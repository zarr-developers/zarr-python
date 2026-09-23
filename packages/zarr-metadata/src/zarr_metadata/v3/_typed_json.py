"""JSON values checked against type annotations.

`parser_for` turns an annotation into a parser: a function of a JSON
value and its location that returns the typed value and every problem
found with it. The annotations it reads are the shapes JSON takes and no
others -- `int`, `float` for any number, `bool`, `str`, `None` for null,
`JSONValue`, a `Literal` of JSON scalars, `tuple[T, ...]` and
`tuple[T1, T2]`, a union of those, an object described by a TypedDict,
`Mapping[str, V]`, a `NewType` as the type it names, and a type alias as
the type it stands for -- which is what keeps it small. An annotation
outside these implies no parser.

A TypedDict is read as the typing spec defines it, which is not always
what its runtime attributes say: `typeddict_keys` reads which keys it
declares, what each holds and whether it is required, and what every
other key may hold. What a parser returns is a value of its annotation:
an array comes back as a tuple, and an object as a new dict of the keys
its type admits, so a key a closed TypedDict does not declare is
reported, as `unknown_key`, and left out.

Nothing here knows what a metadata field is. A caller with a shape of its
own passes a `leaf`, which is asked first for every annotation at every
depth; the parser it returns is used as it is. Parsers are compiled once
per annotation and are pure functions of the value, so the branch of a
union that did not match leaves nothing behind.
"""

from __future__ import annotations

import functools
import types
import typing
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Annotated,
    Final,
    ForwardRef,
    Literal,
    Never,
    NewType,
    NoReturn,
    TypeAlias,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import typing_extensions
from typing_extensions import NoExtraItems, TypeIs, is_typeddict

from zarr_metadata._common import JSONValue
from zarr_metadata.model._validation import ValidationProblem, is_json

if TYPE_CHECKING:
    from zarr_metadata.model._validation import ProblemKind


Loc: TypeAlias = tuple[str | int, ...]
"""Where in a document a value sits: the keys and indices down to it."""

Parsed: TypeAlias = tuple[object, tuple[ValidationProblem, ...]]
"""What a parser returns: the typed value, and every problem found with it."""

Parser: TypeAlias = Callable[[object, Loc], Parsed]
"""One value against one annotation, located at `loc`: compiled once, run many times."""

Leaf: TypeAlias = Callable[[object], "Parser | None"]
"""A caller's own shapes: asked first for every annotation, None to decline."""

# The qualifiers a TypedDict key may carry. `typing` and `typing_extensions`
# may each define one, and a TypedDict may be written with either.
_REQUIRED: Final[tuple[object, ...]] = (typing.Required, typing_extensions.Required)
_NOT_REQUIRED: Final[tuple[object, ...]] = (typing.NotRequired, typing_extensions.NotRequired)
_READ_ONLY: Final[tuple[object, ...]] = (
    typing_extensions.ReadOnly,
    getattr(typing, "ReadOnly", typing_extensions.ReadOnly),
)
_QUALIFIERS: Final[tuple[object, ...]] = (*_REQUIRED, *_NOT_REQUIRED, *_READ_ONLY)

# A `type` statement makes a `typing.TypeAliasType`, which is not the
# `typing_extensions` one on every version that has both.
_ALIASES: Final[tuple[type, ...]] = (
    typing_extensions.TypeAliasType,
    getattr(typing, "TypeAliasType", typing_extensions.TypeAliasType),
)


def problem(
    loc: Loc, message: str, kind: ProblemKind = "invalid_type"
) -> tuple[ValidationProblem, ...]:
    """One problem, as the one-element tuple every parser returns.

    A tuple so that a parser can return it directly and a rule can
    `found.extend(problem(...))` and raise `MetadataValidationError(found)`
    once. The default `kind` names a type mismatch; a value rule passes
    `"invalid_value"`.
    """
    return (ValidationProblem(loc, message, kind),)


def is_integer(value: object) -> TypeIs[int]:
    """A JSON integer: an `int`, and not a `bool`.

    `True` is an `int` in Python and `true` is not a number in JSON, so
    the two have to be told apart everywhere a number is expected.
    """
    return not isinstance(value, bool) and isinstance(value, int)


# --- annotations ---------------------------------------------------------


def strip_annotation(annotation: object) -> tuple[object, tuple[object, ...]]:
    """An annotation's type, and the metadata `Annotated` wrapped it in.

    `Required`, `NotRequired` and `ReadOnly` are qualifiers on a TypedDict
    key, not part of the value's type; peeled with the `Annotated` layers,
    in whatever order they were written.
    """
    metadata: list[object] = []
    while True:
        origin = get_origin(annotation)
        if origin is Annotated:
            inner, *extras = get_args(annotation)
            metadata.extend(extras)
            annotation = inner
        elif origin in _QUALIFIERS:
            (annotation,) = get_args(annotation)
        else:
            return annotation, tuple(metadata)


Qualifier: TypeAlias = Literal["Required", "NotRequired", "ReadOnly"]
"""A qualifier on a TypedDict key, by name: `typing` and `typing_extensions` may each spell one."""


def qualifiers(annotation: object) -> frozenset[Qualifier]:
    """The qualifiers on a TypedDict key, under whatever `Annotated` layers, in whatever order."""
    found: set[Qualifier] = set()
    while True:
        origin = get_origin(annotation)
        if origin is Annotated:
            annotation = get_args(annotation)[0]
            continue
        if origin in _REQUIRED:
            found.add("Required")
        elif origin in _NOT_REQUIRED:
            found.add("NotRequired")
        elif origin in _READ_ONLY:
            found.add("ReadOnly")
        else:
            return frozenset(found)
        (annotation,) = get_args(annotation)


def is_union(annotation: object) -> bool:
    return get_origin(annotation) in (typing.Union, types.UnionType)


def is_alias(annotation: object) -> bool:
    """Whether `annotation` is a type alias that takes no type parameters: `type Level = int`."""
    if not isinstance(annotation, _ALIASES):
        return False
    return len(cast("typing_extensions.TypeAliasType", annotation).__type_params__) == 0


# --- TypedDicts ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TypedDictKeys:
    """What a TypedDict says of an object's keys, read as the typing spec defines it.

    `members` holds every key it declares, its bases' included, with the
    type of the key's value -- qualifiers and `Annotated` metadata peeled
    -- and whether the key is required. `extra_items` is what any other
    key may hold: `Never` when the TypedDict is closed, `object` when it
    is open, and the `extra_items` type otherwise. `declared` is whether
    that was said, by the TypedDict or a base, rather than defaulted: a
    TypedDict that says nothing is open.
    """

    members: Mapping[str, tuple[object, bool]]
    extra_items: object
    declared: bool

    @property
    def required(self) -> frozenset[str]:
        """The keys an object of this TypedDict must have."""
        return frozenset(key for key, (_, required) in self.members.items() if required)

    @property
    def closed(self) -> bool:
        """Whether a key it does not declare is not a key of the type: `closed=True`, or `extra_items=Never`."""
        return self.extra_items is Never or self.extra_items is NoReturn

    @property
    def open(self) -> bool:
        """Whether a key it does not declare may hold anything: the default, or `closed=False`."""
        return self.extra_items is object


@functools.cache
def typeddict_keys(typeddict: type) -> TypedDictKeys:
    """What `typeddict` says of an object's keys; `TypeError` if its annotations do not resolve.

    Whether a key is required follows the spec: a `Required` or
    `NotRequired` qualifier says, and otherwise `total` of the class that
    declared the key does. The qualifiers are read off the annotations as
    evaluated, each where its class was defined, because the runtime's
    `__required_keys__` is worked out before they are: an annotation
    written as a string -- as `from __future__ import annotations` writes
    every one -- hides its qualifier. For a key with no qualifier,
    `__required_keys__` is right, and carries the `total` that applied.

    Openness follows the spec too: `closed=True` closes the TypedDict,
    `extra_items=` types every other key, `closed=False` opens it, and one
    that says none of these is as its bases are -- which the runtime does
    not record -- or open, when no base says either.
    """
    try:
        hints = get_type_hints(typeddict, include_extras=True)
    except NameError as error:
        msg = f"{typeddict.__name__}: {error}; its annotations must resolve in its module"
        raise TypeError(msg) from error
    required_at_runtime = cast("frozenset[str]", getattr(typeddict, "__required_keys__", ()))
    members: dict[str, tuple[object, bool]] = {}
    for key, hint in hints.items():
        said = qualifiers(hint)
        if "Required" in said:
            required = True
        elif "NotRequired" in said:
            required = False
        else:
            required = key in required_at_runtime
        members[key] = (strip_annotation(hint)[0], required)
    extra_items, declared = _openness(typeddict)
    return TypedDictKeys(types.MappingProxyType(members), extra_items, declared)


def _openness(typeddict: type) -> tuple[object, bool]:
    """What a key `typeddict` does not declare may hold, and whether that was said rather than defaulted."""
    extra_items = getattr(typeddict, "__extra_items__", NoExtraItems)
    if extra_items is not NoExtraItems:
        return _extra_items_type(typeddict, extra_items), True
    closed = getattr(typeddict, "__closed__", None)
    if closed is not None:
        return (Never if closed else object), True
    inherited = [_openness(base) for base in _typeddict_bases(typeddict)]
    restricted = {extra for extra, _ in inherited if extra is not object}
    if len(restricted) > 1:
        msg = (
            f"{typeddict.__name__}: its bases disagree on what a key they do not declare may "
            "hold; say it on the class, with closed= or extra_items="
        )
        raise TypeError(msg)
    if len(restricted) == 1:
        return restricted.pop(), True
    return object, any(declared for _, declared in inherited)


def _extra_items_type(typeddict: type, extra_items: object) -> object:
    """The `extra_items` type, evaluated where `typeddict` was defined, `ReadOnly` peeled."""
    if isinstance(extra_items, (str, ForwardRef)):
        holder = type(
            "_ExtraItems",
            (),
            {"__annotations__": {"extra": extra_items}, "__module__": typeddict.__module__},
        )
        try:
            extra_items = get_type_hints(holder, include_extras=True)["extra"]
        except NameError as error:
            msg = f"{typeddict.__name__}: {error}; its extra_items must resolve in its module"
            raise TypeError(msg) from error
    return strip_annotation(extra_items)[0]


def _typeddict_bases(typeddict: type) -> tuple[type, ...]:
    """The TypedDicts `typeddict` was declared with as bases, type arguments dropped."""
    bases: list[type] = []
    for base in getattr(typeddict, "__orig_bases__", ()):
        origin = get_origin(base) or base
        if isinstance(origin, type) and is_typeddict(origin):
            bases.append(origin)
    return tuple(bases)


# --- what a message calls a shape, and which shape a value has -------------


def describe(annotation: object, seen: frozenset[object] = frozenset()) -> str:
    """The annotation as a message would name it: "an integer", "an object"."""
    inner = strip_annotation(annotation)[0]
    if inner is int:
        return "an integer"
    if inner is float:
        return "a number"
    if inner is bool:
        return "a boolean"
    if inner is str:
        return "a string"
    if inner is None or inner is types.NoneType:
        return "null"
    if inner is JSONValue:
        return "a JSON value"
    origin = get_origin(inner)
    if origin is Literal:
        return f"one of {tuple(sorted(get_args(inner), key=repr))!r}"
    if is_union(inner):
        return " or ".join(describe(branch, seen) for branch in get_args(inner))
    if origin is tuple:
        arguments = get_args(inner)
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return f"an array of {describe(arguments[0], seen)} elements"
        if len(arguments) == 2:
            return f"a [{describe(arguments[0], seen)}, {describe(arguments[1], seen)}] pair"
        return f"an array of {len(arguments)} elements"
    if origin in (Mapping, dict):
        return "an object"
    if isinstance(inner, NewType):
        return describe(inner.__supertype__, seen)
    if isinstance(inner, type) and is_typeddict(inner):
        return "an object"
    if is_alias(inner):
        alias = cast("typing_extensions.TypeAliasType", inner)
        if alias in seen:
            return f"a {alias.__name__}"
        return describe(alias.__value__, seen | {alias})
    return "a value"


def shape_of(annotation: object) -> str | None:
    """The top-level JSON shape an annotation admits, for choosing a union branch.

    None means any shape -- a JSON value, a union that mixes them, or a
    shape the caller's `leaf` reads.
    """
    inner = strip_annotation(annotation)[0]
    seen: set[object] = set()
    while (isinstance(inner, NewType) or is_alias(inner)) and inner not in seen:
        seen.add(inner)
        if isinstance(inner, NewType):
            inner = strip_annotation(inner.__supertype__)[0]
        else:
            inner = strip_annotation(cast("typing_extensions.TypeAliasType", inner).__value__)[0]
    if inner is int:
        return "int"
    if inner is float:
        return "number"
    if inner is bool:
        return "bool"
    if inner is str:
        return "str"
    if inner is None or inner is types.NoneType:
        return "null"
    origin = get_origin(inner)
    if origin is Literal:
        # `True` is a bool before it is an int, as `is_integer` says.
        values: tuple[object, ...] = get_args(inner)
        shapes = {shape_of(type(value)) for value in values}
        return shapes.pop() if len(shapes) == 1 else None
    if origin is tuple:
        return "tuple"
    if origin in (Mapping, dict):
        return "mapping"
    if isinstance(inner, type) and is_typeddict(inner):
        return "mapping"
    return None


def has_shape(shape: str | None, value: object) -> bool:
    if shape is None:
        return True
    if shape == "int":
        return is_integer(value)
    if shape == "number":
        return not isinstance(value, bool) and isinstance(value, (int, float))
    if shape == "bool":
        return isinstance(value, bool)
    if shape == "str":
        return isinstance(value, str)
    if shape == "null":
        return value is None
    if shape == "tuple":
        return isinstance(value, (list, tuple))
    return isinstance(value, Mapping)  # "mapping"


# --- the parsers ---------------------------------------------------------


def _scalar(description: str, admits: Callable[[object], bool]) -> Parser:
    def parse(value: object, loc: Loc) -> Parsed:
        if admits(value):
            return value, ()
        return value, problem(loc, f"expected {description}, got {value!r}")

    return parse


_INTEGER: Final = _scalar("an integer", is_integer)
_NUMBER: Final = _scalar(
    "a number", lambda value: not isinstance(value, bool) and isinstance(value, (int, float))
)
_BOOLEAN: Final = _scalar("a boolean", lambda value: isinstance(value, bool))
_STRING: Final = _scalar("a string", lambda value: isinstance(value, str))
_NULL: Final = _scalar("null", lambda value: value is None)
_JSON: Final = _scalar("a JSON value", is_json)


def one_of(allowed: tuple[object, ...]) -> Parser:
    """A member whose type is a closed set of values.

    Equal and of the same type: JSON `true` is not the integer 1, though
    Python says `True == 1`.
    """

    def parse(value: object, loc: Loc) -> Parsed:
        if not any(value == entry and type(value) is type(entry) for entry in allowed):
            return value, problem(
                loc, f"expected one of {allowed!r}, got {value!r}", "invalid_value"
            )
        return value, ()

    return parse


def sequence_of(element: Parser) -> Parser:
    """A member whose type is an array of one element type, parsed element by element."""

    def parse(value: object, loc: Loc) -> Parsed:
        if not isinstance(value, (list, tuple)):
            return value, problem(loc, f"expected a sequence, got {value!r}")
        entries = cast("list[object] | tuple[object, ...]", value)
        parsed: list[object] = []
        found: list[ValidationProblem] = []
        for index, entry in enumerate(entries):
            item, problems = element(entry, (*loc, index))
            parsed.append(item)
            found.extend(problems)
        return tuple(parsed), tuple(found)

    return parse


def fixed_tuple(elements: Sequence[Parser], description: str) -> Parser:
    """A member whose type is an array of a fixed length, parsed position by position."""

    def parse(value: object, loc: Loc) -> Parsed:
        if not isinstance(value, (list, tuple)):
            return value, problem(loc, f"expected {description}, got {value!r}")
        entries = tuple(cast("list[object] | tuple[object, ...]", value))
        if len(entries) != len(elements):
            return entries, problem(loc, f"expected {description}, got {entries!r}")
        parsed: list[object] = []
        found: list[ValidationProblem] = []
        for position, (element, entry) in enumerate(zip(elements, entries, strict=True)):
            item, problems = element(entry, (*loc, position))
            parsed.append(item)
            found.extend(problems)
        return tuple(parsed), tuple(found)

    return parse


def any_of(branches: Sequence[tuple[str | None, Parser]], description: str) -> Parser:
    """A member whose type is a union of shapes, parsed by the branch it fits.

    Each branch is its top-level shape, from `shape_of`, and its parser.
    The branch whose shape the value has is the one that reports -- so an
    element inside a malformed array is located inside the array, rather
    than the whole array being called wrong. A value fitting no branch's
    shape is reported once, by what was expected; one fitting several is
    parsed by the first that accepts it with no problem, and otherwise
    reported by the first that tried.
    """

    def parse(value: object, loc: Loc) -> Parsed:
        first: Parsed | None = None
        for shape, branch in branches:
            if not has_shape(shape, value):
                continue
            result = branch(value, loc)
            if len(result[1]) == 0:
                return result
            if first is None:
                first = result
        if first is None:
            return value, problem(loc, f"expected {description}, got {value!r}")
        return first

    return parse


def object_of(members: Mapping[str, tuple[Parser, bool]], extra: Parser | None) -> Parser:
    """An object with declared keys, each parsed at its own key, and `extra` for every other key.

    `members` maps each declared key to its parser and whether it is
    required; a required key missing is `missing_key`. `extra` parses a
    key the object does not declare; with no `extra` -- a closed
    TypedDict -- such a key is `unknown_key`, and left out of what comes
    back. What comes back is a new dict, keys in the order they came.
    """

    def parse(value: object, loc: Loc) -> Parsed:
        if not isinstance(value, Mapping):
            return value, problem(loc, f"expected an object, got {value!r}")
        entries = cast("Mapping[str, object]", value)
        parsed: dict[str, object] = {}
        found: list[ValidationProblem] = []
        for key, entry in entries.items():
            if key in members:
                continue
            if extra is None:
                found.extend(problem((*loc, key), f"unexpected key {key!r}", "unknown_key"))
                continue
            parsed[key], problems = extra(entry, (*loc, key))
            found.extend(problems)
        for key, (member, required) in members.items():
            if key in entries:
                parsed[key], problems = member(entries[key], (*loc, key))
                found.extend(problems)
            elif required:
                found.extend(problem((*loc, key), f"missing required key {key!r}", "missing_key"))
        return {key: parsed[key] for key in entries if key in parsed}, tuple(found)

    return parse


def mapping_of(value: Parser) -> Parser:
    """A member whose type is an object with any keys, parsed value by value.

    The open counterpart of `object_of`: a `Mapping[str, V]` says nothing
    about which keys there are, only what each value must be.
    """

    def parse(candidate: object, loc: Loc) -> Parsed:
        if not isinstance(candidate, Mapping):
            return candidate, problem(loc, f"expected an object, got {candidate!r}")
        entries = cast("Mapping[str, object]", candidate)
        parsed: dict[str, object] = {}
        found: list[ValidationProblem] = []
        for key, entry in entries.items():
            item, problems = value(entry, (*loc, key))
            parsed[key] = item
            found.extend(problems)
        return parsed, tuple(found)

    return parse


# --- the compiler --------------------------------------------------------

_Building: TypeAlias = dict[object, Parser]
"""The parsers of one compilation, by the TypedDict or alias each reads.

A TypedDict or alias being compiled is in it already, as a parser that
reads through to the one being built: a type that holds itself compiles
to a parser that calls itself.
"""


def _union(inner: object, leaf: Leaf, building: _Building) -> Parser | None:
    branches: list[tuple[str | None, Parser]] = []
    for branch in get_args(inner):
        member = _compile(branch, leaf, building)
        if member is None:
            return None
        branches.append((shape_of(branch), member))
    return any_of(branches, describe(inner))


def _tuple(inner: object, leaf: Leaf, building: _Building) -> Parser | None:
    arguments = get_args(inner)
    if len(arguments) == 2 and arguments[1] is Ellipsis:
        element = _compile(arguments[0], leaf, building)
        return None if element is None else sequence_of(element)
    elements: list[Parser] = []
    for argument in arguments:
        element = _compile(argument, leaf, building)
        if element is None:
            return None
        elements.append(element)
    return fixed_tuple(elements, describe(inner))


def _mapping(inner: object, leaf: Leaf, building: _Building) -> Parser | None:
    arguments = get_args(inner)
    if len(arguments) != 2 or arguments[0] is not str:
        return None
    value = _compile(arguments[1], leaf, building)
    return None if value is None else mapping_of(value)


def _recursive(
    key: object, building: _Building, compile_: Callable[[], Parser | None]
) -> Parser | None:
    """The parser `compile_` builds for `key`, a TypedDict or alias that may hold itself."""
    if key in building:
        return building[key]
    built: list[Parser] = []
    building[key] = lambda value, loc: built[0](value, loc)
    parser = compile_()
    if parser is None:
        return None
    built.append(parser)
    building[key] = parser
    return parser


def _object(typeddict: type, leaf: Leaf, building: _Building) -> Parser | None:
    def compile_() -> Parser | None:
        keys = typeddict_keys(typeddict)
        members: dict[str, tuple[Parser, bool]] = {}
        for key, (annotation, required) in keys.members.items():
            member = _compile(annotation, leaf, building)
            if member is None:
                return None
            members[key] = (member, required)
        if keys.closed:
            return object_of(members, None)
        if keys.open:
            return object_of(members, _JSON)
        extra = _compile(keys.extra_items, leaf, building)
        return None if extra is None else object_of(members, extra)

    return _recursive(typeddict, building, compile_)


def _alias(
    alias: typing_extensions.TypeAliasType, leaf: Leaf, building: _Building
) -> Parser | None:
    return _recursive(alias, building, lambda: _compile(alias.__value__, leaf, building))


def _literal(inner: object) -> Parser | None:
    values: tuple[object, ...] = get_args(inner)
    if not all(value is None or isinstance(value, (str, int)) for value in values):
        return None  # an enum member, or bytes: nothing a JSON document holds
    # Sorted, because the order `get_args` reports is not the order the
    # `Literal` was written in: two `Literal`s over the same values
    # compare and hash equal, so the first one built anywhere in the
    # process is the one every later one resolves to. The parse is a
    # membership test either way; this is so the message listing the
    # values does not depend on import order.
    return one_of(tuple(sorted(values, key=repr)))


def _compile(annotation: object, leaf: Leaf, building: _Building) -> Parser | None:
    inner = strip_annotation(annotation)[0]
    found = leaf(inner)
    if found is not None:
        return found
    if inner is int:
        return _INTEGER
    if inner is float:
        return _NUMBER
    if inner is bool:
        return _BOOLEAN
    if inner is str:
        return _STRING
    if inner is None or inner is types.NoneType:
        return _NULL
    if inner is JSONValue:
        return _JSON
    origin = get_origin(inner)
    if origin is Literal:
        return _literal(inner)
    if is_union(inner):
        return _union(inner, leaf, building)
    if origin is tuple:
        return _tuple(inner, leaf, building)
    if origin in (Mapping, dict):
        return _mapping(inner, leaf, building)
    if isinstance(inner, type) and is_typeddict(inner):
        return _object(inner, leaf, building)
    if isinstance(inner, NewType):
        # A `NewType` is its supertype to a document; the distinction is
        # the code's, for a value it has vouched for.
        return _compile(inner.__supertype__, leaf, building)
    if is_alias(inner):
        return _alias(cast("typing_extensions.TypeAliasType", inner), leaf, building)
    return None


def parser_for(annotation: object, leaf: Leaf) -> Parser | None:
    """The parser an annotation implies, or None if it implies none.

    A small compiler over the shapes JSON takes and no others, listed in
    the module docstring. `leaf` is asked first, here and at every depth
    -- inside a union, an array, an object -- and what it returns is used
    as it is. Closed: an annotation outside these implies no parser. A
    TypedDict whose annotations do not resolve, or whose bases disagree
    on what its other keys hold, is a `TypeError` naming it.
    """
    return _compile(annotation, leaf, {})


def no_leaf(annotation: object) -> Parser | None:
    """The leaf of a caller with no shapes of its own."""
    return None


def parser(annotation: object, leaf: Leaf) -> Parser:
    """The parser an annotation implies; `TypeError` if it implies none."""
    found = parser_for(annotation, leaf)
    if found is None:
        msg = f"{annotation!r} is not a shape JSON takes"
        raise TypeError(msg)
    return found


__all__ = [
    "Leaf",
    "Loc",
    "Parsed",
    "Parser",
    "Qualifier",
    "TypedDictKeys",
    "any_of",
    "describe",
    "fixed_tuple",
    "has_shape",
    "is_alias",
    "is_integer",
    "is_union",
    "mapping_of",
    "no_leaf",
    "object_of",
    "one_of",
    "parser",
    "parser_for",
    "problem",
    "qualifiers",
    "sequence_of",
    "shape_of",
    "strip_annotation",
    "typeddict_keys",
]
