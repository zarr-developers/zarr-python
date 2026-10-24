"""JSON values checked against type annotations.

`parser_for` turns an annotation into a parser. A parser is a function
that takes a JSON value and its location, and returns the typed value and
every problem found with it. `parser_for` supports only the annotations
that describe JSON shapes: `int`, `float` for any number, `bool`, `str`,
`None` for null, `JSONValue`, a `Literal` of JSON scalars, `tuple[T, ...]`
and `tuple[T1, T2]`, a union of those, an object described by a TypedDict,
`Mapping[str, V]`, a `NewType` (parsed as its supertype), and a type alias
(parsed as the type it stands for). This short list keeps the module
small. Any other annotation has no parser. A number type may carry bounds,
written with annotated-types as pydantic reads them, for example
`Annotated[int, Interval(ge=0, le=9)]`. A value outside the bounds is a
problem. `constraints_of` lists the bounds the checker supports.

A TypedDict is read as the typing spec defines it. This is not always
what its runtime attributes say. `typeddict_keys` reads which keys the
TypedDict declares, the type of each, whether each is required, and what
any other key may hold. A parser returns a value of its annotation: an
array comes back as a tuple, and an object comes back as a new dict of the
keys its type allows. A key that a closed TypedDict does not declare is
reported as `unknown_key` and left out.

`check` is the full entry point for a caller that has a JSON value and a
TypedDict. The public module `zarr_metadata.typed_json` exports it.
This module does not know what a metadata field is. A caller with its own
shapes passes a `leaf`. The parser asks the leaf first for every
annotation at every depth, and uses the parser the leaf returns without
changes. Parsers are compiled once per annotation and are pure functions
of the value, so a union branch that did not match leaves no state
behind.

`json_schema` writes the same rules as a JSON Schema. `Schemas` writes
each shape as the checker reads it. It asks a caller's `SchemaLeaf`
first, the same way a parser asks a `Leaf`.
"""

from __future__ import annotations

import functools
import math
import operator
import sys
import types
import typing
import urllib.parse
from collections.abc import Callable, Iterator, Mapping, Sequence
from collections.abc import Set as AbstractSet
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
    TypeGuard,
    TypeVar,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import annotated_types
import typing_extensions
from typing_extensions import NoExtraItems, TypeIs, is_typeddict

from zarr_metadata._common import JSONValue, UserJSONValue
from zarr_metadata._json import (
    ValidationProblem,
    choices,
    copied,
    is_canonical_json,
    is_json,
    is_list_or_tuple,
    is_object,
    listed,
    outside_of,
    refine_json,
    shown,
    with_input,
)

if TYPE_CHECKING:
    from zarr_metadata._json import ProblemKind


T = TypeVar("T")

Loc: TypeAlias = tuple[str | int, ...]
"""The location of a value in a document: the keys and indices that lead to it."""

Parsed: TypeAlias = tuple[object, tuple[ValidationProblem, ...]]
"""The result of a parser: the typed value, and every problem found with it."""

Parser: TypeAlias = Callable[[object, Loc], Parsed]
"""Checks one value at `loc` against one annotation. Compiled once, run many times."""

Leaf: TypeAlias = Callable[[object], "Parser | None"]
"""A caller's own shapes. Asked first for every annotation; returns None to decline."""

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
_ALIASES: Final[tuple[type[typing_extensions.TypeAliasType], ...]] = (
    typing_extensions.TypeAliasType,
    cast(
        "type[typing_extensions.TypeAliasType]",
        getattr(typing, "TypeAliasType", typing_extensions.TypeAliasType),
    ),
)


def problem(
    loc: Loc, message: str, kind: ProblemKind = "invalid_type"
) -> tuple[ValidationProblem, ...]:
    """One problem, as the one-element tuple every parser returns.

    It is a tuple so that a parser can return it directly, and a caller that
    collects problems can call `found.extend(problem(...))`. The default
    `kind` means a type mismatch. A value rule passes `"invalid_value"`.
    """
    return (ValidationProblem(loc, message, kind),)


def is_integer(value: object) -> TypeIs[int]:
    """A JSON integer: an `int`, and not a `bool`.

    `True` is an `int` in Python, but `true` is not a number in JSON, so
    the code must distinguish the two everywhere a number is expected.
    """
    return not isinstance(value, bool) and isinstance(value, int)


# --- annotations ---------------------------------------------------------


def strip_annotation(annotation: object) -> tuple[object, tuple[object, ...]]:
    """An annotation's type, and the metadata `Annotated` wrapped it in.

    `Required`, `NotRequired` and `ReadOnly` are qualifiers on a TypedDict
    key, not part of the value's type. This function removes them together
    with the `Annotated` layers, in any order they were written.
    """
    metadata: list[object] = []
    while True:
        origin = get_origin(annotation)
        if origin is Annotated:
            inner, *extras = get_args(annotation)
            # Put an inner layer's metadata first, the same way `Annotated`
            # flattens nested layers.
            metadata[:0] = extras
            annotation = inner
        elif origin in _QUALIFIERS:
            (annotation,) = get_args(annotation)
        else:
            return annotation, tuple(metadata)


def unqualified(annotation: object) -> object:
    """A TypedDict key's annotation with its qualifiers removed and its `Annotated` metadata kept, since that metadata is part of the value's type."""
    inner, metadata = strip_annotation(annotation)
    if len(metadata) == 0:
        return inner
    return Annotated[(inner, *metadata)]


Qualifier: TypeAlias = Literal["Required", "NotRequired", "ReadOnly"]
"""The name of a qualifier on a TypedDict key. `typing` and `typing_extensions` may each define one."""


def qualifiers(annotation: object) -> frozenset[Qualifier]:
    """The qualifiers on a TypedDict key, under any `Annotated` layers, in any order."""
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


def is_alias(annotation: object) -> TypeGuard[typing_extensions.TypeAliasType]:
    """Whether `annotation` is a type alias that takes no type parameters: `type Level = int`."""
    return isinstance(annotation, _ALIASES) and len(annotation.__type_params__) == 0


@functools.cache
def alias_value(alias: typing_extensions.TypeAliasType) -> object:
    """The type `alias` stands for, evaluated in the module that defined it; `TypeError` if a name in it does not resolve.

    Python evaluates the value of a `type` statement on access. An alias
    made with `TypeAliasType` holds whatever value it was given. An alias
    that refers to itself, such as
    `TypeAliasType("Dtype", str | tuple[tuple[str, "Dtype"], ...])`,
    holds that reference as a string. This function resolves the string in
    the alias's module.
    """
    holder = type(
        "_AliasValue",
        (),
        {"__annotations__": {"value": alias.__value__}, "__module__": alias.__module__},
    )
    try:
        return get_type_hints(holder, include_extras=True)["value"]
    except NameError as error:
        msg = f"{alias.__name__}: {error}; its value must resolve in its module"
        raise TypeError(msg) from error


# --- TypedDicts ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TypedDictKeys:
    """The keys a TypedDict defines for an object, read as the typing spec defines them.

    `members` holds every key the TypedDict declares, including keys from
    its bases. For each key it holds the type of the value and whether the
    key is required. The type has its qualifiers removed and keeps any
    `Annotated` metadata, since a constraint there is part of the type.
    `extra_items` is the type any other key may hold: `Never` when the
    TypedDict is closed, `object` when it is open, and the `extra_items`
    type otherwise. `declared` is whether the TypedDict or a base set this
    explicitly, rather than by default. A TypedDict that sets nothing is
    open.
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
        """Whether undeclared keys are forbidden: `closed=True`, or `extra_items=Never`."""
        extra_items = strip_annotation(self.extra_items)[0]
        return extra_items is Never or extra_items is NoReturn

    @property
    def open(self) -> bool:
        """Whether an undeclared key may hold any value: the default, or `closed=False`."""
        return strip_annotation(self.extra_items)[0] is object


@functools.cache
def typeddict_keys(typeddict: type) -> TypedDictKeys:
    """The keys `typeddict` defines for an object; `TypeError` if its annotations do not resolve.

    Whether a key is required follows the spec. A `Required` or
    `NotRequired` qualifier decides it. Otherwise the `total` setting of
    the class that declared the key decides it. This function reads the
    qualifiers from the evaluated annotations, each evaluated in the
    module where its class was defined. It cannot use the runtime's
    `__required_keys__` for this, because the runtime computes it before
    the annotations are evaluated: an annotation written as a string (and
    `from __future__ import annotations` makes every annotation a string)
    hides its qualifier. For a key with no qualifier, `__required_keys__`
    is correct and reflects the `total` setting that applied.

    Openness also follows the spec. `closed=True` closes the TypedDict,
    `extra_items=` sets the type of every other key, and `closed=False`
    opens it. A TypedDict that sets none of these inherits the setting of
    its bases (the runtime does not record this), or is open when no base
    sets either.
    """
    hints = dict(_hints(typeddict))
    required_at_runtime = cast("frozenset[str]", getattr(typeddict, "__required_keys__", ()))
    optional_at_runtime = cast("frozenset[str]", getattr(typeddict, "__optional_keys__", ()))
    if frozenset(hints) != required_at_runtime | optional_at_runtime:
        msg = (
            f"{typeddict.__name__}: its annotations declare {sorted(hints)!r}, and the runtime "
            f"{sorted(required_at_runtime | optional_at_runtime)!r}"
        )
        raise TypeError(msg)
    members: dict[str, tuple[object, bool]] = {}
    for key, hint in hints.items():
        said = qualifiers(hint)
        if "Required" in said and "NotRequired" in said:
            msg = f"{typeddict.__name__}.{key}: Required and NotRequired both; the spec allows one"
            raise TypeError(msg)
        if "Required" in said:
            required = True
        elif "NotRequired" in said:
            required = False
        else:
            required = key in required_at_runtime
        members[key] = (unqualified(hint), required)
    extra_items, declared = _openness(typeddict)
    return TypedDictKeys(types.MappingProxyType(members), extra_items, declared)


@functools.cache
def _hints(typeddict: type) -> Mapping[str, object]:
    """Every key's annotation, evaluated in the module of the class that declared the key.

    This is what the typing spec requires, and `get_type_hints` does not
    fully do it. `get_type_hints` evaluates every inherited annotation in
    the subclass's own module. If a base in another module wrote a string
    inside an annotation, such as `tuple["Local", ...]`, that string is
    evaluated where `Local` may mean something else, or nothing. This
    happens because the runtime ties a module only to a string at the top
    level of an annotation. `get_type_hints` also looks up names in the
    class's module before the module a string is tied to, so an inherited
    `x: "Foo"` resolves to the subclass's `Foo` when both modules define
    one. To avoid both problems, this function takes a base's keys from
    the base, and evaluates the class's own keys (those no base provided)
    in its module with nothing else in scope. On Python 3.11,
    `typing.TypedDict` does not record a class's bases, so every key is
    evaluated as the class's own. That is correct unless a string nested in
    a base's annotation names something that only the base's module has.
    """
    bases = _typeddict_bases(typeddict)
    hints: dict[str, object] = {}
    for base in bases:
        hints.update(_hints(base))
    written = _written(typeddict)
    inherited = [_written(base) for base in bases]
    own = {
        key: annotation
        for key, annotation in written.items()
        if not any(key in given and given[key] == annotation for given in inherited)
    }
    holder = type("_Own", (), {"__annotations__": own, "__module__": typeddict.__module__})
    try:
        hints.update(get_type_hints(holder, localns={}, include_extras=True))
    except NameError as error:
        msg = f"{typeddict.__name__}: {error}; its annotations must resolve in its module"
        raise TypeError(msg) from error
    return types.MappingProxyType(hints)


def _written(typeddict: type) -> dict[str, object]:
    """A TypedDict's annotations, including its bases', evaluated only as far as they evaluate without errors."""
    if sys.version_info >= (3, 14):
        import annotationlib

        return dict(
            annotationlib.get_annotations(typeddict, format=annotationlib.Format.FORWARDREF)
        )
    return dict(vars(typeddict).get("__annotations__", {}))


def _openness(typeddict: type) -> tuple[object, bool]:
    """The type an undeclared key of `typeddict` may hold, and whether it was set explicitly rather than by default."""
    extra_items = getattr(typeddict, "__extra_items__", NoExtraItems)
    if extra_items is not NoExtraItems:
        return _extra_items_type(typeddict, extra_items), True
    closed = getattr(typeddict, "__closed__", None)
    if closed is not None:
        return (Never if closed else object), True
    inherited = [_openness(base) for base in _typeddict_bases(typeddict)]
    restricted = {extra for extra, _ in inherited if strip_annotation(extra)[0] is not object}
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
    """The `extra_items` type, evaluated in the module of `typeddict`, with `ReadOnly` removed and any `Annotated` metadata kept."""
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
    return unqualified(extra_items)


def _typeddict_bases(typeddict: type) -> tuple[type, ...]:
    """The TypedDict bases `typeddict` was declared with, without type arguments."""
    bases: list[type] = []
    for base in getattr(typeddict, "__orig_bases__", ()):
        origin = get_origin(base) or base
        if isinstance(origin, type) and is_typeddict(origin):
            bases.append(origin)
    return tuple(bases)


# --- names of shapes in messages, and the shape of a value -----------------


def describe(annotation: object, seen: frozenset[object] = frozenset()) -> str:
    """The name of the annotation in an error message: "an integer", "an object"."""
    return _named(annotation, seen)[0]


def _named(annotation: object, seen: frozenset[object]) -> tuple[str, str]:
    """The singular and plural names of the annotation in an error message: "an integer", "integers"."""
    inner = strip_annotation(annotation)[0]
    if inner is int:
        return "an integer", "integers"
    if inner is float:
        return "a number", "numbers"
    if inner is bool:
        return "a boolean", "booleans"
    if inner is str:
        return "a string", "strings"
    if inner is None or inner is types.NoneType:
        return "null", "nulls"
    if inner is JSONValue:
        return "a JSON value", "JSON values"
    origin = get_origin(inner)
    if origin is Literal:
        values = get_args(inner)
        listed = ", ".join(sorted(dict.fromkeys(shown(value) for value in values)))
        return choices(values), f"values in [{listed}]"
    if is_union(inner):
        # Name each shape once, since two TypedDicts are both "an object".
        # A broader shape absorbs a narrower one: "a number" absorbs "an
        # integer", and "a string" absorbs a `Literal` of strings.
        branches = get_args(inner)
        named = dict.fromkeys(_named(branch, seen) for branch in branches)
        if ("a number", "numbers") in named:
            named.pop(("an integer", "integers"), None)
        if ("a string", "strings") in named:
            for branch in filter(_strings_only, branches):
                named.pop(_named(branch, seen), None)
        return " or ".join(one for one, _ in named), " or ".join(many for _, many in named)
    if origin is tuple:
        arguments = get_args(inner)
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            each = _named(arguments[0], seen)[1]
            return f"an array of {each}", f"arrays of {each}"
        if len(arguments) == 2:
            pair = f"[{describe(arguments[0], seen)}, {describe(arguments[1], seen)}] pair"
            return f"a {pair}", f"{pair}s"
        return f"an array of {len(arguments)} elements", f"arrays of {len(arguments)} elements"
    if origin in (Mapping, dict) or (isinstance(inner, type) and is_typeddict(inner)):
        return "an object", "objects"
    if isinstance(inner, NewType):
        return _named(inner.__supertype__, seen)
    if is_alias(inner):
        alias = inner
        if alias in seen or _holds(alias_value(alias), alias, frozenset()):
            return f"a {alias.__name__}", f"{alias.__name__} values"
        return _named(alias_value(alias), seen | {alias})
    return "a value", "values"


def _strings_only(annotation: object) -> bool:
    """Whether `annotation` is a `Literal` of strings, which the name "a string" covers."""
    inner = strip_annotation(annotation)[0]
    return get_origin(inner) is Literal and all(isinstance(value, str) for value in get_args(inner))


def _holds(annotation: object, alias: object, seen: frozenset[object]) -> bool:
    """Whether `alias` occurs in `annotation`, following any aliases in it. Used to tell whether an alias refers to itself."""
    inner = strip_annotation(annotation)[0]
    if inner is alias:
        return True
    if is_alias(inner):
        if inner in seen:
            return False
        value = alias_value(inner)
        return _holds(value, alias, seen | {inner})
    return any(_holds(argument, alias, seen) for argument in get_args(inner))


def shape_of(annotation: object) -> str | None:
    """The top-level JSON shape an annotation allows, used to choose a union branch.

    None means any shape: a JSON value, a union that mixes shapes, or a
    shape that the caller's `leaf` parses.
    """
    inner = strip_annotation(annotation)[0]
    seen: set[object] = set()
    while (isinstance(inner, NewType) or is_alias(inner)) and inner not in seen:
        seen.add(inner)
        if isinstance(inner, NewType):
            inner = strip_annotation(inner.__supertype__)[0]
        else:
            inner = strip_annotation(alias_value(inner))[0]
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
        # `True` counts as a bool, not an int, as in `is_integer`.
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
        return value, problem(loc, f"expected {description}, got {shown(value)}")

    return parse


_INTEGER: Final = _scalar("an integer", is_integer)
_NUMBER: Final = _scalar(
    "a number", lambda value: not isinstance(value, bool) and isinstance(value, (int, float))
)
_BOOLEAN: Final = _scalar("a boolean", lambda value: isinstance(value, bool))
_STRING: Final = _scalar("a string", lambda value: isinstance(value, str))
_NULL: Final = _scalar("null", lambda value: value is None)
_JSON: Final = _scalar("a JSON value", is_json)
_USER_JSON: Final = _scalar("a JSON value", lambda value: is_canonical_json(value, finite=False))


def one_of(allowed: tuple[object, ...]) -> Parser:
    """A member whose type is a closed set of values.

    A value matches an entry when it is equal and of the same type. JSON
    `true` is not the integer 1, though in Python `True == 1`. If the
    value's JSON type matches none of the entries (for example a number,
    where every entry is a string), the problem is a wrong type. If the
    type matches but the value does not, the problem is a wrong value.
    """

    def parse(value: object, loc: Loc) -> Parsed:
        if not any(value == entry and type(value) is type(entry) for entry in allowed):
            return value, (outside_of(loc, value, allowed),)
        return value, ()

    return parse


def sequence_of(element: Parser) -> Parser:
    """A member whose type is an array of one element type, parsed element by element."""

    def parse(value: object, loc: Loc) -> Parsed:
        if not is_list_or_tuple(value):
            return value, problem(loc, f"expected an array, got {shown(value)}")
        entries = value
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
        if not is_list_or_tuple(value):
            return value, problem(loc, f"expected {description}, got {shown(value)}")
        entries = tuple(value)
        if len(entries) != len(elements):
            return entries, problem(loc, f"expected {description}, got {shown(entries)}")
        parsed: list[object] = []
        found: list[ValidationProblem] = []
        for position, (element, entry) in enumerate(zip(elements, entries, strict=True)):
            item, problems = element(entry, (*loc, position))
            parsed.append(item)
            found.extend(problems)
        return tuple(parsed), tuple(found)

    return parse


Branch: TypeAlias = tuple[str | None, Parser, frozenset[str] | None]
"""A branch of a union: its top-level shape (from `shape_of`), its parser, and its keys if it is a TypedDict."""

Tag: TypeAlias = tuple[str, Mapping[tuple[type, object], int]]
"""A key that every branch requires as a `Literal`, and the branch that each of its values selects."""


def any_of(
    branches: Sequence[Branch],
    description: str,
    tag: Tag | None = None,
    literals: frozenset[int] = frozenset(),
) -> Parser:
    """A member whose type is a union of shapes, parsed by the matching branch.

    The parser tries only the branches whose top-level shape matches the
    value. This way a bad element inside an array is reported at its
    position inside the array, rather than the whole array being reported
    as wrong. A value that matches no branch's shape is reported once,
    with the expected shapes. When every branch is a TypedDict that
    requires a key as a `Literal` of values unique to that branch, the
    value of that key selects the branch, and the problems are that
    branch's problems. Otherwise the value is parsed by the first branch
    that reports no problem (among TypedDicts, the one that declares the
    most of the value's keys). If every branch reports problems, the
    parser returns the result of the branch with the fewest problems.
    `literals` holds the index of each branch that is a `Literal`. When
    only such branches tried the value, and it is none of their values,
    it is reported once with the expected shapes, since naming one
    `Literal` would leave out the other branches.
    """

    def parse(value: object, loc: Loc) -> Parsed:
        present = _keys_of(value)
        if tag is not None and present is not None:
            return _by_tag(branches, tag, value, loc)
        clean: list[tuple[int, int, Parsed]] = []
        failed: list[tuple[int, int, Parsed]] = []
        for index, (shape, branch, keys) in enumerate(branches):
            if not has_shape(shape, value):
                continue
            result = branch(value, loc)
            if len(result[1]) != 0:
                failed.append((len(result[1]), index, result))
            elif keys is None or present is None:
                return result
            else:
                clean.append((-len(keys & present), index, result))
        if len(clean) != 0:
            return min(clean, key=lambda found: found[:2])[2]
        if len(failed) != 0 and all(index in literals for _, index, _ in failed):
            # A `Literal` reports one problem, of the kind its own check
            # decided, with the values it allows under `expected`.
            kind = min(failed, key=lambda found: found[:2])[2][1][0].kind
            allowed = [
                item
                for _, _, (_, found) in failed
                for refused in found
                for item in _expected(refused)
            ]
            message = f"expected {description}, got {shown(value)}"
            ctx: dict[str, JSONValue] = {"expected": listed(allowed)}
            return value, (ValidationProblem(loc, message, kind, ctx=ctx),)
        if len(failed) != 0:
            return min(failed, key=lambda found: found[:2])[2]
        return value, problem(loc, f"expected {description}, got {shown(value)}")

    return parse


def _by_tag(branches: Sequence[Branch], tag: Tag, value: object, loc: Loc) -> Parsed:
    """Parse `value` with the branch its tag selects. A missing tag, or a tag value no branch has, is reported at the tag's location."""
    key, picks = tag
    if not is_object(value) or key not in value:
        return value, problem((*loc, key), f"missing required key {key!r}", "missing_key")
    said = value[key]
    index = picks.get((type(said), said)) if _hashable(said) else None
    if index is None:
        return value, (outside_of((*loc, key), said, tuple(entry for _, entry in picks)),)
    return branches[index][1](value, loc)


def _keys_of(value: object) -> AbstractSet[str] | None:
    """The keys of `value` if it is a JSON object, else None."""
    if is_object(value):
        return {key for key in value if isinstance(key, str)}
    return None


def _hashable(value: object) -> bool:
    return value is None or isinstance(value, (str, int, float))


def object_of(members: Mapping[str, tuple[Parser, bool]], extra: Parser | None) -> Parser:
    """An object with declared keys, each parsed at its own key, and `extra` for every other key.

    `members` maps each declared key to its parser and whether it is
    required. A missing required key is a `missing_key` problem. `extra`
    parses a key the object does not declare. With no `extra` (a closed
    TypedDict), such a key is an `unknown_key` problem and is left out of
    the result. The result is a new dict, with keys in their input order.
    """

    def parse(value: object, loc: Loc) -> Parsed:
        if not is_object(value):
            return value, problem(loc, f"expected an object, got {shown(value)}")
        entries = value
        parsed: dict[str, object] = {}
        found: list[ValidationProblem] = []
        for key, entry in entries.items():
            if not isinstance(key, str):
                found.extend(problem(loc, f"non-string key {shown(key)}"))
                continue
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
        return {
            key: parsed[key] for key in entries if isinstance(key, str) and key in parsed
        }, tuple(found)

    return parse


def mapping_of(value: Parser) -> Parser:
    """A member whose type is an object with any keys, parsed value by value.

    The open counterpart of `object_of`: a `Mapping[str, V]` does not
    restrict the keys, only the type of each value.
    """

    def parse(candidate: object, loc: Loc) -> Parsed:
        if not is_object(candidate):
            return candidate, problem(loc, f"expected an object, got {shown(candidate)}")
        entries = candidate
        parsed: dict[str, object] = {}
        found: list[ValidationProblem] = []
        for key, entry in entries.items():
            if not isinstance(key, str):
                found.extend(problem(loc, f"non-string key {shown(key)}"))
                continue
            item, problems = value(entry, (*loc, key))
            parsed[key] = item
            found.extend(problems)
        return parsed, tuple(found)

    return parse


# --- constraints ---------------------------------------------------------

Constraints: TypeAlias = Mapping[str, int | float]
"""The bounds a type carries, keyed by the names pydantic uses: `{"ge": 0, "le": 9}`."""

_BOUNDS: Final[tuple[tuple[type, str], ...]] = (
    (annotated_types.Gt, "gt"),
    (annotated_types.Ge, "ge"),
    (annotated_types.Lt, "lt"),
    (annotated_types.Le, "le"),
)
"""The annotated-types constraints the checker supports, each with its name."""

_FROM_BELOW: Final = frozenset({"gt", "ge"})
"""The lower bounds."""

_FROM_ABOVE: Final = frozenset({"lt", "le"})
"""The upper bounds."""

_EXCLUSIVE: Final = frozenset({"gt", "lt"})
"""The exclusive bounds: a value may not equal them."""

_HOLDS: Final[Mapping[str, Callable[[float, float], bool]]] = {
    "gt": operator.gt,
    "ge": operator.ge,
    "lt": operator.lt,
    "le": operator.le,
}
"""For each bound name, a test of whether a number satisfies that bound."""


def constraints_of(metadata: Sequence[object]) -> dict[str, int | float]:
    """The bounds in an `Annotated` type's metadata, by name: at most one lower and one upper bound.

    The supported bounds are annotated-types' `Gt`, `Ge`, `Lt` and `Le`,
    and an `Interval`, which unpacks to them, as pydantic reads them. A
    note (a string or a `Doc`) does not constrain the value and is
    skipped. Anything else raises `TypeError`:

    - a constraint the checker does not support, such as `MinLen`,
      `Predicate`, or pydantic's `Field`. Otherwise the type would declare
      a rule that the checker does not enforce.
    - a second bound on the same side. pydantic uses the last one.
    - a bound that is not a finite number.

    A bound is stored as the number it equals, as an integer when it is
    one. `Ge(True)`, `Ge(1.0)` and `Ge(1)` are the same bound, because
    Python's `Annotated` cache treats equal metadata as identical and may
    return any of them in place of another.
    """
    said: dict[str, int | float] = {}
    for item in _unpacked(metadata):
        if isinstance(item, (str, typing_extensions.Doc)):
            continue
        name = next((name for kind, name in _BOUNDS if isinstance(item, kind)), None)
        if name is None:
            msg = f"{item!r} is not a constraint the checker reads, which are Gt, Ge, Lt and Le"
            raise TypeError(msg)
        side = _FROM_BELOW if name in _FROM_BELOW else _FROM_ABOVE
        if not side.isdisjoint(said):
            where = "below" if side is _FROM_BELOW else "above"
            msg = f"{item!r} is a second bound from {where}; a type takes one from each side"
            raise TypeError(msg)
        said[name] = _bound(item, name)
    return said


def _unpacked(metadata: Sequence[object]) -> Iterator[object]:
    """`metadata`, with each group of constraints in it (such as an `Interval`) unpacked."""
    for item in metadata:
        if isinstance(item, annotated_types.GroupedMetadata):
            yield from _unpacked(tuple(cast("Sequence[object]", item)))
        else:
            yield item


def _bound(item: object, name: str) -> int | float:
    """The number of bound `item` named `name`, as an integer when it is one; `TypeError` if it is not a finite number."""
    bound = cast("object", getattr(item, name))
    if isinstance(bound, int):
        return int(bound)  # a `bool` becomes its integer, as in the `Annotated` cache
    if isinstance(bound, float) and math.isfinite(bound):
        return int(bound) if bound.is_integer() else bound
    msg = f"{item!r}: a bound is a finite number"
    raise TypeError(msg)


def _constrainable(inner: object, said: Constraints) -> None:
    """Raise `TypeError` for bounds on a type that is not a number: a string, an array, a value of any type."""
    if len(said) != 0 and shape_of(inner) not in ("int", "number"):
        msg = f"{', '.join(said)}: a bound is on a number, and {describe(inner)} is not one"
        raise TypeError(msg)


def constrained(parse: Parser, description: str, said: Constraints) -> Parser:
    """Wrap `parse` to also check the bounds its type carries: one `invalid_value` problem if any bound fails.

    The message states what the type allows, for example "expected an
    integer in [0, 9], got 12", and the problem's `ctx` holds the bounds.
    The bounds are checked only when `parse` reports no problem, in the
    same way that rules run only on a value of their type.
    """
    expected = f"expected {description} {_admitted(said)}"
    ctx: dict[str, JSONValue] = dict(said)
    holds = tuple((_HOLDS[name], bound) for name, bound in said.items())

    def parse_constrained(value: object, loc: Loc) -> Parsed:
        typed, found = parse(value, loc)
        if len(found) != 0:
            return typed, found
        number = cast("float", typed)
        for keeps, bound in holds:
            if not keeps(number, bound):
                message = f"{expected}, got {shown(value)}"
                return typed, (ValidationProblem(loc, message, "invalid_value", ctx=ctx),)
        return typed, ()

    return parse_constrained


def _admitted(said: Constraints) -> str:
    """The range the bounds allow, as written after the type in a message: "in [0, 9]", "in (0, 1)", ">= 1"."""
    low = next(((name, bound) for name, bound in said.items() if name in _FROM_BELOW), None)
    high = next(((name, bound) for name, bound in said.items() if name in _FROM_ABOVE), None)
    if low is not None and high is not None:
        opening = "(" if low[0] in _EXCLUSIVE else "["
        closing = ")" if high[0] in _EXCLUSIVE else "]"
        return f"in {opening}{shown(low[1])}, {shown(high[1])}{closing}"
    if low is not None:
        return f"{'>' if low[0] in _EXCLUSIVE else '>='} {shown(low[1])}"
    name, bound = cast("tuple[str, int | float]", high)
    return f"{'<' if name in _EXCLUSIVE else '<='} {shown(bound)}"


# --- the compiler --------------------------------------------------------

_Building: TypeAlias = dict[object, Parser]
"""The parsers of one compilation, keyed by the TypedDict or alias each one parses.

A TypedDict or alias is added before its compilation finishes, as a
parser that forwards to the one being built. This way a type that refers
to itself compiles to a parser that calls itself.
"""


def _union(inner: object, leaf: Leaf, building: _Building) -> Parser | None:
    arguments = get_args(inner)
    branches: list[Branch] = []
    for branch in arguments:
        member = _compile(branch, leaf, building)
        if member is None:
            return None
        typeddict = strip_annotation(branch)[0]
        keys = (
            frozenset(typeddict_keys(typeddict).members)
            if isinstance(typeddict, type) and is_typeddict(typeddict)
            else None
        )
        branches.append((shape_of(branch), member, keys))
    literals = frozenset(
        index for index, branch in enumerate(arguments) if get_origin(_unaliased(branch)) is Literal
    )
    return any_of(branches, describe(inner), _tag(arguments), literals)


def _unaliased(annotation: object) -> object:
    """`annotation` without its `Annotated` metadata, and with each type alias replaced by its value, until what remains is not an alias."""
    inner = strip_annotation(annotation)[0]
    seen: set[object] = set()
    while is_alias(inner) and inner not in seen:
        seen.add(inner)
        inner = strip_annotation(alias_value(inner))[0]
    return inner


def _expected(refused: ValidationProblem) -> tuple[JSONValue, ...]:
    """The values that `refused`, a `Literal`'s problem, says it allows, from its `ctx`."""
    expected = refused.ctx.get("expected")
    return expected if isinstance(expected, tuple) else ()


def _tag(arguments: Sequence[object]) -> Tag | None:
    """The key that identifies which of these TypedDicts a value is, if there is one.

    Every branch must require the key, as a `Literal` whose values no other
    branch has. This is the discriminator that a union of configurations
    writes as `name`.
    """
    typeddicts = [strip_annotation(argument)[0] for argument in arguments]
    if not all(isinstance(typed, type) and is_typeddict(typed) for typed in typeddicts):
        return None
    keys = [typeddict_keys(cast("type", typed)) for typed in typeddicts]
    shared = set(keys[0].required)
    for each in keys[1:]:
        shared &= each.required
    for key in sorted(shared):
        picks: dict[tuple[type, object], int] = {}
        for index, each in enumerate(keys):
            annotation = strip_annotation(each.members[key][0])[0]
            if get_origin(annotation) is not Literal:
                break
            values: tuple[object, ...] = get_args(annotation)
            if any((type(entry), entry) in picks for entry in values):
                break
            picks.update({(type(entry), entry): index for entry in values})
        else:
            return key, types.MappingProxyType(picks)
    return None


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
    """The parser `compile_` builds for `key`, a TypedDict or alias that may refer to itself."""
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
            # Any JSON value is allowed. A note does not change that, and a
            # bound cannot apply, but the metadata is still checked.
            _constrainable(object, constraints_of(strip_annotation(keys.extra_items)[1]))
            return object_of(members, _JSON)
        extra = _compile(keys.extra_items, leaf, building)
        return None if extra is None else object_of(members, extra)

    return _recursive(typeddict, building, compile_)


def _alias(
    alias: typing_extensions.TypeAliasType, leaf: Leaf, building: _Building
) -> Parser | None:
    return _recursive(alias, building, lambda: _compile(alias_value(alias), leaf, building))


def _literal(inner: object) -> Parser | None:
    values: tuple[object, ...] = get_args(inner)
    if not all(value is None or isinstance(value, (str, int)) for value in values):
        return None  # an enum member, or bytes: a JSON document cannot hold these
    # Sort the values, because `get_args` may not report them in the order
    # this `Literal` was written in. Two `Literal`s over the same values
    # compare and hash equal, so every later one resolves to the first one
    # built anywhere in the process. The parse is a membership test either
    # way. Sorting keeps the message that lists the values independent of
    # import order.
    return one_of(tuple(sorted(values, key=repr)))


def _compile(annotation: object, leaf: Leaf, building: _Building) -> Parser | None:
    inner, metadata = strip_annotation(annotation)
    parse = _compile_type(inner, leaf, building)
    if parse is None or len(metadata) == 0:
        return parse
    said = constraints_of(metadata)
    if len(said) == 0:
        return parse
    _constrainable(inner, said)
    return constrained(parse, describe(inner), said)


def _compile_type(inner: object, leaf: Leaf, building: _Building) -> Parser | None:
    """The parser of a type with `Annotated` already removed."""
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
    if inner is UserJSONValue:
        return _USER_JSON
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
        # In a document, a `NewType` is the same as its supertype. The
        # distinction exists only in code, to mark a value the code has
        # already checked.
        return _compile(inner.__supertype__, leaf, building)
    if is_alias(inner):
        return _alias(inner, leaf, building)
    return None


def parser_for(annotation: object, leaf: Leaf) -> Parser | None:
    """The parser for an annotation, or None if the annotation has none.

    This is a small compiler that supports only the JSON shapes listed in
    the module docstring. It asks `leaf` first, here and at every depth
    (inside a union, an array, or an object), and uses what `leaf` returns
    without changes. The list is closed: any other annotation has no
    parser. A TypedDict whose annotations do not resolve, or whose bases
    disagree on what its other keys hold, raises a `TypeError` that names
    it.
    """
    return _compile(annotation, leaf, {})


def no_leaf(annotation: object) -> Parser | None:
    """The leaf for a caller with no shapes of its own."""
    return None


def parser(annotation: object, leaf: Leaf) -> Parser:
    """The parser for an annotation; `TypeError` naming the part of it that has no parser."""
    found = parser_for(annotation, leaf)
    if found is None:
        inner = strip_annotation(annotation)[0]
        unread = unread_in(inner, leaf) if isinstance(inner, type) and is_typeddict(inner) else None
        msg = unread or f"{annotation!r} is not a shape JSON takes"
        raise TypeError(msg)
    return found


def unread_in(typeddict: type, leaf: Leaf) -> str | None:
    """The part of `typeddict` that has no parser, named down to the TypedDict that contains it; None if every part has one."""
    keys = typeddict_keys(typeddict)
    others: list[tuple[str, object]] = []
    if not (keys.closed or keys.open):
        others.append(("extra_items", keys.extra_items))
    for key, annotation in [*((key, member[0]) for key, member in keys.members.items()), *others]:
        try:
            member = parser_for(annotation, leaf)
        except TypeError as error:
            return f"{typeddict.__name__}.{key}: {error}"
        if member is not None:
            continue
        inner = strip_annotation(annotation)[0]
        if isinstance(inner, type) and is_typeddict(inner):
            deeper = unread_in(inner, leaf)
            if deeper is not None:
                return f"{typeddict.__name__}.{key}: {deeper}"
        return (
            f"{typeddict.__name__}: {key} is not a shape JSON takes; write it as one -- a "
            "number, string, boolean, null, array, object, or an alias of one"
        )
    return None


@functools.cache
def _checker(typeddict: type) -> Parser:
    return parser(typeddict, no_leaf)


def check(
    value: object, shape: type[T], loc: Loc = ()
) -> tuple[T | None, tuple[ValidationProblem, ...]]:
    """Type-check `value` as `shape`, a TypedDict. Returns the typed value or None, and every problem.

    This function first refines `value` to JSON (arrays as tuples, string
    keys, finite floats), then checks it member by member. Each problem is
    located under `loc`. The returned value holds only what `shape` allows.
    A key that a closed TypedDict does not declare is reported as
    `unknown_key` and left out, and the value is still returned. Any other
    problem means the value is not returned (None). The nesting-depth limit
    is counted from the root of the document, where `loc` gives the
    position of `value`, so a `loc` of 255 levels leaves one more level.
    Raises `TypeError` if `shape` is not a TypedDict, or contains a type
    that has no parser.
    """
    if not is_typeddict(shape):
        msg = f"{shape!r} is not a TypedDict"
        raise TypeError(msg)
    refined, problems = refine_json(value, loc)
    if len(problems) != 0:
        return None, with_input(problems, value, loc)
    typed, found = _checker(shape)(refined, loc)
    readable = all(problem.kind == "unknown_key" for problem in found)
    return (cast("T", typed) if readable else None), with_input(found, value, loc)


# --- JSON Schema ---------------------------------------------------------

JSONSchema: TypeAlias = dict[str, JSONValue]
"""A JSON Schema as a JSON object, with arrays as lists, which is the form validators accept."""

SchemaLeaf: TypeAlias = Callable[[object, "Schemas"], "JSONSchema | None"]
"""Writes a caller's own shapes into a schema. Asked first for every annotation, like a `Leaf`; returns None to decline.

It receives the schema being written. This lets a caller's shape contain
other shapes, written with `of`, or be written once in `$defs` with
`defined`.
"""

DIALECT: Final = "https://json-schema.org/draft/2020-12/schema"
"""The dialect of every schema: JSON Schema draft 2020-12, the same dialect pydantic and zod use."""

_KEYWORDS: Final[Mapping[str, str]] = {
    "gt": "exclusiveMinimum",
    "ge": "minimum",
    "lt": "exclusiveMaximum",
    "le": "maximum",
}
"""JSON Schema's keyword for each bound."""

_STRICTER: Final[Mapping[str, Callable[[float, float], float]]] = {
    "exclusiveMinimum": max,
    "minimum": max,
    "exclusiveMaximum": min,
    "maximum": min,
}
"""For each keyword, the function that picks the stricter of two bounds."""


def no_schema_leaf(annotation: object, schemas: Schemas) -> JSONSchema | None:
    """The schema leaf for a caller with no shapes of its own."""
    return None


class Schemas:
    """One JSON Schema being written, and the `$defs` it holds so far.

    `of` writes an annotation as the checker reads it. It asks the leaf
    first at every depth, as `parser_for` asks a `Leaf`. Each TypedDict and
    type alias is written once in `$defs`, under its name (or its name plus
    a number, when another entry already has that name), and referenced
    wherever it occurs. A type that refers to itself becomes a schema that
    refers to itself. `document` returns the whole schema.
    """

    __slots__ = ("_defs", "_leaf", "_names", "_uses")

    def __init__(self, leaf: SchemaLeaf = no_schema_leaf) -> None:
        self._leaf = leaf
        self._defs: dict[str, JSONSchema] = {}
        self._names: dict[object, str] = {}
        self._uses: dict[str, int] = {}

    def of(self, annotation: object) -> JSONSchema:
        """The schema of `annotation` as the checker reads it: its type, with the bounds and notes from `Annotated`.

        Each bound becomes the matching JSON Schema keyword (`Ge(0)` becomes
        `minimum`), and a `Doc` becomes the `description`. The schema keeps
        both the type's bounds and the bounds of its `NewType`, because the
        checker applies both. When both set the same keyword, the stricter
        bound is kept. An annotation the checker supports is written. Any
        other raises `TypeError`, which cannot happen if the caller already
        checked the annotation with `parser`.
        """
        inner, metadata = strip_annotation(annotation)
        schema = self._type(inner)
        if len(metadata) == 0:
            return schema
        notes = [
            item.documentation
            for item in _unpacked(metadata)
            if isinstance(item, typing_extensions.Doc)
        ]
        if len(notes) != 0:
            schema = {**schema, "description": "\n\n".join(notes)}
        for name, bound in constraints_of(metadata).items():
            keyword = _KEYWORDS[name]
            held = cast("int | float | None", schema.get(keyword))
            schema = {**schema, keyword: bound if held is None else _STRICTER[keyword](held, bound)}
        return schema

    def object_of(self, typeddict: type) -> JSONSchema:
        """The inline schema of `typeddict`: its keys, its required keys, and what any other key may hold."""
        keys = typeddict_keys(typeddict)
        schema: JSONSchema = {"type": "object"}
        if len(keys.members) != 0:
            schema["properties"] = {
                key: self.of(annotation) for key, (annotation, _) in keys.members.items()
            }
        required: list[JSONValue] = [key for key in keys.members if key in keys.required]
        if len(required) != 0:
            schema["required"] = required
        if keys.closed:
            schema["additionalProperties"] = False
        elif not keys.open:
            extra = self.of(keys.extra_items)
            if len(extra) != 0:
                schema["additionalProperties"] = extra
        return schema

    def defined(self, key: object, name: str, write: Callable[[], JSONSchema]) -> JSONSchema:
        """A reference to the entry in `$defs` for `key`. `write` writes the entry the first time `key` is requested.

        The entry is named `name`, or `name` plus a number when another key
        already has that name. The entry is reserved before it is written, so
        a schema that refers to itself produces a reference to itself. If
        `write` fails, the reservation is removed, because a later reference
        to it would point to an empty schema, which accepts anything. Any
        entries or use counts that `write` added before it failed are also
        removed.
        """
        name_held = self._names.get(key)
        if name_held is None:
            name_held, number = name, 1
            while name_held in self._defs:
                number += 1
                name_held = f"{name}{number}"
            # If the write fails, restore the previous state: the entries it
            # wrote for the types it contains, the uses it counted, and its
            # own name.
            before = (dict(self._defs), dict(self._names), dict(self._uses))
            self._names[key] = name_held
            self._defs[name_held] = {}
            try:
                self._defs[name_held] = write()
            except BaseException:
                self._defs, self._names, self._uses = before
                raise
        self._uses[name_held] = self._uses.get(name_held, 0) + 1
        return {"$ref": _pointer(name_held)}

    def document(self, root: JSONSchema) -> JSONSchema:
        """The whole schema: its dialect, `root`, and the `$defs`, sorted by name.

        When `root` refers to an entry that nothing else refers to, the
        entry is written inline instead. pydantic does the same for a model
        that does not refer to itself. The result shares no objects with the
        internal state, and no part of it shares objects with another part,
        so a caller may modify it freely.
        """
        defs = dict(self._defs)
        target = next((name for name in defs if root == {"$ref": _pointer(name)}), None)
        if target is not None and self._uses[target] == 1:
            root = defs.pop(target)
        whole: JSONSchema = {"$schema": DIALECT, **root}
        if len(defs) != 0:
            whole["$defs"] = {name: defs[name] for name in sorted(defs)}
        return cast("JSONSchema", copied(whole))

    def _type(self, inner: object) -> JSONSchema:
        """The schema of a type with `Annotated` already removed."""
        found = self._leaf(inner, self)
        if found is not None:
            return found
        if inner is int:
            return {"type": "integer"}
        if inner is float:
            return {"type": "number"}
        if inner is bool:
            return {"type": "boolean"}
        if inner is str:
            return {"type": "string"}
        if inner is None or inner is types.NoneType:
            return {"type": "null"}
        if inner is JSONValue:
            return {}
        origin = get_origin(inner)
        if origin is Literal:
            # Sort the values, as `_literal` does: `get_args` reports a
            # `Literal`'s values in the order of the first equal `Literal`
            # built in the process.
            values: list[JSONValue] = sorted(get_args(inner), key=repr)
            return {"const": values[0]} if len(values) == 1 else {"enum": values}
        if is_union(inner):
            return {"anyOf": [self.of(branch) for branch in get_args(inner)]}
        if origin is tuple:
            return self._tuple(get_args(inner))
        if origin in (Mapping, dict):
            value = self.of(get_args(inner)[1])
            if len(value) == 0:
                return {"type": "object"}
            return {"type": "object", "additionalProperties": value}
        if isinstance(inner, type) and is_typeddict(inner):
            typeddict = inner
            return self.defined(typeddict, typeddict.__name__, lambda: self.object_of(typeddict))
        if isinstance(inner, NewType):
            return self.of(inner.__supertype__)
        if is_alias(inner):
            alias = inner
            return self.defined(alias, alias.__name__, lambda: self.of(alias_value(alias)))
        msg = f"{inner!r} is not a shape JSON takes"
        raise TypeError(msg)

    def _tuple(self, arguments: tuple[object, ...]) -> JSONSchema:
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            items = self.of(arguments[0])
            return {"type": "array"} if len(items) == 0 else {"type": "array", "items": items}
        if len(arguments) == 0:
            return {"type": "array", "maxItems": 0}
        return {
            "type": "array",
            "prefixItems": [self.of(argument) for argument in arguments],
            "items": False,
            "minItems": len(arguments),
        }


def _pointer(name: str) -> str:
    """The reference to the entry in `$defs` named `name`: a JSON pointer, escaped as a URI fragment."""
    escaped = name.replace("~", "~0").replace("/", "~1")
    return f"#/$defs/{urllib.parse.quote(escaped, safe='')}"


def json_schema(shape: type) -> JSONSchema:
    """The JSON Schema for the JSON that `check` accepts as `shape`, a TypedDict.

    The schema uses draft 2020-12 and is a JSON object, with arrays as
    lists and `$schema` first. The schema is built as follows:

    - A TypedDict becomes an object with its keys, its required keys, and
      what any other key may hold (nothing, for a closed TypedDict).
    - A bound becomes the matching JSON Schema keyword: `Ge(0)` becomes
      `minimum`.
    - A `Doc` becomes the `description`. Only a `Doc` produces a
      description; docstrings are for Python readers and are not used.
      This matches zod, which writes only what `.describe()` sets.
    - A `Literal` becomes its values.
    - A union becomes `anyOf` its branches.

    Each TypedDict or type alias is written once in `$defs`, under its
    name, and referenced wherever it occurs. The exception is `shape`
    itself, which is written inline unless it refers to itself.

    One difference comes from JSON Schema itself: it accepts a number with
    no fractional part, such as `1.0`, as an integer, while `check`
    requires `1`. Raises `TypeError` if `shape` is not a TypedDict, or
    contains a type that has no parser, in the same cases as `check`.
    """
    if not is_typeddict(shape):
        msg = f"{shape!r} is not a TypedDict"
        raise TypeError(msg)
    _checker(shape)
    schemas = Schemas()
    return schemas.document(schemas.of(shape))


__all__ = [
    "DIALECT",
    "Branch",
    "Constraints",
    "JSONSchema",
    "Leaf",
    "Loc",
    "Parsed",
    "Parser",
    "Qualifier",
    "SchemaLeaf",
    "Schemas",
    "Tag",
    "TypedDictKeys",
    "alias_value",
    "any_of",
    "check",
    "constrained",
    "constraints_of",
    "describe",
    "fixed_tuple",
    "has_shape",
    "is_alias",
    "is_integer",
    "is_union",
    "json_schema",
    "mapping_of",
    "no_leaf",
    "no_schema_leaf",
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
    "unqualified",
    "unread_in",
]
