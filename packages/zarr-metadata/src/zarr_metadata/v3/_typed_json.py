"""JSON values parsed by type annotation.

A dataclass's fields are its schema, and this module reads that schema.
`parser_for` turns a field annotation into a parser: a function of a
JSON value and its location that returns the typed value and every
problem found with it. The annotations it reads are the shapes JSON
takes and no others -- `int`, `float` for any number, `bool`, `str`,
`JSONValue`, a `Literal` of names, `tuple[T, ...]` and `tuple[T1, T2]`,
a union of those, an object described by a TypedDict or a record
dataclass, `Mapping[str, V]`, a `NewType` as the type it names -- which
is what keeps it small. `UNSET` in a union says the member may be
absent; `is_optional` reads that, and a parser only ever sees a present
value.

Nothing here knows what a metadata entity is. A caller with a shape of
its own -- a field that holds another entity, read through a scope --
passes a `leaf`, which is asked first for every annotation at every
depth; the parser it returns is used as it is.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import is_dataclass
from typing import (
    TYPE_CHECKING,
    Annotated,
    ClassVar,
    Literal,
    NewType,
    NotRequired,
    Required,
    TypeAlias,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from typing_extensions import ReadOnly, TypeIs, is_typeddict

from zarr_metadata._common import JSONValue
from zarr_metadata.model._sentinel import UNSET
from zarr_metadata.model._validation import MetadataValidationError, ValidationProblem, is_json

if TYPE_CHECKING:
    from zarr_metadata.model._validation import ProblemKind


Loc: TypeAlias = "tuple[str | int, ...]"
"""Where in a document a value sits: the keys and indices down to it."""

Parsed: TypeAlias = "tuple[object, tuple[ValidationProblem, ...]]"
"""What a parser returns: the typed value, and every problem found with it."""

Parser: TypeAlias = "Callable[[object, Loc], Parsed]"
"""One value against one annotation, located at `loc`."""

Leaf: TypeAlias = "Callable[[object], Parser | None]"
"""A caller's own shapes: asked first for every annotation, None to decline."""


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


def as_tuples(value: object) -> object:
    """Every JSON array in `value`, at any depth, as a tuple.

    The TypedDicts spell a JSON array as a tuple throughout, so a member
    taken straight from parsed JSON would otherwise hold a list where its
    own type says tuple -- and two documents differing only in that would
    compare unequal.
    """
    if isinstance(value, (list, tuple)):
        entries = cast("list[object] | tuple[object, ...]", value)
        return tuple(as_tuples(entry) for entry in entries)
    if isinstance(value, Mapping):
        entries = cast("Mapping[str, object]", value)
        return {key: as_tuples(entry) for key, entry in entries.items()}
    return value


# --- annotations ---------------------------------------------------------


def strip_annotation(annotation: object) -> tuple[object, tuple[object, ...]]:
    """An annotation's type, and the metadata `Annotated` wrapped it in.

    `NotRequired`, `Required` and `ReadOnly` are qualifiers on a TypedDict
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
        elif origin in (NotRequired, Required, ReadOnly):
            (annotation,) = get_args(annotation)
        else:
            return annotation, tuple(metadata)


def is_not_required(annotation: object) -> bool:
    """Whether a TypedDict key is marked `NotRequired`, under whatever qualifiers."""
    while True:
        origin = get_origin(annotation)
        if origin is NotRequired:
            return True
        if origin is Annotated:
            annotation = get_args(annotation)[0]
        elif origin in (Required, ReadOnly):
            (annotation,) = get_args(annotation)
        else:
            return False


def is_union(annotation: object) -> bool:
    return get_origin(annotation) in (Union, types.UnionType)


def is_optional(annotation: object) -> bool:
    """Whether a field may be absent: its type admits `UNSET`."""
    inner, _ = strip_annotation(annotation)
    return is_union(inner) and any(arg is UNSET for arg in get_args(inner))


def without_unset(inner: object) -> object:
    """The type of a present value: the union less `UNSET`."""
    if not is_union(inner):
        return inner
    arguments = get_args(inner)
    present = [argument for argument in arguments if argument is not UNSET]
    if len(present) == len(arguments):
        return inner
    if len(present) == 1:
        return present[0]
    return Union[tuple(present)]  # noqa: UP007 - built from a tuple, which `|` cannot be


def own_annotations(klass: type) -> dict[str, object]:
    """A class's own annotations, unevaluated.

    From 3.14 a class does not carry an `__annotations__` dict until it is
    asked for one, and asking evaluates every annotation at once -- so a
    `ClassVar` naming something imported only for the type checker would
    fail the whole class. `annotationlib` can hand them back as the text
    they were written as, which is what the callers here want anyway:
    class variables are skipped by text before anything is evaluated.
    Earlier versions leave the dict on the class, strings or values as
    the module chose.
    """
    if sys.version_info >= (3, 14):
        import annotationlib

        return dict(annotationlib.get_annotations(klass, format=annotationlib.Format.STRING))
    return dict(vars(klass).get("__annotations__", {}))


def is_class_var(annotation: object) -> bool:
    """Whether an annotation says `ClassVar`.

    `from __future__ import annotations` leaves them as strings, so this
    reads the text when it gets one -- the same thing `dataclasses` does,
    and for the same reason: resolving the name needs a module namespace
    that is not available while the class is still being built.
    """
    if isinstance(annotation, str):
        stripped = annotation.strip()
        return stripped.startswith(("ClassVar[", "ClassVar", "typing.ClassVar"))
    return annotation is ClassVar or get_origin(annotation) is ClassVar


def field_hints(cls: type) -> dict[str, object]:
    """The dataclass fields of `cls`, resolved, base first.

    Each class's own annotations are resolved in that class's module,
    and class variables are skipped *before* resolving, by text -- so a
    `ClassVar` whose annotation names something imported only for the
    type checker cannot fail registration. `@dataclass` sees the same
    set, in the same order.
    """
    hints: dict[str, object] = {}
    for ancestor in reversed(cls.__mro__):
        raw = {
            name: annotation
            for name, annotation in own_annotations(ancestor).items()
            if not is_class_var(annotation)
        }
        if len(raw) == 0:
            continue
        shell = type("_Fields", (), {"__annotations__": raw, "__module__": ancestor.__module__})
        hints.update(get_type_hints(shell, include_extras=True))
    return hints


def declared_class_vars(cls: type) -> dict[str, type]:
    """Every class variable annotated anywhere in `cls`'s ancestry.

    Mapped to the class that annotated it, so a message can say where the
    requirement comes from. Base first, so a redeclaration names the
    nearest ancestor.
    """
    found: dict[str, type] = {}
    for ancestor in reversed(cls.__mro__):
        for name, annotation in own_annotations(ancestor).items():
            if is_class_var(annotation):
                found[name] = ancestor
    return found


# --- what a message calls a shape, and which shape a value has -------------


def describe(annotation: object) -> str:
    """The annotation as a message would name it: "an integer", "an object"."""
    inner = without_unset(strip_annotation(annotation)[0])
    if inner is int:
        return "an integer"
    if inner is float:
        return "a number"
    if inner is bool:
        return "a boolean"
    if inner is str:
        return "a string"
    if inner is JSONValue:
        return "a JSON value"
    origin = get_origin(inner)
    if origin is Literal:
        return f"one of {tuple(sorted(get_args(inner)))!r}"
    if is_union(inner):
        return " or ".join(describe(branch) for branch in get_args(inner))
    if origin is tuple:
        arguments = get_args(inner)
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return f"an array of {describe(arguments[0])} elements"
        if len(arguments) == 2:
            return f"a [{describe(arguments[0])}, {describe(arguments[1])}] pair"
        return f"an array of {len(arguments)} elements"
    if origin in (Mapping, dict):
        return "an object"
    if isinstance(inner, NewType):
        return describe(inner.__supertype__)
    if is_typeddict(inner) or is_dataclass(inner):
        return "an object"
    return "a value"


def shape_of(annotation: object) -> str | None:
    """The top-level JSON shape an annotation admits, for choosing a union branch.

    None means any shape -- a JSON value, a union that mixes them, or a
    shape the caller's `leaf` reads.
    """
    inner = without_unset(strip_annotation(annotation)[0])
    if inner is int:
        return "int"
    if inner is float:
        return "number"
    if inner is bool:
        return "bool"
    if inner is str:
        return "str"
    origin = get_origin(inner)
    if origin is Literal:
        values = get_args(inner)
        return "int" if all(isinstance(value, int) for value in values) else "str"
    if origin is tuple:
        return "tuple"
    if origin in (Mapping, dict):
        return "mapping"
    if isinstance(inner, NewType):
        return shape_of(inner.__supertype__)
    if is_typeddict(inner) or is_dataclass(inner):
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
    if shape == "tuple":
        return isinstance(value, tuple)
    return isinstance(value, Mapping)  # "mapping"


# --- the parsers ---------------------------------------------------------


def _scalar(description: str, admits: Callable[[object], bool]) -> Parser:
    def parse(value: object, loc: Loc) -> Parsed:
        if admits(value):
            return value, ()
        return value, problem(loc, f"expected {description}, got {value!r}")

    return parse


_INTEGER: Parser = _scalar("an integer", is_integer)
_NUMBER: Parser = _scalar(
    "a number", lambda value: not isinstance(value, bool) and isinstance(value, (int, float))
)
_BOOLEAN: Parser = _scalar("a boolean", lambda value: isinstance(value, bool))
_STRING: Parser = _scalar("a string", lambda value: isinstance(value, str))
_JSON: Parser = _scalar("a JSON value", is_json)


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


def any_of(branches: Sequence[tuple[object, Parser]], description: str) -> Parser:
    """A member whose type is a union of shapes, parsed by the branch it fits.

    The branch whose top-level shape the value has is the one that
    reports -- so an element inside a malformed array is located inside
    the array, rather than the whole array being called wrong. A value
    fitting no branch's shape is reported once, by what was expected;
    one fitting several is parsed by the first that accepts it, and
    reported by the first that does not.
    """

    def parse(value: object, loc: Loc) -> Parsed:
        first: Parsed | None = None
        for annotation, branch in branches:
            if not has_shape(shape_of(annotation), value):
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


Members: TypeAlias = "Mapping[str, tuple[bool, Parser]]"
"""An object's declared keys: whether each is required, and its parser."""


def _keys(
    members: Members, entries: Mapping[str, object], loc: Loc
) -> tuple[dict[str, object], tuple[ValidationProblem, ...]]:
    """The declared keys of one object, each parsed at its own key.

    Closed, like every configuration in this package: a key the type does
    not declare is `unknown_key`, a required one missing is `missing_key`,
    both located at the key. An optional key left out is parsed as
    `UNSET`, so a record never depends on a default for it.
    """
    parsed: dict[str, object] = {}
    found: list[ValidationProblem] = []
    for key in entries:
        if key not in members:
            found.extend(problem((*loc, key), f"unexpected key {key!r}", "unknown_key"))
    for key, (required, member) in members.items():
        if key not in entries:
            if required:
                found.extend(problem((*loc, key), f"missing required key {key!r}", "missing_key"))
            else:
                parsed[key] = UNSET
            continue
        item, problems = member(entries[key], (*loc, key))
        parsed[key] = item
        found.extend(problems)
    return parsed, tuple(found)


def object_of(members: Members) -> Parser:
    """A member that is itself an object with declared keys, kept as the mapping it came as.

    A key the type does not declare is reported and kept: the member
    still says what the document said.
    """

    def parse(value: object, loc: Loc) -> Parsed:
        if not isinstance(value, Mapping):
            return value, problem(loc, f"expected an object, got {value!r}")
        entries = cast("Mapping[str, object]", value)
        parsed, found = _keys(members, entries, loc)
        return {**entries, **{key: item for key, item in parsed.items() if key in entries}}, found

    return parse


def record_of(record: Callable[..., object], members: Members) -> Parser:
    """A member that is itself an object with declared keys, built as a dataclass.

    Built only from an object whose every key read; otherwise the value
    comes back as it came, with the reasons. A record that refuses its
    own values -- a `__post_init__` raising `MetadataValidationError` --
    is reported the same way, located under the object.
    """

    def parse(value: object, loc: Loc) -> Parsed:
        if not isinstance(value, Mapping):
            return value, problem(loc, f"expected an object, got {value!r}")
        entries = cast("Mapping[str, object]", value)
        parsed, found = _keys(members, entries, loc)
        if any(entry.kind != "unknown_key" for entry in found):
            return entries, found
        try:
            return record(**parsed), found
        except MetadataValidationError as refused:
            return entries, (
                *found,
                *(
                    ValidationProblem((*loc, *entry.loc), entry.message, entry.kind)
                    for entry in refused.problems
                ),
            )

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


def _members_of(annotations: Mapping[str, object], leaf: Leaf) -> Members | None:
    """A member table for an object's keys; None if any key's type has no parser."""
    members: dict[str, tuple[bool, Parser]] = {}
    for key, annotation in annotations.items():
        parser = parser_for(annotation, leaf)
        if parser is None:
            return None
        required = not is_not_required(annotation) and not is_optional(annotation)
        members[key] = (required, parser)
    return members


def _union(inner: object, leaf: Leaf) -> Parser | None:
    compiled = [(branch, parser_for(branch, leaf)) for branch in get_args(inner)]
    branches = [(branch, parser) for branch, parser in compiled if parser is not None]
    if len(branches) != len(compiled):
        return None
    return any_of(branches, describe(inner))


def _tuple(inner: object, leaf: Leaf) -> Parser | None:
    arguments = get_args(inner)
    if len(arguments) == 2 and arguments[1] is Ellipsis:
        element = parser_for(arguments[0], leaf)
        return None if element is None else sequence_of(element)
    compiled = [parser_for(argument, leaf) for argument in arguments]
    elements = [element for element in compiled if element is not None]
    if len(elements) != len(compiled):
        return None
    return fixed_tuple(elements, describe(inner))


def _mapping(inner: object, leaf: Leaf) -> Parser | None:
    arguments = get_args(inner)
    if len(arguments) != 2 or arguments[0] is not str:
        return None
    value = parser_for(arguments[1], leaf)
    return None if value is None else mapping_of(value)


def _no_leaf(annotation: object) -> Parser | None:
    return None


def parser_for(annotation: object, leaf: Leaf = _no_leaf) -> Parser | None:
    """The parser a field annotation implies, or None if it implies none.

    A small compiler over the shapes JSON takes and no others, listed in
    the module docstring. `leaf` is asked first, here and at every depth
    -- inside a union, an array, an object -- and what it returns is
    used as it is. Closed: an annotation outside these implies no
    parser, and an entity declaring one is refused at registration.
    The field is written as one of these shapes instead, with any finer
    rule in `__post_init__`.
    """
    inner = without_unset(strip_annotation(annotation)[0])
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
    if inner is JSONValue:
        return _JSON
    if get_origin(inner) is Literal:
        # Sorted, because the order `get_args` reports is not the order
        # the `Literal` was written in: two `Literal`s over the same
        # values compare and hash equal, so the first one built anywhere
        # in the process is the one every later one resolves to. The
        # parse is a membership test either way; this is so the message
        # listing the values does not depend on import order.
        return one_of(tuple(sorted(get_args(inner), key=repr)))
    if is_union(inner):
        return _union(inner, leaf)
    if get_origin(inner) is tuple:
        return _tuple(inner, leaf)
    if is_typeddict(inner):
        members = _members_of(get_type_hints(inner, include_extras=True), leaf)
        return None if members is None else object_of(members)
    if get_origin(inner) in (Mapping, dict):
        return _mapping(inner, leaf)
    if isinstance(inner, NewType):
        # A `NewType` is its supertype to a document; the distinction is
        # the code's, for a value it has vouched for.
        return parser_for(inner.__supertype__, leaf)
    # Last, because `is_dataclass` narrows what pyright knows of `inner`
    # for every line after it.
    if isinstance(inner, type) and is_dataclass(inner):
        members = _members_of(field_hints(inner), leaf)
        return None if members is None else record_of(inner, members)
    return None


def parser(annotation: object, leaf: Leaf = _no_leaf) -> Parser:
    """The parser a field annotation implies; `TypeError` if it implies none."""
    found = parser_for(annotation, leaf)
    if found is None:
        msg = f"{annotation!r} is not a shape JSON takes"
        raise TypeError(msg)
    return found


__all__ = [
    "Leaf",
    "Loc",
    "Members",
    "Parsed",
    "Parser",
    "any_of",
    "as_tuples",
    "declared_class_vars",
    "describe",
    "field_hints",
    "fixed_tuple",
    "has_shape",
    "is_class_var",
    "is_integer",
    "is_not_required",
    "is_optional",
    "is_union",
    "mapping_of",
    "object_of",
    "one_of",
    "own_annotations",
    "parser",
    "parser_for",
    "problem",
    "record_of",
    "sequence_of",
    "shape_of",
    "strip_annotation",
    "without_unset",
]
