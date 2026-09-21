"""What a field annotation says, read off it: the type.

An entity's dataclass fields are its schema, and this module is the
compiler over them. `check_for` turns an annotation into its type check
-- a scalar, a `Literal`, arrays homogeneous or fixed, unions, a nested
object described by a TypedDict or a record dataclass, an object of
undeclared keys, a `NewType` as the type it names: the shapes JSON takes
and no others, which is what keeps it small. `derive_member_types` is
what an entity reads its member table off. Anything finer than a type
-- a bound, a rule about a member, members read together -- is the
entity's own `__post_init__`, in plain code.

Nothing here knows what an entity is. A nested metadata field is the one
shape recognised through `nested_field`, which `_entity` sets.
"""

from __future__ import annotations

# Runtime imports, not `TYPE_CHECKING` ones: the string type aliases below
# (`TypeCheck`, `MemberTypes`) are resolved by `get_type_hints` at class
# creation, and a name that exists only for the type checker is a NameError
# then -- for this package and for any tool introspecting an entity.
import sys
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import is_dataclass
from typing import (
    TYPE_CHECKING,
    Annotated,
    ClassVar,
    Final,
    Literal,
    NewType,
    NotRequired,
    Required,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from typing_extensions import ReadOnly, is_typeddict

from zarr_metadata._common import JSONValue
from zarr_metadata.model._sentinel import UNSET
from zarr_metadata.v3._checks import (
    is_bool,
    is_int,
    is_integer,
    is_json_value,
    is_metadata_field,
    is_str,
    object_of,
    one_of,
    problem,
    sequence_of,
)

if TYPE_CHECKING:
    from zarr_metadata.model._validation import ValidationProblem
    from zarr_metadata.v3._checks import Loc, TypeCheck


class _FromName:
    """The marker behind `FROM_NAME`."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "FROM_NAME"


FROM_NAME: Final = _FromName()
"""Marks a field carried by the metadata envelope's `name`, not its configuration.

    data_type_name: Annotated[str, FROM_NAME]

A member all the same -- `value_problems` judges it -- but not a
configuration key, so it is neither read from nor written to a
`configuration` object. The raw-bytes family is the case: `r<N>` keeps its
width in its name and has no configuration at all.
"""


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


def field_hints(cls: type) -> dict[str, object]:
    """The dataclass fields of `cls`, resolved, base first.

    Each class's own annotations are resolved in that class's module,
    and class variables are skipped *before* resolving, by text -- so a
    `ClassVar` whose annotation names something imported only for the
    type checker cannot fail class creation. `@dataclass` sees the same
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


def is_union(annotation: object) -> bool:
    return get_origin(annotation) in (Union, types.UnionType)


def is_optional(annotation: object) -> bool:
    """Whether a field may be absent: its type admits `UNSET`."""
    inner, _ = strip_annotation(annotation)
    return is_union(inner) and any(arg is UNSET for arg in get_args(inner))


def _no_nested_field(annotation: object) -> bool:
    return False


nested_field: Callable[[object], bool] = _no_nested_field
"""Whether an annotation is a nested metadata field: an entity type, or a union of those with `Opaque`.

The one shape the compiler cannot recognise by itself, because which
classes are entities is `_entity`'s to say; it sets this once at import.
Consulted ahead of every other shape, since an entity is a dataclass too
and must not be walked as a record.
"""


def describe(annotation: object) -> str:
    """The annotation as a message would name it: "an integer", "an object"."""
    inner, _ = strip_annotation(annotation)
    if nested_field(inner):
        return "a metadata field"
    if inner is int:
        return "an integer"
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
        branches = [arg for arg in get_args(inner) if arg is not UNSET]
        return " or ".join(describe(branch) for branch in branches)
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

    None means any shape -- a JSON value, or a union that mixes them.
    """
    inner, _ = strip_annotation(annotation)
    if nested_field(inner):
        return "field"
    if inner is int:
        return "int"
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
    if shape == "bool":
        return isinstance(value, bool)
    if shape == "str":
        return isinstance(value, str)
    if shape == "tuple":
        return isinstance(value, tuple)
    if shape == "mapping":
        return isinstance(value, Mapping)
    return isinstance(value, (str, Mapping))  # "field"


def any_of(branches: Sequence[tuple[object, TypeCheck]], description: str) -> TypeCheck:
    """A member whose type is a union of shapes, judged by the branch it fits.

    The branch whose top-level shape the value has is the one that
    reports -- so an element inside a malformed array is located inside
    the array, rather than the whole array being called wrong. A value
    fitting no branch's shape is reported once, by what was expected.
    """

    def check(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
        fitting = [
            check for annotation, check in branches if has_shape(shape_of(annotation), value)
        ]
        if len(fitting) == 0:
            return problem(loc, f"expected {description}, got {value!r}")
        verdicts = [check(value, loc) for check in fitting]
        return () if any(len(verdict) == 0 for verdict in verdicts) else verdicts[0]

    return check


def fixed_tuple(elements: Sequence[TypeCheck], description: str) -> TypeCheck:
    """A member whose type is an array of a fixed length, checked position by position."""

    def check(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
        if not isinstance(value, tuple) or len(cast("tuple[object, ...]", value)) != len(elements):
            return problem(loc, f"expected {description}, got {value!r}")
        entries = cast("tuple[object, ...]", value)
        return tuple(
            found
            for position, (element, entry) in enumerate(zip(elements, entries, strict=True))
            for found in element(entry, (*loc, position))
        )

    return check


def mapping_of(members: Mapping[str, tuple[bool, TypeCheck]]) -> TypeCheck:
    """A member that is itself an object with declared keys, checked key by key.

    Closed, like every configuration in this package: a key the type does
    not declare is `unknown_key`, a required one missing is `missing_key`,
    both located at the object. Each present member is checked at its own
    key, so a problem inside is located inside.
    """

    def check(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
        if not isinstance(value, Mapping):
            return problem(loc, f"expected an object, got {value!r}")
        entries = cast("Mapping[str, object]", value)
        found: list[ValidationProblem] = []
        for key in entries:
            if key not in members:
                found.extend(problem(loc, f"unexpected key {key!r}", "unknown_key"))
        for key, (required, member) in members.items():
            if key not in entries:
                if required:
                    found.extend(problem(loc, f"missing required key {key!r}", "missing_key"))
                continue
            found.extend(member(entries[key], (*loc, key)))
        return tuple(found)

    return check


def _members_of(annotations: Mapping[str, object]) -> dict[str, tuple[bool, TypeCheck]] | None:
    """A member table for a nested object's keys; None if any key's type has no check."""
    members: dict[str, tuple[bool, TypeCheck]] = {}
    for key, annotation in annotations.items():
        check = check_for(annotation)
        if check is None:
            return None
        inner, _ = strip_annotation(annotation)
        required = get_origin(annotation) is not NotRequired and not is_optional(inner)
        members[key] = (required, check)
    return members


def _compile_literal(inner: object) -> TypeCheck | None:
    # Sorted, because the order `get_args` reports is not the order the
    # `Literal` was written in: two `Literal`s over the same values
    # compare and hash equal, so the first one built anywhere in the
    # process is the one every later one resolves to. The check is a
    # membership test either way; this is so the message listing the
    # values does not depend on import order.
    return one_of(tuple(sorted(cast("tuple[str, ...]", get_args(inner)))))


def _compile_union(inner: object) -> TypeCheck | None:
    branches = [arg for arg in get_args(inner) if arg is not UNSET]
    if len(branches) == 1:
        return check_for(branches[0])
    compiled = [(branch, check_for(branch)) for branch in branches]
    if any(check is None for _, check in compiled):
        return None
    return any_of(
        [(branch, cast("TypeCheck", check)) for branch, check in compiled], describe(inner)
    )


def _compile_tuple(inner: object) -> TypeCheck | None:
    arguments = get_args(inner)
    if len(arguments) == 2 and arguments[1] is Ellipsis:
        element = check_for(arguments[0])
        return None if element is None else sequence_of(element)
    elements = [check_for(argument) for argument in arguments]
    if any(element is None for element in elements):
        return None
    return fixed_tuple([cast("TypeCheck", element) for element in elements], describe(inner))


def _compile_typeddict(inner: object) -> TypeCheck | None:
    members = _members_of(get_type_hints(inner, include_extras=True))
    return None if members is None else mapping_of(members)


def _compile_record(inner: object) -> TypeCheck | None:
    if not isinstance(inner, type):  # pragma: no cover - the predicate says it is
        return None
    members = _members_of(field_hints(inner))
    return None if members is None else mapping_of(members)


def _compile_mapping(inner: object) -> TypeCheck | None:
    # An object of undeclared keys: `Mapping[str, V]`, every value a `V`.
    arguments = get_args(inner)
    if len(arguments) != 2 or arguments[0] is not str:
        return None
    value = check_for(arguments[1])
    return None if value is None else object_of(value)


def _compile_new_type(inner: object) -> TypeCheck | None:
    # A `NewType` is its supertype to a document; the distinction is the
    # code's, for a value it has vouched for.
    return check_for(cast("NewType", inner).__supertype__)


def check_for(annotation: object) -> TypeCheck | None:
    """The type check a field annotation implies, or None if it implies none.

    A small compiler over the shapes JSON takes, and no others: the
    scalars, a `Literal` of names, arrays homogeneous or fixed, unions of
    those, a nested object described by a TypedDict or a record dataclass,
    an object of undeclared keys as `Mapping[str, V]`, a `NewType` as the
    type it names, and a nested metadata field -- an entity type, with or
    without `Opaque`, which `nested_field` recognises. `UNSET` in a union
    says the member may be absent, which is the other half of a table
    entry and is read separately by `is_optional`.

    Closed: an annotation outside these implies no check, and an entity
    declaring one is refused at class creation. The field is written as
    one of these shapes instead, with any finer rule in `__post_init__`.
    """
    inner, _ = strip_annotation(annotation)
    if nested_field(inner):
        return is_metadata_field
    if inner is int:
        return is_int
    if inner is bool:
        return is_bool
    if inner is str:
        return is_str
    if inner is JSONValue:
        return is_json_value
    if get_origin(inner) is Literal:
        return _compile_literal(inner)
    if is_union(inner):
        return _compile_union(inner)
    if get_origin(inner) is tuple:
        return _compile_tuple(inner)
    if is_typeddict(inner):
        return _compile_typeddict(inner)
    if get_origin(inner) in (Mapping, dict):
        return _compile_mapping(inner)
    if isinstance(inner, NewType):
        return _compile_new_type(inner)
    # Last, because `is_dataclass` narrows what pyright knows of `inner`
    # for every line after it.
    if isinstance(inner, type) and is_dataclass(inner):
        return _compile_record(inner)
    return None


def derive_member_types(cls: type) -> tuple[dict[str, tuple[bool, TypeCheck]], list[str]]:
    """The member table an entity's own fields describe.

    Every field is a configuration member unless `FROM_NAME` says it is
    carried by the envelope. Requiredness is whether the type admits
    `UNSET`; the check is whatever `check_for` reads off the type. Also
    returned: the fields no check could be read for, which class
    creation refuses.
    """
    derived: dict[str, tuple[bool, TypeCheck]] = {}
    unread: list[str] = []
    for name, annotation in field_hints(cls).items():
        inner, metadata = strip_annotation(annotation)
        if any(entry is FROM_NAME for entry in metadata):
            continue
        check = check_for(inner)
        if check is None:
            unread.append(name)
            continue
        derived[name] = (not is_optional(inner), check)
    return derived, unread


def element_annotations(inner: object, count: int) -> list[object]:
    """The annotation of each element of a tuple type, one per element held."""
    arguments = get_args(inner)
    if len(arguments) == 2 and arguments[1] is Ellipsis:
        return [arguments[0]] * count
    return list(arguments)


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
    return get_origin(annotation) is ClassVar


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


__all__ = [
    "FROM_NAME",
    "any_of",
    "check_for",
    "declared_class_vars",
    "derive_member_types",
    "describe",
    "element_annotations",
    "field_hints",
    "fixed_tuple",
    "has_shape",
    "is_class_var",
    "is_optional",
    "is_union",
    "mapping_of",
    "nested_field",
    "own_annotations",
    "shape_of",
    "strip_annotation",
]
