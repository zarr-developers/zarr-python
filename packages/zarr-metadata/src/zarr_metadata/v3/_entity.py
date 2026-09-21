"""What every metadata entity can do for itself.

A codec, data type, chunk grid or chunk key encoding is three things at
once: a JSON shape, a set of constraints on the values in that shape, and
a canonical spelling. Keeping the three apart put one entity's knowledge
in four modules and needed a table per axis plus a drift test per table to
hold them together. Here they are one class per entity, and the class is
where methods bind:

- `coerce` is **type-space**: raw metadata in, the entity or the reasons
  it is not that entity out.
- `problems` is **value-space**: the entity is well-typed by construction,
  so this only asks whether its values are in range.
- `to_json` is **canonical**: the simplest spelling meaning the same, typed
  as the entity's own object TypedDict.

The TypedDicts stay: they model the JSON form, and the correspondence is
exact in both directions. A configuration TypedDict unpacked is the
dataclass constructor's signature, and the object TypedDict is what
`to_json` returns. `tests/v3/test_entities.py` asserts the first, so the
two cannot drift.

Everything that needs the document or the codec chain stays outside, in
`zarr_metadata.rules`, because an entity cannot answer it alone.

`coerce` takes a `Context`: the entities in scope for this reading. Most
entities ignore it -- a `gzip` codec is a `gzip` codec whatever else is
registered -- but the ones whose configuration contains other entities do
not. A `struct` data type holds field data types and a `sharding_indexed`
codec holds two codec pipelines, and neither can coerce its own
configuration without knowing what names are in scope inside it.

The shared plumbing lives here too: the member checks every entity needs,
the compiler that reads them off a field annotation, and the walk over a
configuration that applies them. What stays with the entity is its fields
-- the part that is about blosc rather than about entities -- and
everything the layer knows about a member is read from those.
"""

from __future__ import annotations

# Runtime imports, not `TYPE_CHECKING` ones: the string type aliases below
# (`TypeCheck`, `MemberTypes`) are resolved by `get_type_hints` at class
# creation, and a name that exists only for the type checker is a NameError
# then -- for this package and for any tool introspecting an entity.
import sys
import types
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import MISSING, Field, dataclass, fields, is_dataclass, replace
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Annotated,
    ClassVar,
    Final,
    Literal,
    NotRequired,
    Required,
    TypeAlias,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from typing_extensions import ReadOnly, TypeIs, is_typeddict

from zarr_metadata._common import JSONValue
from zarr_metadata.model._sentinel import UNSET
from zarr_metadata.model._validation import (
    MetadataValidationError,
    ValidationProblem,
    is_json,
)
from zarr_metadata.v3._parts import ChunkGrid

if TYPE_CHECKING:
    from typing import Self

    from zarr_metadata.model._validation import ProblemKind
    from zarr_metadata.v3._common import ZarrV3MetadataFieldJSON
    from zarr_metadata.v3._parts import ArrayParts
    from zarr_metadata.v3._registry import Context

EntityT = TypeVar("EntityT", bound="MetadataEntity")

# A real alias, not a string one: entity modules subscript it as
# `Coerced[Self]` in a return annotation, and not all of them defer
# annotation evaluation.
Coerced: TypeAlias = tuple[EntityT | None, tuple[ValidationProblem, ...]]
"""The entity if it could be built, and every problem found.

One direction holds: no entity means at least one problem. The converse
does not -- a survivable problem (an unknown key, an optional member of
the wrong type) comes back *with* the entity, because the entity is
still readable and saying so is more useful than refusing.

So test `entity is None` to decide whether to go on reading, and test the
problems to decide the verdict. They are different questions.
"""

Loc: TypeAlias = "tuple[str | int, ...]"

ExtensionPointField = Literal[
    "data_type", "chunk_grid", "chunk_key_encoding", "codecs", "storage_transformers"
]
"""The v3 array metadata fields whose values name an extension.

Here rather than in `_extension_points` because an entity that contains
other entities has to say which point it is reading them at, and
`_extension_points` also folds `r<N>` names -- which means importing the
data types, which import this.
"""

# Left to infer their `Literal` types rather than widened to
# `ExtensionPointField`: `Context.coerce` overloads on the field, so a
# call written with one of these constants gets the entity type back
# rather than the base. They are still assignable to the alias.
DATA_TYPE: Final = "data_type"
CHUNK_GRID: Final = "chunk_grid"
CHUNK_KEY_ENCODING: Final = "chunk_key_encoding"
CODECS: Final = "codecs"
STORAGE_TRANSFORMERS: Final = "storage_transformers"

StorageClass = Literal["single_byte", "multi_byte", "variable_length"]
"""How one scalar of a data type occupies bytes.

`single_byte` and `multi_byte` are both fixed-size; they differ only in
whether a byte order applies, which is what the `bytes` codec's `endian`
member is about.
"""

CodecKind = Literal["array_array", "array_bytes", "bytes_bytes"]
"""The three pipeline positions the v3 spec sorts codecs into.

Declared by each codec, which is why there is no table of it: a name
does not have a pipeline position, a codec does.
"""

TypeCheck: TypeAlias = "Callable[[object, Loc], tuple[ValidationProblem, ...]]"
"""Whether one value has the type a member declares, and where if not."""

MemberTypes: TypeAlias = "Mapping[str, tuple[bool, TypeCheck]]"
"""Per configuration member: whether it is required, and its type check."""


def problem(
    loc: Loc, message: str, kind: ProblemKind = "invalid_type"
) -> tuple[ValidationProblem, ...]:
    """One problem, as the tuple every check returns."""
    return (ValidationProblem(loc, message, kind),)


def is_integer(value: object) -> TypeIs[int]:
    """A JSON integer: an `int`, and not a `bool`.

    `True` is an `int` in Python and `true` is not a number in JSON, so
    the two have to be told apart everywhere a number is expected.
    """
    return not isinstance(value, bool) and isinstance(value, int)


def is_int(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
    """An integer, and not a bool -- JSON `true` is not the integer 1."""
    if not is_integer(value):
        return problem(loc, f"expected an integer, got {value!r}")
    return ()


def is_str(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
    if not isinstance(value, str):
        return problem(loc, f"expected a string, got {value!r}")
    return ()


def is_bool(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
    if not isinstance(value, bool):
        return problem(loc, f"expected a boolean, got {value!r}")
    return ()


def is_json_value(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
    """Any JSON value at all -- the widest type a member can declare."""
    if not is_json(value):
        return problem(loc, f"expected a JSON value, got {value!r}")
    return ()


def one_of(allowed: tuple[str, ...]) -> TypeCheck:
    """A member whose type is a closed set of names."""

    def check(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
        if value not in allowed:
            return problem(loc, f"expected one of {allowed!r}, got {value!r}", "invalid_value")
        return ()

    return check


def sequence_of(element: TypeCheck) -> TypeCheck:
    """A member whose type is a sequence, checked element by element."""

    def check(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
        if not isinstance(value, (list, tuple)):
            return problem(loc, f"expected a sequence, got {value!r}")
        elements: tuple[object, ...] = tuple(cast("list[object] | tuple[object, ...]", value))
        return tuple(
            found for index, entry in enumerate(elements) for found in element(entry, (*loc, index))
        )

    return check


def _as_tuples(value: object) -> object:
    """Every JSON array in `value`, at any depth, as a tuple.

    The TypedDicts spell a JSON array as a tuple throughout, so a member
    taken straight from parsed JSON would otherwise hold a list where its
    own type says tuple -- and two documents differing only in that would
    compare unequal.
    """
    if isinstance(value, (list, tuple)):
        entries = cast("list[object] | tuple[object, ...]", value)
        return tuple(_as_tuples(entry) for entry in entries)
    if isinstance(value, Mapping):
        entries = cast("Mapping[str, object]", value)
        return {key: _as_tuples(entry) for key, entry in entries.items()}
    return value


def coerce_members(
    configuration: Mapping[str, object], types: MemberTypes
) -> tuple[dict[str, object], tuple[ValidationProblem, ...], frozenset[str]]:
    """The members `types` declares, taken from `configuration`.

    Returns what was accepted, every problem found, and the names of the
    required members that could not be read. Three kinds of problem, and
    they differ in that last part:

    - a key the entity does not declare says the value carries something
      extra, not that it is wrong;
    - an *optional* member of the wrong type leaves that member absent,
      and everything else about the entity is still readable -- a bad
      `index_location` says nothing about whether a shard's pipelines
      are well formed, and silencing them would lose a real judgment;
    - a *required* member missing or of the wrong type does stop it.
      There is no honest reading of a `blosc` whose level is a string.
    """
    problems: list[ValidationProblem] = []
    members: dict[str, object] = {}
    unreadable: set[str] = set()
    for key in configuration:
        if key not in types:
            problems.extend(
                problem(("configuration", key), f"unexpected key {key!r}", "unknown_key")
            )
    for key, (required, check) in types.items():
        if key not in configuration:
            if required:
                problems.extend(
                    problem(("configuration", key), f"missing required key {key!r}", "missing_key")
                )
                unreadable.add(key)
            continue
        # Normalized before the check, so a check only ever sees the tuples
        # the TypedDicts declare -- never the lists raw JSON arrives as.
        value = _as_tuples(configuration[key])
        found = check(value, ("configuration", key))
        problems.extend(found)
        # An unknown key says the value carries something extra, not that
        # it is the wrong type -- so the member is still readable, and
        # dropping it here would make `to_json` lose what was written.
        if all(entry.kind == "unknown_key" for entry in found):
            members[key] = value
        elif required:
            unreadable.add(key)
    return members, tuple(problems), frozenset(unreadable)


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


@dataclass(frozen=True, slots=True)
class Ge:
    """`Annotated[int, Ge(1)]`: the value is at least `bound`."""

    bound: int | float


@dataclass(frozen=True, slots=True)
class Gt:
    """`Annotated[int, Gt(0)]`: the value is more than `bound`."""

    bound: int | float


@dataclass(frozen=True, slots=True)
class Le:
    """`Annotated[int, Le(9)]`: the value is at most `bound`."""

    bound: int | float


@dataclass(frozen=True, slots=True)
class Lt:
    """`Annotated[int, Lt(10)]`: the value is less than `bound`."""

    bound: int | float


@dataclass(frozen=True, slots=True)
class Interval:
    """`Annotated[int, Interval(ge=0, le=9)]`: the value lies within these bounds.

    These five are the `annotated_types` vocabulary -- what pydantic reads
    and msgspec's `Meta` mirrors -- so a reader recognises them. Defined
    here rather than imported, so the package keeps its one dependency.
    A bound is a value rule: it runs only once the member has the type it
    declared, at whatever depth the annotation puts it, so a bound on an
    array's element type judges each element at its own position.
    """

    ge: int | float | None = None
    gt: int | float | None = None
    le: int | float | None = None
    lt: int | float | None = None


def _strip(annotation: object) -> tuple[object, tuple[object, ...]]:
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


def _own_annotations(klass: type) -> dict[str, object]:
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


def _field_hints(cls: type) -> dict[str, object]:
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
            for name, annotation in _own_annotations(ancestor).items()
            if not _is_class_var(annotation)
        }
        if len(raw) == 0:
            continue
        shell = type("_Fields", (), {"__annotations__": raw, "__module__": ancestor.__module__})
        hints.update(get_type_hints(shell, include_extras=True))
    return hints


def _is_union(annotation: object) -> bool:
    return get_origin(annotation) in (Union, types.UnionType)


def is_optional(annotation: object) -> bool:
    """Whether a field may be absent: its type admits `UNSET`."""
    inner, _ = _strip(annotation)
    return _is_union(inner) and any(arg is UNSET for arg in get_args(inner))


def _is_entity_type(candidate: object) -> bool:
    return candidate is Opaque or (
        isinstance(candidate, type) and issubclass(candidate, MetadataEntity)
    )


def _is_entity_or_opaque(candidates: Sequence[object]) -> bool:
    """A nested metadata field: some entity kind, optionally with `Opaque`."""
    return (
        len(candidates) != 0
        and all(_is_entity_type(candidate) for candidate in candidates)
        and any(candidate is not Opaque for candidate in candidates)
    )


def is_metadata_field(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
    """A nested metadata field: a bare name or a named-configuration object.

    Only the envelope's shape. Which entity the name denotes, and whether
    its configuration is well formed, is settled when the containing
    entity reads it in scope.
    """
    if not isinstance(value, (str, Mapping)):
        return problem(loc, f"expected a metadata field, got {value!r}")
    return ()


def describe(annotation: object) -> str:
    """The annotation as a message would name it: "an integer", "an object"."""
    inner, _ = _strip(annotation)
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
    if _is_union(inner):
        branches = [arg for arg in get_args(inner) if arg is not UNSET]
        if _is_entity_or_opaque(branches):
            return "a metadata field"
        return " or ".join(describe(branch) for branch in branches)
    if origin is tuple:
        arguments = get_args(inner)
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return f"an array of {describe(arguments[0])} elements"
        if len(arguments) == 2:
            return f"a [{describe(arguments[0])}, {describe(arguments[1])}] pair"
        return f"an array of {len(arguments)} elements"
    if _is_entity_type(inner):
        return "a metadata field"
    if is_typeddict(inner) or is_dataclass(inner):
        return "an object"
    return "a value"


def _shape(annotation: object) -> str | None:
    """The top-level JSON shape an annotation admits, for choosing a union branch.

    None means any shape -- a JSON value, or a union that mixes them.
    """
    inner, _ = _strip(annotation)
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
    if _is_entity_type(inner):
        return "field"
    if is_typeddict(inner) or is_dataclass(inner):
        return "mapping"
    return None


def _has_shape(shape: str | None, value: object) -> bool:
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
        fitting = [check for annotation, check in branches if _has_shape(_shape(annotation), value)]
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
        inner, _ = _strip(annotation)
        required = get_origin(annotation) is not NotRequired and not is_optional(inner)
        members[key] = (required, check)
    return members


CheckCompiler: TypeAlias = "Callable[[object], TypeCheck | None]"
"""Turns one annotation into its type check -- or None, to decline it after all."""

_CHECK_COMPILERS: Final[list[tuple[Callable[[object], bool], CheckCompiler]]] = []
"""The shapes `check_for` reads, each as (does this annotation have it?, compile it).

Consulted front to back. The built-in shapes are appended below in the
order they must be tried -- a nested metadata field before a record,
because `Opaque` is itself a dataclass -- and `register_check` puts a
registration in front of all of them, so the newest one wins.
"""


def register_check(predicate: Callable[[object], bool], compile: CheckCompiler) -> None:
    """Teach `check_for` an annotation shape it does not read.

        Hex = NewType("Hex", str)
        register_check(lambda annotation: annotation is Hex, lambda annotation: is_hex)

    `predicate` sees the annotation with `Annotated`, `NotRequired` and
    `ReadOnly` peeled; `compile` returns the check for it, calling
    `check_for` itself for any shape inside. A registration is consulted
    before every built-in one, so a package can also replace how a
    built-in shape is judged. The same door the built-ins came through,
    which is what makes the set of shapes open rather than this module's.
    """
    _CHECK_COMPILERS.insert(0, (predicate, compile))


def _builtin(predicate: Callable[[object], bool]) -> Callable[[CheckCompiler], CheckCompiler]:
    """Register a built-in shape, in the order written."""

    def append(compile: CheckCompiler) -> CheckCompiler:
        _CHECK_COMPILERS.append((predicate, compile))
        return compile

    return append


@_builtin(lambda inner: inner is int)
def _compile_int(inner: object) -> TypeCheck | None:
    return is_int


@_builtin(lambda inner: inner is bool)
def _compile_bool(inner: object) -> TypeCheck | None:
    return is_bool


@_builtin(lambda inner: inner is str)
def _compile_str(inner: object) -> TypeCheck | None:
    return is_str


@_builtin(lambda inner: inner is JSONValue)
def _compile_json_value(inner: object) -> TypeCheck | None:
    return is_json_value


@_builtin(lambda inner: get_origin(inner) is Literal)
def _compile_literal(inner: object) -> TypeCheck | None:
    # Sorted, because the order `get_args` reports is not the order the
    # `Literal` was written in: two `Literal`s over the same values
    # compare and hash equal, so the first one built anywhere in the
    # process is the one every later one resolves to. The check is a
    # membership test either way; this is so the message listing the
    # values does not depend on import order.
    return one_of(tuple(sorted(cast("tuple[str, ...]", get_args(inner)))))


@_builtin(_is_union)
def _compile_union(inner: object) -> TypeCheck | None:
    branches = [arg for arg in get_args(inner) if arg is not UNSET]
    if len(branches) == 1:
        return check_for(branches[0])
    if _is_entity_or_opaque(branches):
        return is_metadata_field
    compiled = [(branch, check_for(branch)) for branch in branches]
    if any(check is None for _, check in compiled):
        return None
    return any_of(
        [(branch, cast("TypeCheck", check)) for branch, check in compiled], describe(inner)
    )


@_builtin(lambda inner: get_origin(inner) is tuple)
def _compile_tuple(inner: object) -> TypeCheck | None:
    arguments = get_args(inner)
    if len(arguments) == 2 and arguments[1] is Ellipsis:
        element = check_for(arguments[0])
        return None if element is None else sequence_of(element)
    elements = [check_for(argument) for argument in arguments]
    if any(element is None for element in elements):
        return None
    return fixed_tuple([cast("TypeCheck", element) for element in elements], describe(inner))


# A nested metadata field, before the record shape: `Opaque` is itself a
# dataclass, and an entity type must not be walked as one either.
@_builtin(_is_entity_type)
def _compile_entity(inner: object) -> TypeCheck | None:
    return is_metadata_field


@_builtin(is_typeddict)
def _compile_typeddict(inner: object) -> TypeCheck | None:
    members = _members_of(get_type_hints(inner, include_extras=True))
    return None if members is None else mapping_of(members)


@_builtin(lambda inner: isinstance(inner, type) and is_dataclass(inner))
def _compile_record(inner: object) -> TypeCheck | None:
    if not isinstance(inner, type):  # pragma: no cover - the predicate says it is
        return None
    members = _members_of(_field_hints(inner))
    return None if members is None else mapping_of(members)


def check_for(annotation: object) -> TypeCheck | None:
    """The type check a field annotation implies, or None if it implies none.

    A small compiler over the shapes this package's metadata takes: the
    JSON scalars, a `Literal` of names, arrays homogeneous or fixed,
    unions of those, a nested object described by a TypedDict or a
    record dataclass, and a nested metadata field -- an entity type,
    with or without `Opaque`. `UNSET` in a union says the member may be
    absent, which is the other half of a table entry and is read
    separately by `is_optional`.

    Open: each shape is a registration in `_CHECK_COMPILERS`, and
    `register_check` adds one from outside. None for an annotation no
    registration claims, which the entity then declares a check for by
    hand.
    """
    inner, _ = _strip(annotation)
    for predicate, compile in _CHECK_COMPILERS:
        if predicate(inner):
            return compile(inner)
    return None


def derive_member_types(cls: type) -> tuple[dict[str, tuple[bool, TypeCheck]], list[str]]:
    """The member table an entity's own fields describe.

    Every field is a configuration member unless `FROM_NAME` says it is
    carried by the envelope. Requiredness is whether the type admits
    `UNSET`; the check is whatever `check_for` reads off the type. Also
    returned: the fields no check could be read for, which the entity
    must declare by hand.
    """
    derived: dict[str, tuple[bool, TypeCheck]] = {}
    unread: list[str] = []
    for name, annotation in _field_hints(cls).items():
        inner, metadata = _strip(annotation)
        if any(entry is FROM_NAME for entry in metadata):
            continue
        check = check_for(inner)
        if check is None:
            unread.append(name)
            continue
        derived[name] = (not is_optional(inner), check)
    return derived, unread


MemberRule: TypeAlias = "Callable[..., tuple[ValidationProblem, ...]]"
"""A rule about one member: takes its value, reports relative to it."""

_RULE_MEMBERS: Final[dict[object, tuple[str, ...]]] = {}
"""Which members each `@validates` rule is about, keyed by the function.

A side table rather than an attribute on the function, so the decorator
hands back exactly what it was given -- the declared signature survives,
and the type checker keeps checking the body and its callers.
"""

_Rule = TypeVar("_Rule", bound="Callable[..., tuple[ValidationProblem, ...]]")


def validates(*members: str) -> Callable[[_Rule], _Rule]:
    """Mark a static rule as being about one member, or several alike.

        @staticmethod
        @validates("order")
        def _order_permutes_itself(order: tuple[int, ...]) -> tuple[ValidationProblem, ...]:
            ...

    The rule receives the member's value, already of the type the field
    declares, and only when the member is present; it reports relative
    to the member, so a problem with an empty location is about the
    member itself. Naming several members applies the one rule to each.
    A rule that reads two members together is `value_problems`.
    """

    def mark(rule: _Rule) -> _Rule:
        _RULE_MEMBERS[rule] = members
        return rule

    return mark


def _bound_check(metadata: Sequence[object]) -> TypeCheck | None:
    """The check the bound markers among an annotation's metadata imply, or None."""
    ge = gt = le = lt = None
    for marker in metadata:
        if isinstance(marker, Ge):
            ge = marker.bound
        elif isinstance(marker, Gt):
            gt = marker.bound
        elif isinstance(marker, Le):
            le = marker.bound
        elif isinstance(marker, Lt):
            lt = marker.bound
        elif isinstance(marker, Interval):
            ge = marker.ge if marker.ge is not None else ge
            gt = marker.gt if marker.gt is not None else gt
            le = marker.le if marker.le is not None else le
            lt = marker.lt if marker.lt is not None else lt
    if ge is None and gt is None and le is None and lt is None:
        return None
    if ge is not None and le is not None and gt is None and lt is None:
        expectation = f"an integer in [{ge}, {le}]"
    else:
        comparisons = [
            text
            for bound, text in (
                (ge, f">= {ge}"),
                (gt, f"> {gt}"),
                (le, f"<= {le}"),
                (lt, f"< {lt}"),
            )
            if bound is not None
        ]
        expectation = "an integer " + " and ".join(comparisons)

    def check(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
        # Not a number: the type check's finding, not this one's.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return ()
        within_bounds = (
            (ge is None or value >= ge)
            and (gt is None or value > gt)
            and (le is None or value <= le)
            and (lt is None or value < lt)
        )
        if within_bounds:
            return ()
        return problem(loc, f"expected {expectation}, got {value}", "invalid_value")

    return check


def value_check_for(annotation: object) -> TypeCheck | None:
    """The value check an annotation's metadata implies, at any depth, or None.

    Over the shapes as the entity holds them, not as the JSON spells
    them: this runs after every member has its type and every nested
    entity has been read, so a record is a dataclass instance here and
    an entity is skipped -- it is valid by construction.
    """
    inner, metadata = _strip(annotation)
    own = _bound_check(metadata)
    below: TypeCheck | None = None
    if _is_union(inner):
        branches = [
            (branch, value_check_for(branch)) for branch in get_args(inner) if branch is not UNSET
        ]
        if any(check is not None for _, check in branches):

            def by_branch(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
                for branch, check in branches:
                    if check is not None and _has_shape(_shape(branch), value):
                        return check(value, loc)
                return ()

            below = by_branch
    elif get_origin(inner) is tuple:
        arguments = get_args(inner)
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            element = value_check_for(arguments[0])
            if element is not None:
                each = element

                def per_element(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
                    entries = cast("tuple[object, ...]", value)
                    return tuple(
                        found
                        for position, entry in enumerate(entries)
                        for found in each(entry, (*loc, position))
                    )

                below = per_element
        else:
            positions = [value_check_for(argument) for argument in arguments]
            if any(check is not None for check in positions):

                def per_position(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
                    entries = cast("tuple[object, ...]", value)
                    return tuple(
                        found
                        for position, (check, entry) in enumerate(
                            zip(positions, entries, strict=True)
                        )
                        if check is not None
                        for found in check(entry, (*loc, position))
                    )

                below = per_position
    elif isinstance(inner, type) and is_dataclass(inner) and not _is_entity_type(inner):
        members = {
            name: check
            for name, field_annotation in _field_hints(inner).items()
            if (check := value_check_for(field_annotation)) is not None
        }
        if len(members) != 0:

            def per_field(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
                return tuple(
                    found
                    for name, check in members.items()
                    if (held := getattr(value, name)) is not UNSET
                    for found in check(held, (*loc, name))
                )

            below = per_field
    if own is None and below is None:
        return None
    if below is None:
        return own
    if own is None:
        return below
    outer, inner_check = own, below

    def both(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
        return (*outer(value, loc), *inner_check(value, loc))

    return both


def _contains_entity(annotation: object) -> bool:
    """Whether a value of this type holds a nested metadata field anywhere in it."""
    inner, _ = _strip(annotation)
    if _is_entity_type(inner):
        return True
    origin = get_origin(inner)
    if _is_union(inner):
        return any(_contains_entity(arg) for arg in get_args(inner) if arg is not UNSET)
    if origin is tuple:
        return any(_contains_entity(arg) for arg in get_args(inner) if arg is not Ellipsis)
    if is_typeddict(inner):
        return any(
            _contains_entity(value) for value in get_type_hints(inner, include_extras=True).values()
        )
    if isinstance(inner, type) and is_dataclass(inner):
        return any(_contains_entity(value) for value in _field_hints(inner).values())
    return False


def _as_entity_kind(candidate: object) -> type[MetadataEntity] | None:
    """`candidate` as an entity type, or None if it is not one.

    In a function of its own so that the `isinstance`/`issubclass` pair
    narrows this parameter and not the caller's variable, which the
    caller goes on to read as the annotation it is.
    """
    if isinstance(candidate, type) and issubclass(candidate, MetadataEntity):
        return candidate
    return None


def _entity_kinds(annotation: object) -> list[type[MetadataEntity]]:
    """Every entity type an annotation names, at any depth."""
    inner, _ = _strip(annotation)
    kind = _as_entity_kind(inner)
    if kind is not None:
        return [kind]
    origin = get_origin(inner)
    arguments: tuple[object, ...] = get_args(inner)
    if _is_union(inner):
        return [kind for arg in arguments if arg is not UNSET for kind in _entity_kinds(arg)]
    if origin is tuple:
        return [kind for arg in arguments if arg is not Ellipsis for kind in _entity_kinds(arg)]
    if isinstance(inner, type) and is_dataclass(inner) and not _is_entity_type(inner):
        return [kind for value in _field_hints(inner).values() for kind in _entity_kinds(value)]
    return []


def _point_of(kind: type[MetadataEntity]) -> ExtensionPointField:
    """The point a nested entity kind is resolved at.

    Every nested field's kind has one by the time an entity exists --
    `__init_subclass__` refuses the class otherwise -- so this is the
    narrowing, not a second check.
    """
    point = kind.extension_point
    if point is None:
        msg = f"{kind.__name__} is registered at no single extension point"
        raise TypeError(msg)
    return point


def _element_annotations(inner: object, count: int) -> list[object]:
    """The annotation of each element of a tuple type, one per element held."""
    arguments = get_args(inner)
    if len(arguments) == 2 and arguments[1] is Ellipsis:
        return [arguments[0]] * count
    return list(arguments)


def _fitting_branch(inner: object, value: object) -> object | None:
    """The branch of a union that holds an entity and whose shape `value` has."""
    for branch in get_args(inner):
        if branch is UNSET or not _contains_entity(branch):
            continue
        if _has_shape(_shape(branch), value):
            return branch
    return None


def _resolve(
    annotation: object, value: object, context: Context, loc: Loc
) -> tuple[object, tuple[ValidationProblem, ...]]:
    """`value`, with every nested metadata field in it read as an entity in `context`.

    What `prepare` used to be written for by hand: a field annotated with
    an entity type is resolved through the scope, at the point that kind
    of entity is registered at; an array of them element by element; a
    record holding one field by field. The value has passed its type
    check, so the shapes are the annotation's.
    """
    inner, _ = _strip(annotation)
    candidates = list(get_args(inner)) if _is_union(inner) else [inner]
    if _is_entity_or_opaque(candidates):
        return context.coerce(_point_of(_entity_kinds(inner)[0]), value, loc)
    if _is_union(inner):
        branch = _fitting_branch(inner, value)
        return (value, ()) if branch is None else _resolve(branch, value, context, loc)
    if get_origin(inner) is tuple:
        entries = cast("tuple[object, ...]", value)
        resolved: list[object] = []
        found: list[ValidationProblem] = []
        for position, (element, entry) in enumerate(
            zip(_element_annotations(inner, len(entries)), entries, strict=True)
        ):
            item, problems = _resolve(element, entry, context, (*loc, position))
            resolved.append(item)
            found.extend(problems)
        return tuple(resolved), tuple(found)
    if isinstance(inner, type) and is_dataclass(inner) and not _is_entity_type(inner):
        entries = cast("Mapping[str, object]", value)
        members: dict[str, object] = {}
        found = []
        for name, field_annotation in _field_hints(inner).items():
            if name not in entries:
                continue
            member, problems = _resolve(field_annotation, entries[name], context, (*loc, name))
            members[name] = member
            found.extend(problems)
        return inner(**members), tuple(found)
    return value, ()


def _render(annotation: object, value: object) -> object:
    """`value` as a document would write it: every nested entity in its JSON form."""
    if isinstance(value, MetadataEntity):
        return value.to_json()
    if isinstance(value, Opaque):
        return value.json
    inner, _ = _strip(annotation)
    if _is_union(inner):
        branch = _fitting_branch(inner, value)
        return value if branch is None else _render(branch, value)
    if get_origin(inner) is tuple:
        entries = cast("tuple[object, ...]", value)
        return tuple(
            _render(element, entry)
            for element, entry in zip(
                _element_annotations(inner, len(entries)), entries, strict=True
            )
        )
    if isinstance(inner, type) and is_dataclass(inner) and not _is_entity_type(inner):
        return {
            name: _render(field_annotation, getattr(value, name))
            for name, field_annotation in _field_hints(inner).items()
            if getattr(value, name) is not UNSET
        }
    return value


def _canonicalize(annotation: object, value: object) -> object:
    """`value` with every nested entity in its own canonical form."""
    if isinstance(value, MetadataEntity):
        return value.canonical()
    if isinstance(value, Opaque):
        return value
    inner, _ = _strip(annotation)
    if _is_union(inner):
        branch = _fitting_branch(inner, value)
        return value if branch is None else _canonicalize(branch, value)
    if get_origin(inner) is tuple:
        entries = cast("tuple[object, ...]", value)
        return tuple(
            _canonicalize(element, entry)
            for element, entry in zip(
                _element_annotations(inner, len(entries)), entries, strict=True
            )
        )
    if (
        isinstance(inner, type)
        and is_dataclass(inner)
        and not _is_entity_type(inner)
        and is_dataclass(value)
        and not isinstance(value, type)
    ):
        return replace(
            value,
            **{
                name: _canonicalize(field_annotation, getattr(value, name))
                for name, field_annotation in _field_hints(inner).items()
            },
        )
    return value


ValueRoutine: TypeAlias = "Callable[..., tuple[ValidationProblem, ...]]"
"""An entity's value-space judgment, over the members it was given."""


_MISSING_DEFAULT: Final = object()
"""Distinguishes "declared no default" from a default that is None or UNSET."""


def _no_value_problems(**members: object) -> tuple[ValidationProblem, ...]:
    """An entity whose types admit only valid values has nothing to add."""
    return ()


@dataclass(frozen=True, slots=True)
class Opaque:
    """A metadata field this reading did not turn into an entity.

    Carrying the JSON rather than dropping it is what makes the result a
    real union: `CodecEntity | Opaque` is exhaustive and narrows, where
    `CodecEntity | object` is just `object` and narrows to nothing.

    `reason` is the distinction a reader needs and could not otherwise
    make. `out_of_scope` is a name no entity in this `Context` claims --
    an extension this reader does not model, which is not an error and is
    the reader's cue to resolve it elsewhere. `invalid` is a name that
    *was* claimed and then refused; the reasons are in the problems
    reported alongside.
    """

    json: object
    reason: Literal["out_of_scope", "invalid"]


def _is_class_var(annotation: object) -> bool:
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


def _declared_class_vars(cls: type) -> dict[str, type]:
    """Every class variable annotated anywhere in `cls`'s ancestry.

    Mapped to the class that annotated it, so a message can say where the
    requirement comes from. Base first, so a redeclaration names the
    nearest ancestor.
    """
    found: dict[str, type] = {}
    for ancestor in reversed(cls.__mro__):
        for name, annotation in _own_annotations(ancestor).items():
            if _is_class_var(annotation):
                found[name] = ancestor
    return found


# No `slots=True`, deliberately. It rebuilds the class, which on Python
# 3.11 and 3.12 leaves the zero-argument `super()` *in that same class's
# body* pointing at the class it replaced. Several entities call `super()`
# to narrow `to_json` and to adjust `configuration`, so they would each
# have to spell it `super(Cls, self)`. CPython fixed this in 3.13, so when
# that is the floor this is worth revisiting; the memory saved is small at
# document scale, which is why it has not been.
@dataclass(frozen=True)
class MetadataEntity:
    """One named entity, coerced from its metadata.

    Subclasses add their configuration members as fields, which is what
    makes them well-typed by construction: an instance exists only if
    `coerce` accepted the metadata that produced it. An optional member is
    typed `| None` with a default of `None`, so absence is representable
    and a canonical spelling can leave it out.

    Frozen, so an entity of hashable members is hashable. One holding a
    value out of scope is not, because that value is the JSON the document
    wrote and a JSON object is a `dict` -- the same way any frozen
    dataclass holding a list is unhashable. It cannot be an immutable
    mapping instead: `MappingProxyType` is unhashable too, and anything
    else stops `json.dumps` from serializing what `to_json` returns.

    Most subclasses declare `member_types` and nothing else: the default
    `coerce` and `to_json` are written once here against that table. The
    ones that override are the ones with something particular to say --
    a configuration containing other entities, a name that is a family
    rather than a constant, a member another member renders meaningless.
    """

    extension_point: ClassVar[ExtensionPointField | None] = None
    """Where this kind of entity is registered, if it is registered at one point.

    Set by `CodecEntity`, `DataTypeEntity` and `ChunkGridEntity`. It is
    what makes a field typed as one of those resolvable: the scope is
    asked at that point. `MetadataEntity` itself is the kind of the two
    points that take any entity, so it names none, and a field typed as
    bare `MetadataEntity` is refused at class creation.
    """

    must_understand: ClassVar[bool] = True
    """Whether a reader that does not know this entity may skip it.

    A property of the *kind* of metadata, not of a use of it: a codec is
    something you must understand, every time it appears, because
    ignoring one gives wrong bytes. Consolidated metadata is the opposite
    and is unconditionally skippable. Neither is a per-occurrence choice,
    so neither is a configuration member -- which is why this is a class
    variable and not a field.

    The spec permits `must_understand: false` on a codec; this package
    treats that as an oversight and refuses it. Where the flag does earn
    its keep -- an unknown top-level extension field a reader really can
    skip -- it stays per-occurrence, on `ZarrV3NamedConfig`.
    """

    identifier: ClassVar[str]
    """The name this entity is registered under.

    Usually the `name` the metadata carries. The raw-bytes data types are
    the exception: every `r<N>` spelling is one family, so the family gets
    an invented identifier that no real name can collide with.
    """

    member_types: ClassVar[MemberTypes] = MappingProxyType({})
    """The configuration members, and the type each one takes.

    Read off the dataclass fields at class creation: which members there
    are, which may be absent (the type admits `UNSET`), and the check
    each one's type implies. A class declares an entry itself only for a
    field whose annotation `check_for` cannot compile, and the public
    JSON TypedDict is held to the same keys by `tests/v3/test_entities.py`.
    """

    nested_members: ClassVar[Mapping[str, object]] = MappingProxyType({})
    """The fields that hold other entities, with their annotations.

    Read off the fields at class creation, like `member_types`. These are
    the members `coerce` resolves through the scope, `configuration`
    renders as JSON and `canonical` recurses into -- so an entity that
    contains entities writes nothing for any of that.
    """

    value_checks: ClassVar[Mapping[str, TypeCheck]] = MappingProxyType({})
    """The value rules the field annotations state, member by member.

    A bound in an `Annotated` -- `level: Annotated[int, Interval(ge=0,
    le=9)]`, `chunk_shape: tuple[Annotated[int, Ge(1)], ...]` -- becomes
    a check here at class creation, located at the member and, for an
    element, at its position. Runs with `value_problems`, after the type
    checks, on both the reading path and the constructor.
    """

    member_rules: ClassVar[Mapping[str, tuple[MemberRule, ...]]] = MappingProxyType({})
    """The `@validates` rules, by the member each is about.

    Collected at class creation from the class and its ancestors, the
    nearest definition of a name winning. Run after the annotation bounds
    and before `value_problems`, on both paths.
    """

    configuration_required: ClassVar[bool] = False
    """Whether the bare-name spelling says too little for this entity.

    The spec permits a bare name "if no configuration metadata is
    required", so this is true exactly when some member is required --
    which the fields already say.
    """

    def __init_subclass__(cls, *, base: bool = False, **kwargs: object) -> None:
        """Refuse a subclass that is not an entity this layer can use.

        Every check here has the same shape: something that type-checks
        cleanly and then goes wrong later, somewhere that will not name
        this class. An import-time error in the extension's own module is
        the one place the author is looking.

        `base=True` for a class that exists to add a class variable
        rather than to be an entity -- `CodecEntity`, `IntegerDataType`.
        """
        super().__init_subclass__(**kwargs)
        if base:
            return
        if "problems" in cls.__dict__:
            # Value rules are `value_problems`, a static routine over the
            # members. An override named `problems` is a rule that would
            # never run, and nothing else would say so.
            msg = (
                f"{cls.__name__} defines `problems`; a value rule is a bound on the "
                "field, a `@validates` rule about one member, or `value_problems` "
                "over the members together -- none of them takes an entity"
            )
            raise TypeError(msg)
        if "prepare" in cls.__dict__:
            # A member that is an entity is read from its annotation, so
            # an override named `prepare` is resolution that would never
            # run, and nothing else would say so.
            msg = (
                f"{cls.__name__} defines `prepare`; a member that is an entity is read "
                "from its field annotation, and nothing calls `prepare`"
            )
            raise TypeError(msg)
        if "__post_init__" in cls.__dict__:
            # `coerce` builds through `unchecked`, which bypasses
            # `__init__` and so never reaches `__post_init__`. Rules put
            # there would hold for a hand-built entity and be silently
            # absent for every entity read from a document -- the one
            # direction that matters.
            msg = (
                f"{cls.__name__} defines `__post_init__`, which `unchecked` does "
                "not reach; value rules belong in `value_problems`"
            )
            raise TypeError(msg)
        if "configuration_required" in vars(cls):
            msg = (
                f"{cls.__name__} declares `configuration_required`, which follows "
                "from whether any member is required"
            )
            raise TypeError(msg)
        # Before every guard below, because they read the table.
        declared = dict(vars(cls).get("member_types", {}))
        derived, unread = derive_member_types(cls)
        unsupported = sorted(set(unread) - set(declared))
        if len(unsupported) != 0:
            msg = (
                f"{cls.__name__}: no check can be read off the annotation of "
                f"{', '.join(unsupported)}; declare one in `member_types`"
            )
            raise TypeError(msg)
        optional = {name: is_optional(annotation) for name, annotation in _field_hints(cls).items()}
        misstated = sorted(
            member
            for member, (required, _) in declared.items()
            if member in optional and required == optional[member]
        )
        if len(misstated) != 0:
            # The check is the entity's to write; whether the member may
            # be left out is the field's to say, and a declared entry
            # that disagrees is the drift this derivation exists to rule
            # out.
            msg = (
                f"{cls.__name__} declares {', '.join(misstated)} with a requiredness "
                "its field does not give it"
            )
            raise TypeError(msg)
        cls.member_types = {**derived, **declared}
        cls.configuration_required = any(required for required, _ in cls.member_types.values())
        hints = _field_hints(cls)
        cls.nested_members = {
            name: annotation for name, annotation in hints.items() if _contains_entity(annotation)
        }
        cls.value_checks = {
            name: check
            for name, annotation in hints.items()
            if (check := value_check_for(annotation)) is not None
        }
        attributes: dict[str, object] = {}
        for ancestor in reversed(cls.__mro__):
            attributes.update(vars(ancestor))
        rules: dict[str, list[MemberRule]] = {}
        for attribute in attributes.values():
            function = attribute.__func__ if isinstance(attribute, staticmethod) else attribute
            # Only a function can carry the mark; a table-valued class
            # attribute is not even hashable.
            if not callable(function):
                continue
            for member in _RULE_MEMBERS.get(function, ()):
                if member not in hints:
                    msg = f"{cls.__name__}: `@validates({member!r})` names no field of the entity"
                    raise TypeError(msg)
                rules.setdefault(member, []).append(cast("MemberRule", function))
        cls.member_rules = {member: tuple(found) for member, found in rules.items()}
        unplaced = sorted(
            name
            for name, annotation in cls.nested_members.items()
            if any(kind.extension_point is None for kind in _entity_kinds(annotation))
        )
        if len(unplaced) != 0:
            # `MetadataEntity` itself is registered at no single point, so
            # a field typed as one could not be resolved through a scope.
            msg = (
                f"{cls.__name__}: the entity kind of {', '.join(unplaced)} has no "
                "`extension_point`; annotate it with `CodecEntity`, `DataTypeEntity` "
                "or `ChunkGridEntity`"
            )
            raise TypeError(msg)
        annotated = _declared_class_vars(cls)
        shadowed = [
            name
            for name, annotation in _own_annotations(cls).items()
            if name in annotated and annotated[name] is not cls and not _is_class_var(annotation)
        ]
        if len(shadowed) != 0:
            # A field of that name would go into `member_types`, into the
            # configuration, and into the JSON -- while the class variable
            # it shadows is what every other part of this layer reads.
            msg = (
                f"{cls.__name__} declares {', '.join(shadowed)} as a field, "
                "shadowing a class variable of the same name"
            )
            raise TypeError(msg)
        # A class variable annotated with no value anywhere in the
        # ancestry is one the concrete entity owes: `identifier` for all
        # of them, `kind` for a codec, `bounds` for an integer type.
        # Derived rather than listed, so adding one to a family cannot
        # forget to require it.
        missing = [name for name in annotated if not hasattr(cls, name)]
        if len(missing) != 0:
            msg = f"{cls.__name__} does not declare {', '.join(sorted(missing))}"
            raise TypeError(msg)
        # A member's default decides whether the entity can exist without
        # it, so the two kinds have opposite rules. `@dataclass` has not
        # run yet, so a member declared with `field(...)` is still a
        # `Field` here and its default has to be unwrapped.
        defaulted: dict[str, object] = {}
        for key in cls.member_types:
            declared: object = getattr(cls, key, _MISSING_DEFAULT)
            if type(declared) is Field:
                # `field(...)`, so the default is inside it rather than
                # being the attribute. `@dataclass` has not unwrapped it
                # yet -- this hook runs first.
                spec = cast("Field[object]", declared)
                declared = (
                    _MISSING_DEFAULT
                    if spec.default is MISSING and spec.default_factory is MISSING
                    else spec.default
                )
            defaulted[key] = declared
        # An optional member defaults to UNSET or `configuration` emits it
        # for every instance, so the bare-name spelling becomes
        # unreachable and a document gains a member it never wrote.
        invented = [
            key
            for key, (required, _) in cls.member_types.items()
            if not required and defaulted[key] is not UNSET
        ]
        if len(invented) != 0:
            msg = (
                f"{cls.__name__} gives the optional member(s) "
                f"{', '.join(invented)} a default other than UNSET"
            )
            raise TypeError(msg)
        # A required member with a default is an entity that can be built
        # without it -- and then serializes a document nobody wrote. A
        # conventional starting point is a `create_default` classmethod,
        # named so that asking for one is deliberate.
        presumed = [
            key
            for key, (required, _) in cls.member_types.items()
            if required and defaulted[key] is not _MISSING_DEFAULT
        ]
        if len(presumed) != 0:
            msg = (
                f"{cls.__name__} gives the required member(s) "
                f"{', '.join(presumed)} a default; required members have none"
            )
            raise TypeError(msg)

    @classmethod
    def accepts(cls, name: str) -> bool:
        """Whether `name` denotes this entity.

        Constant for all but the raw-bytes family, where one class covers
        every `r<N>`.
        """
        return name == cls.identifier

    @classmethod
    def coerce(cls, value: object, context: Context) -> Coerced[Self]:
        """`value` as this entity, or the reasons it is not one.

        `context` is the scope this reading is happening in; most entities
        have no use for it and ignore it.
        """
        name, configuration, _ = named_configuration(value)
        if name is None or not cls.accepts(name):
            return None, problem((), f"expected the {cls.identifier!r} entity")
        if configuration is None:
            if cls.configuration_required:
                return None, problem(
                    ("configuration",),
                    f"{cls.identifier!r} requires a configuration",
                    "missing_key",
                )
            configuration = cast("Mapping[str, object]", {})
        members, found, unreadable = coerce_members(configuration, cls.member_types)
        if len(unreadable) == 0:
            # Before judging: a member that is itself an entity has to be
            # one before its container's value rules can ask it anything.
            for name, annotation in cls.nested_members.items():
                if name in members:
                    members[name], nested = _resolve(
                        annotation, members[name], context, ("configuration", name)
                    )
                    found = (*found, *nested)
        if len(unreadable) != 0:
            # A member that could not be read leaves a hole, and the value
            # rules are written over a whole configuration -- blosc's
            # `typesize` requirement reads `shuffle`. Judging around the
            # hole would be guessing, so the type problems stand alone.
            return None, found
        found = (*found, *within((), cls._judge_values(members)))
        if any(entry.kind != "unknown_key" for entry in found):
            return None, found
        # Already asked, so do not ask again on the way in.
        return cls.unchecked(**members), found

    def canonical(self) -> Self:
        """This entity in the simplest form that means the same thing.

        A *transformation*, asked for by `canonicalize_array_metadata_v3`
        and by nothing else. `to_json` does not apply it, because writing
        a document back is not the same as asking for it to be rewritten:
        a reader that reads and writes should not change bytes it was not
        asked to change.

        A contained entity is put in its own canonical form here, by
        walking the fields that hold one. Override where two spellings of
        the entity's *own* members mean the same -- a rectilinear
        dimension's run-length encoding, a `typesize` that `noshuffle`
        ignores -- and start from `super().canonical()`, so the walk is
        not lost.
        """
        nested = type(self).nested_members
        if len(nested) == 0:
            return self
        return replace(
            self,
            **{
                name: _canonicalize(annotation, getattr(self, name))
                for name, annotation in nested.items()
            },
        )

    def configuration(self) -> dict[str, object]:
        """This entity's configuration, as the document would write it.

        Faithful to every member the entity holds: `to_json` is
        serialization, not canonicalization, so nothing is simplified
        here. A contained entity is rendered as its own `to_json`, by
        walking the fields that hold one.

        Absent optional members are left out, which is what makes the
        bare-name spelling reachable. Absence is `UNSET`, never `None`:
        this package holds `None` to mean a JSON `null` the document
        actually wrote, and `scale_offset` is a real case where `null`
        and absent are different documents.

        Deep-copied, because a member can be an arbitrary JSON value: a
        `scale_offset` offset may be an object, and handing the caller
        the entity's own dict would let them mutate a frozen entity
        through the document it returned.
        """
        members = self._configuration_members()
        for name, annotation in type(self).nested_members.items():
            if name in members:
                members[name] = _render(annotation, members[name])
        return deepcopy(members)

    value_problems: ClassVar[ValueRoutine] = staticmethod(_no_value_problems)
    """What the spec disallows among the members taken together.

    The third of three places a value rule lives, for the rule that
    reads two members at once -- blosc's `typesize` against its
    `shuffle`. A bound on one member is on the field; a rule about one
    member is a `@validates` staticmethod. A routine rather than a
    method, because judging values does not need an entity -- and
    needing one would mean an invalid one had been built. Takes
    `Unpack[<Entity>Configuration]`: the same spelling the constructor
    takes, receiving only the members that are present.

    Typed loosely here because the base does not know any entity's
    configuration, and saying so is the truth. Call a specific routine by
    its own name to have the arguments checked.

    Locations are relative to the entity's `configuration`.
    """

    def __post_init__(self) -> None:
        """Refuse to exist with values the spec disallows.

        So an instance is the value guarantee, not just the type one:
        `BloscCodec(clevel=99)` raises rather than serializing a document
        no reader will accept. `coerce` asks `value_problems` first and
        reports, so reading a bad document still returns problems rather
        than raising, and `unchecked` is the door for a caller that has
        already asked.
        """
        found = type(self)._judge_values(self._members())
        if len(found) != 0:
            raise MetadataValidationError(found)

    @classmethod
    def _judge_values(cls, members: Mapping[str, object]) -> tuple[ValidationProblem, ...]:
        """Every value problem among the members.

        The bounds the annotations state first, then the `@validates`
        rules, member by member, then whatever `value_problems` has to
        say about the members together -- one routine for the reading
        path and the constructor, so the two cannot disagree.
        """
        from_annotations = [
            found
            for name, check in cls.value_checks.items()
            if name in members
            for found in check(members[name], (name,))
        ]
        from_rules = [
            ValidationProblem((member, *found.loc), found.message, found.kind)
            for member, member_rules in cls.member_rules.items()
            if member in members
            for rule in member_rules
            for found in rule(members[member])
        ]
        return (*from_annotations, *from_rules, *cls.value_problems(**members))

    def _members(self) -> dict[str, object]:
        """Every member this entity holds, unrendered.

        The dataclass's own fields, which is what `value_problems`
        judges: a member is a member whether or not the JSON spells it
        as a configuration key. The raw-bytes family is the case that
        separates the two -- its width lives in its name, so it has a
        field and no configuration at all.
        """
        return {
            field_.name: value
            for field_ in fields(self)
            if (value := getattr(self, field_.name)) is not UNSET
        }

    def _configuration_members(self) -> dict[str, object]:
        """The members a configuration object would spell out.

        `_members` minus anything the envelope carries some other way.
        """
        return {
            key: value
            for key in type(self).member_types
            if (value := getattr(self, key)) is not UNSET
        }

    @classmethod
    def unchecked(cls, **members: object) -> Self:
        """This entity, without asking whether its values are allowed.

        For a caller that has already asked -- `coerce` does, so that it
        can report the answer instead of raising it. Named so that
        choosing it is deliberate.

        Unchecked means *value*-unchecked. A member this entity does not
        declare, or one with neither a value nor a default, is still a
        `TypeError`: those produce an entity that cannot be repred,
        compared or hashed, which no caller is asking for.
        """
        declared = {field_.name: field_ for field_ in fields(cls)}
        unknown = sorted(members.keys() - declared.keys())
        if len(unknown) != 0:
            msg = f"{cls.__name__} has no member(s) {', '.join(unknown)}"
            raise TypeError(msg)
        entity = object.__new__(cls)
        for name, field_ in declared.items():
            if name in members:
                object.__setattr__(entity, name, members[name])
            elif field_.default is not MISSING:
                object.__setattr__(entity, name, field_.default)
            elif field_.default_factory is not MISSING:  # pragma: no cover - none today
                object.__setattr__(entity, name, field_.default_factory())
            else:
                # Leaving it unset would give an entity whose `repr`,
                # `==` and `hash` raise `AttributeError` on access.
                msg = f"{cls.__name__} is missing a value for {name!r}"
                raise TypeError(msg)
        return entity

    def to_json(self) -> ZarrV3MetadataFieldJSON:
        """This entity as a document would write it.

        Faithful to every member it models: read a document, write it
        back, and those come out as they went in. Ask `canonical` first
        if you want the simplest equivalent spelling.

        A member this entity does not model is not one of them. It is
        reported as `unknown_key` and not held, so writing back drops it
        -- which only a caller who took the problems as data and went on
        past that one can reach, because `from_json` raises on it. A
        caller who needs the bytes preserved has the JSON it passed in,
        and `Opaque` is where unmodelled metadata belongs.

        What is *not* preserved is the envelope's spelling, because the
        entity does not model it: a bare name, `{"name": x}`, and
        `{"name": x, "configuration": {}}` all mean the same and all read
        to the same entity, so all three write back as the bare name.
        `must_understand` follows the entity's own class variable, so it
        is omitted for everything this package models today.

        Subclasses narrow the return type to their own object TypedDict,
        which is the JSON form this dataclass models.
        """
        configuration = self.configuration()
        if len(configuration) == 0 and type(self).must_understand:
            return cast("ZarrV3MetadataFieldJSON", type(self).identifier)
        entry: dict[str, object] = {"name": type(self).identifier}
        if len(configuration) != 0:
            entry["configuration"] = configuration
        if not type(self).must_understand:
            entry["must_understand"] = False
        return cast("ZarrV3MetadataFieldJSON", entry)


@dataclass(frozen=True)
class CodecEntity(MetadataEntity, base=True):
    """An entity that occupies a position in the codec pipeline."""

    extension_point: ClassVar[ExtensionPointField] = CODECS
    """Where a codec is registered, and so where a field typed as one is resolved."""

    kind: ClassVar[CodecKind]

    variable_size: ClassVar[bool] = False
    """Whether this codec's output size depends on the bytes it is given.

    A compressor's does, so a shard index encoded with one has no size
    derivable from metadata alone, and the shard cannot be read.
    """

    def incoming_problems(self, incoming: ArrayParts | None) -> tuple[ValidationProblem, ...]:
        """Why this codec cannot be applied to the array that reaches it.

        `incoming` is None once the chain can no longer say what reaches
        here, and the default answer to that is nothing: declining beats
        guessing. Locations are relative to this codec's `configuration`,
        as `problems`' are; an empty one lands on the codec itself.
        """
        return ()

    def transition(self, incoming: ArrayParts) -> ArrayParts | None:
        """What the next codec in the chain sees, or None if undeterminable.

        Only an array-to-array codec has anything to say: the two later
        kinds end shape propagation by construction, one by consuming the
        array and the other by never having had it.

        The default is None, so a modelled codec that forgets to say how
        it transforms the array stops propagation rather than silently
        claiming to leave it alone. Failing closed here costs a judgment;
        failing open would invent one.
        """
        return None


@dataclass(frozen=True)
class ChunkGridEntity(MetadataEntity, base=True):
    """An entity that divides an array into the parts a pipeline encodes."""

    extension_point: ClassVar[ExtensionPointField] = CHUNK_GRID

    def shape_problems(self, array_shape: object) -> tuple[ValidationProblem, ...]:
        """Why this grid does not divide an array of `array_shape`.

        Locations are relative to the grid's `configuration`. Default:
        nothing, for a grid this package reads but has no such rule for.
        """
        return ()

    def grid(self, array_shape: object) -> ChunkGrid:
        """What this grid divides an array of `array_shape` into.

        The array shape is a parameter because neither determines a grid
        alone: a grid whose own metadata cannot be read still has the
        array's rank, and rank is enough for several rules.
        """
        return ChunkGrid.unreadable(array_shape)


@dataclass(frozen=True)
class DataTypeEntity(MetadataEntity, base=True):
    """An entity that says how the array's scalars are stored.

    Only data types answer that, and every rule that turns on it -- a
    `bytes` codec is pointless before a single-byte type, a struct field
    cannot be variable-length -- asks a data type rather than consulting
    a table of names.
    """

    extension_point: ClassVar[ExtensionPointField] = DATA_TYPE

    scalar_storage: ClassVar[StorageClass]

    def storage_class(self) -> StorageClass | None:
        """How one scalar occupies bytes, or None if undetermined.

        None only for a composite whose parts are not all in scope: an
        answer would be a guess, and the rules that ask decline instead.
        """
        return type(self).scalar_storage

    def fill_value_problems(self, value: object, loc: Loc = ()) -> tuple[ValidationProblem, ...]:
        """Why `value` is not a fill value of this type, if it is not.

        Default: nothing. A data type this package does not model accepts
        whatever its extension says it does, and guessing would reject
        valid documents.
        """
        return ()


def within(prefix: Loc, problems: Sequence[ValidationProblem]) -> tuple[ValidationProblem, ...]:
    """One entity's problems, located in the document that holds it.

    An entity reports relative to its own `configuration`, so that is what
    goes between the field and the member. A problem with an empty
    location is about the entity itself -- a malformed `r<N>` name, a
    codec that cannot encode what reaches it -- and lands on the field.
    """
    return tuple(
        ValidationProblem(
            (*prefix, *(("configuration", *found.loc) if len(found.loc) != 0 else ())),
            found.message,
            found.kind,
        )
        for found in problems
    )


def named_configuration(
    value: object,
) -> tuple[str | None, Mapping[str, object] | None, bool]:
    """Split metadata into `(name, configuration, must_understand)`.

    The shared shape every entity arrives in: a bare name, or an object
    carrying one. A `None` name means the value is not a metadata field at
    all; a `None` configuration means the bare spelling was used.
    """
    if isinstance(value, str):
        return value, None, True
    if not isinstance(value, Mapping):
        return None, None, True
    entry = cast("Mapping[str, object]", value)
    name = entry.get("name")
    if not isinstance(name, str):
        return None, None, True
    configuration = entry.get("configuration")
    must_understand = entry.get("must_understand", True)
    return (
        name,
        cast("Mapping[str, object]", configuration) if isinstance(configuration, Mapping) else None,
        must_understand if isinstance(must_understand, bool) else True,
    )


__all__ = [
    "CHUNK_GRID",
    "CHUNK_KEY_ENCODING",
    "CODECS",
    "DATA_TYPE",
    "FROM_NAME",
    "STORAGE_TRANSFORMERS",
    "CheckCompiler",
    "ChunkGridEntity",
    "CodecEntity",
    "CodecKind",
    "Coerced",
    "DataTypeEntity",
    "ExtensionPointField",
    "Ge",
    "Gt",
    "Interval",
    "Le",
    "Loc",
    "Lt",
    "MemberRule",
    "MemberTypes",
    "MetadataEntity",
    "Opaque",
    "StorageClass",
    "TypeCheck",
    "ValueRoutine",
    "coerce_members",
    "is_bool",
    "is_int",
    "is_integer",
    "is_json_value",
    "is_metadata_field",
    "is_str",
    "named_configuration",
    "one_of",
    "problem",
    "register_check",
    "sequence_of",
    "validates",
    "within",
]
