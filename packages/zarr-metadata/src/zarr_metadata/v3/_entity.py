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

The shared plumbing lives here too: the member checks every entity needs
and the walk over a configuration that applies them. What stays with the
entity is the table saying which members it has -- that is the part that
is about blosc rather than about entities.
"""

from __future__ import annotations

from collections.abc import Mapping as _Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Final, Literal, TypeAlias, TypeVar, cast

from typing_extensions import TypeIs

from zarr_metadata.model._validation import ValidationProblem, is_json
from zarr_metadata.v3._parts import ChunkGrid

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
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
"""The entity, or None and every reason the metadata is not one.

A caller that only wants a verdict reads the problems; one that wants to
go on reading the entity checks for None. Both never happen at once.
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

DATA_TYPE: Final[ExtensionPointField] = "data_type"
CHUNK_GRID: Final[ExtensionPointField] = "chunk_grid"
CHUNK_KEY_ENCODING: Final[ExtensionPointField] = "chunk_key_encoding"
CODECS: Final[ExtensionPointField] = "codecs"

StorageClass = Literal["single_byte", "multi_byte", "variable_length"]
"""How one scalar of a data type occupies bytes.

`single_byte` and `multi_byte` are both fixed-size; they differ only in
whether a byte order applies, which is what the `bytes` codec's `endian`
member is about.
"""

CodecKind = Literal["array_array", "array_bytes", "bytes_bytes"]
"""The three pipeline positions the v3 spec sorts codecs into.

Here rather than in `zarr_metadata.v3.codec.kind` because each codec
declares its own kind, and that module imports every codec to build the
tuples it will no longer need once they all do.
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
    if isinstance(value, _Mapping):
        entries = cast("Mapping[str, object]", value)
        return {key: _as_tuples(entry) for key, entry in entries.items()}
    return value


def coerce_members(
    configuration: Mapping[str, object], types: MemberTypes
) -> tuple[dict[str, object], tuple[ValidationProblem, ...], bool]:
    """The members `types` declares, taken from `configuration`.

    Returns what was accepted, every problem found, and whether the entity
    is still worth building. Three kinds of problem, and they differ in
    that last part:

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
    usable = True
    for key in configuration:
        if key not in types:
            problems.extend(problem(("configuration",), f"unexpected key {key!r}", "unknown_key"))
    for key, (required, check) in types.items():
        if key not in configuration:
            if required:
                problems.extend(
                    problem(("configuration", key), f"missing required key {key!r}", "missing_key")
                )
                usable = False
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
            usable = False
    return members, tuple(problems), usable


# No `slots=True`, deliberately: it rebuilds the class, which leaves the
# zero-argument `super()` in a subclass pointing at the class that was
# replaced. Subclasses call `super()` to narrow `to_json` and to adjust
# `configuration`, so slots would be a trap laid for every entity.
@dataclass(frozen=True)
class MetadataEntity:
    """One named entity, coerced from its metadata.

    Subclasses add their configuration members as fields, which is what
    makes them well-typed by construction: an instance exists only if
    `coerce` accepted the metadata that produced it. An optional member is
    typed `| None` with a default of `None`, so absence is representable
    and a canonical spelling can leave it out.

    Most subclasses declare `member_types` and nothing else: the default
    `coerce` and `to_json` are written once here against that table. The
    ones that override are the ones with something particular to say --
    a configuration containing other entities, a name that is a family
    rather than a constant, a member another member renders meaningless.
    """

    # Keyword-only: it is the envelope's member, not the configuration's,
    # and it would otherwise take the first positional slot of every
    # entity -- so `RawBytesDataType("r16")` would set this instead of
    # the field it reads as.
    must_understand: bool = field(default=True, kw_only=True)

    identifier: ClassVar[str]
    """The name this entity is registered under.

    Usually the `name` the metadata carries. The raw-bytes data types are
    the exception: every `r<N>` spelling is one family, so the family gets
    an invented identifier that no real name can collide with.
    """

    member_types: ClassVar[MemberTypes] = {}
    """The configuration members, and the type each one takes.

    The same keys as the configuration TypedDict, which is the same as the
    constructor signature; `tests/v3/test_entities.py` holds the three
    together.
    """

    configuration_required: ClassVar[bool] = False
    """Whether the bare-name spelling says too little for this entity.

    The spec permits a bare name "if no configuration metadata is
    required", so this is true exactly when some member is required.
    """

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
        name, configuration, must_understand = named_configuration(value)
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
        members, found, usable = coerce_members(configuration, cls.member_types)
        if not usable:
            return None, found
        return cls(must_understand=must_understand, **members), found  # type: ignore[arg-type]

    def configuration(self) -> dict[str, object]:
        """This entity's configuration, in its simplest equivalent form.

        Absent optional members are left out, which is what makes the
        bare-name spelling reachable. Override to drop a member that
        another member renders meaningless.
        """
        return {
            key: value
            for key in type(self).member_types
            if (value := getattr(self, key)) is not None
        }

    def problems(self) -> tuple[ValidationProblem, ...]:
        """Every value of this entity the spec disallows.

        Locations are relative to the entity's `configuration`. Default:
        an entity whose type admits only valid values has nothing to add.
        """
        return ()

    def to_json(self) -> ZarrV3MetadataFieldJSON:
        """This entity in its simplest equivalent spelling.

        A name alone when the name says everything, and the object form
        otherwise. `must_understand` is omitted when true, because that is
        the default and says nothing; an explicit false says something.

        Subclasses narrow the return type to their own object TypedDict,
        which is the JSON form this dataclass models.
        """
        configuration = self.configuration()
        if len(configuration) == 0 and self.must_understand:
            return cast("ZarrV3MetadataFieldJSON", type(self).identifier)
        entry: dict[str, object] = {"name": type(self).identifier}
        if len(configuration) != 0:
            entry["configuration"] = configuration
        if not self.must_understand:
            entry["must_understand"] = False
        return cast("ZarrV3MetadataFieldJSON", entry)


@dataclass(frozen=True)
class CodecEntity(MetadataEntity):
    """An entity that occupies a position in the codec pipeline."""

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
        guessing. Locations are relative to this codec's entry.
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
class ChunkGridEntity(MetadataEntity):
    """An entity that divides an array into the parts a pipeline encodes."""

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
class DataTypeEntity(MetadataEntity):
    """An entity that says how the array's scalars are stored.

    Only data types answer that, and every rule that turns on it -- a
    `bytes` codec is pointless before a single-byte type, a struct field
    cannot be variable-length -- asks a data type rather than consulting
    a table of names.
    """

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
    if not isinstance(value, _Mapping):
        return None, None, True
    entry = cast("Mapping[str, object]", value)
    name = entry.get("name")
    if not isinstance(name, str):
        return None, None, True
    configuration = entry.get("configuration")
    must_understand = entry.get("must_understand", True)
    return (
        name,
        cast("Mapping[str, object]", configuration)
        if isinstance(configuration, _Mapping)
        else None,
        must_understand if isinstance(must_understand, bool) else True,
    )


__all__ = [
    "CHUNK_GRID",
    "CHUNK_KEY_ENCODING",
    "CODECS",
    "DATA_TYPE",
    "ChunkGridEntity",
    "CodecEntity",
    "CodecKind",
    "Coerced",
    "DataTypeEntity",
    "ExtensionPointField",
    "Loc",
    "MemberTypes",
    "MetadataEntity",
    "StorageClass",
    "TypeCheck",
    "coerce_members",
    "is_bool",
    "is_int",
    "is_integer",
    "is_json_value",
    "is_str",
    "named_configuration",
    "one_of",
    "problem",
    "sequence_of",
    "within",
]
