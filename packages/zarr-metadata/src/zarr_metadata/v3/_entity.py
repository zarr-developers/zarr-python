"""What every metadata entity can do for itself.

A codec, data type, chunk grid, chunk key encoding or storage transformer
is one frozen dataclass whose fields are its configuration, and the
field annotations are its schema: `coerce` reads a document's
configuration against them, member by member, with `_typed_json`. A
field typed `CodecEntity | Opaque` holds another entity, read through
the scope the containing one is read in. Everything finer than a type --
a bound, a rule about a member, members read together -- is the
record's own `problems`, which yields them as it finds them; the
constructor stops at the first, `coerce` reports every one, and a
reader may ask a record before building anything.

What an entity writes follows from the same fields: `to_json` is
written once here, the parser's inverse over each field's annotation.
What it simplifies to is its own: `canonical` defaults to the entity
itself, and an entity that contains entities canonicalizes them there.

Composition -- what needs the document or the codec chain -- is the
entity's to answer through `incoming_problems`, `shape_problems`,
`fill_value_problems`, `transition` and `grid`, each taking the part of
the document it needs. The document that composes those answers is
`_document`; which entities are in scope is `_registry`, which refuses,
at registration, an entity `coerce` could not read.
"""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Final,
    Literal,
    TypeAlias,
    TypeVar,
    cast,
    get_args,
)

from zarr_metadata.model._validation import MetadataValidationError, ValidationProblem
from zarr_metadata.v3._typed_json import (
    Loc,
    Parsed,
    Parser,
    RecordWriter,
    Writer,
    as_tuples,
    declared_class_vars,
    field_hints,
    is_integer,
    is_optional,
    is_union,
    parser,
    parser_for,
    problem,
    record_writer,
    strip_annotation,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from typing import Self

    from zarr_metadata._common import JSONValue
    from zarr_metadata.v3._common import ZarrV3MetadataFieldJSON
    from zarr_metadata.v3._parts import ArrayParts, ChunkGrid
    from zarr_metadata.v3._registry import Context

EntityT = TypeVar("EntityT", bound="MetadataEntity")

# A real alias, not a string one, so a return annotation can subscript
# it -- `Coerced[Self]` -- without deferring annotation evaluation.
Coerced: TypeAlias = tuple[EntityT | None, tuple[ValidationProblem, ...]]
"""The entity if it could be built, and every problem found.

One direction holds: no entity means at least one problem. The converse
does not -- a survivable problem, an unknown key, comes back *with* the
entity, because the entity is still readable and saying so is more
useful than refusing. A member of the wrong type is not survivable:
the entity's rules are written over a whole configuration, and an
entity is never built around a hole.

So test `entity is None` to decide whether to go on reading, and test the
problems to decide the verdict. They are different questions.
"""


StorageClass = Literal["single_byte", "multi_byte", "variable_length"]
"""How one scalar of a data type occupies bytes.

`single_byte` and `multi_byte` are both fixed-size; they differ only in
whether a byte order applies, which is what the `bytes` codec's `endian`
member is about.
"""


class _FromName:
    """The marker behind `FROM_NAME`."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "FROM_NAME"


FROM_NAME: Final = _FromName()
"""Marks a field carried by the metadata envelope's `name`, not its configuration.

    data_type_name: Annotated[str, FROM_NAME]

A member all the same -- `__post_init__` judges it -- but not a
configuration key, so it is neither read from nor written to a
`configuration` object. The raw-bytes family is the case: `r<N>` keeps its
width in its name and has no configuration at all.
"""


def is_from_name(annotation: object) -> bool:
    """Whether `FROM_NAME` marks the field: carried by the envelope's name, not a configuration key."""
    return any(entry is FROM_NAME for entry in strip_annotation(annotation)[1])


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
) -> tuple[str | None, Mapping[str, object] | None, tuple[ValidationProblem, ...]]:
    """Split metadata into `(name, configuration, problems)`.

    The shared shape every entity arrives in: a bare name, or an object
    carrying one. A `None` name means the value is not a metadata field
    at all; a `None` configuration means the bare spelling was used, or
    the key was left out. A configuration that is present and not an
    object is the one problem reported, at `("configuration",)`.
    """
    if isinstance(value, str):
        return value, None, ()
    if not isinstance(value, Mapping):
        return None, None, ()
    entry = cast("Mapping[str, object]", value)
    name = entry.get("name")
    if not isinstance(name, str):
        return None, None, ()
    if "configuration" not in entry:
        return name, None, ()
    configuration = entry["configuration"]
    if not isinstance(configuration, Mapping):
        return name, None, problem(("configuration",), f"expected an object, got {configuration!r}")
    return name, cast("Mapping[str, object]", configuration), ()


def is_metadata_field(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
    """A nested metadata field: a bare name or a named-configuration object.

    Only the envelope's shape. Which entity the name denotes, and whether
    its configuration is well formed, is settled when the containing
    entity reads it in scope.
    """
    if not isinstance(value, (str, Mapping)):
        return problem(loc, f"expected a metadata field, got {value!r}")
    return ()


@dataclass(frozen=True, slots=True)
class Opaque:
    """A metadata field this reading did not turn into an entity.

    Carries the JSON the document wrote, so a reader holding a
    `CodecEntity | Opaque` has everything the document said in either
    case, and `reason` says which case it is. `out_of_scope` is a name no
    entity in this `Context` claims -- an extension this reader does not
    model, which is not an error and is the reader's cue to resolve it
    elsewhere. `invalid` is a name that *was* claimed and then refused,
    for the reasons reported alongside.

    Answers `to_json` and `canonical` as an entity does, so a field typed
    `CodecEntity | Opaque` is written and simplified without asking
    which case it holds.
    """

    json: object
    reason: Literal["out_of_scope", "invalid"]

    def to_json(self) -> ZarrV3MetadataFieldJSON:
        """The JSON the document wrote, as it wrote it.

        An `Opaque` inside a built entity is out of scope -- an inner
        name no entity in scope claimed -- and its JSON passed the
        envelope check as a metadata field, which is what the cast says.
        """
        return cast("ZarrV3MetadataFieldJSON", self.json)

    def canonical(self) -> Self:
        """Itself: what was not read cannot be simplified."""
        return self


def unreadable(cls: type[MetadataEntity]) -> str | None:
    """Why `coerce` could not read an instance of `cls`; None if it can.

    An entity's fields are `configuration`, which a `Configured` entity
    narrows to its own `Configuration` record, and at most one field the
    envelope's name fills. Things that type-check cleanly and then go
    wrong somewhere that will not name the class: a field of any other
    name; a `configuration` on an entity that is not `Configured`; a
    configuration that is not a `Configuration`, or a member of it whose
    annotation is not a shape JSON takes; a `__post_init__` of the
    entity's own, whose rules `coerce` would never ask; and a class
    variable a base annotates and nothing sets -- `identifier` for every
    entity, `bounds` for an integer type -- which the first lookup would
    fail. Registration asks, and refuses the class with the answer.
    """
    try:
        hints = field_hints(cls)
    except NameError as unresolved:
        return _unresolved(cls, unresolved)
    for name, annotation in hints.items():
        if name == "configuration" and issubclass(cls, Configured):
            continue
        if is_from_name(annotation):
            continue
        if name == "configuration":
            return (
                f"{cls.__name__} declares `configuration` without `Configured`; an entity with a "
                f"configuration adds it beside its kind: class {cls.__name__}(..., Configured)"
            )
        return (
            f"{cls.__name__} declares a field {name!r}; an entity's fields are `configuration`, "
            "a frozen dataclass of its members, and a name it carries marked FROM_NAME -- put "
            f"{name!r} in the configuration record"
        )
    if issubclass(cls, Configured):
        record = hints["configuration"]
        if not (isinstance(record, type) and issubclass(record, Configuration)):
            return (
                f"{cls.__name__}: configuration is annotated {record!r}; annotate it with a frozen "
                "dataclass subclassing Configuration, one field per configuration member"
            )
        try:
            members = field_hints(record)
        except NameError as unresolved:
            return _unresolved(record, unresolved)
        unread: list[str] = []
        for name, annotation in members.items():
            try:
                accepted = parser_for(annotation, _nested_field) is not None
            except TypeError as refused:
                return f"{cls.__name__}: {name} {refused}"
            if not accepted:
                unread.append(name)
        if len(unread) != 0:
            return (
                f"{cls.__name__}: "
                f"{'; '.join(f'{name} is annotated {members[name]!r}' for name in unread)}"
                ", which is not a shape JSON takes. A member is int, float, bool, str, JSONValue, "
                "a Literal of names, tuple[T, ...] or tuple[T1, T2], a TypedDict or dataclass "
                "record, Mapping[str, V], a NewType, or an entity kind with Opaque "
                "(CodecEntity | Opaque); add | UNSET for an optional member, and put any finer "
                "rule in the function bound as `problems`"
            )
    if "__post_init__" in vars(cls):
        return (
            f"{cls.__name__} defines __post_init__; write its rules as `problems` on its "
            "Configuration record, yielding each: the constructor stops at the first, `coerce` "
            "reports every one"
        )
    annotated = declared_class_vars(cls)
    missing = sorted(name for name in annotated if not hasattr(cls, name))
    if len(missing) != 0:
        owed = ", ".join(f"{name} (annotated by {annotated[name].__name__})" for name in missing)
        return f"{cls.__name__} does not declare {owed}; set each as a class variable"
    return None


def _unresolved(cls: type, unresolved: NameError) -> str:
    return (
        f"{cls.__name__}: a field annotation names {unresolved.name!r}, which is not "
        "defined where the class is; define it at module level, or import it outside "
        "`TYPE_CHECKING`"
    )


def nested_kind(annotation: object) -> type[MetadataEntity] | None:
    """The kind an annotation of the form `Kind | Opaque` names; None if it names no entity.

    The one shape a field holding another entity takes, with `Opaque`
    because that is what the field holds when the inner name is out of
    scope, and a kind -- or a subclass of one, `GzipCodec` -- because a
    scope resolves names by kind. An annotation naming an entity any
    other way is a `TypeError` saying so, which registration reports
    against the field.
    """
    parts = get_args(annotation) if is_union(annotation) else (annotation,)
    entities = [
        part for part in parts if isinstance(part, type) and issubclass(part, MetadataEntity)
    ]
    if len(entities) == 0:
        return None
    kind = entities[0]
    if len(entities) == 1 and set(parts) == {kind, Opaque} and kind_of(kind) is not None:
        return kind
    msg = (
        "holds an entity but is not written as its kind | Opaque (CodecEntity | Opaque), "
        "which is what the field holds when the inner name is out of scope; a kind is a "
        "codec kind, DataTypeEntity, ChunkGridEntity, ChunkKeyEncodingEntity or "
        "StorageTransformerEntity, or a subclass of one"
    )
    raise TypeError(msg)


@dataclass(frozen=True, slots=True)
class _Reading:
    """What one reading hands down into the fields that hold entities.

    The scope the inner entities are read in, and the problems found
    inside them, kept apart from the containing entity's own: its rules
    still run over its own members when only a contained entity is
    wrong.
    """

    context: Context
    nested: list[ValidationProblem]


def _nested_field(annotation: object) -> Parser[_Reading] | None:
    """The parser for a field holding another entity: `Kind | Opaque`, read in the reading's scope.

    The entity layer's one shape of its own, asked by the parser at
    every depth, so a struct's fields' data types and a shard's inner
    pipelines are read the same way. The envelope's shape is the
    containing field's own problem; what is found inside the entity
    goes to the reading.
    """
    kind = nested_kind(annotation)
    if kind is None:
        return None

    def parse(value: object, loc: Loc, reading: _Reading) -> Parsed:
        problems = is_metadata_field(value, loc)
        if len(problems) != 0:
            return value, problems
        entity, found = reading.context.coerce(kind, value, loc)
        reading.nested.extend(found)
        return entity, ()

    return parse


def _nested_field_writer(annotation: object) -> Writer | None:
    """The writer for a field holding another entity: what it holds, as it writes itself."""
    if nested_kind(annotation) is None:
        return None

    def write(value: object) -> JSONValue:
        if isinstance(value, (MetadataEntity, Opaque)):
            return value.to_json()
        msg = f"{value!r} is not an entity or an Opaque"
        raise TypeError(msg)

    return write


@dataclass(frozen=True, slots=True)
class _Plan:
    """How `coerce` reads and `to_json` writes one class, compiled once from its fields."""

    from_name: str | None
    """The field the envelope's name fills, for a family; None for every other entity."""
    parse: Parser[_Reading] | None
    """The configuration record's parser; None for an entity that is not `Configured`."""
    write: Callable[[Configured], dict[str, JSONValue]] | None
    """The entity's configuration as the JSON object it writes; None for one that is not `Configured`."""
    requires_configuration: bool
    """Whether the record has a member the document must write."""


@functools.cache
def _plan(cls: type[MetadataEntity]) -> _Plan:
    """The plan for `cls`, a pure function of the class: its fields are fixed once it exists.

    Each parser is a function of its annotation alone, taking the reading
    it runs in as an argument. `TypeError` for a shape no parser reads,
    which registration refuses first.
    """
    hints = field_hints(cls)
    from_name = next((key for key, annotation in hints.items() if is_from_name(annotation)), None)
    if not issubclass(cls, Configured):
        return _Plan(from_name, None, None, False)
    record = hints["configuration"]
    if not (isinstance(record, type) and issubclass(record, Configuration)):  # pragma: no cover
        msg = f"{cls.__name__}: configuration is annotated {record!r}, not a Configuration"
        raise TypeError(msg)
    required = any(not is_optional(annotation) for annotation in field_hints(record).values())
    writes: RecordWriter = record_writer(record, _nested_field_writer)

    def write(entity: Configured) -> dict[str, JSONValue]:
        return writes(entity.configuration)

    return _Plan(from_name, parser(record, _nested_field), write, required)


@dataclass(frozen=True)
class Configuration:
    """What an entity is configured with: a record of its members, and the rules on them.

    A frozen dataclass whose fields are the configuration's members,
    each a shape JSON takes; the entity names it in its `configuration`
    field. `problems` is where everything finer than a type goes -- a
    bound, a rule about one member, members read together -- yielding
    each problem as it is found, located relative to the configuration.
    A reader stops at the first or collects them all, as it needs: the
    entity's constructor stops at the first, `coerce` reports every one,
    and `BloscOptions(...).problems()` answers without an entity at all.
    """

    def problems(self) -> Iterator[ValidationProblem]:
        """Every reason these values are not allowed, yielded as found. Default: none."""
        yield from ()


@dataclass(frozen=True)
class Configured:
    """The half of an entity that has a configuration.

    An entity whose metadata carries a `configuration` object adds this
    beside its kind -- `class GzipCodec(BytesBytesCodec, Configured)` --
    and narrows the field to its own record: `configuration:
    GzipOptions`. What the layer does with a configuration -- parse it,
    ask its rules, write it back, replace members of it -- is done here
    or asked of this, and an entity that is not `Configured` has none
    of it: its metadata is a bare name.
    """

    configuration: Configuration

    def with_configuration(self, **changes: object) -> Self:
        """This entity with these configuration members changed.

        `codec.with_configuration(typesize=UNSET)` is the record replaced
        member by member and the entity rebuilt around it, so the
        constructor checks the result as it checks any other.
        """
        return replace(self, configuration=replace(self.configuration, **changes))


@dataclass(frozen=True)
class MetadataEntity(ABC):
    """One named entity, coerced from its metadata.

    Subclasses add their configuration members as fields, which is what
    makes them well-typed when read: `coerce` builds one only from
    metadata it accepted. Built by hand, the types are the caller's
    promise -- `__post_init__` judges values, not types. An optional member is
    typed `| UNSET` with a default of `UNSET`, so absence is representable
    -- and distinct from a `null` the document wrote -- and a canonical
    spelling can leave it out.

    Frozen, so an entity of hashable members is hashable. One holding a
    value out of scope is not, because that value is the JSON the document
    wrote and a JSON object is a `dict` -- the same way any frozen
    dataclass holding a list is unhashable. It cannot be an immutable
    mapping instead: `MappingProxyType` is unhashable too, and anything
    else stops `json.dumps` from serializing what `to_json` returns.

    A subclass names its configuration record, a `Configuration` whose
    `problems` holds what the spec says beyond the members' types -- so
    `BloscCodec(BloscOptions(clevel=99))` raises on the first, and
    `coerce` reports every one instead -- and writes `canonical` where
    two spellings of its members mean the same. `coerce` and `to_json`
    are written once here, against what the record says.
    """

    identifier: ClassVar[str]
    """The name this entity is registered under.

    Usually the `name` the metadata carries. The raw-bytes data types are
    the exception: every `r<N>` spelling is one family, so the family gets
    an invented identifier that no real name can collide with.
    """

    @classmethod
    def name_problems(cls, name: str) -> Iterator[ValidationProblem]:
        """Why `name`, which `accepts` claimed, is not a well-formed name of this family.

        For a family, whose names carry data -- `r<N>` -- and which claims
        a malformed member so that it is reported rather than waved
        through as an unknown extension. Locations are relative to the
        entity: `()`. Default: none, for an entity of one name.
        """
        yield from ()

    def __post_init__(self) -> None:
        """Refuse the first problem the rules find, so `BloscCodec(BloscOptions(clevel=99))` raises."""
        plan = _plan(type(self))
        name = self.identifier if plan.from_name is None else getattr(self, plan.from_name)
        first = next(type(self).name_problems(name), None)
        if first is None and isinstance(self, Configured):
            first = next(self.configuration.problems(), None)
        if first is not None:
            raise MetadataValidationError((first,))

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

        The configuration is parsed against the record the `configuration`
        field names, member by member; a member holding another entity is
        read in `context`, the scope this reading is happening in. An
        optional member the document left out is `UNSET` in the record,
        so no field's default decides what a document said. The entity is
        built only when every member of its own read -- its rules are
        written over a whole configuration -- and handed back only when
        everything inside it read too.
        """
        name, given, envelope = named_configuration(value)
        if name is None or not cls.accepts(name):
            return None, problem((), f"expected the {cls.identifier!r} entity")
        if len(envelope) != 0:
            return None, envelope
        plan = _plan(cls)
        members: dict[str, object] = {}
        if plan.from_name is not None:
            members[plan.from_name] = name
        reading = _Reading(context, [])
        own: tuple[ValidationProblem, ...] = ()
        if not issubclass(cls, Configured) or plan.parse is None:
            own = tuple(
                found
                for key in (given or {})
                for found in problem(
                    ("configuration", key), f"unexpected key {key!r}", "unknown_key"
                )
            )
        elif given is None and plan.requires_configuration:
            return None, problem(
                ("configuration",),
                f"{cls.identifier!r} requires a configuration",
                "missing_key",
            )
        else:
            # Arrays as tuples before parsing, so a member holds the
            # tuples its type declares, never the lists raw JSON
            # arrives as.
            members["configuration"], own = plan.parse(
                as_tuples({} if given is None else given), ("configuration",), reading
            )
        found = (*own, *reading.nested)
        if any(entry.kind != "unknown_key" for entry in own):
            # An unknown key is survivable; a member that could not be
            # read is a hole, and judging around it would be guessing.
            return None, found
        # The rules, asked of the name and of the record before anything
        # is built: a name problem lands on the entity, a configuration
        # problem under the configuration.
        record = members.get("configuration")
        refused = (
            *cls.name_problems(name),
            *(within((), tuple(record.problems())) if isinstance(record, Configuration) else ()),
        )
        if len(refused) != 0:
            # Values the spec disallows: reported rather than raised,
            # every one.
            return None, (*found, *refused)
        if any(entry.kind != "unknown_key" for entry in reading.nested):
            # A contained entity could not be read. This entity's own
            # rules ran -- an invalid inner is an `Opaque`, as an
            # out-of-scope one is -- but what is handed back is not an
            # entity that would be asked composition questions it cannot
            # answer.
            return None, found
        return cls(**members), found

    def canonical(self) -> Self:
        """This entity in the simplest form that means the same thing.

        A *transformation*, asked for by `canonicalize_array_metadata_v3`
        and by nothing else. `to_json` does not apply it, because writing
        a document back is not the same as asking for it to be rewritten:
        a reader that reads and writes should not change bytes it was not
        asked to change.

        The default is the entity itself. Override it where two spellings
        of the entity's members mean the same -- a rectilinear
        dimension's run-length encoding, a `typesize` that `noshuffle`
        ignores -- and, in an entity that contains entities, to put those
        in canonical form: `replace(self, inner=canonicalized(self.inner))`.
        """
        return self

    def to_json(self) -> ZarrV3MetadataFieldJSON:
        """This entity as a document would write it.

        Written from the configuration record by the same declaration
        `coerce` reads it by, each member by the writer its annotation
        implies: the bare name when every member it holds is absent, the
        object otherwise, a contained entity through its own `to_json`,
        a JSON-valued member copied so the document is not a handle on
        the entity. Faithful to every member: read a document, write it
        back, and those come out as they went in. Ask `canonical` first
        if you want the simplest equivalent spelling. The envelope's
        spelling is the one thing not preserved, because the entity does
        not model it: a bare name, `{"name": x}` and `{"name": x,
        "configuration": {}}` all read to the same entity.

        An entity whose JSON is not its fields overrides this; none in
        the package does.
        """
        plan = _plan(type(self))
        name = self.identifier
        if plan.from_name is not None:
            carried = getattr(self, plan.from_name)
            name = carried if isinstance(carried, str) else name
        if not isinstance(self, Configured) or plan.write is None:
            return name
        configuration = plan.write(self)
        if len(configuration) == 0:
            return name
        return {"name": name, "configuration": configuration}


@dataclass(frozen=True)
class CodecEntity(MetadataEntity):
    """An entity that occupies a position in the codec pipeline.

    Of one of three kinds, each a base class: `ArrayArrayCodec`,
    `ArrayBytesCodec`, `BytesBytesCodec`. The kind fixes where in the
    pipeline the codec may stand, and what it must answer.
    """

    variable_size: ClassVar[bool]
    """Whether this codec's output size depends on the bytes it is given.

    A compressor's does, so a shard index encoded with one has no size
    derivable from metadata alone, and the shard cannot be read. Every
    codec says, because a default in either direction is a verdict.
    """

    def incoming_problems(self, incoming: ArrayParts | None) -> tuple[ValidationProblem, ...]:
        """Why this codec cannot be applied to the array that reaches it.

        `incoming` is None once the chain can no longer say what reaches
        here, and the default answer to that is nothing: declining beats
        guessing. Locations are relative to this codec's `configuration`;
        an empty one lands on the codec itself.
        """
        return ()


@dataclass(frozen=True)
class ArrayArrayCodec(CodecEntity):
    """A codec that transforms the array: what reaches the next codec is its to say."""

    @abstractmethod
    def transition(self, incoming: ArrayParts) -> ArrayParts | None:
        """What the next codec in the chain sees.

        `incoming` itself if this codec leaves the array's shape, grid and
        data type alone; the parts it hands on if it changes one; None if
        that cannot be determined from the metadata, which ends the
        judgments downstream rather than inventing them.
        """


@dataclass(frozen=True)
class ArrayBytesCodec(CodecEntity):
    """The one codec in a pipeline that turns the array into bytes."""


@dataclass(frozen=True)
class BytesBytesCodec(CodecEntity):
    """A codec that transforms bytes, after the array is gone."""


@dataclass(frozen=True)
class ChunkGridEntity(MetadataEntity):
    """An entity that divides an array into the parts a pipeline encodes."""

    def shape_problems(self, array_shape: object) -> tuple[ValidationProblem, ...]:
        """Why this grid does not divide an array of `array_shape`.

        Locations are relative to the grid's `configuration`. Default:
        nothing, for a grid this package reads but has no such rule for.
        """
        return ()

    @abstractmethod
    def grid(self, array_shape: object) -> ChunkGrid:
        """What this grid divides an array of `array_shape` into.

        The array shape is a parameter because neither determines a grid
        alone: a grid whose own metadata cannot be read still has the
        array's rank, and rank is enough for several rules.
        """


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

    @abstractmethod
    def fill_value_problems(self, value: object, loc: Loc = ()) -> tuple[ValidationProblem, ...]:
        """Why `value` is not a fill value of this type, if it is not.

        Every data type answers this; one that accepts any fill value
        says so with `return ()`.
        """


@dataclass(frozen=True)
class ChunkKeyEncodingEntity(MetadataEntity):
    """An entity that says how a chunk's coordinates become a store key."""


@dataclass(frozen=True)
class StorageTransformerEntity(MetadataEntity):
    """An entity that stands between the codec pipeline and the store."""


KINDS: Final[tuple[type[MetadataEntity], ...]] = (
    DataTypeEntity,
    ChunkGridEntity,
    ChunkKeyEncodingEntity,
    CodecEntity,
    StorageTransformerEntity,
)
"""The kinds of entity: what a scope holds a table of, and what a document's fields are read as."""


def kind_of(cls: type[MetadataEntity]) -> type[MetadataEntity] | None:
    """The kind `cls` is of, or None for a class under none of them."""
    return next((kind for kind in cls.__mro__ if kind in KINDS), None)


__all__ = [
    "FROM_NAME",
    "KINDS",
    "ArrayArrayCodec",
    "ArrayBytesCodec",
    "BytesBytesCodec",
    "ChunkGridEntity",
    "ChunkKeyEncodingEntity",
    "CodecEntity",
    "Coerced",
    "Configuration",
    "Configured",
    "DataTypeEntity",
    "Loc",
    "MetadataEntity",
    "Opaque",
    "StorageClass",
    "StorageTransformerEntity",
    "is_from_name",
    "is_integer",
    "is_metadata_field",
    "kind_of",
    "named_configuration",
    "nested_kind",
    "problem",
    "unreadable",
    "within",
]
