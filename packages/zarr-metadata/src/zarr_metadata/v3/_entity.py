"""What every metadata entity can do for itself.

A codec, data type, chunk grid, chunk key encoding or storage transformer
is one frozen dataclass whose fields are its configuration, and the
field annotations are its schema: `coerce` reads a document's
configuration against them, member by member, with `_typed_json`. A
field typed `CodecEntity | Opaque` holds another entity, read through
the scope the containing one is read in. Everything finer than a type --
a bound, a rule about a member, members read together -- is a function
of the entity's instance that yields problems as it finds them, bound
on the class as `problems`; the constructor stops at the first,
`coerce` reports every one.

What an entity writes and what it simplifies to are its own too:
`to_json` is abstract, a literal of the entity's JSON type, and
`canonical` defaults to the entity itself. An entity that contains
entities writes them with `written` and canonicalizes them with
`canonicalized`, in the same two methods.

Composition -- what needs the document or the codec chain -- is the
entity's to answer through `incoming_problems`, `shape_problems`,
`fill_value_problems`, `transition` and `grid`, each taking the part of
the document it needs. The document that composes those answers is
`_document`; which entities are in scope is `_registry`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Final, Literal, TypeAlias, TypeVar, cast, get_args

from zarr_metadata.model._sentinel import UNSET
from zarr_metadata.model._validation import MetadataValidationError, ValidationProblem
from zarr_metadata.v3._typed_json import (
    Loc,
    Parsed,
    Parser,
    as_tuples,
    declared_class_vars,
    field_hints,
    is_integer,
    is_optional,
    is_union,
    parser,
    parser_for,
    problem,
    strip_annotation,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from typing import Self

    from zarr_metadata.v3._common import ZarrV3MetadataFieldJSON
    from zarr_metadata.v3._parts import ArrayParts, ChunkGrid
    from zarr_metadata.v3._registry import Context
    from zarr_metadata.v3._typed_json import Leaf

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
    """

    json: object
    reason: Literal["out_of_scope", "invalid"]


def written(value: MetadataEntity | Opaque) -> ZarrV3MetadataFieldJSON:
    """A contained metadata field as a document would write it.

    The entity's own JSON, or the JSON an `Opaque` kept. An `Opaque`
    inside a built entity is out of scope -- an inner name no entity in
    scope claimed -- and its JSON passed the envelope check as a metadata
    field, which is what the cast says.
    """
    if isinstance(value, MetadataEntity):
        return value.to_json()
    return cast("ZarrV3MetadataFieldJSON", value.json)


def canonicalized(value: EntityT | Opaque) -> EntityT | Opaque:
    """A contained metadata field in canonical form: the entity's own, or the `Opaque` as it is."""
    if isinstance(value, MetadataEntity):
        return value.canonical()
    return value


def nested_kind(annotation: object) -> type[MetadataEntity] | None:
    """The kind an annotation of the form `Kind | Opaque` names; None if it names no entity.

    The one shape a field holding another entity takes, with `Opaque`
    because that is what the field holds when the inner name is out of
    scope, and a kind -- or a subclass of one, `GzipCodec` -- because a
    scope resolves names by kind. An annotation naming an entity any
    other way is a `TypeError` saying so, which class creation reports
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


def _reading(context: Context | None, nested: list[ValidationProblem]) -> Leaf:
    """How a field holding another entity is parsed: `Kind | Opaque`, read in `context`.

    Asked by the parser at every depth, so a struct's fields' data types
    and a shard's inner pipelines are read the same way. The envelope's
    shape is the containing field's own problem; what is found inside the
    entity is collected in `nested`, apart, because the containing
    entity's rules still run over its own members when only a contained
    entity is wrong. With no `context` the shape is checked and nothing
    is read, which is what class creation asks.
    """

    def leaf(annotation: object) -> Parser | None:
        kind = nested_kind(annotation)
        if kind is None:
            return None

        def parse(value: object, loc: Loc) -> Parsed:
            problems = is_metadata_field(value, loc)
            if len(problems) != 0 or context is None:
                return value, problems
            entity, found = context.coerce(kind, value, loc)
            nested.extend(found)
            return entity, ()

        return parse

    return leaf


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

    A subclass writes its fields; where the spec has something to say
    beyond their types, a function of the instance that yields problems,
    bound as `problems` -- so `BloscCodec(clevel=99)` raises on the
    first, and `coerce` reports every one instead; `to_json`, a literal
    of its JSON type; and `canonical` where two spellings of its members
    mean the same. `coerce` is written once here, against what the
    fields say.
    """

    identifier: ClassVar[str]
    """The name this entity is registered under.

    Usually the `name` the metadata carries. The raw-bytes data types are
    the exception: every `r<N>` spelling is one family, so the family gets
    an invented identifier that no real name can collide with.
    """

    def __init_subclass__(cls, *, base: bool = False, **kwargs: object) -> None:
        """Refuse, at class creation, an entity this layer could not read.

        Three things type-check cleanly and then go wrong somewhere that
        will not name the class: a field whose annotation is not a shape
        JSON takes, which `coerce` could not parse; a `__post_init__` of
        the entity's own, whose rules `coerce` would never ask; and a
        class variable a base annotates and nothing sets -- `identifier`
        for every entity, `bounds` for an integer type -- which the first
        lookup would fail. An import-time error in the extension's own
        module is the one place the author is looking.

        `base=True` for a class that exists to add a class variable
        rather than to be an entity -- `CodecEntity`, `IntegerDataType`.
        """
        super().__init_subclass__(**kwargs)
        if base:
            return
        hints = field_hints(cls)
        unread: list[str] = []
        for name, annotation in hints.items():
            try:
                accepted = parser_for(annotation, _reading(None, [])) is not None
            except TypeError as refused:
                msg = f"{cls.__name__}: {name} {refused}"
                raise TypeError(msg) from None
            if not accepted:
                unread.append(name)
        if len(unread) != 0:
            msg = (
                f"{cls.__name__}: "
                f"{'; '.join(f'{name} is annotated {hints[name]!r}' for name in unread)}"
                ", which is not a shape JSON takes. A field is int, float, bool, str, JSONValue, "
                "a Literal of names, tuple[T, ...] or tuple[T1, T2], a TypedDict or dataclass "
                "record, Mapping[str, V], a NewType, or an entity kind with Opaque "
                "(CodecEntity | Opaque); add | UNSET for an optional member, and put any finer "
                "rule in the function bound as `problems`"
            )
            raise TypeError(msg)
        if "__post_init__" in vars(cls):
            msg = (
                f"{cls.__name__} defines __post_init__; write its rules as a function of the "
                "instance that yields problems and bind it as `problems = <function>`: the "
                "constructor stops at the first problem it yields, `coerce` reports every one"
            )
            raise TypeError(msg)
        annotated = declared_class_vars(cls)
        missing = sorted(name for name in annotated if not hasattr(cls, name))
        if len(missing) != 0:
            owed = ", ".join(
                f"{name} (annotated by {annotated[name].__name__})" for name in missing
            )
            msg = (
                f"{cls.__name__} does not declare {owed}; set each as a class variable, "
                "or pass base=True if this class exists only to be subclassed"
            )
            raise TypeError(msg)

    def problems(self, /) -> Iterator[ValidationProblem]:
        """Every reason this entity's values are not allowed, yielded as found.

        The entity's own rules -- a bound, a rule about one member,
        members read together -- written as a function of the instance
        and bound on the class: `problems = blosc_problems`. Locations
        are relative to the configuration. A consumer stops at the first
        or collects them all, as it needs: the constructor stops at the
        first, `coerce` collects every one. Default: none.
        """
        yield from ()

    def __post_init__(self) -> None:
        """Refuse the first problem `problems` finds, so `BloscCodec(clevel=99)` raises."""
        first = next(self.problems(), None)
        if first is not None:
            raise MetadataValidationError((first,))

    @classmethod
    def _unchecked(cls, members: Mapping[str, object]) -> Self:
        """The instance `cls(**members)` would build, without asking `problems`.

        For `coerce`, which asks `problems` itself and reports every one,
        where the constructor stops at the first.
        """
        entity = object.__new__(cls)
        for name, value in members.items():
            object.__setattr__(entity, name, value)
        return entity

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

        Each configuration member is parsed against its field's
        annotation; a member holding another entity is read in `context`,
        the scope this reading is happening in. An optional member the
        document left out is passed as `UNSET`, so no field's default
        decides what a document said. The entity is built only when every
        member of its own read -- its rules are written over a whole
        configuration -- and handed back only when everything inside it
        read too.
        """
        name, given, _ = named_configuration(value)
        if name is None or not cls.accepts(name):
            return None, problem((), f"expected the {cls.identifier!r} entity")
        hints = field_hints(cls)
        if given is None and any(
            not is_from_name(annotation) and not is_optional(annotation)
            for annotation in hints.values()
        ):
            return None, problem(
                ("configuration",),
                f"{cls.identifier!r} requires a configuration",
                "missing_key",
            )
        configuration: Mapping[str, object] = {} if given is None else given
        nested: list[ValidationProblem] = []
        reading = _reading(context, nested)
        members: dict[str, object] = {}
        own: list[ValidationProblem] = []
        for key in configuration:
            if key not in hints or is_from_name(hints[key]):
                own.extend(
                    problem(("configuration", key), f"unexpected key {key!r}", "unknown_key")
                )
        for key, annotation in hints.items():
            if is_from_name(annotation):
                members[key] = name
            elif key not in configuration:
                if is_optional(annotation):
                    members[key] = UNSET
                else:
                    own.extend(
                        problem(
                            ("configuration", key), f"missing required key {key!r}", "missing_key"
                        )
                    )
            else:
                # Arrays as tuples before parsing, so a member holds the
                # tuples its type declares, never the lists raw JSON
                # arrives as.
                members[key], problems = parser(annotation, reading)(
                    as_tuples(configuration[key]), ("configuration", key)
                )
                own.extend(problems)
        found = (*own, *nested)
        if any(entry.kind != "unknown_key" for entry in own):
            # An unknown key is survivable; a member that could not be
            # read is a hole, and judging around it would be guessing.
            return None, found
        entity = cls._unchecked(members)
        refused = within((), tuple(entity.problems()))
        if len(refused) != 0:
            # Values the spec disallows: reported rather than raised,
            # every one, located under the configuration.
            return None, (*found, *refused)
        if any(entry.kind != "unknown_key" for entry in nested):
            # A contained entity could not be read. This entity's own
            # rules ran -- an invalid inner is an `Opaque`, as an
            # out-of-scope one is -- but what is handed back is not an
            # entity that would be asked composition questions it cannot
            # answer.
            return None, found
        return entity, found

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

    @abstractmethod
    def to_json(self) -> ZarrV3MetadataFieldJSON:
        """This entity as a document would write it: a literal of its own JSON type.

        Faithful to every member it holds: read a document, write it
        back, and those come out as they went in. Ask `canonical` first
        if you want the simplest equivalent spelling. The envelope's
        spelling is the one thing not preserved, because the entity does
        not model it: a bare name, `{"name": x}` and `{"name": x,
        "configuration": {}}` all read to the same entity, and the entity
        writes the bare name when every member it holds is absent.

        Written per entity, as a literal of its own TypedDict and with
        that TypedDict as the declared return type -- narrower than the
        base's, which is what tells a consumer holding a `GzipCodec` that
        it gets a `GzipCodecObject` -- so pyright checks the literal's keys
        and values against it. A contained entity is written with
        `written`.
        """


@dataclass(frozen=True)
class CodecEntity(MetadataEntity, base=True):
    """An entity that occupies a position in the codec pipeline.

    Of one of three kinds, each a base class: `ArrayArrayCodec`,
    `ArrayBytesCodec`, `BytesBytesCodec`. The kind fixes where in the
    pipeline the codec may stand, and what it must answer.
    """

    variable_size: ClassVar[bool] = False
    """Whether this codec's output size depends on the bytes it is given.

    A compressor's does, so a shard index encoded with one has no size
    derivable from metadata alone, and the shard cannot be read.
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
class ArrayArrayCodec(CodecEntity, base=True):
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
class ArrayBytesCodec(CodecEntity, base=True):
    """The one codec in a pipeline that turns the array into bytes."""


@dataclass(frozen=True)
class BytesBytesCodec(CodecEntity, base=True):
    """A codec that transforms bytes, after the array is gone."""


@dataclass(frozen=True)
class ChunkGridEntity(MetadataEntity, base=True):
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
class DataTypeEntity(MetadataEntity, base=True):
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
class ChunkKeyEncodingEntity(MetadataEntity, base=True):
    """An entity that says how a chunk's coordinates become a store key."""


@dataclass(frozen=True)
class StorageTransformerEntity(MetadataEntity, base=True):
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
    "DataTypeEntity",
    "Loc",
    "MetadataEntity",
    "Opaque",
    "StorageClass",
    "StorageTransformerEntity",
    "canonicalized",
    "is_from_name",
    "is_integer",
    "is_metadata_field",
    "kind_of",
    "named_configuration",
    "nested_kind",
    "problem",
    "within",
    "written",
]
