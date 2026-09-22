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
from dataclasses import dataclass, is_dataclass, replace
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

from zarr_metadata.model._sentinel import UNSET
from zarr_metadata.model._validation import (
    MetadataValidationError,
    ValidationProblem,
    is_json,
    refine_json,
    validate_metadata_field_v3,
)
from zarr_metadata.v3._typed_json import (
    Loc,
    Parsed,
    Parser,
    RecordWriter,
    Writer,
    declared_class_vars,
    field_hints,
    is_integer,
    is_optional,
    is_union,
    members_of,
    parser,
    parser_for,
    problem,
    record_of,
    record_writer,
    strip_annotation,
    without_unset,
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

    Built by the reader through `create_unchecked`, from JSON it has
    refined; the constructor checks one built by hand.
    """

    json: JSONValue
    reason: Literal["out_of_scope", "invalid"]

    def __post_init__(self) -> None:
        """Refuse a reason that is not one of the reader's two, and a `json` that is not JSON."""
        reason: object = self.reason
        found = (
            *(
                ()
                if reason in ("out_of_scope", "invalid")
                else problem(("reason",), f"expected 'out_of_scope' or 'invalid', got {reason!r}")
            ),
            *(
                ()
                if is_json(self.json)
                else problem(("json",), f"expected JSON, got {self.json!r}")
            ),
        )
        if len(found) != 0:
            raise MetadataValidationError(found)

    @classmethod
    def create_unchecked(cls, json: JSONValue, reason: Literal["out_of_scope", "invalid"]) -> Self:
        """An `Opaque` built without the constructor's check, for the reader, whose JSON is refined."""
        opaque = object.__new__(cls)
        object.__setattr__(opaque, "json", json)
        object.__setattr__(opaque, "reason", reason)
        return opaque

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

    An entity's fields are `configuration`, which an entity narrows to
    its own `Configuration` record or defaults to the empty one, and at
    most one field the envelope's name fills. Things that type-check
    cleanly and then go wrong somewhere that will not name the class: a
    field of any other name; a configuration that is not a
    `Configuration`, or a member of it whose annotation is not a shape
    JSON takes, or is itself a `Configuration`, whose rules nothing would
    ask; a `__post_init__` of the entity's or its record's own, since the
    rules go in `problems`; and a class variable a base annotates and
    nothing sets -- `identifier` for every entity, `bounds` for an
    integer type -- which the first lookup would fail. Registration asks,
    and refuses the class with the answer.
    """
    try:
        hints = field_hints(cls)
    except NameError as unresolved:
        return _unresolved(cls, unresolved)
    for name, annotation in hints.items():
        if name == "configuration" or is_from_name(annotation):
            continue
        return (
            f"{cls.__name__} declares a field {name!r}; an entity's fields are `configuration`, "
            "a frozen dataclass of its members, and a name it carries marked FROM_NAME -- put "
            f"{name!r} in the configuration record"
        )
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
    if record.__post_init__ is not Configuration.__post_init__:
        return (
            f"{record.__name__} defines __post_init__; write its rules as `problems`, "
            "yielding each: the entity's constructor stops at the first, `coerce` reports "
            "every one"
        )
    unread: list[str] = []
    for name, annotation in members.items():
        inner = without_unset(strip_annotation(annotation)[0])
        if isinstance(inner, type) and issubclass(inner, Configuration):
            return (
                f"{cls.__name__}: {name} is annotated {inner.__name__}, a Configuration, whose "
                "rules nothing would ask; a member that is an object is a plain record "
                "dataclass or a TypedDict, and its rules belong to the entity's configuration"
            )
        try:
            accepted = parser_for(annotation, _nested_field) is not None
        except TypeError as refused:
            return f"{cls.__name__}: {name} {refused}"
        except NameError as unresolved:
            return _unresolved(record, unresolved)
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
            "rule in the record's `problems`"
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
        # The parser hands refined JSON; its own type is `object` because
        # the checker knows no JSON type.
        entity, found = _resolve_field(cast("JSONValue", value), kind, reading.context, loc)
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


def resolve(
    data: object,
    kind: type[EntityT],
    context: Context,
    loc: Loc = (),
) -> tuple[EntityT | Opaque, tuple[ValidationProblem, ...]]:
    """`data`, one metadata field, read as an entity of `kind` in `context`.

    The reader for one field: the first two layers of reading a
    document, applied to a field on its own. The first needs nothing but
    the value -- `data` is refined to JSON, arrays as tuples, and judged
    as a metadata field, an extra member, a `configuration` that is not
    an object or a `must_understand` that is not a boolean or is `false`
    each a problem. The second needs `context`: the identifier in the
    field is related to a concrete class through it, and that class owns
    the validation routine, its `coerce`, which is handed the field.

    What comes back is the entity, or an `Opaque` saying why not. A name
    no class in `context` claims is `out_of_scope` -- an unmodelled
    extension, left unjudged, which is what makes the format open. A
    name claimed and refused, or of another kind than this position
    takes, is `invalid`, for the reasons reported alongside; so is a
    value that is not JSON, or names no entity. `loc` prefixes the
    problems, so they point at where in the containing configuration the
    field sat.
    """
    refined, problems = refine_json(data, loc)
    if refined is None:
        return Opaque.create_unchecked(None, "invalid"), problems
    return _resolve_field(refined, kind, context, loc)


def _resolve_field(
    data: JSONValue, kind: type[EntityT], context: Context, loc: Loc
) -> tuple[EntityT | Opaque, tuple[ValidationProblem, ...]]:
    """A refined field, its envelope judged, then read.

    What a nested field gets: a metadata field is a metadata field
    wherever it appears, so the envelope gets the same structural
    judgment here that the model layer gives a top-level one.
    """
    problems = tuple(
        ValidationProblem((*loc, *found.loc), found.message, found.kind)
        for found in validate_metadata_field_v3(data, allow_must_understand_false=False)
    )
    entity, found = read_field(data, kind, context, loc)
    return entity, (*problems, *found)


def read_field(
    data: JSONValue, kind: type[EntityT], context: Context, loc: Loc = ()
) -> tuple[EntityT | Opaque, tuple[ValidationProblem, ...]]:
    """The second layer for one field whose envelope the first has judged.

    What `resolve` does once the envelope is judged, and what the
    document reader does for a top-level field, whose envelope the model
    layer judged with the document: relate the identifier in `data` to a
    class through `context`, and hand the class the field. The class is
    asked whenever there is one to ask -- a stray member or a malformed
    `must_understand` says nothing about the configuration -- and not
    when the value names no entity or its configuration is not an
    object, which the envelope judgment has said.
    """
    name, _, malformed = named_configuration(data)
    if name is None or len(malformed) != 0:
        return Opaque.create_unchecked(data, "invalid"), ()
    # Asked with the kind's kind, so a class of the wrong kind for this
    # position is found, and told apart from a name nothing claims.
    registered = kind_of(kind)
    entity_type = None if registered is None else context.claimant(registered, name)
    if entity_type is None:
        return Opaque.create_unchecked(data, "out_of_scope"), ()
    if not issubclass(entity_type, kind):
        # In scope, so not for another reader to resolve: the name is an
        # entity of the wrong kind for this position.
        return Opaque.create_unchecked(data, "invalid"), (
            ValidationProblem(
                loc,
                f"expected {_an(kind.__name__)}, got {name!r}, "
                f"{_an(_refinement(entity_type, kind).__name__)}",
                "invalid_value",
            ),
        )
    entity, found = entity_type.coerce(data, context)
    problems = tuple(
        ValidationProblem((*loc, *entry.loc), entry.message, entry.kind) for entry in found
    )
    if entity is None:
        return Opaque.create_unchecked(data, "invalid"), problems
    return entity, problems


def _refinement(entity: type[MetadataEntity], kind: type[MetadataEntity]) -> type[MetadataEntity]:
    """The class just below `kind`'s kind that `entity` is: `ArrayArrayCodec` for a transpose codec."""
    mro = entity.__mro__
    return mro[mro.index(kind_of(kind) or MetadataEntity) - 1]


def _an(noun: str) -> str:
    """`noun` with its indefinite article: `an ArrayArrayCodec`, `a BytesBytesCodec`."""
    return f"an {noun}" if noun[:1].upper() in "AEIOU" else f"a {noun}"


def held_problems(
    value: object, kind: type[MetadataEntity], loc: Loc
) -> tuple[ValidationProblem, ...]:
    """Why `value` is not what a field typed `kind | Opaque` holds: an entity of the kind, or an `Opaque`."""
    if isinstance(value, (kind, Opaque)):
        return ()
    return problem(loc, f"expected an entity of {kind.__name__} or an Opaque, got {value!r}")


def _held_field(annotation: object) -> Parser[None] | None:
    """The check on a value a record holds in a field typed as an entity or as a record.

    The leaf the record's constructor checks with, where `_nested_field`
    is the one a document is read with: a field typed `Kind | Opaque`
    holds an instance of the kind or an `Opaque`, and a field typed as a
    record dataclass holds an instance of it, checked in turn. Every
    other shape is one JSON takes, and a Python value of that shape
    passes the same parser.
    """
    kind = nested_kind(annotation)
    if kind is not None:

        def holds_entity(value: object, loc: Loc, state: None) -> Parsed:
            return value, held_problems(value, kind, loc)

        return holds_entity
    inner = without_unset(strip_annotation(annotation)[0])
    if isinstance(inner, type) and is_dataclass(inner):

        def holds_record(value: object, loc: Loc, state: None) -> Parsed:
            if not isinstance(value, inner):
                return value, problem(loc, f"expected {inner.__name__}, got {value!r}")
            return value, tuple(
                ValidationProblem((*loc, *found.loc), found.message, found.kind)
                for found in mistyped(value)
            )

        return holds_record
    return None


@functools.cache
def _checks(record: type) -> dict[str, tuple[bool, Parser[None]]]:
    """Each field of a record: whether it may be `UNSET`, and the check on a value held in it."""
    return {
        name: (is_optional(annotation), parser(annotation, _held_field))
        for name, annotation in field_hints(record).items()
    }


def mistyped(record: object) -> tuple[ValidationProblem, ...]:
    """Every field of `record` holding a value not of its type, located at the field.

    The runtime half of a record's type. Pyright checks the values a
    caller writes; this checks the ones it cannot see -- through
    `**changes`, through `replace`, through anything typed `object` --
    with the parsers a document is read by, so a record is well-typed
    however it was built.
    """
    found: list[ValidationProblem] = []
    for name, (optional, check) in _checks(type(record)).items():
        value = getattr(record, name)
        if optional and value is UNSET:
            continue
        # An unknown key in a mapping member is the reader's report, not
        # a type: the record holds what the document said, as `coerce`
        # lets it, and a hand-built one may say the same.
        found.extend(
            entry for entry in check(value, (name,), None)[1] if entry.kind != "unknown_key"
        )
    return tuple(found)


@dataclass(frozen=True, slots=True)
class _Plan:
    """How `coerce` reads and `to_json` writes one class, compiled once from its fields."""

    from_name: str | None
    """The field the envelope's name fills, for a family; None for every other entity."""
    record: type[Configuration]
    """The record the entity declares, which its constructor holds it to."""
    read: Callable[
        [object, Loc, _Reading], tuple[Configuration | None, tuple[ValidationProblem, ...]]
    ]
    """The configuration record, built when every member read, and the problems found."""
    write: Callable[[MetadataEntity], dict[str, JSONValue]]
    """The entity's configuration as the JSON object it writes; empty for a bare name."""
    requires_configuration: bool
    """Whether the record has a member the document must write."""


@functools.cache
def _plan(cls: type[MetadataEntity]) -> _Plan:
    """The plan for `cls`, a pure function of the class: its fields are fixed once it exists.

    Each parser is a function of its annotation alone, taking the reading
    it runs in as an argument. `TypeError` for a shape no parser reads,
    which registration refuses first. The record is built through
    `create_unchecked`: its members were type-checked by the parsers the
    constructor would use, and a read does each check once.
    """
    hints = field_hints(cls)
    from_name = next((key for key, annotation in hints.items() if is_from_name(annotation)), None)
    record = hints["configuration"]
    if not (isinstance(record, type) and issubclass(record, Configuration)):  # pragma: no cover
        msg = f"{cls.__name__}: configuration is annotated {record!r}, not a Configuration"
        raise TypeError(msg)
    required = any(not is_optional(annotation) for annotation in field_hints(record).values())
    members = members_of(field_hints(record), _nested_field)
    if members is None:  # pragma: no cover - registration refused the member first
        msg = f"{cls.__name__}: a configuration member is not a shape JSON takes"
        raise TypeError(msg)
    parse = record_of(record.create_unchecked, members)
    writes: RecordWriter = record_writer(record, _nested_field_writer)

    def read(
        value: object, loc: Loc, reading: _Reading
    ) -> tuple[Configuration | None, tuple[ValidationProblem, ...]]:
        typed, found = parse(value, loc, reading)
        # The parser builds the record only from an object whose every
        # member read; anything else comes back as it came.
        return (typed if isinstance(typed, Configuration) else None), found

    def write(entity: MetadataEntity) -> dict[str, JSONValue]:
        return writes(entity.configuration)

    return _Plan(from_name, record, read, write, required)


@dataclass(frozen=True)
class Configuration:
    """What an entity is configured with: a record of its members, and the rules on them.

    A frozen dataclass whose fields are the configuration's members,
    each a shape JSON takes; the entity names it in its `configuration`
    field, and an entity of a bare name defaults it to this one, empty,
    since the spec makes an absent configuration and an empty one the
    same.
    `problems` is where everything finer than a type goes -- a
    bound, a rule about one member, members read together -- yielding
    each problem as it is found, located relative to the configuration.
    A reader stops at the first or collects them all, as it needs: the
    entity's constructor stops at the first, `coerce` reports every one,
    and `BloscOptions(...).problems()` answers without an entity at all.

    The constructor refuses a member of the wrong type, so a record is
    well-typed however it was built -- by hand, through `replace`,
    through an entity's `with_configuration` -- and the rules can trust
    what they read. Values are the rules' business, and the entity's
    constructor asks them.
    """

    def __post_init__(self) -> None:
        """Refuse every member of the wrong type, so `GzipOptions(level="high")` raises."""
        found = mistyped(self)
        if len(found) != 0:
            raise MetadataValidationError(found)

    @classmethod
    def create_unchecked(cls, **members: object) -> Self:
        """This record with these members, built without the constructor's check.

        The one way around the check, for a caller that has just made
        it: the parser, which type-checked every member against the same
        annotations before building the record. Every field is given --
        the parser gives an absent optional member as `UNSET` -- since
        nothing here applies a default. Anything that has not checked
        the members goes through the constructor.
        """
        record = object.__new__(cls)
        for name, value in members.items():
            object.__setattr__(record, name, value)
        return record

    def problems(self) -> Iterator[ValidationProblem]:
        """Every reason these values are not allowed, yielded as found. Default: none."""
        yield from ()


@dataclass(frozen=True)
class MetadataEntity(ABC):
    """One named entity, coerced from its metadata.

    An entity is well-typed and allowed however it was built. `coerce`
    builds one only from metadata it accepted, through `create_unchecked`
    once it has; by hand, the record's constructor refuses a member of
    the wrong type, the entity's refuses a record that is not its own
    and then a value the rules disallow, and `replace` and
    `with_configuration` go through both. An optional member is typed
    `| UNSET` with a default of `UNSET`, so absence is representable --
    and distinct from a `null` the document wrote -- and a canonical
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
    two spellings of its members mean the same. An entity of a bare
    name defaults the field to the empty record. `coerce` and `to_json`
    are written once here, against what the record says.
    """

    configuration: Configuration
    """The record of this entity's members.

    An entity with members narrows it to its own record, `configuration:
    GzipOptions`, its one positional argument. An entity of a bare name
    defaults it to the empty record -- `configuration: Configuration =
    field(default_factory=Configuration)` -- so that `Crc32cCodec()`
    builds; `coerce` passes the record either way.
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
        """Refuse a record that is not this entity's own, then the first problem the rules find.

        The runtime half of the entity's type, as the record's constructor
        is of the record's: `GzipCodec(BloscOptions(...))` and a family
        member carrying a name that is not a string are refused before
        any rule reads them. Then `BloscCodec(BloscOptions(clevel=99))`
        raises on the first problem the rules yield.
        """
        plan = _plan(type(self))
        if not isinstance(self.configuration, plan.record):
            raise MetadataValidationError(
                problem(
                    (),
                    f"expected a {plan.record.__name__} configuration, got "
                    f"{type(self.configuration).__name__}",
                )
            )
        name: object = self.identifier if plan.from_name is None else getattr(self, plan.from_name)
        if not isinstance(name, str):
            raise MetadataValidationError(problem((), f"expected a string name, got {name!r}"))
        first = next(type(self).name_problems(name), None)
        if first is None:
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

    @property
    def name(self) -> str:
        """The name this entity carries, as a document writes it.

        The identifier, except for a family whose name carries a value:
        the raw-bytes family's identifier is invented and belongs in no
        message a reader sees, and its name is the `r24` the document
        wrote. Anything a reader sees wants this; anything looking a
        class up wants `identifier`.
        """
        from_name = _plan(type(self)).from_name
        return type(self).identifier if from_name is None else cast("str", getattr(self, from_name))

    @classmethod
    def create_unchecked(cls, **fields: object) -> Self:
        """This entity with these fields, built without the constructor's checks.

        The one way around them, for a caller that has just made them:
        `coerce`, which type-checked the record and ran the rules before
        building. Every field is given -- the record, and the carried
        name for a family -- since nothing here applies a default.
        Anything that has not checked goes through the constructor.
        """
        entity = object.__new__(cls)
        for name, value in fields.items():
            object.__setattr__(entity, name, value)
        return entity

    def with_configuration(self, **changes: object) -> Self:
        """This entity with these configuration members changed.

        `codec.with_configuration(typesize=UNSET)` is the record rebuilt
        through its constructor, which refuses a member of the wrong
        type, and the entity rebuilt through its own, which refuses a
        value the rules disallow -- the same checks as any construction,
        since pyright cannot see the members through `**changes`. A
        name that is not a member is refused the way `replace` refuses
        it.
        """
        return replace(self, configuration=replace(self.configuration, **changes))

    @classmethod
    def coerce(cls, value: JSONValue, context: Context) -> Coerced[Self]:
        """`value` as this entity, or the reasons it is not one: the class's validation routine.

        `resolve` relates a field's name to this class and hands it the
        field, refined JSON with arrays as tuples, which is what `value`
        is; this is what the class does with it. The configuration is
        parsed against the record the `configuration` field names,
        member by member; a member holding another entity is read in
        `context`, the scope this reading is happening in. An optional
        member the document left out is `UNSET` in the record, so no
        field's default decides what a document said. The entity is
        built only when every member of its own read -- its rules are
        written over a whole configuration -- and handed back only when
        everything inside it read too.

        The envelope is the field's, not the class's, and `resolve`
        judges it: a stray member or a `must_understand` of `false` is
        not reported here. Called on a class no scope has registered,
        this runs with none of registration's refusals having happened.
        """
        name, given, envelope = named_configuration(value)
        if name is None or not cls.accepts(name):
            return None, problem((), f"expected the {cls.identifier!r} entity")
        if len(envelope) != 0:
            return None, envelope
        plan = _plan(cls)
        carried: dict[str, str] = {} if plan.from_name is None else {plan.from_name: name}
        if given is None and plan.requires_configuration:
            return None, problem(
                ("configuration",),
                f"{cls.identifier!r} requires a configuration",
                "missing_key",
            )
        reading = _Reading(context, [])
        # A bare name's record has no members, so any key is an unknown one.
        record, own = plan.read({} if given is None else given, ("configuration",), reading)
        found = (*own, *reading.nested)
        if record is None:
            # An unknown key is survivable; a member that could not be
            # read is a hole, and judging around it would be guessing.
            return None, found
        # The rules, asked of the name and of the record before anything
        # is built: a name problem lands on the entity, a configuration
        # problem under the configuration.
        refused = (*cls.name_problems(name), *within((), tuple(record.problems())))
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
        return cls.create_unchecked(configuration=record, **carried), found

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
        in canonical form: `self.with_configuration(inner=self.inner.canonical())`.
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
        back, and those come out as they went in. The envelope is
        written the entity's way -- the bare name when nothing is
        configured, the object otherwise, no `must_understand`, which
        means what absence means -- because an entity alone has no
        document to be faithful to; `ArrayDocumentV3.to_json` puts back
        the spelling the document used. Ask `canonical` first if you
        want the simplest equivalent spelling.

        An entity whose JSON is not its fields overrides this; none in
        the package does.
        """
        plan = _plan(type(self))
        name = self.identifier
        if plan.from_name is not None:
            carried = getattr(self, plan.from_name)
            name = carried if isinstance(carried, str) else name
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

    def inner_pipelines(
        self, incoming: ArrayParts | None
    ) -> Mapping[str, tuple[Sequence[CodecEntity | Opaque], ArrayParts | None]]:
        """The pipelines this codec holds, by the member holding each, with what each is handed.

        A shard holds two: its `codecs`, handed its inner chunk, and its
        `index_codecs`, handed the shard index. Refinement walks them as
        it walks the pipeline this codec stands in, locating what it
        finds under the member, so a codec that holds pipelines says
        which and what they receive, and judges nothing inside them
        itself. Default: none.
        """
        return {}


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

    twos_complement: ClassVar[bool]
    """Whether this type's scalars are two's complement integers.

    Asked by `cast_value`, whose `out_of_range: "wrap"` is defined only
    for such a target. Owed rather than defaulted: a data type added
    later must decide, because either default would answer for it
    silently -- and getting it wrong in one direction accepts a cast the
    spec does not define.
    """

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
    "DataTypeEntity",
    "EntityT",
    "Loc",
    "MetadataEntity",
    "Opaque",
    "StorageClass",
    "StorageTransformerEntity",
    "held_problems",
    "is_from_name",
    "is_integer",
    "is_metadata_field",
    "kind_of",
    "named_configuration",
    "nested_kind",
    "problem",
    "read_field",
    "resolve",
    "unreadable",
    "within",
]
