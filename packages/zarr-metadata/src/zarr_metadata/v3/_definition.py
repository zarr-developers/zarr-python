"""One metadata field, read against its definition.

Three steps, each feeding the next and needing more than the one before:

1. The type check needs the value and a TypedDict. `check` runs the
   checker compiled from the TypedDict's annotations over the JSON:
   its shape, member by member, every problem located, and a value of
   the TypedDict back -- a key it does not declare is reported and left
   out. A member holding another metadata field -- a shard's codecs, a
   cast's data type -- is annotated with a field alias, `CodecField`,
   and checked as an envelope: a name, or a named configuration. Which
   definition the name denotes is a scope's question, so the check
   needs none.
2. The rules need the definition: plain functions over the checked
   TypedDict, for everything finer than a type -- a bound, members read
   together. `Definition.judge` is the check and then the rules, for a
   caller holding one configuration.
3. The reading needs a scope. `resolve` relates the field's name to a
   definition through a `Context`, judges the configuration, and reads
   each nested field the check found, in the same scope.

A definition is a value, not a class to subclass. It holds the name the
metadata carries, the TypedDict that is the one declaration of the
configuration's JSON, and its rules, and it checks itself when it is
built. What kind of metadata it defines is its type -- `CodecDefinition`,
`ChunkGridDefinition` -- which is how a scope files it. Nothing happens
at class creation.
"""

from __future__ import annotations

import dataclasses
import functools
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Generic,
    Literal,
    TypeAlias,
    TypeGuard,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from typing_extensions import TypeAliasType, TypedDict, TypeVar, is_typeddict

from zarr_metadata._common import JSONValue, ZarrV3NamedConfigJSON
from zarr_metadata._json import ValidationProblem, refine_json
from zarr_metadata._typed_json import (
    Loc,
    Parsed,
    Parser,
    no_leaf,
    parser,
    problem,
    typeddict_keys,
    unread_in,
)
from zarr_metadata.v3._common import ZarrV3MetadataFieldJSON, envelope_problems

if TYPE_CHECKING:
    from collections.abc import Sequence

    from zarr_metadata.v3._registry import Context

C = TypeVar("C", default=Mapping[str, JSONValue])
"""A configuration: the TypedDict a definition declares its JSON as.

Any configuration is a `Mapping[str, JSONValue]`, which is what a kind
written without one stands for: `resolve(field, CodecDefinition, scope)`
reads as `CodecDefinition[Mapping[str, JSONValue]]`, nothing unknown.
"""

T = TypeVar("T")

D = TypeVar("D", bound="Definition[Any]")

Problems: TypeAlias = tuple[ValidationProblem, ...]


def no_rules(*_: object) -> Iterator[ValidationProblem]:
    """The rules of a definition with none, of any kind: everything well typed is allowed."""
    yield from ()


def unchanged(configuration: T) -> T:
    """The canonical form of a configuration with no simpler spelling: itself."""
    return configuration


def unknown_lengths(configuration: object, nested: object, shape: tuple[int, ...]) -> Lengths:
    """The chunk lengths of a grid that says nothing of them: unknown, along each axis of the array."""
    return (None,) * len(shape)


def unknown_chunk(*_: object) -> Chunk:
    """What an array -> array codec that says nothing of it hands on: a chunk nothing is known of."""
    return Chunk()


def no_pipelines(*_: object) -> Mapping[str, Chunk]:
    """The pipelines of a codec that holds none: none."""
    return {}


StorageClass = Literal["single_byte", "multi_byte", "variable_length"]
"""How a data type's values are stored: in single bytes, in several bytes at a time, or each in as many as it needs.

A number of several bytes is stored in a byte order, which the `bytes`
codec's `endian` says. A value made of single bytes -- a `uint8`, or a
struct of `int8` fields -- has no byte order, and a value whose size
varies takes a codec of its own.
"""


def single_byte(*_: object) -> StorageClass:
    """The storage of a data type made of single bytes, which no byte order applies to: `uint8`."""
    return "single_byte"


def multi_byte(*_: object) -> StorageClass:
    """The storage of a data type holding numbers of several bytes, which a byte order applies to: `int16`."""
    return "multi_byte"


def variable_length(*_: object) -> StorageClass:
    """The storage of a data type whose values vary in size: `string`."""
    return "variable_length"


def unknown_storage(*_: object) -> None:
    """The storage of a data type that says nothing of it: unknown."""


class EmptyConfiguration(TypedDict, closed=True):
    """The configuration of a definition with nothing to configure: its field is written with its name alone."""


@dataclass(frozen=True, kw_only=True, slots=True)
class Definition(Generic[C]):
    """One extension's metadata, as JSON: its name, the TypedDict its configuration is, its rules.

    `configuration` is the TypedDict, and so the one declaration of the
    JSON: the checker is compiled from it, the static type of a checked
    configuration is it, and a document's author writes to it. It reads
    as the typing spec defines it -- `total`, `Required`, `NotRequired`,
    `closed` and `extra_items` mean what they mean to a type checker --
    and it says what a key it does not declare is: with `closed=True`, a
    problem, reported and left out; with `extra_items=`, a key of that
    type; with `closed=False`, anything. `rules` yields what the spec
    disallows in a configuration of that type, as it finds each; it is
    handed only a configuration that has passed the check, holding what
    the TypedDict admits and nothing else, and the fields it holds as the
    scope read them -- a struct's field types -- which is nothing when no
    scope read it. `judge` is the two, for a caller holding JSON.

    `canonical` is where two spellings of the configuration that mean the
    same thing are made one.

    Built by hand, a definition refuses what it could not read with: a
    `configuration` that is not a TypedDict, says nothing of the keys it
    does not declare, or has a member no checker reads, which is named;
    and a `name` or rules that are not what they say.
    """

    name: str
    """The name the metadata carries, which a scope files the definition under."""
    configuration: type[C]
    """The TypedDict the configuration is."""
    rules: Callable[[C, Nested], Iterable[ValidationProblem]] = no_rules
    """What the spec disallows in a well-typed configuration and the fields it holds, located in it."""
    canonical: Callable[[C], C] = unchanged
    """A well-typed, allowed configuration in its simplest equivalent spelling.

    Only the definition's own members: a nested field is put in its own
    canonical form by `canonicalize`, which knows where each one sits.
    """

    def __post_init__(self) -> None:
        refusal = _malformed(self) or self._refusal()
        if refusal is not None:
            raise TypeError(refusal)
        try:
            _vet(self.configuration)
        except TypeError as error:
            msg = f"{self.name!r}: {error}"
            raise TypeError(msg) from error

    def _refusal(self) -> str | None:
        """What is wrong with the members a kind adds; None when nothing is, or it adds none."""
        return None

    @property
    def requires_configuration(self) -> bool:
        """Whether a document must write a configuration: whether the TypedDict has a required key."""
        return len(typeddict_keys(self.configuration).required) != 0

    def check(self, value: object, loc: Loc = ()) -> tuple[C | None, Problems]:
        """`value` type-checked as this definition's configuration, each nested field's envelope judged.

        `zarr_metadata.typed_json.check` is the type check alone; this also
        judges the envelope of each metadata field a member holds.
        """
        return _configuration_checked(value, self.configuration, loc)

    def judge(self, value: object, loc: Loc = ()) -> tuple[C | None, Problems]:
        """`value` type-checked, then judged by the rules: the configuration if it holds, and every problem.

        The rules are asked only of a configuration that type-checked and
        whose nested fields are well formed, holding what its TypedDict
        admits and nothing else, so a caller holding JSON never reaches a
        rule with a member of the wrong type, or one the type says cannot
        be there. No scope reads the fields it holds, so the rules see
        none of them read, and a rule about one -- a struct's field of a
        type whose values vary in size -- finds nothing to judge: `resolve`
        reads the field in a scope, and asks every rule.
        """
        configuration, problems = self.check(value, loc)
        if configuration is None:
            return None, problems
        refused = ruled(self, lambda: self.rules(configuration, _nothing_nested()), loc)
        return (configuration if len(refused) == 0 else None), (*problems, *refused)


def _malformed(definition: Definition[Any]) -> str | None:
    """What makes a hand-built definition unusable before its configuration is read; None if nothing does."""
    name = cast("object", definition.name)
    if not isinstance(name, str):
        return f"a definition's name is a string, got {name!r}"
    for member in _function_members(type(definition)):
        value = getattr(definition, member)
        if not callable(value):
            return f"{name!r}: {member} is a function, got {value!r}"
    return None


@functools.cache
def _function_members(kind: type[Definition[Any]]) -> tuple[str, ...]:
    """The members `kind` declares as functions: each one its annotation says is a `Callable`."""
    return tuple(
        member
        for member, annotation in get_type_hints(kind).items()
        if get_origin(annotation) is Callable
    )


RAW_BYTES_NAME: Final = "r*"
"""The name raw bits are filed under: `r*`, as the specification's table of data types writes them.

A document writes `r` and the size in bits -- `r8`, `r16` -- so raw bits
are the one data type whose name carries its configuration. `spelled`
reads such a name as `r*` with the size as its configuration; `r*` itself
is the table's notation, never a name a document writes
(https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/data-types/index.rst#L46-L47).
"""

RAW_BYTES_NAME_PATTERN: Final = re.compile(r"r([0-9]+)")
"""A name that writes raw bits: `r` and a size in bits, matched whole, whether or not the size is allowed.

ASCII digits only: `\\d` would also match every other Unicode decimal, so
`r\uff11\uff16` would be read as sixteen bits, and a third-party name
spelled that way would be taken for raw bits. A size the spec does not
allow -- `r0`, `r12` -- is still raw bits, so it is reported as a bad
size rather than passed as an unknown extension.
"""


def spelled(
    kind: type[Definition[Any]], name: str
) -> tuple[str | None, dict[str, JSONValue] | None]:
    """How a name a document writes reads: the name its definition is filed under, and the configuration the name carries.

    A name is filed as itself and carries nothing, but for raw bits: the
    data type `r16` is filed under `r*` and carries `{"bits": 16}`. `r*`
    itself is filed under nothing, `(None, None)`, since no document
    writes it.
    """
    if kind is DataTypeDefinition:
        if name == RAW_BYTES_NAME:
            return None, None
        written = RAW_BYTES_NAME_PATTERN.fullmatch(name)
        if written is not None:
            return RAW_BYTES_NAME, {"bits": int(written.group(1))}
    return name, None


def _carrying_name(
    definition: Definition[Any], configuration: Mapping[str, JSONValue]
) -> str | None:
    """The name that carries `configuration` for `definition`: `r16` for `r*` with `{"bits": 16}`; None when its name carries nothing."""
    if not isinstance(definition, DataTypeDefinition) or definition.name != RAW_BYTES_NAME:
        return None
    return f"r{configuration['bits']}"


@dataclass(frozen=True, kw_only=True, slots=True)
class DataTypeDefinition(Definition[C]):
    """A data type, and the fill value an array of it takes.

    `fill_value` is the JSON shape of a fill value -- `Int8FillValue`, an
    annotation the checker reads as it reads a configuration's members --
    and `fill_value_rules` is what the spec disallows in a fill value of
    that shape: an integer out of range, a hex string of another width.
    The rules are handed the configuration, the fields it holds as the
    scope read them (a struct's field types), and the typed fill value. A
    data type that says nothing of its fill value takes any JSON.

    `storage` says how its values are stored -- in single bytes, in
    several bytes at a time, or each in as many as it needs -- which is
    what the `bytes` codec asks of the data type it is handed: an
    `endian`, for numbers of several bytes. A struct's is its fields', so
    it is handed the fields the configuration holds as the scope read
    them. A data type that says nothing of it leaves it unknown.

    One named as a document writes raw bits of one size -- `r16` -- is
    refused: that name reads as `r*`, so nothing would ever read it with
    this definition.
    """

    fill_value: object = JSONValue
    """The JSON shape of a fill value, as an annotation: `Int8FillValue`."""
    fill_value_rules: Callable[[C, Nested, Any], Iterable[ValidationProblem]] = no_rules
    """What the spec disallows in a fill value of that shape, located in it."""
    storage: Callable[[C, Nested], StorageClass | None] = unknown_storage
    """How its values are stored, given the configuration and the fields it holds; None when unknown."""

    def _refusal(self) -> str | None:
        if RAW_BYTES_NAME_PATTERN.fullmatch(self.name) is not None:
            return (
                f"{self.name!r} is how a document writes raw bits of one size, which read as "
                f"{RAW_BYTES_NAME!r}; to read raw bits your own way, define {RAW_BYTES_NAME!r}"
            )
        try:
            _fill_value_parser(self.fill_value)
        except TypeError as error:
            return f"{self.name!r}: fill_value: {error}"
        return None


@functools.cache
def _fill_value_parser(annotation: object) -> Parser:
    """The checker for a fill value's JSON shape, compiled once; `TypeError` naming what no checker reads."""
    return parser(annotation, no_leaf)


@dataclass(frozen=True, kw_only=True, slots=True)
class ChunkGridDefinition(Definition[C]):
    """A chunk grid, and the arrays it fits.

    `shape_rules` is what the spec disallows in a grid of this
    configuration over an array of a given shape: a dimension with no
    chunk length, chunks that fall short of one. It is handed the
    configuration, the fields it holds as the scope read them, and the
    shape, and locates its problems in the configuration. A grid that
    says nothing of the shape fits every one.

    `chunk_lengths` is what the first codec of the array's pipeline is
    handed: the lengths the grid's chunks take along each axis of an
    array of a shape it fits -- one for each axis of a regular grid, every
    length a rectilinear grid lists. It is asked only of a grid its shape
    rules accept. A grid that says nothing of it leaves the lengths along
    every axis unknown.
    """

    shape_rules: Callable[[C, Nested, tuple[int, ...]], Iterable[ValidationProblem]] = no_rules
    """What the spec disallows in this grid over an array of a shape, located in the configuration."""
    chunk_lengths: Callable[[C, Nested, tuple[int, ...]], Lengths] = unknown_lengths
    """The lengths its chunks take along each axis of an array of a shape it fits, None where unknown."""


@dataclass(frozen=True, kw_only=True, slots=True)
class ChunkKeyEncodingDefinition(Definition[C]):
    """A chunk key encoding."""


CodecKind = Literal["array_array", "array_bytes", "bytes_bytes"]
"""What a codec does to what it is handed: the three positions a pipeline orders."""

CodecSize = Literal["static", "dynamic"]
"""Whether the size of what a codec gives out is fixed by the size of what it is handed.

`static`: it is -- `bytes` writes each element in its width, `crc32c` adds
four bytes. `dynamic`: it depends on the values -- every compressor.
"""

_UNASKED: Final[Mapping[CodecKind, tuple[str, ...]]] = {
    "array_array": (),
    "array_bytes": ("transition",),
    "bytes_bytes": ("chunk_rules", "transition", "pipelines"),
}
"""The functions no codec of a kind is asked: a bytes -> bytes codec is handed bytes, and only an array -> array codec hands on a chunk."""


@dataclass(frozen=True, kw_only=True, slots=True)
class CodecDefinition(Definition[C]):
    """A codec: what it does to what it is handed, and whether the size of what it gives out is static.

    A codec handed an array -- array -> array, array -> bytes -- says what
    the spec disallows in it handed a `Chunk`: `chunk_rules`, handed the
    configuration, the fields it holds as the scope read them, and the
    chunk, and locating its problems in the configuration -- a `bytes`
    codec without `endian`, handed a multi-byte data type. An array ->
    array codec also says what it hands on: `transition`, the chunk the
    next codec is handed, given the one it is handed -- `transpose`
    permutes the axes, `cast_value` changes the data type. The two are the
    spec's pair: a codec computes what it gives from the shape and data
    type it is handed, and "If the decoded_representation_type is not
    supported, this algorithm must fail with an error"
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L987-L994).
    The transition is asked of every chunk the codec is handed, whatever
    its chunk rules found, so it gives only what holds either way: a
    `transpose` whose `order` has another number of axes hands on lengths
    nothing is known of. A codec that says nothing of what it hands on
    hands the next a chunk nothing is known of.

    A codec that holds pipelines of its own says what each is handed:
    `pipelines`, by the member of its configuration that holds each, the
    chunk its first codec is handed, given the chunk the codec is handed
    -- a shard's inner codecs are handed its inner chunks, and its index
    codecs the shard index. Like the transition, it is asked whatever the
    chunk rules found, and gives only what holds either way. A function
    no codec of its kind is asked -- the chunk rules of a bytes -> bytes
    codec, which is handed bytes -- is refused.
    """

    kind: CodecKind
    size: CodecSize
    chunk_rules: Callable[[C, Nested, Chunk], Iterable[ValidationProblem]] = no_rules
    """What the spec disallows in this codec handed a chunk, located in the configuration."""
    transition: Callable[[C, Nested, Chunk], Chunk] = unknown_chunk
    """The chunk the next codec is handed, given the one this array -> array codec is handed."""
    pipelines: Callable[[C, Nested, Chunk], Mapping[str, Chunk]] = no_pipelines
    """The pipelines it holds, by the member of its configuration that holds each, and the chunk each is handed."""

    def _refusal(self) -> str | None:
        kind: object = self.kind
        if kind not in get_args(CodecKind):
            return f"{self.name!r}: kind is one of {get_args(CodecKind)!r}, got {kind!r}"
        size: object = self.size
        if size not in get_args(CodecSize):
            return f"{self.name!r}: size is one of {get_args(CodecSize)!r}, got {size!r}"
        defaults = {member.name: member.default for member in dataclasses.fields(CodecDefinition)}
        for member in _UNASKED[self.kind]:
            if getattr(self, member) is not defaults[member]:
                return f"{self.name!r}: {member}, which no codec of kind {kind!r} is asked"
        return None


@dataclass(frozen=True, kw_only=True, slots=True)
class StorageTransformerDefinition(Definition[C]):
    """A storage transformer."""


KINDS: Final[tuple[type[Definition[Any]], ...]] = (
    DataTypeDefinition,
    ChunkGridDefinition,
    ChunkKeyEncodingDefinition,
    CodecDefinition,
    StorageTransformerDefinition,
)
"""The kinds of metadata a document holds, each filed apart in a scope."""


def kind_of(definition: Definition[Any]) -> type[Definition[Any]] | None:
    """The kind `definition` is; None for a definition of no kind, which no scope files."""
    return next((kind for kind in KINDS if isinstance(definition, kind)), None)


def as_kind(kind: object) -> type[Definition[Any]]:
    """The kind of metadata `kind` names, type arguments dropped; `TypeError` if it names none.

    A scope files definitions by kind, so a field is read as one of
    `KINDS` -- `CodecDefinition`, or `CodecDefinition[Any]` -- and never
    as the base `Definition` or a class of the caller's own, under which
    nothing is filed: a field read as one would go unjudged.
    """
    origin = get_origin(kind) or kind
    found = next((known for known in KINDS if origin is known), None)
    if found is None:
        names = ", ".join(known.__name__ for known in KINDS)
        msg = f"{kind!r} is not a kind of metadata; read a field as one of {names}"
        raise TypeError(msg)
    return found


DataTypeField = TypeAliasType("DataTypeField", ZarrV3MetadataFieldJSON)
"""A configuration member holding a data type, read in the scope the member's field is read in."""
ChunkGridField = TypeAliasType("ChunkGridField", ZarrV3MetadataFieldJSON)
"""A configuration member holding a chunk grid."""
ChunkKeyEncodingField = TypeAliasType("ChunkKeyEncodingField", ZarrV3MetadataFieldJSON)
"""A configuration member holding a chunk key encoding."""
CodecField = TypeAliasType("CodecField", ZarrV3MetadataFieldJSON)
"""A configuration member holding a codec: a shard's `codecs` is `tuple[CodecField, ...]`."""
StaticCodecField = TypeAliasType("StaticCodecField", ZarrV3MetadataFieldJSON)
"""A configuration member holding a codec of static size: a shard's `index_codecs` is one,
since a reader finds the index by a size it knows before reading it.
"""
StorageTransformerField = TypeAliasType("StorageTransformerField", ZarrV3MetadataFieldJSON)
"""A configuration member holding a storage transformer."""

_FIELD_KINDS: Final[Mapping[object, type[Definition[Any]]]] = {
    DataTypeField: DataTypeDefinition,
    ChunkGridField: ChunkGridDefinition,
    ChunkKeyEncodingField: ChunkKeyEncodingDefinition,
    CodecField: CodecDefinition,
    StaticCodecField: CodecDefinition,
    StorageTransformerField: StorageTransformerDefinition,
}

_STATIC_SIZE: Final[frozenset[object]] = frozenset({StaticCodecField})
"""The field aliases whose codec must be of static size."""


@dataclass(frozen=True, slots=True)
class _NestedField:
    """A metadata field the check met inside a configuration: where it sits, its kind, its JSON.

    What the checker hands back where a field alias is, in place of the
    JSON: `_checked` collects each one and puts its JSON back. Carried in
    the value rather than recorded on the side, so a branch of a union
    that did not match leaves none behind.
    """

    loc: Loc
    kind: type[Definition[Any]]
    json: JSONValue
    static: bool
    """Whether the member holding it takes a codec of static size only."""


def _field(annotation: object) -> Parser | None:
    """The checker's one shape of this module's own: a member holding a metadata field.

    Checked as a bare name or an object, and handed on: the envelope is
    judged, and the name related to a definition, by whoever reads the
    field -- `check` without a scope, `resolve` in one.
    """
    try:
        kind = _FIELD_KINDS.get(annotation)
    except TypeError:  # an unhashable annotation is no field alias
        return None
    if kind is None:
        return None
    static = annotation in _STATIC_SIZE

    def parse(value: object, loc: Loc) -> Parsed:
        if not isinstance(value, (str, Mapping)):
            return value, problem(loc, f"expected a metadata field, got {value!r}")
        # Refined JSON, which the checker only knows as `object`.
        return _NestedField(loc, kind, cast("JSONValue", value), static), ()

    return parse


def _vetting(annotation: object) -> Parser | None:
    """`_field`, refusing what a definition's configuration cannot hold and still be read.

    A member typed with the envelope's own TypedDict --
    `ZarrV3MetadataFieldJSON` -- checks as plain JSON: its name would never
    be related to a definition, nor its configuration judged. And a
    TypedDict that says nothing of the keys it does not declare is open,
    so a key it does not declare would never be reported.
    """
    if annotation is ZarrV3NamedConfigJSON:
        msg = (
            "a member typed ZarrV3MetadataFieldJSON checks as plain JSON, and is never read as "
            "a field; annotate it with the field alias of its kind: "
            + ", ".join(cast("TypeAliasType", alias).__name__ for alias in _FIELD_KINDS)
        )
        raise TypeError(msg)
    if (
        isinstance(annotation, type)
        and is_typeddict(annotation)
        and not typeddict_keys(annotation).declared
    ):
        msg = (
            f"{annotation.__name__} says nothing of the keys it does not declare, so it is "
            "open, and such a key would go unreported; declare it closed=True, or "
            "extra_items= for what such a key holds, or closed=False to take any, "
            "on a typing_extensions.TypedDict"
        )
        raise TypeError(msg)
    return _field(annotation)


@functools.cache
def _vet(configuration: type) -> None:
    """Refuse a configuration no definition could read with, saying what is wrong with it."""
    if not is_typeddict(configuration):
        msg = (
            f"configuration is {configuration!r}; give the TypedDict the configuration's JSON takes"
        )
        raise TypeError(msg)
    _vetting(configuration)
    unread = unread_in(configuration, _vetting)
    if unread is not None:
        msg = f"{unread}; a finer rule goes in the definition's `rules`"
        raise TypeError(msg)


@dataclass(frozen=True, slots=True)
class _Checker:
    """A TypedDict's checker, compiled once, and whether a value of it can hold a nested field."""

    parse: Parser
    nests: bool


@functools.cache
def _checker(shape: type) -> _Checker:
    """The checker for a TypedDict, compiled once; `TypeError` naming what no checker reads."""
    met: list[object] = []

    def leaf(annotation: object) -> Parser | None:
        found = _field(annotation)
        if found is not None:
            met.append(annotation)
        return found

    return _Checker(parser(shape, leaf), len(met) != 0)


def _checked(
    shape: type, value: object, loc: Loc
) -> tuple[object, Problems, tuple[_NestedField, ...]]:
    """`value` checked as `shape`: the typed value, every problem, and the fields nested in it, in order."""
    checker = _checker(shape)
    typed, found = checker.parse(value, loc)
    if not checker.nests:
        return typed, found, ()
    nested: list[_NestedField] = []
    return _put_back(typed, nested), found, tuple(nested)


def _put_back(value: object, nested: list[_NestedField]) -> object:
    """`value` with each nested field the checker handed back put back as its JSON, collected in order."""
    if isinstance(value, _NestedField):
        nested.append(value)
        return value.json
    if isinstance(value, tuple):
        return tuple(_put_back(entry, nested) for entry in cast("tuple[object, ...]", value))
    if isinstance(value, dict):
        entries = cast("dict[str, object]", value)
        return {key: _put_back(entry, nested) for key, entry in entries.items()}
    return value


def _usable(problems: Sequence[ValidationProblem]) -> bool:
    """Whether a value with these problems still reads: an unknown key is survivable, nothing else is."""
    return all(found.kind == "unknown_key" for found in problems)


def asked(definition: Definition[Any], what: str, ask: Callable[[], T], at: Loc | None = None) -> T:
    """What `ask`, a call of `definition`'s `what`, gives.

    A definition's functions are the extension author's code: an error one
    raises says which definition's function raised it, and where it was
    reading, when that is known.
    """
    try:
        return ask()
    except Exception as error:
        where = "" if at is None else f", reading {at!r}"
        error.add_note(f"raised by the {what} of {definition.name!r}{where}")
        raise


def ruled(
    definition: Definition[Any], ask: Callable[[], Iterable[ValidationProblem]], at: Loc
) -> Problems:
    """What `ask`, a call of `definition`'s rules, finds, located under `at`.

    Rules are the extension author's code. What they yield is checked to be
    what they declare, and an error one raises says which definition's rules
    raised it, and where they were reading.
    """
    found = asked(definition, "rules", lambda: tuple(cast("Iterable[object]", ask())), at)
    for item in found:
        if not isinstance(item, ValidationProblem):
            msg = f"{definition.name!r}: its rules yield ValidationProblem values, got {item!r}"
            raise TypeError(msg)
    return _located(at, cast("tuple[ValidationProblem, ...]", found))


def _located(prefix: Loc, problems: Iterable[ValidationProblem]) -> Problems:
    return tuple(
        ValidationProblem((*prefix, *found.loc), found.message, found.kind) for found in problems
    )


def _envelope(field: _NestedField) -> Problems:
    """What is wrong with a nested field's envelope, at the field."""
    return _located(field.loc, envelope_problems(field.json, allow_must_understand_false=False))


def _configuration_checked(
    value: object, shape: type[T], loc: Loc = ()
) -> tuple[T | None, Problems]:
    """`value` checked as `shape`, as `typed_json.check` checks it, and each nested field's envelope judged.

    The step a definition's `judge` starts from. A member typed with a
    field alias holds a metadata field, whose envelope is judged as a
    document's is: a stray member, or a `must_understand` of `false`, is a
    problem of the configuration, and the value does not come back.
    """
    refined, problems = refine_json(value, loc)
    if len(problems) != 0:
        return None, problems
    typed, found, nested = _checked(shape, refined, loc)
    problems = (*found, *(problem for field in nested for problem in _envelope(field)))
    return (cast("T", typed) if _usable(problems) else None), problems


def named_configuration(
    value: object,
) -> tuple[str | None, Mapping[str, object] | None, Problems]:
    """Split a metadata field into `(name, configuration, problems)`.

    A bare name, or an object carrying one. A `None` name means the value
    is not a metadata field at all; a `None` configuration means the bare
    spelling was used, or the key was left out. A configuration that is
    present and not an object is the one problem reported, at
    `("configuration",)`.
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


Unread = Literal["out_of_scope", "invalid"]
"""A field no definition read: nothing in scope claims its name, or it could not be read."""

Resolution = Literal["read"] | Unread
"""What a scope made of a field: read by the definition that claims it, or unread, and why."""


def _nothing_nested() -> Nested:
    """What a field that holds no field, or was not read, holds inside: nothing."""
    return {}


@dataclass(frozen=True, slots=True)
class Resolved(Generic[D]):
    """One metadata field, as read in a scope: its JSON, and what the scope made of it.

    `resolution` is what became of the configuration: read by the
    definition that claims the name, claimed by nothing, or not readable.
    A problem with the envelope around it -- a stray member, a
    `must_understand` of `false` -- is reported with the field, and leaves
    the resolution as it is; so is a problem of a field the configuration
    holds, which is that field's own, with its own resolution in `nested`.
    """

    json: JSONValue
    """The field as written, refined: arrays as tuples. `None` for a value that was not JSON, as for `null`."""
    resolution: Resolution
    definition: D | None
    """The definition that claims the field's name; None when nothing in scope does, or it names none."""
    configuration: Mapping[str, JSONValue] | None
    """The configuration, type-checked and allowed by the rules, when the field was read; None otherwise."""
    nested: Nested = dataclasses.field(default_factory=_nothing_nested)
    """The fields the configuration holds, each as the scope read it, by where it sits in the configuration.

    A struct's field types at `("fields", 0, "data_type")`, a shard's
    codecs at `("codecs", 0)`: what a definition's functions consult about
    the fields inside its own. Empty unless the field was read.
    """


Nested: TypeAlias = Mapping[Loc, Resolved[Any]]
"""The fields a configuration holds, each as the scope read it, by where it sits in the configuration."""


Lengths: TypeAlias = tuple[frozenset[int] | None, ...]
"""Per axis, every length chunks take along it -- a set, since a rectilinear grid's differ -- or None where unknown."""


@dataclass(frozen=True, slots=True)
class Chunk:
    """What a codec is handed: chunks of some lengths along each axis, of a data type.

    What nothing says is None: the lengths along an axis the grid does not
    say, and every part of the chunk handed on by a codec that says nothing
    of what it hands on. A data type field the scope did not read is held
    as written, and says nothing of the values either. A codec's chunk
    rules judge what is known and leave the rest, so a chunk nothing is
    known of, `Chunk()`, is refused nothing.
    """

    lengths: Lengths | None = None
    """Per axis, the lengths the chunks take along it; None when not even the number of axes is known."""
    data_type: Resolved[DataTypeDefinition[Any]] | None = None
    """The data type field of the values, as a scope read it; None when no field says what they are."""

    def __post_init__(self) -> None:
        lengths = cast("object", self.lengths)
        if lengths is not None and not _is_lengths(lengths):
            msg = f"a chunk's lengths are a frozenset of integers or None per axis, got {lengths!r}"
            raise TypeError(msg)
        data_type = cast("object", self.data_type)
        if data_type is not None and not _is_data_type_field(data_type):
            msg = f"a chunk's data type is a data type field a scope read, got {data_type!r}"
            raise TypeError(msg)

    @property
    def rank(self) -> int | None:
        """The number of axes; None when unknown."""
        return None if self.lengths is None else len(self.lengths)


def _is_data_type_field(value: object) -> bool:
    """Whether `value` is a data type field a scope read: one read as a data type, or by nothing."""
    if not isinstance(value, Resolved):
        return False
    definition = cast("Resolved[Any]", value).definition
    return definition is None or isinstance(definition, DataTypeDefinition)


def _is_lengths(value: object) -> TypeGuard[Lengths]:
    """Whether `value` is chunk lengths: per axis, a frozenset of integers, or None."""
    if not isinstance(value, tuple):
        return False
    for axis in cast("tuple[object, ...]", value):
        if axis is None:
            continue
        if not isinstance(axis, frozenset) or not all(
            isinstance(length, int) and not isinstance(length, bool)
            for length in cast("frozenset[object]", axis)
        ):
            return False
    return True


def configuration_of(resolved: Resolved[Any], definition: Definition[C]) -> C | None:
    """The configuration `resolved` holds, typed as `definition` declares it, if `definition` read it.

    `Resolved` holds a configuration as the mapping every one is; asked
    with the definition that read the field, this is the same mapping, as
    its TypedDict. None when another definition read it, or none did.
    """
    if resolved.definition is not definition or resolved.configuration is None:
        return None
    return cast("C", resolved.configuration)


def fill_value_problems(
    data_type: Resolved[DataTypeDefinition[Any]], value: object, loc: Loc = ()
) -> Problems:
    """What is wrong with `value` as a fill value of `data_type`, a data type field a scope read.

    `value` is refined to JSON first: not JSON is the first verdict,
    whatever the data type. It is then checked against the JSON shape the
    data type's definition declares, and judged by its fill value rules, as
    `judge` judges a configuration: a key the shape does not declare is
    reported and left out, and the rules still judge the rest. The rules
    see the fields the configuration holds as the scope read them: a
    struct judges each field's fill value by that field's own type. A data
    type the scope did not read, out of scope or invalid, leaves a JSON fill
    value unjudged. `loc` prefixes every problem.
    """
    refined, problems = refine_json(value, loc)
    definition = data_type.definition
    configuration = data_type.configuration
    if len(problems) != 0 or definition is None or configuration is None:
        return problems
    typed, problems = _fill_value_parser(definition.fill_value)(refined, loc)
    if not _usable(problems):
        return problems
    refused = ruled(
        definition,
        lambda: definition.fill_value_rules(configuration, data_type.nested, typed),
        loc,
    )
    return (*problems, *refused)


def storage_of(data_type: Resolved[DataTypeDefinition[Any]]) -> StorageClass | None:
    """How the values of `data_type`, a data type field a scope read, are stored; None when unknown.

    Unknown when the scope did not read it, or its definition does not
    say. Its `storage` is the extension author's code: what it gives is
    checked to be a storage class, and an error it raises says which data
    type's storage raised it.
    """
    definition = data_type.definition
    configuration = data_type.configuration
    if definition is None or configuration is None:
        return None
    found = asked(
        definition,
        "storage",
        lambda: cast("object", definition.storage(configuration, data_type.nested)),
    )
    if found is not None and found not in get_args(StorageClass):
        msg = (
            f"{definition.name!r}: its storage gives one of {get_args(StorageClass)!r} or None, "
            f"got {found!r}"
        )
        raise TypeError(msg)
    return cast("StorageClass | None", found)


def chunk_grid_lengths(
    chunk_grid: Resolved[ChunkGridDefinition[Any]], shape: tuple[int, ...], loc: Loc = ()
) -> tuple[Lengths, Problems]:
    """The lengths the chunks of `chunk_grid`, a chunk grid field a scope read, take along each axis of an array of `shape`, and what is wrong with the grid over it.

    A chunk has an extent "for each dimension of the array"
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L277-L281),
    so the lengths have an entry for each dimension of `shape`, None where
    nothing says them. The grid's shape rules judge it first, and locate
    their problems in the configuration, under `loc`, where the field
    sits: a regular grid with a chunk length for each of two dimensions,
    over an array of three. Only a grid that fits the shape says its
    lengths; a grid the scope did not read, out of scope or invalid, is
    left unjudged and says none. What a grid's `chunk_lengths` gives is
    checked: lengths of another type are a `TypeError`, and of another
    number of axes than `shape` has a `ValueError`, each a fault in the
    definition, not the field.
    """
    unknown: Lengths = (None,) * len(shape)
    definition = chunk_grid.definition
    configuration = chunk_grid.configuration
    if definition is None or configuration is None:
        return unknown, ()
    at = (*loc, "configuration")
    problems = ruled(
        definition, lambda: definition.shape_rules(configuration, chunk_grid.nested, shape), at
    )
    if len(problems) != 0:
        return unknown, problems
    lengths = asked(
        definition,
        "chunk lengths",
        lambda: cast("object", definition.chunk_lengths(configuration, chunk_grid.nested, shape)),
        at,
    )
    if not _is_lengths(lengths):
        msg = (
            f"{definition.name!r}: its chunk_lengths give a frozenset of integers or None "
            f"per axis, got {lengths!r}"
        )
        raise TypeError(msg)
    if len(lengths) != len(shape):
        msg = (
            f"{definition.name!r}: its chunk_lengths gave {len(lengths)} axes, "
            f"for a shape of {len(shape)}"
        )
        raise ValueError(msg)
    return lengths, ()


def resolve(
    data: object, kind: type[D], context: Context, loc: Loc = ()
) -> tuple[Resolved[D], Problems]:
    """`data`, one metadata field, read as a `kind` in `context`: what the scope made of it, and every problem.

    All three steps for one field. `data` is refined to JSON and its
    envelope judged -- an extra member, a `configuration` that is not an
    object, a `must_understand` that is not a boolean or is `false`, each
    a problem. The name is related to a definition in `context`; the
    configuration is checked against its TypedDict and judged by its
    rules; each nested field the check met is read the same way, in the
    same scope, and what is wrong with one is its own, reported where it
    sits, as with a document's fields. A name nothing claims is
    `out_of_scope`: an unmodelled
    extension, left unjudged, which is what keeps the format open. `loc`
    prefixes every problem. `kind` is one of `KINDS`, with or without
    type arguments; anything else is a `TypeError`.
    """
    asked = as_kind(kind)
    refined, problems = refine_json(data, loc)
    if len(problems) != 0:
        return Resolved(None, "invalid", None, None), problems
    resolved, found = _resolve_field(refined, asked, context, loc)
    return cast("Resolved[D]", resolved), found


def _resolve_field(
    data: JSONValue, kind: type[Definition[Any]], context: Context, loc: Loc
) -> tuple[Resolved[Definition[Any]], Problems]:
    """A refined field with its envelope judged, then read.

    The resolution is what became of the configuration. A stray member or
    a `must_understand` of `false` says nothing about it, so it is reported
    beside the field that was read, which later layers can still judge.
    """
    envelope = _located(loc, envelope_problems(data, allow_must_understand_false=False))
    resolved, found = _read(data, kind, context, loc)
    return resolved, (*envelope, *found)


def _read(
    data: JSONValue, kind: type[Definition[Any]], context: Context, loc: Loc
) -> tuple[Resolved[Definition[Any]], Problems]:
    name, given, malformed = named_configuration(data)
    if name is None:
        return Resolved(data, "invalid", None, None), ()
    definition = context.claimant(kind, name)
    if len(malformed) != 0:
        # A configuration that is not an object, which the envelope's
        # problems say; the name still says what claims the field.
        return Resolved(data, "invalid", definition, None), ()
    if definition is None:
        return Resolved(data, "out_of_scope", None, None), ()
    _, carried = spelled(kind, name)
    if carried is not None:
        return _read_carried(data, definition, given, carried, loc)
    at = (*loc, "configuration")
    if given is None and definition.requires_configuration:
        missing = problem(at, f"{name!r} requires a configuration", "missing_key")
        return Resolved(data, "invalid", definition, None), missing
    typed, found, nested = _checked(definition.configuration, {} if given is None else given, at)
    # The rules may read a field the configuration holds by its name, so
    # they are asked only when each one is named; any other problem with
    # one is its own, reported where it sits, as a document's fields are.
    sound = _usable(found) and all(_named(field) for field in nested)
    configuration = cast("Mapping[str, JSONValue]", typed) if sound else None
    # The fields it holds are read first, so the rules see them as the
    # scope read them; their problems are reported after the rules'.
    within: dict[Loc, Resolved[Any]] = {}
    inside: list[ValidationProblem] = []
    for field in nested:
        inside.extend(_envelope(field))
        inner, found_inside = _read(field.json, field.kind, context, field.loc)
        within[field.loc[len(at) :]] = inner
        inside.extend(found_inside)
        inside.extend(_sized(field, inner.definition))
    own = list(found)
    if configuration is not None:
        own.extend(ruled(definition, lambda: definition.rules(configuration, within), at))
    if configuration is None or not _usable(own):
        return Resolved(data, "invalid", definition, None), (*own, *inside)
    return Resolved(data, "read", definition, configuration, within), (*own, *inside)


def _named(field: _NestedField) -> bool:
    """Whether a field a configuration holds is named, with an object for its configuration if it has one."""
    name, _, malformed = named_configuration(field.json)
    return name is not None and len(malformed) == 0


def _read_carried(
    data: JSONValue,
    definition: Definition[Any],
    given: Mapping[str, object] | None,
    carried: Mapping[str, JSONValue],
    loc: Loc,
) -> tuple[Resolved[Definition[Any]], Problems]:
    """A field whose name carries its configuration -- raw bits, `r16` -- read by the definition its name is filed under.

    The document wrote a name, so what is wrong with what the name carries
    -- a size that is not a positive multiple of 8 -- is a problem of the
    field. A configuration written beside the name holds nothing, so each
    member of one is a key nothing declares.
    """
    _, beside, _ = _checked(
        EmptyConfiguration, {} if given is None else given, (*loc, "configuration")
    )
    configuration, judged = definition.judge(carried)
    problems = (*beside, *(ValidationProblem(loc, found.message, found.kind) for found in judged))
    if configuration is None or not _usable(problems):
        return Resolved(data, "invalid", definition, None), problems
    return Resolved(data, "read", definition, configuration), problems


def _sized(field: _NestedField, definition: Definition[Any] | None) -> Problems:
    """A codec of dynamic size in a member that takes codecs of static size, as a problem at the field.

    A name nothing in scope claims is left unjudged, its size unknown, as
    everything else about it is.
    """
    if not field.static or not isinstance(definition, CodecDefinition):
        return ()
    if definition.size == "static":
        return ()
    name, _, _ = named_configuration(field.json)
    return problem(
        field.loc,
        f"{name!r} is a codec of dynamic size, and only codecs of static size may be used here",
        "invalid_value",
    )


def canonicalize(
    data: object, kind: type[D], context: Context, loc: Loc = ()
) -> tuple[JSONValue | None, Problems]:
    """`data`, one metadata field, in its simplest equivalent spelling, and every problem.

    Only a field without problems has one. A simplest spelling says what
    the author wrote in fewer words, and a key the TypedDict does not
    declare, a stray envelope member or a `must_understand` of `false` is
    something the author wrote that it would erase, so a field with any
    problem comes back None, with its problems. Otherwise each nested
    field goes in its own simplest spelling, and then the definition's
    `canonical` has the rest -- judged again, so a `canonical` that gives
    a configuration that does not hold is a `ValueError`, a fault in the
    definition rather than the field. The envelope takes the fewest
    words every reader takes: a data type with nothing to configure is its
    bare name, as core data types have been written since Zarr v3.0; any
    other field is an object, `{"name": ...}`, since a Zarr v3.0 reader
    takes no short-hand name in `codecs`
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L585-L592);
    and there is no `must_understand`, since `true` is what absence means.
    A name nothing in scope claims comes back as written, since what it
    simplifies to is its own definition's call.
    """
    resolved, problems = resolve(data, kind, context, loc)
    if len(problems) != 0:
        return None, problems
    if resolved.definition is None:
        return resolved.json, ()
    return _canonical_field(resolved.definition, resolved), ()


def _canonical_field(definition: Definition[Any], resolved: Resolved[Any]) -> JSONValue:
    """A field that read, in its simplest equivalent spelling: the fields it holds first, then its own members.

    Each field it holds is spelled from what `resolve` read of it, kept in
    `nested`; one nothing in scope claims keeps the spelling it was written
    in.
    """
    name, _, _ = named_configuration(resolved.json)
    configuration: JSONValue = dict(resolved.configuration or {})
    for loc, inner in resolved.nested.items():
        simplest = (
            inner.json if inner.definition is None else _canonical_field(inner.definition, inner)
        )
        configuration = _replaced(configuration, loc, simplest)
    simplified = cast("Mapping[str, JSONValue]", definition.canonical(configuration))
    _, refused = definition.judge(simplified)
    if len(refused) != 0:
        msg = (
            f"{definition.name!r}: its canonical gave {simplified!r}, which does not hold: "
            f"{list(refused)!r}"
        )
        raise ValueError(msg)
    carrying = _carrying_name(definition, simplified)
    if carrying is not None:
        return carrying
    if len(simplified) != 0:
        return {"name": name, "configuration": simplified}
    return name if isinstance(definition, DataTypeDefinition) else {"name": name}


def _replaced(value: JSONValue, path: Loc, new: JSONValue) -> JSONValue:
    """`value` with what sits at `path` replaced by `new`; `path` comes from a check of `value`."""
    if len(path) == 0:
        return new
    step, rest = path[0], path[1:]
    if isinstance(step, str):
        members = cast("Mapping[str, JSONValue]", value)
        return {**members, step: _replaced(members[step], rest, new)}
    entries = cast("tuple[JSONValue, ...]", value)
    return (*entries[:step], _replaced(entries[step], rest, new), *entries[step + 1 :])


__all__ = [
    "KINDS",
    "RAW_BYTES_NAME",
    "RAW_BYTES_NAME_PATTERN",
    "Chunk",
    "ChunkGridDefinition",
    "ChunkGridField",
    "ChunkKeyEncodingDefinition",
    "ChunkKeyEncodingField",
    "CodecDefinition",
    "CodecField",
    "CodecKind",
    "CodecSize",
    "DataTypeDefinition",
    "DataTypeField",
    "Definition",
    "EmptyConfiguration",
    "Lengths",
    "Nested",
    "Resolution",
    "Resolved",
    "StaticCodecField",
    "StorageClass",
    "StorageTransformerDefinition",
    "StorageTransformerField",
    "Unread",
    "as_kind",
    "asked",
    "canonicalize",
    "chunk_grid_lengths",
    "configuration_of",
    "fill_value_problems",
    "kind_of",
    "multi_byte",
    "named_configuration",
    "no_pipelines",
    "no_rules",
    "resolve",
    "ruled",
    "single_byte",
    "spelled",
    "storage_of",
    "unchanged",
    "unknown_chunk",
    "unknown_lengths",
    "unknown_storage",
    "variable_length",
]
