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
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
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
from zarr_metadata._json import (
    ValidationProblem,
    copied,
    is_object,
    is_tuple,
    json_text,
    refine_json,
    shown,
    with_input,
)
from zarr_metadata._sentinel import UNSET
from zarr_metadata._typed_json import (
    JSONSchema,
    Loc,
    Parsed,
    Parser,
    SchemaLeaf,
    Schemas,
    parser,
    problem,
    typeddict_keys,
    unread_in,
)
from zarr_metadata.v3._common import (
    ENVELOPE_KEYS,
    EXTENSION_NAME_SCHEMA_PATTERN,
    ChunkGridField,
    ChunkKeyEncodingField,
    CodecField,
    DataTypeField,
    StaticCodecField,
    StorageTransformerField,
    envelope_problems,
    name_problem,
    well_named,
)

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
F = TypeVar("F", bound="WithFillValue[Any]")

Problems: TypeAlias = tuple[ValidationProblem, ...]


def no_rules(*_: object) -> Iterator[ValidationProblem]:
    """The rules of a definition with none, of any kind: everything well typed is allowed."""
    yield from ()


def unchanged(configuration: T) -> T:
    """The canonical form of a configuration with no simpler spelling: itself."""
    return configuration


def fill_value_as_written(configuration: object, nested: object, value: T) -> T:
    """The canonical spelling of a fill value of a data type that spells each of its values one way: the fill value as written."""
    return value


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


_FIELD_KINDS: Final[dict[object, type[Definition[Any]]]] = {}
"""Each field alias, and the kind a member annotated with it is read as; filed by each kind as its class is built."""


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
    the TypedDict admits and nothing else, each member within the bounds
    its type carries, and the fields it holds as the scope read them -- a
    struct's field types -- which is nothing when no scope read it.
    `judge` is the two, for a caller holding JSON.

    Each function is handed the configuration as a read-only view, no
    `dict`: `copy.deepcopy` and `json.dumps` refuse it, and a function that
    folds a spelling builds a new mapping, `{**configuration}` without the
    member, rather than editing what it was handed.

    `canonical` is where two spellings of the configuration that mean the
    same thing are made one.

    Built by hand, a definition refuses what it could not read with: a
    `configuration` that is not a TypedDict, says nothing of the keys it
    does not declare, or has a member no checker reads, which is named;
    and a `name` or rules that are not what they say.
    """

    is_kind: ClassVar[bool] = False
    """Whether this class is a kind: what a scope files definitions by.

    Set in a kind's own body and read from it, never inherited: a
    subclass of a kind is a definition of that kind.
    """
    label: ClassVar[str] = "definition"
    """The kind as a message names it: "codec"."""
    field_aliases: ClassVar[tuple[object, ...]] = ()
    """The field aliases a configuration member holding a field of this kind is annotated with: `CodecField` and `StaticCodecField` for a codec."""

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

    @classmethod
    def name_problem(cls, name: str, at: Loc) -> ValidationProblem | None:
        """The problem `name`, at `at`, is when no document of the format writes it for a field of this kind; None when one may: for Zarr v3, when the spec gives an extension such a name."""
        return name_problem(name, at)

    @classmethod
    def well_named(cls, name: str) -> bool:
        """Whether a document of the format may write `name` for a field of this kind, as `name_problem` says."""
        return cls.name_problem(name, ()) is None

    @classmethod
    def spelled(cls, name: str) -> tuple[str | None, dict[str, JSONValue] | None]:
        """How a name a document writes reads: the name its definition is filed under, and the configuration the name carries.

        A name is filed as itself and carries nothing, `(name, None)`. A
        kind whose names carry configuration says otherwise: a v3 data
        type `r16` is filed under `r*` with `{"bits": 16}`. A name no
        document writes, which only files a definition, is `(None, None)`.
        """
        return name, None

    def carrying_name(self, configuration: Mapping[str, JSONValue]) -> str | None:
        """The name that carries `configuration` for this definition, the inverse of `spelled`: `r16` for `r*` with `{"bits": 16}`; None when its names carry nothing."""
        return None

    @classmethod
    def named_configuration(
        cls, value: object
    ) -> tuple[str | None, Mapping[str, object] | None, Problems]:
        """`value`, a field as a document of the format writes it, split into `(name, configuration, problems)`, as the module's `named_configuration` splits a v3 field."""
        return named_configuration(value)

    @classmethod
    def envelope_problems(cls, value: object) -> Problems:
        """Every reason `value` is not a field's envelope as the format writes one for this kind, what the configuration holds left unjudged."""
        return envelope_problems(value, allow_must_understand_false=False)

    @classmethod
    def envelope_json(cls, name: str, configuration: Mapping[str, JSONValue]) -> JSONValue:
        """A field of `name` and `configuration` as a document of the format writes it, in the fewest words every reader takes: for v3, an object."""
        if len(configuration) != 0:
            return {"name": name, "configuration": configuration}
        return {"name": name}

    @classmethod
    def configuration_loc(cls, loc: Loc) -> Loc:
        """Where the configuration of a field at `loc` sits: under `configuration` for v3; at the field for a format that writes the parameters beside the name."""
        return (*loc, "configuration")

    @classmethod
    def name_loc(cls, loc: Loc) -> Loc:
        """Where the name of a field at `loc`, written as an object, sits: under `name` for v3."""
        return (*loc, "name")

    def __init_subclass__(cls, **kwargs: object) -> None:
        # Named, not `super()`: a dataclass with slots is rebuilt, and the
        # cell a bare `super()` reads names the class that was thrown away.
        super(Definition, cls).__init_subclass__(**kwargs)
        # A dataclass with slots is built twice, and the class built last
        # is the one a document is read with: it files its aliases last.
        for alias in cls.__dict__.get("field_aliases", ()):
            _FIELD_KINDS[alias] = cls

    def __post_init__(self) -> None:
        refusal = _malformed(self) or self._refusal()
        if refusal is not None:
            raise TypeError(refusal)
        try:
            _vet(self.configuration)
        except TypeError as error:
            msg = f"{self.name!r}: {error}"
            raise TypeError(msg) from error

    def __repr__(self) -> str:
        # Short, as a reading that holds definitions shows them: in full, a
        # definition's repr is each function it holds, at its address.
        return f"{type(self).__name__}(name={self.name!r})"

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
        configuration, problems = _configuration_checked(value, self.configuration, loc)
        return configuration, with_input(problems, value, loc)

    def judge(self, value: object, loc: Loc = ()) -> tuple[C | None, Problems]:
        """`value` type-checked, then judged by the rules: the configuration if it holds, and every problem.

        The rules are asked only of a configuration that type-checked,
        its bounds kept, and whose nested fields are well formed, holding
        what its TypedDict admits and nothing else, so a caller holding
        JSON never reaches a rule with a member of the wrong type, out of
        its bounds, or one the type says cannot be there. No scope reads the fields it holds, so the rules see
        none of them read, and a rule about one -- a struct's field of a
        type whose values vary in size -- finds nothing to judge: `resolve`
        reads the field in a scope, and asks every rule.
        """
        configuration, problems = self.check(value, loc)
        if configuration is None:
            return None, problems
        refused = ruled(self, lambda: self.rules(read_only(configuration), _nothing_nested()), loc)
        return (configuration if len(refused) == 0 else None), (
            *problems,
            *with_input(refused, value, loc),
        )


def _malformed(definition: Definition[Any]) -> str | None:
    """What makes a hand-built definition unusable before its configuration is read; None if nothing does."""
    name = cast("object", definition.name)
    if not isinstance(name, str):
        return f"a definition's name is a string, got {name!r}"
    for member in _function_members(type(definition)):
        value = getattr(definition, member)
        if not callable(value):
            return f"{name!r}: {member} is a function, got {value!r}"
    kind = type(definition)
    filed, _ = kind.spelled(name)
    bad = None if filed is None else kind.name_problem(name, ())
    if bad is not None:
        return (
            f"{name!r}: {bad.message}, so no document names it, and nothing would ever read "
            "with this definition"
        )
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

RAW_BYTES_NAME_PATTERN: Final = re.compile(r"r([0-9]{1,100})")
"""A name that writes raw bits: `r` and a size in bits of up to a hundred digits, matched whole, whether or not the size is allowed.

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
    return kind.spelled(name)


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class WithFillValue(Definition[C]):
    """A definition of a type whose arrays take a fill value: the JSON shape of one, what the rules disallow in one, and its canonical spelling.

    Not a kind: each format's data type kind derives from it, so
    `fill_value_problems` judges a fill value of either.

    `fill_value` is the JSON shape of a fill value, as an annotation the
    checker reads as it reads a configuration's members, its range among
    it. `fill_value_rules` is what the spec disallows in a fill value of
    that shape that the type cannot say; it is handed the configuration,
    the fields it holds as the scope read them, and the typed fill value.
    `fill_value_canonical` spells a fill value that has no problem in the
    one spelling its value has, so two fill values are one value of the
    type exactly when their canonical spellings are written alike.
    """

    fill_value: object = JSONValue
    """The JSON shape of a fill value, as an annotation: `Int8FillValue`."""
    fill_value_rules: Callable[[C, Nested, Any], Iterable[ValidationProblem]] = no_rules
    """What the spec disallows in a fill value of that shape, located in it."""
    fill_value_canonical: Callable[[C, Nested, Any], JSONValue] = fill_value_as_written
    """A fill value that has no problem, in the one spelling its value has."""

    def _refusal(self) -> str | None:
        try:
            _fill_value_parser(self.fill_value)
        except TypeError as error:
            return f"{self.name!r}: fill_value: {error}"
        return None


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class DataTypeDefinition(WithFillValue[C]):
    """A data type, and the fill value an array of it takes.

    `fill_value` is the JSON shape of a fill value -- `Int8FillValue`, an
    annotation the checker reads as it reads a configuration's members,
    its range among it -- and `fill_value_rules` is what the spec
    disallows in a fill value of that shape that the type cannot say: a
    hex string of another width.
    The rules are handed the configuration, the fields it holds as the
    scope read them (a struct's field types), and the typed fill value. A
    data type that says nothing of its fill value takes any JSON.

    `fill_value_canonical` spells a fill value that has no problem --
    well typed, and allowed by the rules -- in the one spelling its value
    has, so two fill values are one value of the type exactly when their
    canonical spellings are written alike: `"NaN"` and `"0x7fc00000"` are
    one `float32`, and `0.0` and `-0.0` two. It is handed what the rules
    are handed. A data type that says nothing of it spells each of its
    values one way: as written.

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

    is_kind: ClassVar[bool] = True
    label: ClassVar[str] = "data type"
    field_aliases: ClassVar[tuple[object, ...]] = (DataTypeField,)

    @classmethod
    def spelled(cls, name: str) -> tuple[str | None, dict[str, JSONValue] | None]:
        if name == RAW_BYTES_NAME:
            return None, None
        written = RAW_BYTES_NAME_PATTERN.fullmatch(name)
        if written is not None:
            return RAW_BYTES_NAME, {"bits": int(written.group(1))}
        return name, None

    def carrying_name(self, configuration: Mapping[str, JSONValue]) -> str | None:
        if self.name != RAW_BYTES_NAME:
            return None
        return f"r{configuration['bits']}"

    @classmethod
    def envelope_json(cls, name: str, configuration: Mapping[str, JSONValue]) -> JSONValue:
        # A data type with nothing to configure is its bare name, as core
        # data types have been written since Zarr v3.0.
        if len(configuration) != 0:
            return {"name": name, "configuration": configuration}
        return name

    storage: Callable[[C, Nested], StorageClass | None] = unknown_storage
    """How its values are stored, given the configuration and the fields it holds; None when unknown."""

    def _refusal(self) -> str | None:
        if RAW_BYTES_NAME_PATTERN.fullmatch(self.name) is not None:
            return (
                f"{self.name!r} is how a document writes raw bits of one size, which read as "
                f"{RAW_BYTES_NAME!r}; to read raw bits your own way, define {RAW_BYTES_NAME!r}"
            )
        # Named, not a bare `super()`: a dataclass with slots is rebuilt.
        return super(DataTypeDefinition, self)._refusal()


@functools.cache
def _fill_value_parser(annotation: object) -> Parser:
    """The checker for a fill value's JSON shape, compiled once; `TypeError` naming what no checker reads, or a metadata field in it."""
    return parser(annotation, _no_field)


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
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

    is_kind: ClassVar[bool] = True
    label: ClassVar[str] = "chunk grid"
    field_aliases: ClassVar[tuple[object, ...]] = (ChunkGridField,)

    shape_rules: Callable[[C, Nested, tuple[int, ...]], Iterable[ValidationProblem]] = no_rules
    """What the spec disallows in this grid over an array of a shape, located in the configuration."""
    chunk_lengths: Callable[[C, Nested, tuple[int, ...]], Lengths] = unknown_lengths
    """The lengths its chunks take along each axis of an array of a shape it fits, None where unknown."""


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class ChunkKeyEncodingDefinition(Definition[C]):
    """A chunk key encoding."""

    is_kind: ClassVar[bool] = True
    label: ClassVar[str] = "chunk key encoding"
    field_aliases: ClassVar[tuple[object, ...]] = (ChunkKeyEncodingField,)


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


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
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

    is_kind: ClassVar[bool] = True
    label: ClassVar[str] = "codec"
    field_aliases: ClassVar[tuple[object, ...]] = (CodecField, StaticCodecField)

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


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class StorageTransformerDefinition(Definition[C]):
    """A storage transformer."""

    is_kind: ClassVar[bool] = True
    label: ClassVar[str] = "storage transformer"
    field_aliases: ClassVar[tuple[object, ...]] = (StorageTransformerField,)


KINDS: Final[tuple[type[Definition[Any]], ...]] = (
    DataTypeDefinition,
    ChunkGridDefinition,
    ChunkKeyEncodingDefinition,
    CodecDefinition,
    StorageTransformerDefinition,
)
"""The kinds of metadata a document holds, each filed apart in a scope."""


def kind_of(definition: Definition[Any]) -> type[Definition[Any]] | None:
    """The kind `definition` is: the nearest class in its MRO that declares `is_kind`; None for a definition of no kind, which no scope files."""
    return _kind_in(type(definition))


def _kind_in(cls: type) -> type[Definition[Any]] | None:
    return next(
        (
            cast("type[Definition[Any]]", base)
            for base in cls.__mro__
            if vars(base).get("is_kind") is True
        ),
        None,
    )


def as_kind(kind: object) -> type[Definition[Any]]:
    """The kind of metadata `kind` names, type arguments dropped; `TypeError` if it names none.

    A scope files definitions by kind, so a field is read as a kind --
    `CodecDefinition`, or `CodecDefinition[Any]` -- and never as the base
    `Definition` or a class that declares no kind, under which nothing is
    filed: a field read as one would go unjudged.
    """
    origin = get_origin(kind) or kind
    if isinstance(origin, type) and vars(origin).get("is_kind") is True:
        return cast("type[Definition[Any]]", origin)
    names = ", ".join(known.__name__ for known in KINDS)
    msg = (
        f"{kind!r} is not a kind of metadata; read a field as one of {names}, or as a "
        "subclass of Definition that sets is_kind in its own body"
    )
    raise TypeError(msg)


_STATIC_SIZE: Final[frozenset[object]] = frozenset({StaticCodecField})
"""The field aliases whose codec must be of static size."""


def field_kind(annotation: object) -> type[Definition[Any]] | None:
    """The kind of metadata field a member annotated `annotation` holds -- `CodecDefinition` for `CodecField` -- or None when it holds none."""
    try:
        return _FIELD_KINDS.get(annotation)
    except TypeError:  # an unhashable annotation is no field alias
        return None


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
    kind = field_kind(annotation)
    if kind is None:
        return None
    static = annotation in _STATIC_SIZE

    def parse(value: object, loc: Loc) -> Parsed:
        # Refined JSON, which the checker only knows as `object`; what is
        # not a field at all, the kind's envelope says.
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


def _no_field(annotation: object) -> Parser | None:
    """A leaf refusing a field alias: a fill value is a value of its data type, and holds no metadata field."""
    if field_kind(annotation) is not None:
        name = cast("TypeAliasType", annotation).__name__
        msg = f"{name} holds a metadata field, and a fill value is a value of its data type"
        raise TypeError(msg)
    return None


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


def kind_field(kind: type[Definition[Any]]) -> object:
    """The field alias a member holding any field of `kind` is annotated with: the first the kind declares that does not narrow the field, `CodecField` rather than `StaticCodecField`."""
    return next(alias for alias in kind.field_aliases if alias not in _STATIC_SIZE)


_RAW_BYTES_SCHEMA_PATTERN: Final = f"^{RAW_BYTES_NAME_PATTERN.pattern}(?![\\s\\S])"
"""`RAW_BYTES_NAME_PATTERN`, matched whole, as a JSON Schema writes a pattern.

Held to the end of the name by a lookahead for no character at all: a
`$` there would also match before a final newline in a validator that
matches patterns as Python does, so `"r16\\n"`, which names nothing,
would read as raw bits.
"""


def field_json_schema(kind: type[Definition[Any]], context: Context) -> JSONSchema:
    """The JSON Schema of one metadata field read as `kind` in `context`: what `resolve` reads, but for the rules.

    A field one of the definitions in scope reads -- its name, its
    configuration as the TypedDict says, a `must_understand` of `true`
    if any, and its bare name when it needs no configuration -- or a
    name none of them claims, with any configuration: what keeps the
    format open. A field a configuration holds is written the same way,
    in the same scope, and a member taking codecs of static size only
    takes those. JSON Schema draft 2020-12, as `json_schema` writes one;
    the fields it holds, and the configuration of each definition, are in
    `$defs`, under the name of the field alias or TypedDict. What only a
    rule says -- a blosc `typesize` against its `shuffle` -- is not in it,
    so a field it accepts may still have a problem.
    """
    asked = as_kind(kind)
    if not any(issubclass(asked, known) for known in KINDS):
        msg = (
            f"{asked.__name__} is a kind of another format; the JSON Schema writer writes "
            "Zarr v3 fields only"
        )
        raise TypeError(msg)
    schemas = Schemas(field_schemas(context))
    return schemas.document(schemas.of(kind_field(asked)))


def field_schemas(context: Context) -> SchemaLeaf:
    """The schema leaf that writes each field alias as a field of its kind, as `context` reads one: `field_json_schema`'s."""

    def leaf(annotation: object, schemas: Schemas) -> JSONSchema | None:
        kind = field_kind(annotation)
        if kind is None:
            return None
        alias = cast("TypeAliasType", annotation)
        static = annotation in _STATIC_SIZE
        return schemas.defined(
            alias, alias.__name__, lambda: _field_schema(kind, static, context, schemas)
        )

    return leaf


def _field_schema(
    kind: type[Definition[Any]], static: bool, context: Context, schemas: Schemas
) -> JSONSchema:
    """A field of `kind` as `context` reads it: one a definition in scope reads, or one none of them claims."""
    table = context.tables.get(kind, {})
    branches: list[JSONValue] = []
    for definition in table.values():
        if static and cast("CodecDefinition[Any]", definition).size != "static":
            continue
        branches.extend(_read_by(definition, schemas))
    branches.extend(_unclaimed(table))
    return {"anyOf": branches}


def written_name(definition: Definition[Any]) -> JSONSchema:
    """The JSON Schema of each name a document writes for `definition`: its name, or `r` and a size for raw bits."""
    if isinstance(definition, DataTypeDefinition) and definition.name == RAW_BYTES_NAME:
        return {"type": "string", "pattern": _RAW_BYTES_SCHEMA_PATTERN}
    return {"const": definition.name}


def _read_by(definition: Definition[Any], schemas: Schemas) -> list[JSONValue]:
    """The fields `definition` reads: an object of its name and configuration, and its bare name when it needs no configuration.

    Raw bits' name carries their configuration, so what is written beside
    it holds nothing.
    """
    carried = isinstance(definition, DataTypeDefinition) and definition.name == RAW_BYTES_NAME
    name = written_name(definition)
    bare = carried or not definition.requires_configuration
    envelope: JSONSchema = {
        "type": "object",
        "properties": {
            "name": name,
            "configuration": schemas.of(
                EmptyConfiguration if carried else definition.configuration
            ),
            "must_understand": {"const": True},
        },
        "required": ["name"] if bare else ["name", "configuration"],
        "additionalProperties": False,
    }
    return [name, envelope] if bare else [envelope]


def _unclaimed(table: Mapping[str, Definition[Any]]) -> list[JSONValue]:
    """The fields no definition in `table` claims: a name none of them is written with, bare or with any configuration."""
    claimed: list[JSONValue] = [written_name(definition) for definition in table.values()]
    name: JSONSchema = {"type": "string", "pattern": EXTENSION_NAME_SCHEMA_PATTERN}
    if len(claimed) != 0:
        name["not"] = {"anyOf": claimed}
    envelope: JSONSchema = {
        "type": "object",
        "properties": {
            "name": name,
            "configuration": {"type": "object"},
            "must_understand": {"const": True},
        },
        "required": ["name"],
        "additionalProperties": False,
    }
    return [name, envelope]


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
    """`value` checked as `shape`: the typed value, every problem, and the fields nested in it, in order, each put back as it was written."""
    typed, found, nested = _typed(shape, value, loc)
    return _put_back(typed, _as_written), found, nested


def _typed(
    shape: type, value: object, loc: Loc
) -> tuple[object, Problems, tuple[_NestedField, ...]]:
    """`value` checked as `shape`, each nested field still where the checker met it: the typed value, every problem, and those fields, in order."""
    checker = _checker(shape)
    typed, found = checker.parse(value, loc)
    if not checker.nests:
        return typed, found, ()
    nested: list[_NestedField] = []
    _collect(typed, nested)
    return typed, found, tuple(nested)


def _collect(value: object, nested: list[_NestedField]) -> None:
    """Each nested field the checker handed back in `value`, in order."""
    if isinstance(value, _NestedField):
        nested.append(value)
    elif is_tuple(value):
        for entry in value:
            _collect(entry, nested)
    elif isinstance(value, dict):
        for entry in cast("dict[str, object]", value).values():
            _collect(entry, nested)


def _as_written(field: _NestedField) -> JSONValue:
    """A nested field put back as it was written."""
    return field.json


def _declared(field: _NestedField) -> JSONValue:
    """A nested field put back as it was written, but for the members its envelope does not declare, which are reported and left out, as the checker leaves out a key a closed TypedDict does not declare."""
    if not isinstance(field.json, Mapping):
        return field.json
    envelope = cast("Mapping[str, JSONValue]", field.json)
    return {key: value for key, value in envelope.items() if key in ENVELOPE_KEYS}


def _put_back(value: object, put: Callable[[_NestedField], JSONValue]) -> object:
    """`value` with each nested field the checker handed back put back as `put` gives it."""
    if isinstance(value, _NestedField):
        return put(value)
    if is_tuple(value):
        return tuple(_put_back(entry, put) for entry in value)
    if isinstance(value, dict):
        entries = cast("dict[str, object]", value)
        return {key: _put_back(entry, put) for key, entry in entries.items()}
    return value


def _usable(problems: Sequence[ValidationProblem]) -> bool:
    """Whether a value with these problems still reads: an unknown key is survivable, nothing else is."""
    return all(found.kind == "unknown_key" for found in problems)


def read_only(configuration: C) -> C:
    """`configuration` as a definition's functions are handed it: a read-only view of the field's own, so a function that assigns a member fails there.

    The view is of the members: what a member holds, an object among a
    struct's `fields` say, is the field's own, and a function that writes
    into one writes into the field.
    """
    return cast("C", MappingProxyType(cast("Mapping[str, JSONValue]", configuration)))


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
    return tuple(dataclasses.replace(found, loc=(*prefix, *found.loc)) for found in problems)


def _envelope(field: _NestedField) -> Problems:
    """What is wrong with a nested field's envelope, at the field."""
    return _located(field.loc, field.kind.envelope_problems(field.json))


def _configuration_checked(
    value: object, shape: type[T], loc: Loc = ()
) -> tuple[T | None, Problems]:
    """`value` checked as `shape`, as `typed_json.check` checks it, and each nested field's envelope judged.

    The step a definition's `judge` starts from. A member typed with a
    field alias holds a metadata field, whose envelope is judged as a
    document's is: a `must_understand` of `false` is a problem of the
    configuration, and the value does not come back; a stray member is an
    unknown key, reported and left out, and the value still comes back,
    as the checker reports and leaves out a key a closed TypedDict does
    not declare.
    """
    refined, problems = refine_json(value, loc)
    if len(problems) != 0:
        return None, problems
    typed, found, nested = _typed(shape, refined, loc)
    problems = (*found, *(problem for field in nested for problem in _envelope(field)))
    if not _usable(problems):
        return None, problems
    return cast("T", _put_back(typed, _declared)), problems


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
    if not is_object(value):
        return None, None, ()
    entry = value
    name = entry.get("name")
    if not isinstance(name, str):
        return None, None, ()
    if "configuration" not in entry:
        return name, None, ()
    configuration = entry["configuration"]
    if not isinstance(configuration, Mapping):
        return (
            name,
            None,
            problem(("configuration",), f"expected an object, got {shown(configuration)}"),
        )
    return name, cast("Mapping[str, object]", configuration), ()


def _nothing_nested() -> Nested:
    """What a field that holds no field it read holds inside: nothing."""
    return {}


@dataclass(frozen=True, slots=True, kw_only=True)
class Read(Generic[D]):
    """A field a definition in scope read: the name it is written with, the definition, and the configuration it allowed.

    A problem with the envelope around it -- a stray member, a
    `must_understand` of `false` -- is reported with the field and leaves
    it read; so is a problem of a field its configuration holds, which is
    that field's own, as `nested` says. Two fields are equal when they
    read the same, however each was spelled, as `field_key` compares
    them: `"bytes"` and `{"name": "bytes"}` are one field, and so are a
    blosc with and without the `typesize` that `noshuffle` ignores, which
    the definition's `canonical` folds. Equal fields hash alike.
    """

    json: JSONValue
    """The field as written, refined: arrays as tuples; it takes no part in equality."""
    name: str
    """The name it is written with: `"r16"`, though its definition is filed under `r*`."""
    definition: D
    """The definition that read it."""
    configuration: Mapping[str, JSONValue]
    """The configuration, type-checked and allowed by the rules; for raw bits, what the name carries.

    Each field it holds is written as a document writes it, as that
    field's `to_json` writes it, so the configuration says what was read
    however it was spelled: a shard's `"crc32c"` and `{"name": "crc32c"}`
    are one index codec.
    """
    nested: Nested = dataclasses.field(default_factory=_nothing_nested)
    """The fields the configuration holds, each as the scope read it, by where it sits in the configuration.

    A struct's field types at `("fields", 0, "data_type")`, a shard's
    codecs at `("codecs", 0)`: what a definition's functions consult about
    the fields inside its own.
    """
    read_as: type[Definition[Any]] = dataclasses.field(init=False, repr=False)
    """The kind of metadata it was read as: its definition's."""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _FIELDS):
            return NotImplemented
        return field_key(self) == field_key(cast("Resolved[Any]", other))

    def __hash__(self) -> int:
        return hash(field_key(self))

    def __post_init__(self) -> None:
        # The runtime half of the annotations: a field read by hand, as an
        # extension's may be, fails here rather than where a function trusts it.
        definition = cast("object", self.definition)
        kind = kind_of(cast("Definition[Any]", definition))
        refusal = (
            f"a field read is read by a definition of a kind, got {definition!r}"
            if kind is None
            else _misread(definition, kind, self.name)
        )
        if refusal is not None:
            raise TypeError(refusal)
        object.__setattr__(self, "read_as", kind)

    def to_json(self) -> JSONValue:
        """The field as a document writes it, for every reader: its configuration as read, sharing nothing with the field.

        The envelope takes the fewest words every reader takes: a data type
        with nothing to configure is its bare name, as core data types have
        been written since Zarr v3.0; any other field is an object,
        `{"name": ...}`, since a Zarr v3.0 reader takes no bare name in
        `codecs`
        (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L585-L592).
        A name that carries its configuration, as raw bits' does, is
        written alone.
        """
        return copied(cast("JSONValue", document_json(self)))


@dataclass(frozen=True, slots=True, kw_only=True)
class Unclaimed:
    """A field nothing in scope claims: an extension the scope leaves unjudged, which is what keeps the format open.

    Equal to another when it is written with the same name and
    configuration, however each was spelled: nothing in scope interprets
    its configuration, so it compares as JSON text, as `field_key` says.
    """

    json: JSONValue
    """The field as written, refined: arrays as tuples; it takes no part in equality."""
    name: str
    """The name nothing in scope claims."""
    read_as: type[Definition[Any]]
    """The kind of metadata it was read as: what a definition that claimed it would be."""
    configuration: Mapping[str, JSONValue] = dataclasses.field(init=False)
    """The configuration as written, which nothing judged; empty when none is written."""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _FIELDS):
            return NotImplemented
        return field_key(self) == field_key(cast("Resolved[Any]", other))

    def __hash__(self) -> int:
        return hash(field_key(self))

    def __post_init__(self) -> None:
        # The runtime half of the annotations; `read_as` with its type
        # arguments dropped, as `resolve` drops them.
        object.__setattr__(self, "read_as", as_kind(self.read_as))
        name = cast("object", self.name)
        if not isinstance(name, str) or not self.read_as.well_named(name):
            msg = (
                "a field nothing in scope claims is named as a document names a "
                f"{self.read_as.label}, got {name!r}"
            )
            raise TypeError(msg)
        _, written, _ = self.read_as.named_configuration(self.json)
        configuration: Mapping[str, object] = {} if written is None else written
        object.__setattr__(self, "configuration", cast("Mapping[str, JSONValue]", configuration))

    @property
    def definition(self) -> None:
        """The definition that read it: none did."""
        return None

    @property
    def nested(self) -> Nested:
        """The fields its configuration holds as the scope read them: none, since nothing read its configuration."""
        return _nothing_nested()

    def to_json(self) -> JSONValue:
        """The field as a document writes it, sharing nothing with the field: its configuration as written, in the envelope every reader takes, as `Read.to_json` writes one."""
        return copied(cast("JSONValue", document_json(self)))


@dataclass(frozen=True, slots=True, kw_only=True)
class Refused(Generic[D]):
    """A field that could not be read -- not a field at all, not JSON, or refused by the definition that claims its name -- as its problems say."""

    json: JSONValue | UNSET
    """The field as written, refined: arrays as tuples; `UNSET` when it is not JSON, which no document holds."""
    name: str | None
    """The name it is written with; None when it names none."""
    read_as: type[Definition[Any]]
    """The kind of metadata it was read as."""
    definition: D | None = None
    """The definition that claims its name and refused it; None when nothing in scope claims it, or it names none."""
    nested: Nested = dataclasses.field(default_factory=_nothing_nested)
    """The fields its configuration holds, each as the scope read it; empty when its configuration was not checked against its TypedDict."""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _FIELDS):
            return NotImplemented
        return field_key(self) == field_key(cast("Resolved[Any]", other))

    def __hash__(self) -> int:
        return hash(field_key(self))

    def __post_init__(self) -> None:
        # The runtime half of the annotations; `read_as` with its type
        # arguments dropped, as `resolve` drops them.
        kind = as_kind(self.read_as)
        object.__setattr__(self, "read_as", kind)
        definition = cast("object", self.definition)
        refusal = None if definition is None else _misread(definition, kind, self.name)
        if refusal is not None:
            raise TypeError(refusal)


Resolved = TypeAliasType("Resolved", "Read[D] | Unclaimed | Refused[D]", type_params=(D,))
"""One metadata field as a scope read it: read by the definition that claims its name, claimed by nothing, or refused."""

_FIELDS: Final = (Read, Unclaimed, Refused)
"""The three things a scope makes of a field."""


def _misread(definition: object, kind: type[Definition[Any]], name: object) -> str | None:
    """What is wrong with `definition` as the one that read a field named `name` as a `kind`; None when nothing is."""
    if not isinstance(definition, kind):
        return f"a field read as a {kind.__name__} is read by one, got {definition!r}"
    filed = cast("Definition[Any]", definition).name
    if not isinstance(name, str) or spelled(kind, name)[0] != filed:
        return f"a field named {name!r} is read by the definition filed under it, got {filed!r}"
    return None


def field_key(field: Resolved[Any]) -> tuple[object, ...]:
    """What `==` and `hash` compare of a field: what it means, not how it was spelled.

    A field read compares by the definition that read it, its
    configuration in the canonical spelling the definition gives it -- what
    `canonical_of` spells -- as JSON text, and the fields it holds, each by
    its own key. A field nothing claims compares by its name and its
    configuration as written, as JSON text: nothing interprets it. A field
    refused compares by what was written, and by what refused it. So two
    fields are one when what a reader understands of them reads the same,
    and what none interprets is written alike.
    """
    if isinstance(field, Read):
        definition = cast("Definition[Any]", field.definition)
        configuration: JSONValue = dict(field.configuration)
        # A field it holds is compared by its own key, so its place holds
        # nothing before the definition's `canonical` sees the rest, as
        # `_canonical_field` orders it: `canonical` folds only the
        # definition's own members.
        for loc in field.nested:
            configuration = _replaced(configuration, loc, None)
        view = read_only(cast("Mapping[str, JSONValue]", configuration))
        spelled = asked(
            definition,
            "canonical",
            lambda: json_text(cast("JSONValue", dict(definition.canonical(view)))),
        )
        return ("read", definition, spelled, _nested_key(field.nested))
    if isinstance(field, Unclaimed):
        return ("unclaimed", field.read_as, field.name, json_text(field.configuration))
    return (
        "refused",
        field.read_as,
        field.name,
        field.definition,
        UNSET if field.json is UNSET else json_text(field.json),
        _nested_key(field.nested),
    )


def own_key(field: Read[Any]) -> tuple[object, ...]:
    """What `field_key` compares of a read field without the fields it holds: the definition and the canonical spelling of its own members."""
    return field_key(field)[:3]


def _nested_key(nested: Nested) -> tuple[tuple[Loc, tuple[object, ...]], ...]:
    """The fields a configuration holds, each by its key, where it sits."""
    return tuple((loc, field_key(inner)) for loc, inner in nested.items())


def document_json(field: Resolved[Any]) -> JSONValue | UNSET:
    """A field as a document writes it, holding the field's own values: as `to_json` writes it, or as it was written when it was refused, `UNSET` for one that was not JSON.

    What a writer serializes, which changes nothing, so it copies nothing;
    `to_json` is this, copied.
    """
    if isinstance(field, Refused):
        return field.json
    kind = field.read_as
    if isinstance(field, Read) and kind.spelled(field.name)[1] is not None:
        return kind.envelope_json(field.name, {})
    return kind.envelope_json(field.name, field.configuration)


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
    """The data type field of the values, as a scope read it; None when no field says what they are: a document naming none, which its reading holds as `UNSET`, hands the pipeline a chunk of no known type."""

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
    """Whether `value` is a field a scope read as a data type."""
    return isinstance(value, _FIELDS) and cast("Resolved[Any]", value).read_as is DataTypeDefinition


def _is_lengths(value: object) -> TypeGuard[Lengths]:
    """Whether `value` is chunk lengths: per axis, a frozenset of integers, or None."""
    if not is_tuple(value):
        return False
    for axis in value:
        if axis is None:
            continue
        if not isinstance(axis, frozenset) or not all(
            isinstance(length, int) and not isinstance(length, bool)
            for length in cast("frozenset[object]", axis)
        ):
            return False
    return True


def fields_of(resolved: Resolved[Any], loc: Loc = ()) -> Iterator[tuple[Loc, Resolved[Any]]]:
    """`resolved`, a field a scope read, where it sits, then each field it holds and theirs in turn, each where it sits.

    `loc` is where `resolved` sits; a field it holds sits in its
    configuration, at the kind's `configuration_loc` and its place, as `resolve`
    locates its problems. What each holds is its `nested`: a field whose
    configuration was not checked holds none.
    """
    yield loc, resolved
    for place, inner in resolved.nested.items():
        yield from fields_of(inner, (*resolved.read_as.configuration_loc(loc), *place))


def with_problems(
    fields: Iterable[tuple[Loc, Resolved[Any]]], problems: Sequence[ValidationProblem]
) -> Iterator[tuple[Loc, Resolved[Any], Problems]]:
    """Each of `fields`, with where it sits, and the problems among `problems` located in it, in the fields it holds too.

    `fields` and `problems` are one read's: a reading's `fields()` and
    `problems`, or `fields_of` a field and the problems `resolve` gave
    with it. A field's problems are those it was read with, and those the
    document found with it where it stands -- its place in the pipeline,
    the chunk it is handed, the array's shape -- so a field with none is
    valid there, and `canonical_of` spells it. A function of the problems,
    as zod's `treeifyError` is of the issues, grouping them at every
    depth: a problem with a shard's inner codec is the inner codec's, and
    the shard's. Each field comes before the fields it holds, as
    `fields_of` gives them, so the last field whose problems hold a
    problem is the innermost field holding it. A problem in no field --
    with the fill value, with the shape -- is in none's.
    """
    located = list(fields)
    held: dict[Loc, list[ValidationProblem]] = {loc: [] for loc, _ in located}
    for found in problems:
        for depth in range(len(found.loc) + 1):
            holder = held.get(found.loc[:depth])
            if holder is not None:
                holder.append(found)
    for loc, field in located:
        yield loc, field, tuple(held[loc])


def configuration_of(resolved: Resolved[Any], definition: Definition[C]) -> C | None:
    """The configuration `resolved` holds, typed as `definition` declares it, if `definition` read it.

    `Read` holds a configuration as the mapping every one is; asked with
    the definition that read the field -- or one equal to it, as a
    pickled field's is -- this is the same mapping, as its TypedDict. None
    when another definition read it, or none did.
    """
    if not isinstance(resolved, Read) or resolved.definition != definition:
        return None
    return cast("C", resolved.configuration)


def fill_value_problems(data_type: Resolved[F], value: object, loc: Loc = ()) -> Problems:
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
    if len(problems) != 0 or not isinstance(data_type, Read):
        return with_input(problems, value, loc)
    definition, configuration = data_type.definition, data_type.configuration
    typed, problems = _fill_value_parser(definition.fill_value)(refined, loc)
    if not _usable(problems):
        return with_input(problems, value, loc)
    refused = ruled(
        definition,
        lambda: definition.fill_value_rules(read_only(configuration), data_type.nested, typed),
        loc,
    )
    return with_input((*problems, *refused), value, loc)


def canonical_fill_value(data_type: Resolved[F], value: object) -> JSONValue | UNSET:
    """`value`, a fill value of `data_type`, a data type field a scope read, in the one spelling its value has; `UNSET` when it has a problem.

    As the data type's `fill_value_canonical` spells it, so two fill
    values of a data type are one value exactly when their canonical
    spellings are written alike -- the same JSON, as `json.dumps` writes
    it, which `==` is not: it takes `-0.0` for `0.0`. A fill value
    `fill_value_problems` finds a problem with has no canonical spelling,
    as `canonical_of` gives a field with a problem none: `UNSET`, since
    `None` is the JSON `null`, a fill value of a data type the scope did
    not read, which spells a fill value as written.
    """
    if len(fill_value_problems(data_type, value)) != 0:
        return UNSET
    refined, _ = refine_json(value, ())
    return spelled_canonically(data_type, refined)


def spelled_canonically(data_type: Resolved[F], value: JSONValue) -> JSONValue:
    """`value`, a fill value of `data_type` with no problem, in its canonical spelling, as `canonical_fill_value` gives it, without judging it again.

    Its `fill_value_canonical` is the extension author's code: what it
    gives is checked to be JSON, and an error it raises says which data
    type's canonical spelling raised it.
    """
    if not isinstance(data_type, Read):
        return value
    definition, configuration = data_type.definition, data_type.configuration
    spelled = asked(
        definition,
        "fill_value_canonical",
        lambda: cast(
            "object",
            definition.fill_value_canonical(read_only(configuration), data_type.nested, value),
        ),
    )
    refined, problems = refine_json(spelled, ())
    if len(problems) != 0:
        msg = (
            f"{definition.name!r}: its fill_value_canonical gives JSON, got {spelled!r}: "
            f"{problems[0].message}"
        )
        raise TypeError(msg)
    return refined


def storage_of(data_type: Resolved[DataTypeDefinition[Any]]) -> StorageClass | None:
    """How the values of `data_type`, a data type field a scope read, are stored; None when unknown.

    Unknown when the scope did not read it, or its definition does not
    say. Its `storage` is the extension author's code: what it gives is
    checked to be a storage class, and an error it raises says which data
    type's storage raised it.
    """
    if not isinstance(data_type, Read):
        return None
    definition, configuration = data_type.definition, data_type.configuration
    found = asked(
        definition,
        "storage",
        lambda: cast("object", definition.storage(read_only(configuration), data_type.nested)),
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
    if not isinstance(chunk_grid, Read):
        return unknown, ()
    definition, configuration = chunk_grid.definition, chunk_grid.configuration
    at = (*loc, "configuration")
    problems = ruled(
        definition,
        lambda: definition.shape_rules(read_only(configuration), chunk_grid.nested, shape),
        at,
    )
    if len(problems) != 0:
        return unknown, with_input(problems, chunk_grid.json, loc)
    lengths = asked(
        definition,
        "chunk lengths",
        lambda: cast(
            "object", definition.chunk_lengths(read_only(configuration), chunk_grid.nested, shape)
        ),
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
    sits, as with a document's fields. What comes back is `Read` by the
    definition that claims the name; `Unclaimed` when nothing in scope
    claims it, an unmodelled extension left unjudged, which is what keeps
    the format open; or `Refused`, with the problems that say why. `loc`
    prefixes every problem. `kind` is one of the five kinds --
    `CodecDefinition`, `DataTypeDefinition`, `ChunkGridDefinition`,
    `ChunkKeyEncodingDefinition`, `StorageTransformerDefinition` -- with
    or without type arguments; anything else is a `TypeError`.
    """
    asked = as_kind(kind)
    refined, problems = refine_json(data, loc)
    if len(problems) != 0:
        # Not JSON, so not read; its name, if it has one, still says what
        # claims it, and one the spec does not give an extension is a
        # problem here as on the other path, asked of no definition.
        name = asked.named_configuration(data)[0]
        bad = None if name is None else asked.name_problem(name, asked.name_loc(loc))
        claimant = None if name is None or bad is not None else context.claimant(asked, name)
        refused = Refused(json=UNSET, name=name, read_as=asked, definition=claimant)
        found = problems if bad is None else (bad, *problems)
        return cast("Resolved[D]", refused), with_input(found, data, loc)
    resolved, found = _resolve_field(refined, asked, context, loc)
    return cast("Resolved[D]", resolved), with_input(found, data, loc)


def _resolve_field(
    data: JSONValue, kind: type[Definition[Any]], context: Context, loc: Loc
) -> tuple[Resolved[Definition[Any]], Problems]:
    """A refined field with its envelope judged, then read.

    What the scope made of it is what became of the configuration. A
    stray member or a `must_understand` of `false` says nothing about it,
    so it is reported beside the field that was read, which later layers
    can still judge.
    """
    envelope = _located(loc, kind.envelope_problems(data))
    resolved, found = _read(data, kind, context, loc)
    return resolved, (*envelope, *found)


def _read(
    data: JSONValue, kind: type[Definition[Any]], context: Context, loc: Loc
) -> tuple[Resolved[Definition[Any]], Problems]:
    name, given, malformed = kind.named_configuration(data)
    if name is None:
        return Refused(json=data, name=None, read_as=kind), ()
    if not kind.well_named(name):
        # The envelope rule every reader runs first reports it; no
        # definition is asked to claim it.
        return Refused(json=data, name=name, read_as=kind), ()
    definition = context.claimant(kind, name)
    if len(malformed) != 0:
        # A configuration that is not an object, which the envelope's
        # problems say; the name still says what claims the field.
        return Refused(json=data, name=name, read_as=kind, definition=definition), ()
    if definition is None:
        return Unclaimed(json=data, name=name, read_as=kind), ()
    _, carried = kind.spelled(name)
    if carried is not None:
        return _read_carried(data, name, kind, definition, given, carried, loc)
    at = kind.configuration_loc(loc)
    if given is None and definition.requires_configuration:
        missing = problem(at, f"{name!r} requires a configuration", "missing_key")
        return Refused(json=data, name=name, read_as=kind, definition=definition), missing
    typed, found, nested = _typed(definition.configuration, {} if given is None else given, at)
    # The fields it holds are read first, each a frame deeper than this
    # one, so the rules see them as the scope read them; each is put back
    # as a document writes it, so the configuration says what was read
    # however each was spelled. Their problems are reported after the
    # rules'.
    within: dict[Loc, Resolved[Any]] = {}
    written: dict[Loc, JSONValue] = {}
    inside: list[ValidationProblem] = []
    for field in nested:
        inside.extend(_envelope(field))
        inner, found_inside = _read(field.json, field.kind, context, field.loc)
        within[field.loc[len(at) :]] = inner
        # Refined JSON came in, so what was read of it is JSON, or a field
        # refused for what it holds, never for not being JSON.
        written[field.loc] = cast("JSONValue", document_json(inner))
        inside.extend(found_inside)
        inside.extend(_sized(field, inner))
    # The rules may read a field the configuration holds by its name, so
    # they are asked only when each one is named; any other problem with
    # one is its own, reported where it sits, as a document's fields are.
    sound = _usable(found) and all(_named(field) for field in nested)
    configuration = (
        cast("Mapping[str, JSONValue]", _put_back(typed, lambda field: written[field.loc]))
        if sound
        else None
    )
    own = list(found)
    if configuration is not None:
        own.extend(
            ruled(definition, lambda: definition.rules(read_only(configuration), within), at)
        )
    if configuration is None or not _usable(own):
        refused = Refused(json=data, name=name, read_as=kind, definition=definition, nested=within)
        return refused, (*own, *inside)
    read = Read(
        json=data, name=name, definition=definition, configuration=configuration, nested=within
    )
    return read, (*own, *inside)


def _named(field: _NestedField) -> bool:
    """Whether a field a configuration holds is named, with an object for its configuration if it has one."""
    name, _, malformed = field.kind.named_configuration(field.json)
    return name is not None and len(malformed) == 0


def _read_carried(
    data: JSONValue,
    name: str,
    kind: type[Definition[Any]],
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
        EmptyConfiguration,
        {} if given is None else given,
        kind.configuration_loc(loc),
    )
    configuration, judged = definition.judge(carried)
    # What is wrong with what the name carries is the field's: found at
    # the field, where the name is what is there, and what was expected
    # of a member of the configuration is not expected of it.
    problems = (
        *beside,
        *(dataclasses.replace(found, loc=loc, input=UNSET, ctx={}) for found in judged),
    )
    # A name is one word: what it carries that the definition does not
    # declare cannot be left out, as a stray key beside the name can, so
    # anything wrong with what the name carries refuses the field.
    if configuration is None or not _usable(beside) or len(judged) != 0:
        refused = Refused(json=data, name=name, read_as=kind, definition=definition)
        return refused, problems
    return Read(json=data, name=name, definition=definition, configuration=configuration), problems


def _sized(field: _NestedField, inner: Resolved[Any]) -> Problems:
    """A codec of dynamic size in a member that takes codecs of static size, as a problem at the field.

    A name nothing in scope claims is left unjudged, its size unknown, as
    everything else about it is.
    """
    definition = inner.definition
    if not field.static or not isinstance(definition, CodecDefinition):
        return ()
    if definition.size == "static":
        return ()
    return problem(
        field.loc,
        f"{inner.name!r} is a codec of dynamic size, and only codecs of static size may be used "
        "here",
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
    A name nothing in scope claims keeps the configuration it was written
    with, since what it simplifies to is its own definition's call. A
    field a scope has read already is spelled so by `canonical_of`,
    given its problems.
    """
    resolved, problems = resolve(data, kind, context, loc)
    return canonical_of(resolved, problems), problems


def canonical_of(
    resolved: Resolved[Any], problems: Sequence[ValidationProblem]
) -> JSONValue | None:
    """`resolved`, a field a scope read, in its simplest equivalent spelling, as `canonicalize` spells one; None when it has a problem.

    `problems` are the field's, as `resolve` gives them, or
    `with_problems` gives each field of a reading. Only a field with none
    has a simplest spelling: a simpler spelling of one with a problem
    would erase what its author wrote -- a key its TypedDict does not
    declare -- or spell what does not hold. What is spelled is what the
    scope read, without reading the field again.
    """
    if len(problems) != 0:
        return None
    return _simplest(resolved)


def _simplest(field: Resolved[Any]) -> JSONValue | None:
    """A field in its simplest spelling; None when it, or a field it holds, was refused, which has none."""
    if isinstance(field, Read):
        return _canonical_field(field)
    if isinstance(field, Unclaimed):
        return field.to_json()
    return None


def _canonical_field(resolved: Read[Any]) -> JSONValue | None:
    """A field that read, in its simplest equivalent spelling: the fields it holds first, then its own members; None when one it holds was refused."""
    definition, name = resolved.definition, resolved.name
    configuration: JSONValue = dict(resolved.configuration)
    for loc, inner in resolved.nested.items():
        simplest = _simplest(inner)
        if simplest is None:
            return None
        configuration = _replaced(configuration, loc, simplest)
    # The view every function of a definition is handed; what `canonical`
    # gives, the view itself when nothing is folded, is taken as a dict.
    view = read_only(cast("Mapping[str, JSONValue]", configuration))
    simplified = asked(
        definition,
        "canonical",
        lambda: dict(cast("Mapping[str, JSONValue]", definition.canonical(view))),
    )
    _, refused = definition.judge(simplified)
    if len(refused) != 0:
        msg = (
            f"{definition.name!r}: its canonical gave {simplified!r}, which does not hold: "
            f"{list(refused)!r}"
        )
        raise ValueError(msg)
    carrying = definition.carrying_name(simplified)
    if carrying is not None:
        return carrying
    return resolved.read_as.envelope_json(name, simplified)


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
    "Read",
    "Refused",
    "Resolved",
    "StaticCodecField",
    "StorageClass",
    "StorageTransformerDefinition",
    "StorageTransformerField",
    "Unclaimed",
    "as_kind",
    "asked",
    "canonical_fill_value",
    "canonical_of",
    "canonicalize",
    "chunk_grid_lengths",
    "configuration_of",
    "field_json_schema",
    "field_key",
    "field_kind",
    "field_schemas",
    "fill_value_as_written",
    "fill_value_problems",
    "kind_of",
    "multi_byte",
    "named_configuration",
    "no_pipelines",
    "no_rules",
    "own_key",
    "read_only",
    "resolve",
    "ruled",
    "single_byte",
    "spelled",
    "spelled_canonically",
    "storage_of",
    "unchanged",
    "unknown_chunk",
    "unknown_lengths",
    "unknown_storage",
    "variable_length",
    "well_named",
    "with_problems",
    "written_name",
]
