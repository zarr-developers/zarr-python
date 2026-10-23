"""The kinds of metadata in a Zarr v3 document, and what reading one needs that is specific to v3.

The five kinds are `DataTypeDefinition`, `ChunkGridDefinition`,
`ChunkKeyEncodingDefinition`, `CodecDefinition` and
`StorageTransformerDefinition`. This module also holds the chunk that a
codec receives, how a data type's values are stored, the raw bits name
`r*`, and the JSON Schema of a v3 field. The reading itself is in
`zarr_metadata.definition`, which both formats share.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Final,
    Literal,
    TypeAlias,
    TypeGuard,
    cast,
    get_args,
)

from typing_extensions import TypeAliasType

from zarr_metadata._json import ValidationProblem, is_tuple, with_input
from zarr_metadata._typed_json import JSONSchema, Loc, SchemaLeaf, Schemas, is_alias, problem
from zarr_metadata.definition._definition import (
    AcceptedField,
    C,
    Definition,
    EmptyConfiguration,
    Nested,
    Problems,
    ResolvedField,
    WithFillValue,
    as_kind,
    asked,
    field_kind,
    is_field,
    no_rules,
    read_only,
    ruled,
)
from zarr_metadata.definition._envelope import EXTENSION_NAME_SCHEMA_PATTERN
from zarr_metadata.v3._common import (
    ZarrV3ChunkGridField,
    ZarrV3ChunkKeyEncodingField,
    ZarrV3CodecField,
    ZarrV3DataTypeField,
    ZarrV3StaticCodecField,
    ZarrV3StorageTransformerField,
)

if TYPE_CHECKING:
    from zarr_metadata._common import JSONValue
    from zarr_metadata.definition._context import Context


def unknown_lengths(configuration: object, nested: object, shape: tuple[int, ...]) -> Lengths:
    """The chunk lengths of a grid that does not declare them: unknown along each axis of the array."""
    return (None,) * len(shape)


def unknown_chunk(*_: object) -> Chunk:
    """The output of an array -> array codec that does not declare its output: a chunk with nothing known about it."""
    return Chunk()


def no_pipelines(*_: object) -> Mapping[str, Chunk]:
    """The pipelines of a codec that contains none: an empty mapping."""
    return {}


StorageClass = Literal["single_byte", "multi_byte", "variable_length"]
"""How a data type's values are stored: in single bytes, in several bytes at a time, or each value in as many bytes as it needs.

A multi-byte number is stored in a byte order, which the `bytes` codec's
`endian` sets. A value made of single bytes (a `uint8`, or a struct of
`int8` fields) has no byte order. A value whose size varies needs its
own codec.
"""


STORAGE_CLASSES: Final[tuple[StorageClass, ...]] = get_args(StorageClass)
"""The storage classes. `storage_of` checks a definition's result against these."""


def single_byte(*_: object) -> StorageClass:
    """The storage of a data type made of single bytes, where byte order does not apply, such as `uint8`."""
    return "single_byte"


def multi_byte(*_: object) -> StorageClass:
    """The storage of a data type with multi-byte numbers, where byte order applies, such as `int16`."""
    return "multi_byte"


def variable_length(*_: object) -> StorageClass:
    """The storage of a data type whose values vary in size, such as `string`."""
    return "variable_length"


def unknown_storage(*_: object) -> None:
    """The storage of a data type that does not declare it: unknown."""


RAW_BYTES_NAME: Final = "r*"
"""The name raw bits are filed under: `r*`, as written in the specification's table of data types.

A document writes `r` followed by the size in bits (`r8`, `r16`), so raw
bits are the only data type whose name encodes its configuration.
`filed_under` reads such a name as `r*` with the size as its
configuration. `r*` itself is the table's notation, and no document uses
it as a name
(https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/data-types/index.rst#L46-L47).
"""


RAW_BYTES_NAME_PATTERN: Final = re.compile(r"r([0-9]{1,100})")
"""A raw bits name: `r` followed by a size in bits of up to a hundred digits, matched against the whole name, whether or not the size is allowed.

Only ASCII digits match. `\\d` would also match every other Unicode
decimal digit, so `r\uff11\uff16` would be read as sixteen bits, and a
third-party name spelled that way would be mistaken for raw bits. A size
the spec does not allow (`r0`, `r12`) is still raw bits, so it is
reported as a bad size instead of being treated as an unknown extension.
A hundred digits is more than any real size. `r` followed by more digits
than that is an ordinary name, which the spec allows, and which nothing
in scope claims.
"""


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class DataTypeDefinition(WithFillValue[C], kind=True, format=3):
    """A data type, and the fill value of an array with that data type.

    `fill_value` is the JSON shape of a fill value, such as
    `Int8FillValue`. It is an annotation that the checker reads the same
    way it reads a configuration's members, including any range.
    `fill_value_rules` yields the spec violations in a fill value of that
    shape that the type cannot express, such as a hex string of the wrong
    width.
    The rules receive the configuration, the nested fields as the scope
    read them (a struct's field types), and the typed fill value. A data
    type that does not declare its fill value accepts any JSON.

    `fill_value_canonical` writes a fill value without problems (well
    typed and allowed by the rules) in the single canonical spelling of
    its value. So two fill values are the same value of the type exactly
    when their canonical spellings are written the same way: `"NaN"` and
    `"0x7fc00000"` are the same `float32`, but `0.0` and `-0.0` are two
    different values. It receives the same arguments as the rules. A data
    type that does not declare it has one spelling per value: the value as
    written.

    `storage` declares how the values are stored: in single bytes, in
    several bytes at a time, or each value in as many bytes as it needs.
    The `bytes` codec uses this to decide whether the data type it
    receives needs an `endian` (it does for multi-byte numbers). A
    struct's storage depends on its fields, so `storage` receives the
    nested fields as the scope read them. A data type that does not
    declare it leaves its storage unknown.

    A data type named the way a document writes raw bits of one size (such
    as `r16`) is rejected. That name is read as `r*`, so this definition
    would never be used.
    """

    label: ClassVar[str] = "data type"
    field_aliases: ClassVar[tuple[TypeAliasType, ...]] = (ZarrV3DataTypeField,)

    @classmethod
    def filed_under(cls, name: str) -> tuple[str | None, dict[str, JSONValue] | None]:
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
        # A data type with nothing to configure is written as its bare name,
        # which is how core data types have been written since Zarr v3.0.
        if len(configuration) != 0:
            return {"name": name, "configuration": configuration}
        return name

    storage: Callable[[C, Nested], StorageClass | None] = unknown_storage
    """How the values are stored, given the configuration and the nested fields; None when unknown."""

    def _refusal(self) -> str | None:
        if RAW_BYTES_NAME_PATTERN.fullmatch(self.name) is not None:
            return (
                f"{self.name!r} is how a document writes raw bits of one size, which read as "
                f"{RAW_BYTES_NAME!r}; to read raw bits your own way, define {RAW_BYTES_NAME!r}"
            )
        # Use the explicit form, not bare `super()`, because a dataclass
        # with slots is rebuilt.
        return super(DataTypeDefinition, self)._refusal()


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class ChunkGridDefinition(Definition[C], kind=True, format=3):
    """A chunk grid, and the array shapes it fits.

    `shape_rules` yields the spec violations in a grid with this
    configuration over an array of a given shape, such as a dimension with
    no chunk length, or chunks that do not cover the dimension. It receives
    the configuration, the nested fields as the scope read them, and the
    shape, and it locates its problems in the configuration. A grid that
    does not declare shape rules fits every shape.

    `chunk_lengths` determines what the first codec of the array's
    pipeline receives: the lengths of the grid's chunks along each axis of
    an array whose shape the grid fits. A regular grid has one length per
    axis, and a rectilinear grid has every length it lists. It is called
    only for a grid that its shape rules accept. A grid that does not
    declare it leaves the lengths along every axis unknown.
    """

    label: ClassVar[str] = "chunk grid"
    field_aliases: ClassVar[tuple[TypeAliasType, ...]] = (ZarrV3ChunkGridField,)

    shape_rules: Callable[[C, Nested, tuple[int, ...]], Iterable[ValidationProblem]] = no_rules
    """The spec violations in this grid over an array of a given shape, located in the configuration."""
    chunk_lengths: Callable[[C, Nested, tuple[int, ...]], Lengths] = unknown_lengths
    """The lengths of the chunks along each axis of an array whose shape the grid fits; None where unknown."""


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class ChunkKeyEncodingDefinition(Definition[C], kind=True, format=3):
    """A chunk key encoding."""

    label: ClassVar[str] = "chunk key encoding"
    field_aliases: ClassVar[tuple[TypeAliasType, ...]] = (ZarrV3ChunkKeyEncodingField,)


CodecKind = Literal["array_array", "array_bytes", "bytes_bytes"]
"""What a codec converts its input to: the three positions a pipeline puts in order."""


CodecSize = Literal["static", "dynamic"]
"""Whether the size of a codec's output is determined by the size of its input.

`static`: it is. For example, `bytes` writes each element at its width,
and `crc32c` adds four bytes. `dynamic`: the size depends on the values,
as for every compressor.
"""


_UNASKED: Final[Mapping[CodecKind, tuple[str, ...]]] = {
    "array_array": (),
    "array_bytes": ("transition",),
    "bytes_bytes": ("chunk_rules", "transition", "pipelines"),
}
"""The functions that are never called for a codec of each kind. A bytes -> bytes codec receives bytes, and only an array -> array codec passes on a chunk."""


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class CodecDefinition(Definition[C], kind=True, format=3):
    """A codec: what it converts its input to, and whether the size of its output is static.

    A codec that receives an array (array -> array, array -> bytes)
    declares the spec violations for a given input `Chunk` with
    `chunk_rules`. It receives the configuration, the nested fields as the
    scope read them, and the chunk, and it locates its problems in the
    configuration. For example, a `bytes` codec without `endian` that
    receives a multi-byte data type is a problem. An array -> array codec
    also declares its output with `transition`: given its input chunk, it
    returns the chunk the next codec receives. For example, `transpose`
    permutes the axes, and `cast_value` changes the data type. These two
    functions correspond to the spec: a codec computes its output from the
    shape and data type of its input, and "If the
    decoded_representation_type is not supported, this algorithm must
    fail with an error"
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L987-L994).
    The transition is called for every input chunk, whatever the chunk
    rules found, so it must return only what is true in either case. For
    example, a `transpose` whose `order` has the wrong number of axes
    passes on unknown lengths. A codec that does not declare its output
    passes the next codec a chunk with nothing known about it.

    A codec that contains its own pipelines declares what each one
    receives with `pipelines`. Given the codec's input chunk, it returns
    the chunk that the first codec of each pipeline receives, keyed by the
    configuration member that holds the pipeline. For example, a shard's
    inner codecs receive its inner chunks, and its index codecs receive
    the shard index. Like the transition, it is called whatever the chunk
    rules found, and must return only what is true in either case. A
    function that is never called for a codec of its kind is rejected,
    such as the chunk rules of a bytes -> bytes codec, which receives
    bytes.
    """

    label: ClassVar[str] = "codec"
    field_aliases: ClassVar[tuple[TypeAliasType, ...]] = (ZarrV3CodecField, ZarrV3StaticCodecField)

    kind: CodecKind
    size: CodecSize
    chunk_rules: Callable[[C, Nested, Chunk], Iterable[ValidationProblem]] = no_rules
    """The spec violations for this codec with a given input chunk, located in the configuration."""
    transition: Callable[[C, Nested, Chunk], Chunk] = unknown_chunk
    """The chunk the next codec receives, given the chunk this array -> array codec receives."""
    pipelines: Callable[[C, Nested, Chunk], Mapping[str, Chunk]] = no_pipelines
    """The chunk each contained pipeline receives, keyed by the configuration member that holds the pipeline."""

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

    @classmethod
    def alias_problems(
        cls, alias: TypeAliasType, definition: Definition[Any], loc: Loc
    ) -> Problems:
        """A codec of dynamic size in a member annotated with `ZarrV3StaticCodecField`, which admits only codecs of static size, is a problem at the field."""
        if alias not in _STATIC_SIZE:
            return ()
        if cast("CodecDefinition[Any]", definition).size == "static":
            return ()
        return problem(
            loc,
            f"{definition.name!r} is a codec of dynamic size, and only codecs of static size "
            "may be used here",
            "invalid_value",
        )


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class StorageTransformerDefinition(Definition[C], kind=True, format=3):
    """A storage transformer."""

    label: ClassVar[str] = "storage transformer"
    field_aliases: ClassVar[tuple[TypeAliasType, ...]] = (ZarrV3StorageTransformerField,)


KINDS: Final[tuple[type[Definition[Any]], ...]] = (
    DataTypeDefinition,
    ChunkGridDefinition,
    ChunkKeyEncodingDefinition,
    CodecDefinition,
    StorageTransformerDefinition,
)
"""The kinds of metadata in a document. A scope files each kind separately."""


_STATIC_SIZE: Final[frozenset[TypeAliasType]] = frozenset({ZarrV3StaticCodecField})
"""The field aliases whose codec must be of static size."""


def kind_field(kind: type[Definition[Any]]) -> object:
    """The field alias for a member that holds any field of `kind`: the first alias the kind declares that does not restrict the field. For example, `ZarrV3CodecField` instead of `ZarrV3StaticCodecField`."""
    return next(alias for alias in kind.field_aliases if alias not in _STATIC_SIZE)


_RAW_BYTES_SCHEMA_PATTERN: Final = f"^{RAW_BYTES_NAME_PATTERN.pattern}(?![\\s\\S])"
"""`RAW_BYTES_NAME_PATTERN`, matched against the whole name, as a JSON Schema pattern.

A lookahead for "no character at all" anchors the match to the end of
the name. A `$` would also match before a final newline in a validator
that matches patterns the way Python does, so `"r16\\n"`, which is not a
valid name, would be read as raw bits.
"""


def field_json_schema(kind: type[Definition[Any]], context: Context) -> JSONSchema:
    """The JSON Schema of one metadata field read as `kind` in `context`: everything `resolve` checks, except the rules.

    The schema accepts a field that one of the definitions in scope reads
    (its name, its configuration as the TypedDict declares it, an optional
    `must_understand` of `true`, and its bare name when it needs no
    configuration). It also accepts a name that no definition claims, with
    any configuration, which is what keeps the format open. A field inside
    a configuration is written the same way, in the same scope, and a
    member that accepts only codecs of static size accepts only those. The
    output is JSON Schema draft 2020-12, as `json_schema` writes it. The
    nested fields and the configuration of each definition are in
    `$defs`, under the name of the field alias or TypedDict. Constraints
    that only a rule enforces (a blosc `typesize` checked against its
    `shuffle`) are not in the schema, so a field it accepts may still have
    a problem.
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
    """The schema leaf that `field_json_schema` uses. It writes each field alias as a field of its kind, as `context` reads it."""

    def leaf(annotation: object, schemas: Schemas) -> JSONSchema | None:
        kind = field_kind(annotation)
        if kind is None or not is_alias(annotation):
            return None
        alias = annotation
        static = annotation in _STATIC_SIZE
        return schemas.defined(
            alias, alias.__name__, lambda: _field_schema(kind, static, context, schemas)
        )

    return leaf


def _field_schema(
    kind: type[Definition[Any]], static: bool, context: Context, schemas: Schemas
) -> JSONSchema:
    """The schema of a field of `kind` as `context` reads it: a field that a definition in scope reads, or a field that none of them claims."""
    table = context.tables.get(kind, {})
    branches: list[JSONValue] = []
    for definition in table.values():
        if static and cast("CodecDefinition[Any]", definition).size != "static":
            continue
        branches.extend(_read_by(definition, schemas))
    branches.extend(_unclaimed(table))
    return {"anyOf": branches}


def written_name(definition: Definition[Any]) -> JSONSchema:
    """The JSON Schema of the names a document may use for `definition`: its name, or `r` followed by a size for raw bits."""
    if isinstance(definition, DataTypeDefinition) and definition.name == RAW_BYTES_NAME:
        return {"type": "string", "pattern": _RAW_BYTES_SCHEMA_PATTERN}
    return {"const": definition.name}


def _read_by(definition: Definition[Any], schemas: Schemas) -> list[JSONValue]:
    """The schemas of the fields `definition` reads: an object with its name and configuration, and its bare name when it needs no configuration.

    The raw bits name encodes the configuration, so the configuration
    written next to it must be empty.
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
    """The schemas of the fields that no definition in `table` claims: a name that none of them uses, either bare or with any configuration."""
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


Lengths: TypeAlias = tuple[frozenset[int] | None, ...]
"""For each axis, every chunk length along that axis (a set, because a rectilinear grid's lengths differ), or None where unknown."""


@dataclass(frozen=True, slots=True)
class Chunk:
    """The input a codec receives: chunks with given lengths along each axis, and a data type.

    Unknown parts are None. This includes the lengths along an axis that
    the grid does not declare, and every part of the chunk output by a
    codec that does not declare its output. A data type field that the
    scope did not read is kept as written, and it also says nothing about
    the values. A codec's chunk rules check only what is known, so a chunk
    with nothing known about it, `Chunk()`, never causes a problem.
    """

    lengths: Lengths | None = None
    """For each axis, the chunk lengths along it; None when even the number of axes is unknown."""
    data_type: ResolvedField[DataTypeDefinition[Any]] | None = None
    """The data type field of the values, as a scope read it. None when no field declares the data type: a document with no data type (its reading holds `UNSET`) passes the pipeline a chunk of unknown type."""

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
    """Whether `value` is a field that a scope read as a data type."""
    return is_field(value) and value.read_as is DataTypeDefinition


def _is_lengths(value: object) -> TypeGuard[Lengths]:
    """Whether `value` is valid chunk lengths: for each axis, a frozenset of integers or None."""
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


def storage_of(data_type: ResolvedField[DataTypeDefinition[Any]]) -> StorageClass | None:
    """How the values of `data_type` (a data type field a scope read) are stored; None when unknown.

    The storage is unknown when the scope did not read the data type, or
    its definition does not declare storage. The definition's `storage` is
    the extension author's code. Its result is checked to be a storage
    class. If it raises an error, a note is added that says which data
    type's storage raised it.
    """
    if not isinstance(data_type, AcceptedField):
        return None
    definition, configuration = data_type.definition, data_type.configuration
    found = asked(
        definition,
        "storage",
        lambda: cast("object", definition.storage(read_only(configuration), data_type.nested)),
    )
    if found is not None and found not in STORAGE_CLASSES:
        msg = (
            f"{definition.name!r}: its storage gives one of {STORAGE_CLASSES!r} or None, "
            f"got {found!r}"
        )
        raise TypeError(msg)
    return found


def chunk_grid_lengths(
    chunk_grid: ResolvedField[ChunkGridDefinition[Any]], shape: tuple[int, ...], loc: Loc = ()
) -> tuple[Lengths, Problems]:
    """The chunk lengths of `chunk_grid` (a chunk grid field a scope read) along each axis of an array of `shape`, and the problems with the grid for that shape.

    A chunk has an extent "for each dimension of the array"
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L277-L281),
    so the lengths have an entry for each dimension of `shape`, with None
    where they are unknown. The grid's shape rules check it first, and
    locate their problems in the configuration, under `loc` (the location
    of the field). For example, a regular grid with chunk lengths for two
    dimensions over an array with three dimensions is a problem. Only a
    grid that fits the shape returns its lengths. A grid that the scope
    did not read (it is out of scope or invalid) is not checked and
    returns no lengths. The result of a grid's `chunk_lengths` is checked:
    lengths of the wrong type raise `TypeError`, and lengths with a
    different number of axes than `shape` raise `ValueError`. Both
    indicate a bug in the definition, not in the field.
    """
    unknown: Lengths = (None,) * len(shape)
    if not isinstance(chunk_grid, AcceptedField):
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


__all__ = [
    "KINDS",
    "RAW_BYTES_NAME",
    "RAW_BYTES_NAME_PATTERN",
    "Chunk",
    "ChunkGridDefinition",
    "ChunkKeyEncodingDefinition",
    "CodecDefinition",
    "CodecKind",
    "CodecSize",
    "DataTypeDefinition",
    "Lengths",
    "StorageClass",
    "StorageTransformerDefinition",
    "chunk_grid_lengths",
    "field_json_schema",
    "field_schemas",
    "multi_byte",
    "no_pipelines",
    "single_byte",
    "storage_of",
    "unknown_chunk",
    "unknown_lengths",
    "unknown_storage",
    "variable_length",
    "written_name",
]
