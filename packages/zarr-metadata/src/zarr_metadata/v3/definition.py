"""The kinds of metadata in a Zarr v3 document, and the scopes this package provides for it.

The kind of metadata a definition defines is its type: `CodecDefinition`
(with the codec's `kind`, and its `size`: `"static"` if the output size
is determined by the input size, or `"dynamic"` if it depends on the
values), `DataTypeDefinition`, `ChunkGridDefinition`,
`ChunkKeyEncodingDefinition`, `StorageTransformerDefinition`, and, for
the members of a document that the spec does not define,
`ArrayExtraFieldDefinition` and `GroupExtraFieldDefinition`. What a
definition is and how a field is read are in `zarr_metadata.definition`,
which both formats share. `ZarrV3Context` is the v3 scope, with one slot
for each kind: `data_types`, `chunk_grids`, `chunk_key_encodings`,
`codecs`, `storage_transformers`, `array_extra_fields` and
`group_extra_fields`. `CORE` and `CORE_AND_EXTENSIONS` are the v3 scopes
this package provides.

A scope reads whole documents as well as fields.
`validate_array_metadata_v3(document, context=CONTEXT)`, from
`zarr_metadata.model`, reads each extension point of a v3 array document
through the definitions in `CONTEXT`, and so do the model's `from_json` and
`from_key_value`. The fill value is checked against the data type it
names, by that data type's definition. The chunk grid is checked against
the shape, by the grid's definition. The codecs are checked as a
pipeline, each by its definition against the chunk it receives.

A member that holds another metadata field is annotated with the field
alias of its kind (a shard's `codecs: tuple[ZarrV3CodecField, ...]`) and
is read in the same scope as its parent field. A problem with the nested
field belongs to that field and is reported at its location. The parent
field is still read, as a document containing the nested field would be.
A member that accepts only codecs of static size is annotated
`ZarrV3StaticCodecField`. A shard's `index_codecs` uses this, because a
reader locates the index from a size it knows before reading it. A codec
of dynamic size in such a member is a problem at its location, when the
field is read in a scope. A name that nothing claims is not checked, and
its size is unknown, like everything else about it.
`ZarrV3MetadataFieldJSON` describes the same JSON, but it is checked
only as JSON, so a definition rejects a member typed with it. An
extension with nothing to configure uses `EmptyConfiguration`, and is
written with its name only.

A data type also declares how its values are stored. `storage` is a
function of the configuration and the nested fields that returns a
`StorageClass`: single bytes, several bytes at a time, or each value in
as many bytes as it needs. A struct's storage depends on its fields.
`storage_of(data_type)` calls it for a data type field the scope read.
The `bytes` codec requires an `endian` for multi-byte numbers, and a
struct rejects a field whose values vary in size. A data type that does
not declare storage leaves it unknown.

A chunk grid declares which array shapes it fits with `shape_rules`, a
function that yields the spec violations in a grid with its
configuration over an array of a given shape (a dimension with no chunk
length, or chunks that do not cover the dimension), located in the
configuration. A grid that does not declare shape rules fits every
shape. It also declares the lengths of its chunks along each axis of an
array it fits, with `chunk_lengths`: a set per axis, because a
rectilinear grid's chunks differ in length. A reading stores both
results for the grid it read: an entry for each dimension of the shape,
with None where the lengths are unknown.

A codec is checked against its input. The array passes its first codec
a `Chunk`: the lengths of the grid's chunks along each of the array's
dimensions, and the array's data type field, with None for unknown
parts. A codec that receives an array declares the spec violations for
a given chunk with `chunk_rules`, located in its configuration (for
example, a `transpose` whose `order` has the wrong number of axes). An
array -> array codec declares what it passes to the next codec with
`transition`, whatever its chunk rules found. For example, `transpose`
permutes the axes. A reading reads the codec fields as a pipeline. It
checks their order (array -> array codecs, one array -> bytes codec,
bytes -> bytes codecs), then checks each codec against the chunk it
receives, and records a `Stage` for each codec with that chunk. A codec
that contains its own pipelines declares what each one receives with
`pipelines`, keyed by the configuration member that holds each
pipeline. For example, a shard's inner codecs receive its inner chunks,
and its index codecs receive the shard index. Each pipeline is read the
same way, and its stages are stored as the codec's `Stage.inner`.
The reader does not guess. After a codec the scope did not read, or
after a codec that does not declare its output, the next codec receives
a chunk with nothing known about it, `Chunk()`, which never causes a
problem. A later codec can still declare its own output.

Raw bits are the only data type whose name encodes its configuration. A
document writes `r` followed by the size in bits, and `r16` is read as
`r*` (the notation in the specification's table of data types) with
`{"bits": 16}`. A reader that wants to read raw bits its own way defines
`r*`. A data type named `r16` is rejected, because that name is read as
`r*`. `r*` itself is notation, and a document that uses it as a name
refers to nothing in any scope.

**JSON Schema.** `zarr_metadata.json_schema` writes a v3 field, or a whole
`zarr.json`, as a JSON Schema.
"""

from zarr_metadata.v3._common import (
    ZarrV3ChunkGridField,
    ZarrV3ChunkKeyEncodingField,
    ZarrV3CodecField,
    ZarrV3DataTypeField,
    ZarrV3StaticCodecField,
    ZarrV3StorageTransformerField,
)
from zarr_metadata.v3._context import ZarrV3Context, ZarrV3ContextSlots
from zarr_metadata.v3._core import CORE, CORE_AND_EXTENSIONS
from zarr_metadata.v3._definition import (
    ArrayExtraFieldDefinition,
    Chunk,
    ChunkGridDefinition,
    ChunkKeyEncodingDefinition,
    CodecDefinition,
    CodecKind,
    CodecSize,
    DataTypeDefinition,
    ExtraFieldDefinition,
    GroupExtraFieldDefinition,
    Lengths,
    StorageClass,
    StorageTransformerDefinition,
    storage_of,
)
from zarr_metadata.v3._pipeline import Stage

__all__ = [
    "CORE",
    "CORE_AND_EXTENSIONS",
    "ArrayExtraFieldDefinition",
    "Chunk",
    "ChunkGridDefinition",
    "ChunkKeyEncodingDefinition",
    "CodecDefinition",
    "CodecKind",
    "CodecSize",
    "DataTypeDefinition",
    "ExtraFieldDefinition",
    "GroupExtraFieldDefinition",
    "Lengths",
    "Stage",
    "StorageClass",
    "StorageTransformerDefinition",
    "ZarrV3ChunkGridField",
    "ZarrV3ChunkKeyEncodingField",
    "ZarrV3CodecField",
    "ZarrV3Context",
    "ZarrV3ContextSlots",
    "ZarrV3DataTypeField",
    "ZarrV3StaticCodecField",
    "ZarrV3StorageTransformerField",
    "storage_of",
]
