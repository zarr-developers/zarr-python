"""The extension layer: what an entity is, and what is in scope.

Every Zarr v3 extension point -- codecs, data types, chunk grids, chunk
key encodings, storage transformers -- is modelled as a class that
answers for itself. This module is the public door to that layer, for two
kinds of caller.

**Reading metadata.** The concrete entities know things a document does
not spell out: what a data type's scalars are, which position a codec
occupies in the pipeline, what a chunk grid divides an array into. Reach
them through a scope:

    from zarr_metadata.v3.entity import CORE_AND_EXTENSIONS

    data_type, problems = CORE_AND_EXTENSIONS.coerce("data_type", "int32")
    data_type.storage_class()        # 'multi_byte'

**Writing an extension.** Subclass `CodecEntity`, `DataTypeEntity`,
`ChunkGridEntity` or `MetadataEntity`, declare `identifier` and
`member_types`, and put it in a `Context`:

    @dataclass(frozen=True)
    class AcmeLz4Codec(CodecEntity):
        acceleration: int = 1

        identifier: ClassVar[str] = "acme.lz4"
        kind: ClassVar[CodecKind] = "bytes_bytes"
        member_types: ClassVar[MemberTypes] = {"acceleration": (False, is_int)}

    SCOPE = Context({**CORE_AND_EXTENSIONS.entities,
                     "codecs": {**CORE_AND_EXTENSIONS.entities["codecs"],
                                AcmeLz4Codec.identifier: AcmeLz4Codec}})

    validate_array_metadata_v3(document, context=SCOPE)

A name in no scope is not rejected -- that is what extension openness
means -- so registering yours is how you get it judged rather than waved
through.

One known friction, under mypy only. An entity's `to_json` returns its own
object TypedDict, and mypy does not accept that where a
`ZarrV3MetadataFieldJSON` is wanted: it reads every TypedDict as
`Mapping[str, object]`, never as the `Mapping[str, JSONValue]` the
envelope declares. So putting `to_json()` output straight into a `codecs`
list needs a `cast` under mypy. Pyright accepts it.

That conversion is sound here, which is why pyright is the one that is
right. The rule mypy is applying exists because an ordinary TypedDict may
carry extra items of types it never declared, so the union of the
declared value types does not bound what is in the mapping. Every
TypedDict in this package is `closed` (PEP 728), which forbids exactly
that, and pyright implements PEP 728. Mypy does not yet -- see
python/mypy#8994 and python/mypy#18439.

The type therefore stays as it is. Widening `configuration` to
`Mapping[str, object]` or `Mapping[str, Any]` would satisfy mypy by
making the annotation say something false: a configuration's values are
JSON, and that is worth more than one checker's `cast`.
"""

from __future__ import annotations

from zarr_metadata.v3._chain import chain_problems, order_problems
from zarr_metadata.v3._document import ArrayDocumentV3, array_problems_v3, read_array_v3
from zarr_metadata.v3._entity import (
    CHUNK_GRID,
    CHUNK_KEY_ENCODING,
    CODECS,
    DATA_TYPE,
    STORAGE_TRANSFORMERS,
    ChunkGridEntity,
    CodecEntity,
    CodecKind,
    Coerced,
    DataTypeEntity,
    ExtensionPointField,
    Loc,
    MemberTypes,
    MetadataEntity,
    StorageClass,
    TypeCheck,
    coerce_members,
    is_bool,
    is_int,
    is_integer,
    is_json_value,
    is_str,
    named_configuration,
    one_of,
    problem,
    sequence_of,
    within,
)
from zarr_metadata.v3._parts import UNKNOWN_GRID, ArrayParts, ChunkGrid, Extents, shard_index_grid
from zarr_metadata.v3._registry import CORE, CORE_AND_EXTENSIONS, Context
from zarr_metadata.v3.data_type._families import (
    FLOAT_SPECIALS,
    ComplexDataType,
    FloatDataType,
    IntegerDataType,
    NumpyTimeDataType,
    as_sequence,
    byte_values,
)

__all__ = [
    "CHUNK_GRID",
    "CHUNK_KEY_ENCODING",
    "CODECS",
    "CORE",
    "CORE_AND_EXTENSIONS",
    "DATA_TYPE",
    "FLOAT_SPECIALS",
    "STORAGE_TRANSFORMERS",
    "UNKNOWN_GRID",
    "ArrayDocumentV3",
    "ArrayParts",
    "ChunkGrid",
    "ChunkGridEntity",
    "CodecEntity",
    "CodecKind",
    "Coerced",
    "ComplexDataType",
    "Context",
    "DataTypeEntity",
    "ExtensionPointField",
    "Extents",
    "FloatDataType",
    "IntegerDataType",
    "Loc",
    "MemberTypes",
    "MetadataEntity",
    "NumpyTimeDataType",
    "StorageClass",
    "TypeCheck",
    "array_problems_v3",
    "as_sequence",
    "byte_values",
    "chain_problems",
    "coerce_members",
    "is_bool",
    "is_int",
    "is_integer",
    "is_json_value",
    "is_str",
    "named_configuration",
    "one_of",
    "order_problems",
    "problem",
    "read_array_v3",
    "sequence_of",
    "shard_index_grid",
    "within",
]
