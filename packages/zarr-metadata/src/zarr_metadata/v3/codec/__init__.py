"""Zarr v3 codec spec types.

Each codec defined by the spec or by zarr-extensions has its own submodule
(`blosc`, `bytes`, `cast_value`, `crc32c`, `gzip`, `scale_offset`,
`sharding_indexed`, `transpose`, `vlen_bytes`, `vlen_utf8`, `zstd`). Import a
codec's types and its definition from its submodule, or from this package,
which re-exports every name its submodules export: `<X>CodecMetadata`, the
canonical type for the codec's permitted JSON shapes (object form plus,
where the spec allows, a bare-string short-hand form), `<X>CodecObject`,
`<X>_CODEC`, and, for a codec that has a configuration,
`<X>CodecConfiguration`.

For the JSON of any codec entry (one of the entries in array metadata's
`codecs`), import `ZarrV3MetadataFieldJSON` from `zarr_metadata.v3`. A
configuration member that holds a codec, such as sharding's inner
pipelines, is annotated `ZarrV3CodecField` from
`zarr_metadata.v3.definition` instead, so that the reader reads it as a
codec.

Each codec's pipeline position (`array -> array`, `array -> bytes`,
`bytes -> bytes`) is its definition's `kind`, in
`zarr_metadata.v3.definition`.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/index.html
"""

from zarr_metadata.v3.codec.blosc import (
    BLOSC_CNAME,
    BLOSC_CODEC,
    BLOSC_CODEC_NAME,
    BLOSC_NO_SHUFFLE,
    BLOSC_SHUFFLE,
    BloscCName,
    BloscCodecConfiguration,
    BloscCodecMetadata,
    BloscCodecName,
    BloscCodecObject,
    BloscShuffle,
)
from zarr_metadata.v3.codec.bytes import (
    BYTES_CODEC,
    BYTES_CODEC_NAME,
    ENDIANNESS,
    BytesCodecConfiguration,
    BytesCodecMetadata,
    BytesCodecName,
    BytesCodecObject,
    Endianness,
)
from zarr_metadata.v3.codec.cast_value import (
    CAST_OUT_OF_RANGE_MODE,
    CAST_ROUNDING_MODE,
    CAST_VALUE_CODEC,
    CAST_VALUE_CODEC_NAME,
    CastOutOfRangeMode,
    CastRoundingMode,
    CastValueCodecConfiguration,
    CastValueCodecMetadata,
    CastValueCodecName,
    CastValueCodecObject,
    ScalarMap,
    ScalarMapEntry,
)
from zarr_metadata.v3.codec.crc32c import (
    CRC32C_CODEC,
    CRC32C_CODEC_NAME,
    Crc32cCodecMetadata,
    Crc32cCodecName,
    Crc32cCodecObject,
)
from zarr_metadata.v3.codec.gzip import (
    GZIP_CODEC,
    GZIP_CODEC_NAME,
    GzipCodecConfiguration,
    GzipCodecMetadata,
    GzipCodecName,
    GzipCodecObject,
)
from zarr_metadata.v3.codec.scale_offset import (
    SCALE_OFFSET_CODEC,
    SCALE_OFFSET_CODEC_NAME,
    ScaleOffsetCodecConfiguration,
    ScaleOffsetCodecMetadata,
    ScaleOffsetCodecName,
    ScaleOffsetCodecObject,
)
from zarr_metadata.v3.codec.sharding_indexed import (
    SHARDING_INDEX_LOCATION,
    SHARDING_INDEXED_CODEC,
    SHARDING_INDEXED_CODEC_NAME,
    ShardingIndexedCodecConfiguration,
    ShardingIndexedCodecMetadata,
    ShardingIndexedCodecName,
    ShardingIndexedCodecObject,
    ShardingIndexLocation,
)
from zarr_metadata.v3.codec.transpose import (
    TRANSPOSE_CODEC,
    TRANSPOSE_CODEC_NAME,
    TransposeCodecConfiguration,
    TransposeCodecMetadata,
    TransposeCodecName,
    TransposeCodecObject,
)
from zarr_metadata.v3.codec.vlen_bytes import (
    VLEN_BYTES_CODEC,
    VLEN_BYTES_CODEC_NAME,
    VlenBytesCodecMetadata,
    VlenBytesCodecName,
    VlenBytesCodecObject,
)
from zarr_metadata.v3.codec.vlen_utf8 import (
    VLEN_UTF8_CODEC,
    VLEN_UTF8_CODEC_NAME,
    VlenUtf8CodecMetadata,
    VlenUtf8CodecName,
    VlenUtf8CodecObject,
)
from zarr_metadata.v3.codec.zstd import (
    ZSTD_CODEC,
    ZSTD_CODEC_NAME,
    ZSTD_MAX_LEVEL,
    ZSTD_MIN_LEVEL,
    ZstdCodecConfiguration,
    ZstdCodecMetadata,
    ZstdCodecName,
    ZstdCodecObject,
)

__all__ = [
    "BLOSC_CNAME",
    "BLOSC_CODEC",
    "BLOSC_CODEC_NAME",
    "BLOSC_NO_SHUFFLE",
    "BLOSC_SHUFFLE",
    "BYTES_CODEC",
    "BYTES_CODEC_NAME",
    "CAST_OUT_OF_RANGE_MODE",
    "CAST_ROUNDING_MODE",
    "CAST_VALUE_CODEC",
    "CAST_VALUE_CODEC_NAME",
    "CRC32C_CODEC",
    "CRC32C_CODEC_NAME",
    "ENDIANNESS",
    "GZIP_CODEC",
    "GZIP_CODEC_NAME",
    "SCALE_OFFSET_CODEC",
    "SCALE_OFFSET_CODEC_NAME",
    "SHARDING_INDEXED_CODEC",
    "SHARDING_INDEXED_CODEC_NAME",
    "SHARDING_INDEX_LOCATION",
    "TRANSPOSE_CODEC",
    "TRANSPOSE_CODEC_NAME",
    "VLEN_BYTES_CODEC",
    "VLEN_BYTES_CODEC_NAME",
    "VLEN_UTF8_CODEC",
    "VLEN_UTF8_CODEC_NAME",
    "ZSTD_CODEC",
    "ZSTD_CODEC_NAME",
    "ZSTD_MAX_LEVEL",
    "ZSTD_MIN_LEVEL",
    "BloscCName",
    "BloscCodecConfiguration",
    "BloscCodecMetadata",
    "BloscCodecName",
    "BloscCodecObject",
    "BloscShuffle",
    "BytesCodecConfiguration",
    "BytesCodecMetadata",
    "BytesCodecName",
    "BytesCodecObject",
    "CastOutOfRangeMode",
    "CastRoundingMode",
    "CastValueCodecConfiguration",
    "CastValueCodecMetadata",
    "CastValueCodecName",
    "CastValueCodecObject",
    "Crc32cCodecMetadata",
    "Crc32cCodecName",
    "Crc32cCodecObject",
    "Endianness",
    "GzipCodecConfiguration",
    "GzipCodecMetadata",
    "GzipCodecName",
    "GzipCodecObject",
    "ScalarMap",
    "ScalarMapEntry",
    "ScaleOffsetCodecConfiguration",
    "ScaleOffsetCodecMetadata",
    "ScaleOffsetCodecName",
    "ScaleOffsetCodecObject",
    "ShardingIndexLocation",
    "ShardingIndexedCodecConfiguration",
    "ShardingIndexedCodecMetadata",
    "ShardingIndexedCodecName",
    "ShardingIndexedCodecObject",
    "TransposeCodecConfiguration",
    "TransposeCodecMetadata",
    "TransposeCodecName",
    "TransposeCodecObject",
    "VlenBytesCodecMetadata",
    "VlenBytesCodecName",
    "VlenBytesCodecObject",
    "VlenUtf8CodecMetadata",
    "VlenUtf8CodecName",
    "VlenUtf8CodecObject",
    "ZstdCodecConfiguration",
    "ZstdCodecMetadata",
    "ZstdCodecName",
    "ZstdCodecObject",
]
