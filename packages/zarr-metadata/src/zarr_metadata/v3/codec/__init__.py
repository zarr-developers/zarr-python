"""
Zarr v3 codec spec types.

Each codec defined by the spec or by zarr-extensions has its own submodule
(`blosc`, `bytes`, `cast_value`, `crc32c`, `gzip`, `scale_offset`,
`sharding_indexed`, `transpose`, `zstd`).

The `<X>CodecMetadata` aliases re-exported here are the canonical type for
each codec's permitted JSON shapes (object form plus, where the spec allows,
a bare-string short-hand form). For the underlying `<X>CodecObject`,
`<X>CodecConfiguration`, etc., import directly from the leaf submodule.

For the JSON of any codec entry -- one of array metadata's `codecs` --
import `ZarrV3MetadataFieldJSON` from `zarr_metadata.v3`. A configuration
member holding a codec, as sharding's inner pipelines do, is annotated
`CodecField` from `zarr_metadata.v3.definition` instead, so it is read as
the codec it is.

Each codec's pipeline position (`array -> array`, `array -> bytes`,
`bytes -> bytes`) is its definition's `kind`, in
`zarr_metadata.v3.definition`.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/index.html
"""

from zarr_metadata.v3.codec.blosc import BloscCodecMetadata
from zarr_metadata.v3.codec.bytes import BytesCodecMetadata
from zarr_metadata.v3.codec.cast_value import CastValueCodecMetadata
from zarr_metadata.v3.codec.crc32c import Crc32cCodecMetadata
from zarr_metadata.v3.codec.gzip import GzipCodecMetadata
from zarr_metadata.v3.codec.scale_offset import ScaleOffsetCodecMetadata
from zarr_metadata.v3.codec.sharding_indexed import ShardingIndexedCodecMetadata
from zarr_metadata.v3.codec.transpose import TransposeCodecMetadata
from zarr_metadata.v3.codec.zstd import ZstdCodecMetadata

__all__ = [
    "BloscCodecMetadata",
    "BytesCodecMetadata",
    "CastValueCodecMetadata",
    "Crc32cCodecMetadata",
    "GzipCodecMetadata",
    "ScaleOffsetCodecMetadata",
    "ShardingIndexedCodecMetadata",
    "TransposeCodecMetadata",
    "ZstdCodecMetadata",
]
