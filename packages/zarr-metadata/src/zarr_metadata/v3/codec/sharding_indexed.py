"""
Sharding-indexed codec types.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/index.html
"""

from typing import Final, Literal, NotRequired

from typing_extensions import TypedDict

from zarr_metadata.v3._common import ZarrV3MetadataFieldJSON

SHARDING_INDEXED_CODEC_NAME: Final = "sharding_indexed"
"""The `name` field value of the `sharding_indexed` codec."""

ShardingIndexedCodecName = Literal["sharding_indexed"]
"""Literal type of the `name` field of the `sharding_indexed` codec."""

ShardingIndexLocation = Literal["start", "end"]
"""Literal type of the position of the shard index within the encoded shard."""

SHARDING_INDEX_LOCATION: Final = ("start", "end")
"""Tuple of permitted values for the `index_location` field of the `sharding_indexed` codec."""


class ShardingIndexedCodecConfiguration(TypedDict):
    """
    Configuration for the Zarr v3 `sharding_indexed` codec.

    `chunk_shape` is the shape of inner chunks along each dimension;
    it must evenly divide the shard shape.

    `codecs` is the codec pipeline applied to each inner chunk; exactly
    one array-to-bytes codec is required.

    `index_codecs` is the codec pipeline applied to the shard index;
    it must be deterministic (no variable-size compression).

    `index_location` defaults to `"end"` per the spec.
    """

    chunk_shape: tuple[int, ...]
    codecs: tuple[ZarrV3MetadataFieldJSON, ...]
    index_codecs: tuple[ZarrV3MetadataFieldJSON, ...]
    index_location: NotRequired[ShardingIndexLocation]


class ShardingIndexedCodecObject(TypedDict):
    """`sharding_indexed` codec metadata in object form."""

    name: ShardingIndexedCodecName
    configuration: ShardingIndexedCodecConfiguration


ShardingIndexedCodecMetadata = ShardingIndexedCodecObject
"""Permitted JSON shape for `sharding_indexed` codec metadata.

The configuration has multiple required keys (`chunk_shape`, `codecs`,
`index_codecs`), so only the object form is valid; the short-hand-name
form is not permitted by the spec for this codec.
"""

__all__ = [
    "SHARDING_INDEXED_CODEC_NAME",
    "SHARDING_INDEX_LOCATION",
    "ShardingIndexLocation",
    "ShardingIndexedCodecConfiguration",
    "ShardingIndexedCodecMetadata",
    "ShardingIndexedCodecName",
    "ShardingIndexedCodecObject",
]
