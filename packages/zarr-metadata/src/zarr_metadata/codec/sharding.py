"""
Sharding codec types.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/index.html
"""

from typing import Final, Literal, NotRequired, TypedDict

from zarr_metadata.codec import Codec

SHARDING_CODEC_NAME: Final = "sharding_indexed"
"""The `name` field value of the `sharding_indexed` codec."""

ShardingCodecName = Literal["sharding_indexed"]
"""Literal type of the `name` field of the `sharding_indexed` codec."""

SHARDING_INDEX_LOCATION_START: Final = "start"
SHARDING_INDEX_LOCATION_END: Final = "end"

IndexLocation = Literal["start", "end"]
"""Position of the shard index within the encoded shard."""


class ShardingCodecConfiguration(TypedDict):
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
    codecs: tuple[Codec, ...]
    index_codecs: tuple[Codec, ...]
    index_location: NotRequired[IndexLocation]


class ShardingCodec(TypedDict):
    """`sharding_indexed` codec metadata."""

    name: ShardingCodecName
    configuration: ShardingCodecConfiguration


__all__ = [
    "SHARDING_CODEC_NAME",
    "SHARDING_INDEX_LOCATION_END",
    "SHARDING_INDEX_LOCATION_START",
    "IndexLocation",
    "ShardingCodec",
    "ShardingCodecConfiguration",
    "ShardingCodecName",
]
