"""
Sharding codec configuration.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/index.html
"""

from typing import Literal, NotRequired, TypedDict

from zarr_metadata.codec import Codec
from zarr_metadata.common import NamedRequiredConfig

ShardingCodecName = Literal["sharding_indexed"]
"""The ``name`` field value of a ``sharding_indexed`` codec envelope."""


class ShardingCodecConfiguration(TypedDict):
    """
    Configuration for the Zarr v3 ``sharding_indexed`` codec.

    `chunk_shape` is the shape of inner chunks along each dimension;
    it must evenly divide the shard shape.

    `codecs` is the codec pipeline applied to each inner chunk; exactly
    one array-to-bytes codec is required.

    `index_codecs` is the codec pipeline applied to the shard index;
    it must be deterministic (no variable-size compression).

    `index_location` defaults to ``"end"`` per the spec.
    """

    chunk_shape: tuple[int, ...]
    codecs: tuple[Codec, ...]
    index_codecs: tuple[Codec, ...]
    index_location: NotRequired[Literal["start", "end"]]


ShardingCodec = NamedRequiredConfig[ShardingCodecName, ShardingCodecConfiguration]
"""Full ``sharding_indexed`` codec named-config envelope."""


__all__ = [
    "ShardingCodec",
    "ShardingCodecConfiguration",
    "ShardingCodecName",
]
