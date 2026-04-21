"""
Blosc codec types.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/blosc/index.html
"""

from typing import Literal, TypedDict

BloscCodecName = Literal["blosc"]
"""The ``name`` field value of a ``blosc`` codec envelope."""

Shuffle = Literal["noshuffle", "shuffle", "bitshuffle"]
"""Blosc shuffle mode names."""

CName = Literal["lz4", "lz4hc", "blosclz", "snappy", "zlib", "zstd"]
"""Blosc compressor identifiers."""


class BloscCodecConfiguration(TypedDict):
    """Configuration for the Zarr v3 ``blosc`` codec."""

    cname: CName
    clevel: int
    shuffle: Shuffle
    blocksize: int
    typesize: int


class BloscCodec(TypedDict):
    """Full ``blosc`` codec named-config envelope."""

    name: BloscCodecName
    configuration: BloscCodecConfiguration


__all__ = [
    "BloscCodec",
    "BloscCodecConfiguration",
    "BloscCodecName",
    "CName",
    "Shuffle",
]
