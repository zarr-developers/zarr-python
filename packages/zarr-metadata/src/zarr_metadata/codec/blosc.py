"""
Blosc codec types.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/blosc/index.html
"""

from typing import Final, Literal, TypedDict

BLOSC_CODEC_NAME: Final = "blosc"
"""The ``name`` field value of the ``blosc`` codec."""

BloscCodecName = Literal["blosc"]
"""Literal type of the ``name`` field of the ``blosc`` codec."""

BLOSC_SHUFFLE_NOSHUFFLE: Final = "noshuffle"
BLOSC_SHUFFLE_SHUFFLE: Final = "shuffle"
BLOSC_SHUFFLE_BITSHUFFLE: Final = "bitshuffle"

BloscShuffle = Literal["noshuffle", "shuffle", "bitshuffle"]
"""Blosc shuffle mode names."""

BLOSC_CNAME_LZ4: Final = "lz4"
BLOSC_CNAME_LZ4HC: Final = "lz4hc"
BLOSC_CNAME_BLOSCLZ: Final = "blosclz"
BLOSC_CNAME_SNAPPY: Final = "snappy"
BLOSC_CNAME_ZLIB: Final = "zlib"
BLOSC_CNAME_ZSTD: Final = "zstd"

BloscCName = Literal["lz4", "lz4hc", "blosclz", "snappy", "zlib", "zstd"]
"""Blosc compressor identifiers."""


class BloscCodecConfiguration(TypedDict):
    """Configuration for the Zarr v3 ``blosc`` codec."""

    cname: BloscCName
    clevel: int
    shuffle: BloscShuffle
    blocksize: int
    typesize: int


class BloscCodec(TypedDict):
    """``blosc`` codec metadata."""

    name: BloscCodecName
    configuration: BloscCodecConfiguration


__all__ = [
    "BLOSC_CNAME_BLOSCLZ",
    "BLOSC_CNAME_LZ4",
    "BLOSC_CNAME_LZ4HC",
    "BLOSC_CNAME_SNAPPY",
    "BLOSC_CNAME_ZLIB",
    "BLOSC_CNAME_ZSTD",
    "BLOSC_CODEC_NAME",
    "BLOSC_SHUFFLE_BITSHUFFLE",
    "BLOSC_SHUFFLE_NOSHUFFLE",
    "BLOSC_SHUFFLE_SHUFFLE",
    "BloscCName",
    "BloscCodec",
    "BloscCodecConfiguration",
    "BloscCodecName",
    "BloscShuffle",
]
