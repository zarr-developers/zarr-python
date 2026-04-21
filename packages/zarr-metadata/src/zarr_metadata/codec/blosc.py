"""
Blosc codec configuration types (Zarr v3 spec + numcodecs/v2 form).
"""

from typing import Literal, NotRequired, TypedDict

from zarr_metadata.common import NamedRequiredConfig

BloscCodecName = Literal["blosc"]
"""The ``name`` field value of a ``blosc`` codec envelope."""

Shuffle = Literal["noshuffle", "shuffle", "bitshuffle"]
"""Blosc shuffle mode names (v3 spec)."""

CName = Literal["lz4", "lz4hc", "blosclz", "snappy", "zlib", "zstd"]
"""Blosc compressor identifiers."""


class BloscCodecConfigurationNumcodecs(TypedDict):
    """
    Blosc configuration for Zarr v2 / numcodecs-flavored callers.

    ``shuffle`` is an integer code (the numcodecs convention) rather than
    a named literal.
    """

    cname: CName
    clevel: int
    shuffle: int
    blocksize: int
    typesize: NotRequired[int]


class BloscCodecConfigurationV1(TypedDict):
    """
    Blosc configuration for Zarr v3 spec (version 1 of the blosc codec).

    ``shuffle`` is a named string literal.
    """

    cname: CName
    clevel: int
    shuffle: Shuffle
    blocksize: int
    typesize: int


BloscCodecConfiguration = BloscCodecConfigurationV1 | BloscCodecConfigurationNumcodecs
"""Any supported blosc configuration shape."""

BloscCodec = NamedRequiredConfig[BloscCodecName, BloscCodecConfiguration]
"""Full ``blosc`` codec named-config envelope."""


__all__ = [
    "BloscCodec",
    "BloscCodecConfiguration",
    "BloscCodecConfigurationNumcodecs",
    "BloscCodecConfigurationV1",
    "BloscCodecName",
    "CName",
    "Shuffle",
]
