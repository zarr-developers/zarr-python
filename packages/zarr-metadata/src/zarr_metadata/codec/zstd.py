"""
Zstandard codec types.

See https://github.com/zarr-developers/zarr-specs/pull/256 (unmerged at
time of writing; the configuration shape below reflects the proposed
specification).
"""

from typing import Final, Literal, TypedDict

ZSTD_CODEC_NAME: Final = "zstd"
"""The ``name`` field value of a ``zstd`` codec envelope."""

ZstdCodecName = Literal["zstd"]
"""Literal type of the ``name`` field of a ``zstd`` codec envelope."""


class ZstdCodecConfiguration(TypedDict):
    """
    Configuration for the Zarr v3 ``zstd`` codec.

    Both fields are required per the proposed specification.
    """

    level: int
    checksum: bool


class ZstdCodec(TypedDict):
    """Full ``zstd`` codec named-config envelope."""

    name: ZstdCodecName
    configuration: ZstdCodecConfiguration


__all__ = [
    "ZSTD_CODEC_NAME",
    "ZstdCodec",
    "ZstdCodecConfiguration",
    "ZstdCodecName",
]
