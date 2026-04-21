"""
Transpose codec types.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/transpose/index.html
"""

from typing import Final, Literal, TypedDict

TRANSPOSE_CODEC_NAME: Final = "transpose"
"""The `name` field value of the `transpose` codec."""

TransposeCodecName = Literal["transpose"]
"""Literal type of the `name` field of the `transpose` codec."""


class TransposeCodecConfiguration(TypedDict):
    """
    Configuration for the Zarr v3 `transpose` codec.

    `order` is a permutation of the dimension indices 0..n-1 that
    specifies the dimension reordering applied during encoding.
    """

    order: tuple[int, ...]


class TransposeCodec(TypedDict):
    """`transpose` codec metadata."""

    name: TransposeCodecName
    configuration: TransposeCodecConfiguration


__all__ = [
    "TRANSPOSE_CODEC_NAME",
    "TransposeCodec",
    "TransposeCodecConfiguration",
    "TransposeCodecName",
]
