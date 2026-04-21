"""
Transpose codec types.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/transpose/index.html
"""

from typing import Final, Literal, TypedDict

TRANSPOSE_CODEC_NAME: Final = "transpose"
"""The ``name`` field value of a ``transpose`` codec envelope."""

TransposeCodecName = Literal["transpose"]
"""Literal type of the ``name`` field of a ``transpose`` codec envelope."""


class TransposeCodecConfiguration(TypedDict):
    """
    Configuration for the Zarr v3 ``transpose`` codec.

    ``order`` is a permutation of the dimension indices 0..n-1 that
    specifies the dimension reordering applied during encoding.
    """

    order: tuple[int, ...]


class TransposeCodec(TypedDict):
    """Full ``transpose`` codec named-config envelope."""

    name: TransposeCodecName
    configuration: TransposeCodecConfiguration


__all__ = [
    "TRANSPOSE_CODEC_NAME",
    "TransposeCodec",
    "TransposeCodecConfiguration",
    "TransposeCodecName",
]
