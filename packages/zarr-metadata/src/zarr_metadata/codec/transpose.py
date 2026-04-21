"""
Transpose codec configuration.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/transpose/index.html
"""

from typing import Literal, TypedDict

from zarr_metadata.common import NamedRequiredConfig

TransposeCodecName = Literal["transpose"]
"""The ``name`` field value of a ``transpose`` codec envelope."""


class TransposeCodecConfiguration(TypedDict):
    """
    Configuration for the Zarr v3 ``transpose`` codec.

    `order` is a permutation of the dimension indices 0..n-1 that
    specifies the dimension reordering applied during encoding.
    """

    order: tuple[int, ...]


TransposeCodec = NamedRequiredConfig[TransposeCodecName, TransposeCodecConfiguration]
"""Full ``transpose`` codec named-config envelope."""


__all__ = [
    "TransposeCodec",
    "TransposeCodecConfiguration",
    "TransposeCodecName",
]
