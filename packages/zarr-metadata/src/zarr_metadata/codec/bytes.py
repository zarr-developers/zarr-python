"""
Bytes codec configuration.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/bytes/index.html
"""

from typing import Literal, NotRequired, TypedDict

from zarr_metadata.common import NamedRequiredConfig

BytesCodecName = Literal["bytes"]
"""The ``name`` field value of a ``bytes`` codec envelope."""


class BytesCodecConfiguration(TypedDict):
    """
    Configuration for the Zarr v3 ``bytes`` codec.

    The `endian` field is required for multi-byte data types and absent
    for single-byte types. Consumers that always expect a value must
    tolerate its absence.
    """

    endian: NotRequired[Literal["little", "big"]]


BytesCodec = NamedRequiredConfig[BytesCodecName, BytesCodecConfiguration]
"""Full ``bytes`` codec named-config envelope."""


__all__ = [
    "BytesCodec",
    "BytesCodecConfiguration",
    "BytesCodecName",
]
