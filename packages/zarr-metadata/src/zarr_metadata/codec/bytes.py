"""
Bytes codec types.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/bytes/index.html
"""

from typing import Literal, NotRequired, TypedDict

BytesCodecName = Literal["bytes"]
"""The ``name`` field value of a ``bytes`` codec envelope."""


class BytesCodecConfiguration(TypedDict):
    """
    Configuration for the Zarr v3 ``bytes`` codec.

    The ``endian`` field is required for multi-byte data types and absent
    for single-byte types. Consumers that always expect a value must
    tolerate its absence.
    """

    endian: NotRequired[Literal["little", "big"]]


class BytesCodec(TypedDict):
    """Full ``bytes`` codec named-config envelope."""

    name: BytesCodecName
    configuration: BytesCodecConfiguration


__all__ = [
    "BytesCodec",
    "BytesCodecConfiguration",
    "BytesCodecName",
]
