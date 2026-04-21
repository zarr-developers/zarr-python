"""
Bytes codec types.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/bytes/index.html
"""

from typing import Final, Literal, NotRequired, TypedDict

BYTES_CODEC_NAME: Final = "bytes"
"""The `name` field value of the `bytes` codec."""

BytesCodecName = Literal["bytes"]
"""Literal type of the `name` field of the `bytes` codec."""

BYTES_ENDIAN_LITTLE: Final = "little"
BYTES_ENDIAN_BIG: Final = "big"

Endian = Literal["little", "big"]
"""Byte order of multi-byte numeric data."""


class BytesCodecConfiguration(TypedDict):
    """
    Configuration for the Zarr v3 `bytes` codec.

    The `endian` field is required for multi-byte data types and absent
    for single-byte types. Consumers that always expect a value must
    tolerate its absence.
    """

    endian: NotRequired[Endian]


class BytesCodec(TypedDict):
    """`bytes` codec metadata."""

    name: BytesCodecName
    configuration: BytesCodecConfiguration


__all__ = [
    "BYTES_CODEC_NAME",
    "BYTES_ENDIAN_BIG",
    "BYTES_ENDIAN_LITTLE",
    "BytesCodec",
    "BytesCodecConfiguration",
    "BytesCodecName",
    "Endian",
]
