"""
Bytes codec types.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/bytes/index.html
"""

from typing import Final, Literal, NotRequired, TypedDict

BYTES_CODEC_NAME: Final = "bytes"
"""The `name` field value of the `bytes` codec."""

BytesCodecName = Literal["bytes"]
"""Literal type of the `name` field of the `bytes` codec."""

Endian = Literal["little", "big"]
"""Byte order of multi-byte numeric data."""


class BytesCodecConfiguration(TypedDict):
    """
    Configuration for the Zarr v3 `bytes` codec.

    The `endian` field is required for multi-byte data types.
    """

    endian: NotRequired[Endian]


class BytesCodecObject(TypedDict):
    """`bytes` codec metadata in object form."""

    name: BytesCodecName
    configuration: BytesCodecConfiguration


BytesCodecMetadata = BytesCodecObject | BytesCodecName
"""Permitted JSON shapes for `bytes` codec metadata.

The configuration has no required keys (`endian` is conditionally required
at runtime based on data type), so the spec's short-hand-name form is
permitted in addition to the object form.
"""


__all__ = [
    "BYTES_CODEC_NAME",
    "BytesCodecConfiguration",
    "BytesCodecMetadata",
    "BytesCodecName",
    "BytesCodecObject",
    "Endian",
]
