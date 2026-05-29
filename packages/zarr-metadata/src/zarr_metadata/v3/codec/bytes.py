"""
Bytes codec types.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/bytes/index.html
"""

from typing import Final, Literal, NotRequired

from typing_extensions import TypedDict

BYTES_CODEC_NAME: Final = "bytes"
"""The `name` field value of the `bytes` codec."""

BytesCodecName = Literal["bytes"]
"""Literal type of the `name` field of the `bytes` codec."""

Endian = Literal["little", "big"]
"""Literal type of byte order of multi-byte numeric data."""

ENDIAN: Final = ("little", "big")
"""Tuple of permitted values for the `endian` field of the `bytes` codec."""


class BytesCodecConfiguration(TypedDict):
    """
    Configuration for the Zarr v3 `bytes` codec.

    The `endian` field is required for multi-byte data types.
    """

    endian: NotRequired[Endian]


class BytesCodecObject(TypedDict):
    """`bytes` codec metadata in object form.

    `configuration` is itself optional — when no configuration fields are
    set, the entire `configuration` key may be omitted. This matches the
    bare-string short-hand form (`BytesCodecName`) at the canonical data
    level; both encodings describe a `bytes` codec with default settings.
    """

    name: BytesCodecName
    configuration: NotRequired[BytesCodecConfiguration]


BytesCodecMetadata = BytesCodecObject | BytesCodecName
"""Permitted JSON shapes for `bytes` codec metadata.

The configuration has no required keys (`endian` is conditionally required
at runtime based on data type), so the spec's short-hand-name form is
permitted in addition to the object form, and the object form may itself
omit `configuration` entirely.
"""

__all__ = [
    "BYTES_CODEC_NAME",
    "ENDIAN",
    "BytesCodecConfiguration",
    "BytesCodecMetadata",
    "BytesCodecName",
    "BytesCodecObject",
    "Endian",
]
