"""
CRC32C codec types.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/crc32c/index.html

The CRC32C codec has no configuration fields, so the ``configuration``
key is absent from the metadata.
"""

from typing import Final, Literal, NotRequired, TypedDict

from zarr_metadata.common import JSON

CRC32C_CODEC_NAME: Final = "crc32c"
"""The ``name`` field value of the ``crc32c`` codec."""

Crc32cCodecName = Literal["crc32c"]
"""Literal type of the ``name`` field of the ``crc32c`` codec."""


class Crc32cCodec(TypedDict):
    """``crc32c`` codec metadata.

    Per spec the codec has no configuration fields. ``configuration`` is
    optional and, if present, should be an empty mapping.
    """

    name: Crc32cCodecName
    configuration: NotRequired[dict[str, JSON]]


__all__ = [
    "CRC32C_CODEC_NAME",
    "Crc32cCodec",
    "Crc32cCodecName",
]
