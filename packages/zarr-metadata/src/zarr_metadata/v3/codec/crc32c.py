"""
CRC32C codec types.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/crc32c/index.html

The CRC32C codec has no configuration fields, so the `configuration`
key is absent from the metadata.
"""

from typing import Final, Literal, NotRequired

from typing_extensions import TypedDict

CRC32C_CODEC_NAME: Final = "crc32c"
"""The `name` field value of the `crc32c` codec."""

Crc32cCodecName = Literal["crc32c"]
"""Literal type of the `name` field of the `crc32c` codec."""


class Empty(TypedDict, closed=True):  # type: ignore[call-arg]
    """An empty mapping"""


class Crc32cCodecObject(TypedDict):
    """`crc32c` codec metadata in object form.

    Per spec the codec has no configuration fields. `configuration` is
    optional and, if present, should be an empty mapping.
    """

    name: Crc32cCodecName
    configuration: NotRequired[Empty]


Crc32cCodecMetadata = Crc32cCodecObject | Crc32cCodecName
"""Permitted JSON shapes for `crc32c` codec metadata.

The spec's Extension definition allows extensions with no required
configuration to be encoded as a bare short-hand name. CRC32C has no
configuration, so both forms are valid.
"""


__all__ = [
    "CRC32C_CODEC_NAME",
    "Crc32cCodecMetadata",
    "Crc32cCodecName",
    "Crc32cCodecObject",
]
