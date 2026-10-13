"""
CRC32C codec types.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/crc32c/index.html

The CRC32C codec has no configuration fields, so the `configuration`
key is absent from the metadata.
"""

from typing import Final, Literal, NotRequired

from typing_extensions import TypedDict

from zarr_metadata.v3._definition import CodecDefinition

CRC32C_CODEC_NAME: Final = "crc32c"
"""The `name` field value of the `crc32c` codec."""

Crc32cCodecName = Literal["crc32c"]
"""Literal type of the `name` field of the `crc32c` codec."""


class Empty(TypedDict, closed=True):
    """An empty mapping"""


class Crc32cCodecObject(TypedDict, closed=True):
    """`crc32c` codec metadata in object form.

    Per spec the codec has no configuration fields. `configuration` is
    optional and, if present, should be an empty mapping.
      https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/codecs/crc32c/index.rst#L63-L66
    """

    name: Crc32cCodecName
    configuration: NotRequired[Empty]
    must_understand: NotRequired[bool]


Crc32cCodecMetadata = Crc32cCodecObject | Crc32cCodecName
"""Permitted JSON shapes for `crc32c` codec metadata.

The spec's Extension definition allows extensions with no required
configuration to be encoded as a bare short-hand name. CRC32C has no
configuration, so both forms are valid.
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1562-L1564
"""


CRC32C_CODEC: Final = CodecDefinition(
    name=CRC32C_CODEC_NAME, configuration=Empty, kind="bytes_bytes", size="static"
)
"""The `crc32c` codec: a checksum has nothing to configure."""

__all__ = [
    "CRC32C_CODEC",
    "CRC32C_CODEC_NAME",
    "Crc32cCodecMetadata",
    "Crc32cCodecName",
    "Crc32cCodecObject",
]
