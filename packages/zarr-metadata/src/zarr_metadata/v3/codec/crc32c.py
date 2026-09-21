"""
CRC32C codec types.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/crc32c/index.html

The CRC32C codec has no configuration fields, so the `configuration`
key is absent from the metadata.
"""

from dataclasses import dataclass
from typing import ClassVar, Final, Literal, NotRequired

from typing_extensions import TypedDict

from zarr_metadata.v3._entity import (
    BytesBytesCodec,
)

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


__all__ = [
    "CRC32C_CODEC_NAME",
    "Crc32cCodec",
    "Crc32cCodecMetadata",
    "Crc32cCodecName",
    "Crc32cCodecObject",
]


@dataclass(frozen=True)
class Crc32cCodec(BytesBytesCodec):
    """The `crc32c` codec, coerced from its metadata.

    The name says everything: a checksum has nothing to configure.
    """

    identifier: ClassVar[str] = CRC32C_CODEC_NAME

    def to_json(self) -> Crc32cCodecName:
        return "crc32c"
