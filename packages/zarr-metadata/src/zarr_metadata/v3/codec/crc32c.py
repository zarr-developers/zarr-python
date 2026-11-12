"""
CRC32C codec types.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/crc32c/index.html

The CRC32C codec has no configuration fields, so the `configuration`
key is absent from the metadata.
"""

from typing import Final

from zarr_metadata._common import EmptyConfiguration
from zarr_metadata.v3._definition import CodecDefinition
from zarr_metadata.v3._types.codec import (
    CRC32C_CODEC_NAME,
    Crc32cCodecMetadata,
    Crc32cCodecName,
    Crc32cCodecObject,
)

CRC32C_CODEC: Final = CodecDefinition(
    name=CRC32C_CODEC_NAME, configuration=EmptyConfiguration, kind="bytes_bytes", size="static"
)
"""The `crc32c` codec. A checksum has no configuration."""

__all__ = [
    "CRC32C_CODEC",
    "CRC32C_CODEC_NAME",
    "Crc32cCodecMetadata",
    "Crc32cCodecName",
    "Crc32cCodecObject",
]
