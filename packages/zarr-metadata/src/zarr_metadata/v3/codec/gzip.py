"""
Gzip codec types.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/gzip/index.html
"""

from typing import Final

from zarr_metadata.v3._definition import BytesBytesCodecDefinition
from zarr_metadata.v3._types.codec import (
    GZIP_CODEC_NAME,
    GzipCodecConfiguration,
    GzipCodecMetadata,
    GzipCodecName,
    GzipCodecObject,
)

GZIP_CODEC: Final = BytesBytesCodecDefinition(
    name=GZIP_CODEC_NAME,
    configuration=GzipCodecConfiguration,
    size="dynamic",
)
"""The `gzip` codec."""

__all__ = [
    "GZIP_CODEC",
    "GZIP_CODEC_NAME",
    "GzipCodecConfiguration",
    "GzipCodecMetadata",
    "GzipCodecName",
    "GzipCodecObject",
]
