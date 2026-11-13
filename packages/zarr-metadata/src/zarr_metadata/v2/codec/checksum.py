"""The v2 checksums that numcodecs 0.16 configures: `crc32`, `crc32c`, `adler32` and `fletcher32`."""

from __future__ import annotations

from typing import Final, Literal

from zarr_metadata._common import EmptyConfiguration
from zarr_metadata.v2._definition import ZarrV2CodecDefinition
from zarr_metadata.v2._types.codec import ZarrV2Checksum32Parameters


def _fill_location(
    configuration: ZarrV2Checksum32Parameters, default: Literal["start", "end"]
) -> ZarrV2Checksum32Parameters:
    """Fill in a `location` that is omitted or null with `default`."""
    location = configuration.get("location")
    return {"location": default if location is None else location}


def _checksum_canonical(configuration: ZarrV2Checksum32Parameters) -> ZarrV2Checksum32Parameters:
    """Fill in a `location` that is omitted or null with `"start"`, the default of `numcodecs.CRC32` and `numcodecs.Adler32`."""
    return _fill_location(configuration, "start")


def _crc32c_canonical(configuration: ZarrV2Checksum32Parameters) -> ZarrV2Checksum32Parameters:
    """Fill in a `location` that is omitted or null with `"end"`, the default of `numcodecs.CRC32C`."""
    return _fill_location(configuration, "end")


CRC32_CODEC_V2: Final = ZarrV2CodecDefinition(
    name="crc32", configuration=ZarrV2Checksum32Parameters, canonical=_checksum_canonical
)
"""`numcodecs.CRC32`."""
CRC32C_CODEC_V2: Final = ZarrV2CodecDefinition(
    name="crc32c", configuration=ZarrV2Checksum32Parameters, canonical=_crc32c_canonical
)
"""`numcodecs.CRC32C`. Its `location` defaults to the end."""
ADLER32_CODEC_V2: Final = ZarrV2CodecDefinition(
    name="adler32", configuration=ZarrV2Checksum32Parameters, canonical=_checksum_canonical
)
"""`numcodecs.Adler32`."""
FLETCHER32_CODEC_V2: Final = ZarrV2CodecDefinition(
    name="fletcher32", configuration=EmptyConfiguration
)
"""`numcodecs.Fletcher32`. It has no configuration."""

__all__ = [
    "ADLER32_CODEC_V2",
    "CRC32C_CODEC_V2",
    "CRC32_CODEC_V2",
    "FLETCHER32_CODEC_V2",
    "ZarrV2Checksum32Parameters",
]
