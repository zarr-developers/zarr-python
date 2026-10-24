"""The v2 checksums that numcodecs 0.16 configures: `crc32`, `crc32c`, `adler32` and `fletcher32`."""

from __future__ import annotations

from typing import Final, Literal, NotRequired

from typing_extensions import ReadOnly, TypedDict

from zarr_metadata.definition._definition import EmptyConfiguration
from zarr_metadata.v2._definition import ZarrV2CodecDefinition


class ZarrV2Checksum32Parameters(TypedDict, closed=True):
    """`numcodecs.CRC32(location=None)` and the other 32-bit checksums. `location` is where the checksum is placed. When it is omitted or null, it is the start."""

    location: NotRequired[ReadOnly[Literal["start", "end"] | None]]


def _checksum_canonical(configuration: ZarrV2Checksum32Parameters) -> ZarrV2Checksum32Parameters:
    """Fill in a `location` that is omitted or null with the default, `"start"`."""
    location = configuration.get("location")
    return {"location": "start" if location is None else location}


CRC32_CODEC_V2: Final = ZarrV2CodecDefinition(
    name="crc32", configuration=ZarrV2Checksum32Parameters, canonical=_checksum_canonical
)
"""`numcodecs.CRC32`."""
CRC32C_CODEC_V2: Final = ZarrV2CodecDefinition(
    name="crc32c", configuration=ZarrV2Checksum32Parameters, canonical=_checksum_canonical
)
"""`numcodecs.CRC32C`."""
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
