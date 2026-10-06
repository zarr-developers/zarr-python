"""The v2 checksums numcodecs 0.16 configures: `crc32`, `crc32c`, `adler32` and `fletcher32`."""

from __future__ import annotations

from typing import Final, Literal, NotRequired

from typing_extensions import ReadOnly, TypedDict

from zarr_metadata.v2._definition import ZarrV2CodecDefinition
from zarr_metadata.v3._definition import EmptyConfiguration


class ZarrV2Checksum32Parameters(TypedDict, closed=True):
    """`numcodecs.CRC32(location=None)`, and the other 32-bit checksums: where the checksum sits, at the start by default."""

    location: NotRequired[ReadOnly[Literal["start", "end"]]]


CRC32_V2: Final = ZarrV2CodecDefinition(name="crc32", configuration=ZarrV2Checksum32Parameters)
"""`numcodecs.CRC32`."""
CRC32C_V2: Final = ZarrV2CodecDefinition(name="crc32c", configuration=ZarrV2Checksum32Parameters)
"""`numcodecs.CRC32C`."""
ADLER32_V2: Final = ZarrV2CodecDefinition(name="adler32", configuration=ZarrV2Checksum32Parameters)
"""`numcodecs.Adler32`."""
FLETCHER32_V2: Final = ZarrV2CodecDefinition(name="fletcher32", configuration=EmptyConfiguration)
"""`numcodecs.Fletcher32`: nothing to configure."""

__all__ = ["ADLER32_V2", "CRC32C_V2", "CRC32_V2", "FLETCHER32_V2", "ZarrV2Checksum32Parameters"]
