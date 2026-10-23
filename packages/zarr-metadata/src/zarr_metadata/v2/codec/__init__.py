"""Zarr v2 codecs: the configuration shape, and one definition per numcodecs id that this package models.

In v2, compressors and filters are numcodecs configuration dicts: a required
`id` field naming the codec, plus codec-specific parameters.

Each codec's definition and parameter TypedDict are in its submodule
(`checksum`, `compression`, `filters`, `vlen`). This module holds the
configuration shape and `V2_CODECS`, the tuple of every definition.
"""

from typing import Any, Final

from zarr_metadata.v2._codec_json import ZarrV2CodecMetadata
from zarr_metadata.v2._definition import ZarrV2CodecDefinition
from zarr_metadata.v2.codec.checksum import ADLER32_V2, CRC32_V2, CRC32C_V2, FLETCHER32_V2
from zarr_metadata.v2.codec.compression import (
    BLOSC_V2,
    BZ2_V2,
    GZIP_V2,
    LZ4_V2,
    LZMA_V2,
    ZLIB_V2,
    ZSTD_V2,
)
from zarr_metadata.v2.codec.filters import (
    ASTYPE_V2,
    BITROUND_V2,
    DELTA_V2,
    FIXEDSCALEOFFSET_V2,
    PACKBITS_V2,
    QUANTIZE_V2,
    SHUFFLE_V2,
)
from zarr_metadata.v2.codec.vlen import VLEN_ARRAY_V2, VLEN_BYTES_V2, VLEN_UTF8_V2

V2_CODECS: Final[tuple[ZarrV2CodecDefinition[Any], ...]] = (
    ZLIB_V2,
    GZIP_V2,
    BZ2_V2,
    LZMA_V2,
    BLOSC_V2,
    ZSTD_V2,
    LZ4_V2,
    SHUFFLE_V2,
    DELTA_V2,
    FIXEDSCALEOFFSET_V2,
    QUANTIZE_V2,
    BITROUND_V2,
    ASTYPE_V2,
    PACKBITS_V2,
    VLEN_UTF8_V2,
    VLEN_BYTES_V2,
    VLEN_ARRAY_V2,
    CRC32_V2,
    CRC32C_V2,
    ADLER32_V2,
    FLETCHER32_V2,
)
"""Every codec that numcodecs 0.16 configures and that this package models."""

__all__ = [
    "V2_CODECS",
    "ZarrV2CodecMetadata",
]
