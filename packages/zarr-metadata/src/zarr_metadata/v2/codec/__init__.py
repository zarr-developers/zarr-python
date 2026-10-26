"""Zarr v2 codecs: the configuration shape, and one definition per numcodecs id that this package models.

In v2, compressors and filters are numcodecs configuration dicts: a required
`id` field naming the codec, plus codec-specific parameters.

Each codec's definition and parameter TypedDict are in its submodule
(`checksum`, `compression`, `filters`, `vlen`). This module holds the
configuration shape and `CODECS_V2`, the tuple of every definition.
"""

from typing import Any, Final

from zarr_metadata.v2._definition import ZarrV2CodecDefinition
from zarr_metadata.v2._field_json import ZarrV2CodecMetadata
from zarr_metadata.v2.codec.checksum import (
    ADLER32_CODEC_V2,
    CRC32_CODEC_V2,
    CRC32C_CODEC_V2,
    FLETCHER32_CODEC_V2,
)
from zarr_metadata.v2.codec.compression import (
    BLOSC_CODEC_V2,
    BZ2_CODEC_V2,
    GZIP_CODEC_V2,
    LZ4_CODEC_V2,
    LZMA_CODEC_V2,
    ZLIB_CODEC_V2,
    ZSTD_CODEC_V2,
)
from zarr_metadata.v2.codec.filters import (
    ASTYPE_CODEC_V2,
    BITROUND_CODEC_V2,
    DELTA_CODEC_V2,
    FIXEDSCALEOFFSET_CODEC_V2,
    PACKBITS_CODEC_V2,
    QUANTIZE_CODEC_V2,
    SHUFFLE_CODEC_V2,
)
from zarr_metadata.v2.codec.vlen import VLEN_ARRAY_CODEC_V2, VLEN_BYTES_CODEC_V2, VLEN_UTF8_CODEC_V2

CODECS_V2: Final[tuple[ZarrV2CodecDefinition[Any], ...]] = (
    ZLIB_CODEC_V2,
    GZIP_CODEC_V2,
    BZ2_CODEC_V2,
    LZMA_CODEC_V2,
    BLOSC_CODEC_V2,
    ZSTD_CODEC_V2,
    LZ4_CODEC_V2,
    SHUFFLE_CODEC_V2,
    DELTA_CODEC_V2,
    FIXEDSCALEOFFSET_CODEC_V2,
    QUANTIZE_CODEC_V2,
    BITROUND_CODEC_V2,
    ASTYPE_CODEC_V2,
    PACKBITS_CODEC_V2,
    VLEN_UTF8_CODEC_V2,
    VLEN_BYTES_CODEC_V2,
    VLEN_ARRAY_CODEC_V2,
    CRC32_CODEC_V2,
    CRC32C_CODEC_V2,
    ADLER32_CODEC_V2,
    FLETCHER32_CODEC_V2,
)
"""Every codec that numcodecs 0.16 configures and that this package models."""

__all__ = [
    "CODECS_V2",
    "ZarrV2CodecMetadata",
]
