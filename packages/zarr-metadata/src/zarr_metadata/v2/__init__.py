"""Zarr v2 metadata types, and the front door to the v2 definitions.

Each document type is in the submodule for its node or file: `array`
(`.zarray`), `group` (`.zgroup`), `attributes` (`.zattrs`) and
`consolidated` (`.zmetadata`). The codecs and data types are in `codec`
and `data_type`, and the v2 kinds and scope are in `definition`.

This module also exports every v2 definition (`GZIP_CODEC_V2`,
`FLOAT_DATA_TYPE_V2`, ...), the scope `CORE_V2_CONTEXT`, and the scope class
`ZarrV2Context`, so that one import builds or updates a scope. Each is
the same object as in its own submodule, where a definition's
configuration types and constants are.
"""

from typing import TYPE_CHECKING

from zarr_metadata._renamed import renamed_scope
from zarr_metadata.v2._core import RENAMED_SCOPES_V2
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
from zarr_metadata.v2.codec.vlen import (
    VLEN_ARRAY_CODEC_V2,
    VLEN_BYTES_CODEC_V2,
    VLEN_UTF8_CODEC_V2,
)
from zarr_metadata.v2.data_type.fixed_width import (
    BYTES_DATA_TYPE_V2,
    STR_DATA_TYPE_V2,
    VOID_DATA_TYPE_V2,
)
from zarr_metadata.v2.data_type.object import OBJECT_DATA_TYPE_V2
from zarr_metadata.v2.data_type.scalar import (
    BOOL_DATA_TYPE_V2,
    COMPLEX_DATA_TYPE_V2,
    FLOAT_DATA_TYPE_V2,
    INT_DATA_TYPE_V2,
    UINT_DATA_TYPE_V2,
)
from zarr_metadata.v2.data_type.struct import STRUCT_DATA_TYPE_V2
from zarr_metadata.v2.data_type.time import (
    DATETIME64_DATA_TYPE_V2,
    TIMEDELTA64_DATA_TYPE_V2,
)
from zarr_metadata.v2.definition import CORE_V2_CONTEXT, ZarrV2Context

__all__ = [
    "ADLER32_CODEC_V2",
    "ASTYPE_CODEC_V2",
    "BITROUND_CODEC_V2",
    "BLOSC_CODEC_V2",
    "BOOL_DATA_TYPE_V2",
    "BYTES_DATA_TYPE_V2",
    "BZ2_CODEC_V2",
    "COMPLEX_DATA_TYPE_V2",
    "CORE_V2_CONTEXT",
    "CRC32C_CODEC_V2",
    "CRC32_CODEC_V2",
    "DATETIME64_DATA_TYPE_V2",
    "DELTA_CODEC_V2",
    "FIXEDSCALEOFFSET_CODEC_V2",
    "FLETCHER32_CODEC_V2",
    "FLOAT_DATA_TYPE_V2",
    "GZIP_CODEC_V2",
    "INT_DATA_TYPE_V2",
    "LZ4_CODEC_V2",
    "LZMA_CODEC_V2",
    "OBJECT_DATA_TYPE_V2",
    "PACKBITS_CODEC_V2",
    "QUANTIZE_CODEC_V2",
    "SHUFFLE_CODEC_V2",
    "STRUCT_DATA_TYPE_V2",
    "STR_DATA_TYPE_V2",
    "TIMEDELTA64_DATA_TYPE_V2",
    "UINT_DATA_TYPE_V2",
    "VLEN_ARRAY_CODEC_V2",
    "VLEN_BYTES_CODEC_V2",
    "VLEN_UTF8_CODEC_V2",
    "VOID_DATA_TYPE_V2",
    "ZLIB_CODEC_V2",
    "ZSTD_CODEC_V2",
    "ZarrV2Context",
]


if not TYPE_CHECKING:
    # Hidden from type checkers, which would otherwise read every missing
    # name of this module as a scope.
    def __getattr__(name: str) -> object:
        """`CORE_V2`, the scope's name before it ended in `_CONTEXT`, with a `DeprecationWarning`."""
        return renamed_scope(__name__, name, RENAMED_SCOPES_V2)
