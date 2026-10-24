"""Zarr v2 metadata types, and the front door to the v2 definitions.

Each document type is in the submodule for its node or file: `array`
(`.zarray`), `group` (`.zgroup`), `attributes` (`.zattrs`) and
`consolidated` (`.zmetadata`). The codecs and data types are in `codec`
and `data_type`, and the v2 kinds and scope are in `definition`.

This module also exports every v2 definition (`GZIP_V2`, `FLOAT_V2`, ...),
the scope `CORE_V2`, and the scope class `ZarrV2Context`, so that one
import builds or extends a scope. Each is the same object as in its own
submodule, where a definition's configuration types and constants are.
"""

from zarr_metadata.v2.codec.checksum import (
    ADLER32_V2,
    CRC32_V2,
    CRC32C_V2,
    FLETCHER32_V2,
)
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
from zarr_metadata.v2.codec.vlen import (
    VLEN_ARRAY_V2,
    VLEN_BYTES_V2,
    VLEN_UTF8_V2,
)
from zarr_metadata.v2.data_type.fixed_width import (
    BYTES_V2,
    STR_V2,
    VOID_V2,
)
from zarr_metadata.v2.data_type.object import OBJECT_V2
from zarr_metadata.v2.data_type.scalar import (
    BOOL_V2,
    COMPLEX_V2,
    FLOAT_V2,
    INT_V2,
    UINT_V2,
)
from zarr_metadata.v2.data_type.struct import STRUCT_V2
from zarr_metadata.v2.data_type.time import (
    DATETIME64_V2,
    TIMEDELTA64_V2,
)
from zarr_metadata.v2.definition import CORE_V2, ZarrV2Context

__all__ = [
    "ADLER32_V2",
    "ASTYPE_V2",
    "BITROUND_V2",
    "BLOSC_V2",
    "BOOL_V2",
    "BYTES_V2",
    "BZ2_V2",
    "COMPLEX_V2",
    "CORE_V2",
    "CRC32C_V2",
    "CRC32_V2",
    "DATETIME64_V2",
    "DELTA_V2",
    "FIXEDSCALEOFFSET_V2",
    "FLETCHER32_V2",
    "FLOAT_V2",
    "GZIP_V2",
    "INT_V2",
    "LZ4_V2",
    "LZMA_V2",
    "OBJECT_V2",
    "PACKBITS_V2",
    "QUANTIZE_V2",
    "SHUFFLE_V2",
    "STRUCT_V2",
    "STR_V2",
    "TIMEDELTA64_V2",
    "UINT_V2",
    "VLEN_ARRAY_V2",
    "VLEN_BYTES_V2",
    "VLEN_UTF8_V2",
    "VOID_V2",
    "ZLIB_V2",
    "ZSTD_V2",
    "ZarrV2Context",
]
