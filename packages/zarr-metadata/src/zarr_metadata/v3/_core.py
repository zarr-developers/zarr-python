"""The scopes this package provides for Zarr v3, because "is this valid?" has two useful answers.

`CORE` contains what the Zarr v3 specification itself defines, so a
document read in it uses nothing that an implementation could reject as
optional. `CORE_AND_EXTENSIONS` adds the extensions that `zarr-extensions`
registers and this package defines. A name in neither scope is not
rejected, because that is what keeps the format open. Instead it is read
as an `UnclaimedField` and not checked.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from zarr_metadata.definition._context import Context, ZarrV3Context
from zarr_metadata.v3.chunk_grid.rectilinear import RECTILINEAR_CHUNK_GRID
from zarr_metadata.v3.chunk_grid.regular import REGULAR_CHUNK_GRID
from zarr_metadata.v3.chunk_key_encoding.default import DEFAULT_CHUNK_KEY_ENCODING
from zarr_metadata.v3.chunk_key_encoding.v2 import V2_CHUNK_KEY_ENCODING
from zarr_metadata.v3.codec.blosc import BLOSC_CODEC
from zarr_metadata.v3.codec.bytes import BYTES_CODEC
from zarr_metadata.v3.codec.cast_value import CAST_VALUE_CODEC
from zarr_metadata.v3.codec.crc32c import CRC32C_CODEC
from zarr_metadata.v3.codec.gzip import GZIP_CODEC
from zarr_metadata.v3.codec.scale_offset import SCALE_OFFSET_CODEC
from zarr_metadata.v3.codec.sharding_indexed import SHARDING_INDEXED_CODEC
from zarr_metadata.v3.codec.transpose import TRANSPOSE_CODEC
from zarr_metadata.v3.codec.zstd import ZSTD_CODEC
from zarr_metadata.v3.data_type.bool import BOOL_DATA_TYPE
from zarr_metadata.v3.data_type.bytes import BYTES_DATA_TYPE
from zarr_metadata.v3.data_type.complex64 import COMPLEX64_DATA_TYPE
from zarr_metadata.v3.data_type.complex128 import COMPLEX128_DATA_TYPE
from zarr_metadata.v3.data_type.float16 import FLOAT16_DATA_TYPE
from zarr_metadata.v3.data_type.float32 import FLOAT32_DATA_TYPE
from zarr_metadata.v3.data_type.float64 import FLOAT64_DATA_TYPE
from zarr_metadata.v3.data_type.int8 import INT8_DATA_TYPE
from zarr_metadata.v3.data_type.int16 import INT16_DATA_TYPE
from zarr_metadata.v3.data_type.int32 import INT32_DATA_TYPE
from zarr_metadata.v3.data_type.int64 import INT64_DATA_TYPE
from zarr_metadata.v3.data_type.numpy_datetime64 import NUMPY_DATETIME64_DATA_TYPE
from zarr_metadata.v3.data_type.numpy_timedelta64 import NUMPY_TIMEDELTA64_DATA_TYPE
from zarr_metadata.v3.data_type.raw import RAW_BYTES_DATA_TYPE
from zarr_metadata.v3.data_type.string import STRING_DATA_TYPE
from zarr_metadata.v3.data_type.struct import STRUCT_DATA_TYPE
from zarr_metadata.v3.data_type.uint8 import UINT8_DATA_TYPE
from zarr_metadata.v3.data_type.uint16 import UINT16_DATA_TYPE
from zarr_metadata.v3.data_type.uint32 import UINT32_DATA_TYPE
from zarr_metadata.v3.data_type.uint64 import UINT64_DATA_TYPE

if TYPE_CHECKING:
    from zarr_metadata.definition._definition import Definition

_CORE: Final[tuple[Definition[Any], ...]] = (
    BLOSC_CODEC,
    BYTES_CODEC,
    CRC32C_CODEC,
    GZIP_CODEC,
    SHARDING_INDEXED_CODEC,
    TRANSPOSE_CODEC,
    BOOL_DATA_TYPE,
    INT8_DATA_TYPE,
    INT16_DATA_TYPE,
    INT32_DATA_TYPE,
    INT64_DATA_TYPE,
    UINT8_DATA_TYPE,
    UINT16_DATA_TYPE,
    UINT32_DATA_TYPE,
    UINT64_DATA_TYPE,
    FLOAT16_DATA_TYPE,
    FLOAT32_DATA_TYPE,
    FLOAT64_DATA_TYPE,
    COMPLEX64_DATA_TYPE,
    COMPLEX128_DATA_TYPE,
    RAW_BYTES_DATA_TYPE,
    REGULAR_CHUNK_GRID,
    DEFAULT_CHUNK_KEY_ENCODING,
    V2_CHUNK_KEY_ENCODING,
)
"""What the Zarr v3 specification itself defines."""

_EXTENSIONS: Final[tuple[Definition[Any], ...]] = (
    CAST_VALUE_CODEC,
    SCALE_OFFSET_CODEC,
    ZSTD_CODEC,
    BYTES_DATA_TYPE,
    STRING_DATA_TYPE,
    NUMPY_DATETIME64_DATA_TYPE,
    NUMPY_TIMEDELTA64_DATA_TYPE,
    STRUCT_DATA_TYPE,
    RECTILINEAR_CHUNK_GRID,
)
"""The extensions that `zarr-extensions` registers and this package defines."""

CORE: Final[ZarrV3Context] = Context.of(*_CORE)
"""A scope with only what the Zarr v3 specification defines."""

CORE_AND_EXTENSIONS: Final[ZarrV3Context] = Context.of(*_CORE, *_EXTENSIONS)
"""A scope with what the specification defines, plus what `zarr-extensions` registers."""


__all__ = [
    "CORE",
    "CORE_AND_EXTENSIONS",
]
