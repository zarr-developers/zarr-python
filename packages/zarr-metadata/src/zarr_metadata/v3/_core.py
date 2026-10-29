"""The scopes this package provides for Zarr v3, because "is this valid?" has two useful answers.

`CORE_CONTEXT` contains what the Zarr v3 specification itself defines, so a
document read in it uses nothing that an implementation could reject as
optional. `CORE_AND_EXTENSIONS_CONTEXT` adds the extensions that `zarr-extensions`
registers and this package defines. `ZARR_PYTHON_CONTEXT` adds the data types that
zarr-python writes but no specification registers, whose spelling
zarr-python may still change. A name in none of these scopes is not
rejected, because that is what keeps the format open. Instead it is read
as an `UnclaimedField` and not checked.
"""

from __future__ import annotations

from typing import Final

from zarr_metadata.v3._context import ZarrV3Context
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
from zarr_metadata.v3.codec.vlen_bytes import VLEN_BYTES_CODEC
from zarr_metadata.v3.codec.vlen_utf8 import VLEN_UTF8_CODEC
from zarr_metadata.v3.codec.zstd import ZSTD_CODEC
from zarr_metadata.v3.consolidated import CONSOLIDATED_METADATA_EXTRA_FIELD
from zarr_metadata.v3.data_type.bool import BOOL_DATA_TYPE
from zarr_metadata.v3.data_type.bytes import BYTES_DATA_TYPE
from zarr_metadata.v3.data_type.complex64 import COMPLEX64_DATA_TYPE
from zarr_metadata.v3.data_type.complex128 import COMPLEX128_DATA_TYPE
from zarr_metadata.v3.data_type.fixed_length_utf32 import FIXED_LENGTH_UTF32_DATA_TYPE
from zarr_metadata.v3.data_type.float16 import FLOAT16_DATA_TYPE
from zarr_metadata.v3.data_type.float32 import FLOAT32_DATA_TYPE
from zarr_metadata.v3.data_type.float64 import FLOAT64_DATA_TYPE
from zarr_metadata.v3.data_type.int8 import INT8_DATA_TYPE
from zarr_metadata.v3.data_type.int16 import INT16_DATA_TYPE
from zarr_metadata.v3.data_type.int32 import INT32_DATA_TYPE
from zarr_metadata.v3.data_type.int64 import INT64_DATA_TYPE
from zarr_metadata.v3.data_type.null_terminated_bytes import NULL_TERMINATED_BYTES_DATA_TYPE
from zarr_metadata.v3.data_type.numpy_datetime64 import NUMPY_DATETIME64_DATA_TYPE
from zarr_metadata.v3.data_type.numpy_timedelta64 import NUMPY_TIMEDELTA64_DATA_TYPE
from zarr_metadata.v3.data_type.raw import RAW_BYTES_DATA_TYPE
from zarr_metadata.v3.data_type.string import STRING_DATA_TYPE
from zarr_metadata.v3.data_type.struct import STRUCT_DATA_TYPE
from zarr_metadata.v3.data_type.uint8 import UINT8_DATA_TYPE
from zarr_metadata.v3.data_type.uint16 import UINT16_DATA_TYPE
from zarr_metadata.v3.data_type.uint32 import UINT32_DATA_TYPE
from zarr_metadata.v3.data_type.uint64 import UINT64_DATA_TYPE
from zarr_metadata.v3.data_type.variable_length_bytes import VARIABLE_LENGTH_BYTES_DATA_TYPE

CORE_CONTEXT: Final = ZarrV3Context(
    data_types=(
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
    ),
    chunk_grids=(REGULAR_CHUNK_GRID,),
    chunk_key_encodings=(DEFAULT_CHUNK_KEY_ENCODING, V2_CHUNK_KEY_ENCODING),
    codecs=(
        BLOSC_CODEC,
        BYTES_CODEC,
        CRC32C_CODEC,
        GZIP_CODEC,
        SHARDING_INDEXED_CODEC,
        TRANSPOSE_CODEC,
    ),
    group_extra_fields=(CONSOLIDATED_METADATA_EXTRA_FIELD,),
)
"""A scope with only what the Zarr v3 specification defines, including the envelope of a group's `consolidated_metadata`, which the spec fixes."""

CORE_AND_EXTENSIONS_CONTEXT: Final = CORE_CONTEXT.update(
    data_types=[
        *CORE_CONTEXT.data_types.values(),
        BYTES_DATA_TYPE,
        STRING_DATA_TYPE,
        FIXED_LENGTH_UTF32_DATA_TYPE,
        NUMPY_DATETIME64_DATA_TYPE,
        NUMPY_TIMEDELTA64_DATA_TYPE,
        STRUCT_DATA_TYPE,
    ],
    chunk_grids=[*CORE_CONTEXT.chunk_grids.values(), RECTILINEAR_CHUNK_GRID],
    codecs=[
        *CORE_CONTEXT.codecs.values(),
        CAST_VALUE_CODEC,
        SCALE_OFFSET_CODEC,
        VLEN_BYTES_CODEC,
        VLEN_UTF8_CODEC,
        ZSTD_CODEC,
    ],
)
"""A scope with what the specification defines, plus the extensions that `zarr-extensions` registers and this package defines."""

ZARR_PYTHON_CONTEXT: Final = CORE_AND_EXTENSIONS_CONTEXT.update(
    data_types=[
        *CORE_AND_EXTENSIONS_CONTEXT.data_types.values(),
        NULL_TERMINATED_BYTES_DATA_TYPE,
        VARIABLE_LENGTH_BYTES_DATA_TYPE,
    ],
)
"""A scope with `CORE_AND_EXTENSIONS_CONTEXT`, plus the data types that zarr-python writes but no specification registers: `null_terminated_bytes` and `variable_length_bytes`.

zarr-python warns when it writes these that their spelling may change, so
they are kept out of `CORE_AND_EXTENSIONS_CONTEXT`. Read in this scope to check
strictly what zarr-python writes.
"""

__all__ = [
    "CORE_AND_EXTENSIONS_CONTEXT",
    "CORE_CONTEXT",
    "ZARR_PYTHON_CONTEXT",
]
