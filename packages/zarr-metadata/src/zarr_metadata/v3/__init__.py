"""Zarr v3 metadata types, and the front door to the v3 definitions.

This module holds the JSON of any metadata field (a bare name or a
`ZarrV3NamedConfigJSON` envelope) and the node name and path
types. Each document type is in the submodule for its node: `array`,
`group` and `consolidated`. The extensions are in `codec`, `data_type`,
`chunk_grid` and `chunk_key_encoding`, and the v3 kinds and scopes are in
`definition`.

This module also exports every v3 definition (`BYTES_CODEC`,
`FLOAT64_DATA_TYPE`, `REGULAR_CHUNK_GRID`, ...), the scopes `CORE` and
`CORE_AND_EXTENSIONS`, and the scope class `ZarrV3Context`, so that one
import builds or extends a scope. Each is the same object as in its own
submodule, where a definition's configuration types and constants are.
"""

from zarr_metadata._common import ZarrV3NamedConfigJSON
from zarr_metadata.v3._common import ZarrV3ExtensionField, ZarrV3MetadataFieldJSON
from zarr_metadata.v3._hierarchy import NodeName, NodePath
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
from zarr_metadata.v3.definition import (
    CORE,
    CORE_AND_EXTENSIONS,
    ZarrV3Context,
)

__all__ = [
    "BLOSC_CODEC",
    "BOOL_DATA_TYPE",
    "BYTES_CODEC",
    "BYTES_DATA_TYPE",
    "CAST_VALUE_CODEC",
    "COMPLEX64_DATA_TYPE",
    "COMPLEX128_DATA_TYPE",
    "CORE",
    "CORE_AND_EXTENSIONS",
    "CRC32C_CODEC",
    "DEFAULT_CHUNK_KEY_ENCODING",
    "FLOAT16_DATA_TYPE",
    "FLOAT32_DATA_TYPE",
    "FLOAT64_DATA_TYPE",
    "GZIP_CODEC",
    "INT8_DATA_TYPE",
    "INT16_DATA_TYPE",
    "INT32_DATA_TYPE",
    "INT64_DATA_TYPE",
    "NUMPY_DATETIME64_DATA_TYPE",
    "NUMPY_TIMEDELTA64_DATA_TYPE",
    "RAW_BYTES_DATA_TYPE",
    "RECTILINEAR_CHUNK_GRID",
    "REGULAR_CHUNK_GRID",
    "SCALE_OFFSET_CODEC",
    "SHARDING_INDEXED_CODEC",
    "STRING_DATA_TYPE",
    "STRUCT_DATA_TYPE",
    "TRANSPOSE_CODEC",
    "UINT8_DATA_TYPE",
    "UINT16_DATA_TYPE",
    "UINT32_DATA_TYPE",
    "UINT64_DATA_TYPE",
    "V2_CHUNK_KEY_ENCODING",
    "ZSTD_CODEC",
    "NodeName",
    "NodePath",
    "ZarrV3Context",
    "ZarrV3ExtensionField",
    "ZarrV3MetadataFieldJSON",
    "ZarrV3NamedConfigJSON",
]
