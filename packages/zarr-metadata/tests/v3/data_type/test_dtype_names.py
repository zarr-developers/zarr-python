"""Validate every primitive Zarr v3 data-type name string.

Primitive dtypes are encoded as bare strings in the `data_type` field of a
v3 array metadata document (e.g. `"int32"`, `"float64"`). Each must
validate as its declared per-dtype `*Name` literal type.
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter

from zarr_metadata.v3.data_type.bool import BOOL_DATA_TYPE_NAME, BoolDataTypeName
from zarr_metadata.v3.data_type.bytes import BYTES_DATA_TYPE_NAME, BytesDataTypeName
from zarr_metadata.v3.data_type.complex64 import COMPLEX64_DATA_TYPE_NAME, Complex64DataTypeName
from zarr_metadata.v3.data_type.complex128 import COMPLEX128_DATA_TYPE_NAME, Complex128DataTypeName
from zarr_metadata.v3.data_type.float16 import FLOAT16_DATA_TYPE_NAME, Float16DataTypeName
from zarr_metadata.v3.data_type.float32 import FLOAT32_DATA_TYPE_NAME, Float32DataTypeName
from zarr_metadata.v3.data_type.float64 import FLOAT64_DATA_TYPE_NAME, Float64DataTypeName
from zarr_metadata.v3.data_type.int8 import INT8_DATA_TYPE_NAME, Int8DataTypeName
from zarr_metadata.v3.data_type.int16 import INT16_DATA_TYPE_NAME, Int16DataTypeName
from zarr_metadata.v3.data_type.int32 import INT32_DATA_TYPE_NAME, Int32DataTypeName
from zarr_metadata.v3.data_type.int64 import INT64_DATA_TYPE_NAME, Int64DataTypeName
from zarr_metadata.v3.data_type.raw import raw_bytes_dtype_name
from zarr_metadata.v3.data_type.string import STRING_DATA_TYPE_NAME, StringDataTypeName
from zarr_metadata.v3.data_type.uint8 import UINT8_DATA_TYPE_NAME, Uint8DataTypeName
from zarr_metadata.v3.data_type.uint16 import UINT16_DATA_TYPE_NAME, Uint16DataTypeName
from zarr_metadata.v3.data_type.uint32 import UINT32_DATA_TYPE_NAME, Uint32DataTypeName
from zarr_metadata.v3.data_type.uint64 import UINT64_DATA_TYPE_NAME, Uint64DataTypeName

# (name_string, per-dtype literal type)
PRIMITIVE_DTYPES = [
    (BOOL_DATA_TYPE_NAME, BoolDataTypeName),
    (INT8_DATA_TYPE_NAME, Int8DataTypeName),
    (INT16_DATA_TYPE_NAME, Int16DataTypeName),
    (INT32_DATA_TYPE_NAME, Int32DataTypeName),
    (INT64_DATA_TYPE_NAME, Int64DataTypeName),
    (UINT8_DATA_TYPE_NAME, Uint8DataTypeName),
    (UINT16_DATA_TYPE_NAME, Uint16DataTypeName),
    (UINT32_DATA_TYPE_NAME, Uint32DataTypeName),
    (UINT64_DATA_TYPE_NAME, Uint64DataTypeName),
    (FLOAT16_DATA_TYPE_NAME, Float16DataTypeName),
    (FLOAT32_DATA_TYPE_NAME, Float32DataTypeName),
    (FLOAT64_DATA_TYPE_NAME, Float64DataTypeName),
    (COMPLEX64_DATA_TYPE_NAME, Complex64DataTypeName),
    (COMPLEX128_DATA_TYPE_NAME, Complex128DataTypeName),
    (STRING_DATA_TYPE_NAME, StringDataTypeName),
    (BYTES_DATA_TYPE_NAME, BytesDataTypeName),
]


@pytest.mark.parametrize(("name", "literal_type"), PRIMITIVE_DTYPES, ids=lambda x: str(x))
def test_primitive_against_literal(name: str, literal_type: object) -> None:
    """The dtype name validates against its declared Literal type."""
    TypeAdapter(literal_type).validate_python(name)


@pytest.mark.parametrize("raw_name", ["r8", "r16", "r24", "r256", "r1024"], ids=str)
def test_raw_bytes_name(raw_name: str) -> None:
    """`r<N>` names pass the raw_bytes_dtype_name validator."""
    raw_bytes_dtype_name(raw_name)
