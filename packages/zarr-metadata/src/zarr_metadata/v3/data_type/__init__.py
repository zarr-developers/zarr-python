"""Zarr v3 data type spec types.

Each v3 data type has its own submodule:

- Core primitives: `bool`, `int8`/`16`/`32`/`64`, `uint8`/`16`/`32`/`64`,
  `float16`/`32`/`64`, `complex64`/`128`, `raw` (for `r<N>`)
- zarr-extensions: `bytes`, `string`, `fixed_length_utf32`,
  `numpy_datetime64`, `numpy_timedelta64`, `struct`
- zarr-python, registered by no specification: `null_terminated_bytes`,
  `variable_length_bytes`, `zarr_python_raw_bytes` (for `raw_bytes`)

Import a data type's types and its definition from its submodule, or from
this package, which re-exports every name its submodules export:

- `<X>DataTypeName` -- the literal type of the dtype's `data_type` string
  (or, for named-config dtypes, the literal value of their `name` field)
- `<X>FillValue` -- the permitted JSON shape of the `fill_value` field
- the envelope and configuration TypedDicts of named-config dtypes
  (`fixed_length_utf32`, `null_terminated_bytes`, `numpy_datetime64`,
  `numpy_timedelta64`, `struct`, `raw_bytes`), the branded
  `HexFloat<N>` / `Base64Bytes` types, and their validator functions
- `<X>_DATA_TYPE`, the definition

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from zarr_metadata.v3.data_type.bool import (
    BOOL_DATA_TYPE,
    BOOL_DATA_TYPE_NAME,
    BoolDataTypeName,
    BoolFillValue,
)
from zarr_metadata.v3.data_type.bytes import (
    BYTES_DATA_TYPE,
    BYTES_DATA_TYPE_NAME,
    Base64Bytes,
    BytesDataTypeName,
    BytesFillValue,
    base64_bytes,
)
from zarr_metadata.v3.data_type.complex64 import (
    COMPLEX64_DATA_TYPE,
    COMPLEX64_DATA_TYPE_NAME,
    Complex64Component,
    Complex64DataTypeName,
    Complex64FillValue,
)
from zarr_metadata.v3.data_type.complex128 import (
    COMPLEX128_DATA_TYPE,
    COMPLEX128_DATA_TYPE_NAME,
    Complex128Component,
    Complex128DataTypeName,
    Complex128FillValue,
)
from zarr_metadata.v3.data_type.fixed_length_utf32 import (
    FIXED_LENGTH_UTF32_DATA_TYPE,
    FIXED_LENGTH_UTF32_DATA_TYPE_NAME,
    FixedLengthUtf32,
    FixedLengthUtf32Configuration,
    FixedLengthUtf32DataTypeName,
    FixedLengthUtf32FillValue,
)
from zarr_metadata.v3.data_type.float16 import (
    CANONICAL_NAN_HEX_FLOAT16,
    CANONICAL_NEGATIVE_INFINITY_HEX_FLOAT16,
    CANONICAL_POSITIVE_INFINITY_HEX_FLOAT16,
    FLOAT16_DATA_TYPE,
    FLOAT16_DATA_TYPE_NAME,
    Float16DataTypeName,
    Float16FillValue,
    Float16SpecialFillValue,
    HexFloat16,
    hex_float16,
)
from zarr_metadata.v3.data_type.float32 import (
    CANONICAL_NAN_HEX_FLOAT32,
    CANONICAL_NEGATIVE_INFINITY_HEX_FLOAT32,
    CANONICAL_POSITIVE_INFINITY_HEX_FLOAT32,
    FLOAT32_DATA_TYPE,
    FLOAT32_DATA_TYPE_NAME,
    Float32DataTypeName,
    Float32FillValue,
    Float32SpecialFillValue,
    HexFloat32,
    hex_float32,
)
from zarr_metadata.v3.data_type.float64 import (
    CANONICAL_NAN_HEX_FLOAT64,
    CANONICAL_NEGATIVE_INFINITY_HEX_FLOAT64,
    CANONICAL_POSITIVE_INFINITY_HEX_FLOAT64,
    FLOAT64_DATA_TYPE,
    FLOAT64_DATA_TYPE_NAME,
    Float64DataTypeName,
    Float64FillValue,
    Float64SpecialFillValue,
    HexFloat64,
    hex_float64,
)
from zarr_metadata.v3.data_type.int8 import (
    INT8_DATA_TYPE,
    INT8_DATA_TYPE_NAME,
    Int8DataTypeName,
    Int8FillValue,
)
from zarr_metadata.v3.data_type.int16 import (
    INT16_DATA_TYPE,
    INT16_DATA_TYPE_NAME,
    Int16DataTypeName,
    Int16FillValue,
)
from zarr_metadata.v3.data_type.int32 import (
    INT32_DATA_TYPE,
    INT32_DATA_TYPE_NAME,
    Int32DataTypeName,
    Int32FillValue,
)
from zarr_metadata.v3.data_type.int64 import (
    INT64_DATA_TYPE,
    INT64_DATA_TYPE_NAME,
    Int64DataTypeName,
    Int64FillValue,
)
from zarr_metadata.v3.data_type.null_terminated_bytes import (
    NULL_TERMINATED_BYTES_DATA_TYPE,
    NULL_TERMINATED_BYTES_DATA_TYPE_NAME,
    NullTerminatedBytes,
    NullTerminatedBytesConfiguration,
    NullTerminatedBytesDataTypeName,
    NullTerminatedBytesFillValue,
)
from zarr_metadata.v3.data_type.numpy_datetime64 import (
    NUMPY_DATETIME64_DATA_TYPE,
    NUMPY_DATETIME64_DATA_TYPE_NAME,
    NUMPY_TIME_UNIT,
    NumpyDatetime64,
    NumpyDatetime64Configuration,
    NumpyDatetime64DataTypeName,
    NumpyDatetime64FillValue,
    NumpyTimeUnit,
)
from zarr_metadata.v3.data_type.numpy_timedelta64 import (
    NUMPY_TIMEDELTA64_DATA_TYPE,
    NUMPY_TIMEDELTA64_DATA_TYPE_NAME,
    NumpyTimedelta64,
    NumpyTimedelta64Configuration,
    NumpyTimedelta64DataTypeName,
    NumpyTimedelta64FillValue,
)
from zarr_metadata.v3.data_type.raw import (
    RAW_BYTES_DATA_TYPE,
    RawBytesConfiguration,
    RawBytesDataTypeName,
    RawBytesFillValue,
    raw_bytes_dtype_name,
)
from zarr_metadata.v3.data_type.string import (
    STRING_DATA_TYPE,
    STRING_DATA_TYPE_NAME,
    StringDataTypeName,
    StringFillValue,
)
from zarr_metadata.v3.data_type.struct import (
    STRUCT_DATA_TYPE,
    STRUCT_DATA_TYPE_NAME,
    Struct,
    StructConfiguration,
    StructDataTypeName,
    StructField,
    StructFillValue,
)
from zarr_metadata.v3.data_type.uint8 import (
    UINT8_DATA_TYPE,
    UINT8_DATA_TYPE_NAME,
    Uint8DataTypeName,
    Uint8FillValue,
)
from zarr_metadata.v3.data_type.uint16 import (
    UINT16_DATA_TYPE,
    UINT16_DATA_TYPE_NAME,
    Uint16DataTypeName,
    Uint16FillValue,
)
from zarr_metadata.v3.data_type.uint32 import (
    UINT32_DATA_TYPE,
    UINT32_DATA_TYPE_NAME,
    Uint32DataTypeName,
    Uint32FillValue,
)
from zarr_metadata.v3.data_type.uint64 import (
    UINT64_DATA_TYPE,
    UINT64_DATA_TYPE_NAME,
    Uint64DataTypeName,
    Uint64FillValue,
)
from zarr_metadata.v3.data_type.variable_length_bytes import (
    VARIABLE_LENGTH_BYTES_DATA_TYPE,
    VARIABLE_LENGTH_BYTES_DATA_TYPE_NAME,
    VariableLengthBytesDataTypeName,
    VariableLengthBytesFillValue,
)
from zarr_metadata.v3.data_type.zarr_python_raw_bytes import (
    ZARR_PYTHON_RAW_BYTES_DATA_TYPE,
    ZARR_PYTHON_RAW_BYTES_DATA_TYPE_NAME,
    ZarrPythonRawBytes,
    ZarrPythonRawBytesConfiguration,
    ZarrPythonRawBytesDataTypeName,
    ZarrPythonRawBytesFillValue,
)

__all__ = [
    "BOOL_DATA_TYPE",
    "BOOL_DATA_TYPE_NAME",
    "BYTES_DATA_TYPE",
    "BYTES_DATA_TYPE_NAME",
    "CANONICAL_NAN_HEX_FLOAT16",
    "CANONICAL_NAN_HEX_FLOAT32",
    "CANONICAL_NAN_HEX_FLOAT64",
    "CANONICAL_NEGATIVE_INFINITY_HEX_FLOAT16",
    "CANONICAL_NEGATIVE_INFINITY_HEX_FLOAT32",
    "CANONICAL_NEGATIVE_INFINITY_HEX_FLOAT64",
    "CANONICAL_POSITIVE_INFINITY_HEX_FLOAT16",
    "CANONICAL_POSITIVE_INFINITY_HEX_FLOAT32",
    "CANONICAL_POSITIVE_INFINITY_HEX_FLOAT64",
    "COMPLEX64_DATA_TYPE",
    "COMPLEX64_DATA_TYPE_NAME",
    "COMPLEX128_DATA_TYPE",
    "COMPLEX128_DATA_TYPE_NAME",
    "FIXED_LENGTH_UTF32_DATA_TYPE",
    "FIXED_LENGTH_UTF32_DATA_TYPE_NAME",
    "FLOAT16_DATA_TYPE",
    "FLOAT16_DATA_TYPE_NAME",
    "FLOAT32_DATA_TYPE",
    "FLOAT32_DATA_TYPE_NAME",
    "FLOAT64_DATA_TYPE",
    "FLOAT64_DATA_TYPE_NAME",
    "INT8_DATA_TYPE",
    "INT8_DATA_TYPE_NAME",
    "INT16_DATA_TYPE",
    "INT16_DATA_TYPE_NAME",
    "INT32_DATA_TYPE",
    "INT32_DATA_TYPE_NAME",
    "INT64_DATA_TYPE",
    "INT64_DATA_TYPE_NAME",
    "NULL_TERMINATED_BYTES_DATA_TYPE",
    "NULL_TERMINATED_BYTES_DATA_TYPE_NAME",
    "NUMPY_DATETIME64_DATA_TYPE",
    "NUMPY_DATETIME64_DATA_TYPE_NAME",
    "NUMPY_TIMEDELTA64_DATA_TYPE",
    "NUMPY_TIMEDELTA64_DATA_TYPE_NAME",
    "NUMPY_TIME_UNIT",
    "RAW_BYTES_DATA_TYPE",
    "STRING_DATA_TYPE",
    "STRING_DATA_TYPE_NAME",
    "STRUCT_DATA_TYPE",
    "STRUCT_DATA_TYPE_NAME",
    "UINT8_DATA_TYPE",
    "UINT8_DATA_TYPE_NAME",
    "UINT16_DATA_TYPE",
    "UINT16_DATA_TYPE_NAME",
    "UINT32_DATA_TYPE",
    "UINT32_DATA_TYPE_NAME",
    "UINT64_DATA_TYPE",
    "UINT64_DATA_TYPE_NAME",
    "VARIABLE_LENGTH_BYTES_DATA_TYPE",
    "VARIABLE_LENGTH_BYTES_DATA_TYPE_NAME",
    "ZARR_PYTHON_RAW_BYTES_DATA_TYPE",
    "ZARR_PYTHON_RAW_BYTES_DATA_TYPE_NAME",
    "Base64Bytes",
    "BoolDataTypeName",
    "BoolFillValue",
    "BytesDataTypeName",
    "BytesFillValue",
    "Complex64Component",
    "Complex64DataTypeName",
    "Complex64FillValue",
    "Complex128Component",
    "Complex128DataTypeName",
    "Complex128FillValue",
    "FixedLengthUtf32",
    "FixedLengthUtf32Configuration",
    "FixedLengthUtf32DataTypeName",
    "FixedLengthUtf32FillValue",
    "Float16DataTypeName",
    "Float16FillValue",
    "Float16SpecialFillValue",
    "Float32DataTypeName",
    "Float32FillValue",
    "Float32SpecialFillValue",
    "Float64DataTypeName",
    "Float64FillValue",
    "Float64SpecialFillValue",
    "HexFloat16",
    "HexFloat32",
    "HexFloat64",
    "Int8DataTypeName",
    "Int8FillValue",
    "Int16DataTypeName",
    "Int16FillValue",
    "Int32DataTypeName",
    "Int32FillValue",
    "Int64DataTypeName",
    "Int64FillValue",
    "NullTerminatedBytes",
    "NullTerminatedBytesConfiguration",
    "NullTerminatedBytesDataTypeName",
    "NullTerminatedBytesFillValue",
    "NumpyDatetime64",
    "NumpyDatetime64Configuration",
    "NumpyDatetime64DataTypeName",
    "NumpyDatetime64FillValue",
    "NumpyTimeUnit",
    "NumpyTimedelta64",
    "NumpyTimedelta64Configuration",
    "NumpyTimedelta64DataTypeName",
    "NumpyTimedelta64FillValue",
    "RawBytesConfiguration",
    "RawBytesDataTypeName",
    "RawBytesFillValue",
    "StringDataTypeName",
    "StringFillValue",
    "Struct",
    "StructConfiguration",
    "StructDataTypeName",
    "StructField",
    "StructFillValue",
    "Uint8DataTypeName",
    "Uint8FillValue",
    "Uint16DataTypeName",
    "Uint16FillValue",
    "Uint32DataTypeName",
    "Uint32FillValue",
    "Uint64DataTypeName",
    "Uint64FillValue",
    "VariableLengthBytesDataTypeName",
    "VariableLengthBytesFillValue",
    "ZarrPythonRawBytes",
    "ZarrPythonRawBytesConfiguration",
    "ZarrPythonRawBytesDataTypeName",
    "ZarrPythonRawBytesFillValue",
    "base64_bytes",
    "hex_float16",
    "hex_float32",
    "hex_float64",
    "raw_bytes_dtype_name",
]
