"""
Zarr v3 data type spec types.

Each v3 data type has its own submodule:

- Core primitives: `bool`, `int8`/`16`/`32`/`64`, `uint8`/`16`/`32`/`64`,
  `float16`/`32`/`64`, `complex64`/`128`, `raw` (for `r<N>`)
- zarr-extensions: `bytes`, `string`, `numpy_datetime64`, `numpy_timedelta64`,
  `struct`

The two canonical types per dtype are re-exported here:

- `<X>DataTypeName` -- the literal type of the dtype's `data_type` string
  (or, for named-config dtypes, the literal value of their `name` field)
- `<X>FillValue` -- the permitted JSON shape of the `fill_value` field

Named-config dtypes (`numpy_datetime64`, `numpy_timedelta64`, `struct`) also
expose their envelope TypedDict here. For configuration TypedDicts, branded
`HexFloat<N>` / `Base64Bytes` types, and the corresponding validator
functions, import directly from the leaf submodule.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from zarr_metadata.v3.data_type.bool import BoolDataTypeName, BoolFillValue
from zarr_metadata.v3.data_type.bytes import BytesDataTypeName, BytesFillValue
from zarr_metadata.v3.data_type.complex64 import Complex64DataTypeName, Complex64FillValue
from zarr_metadata.v3.data_type.complex128 import (
    Complex128DataTypeName,
    Complex128FillValue,
)
from zarr_metadata.v3.data_type.float16 import Float16DataTypeName, Float16FillValue
from zarr_metadata.v3.data_type.float32 import Float32DataTypeName, Float32FillValue
from zarr_metadata.v3.data_type.float64 import Float64DataTypeName, Float64FillValue
from zarr_metadata.v3.data_type.int8 import Int8DataTypeName, Int8FillValue
from zarr_metadata.v3.data_type.int16 import Int16DataTypeName, Int16FillValue
from zarr_metadata.v3.data_type.int32 import Int32DataTypeName, Int32FillValue
from zarr_metadata.v3.data_type.int64 import Int64DataTypeName, Int64FillValue
from zarr_metadata.v3.data_type.numpy_datetime64 import (
    NumpyDatetime64,
    NumpyDatetime64DataTypeName,
    NumpyDatetime64FillValue,
)
from zarr_metadata.v3.data_type.numpy_timedelta64 import (
    NumpyTimedelta64,
    NumpyTimedelta64DataTypeName,
    NumpyTimedelta64FillValue,
)
from zarr_metadata.v3.data_type.raw import RawBytesDataTypeName, RawBytesFillValue
from zarr_metadata.v3.data_type.string import StringDataTypeName, StringFillValue
from zarr_metadata.v3.data_type.struct import (
    Struct,
    StructDataTypeName,
    StructFillValue,
)
from zarr_metadata.v3.data_type.uint8 import Uint8DataTypeName, Uint8FillValue
from zarr_metadata.v3.data_type.uint16 import Uint16DataTypeName, Uint16FillValue
from zarr_metadata.v3.data_type.uint32 import Uint32DataTypeName, Uint32FillValue
from zarr_metadata.v3.data_type.uint64 import Uint64DataTypeName, Uint64FillValue

__all__ = [
    "BoolDataTypeName",
    "BoolFillValue",
    "BytesDataTypeName",
    "BytesFillValue",
    "Complex64DataTypeName",
    "Complex64FillValue",
    "Complex128DataTypeName",
    "Complex128FillValue",
    "Float16DataTypeName",
    "Float16FillValue",
    "Float32DataTypeName",
    "Float32FillValue",
    "Float64DataTypeName",
    "Float64FillValue",
    "Int8DataTypeName",
    "Int8FillValue",
    "Int16DataTypeName",
    "Int16FillValue",
    "Int32DataTypeName",
    "Int32FillValue",
    "Int64DataTypeName",
    "Int64FillValue",
    "NumpyDatetime64",
    "NumpyDatetime64DataTypeName",
    "NumpyDatetime64FillValue",
    "NumpyTimedelta64",
    "NumpyTimedelta64DataTypeName",
    "NumpyTimedelta64FillValue",
    "RawBytesDataTypeName",
    "RawBytesFillValue",
    "StringDataTypeName",
    "StringFillValue",
    "Struct",
    "StructDataTypeName",
    "StructFillValue",
    "Uint8DataTypeName",
    "Uint8FillValue",
    "Uint16DataTypeName",
    "Uint16FillValue",
    "Uint32DataTypeName",
    "Uint32FillValue",
    "Uint64DataTypeName",
    "Uint64FillValue",
]
