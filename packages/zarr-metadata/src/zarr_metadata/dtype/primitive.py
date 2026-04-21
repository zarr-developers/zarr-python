"""
Primitive Zarr v3 data types.

All primitives are spec-defined as bare-string `data_type` values -- they
carry no configuration.

See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#data-types
"""

from typing import Final, Literal

BOOL_DTYPE_NAME: Final = "bool"
INT8_DTYPE_NAME: Final = "int8"
INT16_DTYPE_NAME: Final = "int16"
INT32_DTYPE_NAME: Final = "int32"
INT64_DTYPE_NAME: Final = "int64"
UINT8_DTYPE_NAME: Final = "uint8"
UINT16_DTYPE_NAME: Final = "uint16"
UINT32_DTYPE_NAME: Final = "uint32"
UINT64_DTYPE_NAME: Final = "uint64"
FLOAT16_DTYPE_NAME: Final = "float16"
FLOAT32_DTYPE_NAME: Final = "float32"
FLOAT64_DTYPE_NAME: Final = "float64"
COMPLEX64_DTYPE_NAME: Final = "complex64"
COMPLEX128_DTYPE_NAME: Final = "complex128"

PrimitiveDTypeName = Literal[
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
]
"""Literal union of every core v3 primitive data type name."""


__all__ = [
    "BOOL_DTYPE_NAME",
    "COMPLEX64_DTYPE_NAME",
    "COMPLEX128_DTYPE_NAME",
    "FLOAT16_DTYPE_NAME",
    "FLOAT32_DTYPE_NAME",
    "FLOAT64_DTYPE_NAME",
    "INT8_DTYPE_NAME",
    "INT16_DTYPE_NAME",
    "INT32_DTYPE_NAME",
    "INT64_DTYPE_NAME",
    "UINT8_DTYPE_NAME",
    "UINT16_DTYPE_NAME",
    "UINT32_DTYPE_NAME",
    "UINT64_DTYPE_NAME",
    "PrimitiveDTypeName",
]
