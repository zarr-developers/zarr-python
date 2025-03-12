from __future__ import annotations

from typing import TYPE_CHECKING, Any, get_args

import numpy as np

from zarr.core.dtype.common import _NUMPY_SUPPORTS_VLEN_STRING

if TYPE_CHECKING:
    import numpy.typing as npt

    from zarr.core.common import JSON

from zarr.core.dtype._numpy import (
    Bool,
    Complex64,
    Complex128,
    DateTime64,
    FixedLengthAsciiString,
    FixedLengthBytes,
    FixedLengthUnicodeString,
    Float16,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    Structured,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    VariableLengthString,
)
from zarr.core.dtype.registry import DataTypeRegistry
from zarr.core.dtype.wrapper import DTypeWrapper

__all__ = [
    "Complex64",
    "Complex128",
    "DTypeWrapper",
    "DateTime64",
    "FixedLengthAsciiString",
    "FixedLengthBytes",
    "FixedLengthUnicodeString",
    "Float16",
    "Float32",
    "Float64",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "Structured",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "VariableLengthString",
    "data_type_registry",
    "parse_data_type",
]

data_type_registry = DataTypeRegistry()

INTEGER_DTYPE = Int8 | Int16 | Int32 | Int64 | UInt8 | UInt16 | UInt32 | UInt64
FLOAT_DTYPE = Float16 | Float32 | Float64
COMPLEX_DTYPE = Complex64 | Complex128
STRING_DTYPE = FixedLengthUnicodeString | VariableLengthString | FixedLengthAsciiString
DTYPE = (
    Bool
    | INTEGER_DTYPE
    | FLOAT_DTYPE
    | COMPLEX_DTYPE
    | STRING_DTYPE
    | FixedLengthBytes
    | Structured
    | DateTime64
)

for dtype in get_args(DTYPE):
    data_type_registry.register(dtype._zarr_v3_name, dtype)


def get_data_type_from_numpy(dtype: npt.DTypeLike) -> DTypeWrapper[Any, Any]:
    data_type_registry.lazy_load()
    if not isinstance(dtype, np.dtype):
        if dtype in (str, "str"):
            if _NUMPY_SUPPORTS_VLEN_STRING:
                np_dtype = np.dtype("T")
            else:
                np_dtype = np.dtype("O")
        elif isinstance(dtype, list):
            # this is a valid _VoidDTypeLike check
            np_dtype = np.dtype([tuple(d) for d in dtype])
        else:
            np_dtype = np.dtype(dtype)
    else:
        np_dtype = dtype
    return data_type_registry.match_dtype(np_dtype)


def get_data_type_from_dict(dtype: dict[str, JSON]) -> DTypeWrapper[Any, Any]:
    return data_type_registry.match_json(dtype)


def parse_data_type(
    dtype: npt.DTypeLike | DTypeWrapper[Any, Any] | dict[str, JSON],
) -> DTypeWrapper[Any, Any]:
    if isinstance(dtype, DTypeWrapper):
        return dtype
    elif isinstance(dtype, dict):
        return get_data_type_from_dict(dtype)
    else:
        return get_data_type_from_numpy(dtype)
