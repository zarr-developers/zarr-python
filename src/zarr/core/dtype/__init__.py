from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

from zarr.core.dtype.common import DataTypeValidationError
from zarr.core.dtype.npy.bool import Bool
from zarr.core.dtype.npy.complex import Complex64, Complex128
from zarr.core.dtype.npy.float import Float16, Float32, Float64
from zarr.core.dtype.npy.int import Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64
from zarr.core.dtype.npy.sized import (
    FixedLengthAscii,
    FixedLengthBytes,
    FixedLengthUnicode,
    Structured,
)
from zarr.core.dtype.npy.time import DateTime64, TimeDelta64

if TYPE_CHECKING:
    from zarr.core.common import ZarrFormat

import numpy as np
import numpy.typing as npt

from zarr.core.common import JSON
from zarr.core.dtype.npy.string import (
    _NUMPY_SUPPORTS_VLEN_STRING,
    VariableLengthString,
)
from zarr.core.dtype.registry import DataTypeRegistry
from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType

__all__ = [
    "Bool",
    "Complex64",
    "Complex128",
    "DataTypeRegistry",
    "DataTypeValidationError",
    "DateTime64",
    "FixedLengthAscii",
    "FixedLengthBytes",
    "FixedLengthUnicode",
    "Float16",
    "Float32",
    "Float64",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "Structured",
    "TBaseDType",
    "TBaseScalar",
    "TimeDelta64",
    "TimeDelta64",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "VariableLengthString",
    "ZDType",
    "data_type_registry",
    "parse_data_type",
]

data_type_registry = DataTypeRegistry()

IntegerDType = Int8 | Int16 | Int32 | Int64 | UInt8 | UInt16 | UInt32 | UInt64
INTEGER_DTYPE = Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64

FloatDType = Float16 | Float32 | Float64
FLOAT_DTYPE = Float16, Float32, Float64

ComplexFloatDType = Complex64 | Complex128
COMPLEX_FLOAT_DTYPE = Complex64, Complex128

StringDType = FixedLengthUnicode | VariableLengthString | FixedLengthAscii
STRING_DTYPE = FixedLengthUnicode, VariableLengthString, FixedLengthAscii

TimeDType = DateTime64 | TimeDelta64
TIME_DTYPE = DateTime64, TimeDelta64

AnyDType = (
    Bool
    | IntegerDType
    | FloatDType
    | ComplexFloatDType
    | StringDType
    | FixedLengthBytes
    | Structured
    | TimeDType
)
# mypy has trouble inferring the type of variablelengthstring dtype, because its class definition
# depends on the installed numpy version. That's why the type: ignore statement is needed here.
ANY_DTYPE: tuple[type[ZDType[TBaseDType, TBaseScalar]], ...] = (  # type: ignore[assignment]
    Bool,
    *INTEGER_DTYPE,
    *FLOAT_DTYPE,
    *COMPLEX_FLOAT_DTYPE,
    *STRING_DTYPE,
    FixedLengthBytes,
    Structured,
    *TIME_DTYPE,
)

ZDTypeLike: TypeAlias = npt.DTypeLike | ZDType[TBaseDType, TBaseScalar] | dict[str, JSON]

for dtype in ANY_DTYPE:
    data_type_registry.register(dtype._zarr_v3_name, dtype)


# TODO: find a better name for this function
def get_data_type_from_native_dtype(dtype: npt.DTypeLike) -> ZDType[TBaseDType, TBaseScalar]:
    """
    Get a data type wrapper (an instance of ``ZDType``) from a native data type, e.g. a numpy dtype.
    """
    data_type_registry.lazy_load()
    if not isinstance(dtype, np.dtype):
        # TODO: This check has a lot of assumptions in it! Chiefly, we assume that the
        # numpy object dtype contains variable length strings, which is not in general true
        # When / if zarr python supports ragged arrays, for example, this check will fail!
        if dtype in (str, "str", "|T16", "O", "|O", np.dtypes.ObjectDType()):
            if _NUMPY_SUPPORTS_VLEN_STRING:
                na_dtype = np.dtype("T")
            else:
                na_dtype = np.dtype("O")
        elif isinstance(dtype, list):
            # this is a valid _VoidDTypeLike check
            na_dtype = np.dtype([tuple(d) for d in dtype])
        else:
            na_dtype = np.dtype(dtype)
    else:
        na_dtype = dtype
    return data_type_registry.match_dtype(na_dtype)


def get_data_type_from_json(
    dtype: JSON, zarr_format: ZarrFormat
) -> ZDType[TBaseDType, TBaseScalar]:
    return data_type_registry.match_json(dtype, zarr_format=zarr_format)


def parse_data_type(dtype: ZDTypeLike, zarr_format: ZarrFormat) -> ZDType[TBaseDType, TBaseScalar]:
    """
    Interpret the input as a ZDType instance.
    """
    if isinstance(dtype, ZDType):
        return dtype
    elif isinstance(dtype, dict):
        # This branch assumes that the data type has been specified in the JSON form
        # but it's also possible for numpy data types to be specified as dictionaries, which will
        # cause an error in the `get_data_type_from_json`, but that's ok for now
        return get_data_type_from_json(dtype, zarr_format=zarr_format)  # type: ignore[arg-type]
    else:
        return get_data_type_from_native_dtype(dtype)
