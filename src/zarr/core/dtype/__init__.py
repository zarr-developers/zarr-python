from __future__ import annotations

from typing import TYPE_CHECKING, Final, TypeAlias

from zarr.core.dtype.common import (
    DataTypeValidationError,
    DTypeJSON,
)
from zarr.core.dtype.npy.bool import Bool
from zarr.core.dtype.npy.bytes import NullTerminatedBytes, RawBytes, VariableLengthBytes
from zarr.core.dtype.npy.complex import Complex64, Complex128
from zarr.core.dtype.npy.float import Float16, Float32, Float64
from zarr.core.dtype.npy.int import Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64
from zarr.core.dtype.npy.structured import (
    Structured,
)
from zarr.core.dtype.npy.time import DateTime64, TimeDelta64

if TYPE_CHECKING:
    from zarr.core.common import ZarrFormat

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

from zarr.core.common import JSON
from zarr.core.dtype.npy.string import (
    FixedLengthUTF32,
    VariableLengthUTF8,
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
    "FixedLengthUTF32",
    "Float16",
    "Float32",
    "Float64",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "NullTerminatedBytes",
    "RawBytes",
    "Structured",
    "TBaseDType",
    "TBaseScalar",
    "TimeDelta64",
    "TimeDelta64",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "VariableLengthUTF8",
    "ZDType",
    "data_type_registry",
    "parse_data_type",
]

data_type_registry = DataTypeRegistry()

IntegerDType = Int8 | Int16 | Int32 | Int64 | UInt8 | UInt16 | UInt32 | UInt64
INTEGER_DTYPE: Final = Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64

FloatDType = Float16 | Float32 | Float64
FLOAT_DTYPE: Final = Float16, Float32, Float64

ComplexFloatDType = Complex64 | Complex128
COMPLEX_FLOAT_DTYPE: Final = Complex64, Complex128

StringDType = FixedLengthUTF32 | VariableLengthUTF8
STRING_DTYPE: Final = FixedLengthUTF32, VariableLengthUTF8

TimeDType = DateTime64 | TimeDelta64
TIME_DTYPE: Final = DateTime64, TimeDelta64

BytesDType = RawBytes | NullTerminatedBytes | VariableLengthBytes
BYTES_DTYPE: Final = RawBytes, NullTerminatedBytes, VariableLengthBytes

AnyDType = (
    Bool
    | IntegerDType
    | FloatDType
    | ComplexFloatDType
    | StringDType
    | BytesDType
    | Structured
    | TimeDType
    | VariableLengthBytes
)
# mypy has trouble inferring the type of variablelengthstring dtype, because its class definition
# depends on the installed numpy version. That's why the type: ignore statement is needed here.
ANY_DTYPE: Final = (
    Bool,
    *INTEGER_DTYPE,
    *FLOAT_DTYPE,
    *COMPLEX_FLOAT_DTYPE,
    *STRING_DTYPE,
    *BYTES_DTYPE,
    Structured,
    *TIME_DTYPE,
    VariableLengthBytes,
)

# These are aliases for variable-length UTF-8 strings
# We handle them when a user requests a data type instead of using NumPy's dtype inferece because
# the default NumPy behavior -- to inspect the user-provided array data and choose
# an appropriately sized U dtype -- is unworkable for Zarr.
VLEN_UTF8_ALIAS: Final = ("str", str, "string")

# This type models inputs that can be coerced to a ZDType
ZDTypeLike: TypeAlias = npt.DTypeLike | ZDType[TBaseDType, TBaseScalar] | Mapping[str, JSON] | str

for dtype in ANY_DTYPE:
    # mypy does not know that all the elements of ANY_DTYPE are subclasses of ZDType
    data_type_registry.register(dtype._zarr_v3_name, dtype)  # type: ignore[arg-type]


# TODO: find a better name for this function
def get_data_type_from_native_dtype(dtype: npt.DTypeLike) -> ZDType[TBaseDType, TBaseScalar]:
    """
    Get a data type wrapper (an instance of ``ZDType``) from a native data type, e.g. a numpy dtype.
    """
    if not isinstance(dtype, np.dtype):
        na_dtype: np.dtype[np.generic]
        if isinstance(dtype, list):
            # this is a valid _VoidDTypeLike check
            na_dtype = np.dtype([tuple(d) for d in dtype])
        else:
            na_dtype = np.dtype(dtype)
    else:
        na_dtype = dtype
    return data_type_registry.match_dtype(dtype=na_dtype)


def get_data_type_from_json(
    dtype_spec: DTypeJSON, *, zarr_format: ZarrFormat
) -> ZDType[TBaseDType, TBaseScalar]:
    """
    Given a JSON representation of a data type and a Zarr format version,
    attempt to create a ZDType instance from the registered ZDType classes.
    """
    return data_type_registry.match_json(dtype_spec, zarr_format=zarr_format)


def parse_data_type(
    dtype_spec: ZDTypeLike,
    *,
    zarr_format: ZarrFormat,
) -> ZDType[TBaseDType, TBaseScalar]:
    """
    Interpret the input as a ZDType instance.
    """
    if isinstance(dtype_spec, ZDType):
        return dtype_spec
    # dict and zarr_format 3 means that we have a JSON object representation of the dtype
    if zarr_format == 3 and isinstance(dtype_spec, Mapping):
        return get_data_type_from_json(dtype_spec, zarr_format=3)
    if dtype_spec in VLEN_UTF8_ALIAS:
        # If the dtype request is one of the aliases for variable-length UTF-8 strings,
        # return that dtype.
        return VariableLengthUTF8()  # type: ignore[return-value]
    # otherwise, we have either a numpy dtype string, or a zarr v3 dtype string, and in either case
    # we can create a numpy dtype from it, and do the dtype inference from that
    return get_data_type_from_native_dtype(dtype_spec)  # type: ignore[arg-type]
