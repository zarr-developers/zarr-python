from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, Literal, Self, TypeGuard, TypeVar, cast, get_args

import numpy as np
import numpy.typing as npt

from zarr.abc.metadata import Metadata
from zarr.core.common import JSON, ZarrFormat
from zarr.core.strings import _NUMPY_SUPPORTS_VLEN_STRING
from zarr.registry import get_data_type_from_dict, register_data_type

Endianness = Literal["little", "big", "native"]
DataTypeFlavor = Literal["boolean", "numeric", "string", "bytes"]
JSONFloat = float | Literal["NaN", "Infinity", "-Infinity"]


def endianness_to_numpy_str(endianness: Endianness | None) -> Literal[">", "<", "=", "|"]:
    match endianness:
        case "little":
            return "<"
        case "big":
            return ">"
        case "native":
            return "="
        case None:
            return "|"
    raise ValueError(
        f"Invalid endianness: {endianness}. Expected one of {get_args(endianness)} or None"
    )


def check_json_bool(data: JSON) -> TypeGuard[bool]:
    return bool(isinstance(data, bool))


def check_json_str(data: JSON) -> TypeGuard[str]:
    return bool(isinstance(data, str))


def check_json_int(data: JSON) -> TypeGuard[int]:
    return bool(isinstance(data, int))


def check_json_float(data: JSON) -> TypeGuard[float]:
    if data == "NaN" or data == "Infinity" or data == "-Infinity":
        return True
    else:
        return bool(isinstance(data, float | int))


def check_json_complex_float(data: JSON) -> TypeGuard[tuple[JSONFloat, JSONFloat]]:
    return (
        not isinstance(data, str)
        and isinstance(data, Sequence)
        and len(data) == 2
        and check_json_float(data[0])
        and check_json_float(data[1])
    )


def float_to_json_v2(data: float | np.floating[Any]) -> JSONFloat:
    if np.isnan(data):
        return "NaN"
    elif np.isinf(data):
        return "Infinity" if data > 0 else "-Infinity"
    return float(data)


def float_to_json_v3(data: float | np.floating[Any]) -> JSONFloat:
    # v3 can in principle handle distinct NaN values, but numpy does not represent these explicitly
    # so we just re-use the v2 routine here
    return float_to_json_v2(data)


def float_to_json(data: float | np.floating[Any], zarr_format: ZarrFormat) -> JSONFloat:
    """
    convert a float to JSON as per the zarr v3 spec
    """
    if zarr_format == 2:
        return float_to_json_v2(data)
    else:
        return float_to_json_v3(data)
    raise ValueError(f"Invalid zarr format: {zarr_format}. Expected 2 or 3.")


def complex_to_json_v2(data: complex | np.complexfloating) -> JSONFloat:
    return float_to_json_v2(data)


def complex_to_json_v3(data: complex | np.complexfloating) -> tuple[JSONFloat, JSONFloat]:
    return float_to_json_v3(data.real), float_to_json_v3(data.imag)


def complex_to_json(
    data: complex | np.complexfloating, zarr_format: ZarrFormat
) -> tuple[JSONFloat, JSONFloat] | JSONFloat:
    if zarr_format == 2:
        return complex_to_json_v2(data)
    else:
        return complex_to_json_v3(data)
    raise ValueError(f"Invalid zarr format: {zarr_format}. Expected 2 or 3.")


def float_from_json_v2(data: JSONFloat, dtype: np.floating[Any]) -> np.floating[Any]:
    if data == "NaN":
        _data = np.nan
    elif data == "Infinity":
        _data = np.inf
    elif data == "-Infinity":
        _data = -np.inf
    else:
        _data = data
    return dtype.type(_data)


def float_from_json_v3(data: JSONFloat, dtype: Any) -> np.floating[Any]:
    # todo: support the v3-specific NaN handling
    return float_from_json_v2(data, dtype)


def float_from_json(data: JSONFloat, dtype: Any, zarr_format: ZarrFormat) -> np.floating[Any]:
    if zarr_format == 2:
        return float_from_json_v2(data, dtype)
    else:
        return float_from_json_v3(data, dtype)
    raise ValueError(f"Invalid zarr format: {zarr_format}. Expected 2 or 3.")


def complex_from_json_v2(data: JSONFloat, dtype: Any) -> np.complexfloating:
    return dtype.type(data)


def complex_from_json_v3(data: tuple[JSONFloat, JSONFloat], dtype: Any) -> np.complexfloating:
    return dtype.type(complex(*data))


def complex_from_json(
    data: tuple[JSONFloat, JSONFloat], dtype: Any, zarr_format: ZarrFormat
) -> np.complexfloating:
    if zarr_format == 2:
        return complex_from_json_v2(data, dtype)
    else:
        if check_json_complex_float(data):
            return complex_from_json_v3(data, dtype)
        else:
            raise TypeError(f"Invalid type: {data}. Expected a sequence of two numbers.")
    raise ValueError(f"Invalid zarr format: {zarr_format}. Expected 2 or 3.")

TDType = TypeVar("TDType", bound=np.dtype[Any])
TScalar = TypeVar("TScalar", bound=np.generic)

@dataclass(frozen=True, kw_only=True)
class Flexible:
    length: int

class DTypeWrapper(Generic[TDType, TScalar], ABC, Metadata):
    name: ClassVar[str]
    dtype_cls: ClassVar[type[TDType]] # this class will create a numpy dtype
    kind: ClassVar[DataTypeFlavor]
    _default_value: object

    def to_dict(self) -> dict[str, JSON]:
        return {"name": self.name}

    def default_value(self: Self, *, endianness: Endianness | None = None) -> TScalar:
        return cast(np.generic, self.to_numpy(endianness=endianness).type(self._default_value))

    def to_numpy(self: Self, *, endianness: Endianness | None = None) -> TDType:
        endian_str = endianness_to_numpy_str(endianness)
        return self.dtype_cls().newbyteorder(endian_str)

    @abstractmethod
    def to_json_value(self, data: np.generic, *, zarr_format: ZarrFormat) -> JSON:
        """
        Convert a single value to JSON-serializable format. Depends on the zarr format.
        """
        raise NotImplementedError

    @abstractmethod
    def from_json_value(
        self: Self, data: JSON, *, zarr_format: ZarrFormat, endianness: Endianness | None = None
    ) -> TScalar:
        """
        Read a JSON-serializable value as a numpy scalar
        """
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class Bool(DTypeWrapper[np.dtypes.BoolDType, np.bool_]):
    name = "bool"
    kind = "boolean"
    default = False

    def to_json_value(self, data: np.generic, zarr_format: ZarrFormat) -> bool:
        return bool(data)

    def from_json_value(
        self, data: JSON, *, zarr_format: ZarrFormat, endianness: Endianness | None = None
    ) -> np.bool_:
        if check_json_bool(data):
            return self.to_numpy(endianness=endianness).type(data)
        raise TypeError(f"Invalid type: {data}. Expected a boolean.")


register_data_type(Bool)

class BaseInt(DTypeWrapper[TDType, TScalar]):
    kind = "numeric"
    default = 0

    def to_json_value(self, data: np.generic, zarr_format: ZarrFormat) -> int:
        return int(data)

    def from_json_value(
        self, data: JSON, *, zarr_format: ZarrFormat, endianness: Endianness | None = None
    ) -> TScalar:
        if check_json_int(data):
            return self.to_numpy(endianness=endianness).type(data)
        raise TypeError(f"Invalid type: {data}. Expected an integer.")


@dataclass(frozen=True, kw_only=True)
class Int8(BaseInt[np.dtypes.Int8DType, np.int8]):
    name = "int8"


register_data_type(Int8)


@dataclass(frozen=True, kw_only=True)
class UInt8(DTypeWrapper[np.dtypes.UInt8DType, np.uint8]):
    name = "uint8"


register_data_type(UInt8)


@dataclass(frozen=True, kw_only=True)
class Int16(DTypeWrapper[np.dtypes.Int16DType, np.int16]):
    name = "int16"


register_data_type(Int16)


@dataclass(frozen=True, kw_only=True)
class UInt16(DTypeWrapper[np.dtypes.UInt16DType, np.uint16]):
    name = "uint16"

register_data_type(UInt16)


@dataclass(frozen=True, kw_only=True)
class Int32(DTypeWrapper[np.dtypes.Int32DType, np.int32]):
    name = "int32"


register_data_type(Int32)


@dataclass(frozen=True, kw_only=True)
class UInt32(DTypeWrapper[np.dtypes.UInt32DType, np.uint32]):
    name = "uint32"

register_data_type(UInt32)


@dataclass(frozen=True, kw_only=True)
class Int64(DTypeWrapper[np.dtypes.Int64DType, np.int64]):
    name = "int64"


register_data_type(Int64)


@dataclass(frozen=True, kw_only=True)
class UInt64(DTypeWrapper[np.dtypes.UInt64DType, np.uint64]):
    name = "uint64"



register_data_type(UInt64)


class FloatBase(DTypeWrapper[TDType, TScalar]):
    kind = "numeric"
    default = 0.0

    def to_json_value(self, data: np.generic, zarr_format: ZarrFormat) -> JSONFloat:
        return float_to_json(data, zarr_format)

    def from_json_value(
        self, data: JSON, *, zarr_format: ZarrFormat, endianness: Endianness | None = None
    ) -> TScalar:
        if check_json_float(data):
            return self.to_numpy(endianness=endianness).type(float_from_json)(data, zarr_format)
        raise TypeError(f"Invalid type: {data}. Expected a float.")

@dataclass(frozen=True, kw_only=True)
class Float16(DTypeWrapper[np.dtypes.Float16DType, np.float16]):
    name = "float16"


register_data_type(Float16)


@dataclass(frozen=True, kw_only=True)
class Float32(DTypeWrapper[np.dtypes.Float32DType, np.float32]):
    name = "float32"
 

register_data_type(Float32)


@dataclass(frozen=True, kw_only=True)
class Float64(DTypeWrapper[np.dtypes.Float64DType, np.float64]):
    name = "float64"


register_data_type(Float64)


@dataclass(frozen=True, kw_only=True)
class Complex64(DTypeWrapper[np.dtypes.Complex64DType, np.complex64]):
    name = "complex64"
    kind = "numeric"
    default = 0.0 + 0.0j

    def to_json_value(
        self, data: np.generic, zarr_format: ZarrFormat
    ) -> tuple[JSONFloat, JSONFloat]:
        return complex_to_json(data, zarr_format)

    def from_json_value(
        self, data: JSON, *, zarr_format: ZarrFormat, endianness: Endianness | None = None
    ) -> np.complex64:
        if check_json_complex_float(data):
            return complex_from_json(
                data, dtype=self.to_numpy(endianness=endianness), zarr_format=zarr_format
            )
        raise TypeError(f"Invalid type: {data}. Expected a complex float.")


register_data_type(Complex64)


@dataclass(frozen=True, kw_only=True)
class Complex128(DTypeWrapper[np.dtypes.Complex128DType, np.complex128]):
    name = "complex128"
    kind = "numeric"
    dtype_cls = np.dtypes.Complex128DType
    default = 0.0 + 0.0j

    def to_json_value(
        self, data: np.generic, zarr_format: ZarrFormat
    ) -> tuple[JSONFloat, JSONFloat]:
        return complex_to_json(data, zarr_format)

    def from_json_value(
        self, data: JSON, *, zarr_format: ZarrFormat, endianness: Endianness | None = None
    ) -> np.complex128:
        if check_json_complex_float(data):
            return complex_from_json(
                data, dtype=self.to_numpy(endianness=endianness), zarr_format=zarr_format
            )
        raise TypeError(f"Invalid type: {data}. Expected a complex float.")


register_data_type(Complex128)


@dataclass(frozen=True, kw_only=True)
class StaticByteString(DTypeWrapper[np.dtypes.BytesDType, np.bytes_], Flexible):
    name = "numpy/static_byte_string"
    kind = "string"
    default = b""

    def to_dict(self) -> dict[str, JSON]:
        return {"name": self.name, "configuration": {"capacity": self.length}}

    def to_numpy(self, endianness: Endianness | None = "native") -> np.dtypes.BytesDType:
        endianness_code = endianness_to_numpy_str(endianness)
        return self.dtype_cls(self.length).newbyteorder(endianness_code)

    def to_json_value(
        self, data: np.generic, *, zarr_format: ZarrFormat, endianness: Endianness | None = None
    ) -> str:
        return data.tobytes().decode("ascii")

    def from_json_value(
        self, data: JSON, *, zarr_format: ZarrFormat, endianness: Endianness | None = None
    ) -> np.bytes_:
        if check_json_bool(data):
            return self.to_numpy(endianness=endianness).type(data.encode("ascii"))
        raise TypeError(f"Invalid type: {data}. Expected a string.")


register_data_type(StaticByteString)


@dataclass(frozen=True, kw_only=True)
class StaticRawBytes(DTypeWrapper[np.dtypes.VoidDType, np.void], Flexible):
    name = "r*"
    kind = "bytes"
    default = b""


    def to_dict(self) -> dict[str, JSON]:
        return {"name": f"r{self.length * 8}"}

    def to_numpy(self, endianness: Endianness | None = "native") -> np.dtypes.VoidDType:
        # this needs to be overridden because numpy does not allow creating a void type
        # by invoking np.dtypes.VoidDType directly
        endianness_code = endianness_to_numpy_str(endianness)
        return np.dtype(f'{endianness_code}V{self.length}')

    def to_json_value(self, data: np.generic, *, zarr_format: ZarrFormat) -> tuple[int, ...]:
        return tuple(*data.tobytes())

    def from_json_value(
        self, data: JSON, *, zarr_format: ZarrFormat, endianness: Endianness | None = None
    ) -> np.void:
        # todo: check that this is well-formed
        return self.to_numpy(endianness=endianness).type(bytes(data))


register_data_type(StaticRawBytes)

if _NUMPY_SUPPORTS_VLEN_STRING:

    @dataclass(frozen=True, kw_only=True)
    class VlenString(DTypeWrapper[np.dtypes.StringDType, str]):
        name = "numpy/vlen_string"
        kind = "string"
        dtype_cls = np.dtypes.StringDType
        default = ""

        def to_dict(self) -> dict[str, JSON]:
            return {"name": self.name}

        def to_numpy(
            self, endianness: Endianness | None = "native"
        ) -> np.dtypes.StringDType:
            endianness_code = endianness_to_numpy_str(endianness)
            return np.dtype(endianness_code + self.numpy_character_code)

        def to_json_value(self, data: np.generic, *, zarr_format: ZarrFormat) -> str:
            return str(data)

        def from_json_value(
            self, data: JSON, *, zarr_format: ZarrFormat, endianness: Endianness | None = None
        ) -> str:
            return self.to_numpy(endianness=endianness).type(data)

else:

    @dataclass(frozen=True, kw_only=True)
    class VlenString(DTypeWrapper[np.dtypes.ObjectDType, str]):
        name = "numpy/vlen_string"
        kind = "string"
        dtype_cls = np.dtypes.ObjectDType
        default = ""

        def to_dict(self) -> dict[str, JSON]:
            return {"name": self.name}

        def to_numpy(
            self, endianness: Endianness | None = None
        ) -> np.dtype[np.dtypes.ObjectDType]:
            return super().to_numpy(endianness=endianness)

        def to_json_value(self, data, *, zarr_format: ZarrFormat) -> str:
            return str(data)

        def from_json_value(
            self, data: JSON, *, zarr_format: ZarrFormat, endianness: Endianness | None = None
        ) -> str:
            return self.to_numpy(endianness=endianness).type(data)


register_data_type(VlenString)


@dataclass(frozen=True, kw_only=True)
class StaticUnicodeString(DTypeWrapper[np.dtypes.StrDType, np.str_], Flexible):
    name = "numpy/static_unicode_string"
    kind = "string"
    default = ""

    def to_dict(self) -> dict[str, JSON]:
        return {"name": self.name, "configuration": {"capacity": self.length}}

    def to_numpy(self, endianness: Endianness | None = "native") -> np.dtypes.StrDType:
        endianness_code = endianness_to_numpy_str(endianness)
        return np.dtype(endianness_code + self.numpy_character_code + str(self.length))

    def to_json_value(self, data: np.generic, *, zarr_format: ZarrFormat) -> str:
        return str(data)

    def from_json_value(
        self, data: JSON, *, zarr_format: ZarrFormat, endianness: Endianness | None = None
    ) -> np.str_:
        if not check_json_bool(data):
            raise TypeError(f"Invalid type: {data}. Expected a string.")
        return self.to_numpy(endianness=endianness).type(data)


register_data_type(StaticUnicodeString)


def resolve_dtype(dtype: npt.DTypeLike | DTypeWrapper | dict[str, JSON]) -> DTypeWrapper:
    from zarr.registry import get_data_type_from_numpy

    if isinstance(dtype, DTypeWrapper):
        return dtype
    elif isinstance(dtype, dict):
        return get_data_type_from_dict(dtype)
    else:
        return get_data_type_from_numpy(dtype)


INTEGER_DTYPE = Int8 | Int16 | Int32 | Int64 | UInt8 | UInt16 | UInt32 | UInt64
FLOAT_DTYPE = Float16 | Float32 | Float64
COMPLEX_DTYPE = Complex64 | Complex128
STRING_DTYPE = StaticUnicodeString | VlenString | StaticByteString
