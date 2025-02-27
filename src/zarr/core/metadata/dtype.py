from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Self, TypeGuard, cast, get_args

import numpy as np
import numpy.typing as npt

from zarr.abc.metadata import Metadata
from zarr.core.common import JSON, ZarrFormat
from zarr.core.strings import _NUMPY_SUPPORTS_VLEN_STRING
from zarr.registry import register_data_type

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


def check_json_int(data: JSON) -> TypeGuard[int]:
    return bool(isinstance(data, int))


def check_json_float(data: JSON) -> TypeGuard[float]:
    if data == "NaN" or data == "Infinity" or data == "-Infinity":
        return True
    else:
        return bool(isinstance(data, float))


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


def complex_to_json_v2(data: complex | np.complex_) -> JSONFloat:
    return float_to_json_v2(data)


def complex_to_json_v3(data: complex | np.complex_) -> tuple[JSONFloat, JSONFloat]:
    return float_to_json_v3(data.real), float_to_json_v3(data.imag)


def complex_to_json(
    data: complex | np.complex_, zarr_format: ZarrFormat
) -> tuple[JSONFloat, JSONFloat] | JSONFloat:
    if zarr_format == 2:
        return complex_to_json_v2(data)
    else:
        return complex_to_json_v3(data)
    raise ValueError(f"Invalid zarr format: {zarr_format}. Expected 2 or 3.")


def float_from_json_v2(data: JSONFloat, dtype: np.floating[Any]) -> np.float_:
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


def complex_from_json_v2(data: JSONFloat, dtype: Any) -> np.complex_:
    return dtype.type(data)


def complex_from_json_v3(data: tuple[JSONFloat, JSONFloat], dtype: Any) -> np.complex_:
    return dtype.type(data[0] + 1j * data[1])


def complex_from_json(
    data: tuple[JSONFloat, JSONFloat], dtype: Any, zarr_format: ZarrFormat
) -> np.complex_:
    if zarr_format == 2:
        return complex_from_json_v2(data, dtype)
    else:
        return complex_from_json_v3(data, dtype)
    raise ValueError(f"Invalid zarr format: {zarr_format}. Expected 2 or 3.")


@dataclass(frozen=True, kw_only=True)
class Flexible:
    length: int


class DTypeBase(ABC, Metadata):
    name: ClassVar[str]
    numpy_character_code: ClassVar[str]
    item_size: ClassVar[int | None]
    kind: ClassVar[DataTypeFlavor]
    default: object

    def __init_subclass__(cls, **kwargs: object) -> None:
        required_attrs = ["name", "numpy_character_code", "item_size", "kind", "default"]
        for attr in required_attrs:
            if not hasattr(cls, attr):
                raise ValueError(f"{attr} is a required attribute for a Zarr dtype.")

        return super().__init_subclass__(**kwargs)

    def to_dict(self) -> dict[str, JSON]:
        return {"name": self.name}

    @classmethod
    def from_numpy(cls, dtype: npt.DTypeLike) -> Self:
        if np.dtype(dtype).char != cls.numpy_character_code:
            raise ValueError(
                f"Invalid dtype {dtype}. Expected dtype with character code == {cls.numpy_character_code}."
            )
        return cls()

    def default_value(self: Self, *, endianness: Endianness | None = None) -> np.generic:
        return cast(np.generic, self.to_numpy(endianness=endianness).type(self.default))

    def to_numpy(self: Self, *, endianness: Endianness | None = None) -> np.dtype[Any]:
        endian_str = endianness_to_numpy_str(endianness)
        return np.dtype(endian_str + self.numpy_character_code)

    @abstractmethod
    def to_json_value(self, data: np.generic, *, zarr_format: ZarrFormat) -> JSON:
        """
        Convert a single value to JSON-serializable format. Depends on the zarr format.
        """
        raise NotImplementedError

    @abstractmethod
    def from_json_value(
        self: Self, data: JSON, *, zarr_format: ZarrFormat, endianness: Endianness | None = None
    ) -> np.generic:
        """
        Read a JSON-serializable value as a numpy scalar
        """
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class Bool(DTypeBase):
    name = "bool"
    item_size = 1
    kind = "boolean"
    numpy_character_code = "?"
    default = False

    def to_numpy(self, *, endianness: Endianness | None = None) -> np.dtypes.BoolDType:
        return super().to_numpy(endianness=endianness)

    def to_json_value(self, data: np.generic, zarr_format: ZarrFormat) -> bool:
        return bool(data)

    def from_json_value(
        self, data: JSON, *, zarr_format: ZarrFormat, endianness: Endianness | None = None
    ) -> np.bool_:
        if check_json_bool(data):
            return self.to_numpy(endianness=endianness).type(data)
        raise TypeError(f"Invalid type: {data}. Expected a boolean.")


register_data_type(Bool)


@dataclass(frozen=True, kw_only=True)
class Int8(DTypeBase):
    name = "int8"
    item_size = 1
    kind = "numeric"
    numpy_character_code = "b"
    default = 0

    def to_numpy(self, *, endianness: Endianness | None = None) -> np.dtypes.Int8DType:
        return super().to_numpy(endianness=endianness)

    def to_json_value(self, data: np.generic) -> int:
        return int(data)

    def from_json_value(
        self, data: JSON, *, zarr_format: ZarrFormat, endianness: Endianness | None = None
    ) -> np.int8:
        if check_json_int(data):
            return self.to_numpy(endianness=endianness).type(data)
        raise TypeError(f"Invalid type: {data}. Expected an integer.")


register_data_type(Int8)


@dataclass(frozen=True, kw_only=True)
class UInt8(DTypeBase):
    name = "uint8"
    item_size = 2
    kind = "numeric"
    numpy_character_code = "B"
    default = 0

    def to_numpy(self, *, endianness: Endianness | None = None) -> np.dtypes.UInt8DType:
        return super().to_numpy(endianness=endianness)

    def to_json_value(self, data: np.generic) -> int:
        return int(data)

    def from_json_value(
        self, data: JSON, *, zarr_format: ZarrFormat, endianness: Endianness | None = None
    ) -> np.uint8:
        if check_json_int(data):
            return self.to_numpy(endianness=endianness).type(data)
        raise TypeError(f"Invalid type: {data}. Expected an integer.")


register_data_type(UInt8)


@dataclass(frozen=True, kw_only=True)
class Int16(DTypeBase):
    name = "int16"
    item_size = 2
    kind = "numeric"
    numpy_character_code = "h"
    default = 0

    def to_numpy(self, *, endianness: Endianness | None = None) -> np.dtypes.Int16DType:
        return super().to_numpy(endianness=endianness)

    def to_json_value(self, data: np.generic) -> int:
        return int(data)

    def from_json_value(
        self, data: JSON, *, zarr_format: ZarrFormat, endianness: Endianness | None = None
    ) -> np.int16:
        if check_json_int(data):
            return self.to_numpy(endianness=endianness).type(data)
        raise TypeError(f"Invalid type: {data}. Expected an integer.")


register_data_type(Int16)


@dataclass(frozen=True, kw_only=True)
class UInt16(DTypeBase):
    name = "uint16"
    item_size = 2
    kind = "numeric"
    numpy_character_code = "H"
    default = 0

    def to_numpy(self, *, endianness: Endianness | None = None) -> np.dtypes.UInt16DType:
        return super().to_numpy(endianness=endianness)

    def to_json_value(self, data: np.generic) -> int:
        return int(data)

    def from_json_value(
        self, data: JSON, *, zarr_format: ZarrFormat, endianness: Endianness | None = None
    ) -> np.uint16:
        if check_json_int(data):
            return self.to_numpy(endianness=endianness).type(data)
        raise TypeError(f"Invalid type: {data}. Expected an integer.")


register_data_type(UInt16)


@dataclass(frozen=True, kw_only=True)
class Int32(DTypeBase):
    name = "int32"
    item_size = 4
    kind = "numeric"
    numpy_character_code = "i"
    default = 0

    def to_numpy(self, *, endianness: Endianness | None = None) -> np.dtypes.Int32DType:
        return super().to_numpy(endianness=endianness)

    def to_json_value(self, data: np.generic) -> int:
        return int(data)

    def from_json_value(
        self, data: JSON, *, zarr_format: ZarrFormat, endianness: Endianness | None = None
    ) -> np.int32:
        if check_json_int(data):
            return self.to_numpy(endianness=endianness).type(data)
        raise TypeError(f"Invalid type: {data}. Expected an integer.")


register_data_type(Int32)


@dataclass(frozen=True, kw_only=True)
class UInt32(DTypeBase):
    name = "uint32"
    item_size = 4
    kind = "numeric"
    numpy_character_code = "I"
    default = 0

    def to_numpy(self, *, endianness: Endianness | None = None) -> np.dtypes.UInt32DType:
        return super().to_numpy(endianness=endianness)

    def to_json_value(self, data: np.generic) -> int:
        return int(data)

    def from_json_value(
        self, data: JSON, *, zarr_format: ZarrFormat, endianness: Endianness | None = None
    ) -> np.uint32:
        if check_json_int(data):
            return self.to_numpy(endianness=endianness).type(data)
        raise TypeError(f"Invalid type: {data}. Expected an integer.")


register_data_type(UInt32)


@dataclass(frozen=True, kw_only=True)
class Int64(DTypeBase):
    name = "int64"
    item_size = 8
    kind = "numeric"
    numpy_character_code = "l"
    default = 0

    def to_numpy(self, *, endianness: Endianness | None = None) -> np.dtypes.Int64DType:
        return super().to_numpy(endianness=endianness)

    def to_json_value(self, data: np.generic) -> int:
        return int(data)

    def from_json_value(self, data: JSON, *, endianness: Endianness | None = None) -> np.int64:
        if check_json_int(data):
            return self.to_numpy(endianness=endianness).type(data)
        raise TypeError(f"Invalid type: {data}. Expected an integer.")


register_data_type(Int64)


@dataclass(frozen=True, kw_only=True)
class UInt64(DTypeBase):
    name = "uint64"
    item_size = 8
    kind = "numeric"
    numpy_character_code = "L"
    default = 0

    def to_numpy(self, *, endianness: Endianness | None = None) -> np.dtypes.UInt64DType:
        return super().to_numpy(endianness=endianness)

    def to_json_value(self, data: np.generic) -> int:
        return int(data)

    def from_json_value(self, data: JSON, *, endianness: Endianness | None = None) -> np.uint64:
        if check_json_int(data):
            return self.to_numpy(endianness=endianness).type(data)
        raise TypeError(f"Invalid type: {data}. Expected an integer.")


register_data_type(UInt64)


@dataclass(frozen=True, kw_only=True)
class Float16(DTypeBase):
    name = "float16"
    item_size = 2
    kind = "numeric"
    numpy_character_code = "e"
    default = 0.0

    def to_numpy(self, *, endianness: Endianness | None = None) -> np.dtypes.Float16DType:
        return super().to_numpy(endianness=endianness)

    def to_json_value(self, data: np.generic) -> float:
        return float(data)

    def from_json_value(self, data: JSON, *, endianness: Endianness | None = None) -> np.float16:
        if check_json_float(data):
            return self.to_numpy(endianness=endianness).type(data)
        raise TypeError(f"Invalid type: {data}. Expected a float.")


register_data_type(Float16)


@dataclass(frozen=True, kw_only=True)
class Float32(DTypeBase):
    name = "float32"
    item_size = 4
    kind = "numeric"
    numpy_character_code = "f"
    default = 0.0

    def to_numpy(self, *, endianness: Endianness | None = None) -> np.dtypes.Float32DType:
        return super().to_numpy(endianness=endianness)

    def to_json_value(self, data: np.generic) -> float:
        return float(data)

    def from_json_value(self, data: JSON, *, endianness: Endianness | None = None) -> np.float32:
        if check_json_float(data):
            return self.to_numpy(endianness=endianness).type(data)
        raise TypeError(f"Invalid type: {data}. Expected a float.")


register_data_type(Float32)


@dataclass(frozen=True, kw_only=True)
class Float64(DTypeBase):
    name = "float64"
    item_size = 8
    kind = "numeric"
    numpy_character_code = "d"
    default = 0.0

    def to_numpy(self, *, endianness: Endianness | None = None) -> np.dtypes.Float64DType:
        return super().to_numpy(endianness=endianness)

    def to_json_value(self, data: np.generic) -> float:
        return float(data)

    def from_json_value(self, data: JSON, *, endianness: Endianness | None = None) -> np.float64:
        if check_json_float(data):
            return float_from_json(data, dtype=self.to_numpy(endianness=endianness))
        raise TypeError(f"Invalid type: {data}. Expected a float.")


register_data_type(Float64)


@dataclass(frozen=True, kw_only=True)
class Complex64(DTypeBase):
    name = "complex64"
    item_size = 16
    kind = "numeric"
    numpy_character_code = "F"
    default = 0.0 + 0.0j

    def to_numpy(self, *, endianness: Endianness | None = None) -> np.dtypes.Complex64DType:
        return super().to_numpy(endianness=endianness)

    def to_json_value(self, data: np.generic) -> float:
        return float(data)

    def from_json_value(self, data: JSON, *, endianness: Endianness | None = None) -> np.complex64:
        if check_json_float(data):
            return self.to_numpy(endianness=endianness).type(data)
        raise TypeError(f"Invalid type: {data}. Expected a float.")


register_data_type(Complex64)


@dataclass(frozen=True, kw_only=True)
class Complex128(DTypeBase):
    name = "complex64"
    item_size = 32
    kind = "numeric"
    numpy_character_code = "D"
    default = 0.0 + 0.0j

    def to_numpy(self, *, endianness: Endianness | None = None) -> np.dtypes.Complex128DType:
        return super().to_numpy(endianness=endianness)


register_data_type(Complex128)


@dataclass(frozen=True, kw_only=True)
class StaticByteString(DTypeBase, Flexible):
    name = "numpy/static_byte_string"
    kind = "string"
    numpy_character_code = "S"
    item_size = 1
    default = b""

    @classmethod
    def from_numpy(cls: type[Self], dtype: npt.DTypeLike) -> Self:
        dtype = np.dtype(dtype)
        if dtype.kind != cls.numpy_character_code:
            raise ValueError(f"Invalid dtype {dtype}. Expected a string dtype.")
        return cls(length=dtype.itemsize)

    def to_dict(self) -> dict[str, JSON]:
        return {"name": self.name, "configuration": {"capacity": self.length}}

    def to_numpy(self, endianness: Endianness | None = "native") -> np.dtype[np.bytes_]:
        endianness_code = endianness_to_numpy_str(endianness)
        return np.dtype(endianness_code + self.numpy_character_code + str(self.length))


@dataclass(frozen=True, kw_only=True)
class StaticRawBytes(DTypeBase, Flexible):
    name = "r*"
    kind = "bytes"
    numpy_character_code = "V"
    item_size = 1
    default = b""

    @classmethod
    def from_numpy(cls: type[Self], dtype: npt.DTypeLike) -> Self:
        dtype = np.dtype(dtype)
        if dtype.kind != "V":
            raise ValueError(f"Invalid dtype {dtype}. Expected a bytes dtype.")
        return cls(length=dtype.itemsize)

    def to_dict(self) -> dict[str, JSON]:
        return {"name": f"r{self.length * 8}"}

    def to_numpy(self, endianness: Endianness | None = "native") -> np.dtype[np.void]:
        endianness_code = endianness_to_numpy_str(endianness)
        return np.dtype(endianness_code + self.numpy_character_code + str(self.length))


register_data_type(StaticByteString)

if _NUMPY_SUPPORTS_VLEN_STRING:

    @dataclass(frozen=True, kw_only=True)
    class VlenString(DTypeBase):
        name = "numpy/vlen_string"
        kind = "string"
        numpy_character_code = "T"
        # this uses UTF-8, so the encoding of a code point varies between
        # 1 and 4 bytes
        item_size = None
        default = ""

        def to_dict(self) -> dict[str, JSON]:
            return {"name": self.name}

        def to_numpy(
            self, endianness: Endianness | None = "native"
        ) -> np.dtype[np.dtypes.StringDType]:
            endianness_code = endianness_to_numpy_str(endianness)
            return np.dtype(endianness_code + self.numpy_character_code)

else:

    @dataclass(frozen=True, kw_only=True)
    class VlenString(DTypeBase):
        name = "numpy/vlen_string"
        kind = "string"
        numpy_character_code = "O"
        item_size = None
        default = ""

        def to_dict(self) -> dict[str, JSON]:
            return {"name": self.name}

        def to_numpy(
            self, endianness: Endianness | None = "native"
        ) -> np.dtype[np.dtypes.ObjectDType]:
            endianness_code = endianness_to_numpy_str(endianness)
            return np.dtype(endianness_code + self.numpy_character_code)


register_data_type(VlenString)


@dataclass(frozen=True, kw_only=True)
class StaticUnicodeString(DTypeBase, Flexible):
    name = "numpy/static_unicode_string"
    kind = "string"
    numpy_character_code = "U"
    item_size = 4
    default = ""

    @classmethod
    def from_numpy(cls: type[Self], dtype: npt.DTypeLike) -> Self:
        dtype = np.dtype(dtype)
        if dtype.kind != "U":
            raise ValueError(f"Invalid dtype {dtype}. Expected a string dtype.")
        return cls(length=dtype.itemsize)

    def to_dict(self) -> dict[str, JSON]:
        return {"name": self.name, "configuration": {"capacity": self.length}}

    def to_numpy(self, endianness: Endianness | None = "native") -> np.dtype[np.str_]:
        endianness_code = endianness_to_numpy_str(endianness)
        return np.dtype(endianness_code + self.numpy_character_code + str(self.length))


register_data_type(StaticUnicodeString)


def resolve_dtype(dtype: npt.DTypeLike | DTypeBase) -> DTypeBase:
    from zarr.registry import get_data_type_from_numpy

    if isinstance(dtype, DTypeBase):
        return dtype
    cls = get_data_type_from_numpy(dtype)
    return cls.from_numpy(dtype)


register_data_type(StaticRawBytes)

INTEGER_DTYPE = Int8 | Int16 | Int32 | Int64 | UInt8 | UInt16 | UInt32 | UInt64
FLOAT_DTYPE = Float16 | Float32 | Float64
COMPLEX_DTYPE = Complex64 | Complex128
STRING_DTYPE = StaticUnicodeString | VlenString | StaticByteString
