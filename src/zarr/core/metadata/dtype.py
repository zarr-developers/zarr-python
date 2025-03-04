from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from importlib.metadata import EntryPoint
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Literal,
    Self,
    TypeGuard,
    TypeVar,
    cast,
    get_args,
)

import numpy as np
import numpy.typing as npt
from typing_extensions import get_original_bases

from zarr.abc.metadata import Metadata
from zarr.core.strings import _NUMPY_SUPPORTS_VLEN_STRING

if TYPE_CHECKING:
    from zarr.core.common import JSON, ZarrFormat

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
    """
    Check if a JSON value represents a boolean.
    """
    return bool(isinstance(data, bool))


def check_json_str(data: JSON) -> TypeGuard[str]:
    """
    Check if a JSON value represents a string.
    """
    return bool(isinstance(data, str))


def check_json_int(data: JSON) -> TypeGuard[int]:
    """
    Check if a JSON value represents an integer.
    """
    return bool(isinstance(data, int))


def check_json_float_v2(data: JSON) -> TypeGuard[float]:
    if data == "NaN" or data == "Infinity" or data == "-Infinity":
        return True
    else:
        return bool(isinstance(data, float | int))


def check_json_float_v3(data: JSON) -> TypeGuard[float]:
    # TODO: handle the special JSON serialization of different NaN values
    return check_json_float_v2(data)


def check_json_float(data: JSON, zarr_format: ZarrFormat) -> TypeGuard[float]:
    if zarr_format == 2:
        return check_json_float_v2(data)
    else:
        return check_json_float_v3(data)


def check_json_complex_float_v3(data: JSON) -> TypeGuard[tuple[JSONFloat, JSONFloat]]:
    """
    Check if a JSON value represents a complex float, as per the zarr v3 spec
    """
    return (
        not isinstance(data, str)
        and isinstance(data, Sequence)
        and len(data) == 2
        and check_json_float_v3(data[0])
        and check_json_float_v3(data[1])
    )


def check_json_complex_float_v2(data: JSON) -> TypeGuard[tuple[JSONFloat, JSONFloat]]:
    """
    Check if a JSON value represents a complex float, as per the behavior of zarr-python 2.x
    """
    return (
        not isinstance(data, str)
        and isinstance(data, Sequence)
        and len(data) == 2
        and check_json_float_v2(data[0])
        and check_json_float_v2(data[1])
    )


def check_json_complex_float(
    data: JSON, zarr_format: ZarrFormat
) -> TypeGuard[tuple[JSONFloat, JSONFloat]]:
    if zarr_format == 2:
        return check_json_complex_float_v2(data)
    else:
        return check_json_complex_float_v3(data)


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


def complex_to_json_v2(data: complex | np.complexfloating[Any]) -> tuple[JSONFloat, JSONFloat]:
    return float_to_json_v2(data.real), float_to_json_v2(data.imag)


def complex_to_json_v3(data: complex | np.complexfloating[Any]) -> tuple[JSONFloat, JSONFloat]:
    return float_to_json_v3(data.real), float_to_json_v3(data.imag)


def complex_to_json(
    data: complex | np.complexfloating[Any], zarr_format: ZarrFormat
) -> tuple[JSONFloat, JSONFloat] | JSONFloat:
    if zarr_format == 2:
        return complex_to_json_v2(data)
    else:
        return complex_to_json_v3(data)
    raise ValueError(f"Invalid zarr format: {zarr_format}. Expected 2 or 3.")


def structured_scalar_to_json(data: bytes, zarr_format: ZarrFormat) -> str:
    if zarr_format == 2:
        return base64.b64encode(data).decode("ascii")
    raise NotImplementedError(f"Invalid zarr format: {zarr_format}. Expected 2.")


def structured_scalar_from_json(data: JSON, zarr_format: ZarrFormat) -> bytes:
    if zarr_format == 2:
        return base64.b64decode(data.encode("ascii"))
    raise NotImplementedError(f"Invalid zarr format: {zarr_format}. Expected 2.")


def float_from_json_v2(data: JSONFloat) -> float:
    match data:
        case "NaN":
            return float("nan")
        case "Infinity":
            return float("inf")
        case "-Infinity":
            return float("-inf")
        case _:
            return float(data)


def float_from_json_v3(data: JSONFloat) -> float:
    # todo: support the v3-specific NaN handling
    return float_from_json_v2(data)


def float_from_json(data: JSONFloat, zarr_format: ZarrFormat) -> float:
    if zarr_format == 2:
        return float_from_json_v2(data)
    else:
        return float_from_json_v3(data)


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
        if check_json_complex_float_v3(data):
            return complex_from_json_v3(data, dtype)
        else:
            raise TypeError(f"Invalid type: {data}. Expected a sequence of two numbers.")
    raise ValueError(f"Invalid zarr format: {zarr_format}. Expected 2 or 3.")


TDType = TypeVar("TDType", bound=np.dtype[Any])
TScalar = TypeVar("TScalar", bound=np.generic | str)


@dataclass(frozen=True, kw_only=True)
class DTypeWrapper(Generic[TDType, TScalar], ABC, Metadata):
    name: ClassVar[str]
    dtype_cls: ClassVar[type[TDType]]  # this class will create a numpy dtype
    kind: ClassVar[DataTypeFlavor]
    default_value: ClassVar[TScalar]
    endianness: Endianness | None = "native"

    def __init_subclass__(cls) -> None:
        # Subclasses will bind the first generic type parameter to an attribute of the class
        # TODO: wrap this in some *very informative* error handling
        generic_args = get_args(get_original_bases(cls)[0])
        cls.dtype_cls = generic_args[0]
        return super().__init_subclass__()

    def to_dict(self) -> dict[str, JSON]:
        return {"name": self.name}

    def cast_value(self: Self, value: object) -> TScalar:
        return cast(np.generic, self.unwrap().type(value))

    @classmethod
    def check_dtype(cls: type[Self], dtype: TDType) -> TypeGuard[TDType]:
        """
        Check that a dtype matches the dtype_cls class attribute
        """
        return type(dtype) is cls.dtype_cls

    @classmethod
    def wrap(cls: type[Self], dtype: TDType) -> Self:
        if cls.check_dtype(dtype):
            return cls._wrap_unsafe(dtype)
        raise TypeError(f"Invalid dtype: {dtype}. Expected an instance of {cls.dtype_cls}.")

    @classmethod
    @abstractmethod
    def _wrap_unsafe(cls: type[Self], dtype: TDType) -> Self:
        raise NotImplementedError

    def unwrap(self: Self) -> TDType:
        endian_str = endianness_to_numpy_str(self.endianness)
        return self.dtype_cls().newbyteorder(endian_str)

    def with_endianness(self: Self, endianness: Endianness) -> Self:
        return replace(self, endianness=endianness)

    @abstractmethod
    def to_json_value(self, data: np.generic, *, zarr_format: ZarrFormat) -> JSON:
        """
        Convert a single value to JSON-serializable format. Depends on the zarr format.
        """
        raise NotImplementedError

    @abstractmethod
    def from_json_value(self: Self, data: JSON, *, zarr_format: ZarrFormat) -> TScalar:
        """
        Read a JSON-serializable value as a numpy scalar
        """
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class Bool(DTypeWrapper[np.dtypes.BoolDType, np.bool_]):
    name = "bool"
    kind = "boolean"
    default_value = np.False_

    @classmethod
    def _wrap_unsafe(cls, dtype: np.dtypes.BoolDType) -> Self:
        return cls()

    def to_json_value(self, data: np.generic, zarr_format: ZarrFormat) -> bool:
        return bool(data)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.bool_:
        if check_json_bool(data):
            return self.unwrap().type(data)
        raise TypeError(f"Invalid type: {data}. Expected a boolean.")


class IntWrapperBase(DTypeWrapper[TDType, TScalar]):
    kind = "numeric"

    @classmethod
    def _wrap_unsafe(cls, dtype: TDType) -> Self:
        return cls()

    def to_json_value(self, data: np.generic, zarr_format: ZarrFormat) -> int:
        return int(data)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> TScalar:
        if check_json_int(data):
            return self.unwrap().type(data)
        raise TypeError(f"Invalid type: {data}. Expected an integer.")


@dataclass(frozen=True, kw_only=True)
class Int8(IntWrapperBase[np.dtypes.Int8DType, np.int8]):
    name = "int8"
    default_value = np.int8(0)


@dataclass(frozen=True, kw_only=True)
class UInt8(IntWrapperBase[np.dtypes.UInt8DType, np.uint8]):
    name = "uint8"
    default_value = np.uint8(0)


@dataclass(frozen=True, kw_only=True)
class Int16(IntWrapperBase[np.dtypes.Int16DType, np.int16]):
    name = "int16"
    default_value = np.int16(0)


@dataclass(frozen=True, kw_only=True)
class UInt16(IntWrapperBase[np.dtypes.UInt16DType, np.uint16]):
    name = "uint16"
    default_value = np.uint16(0)


@dataclass(frozen=True, kw_only=True)
class Int32(IntWrapperBase[np.dtypes.Int32DType, np.int32]):
    name = "int32"
    default_value = np.int32(0)


@dataclass(frozen=True, kw_only=True)
class UInt32(IntWrapperBase[np.dtypes.UInt32DType, np.uint32]):
    name = "uint32"
    default_value = np.uint32(0)


@dataclass(frozen=True, kw_only=True)
class Int64(IntWrapperBase[np.dtypes.Int64DType, np.int64]):
    name = "int64"
    default_value = np.int64(0)


@dataclass(frozen=True, kw_only=True)
class UInt64(IntWrapperBase[np.dtypes.UInt64DType, np.uint64]):
    name = "uint64"
    default_value = np.uint64(0)


class FloatWrapperBase(DTypeWrapper[TDType, TScalar]):
    kind = "numeric"

    @classmethod
    def _wrap_unsafe(cls, dtype: TDType) -> Self:
        return cls()

    def to_json_value(self, data: np.generic, zarr_format: ZarrFormat) -> JSONFloat:
        return float_to_json(data, zarr_format)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> TScalar:
        if check_json_float_v2(data):
            return self.unwrap().type(float_from_json(data, zarr_format))
        raise TypeError(f"Invalid type: {data}. Expected a float.")


@dataclass(frozen=True, kw_only=True)
class Float16(FloatWrapperBase[np.dtypes.Float16DType, np.float16]):
    name = "float16"
    default_value = np.float16(0)


@dataclass(frozen=True, kw_only=True)
class Float32(FloatWrapperBase[np.dtypes.Float32DType, np.float32]):
    name = "float32"
    default_value = np.float32(0)


@dataclass(frozen=True, kw_only=True)
class Float64(FloatWrapperBase[np.dtypes.Float64DType, np.float64]):
    name = "float64"
    default_value = np.float64(0)


@dataclass(frozen=True, kw_only=True)
class Complex64(DTypeWrapper[np.dtypes.Complex64DType, np.complex64]):
    name = "complex64"
    kind = "numeric"
    default_value = np.complex64(0)

    @classmethod
    def _wrap_unsafe(cls, dtype: np.dtypes.Complex64DType) -> Self:
        return cls()

    def to_json_value(
        self, data: np.generic, zarr_format: ZarrFormat
    ) -> tuple[JSONFloat, JSONFloat]:
        return complex_to_json(data, zarr_format)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.complex64:
        if check_json_complex_float_v3(data):
            return complex_from_json(data, dtype=self.unwrap(), zarr_format=zarr_format)
        raise TypeError(f"Invalid type: {data}. Expected a complex float.")


@dataclass(frozen=True, kw_only=True)
class Complex128(DTypeWrapper[np.dtypes.Complex128DType, np.complex128]):
    name = "complex128"
    kind = "numeric"
    default_value = np.complex128(0)

    @classmethod
    def _wrap_unsafe(cls, dtype: np.dtypes.Complex128DType) -> Self:
        return cls()

    def to_json_value(
        self, data: np.generic, zarr_format: ZarrFormat
    ) -> tuple[JSONFloat, JSONFloat]:
        return complex_to_json(data, zarr_format)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.complex128:
        if check_json_complex_float_v3(data):
            return complex_from_json(data, dtype=self.unwrap(), zarr_format=zarr_format)
        raise TypeError(f"Invalid type: {data}. Expected a complex float.")


@dataclass(frozen=True, kw_only=True)
class FlexibleWrapperBase(DTypeWrapper[TDType, TScalar]):
    item_size_bits: ClassVar[int]
    length: int

    @classmethod
    def _wrap_unsafe(cls, dtype: TDType) -> Self:
        return cls(length=dtype.itemsize // (cls.item_size_bits // 8))

    def unwrap(self) -> TDType:
        endianness_code = endianness_to_numpy_str(self.endianness)
        return self.dtype_cls(self.length).newbyteorder(endianness_code)


@dataclass(frozen=True, kw_only=True)
class StaticByteString(FlexibleWrapperBase[np.dtypes.BytesDType, np.bytes_]):
    name = "numpy/static_byte_string"
    kind = "string"
    default_value = np.bytes_(0)
    item_size_bits = 8

    def to_dict(self) -> dict[str, JSON]:
        return {"name": self.name, "configuration": {"length": self.length}}

    def to_json_value(self, data: np.generic, *, zarr_format: ZarrFormat) -> str:
        return base64.standard_b64encode(data).decode("ascii")

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.bytes_:
        if check_json_str(data):
            return self.unwrap().type(data.encode("ascii"))
        raise TypeError(f"Invalid type: {data}. Expected a string.")


@dataclass(frozen=True, kw_only=True)
class StaticRawBytes(FlexibleWrapperBase[np.dtypes.VoidDType, np.void]):
    name = "r*"
    kind = "bytes"
    default_value = np.void(b"")
    item_size_bits = 8

    def to_dict(self) -> dict[str, JSON]:
        return {"name": f"r{self.length * self.item_size_bits}"}

    def unwrap(self) -> np.dtypes.VoidDType:
        # this needs to be overridden because numpy does not allow creating a void type
        # by invoking np.dtypes.VoidDType directly
        endianness_code = endianness_to_numpy_str(self.endianness)
        return np.dtype(f"{endianness_code}V{self.length}")

    def to_json_value(self, data: np.generic, *, zarr_format: ZarrFormat) -> tuple[int, ...]:
        return base64.standard_b64encode(data).decode("ascii")

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.void:
        # todo: check that this is well-formed
        return self.unwrap().type(bytes(data))


@dataclass(frozen=True, kw_only=True)
class StaticUnicodeString(FlexibleWrapperBase[np.dtypes.StrDType, np.str_]):
    name = "numpy/static_unicode_string"
    kind = "string"
    default_value = np.str_("")
    item_size_bits = 32  # UCS4 is 32 bits per code point

    def to_dict(self) -> dict[str, JSON]:
        return {"name": self.name, "configuration": {"length": self.length}}

    def to_json_value(self, data: np.generic, *, zarr_format: ZarrFormat) -> str:
        return str(data)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.str_:
        if not check_json_str(data):
            raise TypeError(f"Invalid type: {data}. Expected a string.")
        return self.unwrap().type(data)


if _NUMPY_SUPPORTS_VLEN_STRING:

    @dataclass(frozen=True, kw_only=True)
    class VariableLengthString(DTypeWrapper[np.dtypes.StringDType, str]):
        name = "numpy/vlen_string"
        kind = "string"
        default_value = ""

        @classmethod
        def _wrap_unsafe(cls, dtype: np.dtypes.StringDType) -> Self:
            return cls()

        def to_dict(self) -> dict[str, JSON]:
            return {"name": self.name}

        def unwrap(self) -> np.dtypes.StringDType:
            # StringDType does not have endianness, so we ignore it here
            return self.dtype_cls()

        def to_json_value(self, data: np.generic, *, zarr_format: ZarrFormat) -> str:
            return str(data)

        def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> str:
            if not check_json_str(data):
                raise TypeError(f"Invalid type: {data}. Expected a string.")
            return self.unwrap().type(data)

else:

    @dataclass(frozen=True, kw_only=True)
    class VariableLengthString(DTypeWrapper[np.dtypes.ObjectDType, str]):
        name = "numpy/vlen_string"
        kind = "string"
        default_value = np.object_("")
        endianness: Endianness = field(default=None)

        def __post_init__(self) -> None:
            if self.endianness is not None:
                raise ValueError("VariableLengthString does not support endianness.")

        def to_dict(self) -> dict[str, JSON]:
            return {"name": self.name}

        @classmethod
        def _wrap_unsafe(cls, dtype: np.dtypes.ObjectDType) -> Self:
            return cls()

        def unwrap(self) -> np.dtypes.ObjectDType:
            return super().unwrap()

        def to_json_value(self, data: np.generic, *, zarr_format: ZarrFormat) -> str:
            return str(data)

        def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> str:
            if not check_json_str(data):
                raise TypeError(f"Invalid type: {data}. Expected a string.")
            return self.unwrap().type(data)


@dataclass(frozen=True, kw_only=True)
class StructuredDtype(DTypeWrapper[np.dtypes.VoidDType, np.void]):
    name = "numpy/struct"
    kind = "struct"
    fields: tuple[tuple[str, DTypeWrapper[Any, Any], int], ...]

    @classmethod
    def check_dtype(cls, dtype: np.dtypes.DTypeLike) -> TypeGuard[np.dtypes.VoidDType]:
        """
        Check that this dtype is a numpy structured dtype
        """
        return super().check_dtype(dtype) and dtype.fields is not None

    @classmethod
    def _wrap_unsafe(cls, dtype: np.dtypes.VoidDType) -> Self:
        fields: list[tuple[str, DTypeWrapper[Any, Any], int]] = []

        if dtype.fields is None:
            raise ValueError("numpy dtype has no fields")

        for key, (dtype_instance, offset) in dtype.fields.items():
            dtype_wrapped = data_type_registry.match_dtype(dtype_instance)
            fields.append((key, dtype_wrapped, offset))

        return cls(fields=tuple(fields))

    def to_json_value(self, data: np.generic, *, zarr_format: ZarrFormat) -> str:
        return structured_scalar_to_json(data.tobytes(), zarr_format)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.void:
        if not check_json_str(data):
            raise TypeError(f"Invalid type: {data}. Expected a string.")
        as_bytes = structured_scalar_from_json(data, zarr_format=zarr_format)
        dtype = self.unwrap()
        return np.array([as_bytes], dtype=dtype.str).view(dtype)[0]


def get_data_type_from_numpy(dtype: npt.DTypeLike) -> DTypeWrapper:
    if dtype in (str, "str"):
        if _NUMPY_SUPPORTS_VLEN_STRING:
            np_dtype = np.dtype("T")
        else:
            np_dtype = np.dtype("O")
    else:
        np_dtype = np.dtype(dtype)
    data_type_registry.lazy_load()
    for val in data_type_registry.contents.values():
        return val.wrap(np_dtype)
    raise ValueError(
        f"numpy dtype '{dtype}' does not have a corresponding Zarr dtype in: {list(data_type_registry.contents)}."
    )


def get_data_type_from_dict(dtype: dict[str, JSON]) -> DTypeWrapper:
    data_type_registry.lazy_load()
    dtype_name = dtype["name"]
    dtype_cls = data_type_registry.get(dtype_name)
    if dtype_cls is None:
        raise ValueError(f"No data type class matching name {dtype_name}")
    return dtype_cls.from_dict(dtype.get("configuration", {}))


def resolve_dtype(
    dtype: npt.DTypeLike | DTypeWrapper[Any, Any] | dict[str, JSON],
) -> DTypeWrapper[Any, Any]:
    if isinstance(dtype, DTypeWrapper):
        return dtype
    elif isinstance(dtype, dict):
        return get_data_type_from_dict(dtype)
    else:
        return get_data_type_from_numpy(dtype)


def get_data_type_by_name(
    dtype: str, configuration: dict[str, JSON] | None = None
) -> DTypeWrapper[Any, Any]:
    data_type_registry.lazy_load()
    if configuration is None:
        _configuration = {}
    else:
        _configuration = configuration
    maybe_dtype_cls = data_type_registry.get(dtype)
    if maybe_dtype_cls is None:
        raise ValueError(f"No data type class matching name {dtype}")
    return maybe_dtype_cls.from_dict(_configuration)


@dataclass(frozen=True, kw_only=True)
class DataTypeRegistry:
    contents: dict[str, type[DTypeWrapper[Any, Any]]] = field(default_factory=dict, init=False)
    lazy_load_list: list[EntryPoint] = field(default_factory=list, init=False)

    def lazy_load(self) -> None:
        for e in self.lazy_load_list:
            self.register(e.load())

        self.lazy_load_list.clear()

    def register(self: Self, cls: type[DTypeWrapper[Any, Any]]) -> None:
        # don't register the same dtype twice
        if cls.name not in self.contents or self.contents[cls.name] != cls:
            self.contents[cls.name] = cls

    def get(self, key: str) -> type[DTypeWrapper[Any, Any]]:
        return self.contents[key]

    def match_dtype(self, dtype: npt.DTypeLike) -> DTypeWrapper[Any, Any]:
        data_type_registry.lazy_load()
        for val in data_type_registry.contents.values():
            try:
                return val._wrap_unsafe(dtype)
            except ValueError:
                pass
        raise ValueError(f"No data type wrapper found that matches {dtype}")


def register_data_type(cls: type[DTypeWrapper[Any, Any]]) -> None:
    data_type_registry.register(cls)


data_type_registry = DataTypeRegistry()

INTEGER_DTYPE = Int8 | Int16 | Int32 | Int64 | UInt8 | UInt16 | UInt32 | UInt64
FLOAT_DTYPE = Float16 | Float32 | Float64
COMPLEX_DTYPE = Complex64 | Complex128
STRING_DTYPE = StaticUnicodeString | VariableLengthString | StaticByteString
for dtype in get_args(
    Bool | INTEGER_DTYPE | FLOAT_DTYPE | COMPLEX_DTYPE | STRING_DTYPE | StaticRawBytes
):
    register_data_type(dtype)
