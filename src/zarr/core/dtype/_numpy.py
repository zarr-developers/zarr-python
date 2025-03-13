from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, TypeGuard, cast, get_args

import numpy as np

from zarr.core.dtype.common import (
    _NUMPY_SUPPORTS_VLEN_STRING,
    DataTypeValidationError,
    Endianness,
    JSONFloat,
    bytes_from_json,
    bytes_to_json,
    check_json_bool,
    check_json_complex_float,
    check_json_complex_float_v3,
    check_json_float_v2,
    check_json_int,
    check_json_str,
    complex_from_json,
    complex_to_json,
    datetime_from_json,
    datetime_to_json,
    endianness_from_numpy_str,
    endianness_to_numpy_str,
    float_from_json,
    float_to_json,
)
from zarr.core.dtype.wrapper import DTypeWrapper, TDType

if TYPE_CHECKING:
    from zarr.core.common import JSON, ZarrFormat


@dataclass(frozen=True, kw_only=True)
class Bool(DTypeWrapper[np.dtypes.BoolDType, np.bool_]):
    """
    Wrapper for numpy boolean dtype.

    Attributes
    ----------
    name : str
        The name of the dtype.
    dtype_cls : ClassVar[type[np.dtypes.BoolDType]]
        The numpy dtype class.
    """

    _zarr_v3_name = "bool"
    dtype_cls: ClassVar[type[np.dtypes.BoolDType]] = np.dtypes.BoolDType

    def to_dict(self) -> dict[str, JSON]:
        return {"name": self.get_name(zarr_format=3)}

    def default_value(self) -> np.bool_:
        """
        Get the default value for the boolean dtype.

        Returns
        -------
        np.bool_
            The default value.
        """
        return np.False_

    @classmethod
    def _from_dtype_unsafe(cls, dtype: np.dtypes.BoolDType) -> Self:
        """
        Wrap a numpy boolean dtype without checking.

        Parameters
        ----------
        dtype : np.dtypes.BoolDType
            The numpy dtype to wrap.

        Returns
        -------
        Self
            The wrapped dtype.
        """
        return cls()

    def to_dtype(self) -> np.dtypes.BoolDType:
        return self.dtype_cls()

    def to_json_value(self, data: np.bool_, zarr_format: ZarrFormat) -> bool:
        """
        Convert a boolean value to JSON-serializable format.

        Parameters
        ----------
        data : object
            The value to convert.
        zarr_format : ZarrFormat
            The zarr format version.

        Returns
        -------
        bool
            The JSON-serializable format.
        """
        return bool(data)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.bool_:
        """
        Read a JSON-serializable value as a numpy boolean scalar.

        Parameters
        ----------
        data : JSON
            The JSON-serializable value.
        zarr_format : ZarrFormat
            The zarr format version.

        Returns
        -------
        np.bool_
            The numpy boolean scalar.
        """
        if check_json_bool(data):
            return self.cast_value(data)
        raise TypeError(f"Invalid type: {data}. Expected a boolean.")


@dataclass(frozen=True, kw_only=True)
class Int8(DTypeWrapper[np.dtypes.Int8DType, np.int8]):
    dtype_cls = np.dtypes.Int8DType
    _zarr_v3_name = "int8"

    @classmethod
    def _from_dtype_unsafe(cls, dtype: np.dtypes.Int8DType) -> Self:
        return cls()

    def to_dtype(self) -> np.dtypes.Int8DType:
        return self.dtype_cls()

    def to_dict(self) -> dict[str, JSON]:
        return {"name": self.get_name(zarr_format=3)}

    def default_value(self) -> np.int8:
        return self.to_dtype().type(0)

    def to_json_value(self, data: np.int8, zarr_format: ZarrFormat) -> int:
        return int(data)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.int8:
        if check_json_int(data):
            return self.cast_value(data)
        raise TypeError(f"Invalid type: {data}. Expected an integer.")


@dataclass(frozen=True, kw_only=True)
class UInt8(DTypeWrapper[np.dtypes.UInt8DType, np.uint8]):
    dtype_cls = np.dtypes.UInt8DType
    _zarr_v3_name = "uint8"

    @classmethod
    def _from_dtype_unsafe(cls, dtype: np.dtypes.UInt8DType) -> Self:
        return cls()

    def to_dtype(self) -> np.dtypes.UInt8DType:
        return self.dtype_cls()

    def to_dict(self) -> dict[str, JSON]:
        return {"name": self.get_name(zarr_format=3)}

    def default_value(self) -> np.uint8:
        return self.to_dtype().type(0)

    def to_json_value(self, data: np.uint8, zarr_format: ZarrFormat) -> int:
        return int(data)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.uint8:
        if check_json_int(data):
            return self.cast_value(data)
        raise TypeError(f"Invalid type: {data}. Expected an integer.")


@dataclass(frozen=True, kw_only=True)
class Int16(DTypeWrapper[np.dtypes.Int16DType, np.int16]):
    dtype_cls = np.dtypes.Int16DType
    _zarr_v3_name = "int16"
    endianness: Endianness | None = "native"

    @classmethod
    def _from_dtype_unsafe(cls, dtype: np.dtypes.Int16DType) -> Self:
        return cls(endianness=endianness_from_numpy_str(dtype.byteorder))

    def to_dtype(self) -> np.dtypes.Int16DType:
        return self.dtype_cls().newbyteorder(endianness_to_numpy_str(self.endianness))

    def to_dict(self) -> dict[str, JSON]:
        return {"name": self.get_name(zarr_format=3)}

    def default_value(self) -> np.int16:
        return self.cast_value(0)

    def to_json_value(self, data: np.int16, zarr_format: ZarrFormat) -> int:
        return int(data)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.int16:
        if check_json_int(data):
            return self.cast_value(data)
        raise TypeError(f"Invalid type: {data}. Expected an integer.")


@dataclass(frozen=True, kw_only=True)
class UInt16(DTypeWrapper[np.dtypes.UInt16DType, np.uint16]):
    dtype_cls = np.dtypes.UInt16DType
    _zarr_v3_name = "uint16"
    endianness: Endianness | None = "native"

    @classmethod
    def _from_dtype_unsafe(cls, dtype: np.dtypes.UInt16DType) -> Self:
        return cls(endianness=endianness_from_numpy_str(dtype.byteorder))

    def to_dtype(self) -> np.dtypes.UInt16DType:
        return self.dtype_cls().newbyteorder(endianness_to_numpy_str(self.endianness))

    def to_dict(self) -> dict[str, JSON]:
        return {"name": self.get_name(zarr_format=3)}

    def default_value(self) -> np.uint16:
        return self.cast_value(0)

    def to_json_value(self, data: np.uint16, zarr_format: ZarrFormat) -> int:
        return int(data)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.uint16:
        if check_json_int(data):
            return self.cast_value(data)
        raise TypeError(f"Invalid type: {data}. Expected an integer.")


@dataclass(frozen=True, kw_only=True)
class Int32(DTypeWrapper[np.dtypes.Int32DType, np.int32]):
    dtype_cls = np.dtypes.Int32DType
    _zarr_v3_name = "int32"
    endianness: Endianness | None = "native"

    @classmethod
    def _from_dtype_unsafe(cls, dtype: np.dtypes.Int32DType) -> Self:
        return cls(endianness=endianness_from_numpy_str(dtype.byteorder))

    def to_dtype(self) -> np.dtypes.Int32DType:
        return self.dtype_cls().newbyteorder(endianness_to_numpy_str(self.endianness))

    def to_dict(self) -> dict[str, JSON]:
        return {"name": self.get_name(zarr_format=3)}

    def default_value(self) -> np.int32:
        return self.cast_value(0)

    def to_json_value(self, data: np.int32, zarr_format: ZarrFormat) -> int:
        return int(data)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.int32:
        if check_json_int(data):
            return self.cast_value(data)
        raise TypeError(f"Invalid type: {data}. Expected an integer.")


@dataclass(frozen=True, kw_only=True)
class UInt32(DTypeWrapper[np.dtypes.UInt32DType, np.uint32]):
    dtype_cls = np.dtypes.UInt32DType
    _zarr_v3_name = "uint32"
    endianness: Endianness | None = "native"

    @classmethod
    def _from_dtype_unsafe(cls, dtype: np.dtypes.UInt32DType) -> Self:
        return cls(endianness=endianness_from_numpy_str(dtype.byteorder))

    def to_dtype(self) -> np.dtypes.UInt32DType:
        return self.dtype_cls().newbyteorder(endianness_to_numpy_str(self.endianness))

    def to_dict(self) -> dict[str, JSON]:
        return {"name": self.get_name(zarr_format=3)}

    def default_value(self) -> np.uint32:
        return self.cast_value(0)

    def to_json_value(self, data: np.uint32, zarr_format: ZarrFormat) -> int:
        return int(data)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.uint32:
        if check_json_int(data):
            return self.cast_value(data)
        raise TypeError(f"Invalid type: {data}. Expected an integer.")


@dataclass(frozen=True, kw_only=True)
class Int64(DTypeWrapper[np.dtypes.Int64DType, np.int64]):
    dtype_cls = np.dtypes.Int64DType
    _zarr_v3_name = "int64"
    endianness: Endianness | None = "native"

    @classmethod
    def _from_dtype_unsafe(cls, dtype: np.dtypes.Int64DType) -> Self:
        return cls(endianness=endianness_from_numpy_str(dtype.byteorder))

    def to_dtype(self) -> np.dtypes.Int64DType:
        return self.dtype_cls().newbyteorder(endianness_to_numpy_str(self.endianness))

    def to_dict(self) -> dict[str, JSON]:
        return {"name": self.get_name(zarr_format=3)}

    def default_value(self) -> np.int64:
        return self.cast_value(0)

    def to_json_value(self, data: np.int64, zarr_format: ZarrFormat) -> int:
        return int(data)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.int64:
        if check_json_int(data):
            return self.cast_value(data)
        raise TypeError(f"Invalid type: {data}. Expected an integer.")


@dataclass(frozen=True, kw_only=True)
class UInt64(DTypeWrapper[np.dtypes.UInt64DType, np.uint64]):
    dtype_cls = np.dtypes.UInt64DType
    _zarr_v3_name = "uint64"
    endianness: Endianness | None = "native"

    @classmethod
    def _from_dtype_unsafe(cls, dtype: np.dtypes.UInt64DType) -> Self:
        return cls(endianness=endianness_from_numpy_str(dtype.byteorder))

    def to_dtype(self) -> np.dtypes.UInt64DType:
        return self.dtype_cls().newbyteorder(endianness_to_numpy_str(self.endianness))

    def to_dict(self) -> dict[str, JSON]:
        return {"name": self.get_name(zarr_format=3)}

    def default_value(self) -> np.uint64:
        return self.cast_value(0)

    def to_json_value(self, data: np.uint64, zarr_format: ZarrFormat) -> int:
        return int(data)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.uint64:
        if check_json_int(data):
            return self.cast_value(data)
        raise TypeError(f"Invalid type: {data}. Expected an integer.")


@dataclass(frozen=True, kw_only=True)
class Float16(DTypeWrapper[np.dtypes.Float16DType, np.float16]):
    dtype_cls = np.dtypes.Float16DType
    _zarr_v3_name = "float16"
    endianness: Endianness | None = "native"

    @classmethod
    def _from_dtype_unsafe(cls, dtype: np.dtypes.Float16DType) -> Self:
        return cls(endianness=endianness_from_numpy_str(dtype.byteorder))

    def to_dtype(self) -> np.dtypes.Float16DType:
        return self.dtype_cls().newbyteorder(endianness_to_numpy_str(self.endianness))

    def to_dict(self) -> dict[str, JSON]:
        return {"name": self.get_name(zarr_format=3)}

    def default_value(self) -> np.float16:
        return self.to_dtype().type(0.0)

    def to_json_value(self, data: np.float16, zarr_format: ZarrFormat) -> JSONFloat:
        return float_to_json(data, zarr_format)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.float16:
        if check_json_float_v2(data):
            return self.to_dtype().type(float_from_json(data, zarr_format))
        raise TypeError(f"Invalid type: {data}. Expected a float.")


@dataclass(frozen=True, kw_only=True)
class Float32(DTypeWrapper[np.dtypes.Float32DType, np.float32]):
    dtype_cls = np.dtypes.Float32DType
    _zarr_v3_name = "float32"
    endianness: Endianness | None = "native"

    @classmethod
    def _from_dtype_unsafe(cls, dtype: np.dtypes.Float32DType) -> Self:
        return cls(endianness=endianness_from_numpy_str(dtype.byteorder))

    def to_dtype(self) -> np.dtypes.Float32DType:
        return self.dtype_cls().newbyteorder(endianness_to_numpy_str(self.endianness))

    def to_dict(self) -> dict[str, JSON]:
        return {"name": self.get_name(zarr_format=3)}

    def cast_value(self, value: object) -> np.float32:
        return self.to_dtype().type(value)

    def default_value(self) -> np.float32:
        return self.to_dtype().type(0.0)

    def to_json_value(self, data: np.float32, zarr_format: ZarrFormat) -> JSONFloat:
        return float_to_json(data, zarr_format)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.float32:
        if check_json_float_v2(data):
            return self.to_dtype().type(float_from_json(data, zarr_format))
        raise TypeError(f"Invalid type: {data}. Expected a float.")


@dataclass(frozen=True, kw_only=True)
class Float64(DTypeWrapper[np.dtypes.Float64DType, np.float64]):
    dtype_cls = np.dtypes.Float64DType
    _zarr_v3_name = "float64"
    endianness: Endianness | None = "native"

    @classmethod
    def _from_dtype_unsafe(cls, dtype: np.dtypes.Float64DType) -> Self:
        return cls(endianness=endianness_from_numpy_str(dtype.byteorder))

    def to_dtype(self) -> np.dtypes.Float64DType:
        return self.dtype_cls().newbyteorder(endianness_to_numpy_str(self.endianness))

    def to_dict(self) -> dict[str, JSON]:
        return {"name": self.get_name(zarr_format=3)}

    def default_value(self) -> np.float64:
        return self.to_dtype().type(0.0)

    def to_json_value(self, data: np.float64, zarr_format: ZarrFormat) -> JSONFloat:
        return float_to_json(data, zarr_format)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.float64:
        if check_json_float_v2(data):
            return self.to_dtype().type(float_from_json(data, zarr_format))
        raise TypeError(f"Invalid type: {data}. Expected a float.")


@dataclass(frozen=True, kw_only=True)
class Complex64(DTypeWrapper[np.dtypes.Complex64DType, np.complex64]):
    dtype_cls = np.dtypes.Complex64DType
    _zarr_v3_name = "complex64"
    endianness: Endianness | None = "native"

    @classmethod
    def _from_dtype_unsafe(cls, dtype: np.dtypes.Complex64DType) -> Self:
        return cls()

    def to_dtype(self) -> np.dtypes.Complex64DType:
        return self.dtype_cls().newbyteorder(endianness_to_numpy_str(self.endianness))

    def to_dict(self) -> dict[str, JSON]:
        return {"name": self.get_name(zarr_format=3)}

    def default_value(self) -> np.complex64:
        return np.complex64(0.0)

    def to_json_value(
        self, data: np.complex64, zarr_format: ZarrFormat
    ) -> tuple[JSONFloat, JSONFloat]:
        return complex_to_json(data, zarr_format)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.complex64:
        if check_json_complex_float(data, zarr_format=zarr_format):
            return complex_from_json(data, dtype=self.to_dtype(), zarr_format=zarr_format)
        raise TypeError(f"Invalid type: {data}. Expected a complex float.")


@dataclass(frozen=True, kw_only=True)
class Complex128(DTypeWrapper[np.dtypes.Complex128DType, np.complex128]):
    dtype_cls = np.dtypes.Complex128DType
    _zarr_v3_name = "complex128"
    endianness: Endianness | None = "native"

    @classmethod
    def _from_dtype_unsafe(cls, dtype: np.dtypes.Complex128DType) -> Self:
        return cls(endianness=endianness_from_numpy_str(dtype.byteorder))

    def to_dtype(self) -> np.dtypes.Complex128DType:
        return self.dtype_cls().newbyteorder(endianness_to_numpy_str(self.endianness))

    def to_dict(self) -> dict[str, JSON]:
        return {"name": self.get_name(zarr_format=3)}

    def default_value(self) -> np.complex128:
        return np.complex128(0.0)

    def to_json_value(
        self, data: np.complex128, zarr_format: ZarrFormat
    ) -> tuple[JSONFloat, JSONFloat]:
        return complex_to_json(data, zarr_format)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.complex128:
        if check_json_complex_float_v3(data):
            return complex_from_json(data, dtype=self.to_dtype(), zarr_format=zarr_format)
        raise TypeError(f"Invalid type: {data}. Expected a complex float.")


@dataclass(frozen=True, kw_only=True)
class FixedLengthAsciiString(DTypeWrapper[np.dtypes.BytesDType[Any], np.bytes_]):
    dtype_cls = np.dtypes.BytesDType
    _zarr_v3_name = "numpy.static_byte_string"
    item_size_bits: ClassVar[int] = 8
    length: int = 1

    @classmethod
    def _from_dtype_unsafe(cls, dtype: np.dtypes.BytesDType) -> Self:
        return cls(length=dtype.itemsize // (cls.item_size_bits // 8))

    def to_dtype(self) -> np.dtypes.BytesDType:
        return self.dtype_cls(self.length)

    def default_value(self) -> np.bytes_:
        return np.bytes_(b"")

    def to_dict(self) -> dict[str, JSON]:
        return {"name": self.get_name(zarr_format=3), "configuration": {"length": self.length}}

    def to_json_value(self, data: np.bytes_, *, zarr_format: ZarrFormat) -> str:
        return base64.standard_b64encode(data).decode("ascii")

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.bytes_:
        if check_json_str(data):
            return self.to_dtype().type(base64.standard_b64decode(data.encode("ascii")))
        raise TypeError(f"Invalid type: {data}. Expected a string.")


@dataclass(frozen=True, kw_only=True)
class FixedLengthBytes(DTypeWrapper[np.dtypes.VoidDType[Any], np.void]):
    dtype_cls = np.dtypes.VoidDType
    _zarr_v3_name = "r*"
    item_size_bits: ClassVar[int] = 8
    length: int = 1

    @classmethod
    def _from_dtype_unsafe(cls, dtype: np.dtypes.VoidDType[Any]) -> Self:
        return cls(length=dtype.itemsize // (cls.item_size_bits // 8))

    def default_value(self) -> np.void:
        return self.cast_value(("\x00" * self.length).encode("ascii"))

    def to_dtype(self) -> np.dtypes.VoidDType[Any]:
        # Numpy does not allow creating a void type
        # by invoking np.dtypes.VoidDType directly
        return np.dtype(f"V{self.length}")

    def get_name(self, zarr_format: ZarrFormat) -> str:
        if zarr_format == 2:
            return super().get_name(zarr_format=zarr_format)
        # note that we don't return self._zarr_v3_name
        # because the name is parametrized by the length
        return f"r{self.length * self.item_size_bits}"

    def to_dict(self) -> dict[str, JSON]:
        return {"name": self.get_name(zarr_format=3)}

    @classmethod
    def check_dtype(cls: type[Self], dtype: TDType) -> TypeGuard[np.dtypes.VoidDType[Any]]:
        """
        Reject structured dtypes by ensuring that dtype.fields is None

        Parameters
        ----------
        dtype : TDType
            The dtype to check.

        Returns
        -------
        Bool
            True if the dtype matches, False otherwise.
        """
        return super().check_dtype(dtype) and dtype.fields is None

    @classmethod
    def check_dict(cls, data: dict[str, JSON]) -> TypeGuard[dict[str, JSON]]:
        # Overriding the base class implementation because the r* dtype
        # does not have a name that will can appear in array metadata
        # Instead, array metadata will contain names like "r8", "r16", etc
        return (
            isinstance(data, dict)
            and "name" in data
            and isinstance(data["name"], str)
            and re.match(r"^r\d+$", data["name"])
        )

    def to_json_value(self, data: np.void, *, zarr_format: ZarrFormat) -> str:
        return base64.standard_b64encode(data.tobytes()).decode("ascii")

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.void:
        if check_json_str(data):
            return self.to_dtype().type(base64.standard_b64decode(data))
        raise DataTypeValidationError(f"Invalid type: {data}. Expected a string.")


@dataclass(frozen=True, kw_only=True)
class FixedLengthUnicodeString(DTypeWrapper[np.dtypes.StrDType[int], np.str_]):
    dtype_cls = np.dtypes.StrDType
    _zarr_v3_name = "numpy.fixed_length_unicode_string"
    item_size_bits: ClassVar[int] = 32  # UCS4 is 32 bits per code point
    endianness: Endianness | None = "native"
    length: int = 1

    @classmethod
    def _from_dtype_unsafe(cls, dtype: np.dtypes.StrDType[int]) -> Self:
        return cls(
            length=dtype.itemsize // (cls.item_size_bits // 8),
            endianness=endianness_from_numpy_str(dtype.byteorder),
        )

    def to_dtype(self) -> np.dtypes.StrDType[int]:
        return self.dtype_cls(self.length).newbyteorder(endianness_to_numpy_str(self.endianness))

    def default_value(self) -> np.str_:
        return np.str_("")

    def to_dict(self) -> dict[str, JSON]:
        return {"name": self.get_name(zarr_format=3), "configuration": {"length": self.length}}

    def to_json_value(self, data: np.str_, *, zarr_format: ZarrFormat) -> str:
        return str(data)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.str_:
        if not check_json_str(data):
            raise TypeError(f"Invalid type: {data}. Expected a string.")
        return self.cast_value(data)


if _NUMPY_SUPPORTS_VLEN_STRING:

    @dataclass(frozen=True, kw_only=True)
    class VariableLengthString(DTypeWrapper[np.dtypes.StringDType, str]):
        dtype_cls = np.dtypes.StringDType
        _zarr_v3_name = "numpy.vlen_string"

        @classmethod
        def _from_dtype_unsafe(cls, dtype: np.dtypes.StringDType) -> Self:
            return cls()

        def default_value(self) -> str:
            return ""

        def cast_value(self, value: object) -> str:
            return str(value)

        def to_dict(self) -> dict[str, JSON]:
            return {"name": self.get_name(zarr_format=3)}

        def to_dtype(self) -> np.dtypes.StringDType:
            return self.dtype_cls()

        def to_json_value(self, data: object, *, zarr_format: ZarrFormat) -> str:
            return str(data)

        def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> str:
            if not check_json_str(data):
                raise TypeError(f"Invalid type: {data}. Expected a string.")
            return self.cast_value(data)

else:

    @dataclass(frozen=True, kw_only=True)
    class VariableLengthString(DTypeWrapper[np.dtypes.ObjectDType, str]):
        dtype_cls = np.dtypes.ObjectDType
        _zarr_v3_name = "numpy.vlen_string"

        @classmethod
        def _from_dtype_unsafe(cls, dtype: np.dtypes.ObjectDType) -> Self:
            return cls()

        def to_dtype(self) -> np.dtypes.ObjectDType:
            return self.dtype_cls()

        def cast_value(self, value: object) -> str:
            return str(value)

        def default_value(self) -> str:
            return ""

        def to_dict(self) -> dict[str, JSON]:
            return {"name": self.get_name(zarr_format=3)}

        def to_json_value(self, data: str, *, zarr_format: ZarrFormat) -> str:
            return data

        def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> str:
            """
            String literals pass through
            """
            if not check_json_str(data):
                raise TypeError(f"Invalid type: {data}. Expected a string.")
            return data


DateUnit = Literal["Y", "M", "W", "D"]
TimeUnit = Literal["h", "m", "s", "ms", "us", "μs", "ns", "ps", "fs", "as"]


@dataclass(frozen=True, kw_only=True)
class DateTime64(DTypeWrapper[np.dtypes.DateTime64DType, np.datetime64]):
    dtype_cls = np.dtypes.DateTime64DType
    _zarr_v3_name = "numpy.datetime64"
    unit: DateUnit | TimeUnit = "s"
    endianness: Endianness = "native"

    def default_value(self) -> np.datetime64:
        return np.datetime64("NaT")

    def to_dict(self) -> dict[str, JSON]:
        return {"name": self.get_name(zarr_format=3), "configuration": {"unit": self.unit}}

    @classmethod
    def _from_dtype_unsafe(cls, dtype: np.dtypes.DateTime64DType) -> Self:
        unit = dtype.name[dtype.name.rfind("[") + 1 : dtype.name.rfind("]")]
        if unit not in get_args(DateUnit | TimeUnit):
            raise DataTypeValidationError('Invalid unit for "numpy.datetime64"')
        return cls(unit=unit, endianness=endianness_from_numpy_str(dtype.byteorder))

    def cast_value(self, value: object) -> np.datetime64:
        return self.to_dtype().type(value, self.unit)

    def to_dtype(self) -> np.dtypes.DateTime64DType:
        # Numpy does not allow creating datetime64 via
        # np.dtypes.DateTime64Dtype()
        return np.dtype(f"datetime64[{self.unit}]").newbyteorder(
            endianness_to_numpy_str(self.endianness)
        )

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.datetime64:
        if check_json_int(data):
            return datetime_from_json(data, self.unit)
        raise TypeError(f"Invalid type: {data}. Expected an integer.")

    def to_json_value(self, data: np.datetime64, *, zarr_format: ZarrFormat) -> int:
        return datetime_to_json(data)


@dataclass(frozen=True, kw_only=True)
class Structured(DTypeWrapper[np.dtypes.VoidDType, np.void]):
    dtype_cls = np.dtypes.VoidDType
    _zarr_v3_name = "numpy.structured"
    fields: tuple[tuple[str, DTypeWrapper[Any, Any]], ...]

    def default_value(self) -> np.void:
        return self.cast_value(0)

    def cast_value(self, value: object) -> np.void:
        return np.array([value], dtype=self.to_dtype())[0]

    @classmethod
    def check_dtype(cls, dtype: np.dtypes.DTypeLike) -> TypeGuard[np.dtypes.VoidDType]:
        """
        Check that this dtype is a numpy structured dtype

        Parameters
        ----------
        dtype : np.dtypes.DTypeLike
            The dtype to check.

        Returns
        -------
        TypeGuard[np.dtypes.VoidDType]
            True if the dtype matches, False otherwise.
        """
        return super().check_dtype(dtype) and dtype.fields is not None

    @classmethod
    def _from_dtype_unsafe(cls, dtype: np.dtypes.VoidDType) -> Self:
        from zarr.core.dtype import get_data_type_from_numpy

        fields: list[tuple[str, DTypeWrapper[Any, Any]]] = []

        if dtype.fields is None:
            raise ValueError("numpy dtype has no fields")

        for key, (dtype_instance, _) in dtype.fields.items():
            dtype_wrapped = get_data_type_from_numpy(dtype_instance)
            fields.append((key, dtype_wrapped))

        return cls(fields=tuple(fields))

    def get_name(self, zarr_format: ZarrFormat) -> str | list[tuple[str, str]]:
        if zarr_format == 2:
            return [[k, d.get_name(zarr_format=2)] for k, d in self.fields]
        return self._zarr_v3_name

    def to_dict(self) -> dict[str, JSON]:
        base_dict = {"name": self.get_name(zarr_format=3)}
        field_configs = [(f_name, f_dtype.to_dict()) for f_name, f_dtype in self.fields]
        base_dict["configuration"] = {"fields": field_configs}
        return base_dict

    @classmethod
    def check_dict(cls, data: JSON) -> bool:
        return (
            isinstance(data, dict)
            and "name" in data
            and "configuration" in data
            and "fields" in data["configuration"]
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        if cls.check_dict(data):
            from zarr.core.dtype import get_data_type_from_dict

            fields = tuple(
                (f_name, get_data_type_from_dict(f_dtype))
                for f_name, f_dtype in data["configuration"]["fields"]
            )
            return cls(fields=fields)
        raise DataTypeValidationError(f"Invalid JSON representation of data type {cls}.")

    def to_dtype(self) -> np.dtypes.VoidDType:
        return cast(np.void, np.dtype([(key, dtype.to_dtype()) for (key, dtype) in self.fields]))

    def to_json_value(self, data: np.generic, *, zarr_format: ZarrFormat) -> str:
        return bytes_to_json(data.tobytes(), zarr_format)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.void:
        if not check_json_str(data):
            raise TypeError(f"Invalid type: {data}. Expected a string.")
        as_bytes = bytes_from_json(data, zarr_format=zarr_format)
        dtype = self.to_dtype()
        return cast(np.void, np.array([as_bytes], dtype=dtype.str).view(dtype)[0])
