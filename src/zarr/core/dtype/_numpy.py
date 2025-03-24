from __future__ import annotations

import base64
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    Self,
    SupportsComplex,
    SupportsFloat,
    SupportsIndex,
    SupportsInt,
    TypeGuard,
    TypeVar,
    cast,
    get_args,
)

import numpy as np

from zarr.core.dtype.common import (
    DataTypeValidationError,
    Endianness,
    bytes_from_json,
    bytes_to_json,
    check_json_bool,
    check_json_complex_float,
    check_json_float,
    check_json_int,
    check_json_str,
    complex_float_from_json,
    complex_float_to_json,
    datetime_from_json,
    datetime_to_json,
    float_from_json,
    float_to_json,
)
from zarr.core.dtype.wrapper import ZDType, _BaseDType, _BaseScalar

if TYPE_CHECKING:
    from zarr.core.common import JSON, ZarrFormat

EndiannessNumpy = Literal[">", "<", "|", "="]
IntLike = SupportsInt | SupportsIndex | bytes | str
FloatLike = SupportsIndex | SupportsFloat | bytes | str
ComplexLike = SupportsFloat | SupportsIndex | SupportsComplex | bytes | str | None


@dataclass(frozen=True)
class HasEndianness:
    """
    This is a mix-in class for data types with an endianness attribute
    """

    endianness: Endianness | None = "little"


@dataclass(frozen=True)
class HasLength:
    """
    This is a mix-in class for data types with a length attribute
    """

    length: int


@dataclass(frozen=True, kw_only=True, slots=True)
class Bool(ZDType[np.dtypes.BoolDType, np.bool_]):
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
    _zarr_v2_names: ClassVar[tuple[str, ...]] = ("|b1",)
    dtype_cls = np.dtypes.BoolDType

    @classmethod
    def _from_dtype_unsafe(cls, dtype: _BaseDType) -> Self:
        return cls()

    def to_dtype(self: Self) -> np.dtypes.BoolDType:
        return self.dtype_cls()

    @classmethod
    def check_json(cls, data: JSON, zarr_format: ZarrFormat) -> TypeGuard[Literal["bool", "|b1"]]:
        """
        Check that the input is a valid JSON representation of a bool.
        """
        if zarr_format == 2:
            return data in cls._zarr_v2_names
        elif zarr_format == 3:
            return data == cls._zarr_v3_name
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def to_json(self, zarr_format: ZarrFormat) -> str:
        if zarr_format == 2:
            return self.to_dtype().str
        elif zarr_format == 3:
            return self._zarr_v3_name
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    @classmethod
    def _from_json_unsafe(cls, data: JSON, zarr_format: ZarrFormat) -> Self:
        return cls()

    def default_value(self) -> np.bool_:
        """
        Get the default value for the boolean dtype.

        Returns
        -------
        np.bool_
            The default value.
        """
        return np.False_

    def to_json_value(self, data: object, zarr_format: ZarrFormat) -> bool:
        """
        Convert a scalar to a python bool.

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
            return self._cast_value_unsafe(data)
        raise TypeError(f"Invalid type: {data}. Expected a boolean.")

    def check_value(self, data: object) -> bool:
        # Anything can become a bool
        return True

    def cast_value(self, value: object) -> np.bool_:
        return self._cast_value_unsafe(value)

    def _cast_value_unsafe(self, value: object) -> np.bool_:
        return np.bool_(value)


_NumpyIntDType = (
    np.dtypes.Int8DType
    | np.dtypes.Int16DType
    | np.dtypes.Int32DType
    | np.dtypes.Int64DType
    | np.dtypes.UInt8DType
    | np.dtypes.UInt16DType
    | np.dtypes.UInt32DType
    | np.dtypes.UInt64DType
)
_NumpyIntScalar = (
    np.int8 | np.int16 | np.int32 | np.int64 | np.uint8 | np.uint16 | np.uint32 | np.uint64
)
TIntDType_co = TypeVar("TIntDType_co", bound=_NumpyIntDType, covariant=True)
TIntScalar_co = TypeVar("TIntScalar_co", bound=_NumpyIntScalar, covariant=True)


@dataclass(frozen=True)
class BaseInt(ZDType[TIntDType_co, TIntScalar_co]):
    # This attribute holds the possible zarr v2 JSON names for the data type
    _zarr_v2_names: ClassVar[tuple[str, ...]]

    def to_json(self, zarr_format: ZarrFormat) -> str:
        """
        Convert the wrapped data type to a JSON-serializable form.

        Parameters
        ----------
        zarr_format : ZarrFormat
            The zarr format version.

        Returns
        -------
        str
            The JSON-serializable representation of the wrapped data type
        """
        if zarr_format == 2:
            return self.to_dtype().str
        elif zarr_format == 3:
            return self._zarr_v3_name
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    @classmethod
    def check_json(cls, data: JSON, zarr_format: ZarrFormat) -> TypeGuard[JSON]:
        """
        Check that the input is a valid JSON representation of this data type.
        """
        if zarr_format == 2:
            return data in cls._zarr_v2_names
        elif zarr_format == 3:
            return data == cls._zarr_v3_name
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def check_value(self, value: object) -> TypeGuard[IntLike]:
        return isinstance(value, IntLike)

    def _cast_value_unsafe(self, value: object) -> TIntScalar_co:
        if self.check_value(value):
            return self.to_dtype().type(value)  # type: ignore[return-value]
        raise TypeError(f"Invalid type: {value}. Expected a value castable to an integer.")

    def default_value(self) -> TIntScalar_co:
        """
        Get the default value, which is 0 cast to this dtype

        Returns
        -------
        Int scalar
            The default value.
        """
        return self._cast_value_unsafe(0)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> TIntScalar_co:
        """
        Read a JSON-serializable value as a numpy int scalar.

        Parameters
        ----------
        data : JSON
            The JSON-serializable value.
        zarr_format : ZarrFormat
            The zarr format version.

        Returns
        -------
        TScalar_co
            The numpy scalar.
        """
        if check_json_int(data):
            return self._cast_value_unsafe(data)
        raise TypeError(f"Invalid type: {data}. Expected an integer.")

    def to_json_value(self, data: object, zarr_format: ZarrFormat) -> int:
        """
        Convert an object to JSON-serializable scalar.

        Parameters
        ----------
        data : _BaseScalar
            The value to convert.
        zarr_format : ZarrFormat
            The zarr format version.

        Returns
        -------
        int
            The JSON-serializable form of the scalar.
        """
        return int(self.cast_value(data))


@dataclass(frozen=True, kw_only=True)
class Int8(BaseInt[np.dtypes.Int8DType, np.int8]):
    dtype_cls = np.dtypes.Int8DType
    _zarr_v3_name = "int8"
    _zarr_v2_names: ClassVar[tuple[str, ...]] = ("|i1",)

    @classmethod
    def _from_dtype_unsafe(cls, dtype: _BaseDType) -> Self:
        return cls()

    def to_dtype(self: Self) -> np.dtypes.Int8DType:
        return self.dtype_cls()

    @classmethod
    def _from_json_unsafe(cls, data: JSON, zarr_format: ZarrFormat) -> Self:
        return cls()


@dataclass(frozen=True, kw_only=True)
class UInt8(BaseInt[np.dtypes.UInt8DType, np.uint8]):
    dtype_cls = np.dtypes.UInt8DType
    _zarr_v3_name = "uint8"
    _zarr_v2_names: ClassVar[tuple[str, ...]] = ("|u1",)

    @classmethod
    def _from_dtype_unsafe(cls, dtype: _BaseDType) -> Self:
        return cls()

    def to_dtype(self: Self) -> np.dtypes.UInt8DType:
        return self.dtype_cls()

    @classmethod
    def _from_json_unsafe(cls, data: JSON, zarr_format: ZarrFormat) -> Self:
        return cls()


@dataclass(frozen=True, kw_only=True)
class Int16(BaseInt[np.dtypes.Int16DType, np.int16], HasEndianness):
    dtype_cls = np.dtypes.Int16DType
    _zarr_v3_name = "int16"
    _zarr_v2_names: ClassVar[tuple[str, ...]] = (">i2", "<i2")

    @classmethod
    def _from_dtype_unsafe(cls, dtype: _BaseDType) -> Self:
        byte_order = cast("EndiannessNumpy", dtype.byteorder)
        return cls(endianness=endianness_from_numpy_str(byte_order))

    def to_dtype(self) -> np.dtypes.Int16DType:
        byte_order = endianness_to_numpy_str(self.endianness)
        return self.dtype_cls().newbyteorder(byte_order)

    @classmethod
    def _from_json_unsafe(cls, data: JSON, zarr_format: ZarrFormat) -> Self:
        if zarr_format == 2:
            # This ensures that we get the endianness correct without annoying string parsing
            return cls.from_dtype(np.dtype(data))  # type: ignore[arg-type]
        elif zarr_format == 3:
            return cls()
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover


@dataclass(frozen=True, kw_only=True)
class UInt16(BaseInt[np.dtypes.UInt16DType, np.uint16], HasEndianness):
    dtype_cls = np.dtypes.UInt16DType
    _zarr_v3_name = "uint16"
    _zarr_v2_names: ClassVar[tuple[str, ...]] = (">u2", "<u2")

    @classmethod
    def _from_dtype_unsafe(cls, dtype: _BaseDType) -> Self:
        byte_order = cast("EndiannessNumpy", dtype.byteorder)
        return cls(endianness=endianness_from_numpy_str(byte_order))

    def to_dtype(self) -> np.dtypes.UInt16DType:
        byte_order = endianness_to_numpy_str(self.endianness)
        return self.dtype_cls().newbyteorder(byte_order)

    @classmethod
    def _from_json_unsafe(cls, data: JSON, zarr_format: ZarrFormat) -> Self:
        if zarr_format == 2:
            return cls.from_dtype(np.dtype(data))  # type: ignore[arg-type]
        elif zarr_format == 3:
            return cls()
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover


@dataclass(frozen=True, kw_only=True)
class Int32(BaseInt[np.dtypes.Int32DType, np.int32], HasEndianness):
    dtype_cls = np.dtypes.Int32DType
    _zarr_v3_name = "int32"
    _zarr_v2_names: ClassVar[tuple[str, ...]] = (">i4", "<i4")

    @classmethod
    def from_dtype(cls: type[Self], dtype: _BaseDType) -> Self:
        # We override the base implementation to address a windows-specific, pre-numpy 2 issue where
        # ``np.dtype('i')`` is an instance of ``np.dtypes.IntDType`` that acts like `int32` instead of ``np.dtype('int32')``
        # In this case, ``type(np.dtype('i')) == np.dtypes.Int32DType``  will evaluate to ``True``,
        # despite the two classes being different. Thus we will create an instance of `cls` with the
        # latter dtype, after pulling in the byte order of the input
        if dtype == np.dtypes.Int32DType():
            return cls._from_dtype_unsafe(np.dtypes.Int32DType().newbyteorder(dtype.byteorder))
        else:
            return super().from_dtype(dtype)

    @classmethod
    def _from_dtype_unsafe(cls, dtype: _BaseDType) -> Self:
        byte_order = cast("EndiannessNumpy", dtype.byteorder)
        return cls(endianness=endianness_from_numpy_str(byte_order))

    def to_dtype(self) -> np.dtypes.Int32DType:
        byte_order = endianness_to_numpy_str(self.endianness)
        return self.dtype_cls().newbyteorder(byte_order)

    @classmethod
    def _from_json_unsafe(cls, data: JSON, zarr_format: ZarrFormat) -> Self:
        if zarr_format == 2:
            return cls.from_dtype(np.dtype(data))  # type: ignore[arg-type]
        elif zarr_format == 3:
            return cls()
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover


@dataclass(frozen=True, kw_only=True)
class UInt32(BaseInt[np.dtypes.UInt32DType, np.uint32], HasEndianness):
    dtype_cls = np.dtypes.UInt32DType
    _zarr_v3_name = "uint32"
    _zarr_v2_names: ClassVar[tuple[str, ...]] = (">u4", "<u4")

    @classmethod
    def _from_dtype_unsafe(cls, dtype: _BaseDType) -> Self:
        byte_order = cast("EndiannessNumpy", dtype.byteorder)
        return cls(endianness=endianness_from_numpy_str(byte_order))

    def to_dtype(self) -> np.dtypes.UInt32DType:
        byte_order = endianness_to_numpy_str(self.endianness)
        return self.dtype_cls().newbyteorder(byte_order)

    @classmethod
    def _from_json_unsafe(cls, data: JSON, zarr_format: ZarrFormat) -> Self:
        if zarr_format == 2:
            return cls.from_dtype(np.dtype(data))  # type: ignore[arg-type]
        elif zarr_format == 3:
            return cls()
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover


@dataclass(frozen=True, kw_only=True)
class Int64(BaseInt[np.dtypes.Int64DType, np.int64], HasEndianness):
    dtype_cls = np.dtypes.Int64DType
    _zarr_v3_name = "int64"
    _zarr_v2_names: ClassVar[tuple[str, ...]] = (">i8", "<i8")

    @classmethod
    def _from_dtype_unsafe(cls, dtype: _BaseDType) -> Self:
        byte_order = cast("EndiannessNumpy", dtype.byteorder)
        return cls(endianness=endianness_from_numpy_str(byte_order))

    def to_dtype(self) -> np.dtypes.Int64DType:
        byte_order = endianness_to_numpy_str(self.endianness)
        return self.dtype_cls().newbyteorder(byte_order)

    @classmethod
    def _from_json_unsafe(cls, data: JSON, zarr_format: ZarrFormat) -> Self:
        if zarr_format == 2:
            return cls.from_dtype(np.dtype(data))  # type: ignore[arg-type]
        elif zarr_format == 3:
            return cls()
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover


@dataclass(frozen=True, kw_only=True)
class UInt64(BaseInt[np.dtypes.UInt64DType, np.uint64], HasEndianness):
    dtype_cls = np.dtypes.UInt64DType
    _zarr_v3_name = "uint64"
    _zarr_v2_names: ClassVar[tuple[str, ...]] = (">u8", "<u8")

    @classmethod
    def _from_dtype_unsafe(cls, dtype: _BaseDType) -> Self:
        byte_order = cast("EndiannessNumpy", dtype.byteorder)
        return cls(endianness=endianness_from_numpy_str(byte_order))

    def to_dtype(self) -> np.dtypes.UInt64DType:
        byte_order = endianness_to_numpy_str(self.endianness)
        return self.dtype_cls().newbyteorder(byte_order)

    @classmethod
    def _from_json_unsafe(cls, data: JSON, zarr_format: ZarrFormat) -> Self:
        if zarr_format == 2:
            return cls.from_dtype(np.dtype(data))  # type: ignore[arg-type]
        elif zarr_format == 3:
            return cls()
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover


TFloatDType_co = TypeVar(
    "TFloatDType_co",
    bound=np.dtypes.Float16DType | np.dtypes.Float32DType | np.dtypes.Float64DType,
    covariant=True,
)
TFloatScalar_co = TypeVar(
    "TFloatScalar_co", bound=np.float16 | np.float32 | np.float64, covariant=True
)


@dataclass(frozen=True)
class BaseFloat(ZDType[TFloatDType_co, TFloatScalar_co], HasEndianness):
    # This attribute holds the possible zarr v2 JSON names for the data type
    _zarr_v2_names: ClassVar[tuple[str, ...]]

    @classmethod
    def _from_dtype_unsafe(cls, dtype: _BaseDType) -> Self:
        byte_order = cast("EndiannessNumpy", dtype.byteorder)
        return cls(endianness=endianness_from_numpy_str(byte_order))

    def to_dtype(self) -> TFloatDType_co:
        byte_order = endianness_to_numpy_str(self.endianness)
        return self.dtype_cls().newbyteorder(byte_order)  # type: ignore[return-value]

    def to_json(self, zarr_format: ZarrFormat) -> str:
        """
        Convert the wrapped data type to a JSON-serializable form.

        Parameters
        ----------
        zarr_format : ZarrFormat
            The zarr format version.

        Returns
        -------
        str
            The JSON-serializable representation of the wrapped data type
        """
        if zarr_format == 2:
            return self.to_dtype().str
        elif zarr_format == 3:
            return self._zarr_v3_name
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    @classmethod
    def _from_json_unsafe(cls, data: JSON, zarr_format: ZarrFormat) -> Self:
        if zarr_format == 2:
            return cls.from_dtype(np.dtype(data))  # type: ignore[arg-type]
        elif zarr_format == 3:
            return cls()
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    @classmethod
    def check_json(cls, data: JSON, zarr_format: ZarrFormat) -> TypeGuard[JSON]:
        """
        Check that the input is a valid JSON representation of this data type.
        """
        if zarr_format == 2:
            return data in cls._zarr_v2_names
        elif zarr_format == 3:
            return data == cls._zarr_v3_name
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def check_value(self, value: object) -> TypeGuard[FloatLike]:
        return isinstance(value, FloatLike)

    def _cast_value_unsafe(self, value: object) -> TFloatScalar_co:
        if self.check_value(value):
            return self.to_dtype().type(value)  # type: ignore[return-value]
        raise TypeError(f"Invalid type: {value}. Expected a value castable to a float.")

    def default_value(self) -> TFloatScalar_co:
        """
        Get the default value, which is 0 cast to this dtype

        Returns
        -------
        Int scalar
            The default value.
        """
        return self._cast_value_unsafe(0)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> TFloatScalar_co:
        """
        Read a JSON-serializable value as a numpy float.

        Parameters
        ----------
        data : JSON
            The JSON-serializable value.
        zarr_format : ZarrFormat
            The zarr format version.

        Returns
        -------
        TScalar_co
            The numpy float.
        """
        if check_json_float(data, zarr_format=zarr_format):
            return self._cast_value_unsafe(float_from_json(data, zarr_format=zarr_format))
        raise TypeError(
            f"Invalid type: {data}. Expected a float or a special string encoding of a float."
        )

    def to_json_value(self, data: object, zarr_format: ZarrFormat) -> float | str:
        """
        Convert an object to a JSON-serializable float.

        Parameters
        ----------
        data : _BaseScalar
            The value to convert.
        zarr_format : ZarrFormat
            The zarr format version.

        Returns
        -------
        JSON
            The JSON-serializable form of the float, which is potentially a number or a string.
            See the zarr specifications for details on the JSON encoding for floats.
        """
        return float_to_json(self._cast_value_unsafe(data), zarr_format=zarr_format)


@dataclass(frozen=True, kw_only=True)
class Float16(BaseFloat[np.dtypes.Float16DType, np.float16]):
    dtype_cls = np.dtypes.Float16DType
    _zarr_v3_name = "float16"
    _zarr_v2_names: ClassVar[tuple[str, ...]] = (">f2", "<f2")


@dataclass(frozen=True, kw_only=True)
class Float32(BaseFloat[np.dtypes.Float32DType, np.float32]):
    dtype_cls = np.dtypes.Float32DType
    _zarr_v3_name = "float32"
    _zarr_v2_names: ClassVar[tuple[str, ...]] = (">f4", "<f4")


@dataclass(frozen=True, kw_only=True)
class Float64(BaseFloat[np.dtypes.Float64DType, np.float64]):
    dtype_cls = np.dtypes.Float64DType
    _zarr_v3_name = "float64"
    _zarr_v2_names: ClassVar[tuple[str, ...]] = (">f8", "<f8")


TComplexDType_co = TypeVar(
    "TComplexDType_co", bound=np.dtypes.Complex64DType | np.dtypes.Complex128DType, covariant=True
)
TComplexScalar_co = TypeVar("TComplexScalar_co", bound=np.complex64 | np.complex128, covariant=True)


@dataclass(frozen=True)
class BaseComplex(ZDType[TComplexDType_co, TComplexScalar_co], HasEndianness):
    # This attribute holds the possible zarr v2 JSON names for the data type
    _zarr_v2_names: ClassVar[tuple[str, ...]]

    @classmethod
    def _from_dtype_unsafe(cls, dtype: _BaseDType) -> Self:
        byte_order = cast("EndiannessNumpy", dtype.byteorder)
        return cls(endianness=endianness_from_numpy_str(byte_order))

    def to_dtype(self) -> TComplexDType_co:
        byte_order = endianness_to_numpy_str(self.endianness)
        return self.dtype_cls().newbyteorder(byte_order)  # type: ignore[return-value]

    def to_json(self, zarr_format: ZarrFormat) -> str:
        """
        Convert the wrapped data type to a JSON-serializable form.

        Parameters
        ----------
        zarr_format : ZarrFormat
            The zarr format version.

        Returns
        -------
        str
            The JSON-serializable representation of the wrapped data type
        """
        if zarr_format == 2:
            return self.to_dtype().str
        elif zarr_format == 3:
            return self._zarr_v3_name
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    @classmethod
    def _from_json_unsafe(cls, data: JSON, zarr_format: ZarrFormat) -> Self:
        if zarr_format == 2:
            return cls.from_dtype(np.dtype(data))  # type: ignore[arg-type]
        elif zarr_format == 3:
            return cls()
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    @classmethod
    def check_json(cls, data: JSON, zarr_format: ZarrFormat) -> TypeGuard[JSON]:
        """
        Check that the input is a valid JSON representation of this data type.
        """
        if zarr_format == 2:
            return data in cls._zarr_v2_names
        elif zarr_format == 3:
            return data == cls._zarr_v3_name
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def check_value(self, value: object) -> bool:
        return isinstance(value, ComplexLike)

    def _cast_value_unsafe(self, value: object) -> TComplexScalar_co:
        if self.check_value(value):
            return self.to_dtype().type(value)  # type: ignore[arg-type, return-value]
        raise TypeError(f"Invalid type: {value}. Expected a value castable to a complex scalar.")

    def default_value(self) -> TComplexScalar_co:
        """
        Get the default value, which is 0 cast to this dtype

        Returns
        -------
        Int scalar
            The default value.
        """
        return self._cast_value_unsafe(0)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> TComplexScalar_co:
        """
        Read a JSON-serializable value as a numpy float.

        Parameters
        ----------
        data : JSON
            The JSON-serializable value.
        zarr_format : ZarrFormat
            The zarr format version.

        Returns
        -------
        TScalar_co
            The numpy float.
        """
        if check_json_complex_float(data, zarr_format=zarr_format):
            return self._cast_value_unsafe(complex_float_from_json(data, zarr_format=zarr_format))
        raise TypeError(
            f"Invalid type: {data}. Expected a float or a special string encoding of a float."
        )

    def to_json_value(self, data: object, zarr_format: ZarrFormat) -> JSON:
        """
        Convert an object to a JSON-serializable float.

        Parameters
        ----------
        data : _BaseScalar
            The value to convert.
        zarr_format : ZarrFormat
            The zarr format version.

        Returns
        -------
        JSON
            The JSON-serializable form of the complex number, which is a list of two floats,
            each of which is encoding according to a zarr-format-specific encoding.
        """
        return complex_float_to_json(self.cast_value(data), zarr_format=zarr_format)


@dataclass(frozen=True, kw_only=True)
class Complex64(BaseComplex[np.dtypes.Complex64DType, np.complex64]):
    dtype_cls = np.dtypes.Complex64DType
    _zarr_v3_name = "complex64"
    _zarr_v2_names: ClassVar[tuple[str, ...]] = (">c8", "<c8")


@dataclass(frozen=True, kw_only=True)
class Complex128(BaseComplex[np.dtypes.Complex128DType, np.complex128], HasEndianness):
    dtype_cls = np.dtypes.Complex128DType
    _zarr_v3_name = "complex128"
    _zarr_v2_names: ClassVar[tuple[str, ...]] = (">c16", "<c16")


@dataclass(frozen=True, kw_only=True)
class FixedLengthAscii(ZDType[np.dtypes.BytesDType[int], np.bytes_], HasLength):
    dtype_cls = np.dtypes.BytesDType
    _zarr_v3_name = "numpy.fixed_length_ascii"
    item_size_bits: ClassVar[int] = 8

    @classmethod
    def _from_dtype_unsafe(cls, dtype: _BaseDType) -> Self:
        return cls(length=dtype.itemsize // (cls.item_size_bits // 8))

    def to_dtype(self) -> np.dtypes.BytesDType[int]:
        return self.dtype_cls(self.length)

    @classmethod
    def check_json(cls, data: JSON, zarr_format: ZarrFormat) -> TypeGuard[JSON]:
        """
        Check that the input is a valid JSON representation of a numpy S dtype.
        """
        if zarr_format == 2:
            # match |S1, |S2, etc
            return isinstance(data, str) and re.match(r"^\|S\d+$", data) is not None
        elif zarr_format == 3:
            return (
                isinstance(data, dict)
                and "name" in data
                and data["name"] == cls._zarr_v3_name
                and "configuration" in data
                and isinstance(data["configuration"], dict)
                and "length_bits" in data["configuration"]
                and isinstance(data["configuration"]["length_bits"], int)
            )
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def to_json(self, zarr_format: ZarrFormat) -> JSON:
        if zarr_format == 2:
            return self.to_dtype().str
        elif zarr_format == 3:
            return {
                "name": self._zarr_v3_name,
                "configuration": {"length_bits": self.length * self.item_size_bits},
            }
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    @classmethod
    def _from_json_unsafe(cls, data: JSON, zarr_format: ZarrFormat) -> Self:
        if zarr_format == 2:
            return cls.from_dtype(np.dtype(data))  # type: ignore[arg-type]
        elif zarr_format == 3:
            return cls(length=data["configuration"]["length_bits"] // cls.item_size_bits)  # type: ignore[arg-type, index, call-overload, operator]
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def default_value(self) -> np.bytes_:
        return np.bytes_(b"")

    def to_json_value(self, data: object, *, zarr_format: ZarrFormat) -> str:
        return base64.standard_b64encode(data).decode("ascii")  # type: ignore[arg-type]

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.bytes_:
        if check_json_str(data):
            return self.to_dtype().type(base64.standard_b64decode(data.encode("ascii")))
        raise TypeError(f"Invalid type: {data}. Expected a string.")

    def check_value(self, data: object) -> bool:
        return isinstance(data, np.bytes_ | str | bytes)

    def _cast_value_unsafe(self, value: object) -> np.bytes_:
        return self.to_dtype().type(value)


@dataclass(frozen=True, kw_only=True)
class FixedLengthBytes(ZDType[np.dtypes.VoidDType[int], np.void], HasLength):
    # np.dtypes.VoidDType is specified in an odd way in numpy
    # it cannot be used to create instances of the dtype
    # so we have to tell mypy to ignore this here
    dtype_cls = np.dtypes.VoidDType  # type: ignore[assignment]
    _zarr_v3_name = "numpy.void"
    item_size_bits: ClassVar[int] = 8

    @classmethod
    def _from_dtype_unsafe(cls, dtype: _BaseDType) -> Self:
        return cls(length=dtype.itemsize // (cls.item_size_bits // 8))

    def to_dtype(self) -> np.dtypes.VoidDType[int]:
        # Numpy does not allow creating a void type
        # by invoking np.dtypes.VoidDType directly
        return cast("np.dtypes.VoidDType[int]", np.dtype(f"V{self.length}"))

    @classmethod
    def check_json(cls, data: JSON, zarr_format: ZarrFormat) -> TypeGuard[JSON]:
        if zarr_format == 2:
            # Check that the dtype is |V1, |V2, ...
            return isinstance(data, str) and re.match(r"^\|V\d+$", data) is not None
        elif zarr_format == 3:
            return (
                isinstance(data, dict)
                and "name" in data
                and isinstance(data["name"], str)
                and (re.match(r"^r\d+$", data["name"]) is not None)
            )
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def to_json(self, zarr_format: ZarrFormat) -> JSON:
        if zarr_format == 2:
            return self.to_dtype().str
        elif zarr_format == 3:
            return {"name": f"r{self.length * self.item_size_bits}"}
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    @classmethod
    def _from_json_unsafe(cls, data: JSON, zarr_format: ZarrFormat) -> Self:
        if zarr_format == 2:
            return cls.from_dtype(np.dtype(data))  # type: ignore[arg-type]
        elif zarr_format == 3:
            return cls(length=int(data["name"][1:]) // cls.item_size_bits)  # type: ignore[arg-type, index, call-overload]
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    @classmethod
    def check_dtype(cls: type[Self], dtype: _BaseDType) -> TypeGuard[np.dtypes.VoidDType[Any]]:
        """
        Numpy void dtype comes in two forms:
        * If the ``fields`` attribute is ``None``, then the dtype represents N raw bytes.
        * If the ``fields`` attribute is not ``None``, then the dtype represents a structured dtype,

        In this check we ensure that ``fields`` is ``None``.

        Parameters
        ----------
        dtype : TDType
            The dtype to check.

        Returns
        -------
        Bool
            True if the dtype matches, False otherwise.
        """
        return cls.dtype_cls is type(dtype) and dtype.fields is None  # type: ignore[has-type]

    def default_value(self) -> np.void:
        return self.to_dtype().type(("\x00" * self.length).encode("ascii"))

    def to_json_value(self, data: object, *, zarr_format: ZarrFormat) -> str:
        return base64.standard_b64encode(self.cast_value(data).tobytes()).decode("ascii")

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.void:
        if check_json_str(data):
            return self.to_dtype().type(base64.standard_b64decode(data))
        raise DataTypeValidationError(f"Invalid type: {data}. Expected a string.")

    def check_value(self, data: object) -> bool:
        return isinstance(data, np.bytes_ | str | bytes | np.void)

    def _cast_value_unsafe(self, value: object) -> np.void:
        return self.to_dtype().type(value)  # type: ignore[call-overload, no-any-return]


@dataclass(frozen=True, kw_only=True)
class FixedLengthUnicode(ZDType[np.dtypes.StrDType[int], np.str_], HasEndianness, HasLength):
    dtype_cls = np.dtypes.StrDType
    _zarr_v3_name = "numpy.fixed_length_ucs4"
    item_size_bits: ClassVar[int] = 32  # UCS4 is 32 bits per code point

    @classmethod
    def _from_dtype_unsafe(cls, dtype: _BaseDType) -> Self:
        byte_order = cast("EndiannessNumpy", dtype.byteorder)
        return cls(
            length=dtype.itemsize // (cls.item_size_bits // 8),
            endianness=endianness_from_numpy_str(byte_order),
        )

    def to_dtype(self) -> np.dtypes.StrDType[int]:
        byte_order = endianness_to_numpy_str(self.endianness)
        return self.dtype_cls(self.length).newbyteorder(byte_order)

    @classmethod
    def check_json(cls, data: JSON, zarr_format: ZarrFormat) -> TypeGuard[JSON]:
        """
        Check that the input is a valid JSON representation of a numpy S dtype.
        """
        if zarr_format == 2:
            # match >U1, <U2, etc
            return isinstance(data, str) and re.match(r"^[><]U\d+$", data) is not None
        elif zarr_format == 3:
            return (
                isinstance(data, dict)
                and "name" in data
                and data["name"] == cls._zarr_v3_name
                and "configuration" in data
                and isinstance(data["configuration"], dict)
                and "length_bits" in data["configuration"]
                and isinstance(data["configuration"]["length_bits"], int)
            )
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def to_json(self, zarr_format: ZarrFormat) -> JSON:
        if zarr_format == 2:
            return self.to_dtype().str
        elif zarr_format == 3:
            return {
                "name": self._zarr_v3_name,
                "configuration": {"length_bits": self.length * self.item_size_bits},
            }
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    @classmethod
    def _from_json_unsafe(cls, data: JSON, zarr_format: ZarrFormat) -> Self:
        if zarr_format == 2:
            return cls.from_dtype(np.dtype(data))  # type: ignore[arg-type]
        elif zarr_format == 3:
            return cls(length=data["configuration"]["length_bits"] // cls.item_size_bits)  # type: ignore[arg-type, index, call-overload, operator]
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def default_value(self) -> np.str_:
        return np.str_("")

    def to_json_value(self, data: object, *, zarr_format: ZarrFormat) -> str:
        return str(data)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.str_:
        if not check_json_str(data):
            raise TypeError(f"Invalid type: {data}. Expected a string.")
        return self.to_dtype().type(data)

    def check_value(self, data: object) -> bool:
        return isinstance(data, str | np.str_ | bytes)

    def _cast_value_unsafe(self, value: object) -> np.str_:
        return self.to_dtype().type(value)


_NUMPY_SUPPORTS_VLEN_STRING = hasattr(np.dtypes, "StringDType")


if _NUMPY_SUPPORTS_VLEN_STRING:

    @dataclass(frozen=True, kw_only=True)
    class VariableLengthString(ZDType[np.dtypes.StringDType, str]):  # type: ignore[type-var]
        dtype_cls = np.dtypes.StringDType
        _zarr_v3_name = "numpy.variable_length_utf8"

        @classmethod
        def _from_dtype_unsafe(cls, dtype: _BaseDType) -> Self:
            return cls()

        def to_dtype(self) -> np.dtypes.StringDType:
            return self.dtype_cls()

        @classmethod
        def check_json(cls, data: JSON, zarr_format: ZarrFormat) -> TypeGuard[JSON]:
            """
            Check that the input is a valid JSON representation of a numpy string dtype.
            """
            if zarr_format == 2:
                # TODO: take the entire metadata document in here, and
                # check the compressors / filters for vlen-utf8
                # Note that we are checking for the object dtype name.
                return data == "|O"
            elif zarr_format == 3:
                return data == cls._zarr_v3_name
            raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

        def to_json(self, zarr_format: ZarrFormat) -> JSON:
            if zarr_format == 2:
                # Note: unlike many other numpy data types, we don't serialize the .str attribute
                # of the data type to JSON. This is because Zarr was using `|O` for strings before the
                # numpy variable length string data type existed, and we want to be consistent with
                # that practice
                return "|O"
            elif zarr_format == 3:
                return self._zarr_v3_name
            raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

        @classmethod
        def _from_json_unsafe(cls, data: JSON, zarr_format: ZarrFormat) -> Self:
            return cls()

        def default_value(self) -> str:
            return ""

        def to_json_value(self, data: object, *, zarr_format: ZarrFormat) -> str:
            return str(data)

        def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> str:
            if not check_json_str(data):
                raise TypeError(f"Invalid type: {data}. Expected a string.")
            return data

        def check_value(self, data: object) -> bool:
            return isinstance(data, str)

        def _cast_value_unsafe(self, value: object) -> str:
            return str(value)

else:

    @dataclass(frozen=True, kw_only=True)
    class VariableLengthString(ZDType[np.dtypes.ObjectDType, str]):  # type: ignore[no-redef]
        dtype_cls = np.dtypes.ObjectDType
        _zarr_v3_name = "numpy.variable_length_utf8"

        @classmethod
        def _from_dtype_unsafe(cls, dtype: _BaseDType) -> Self:
            return cls()

        def to_dtype(self) -> np.dtypes.ObjectDType:
            return self.dtype_cls()

        @classmethod
        def check_json(cls, data: JSON, zarr_format: ZarrFormat) -> TypeGuard[JSON]:
            """
            Check that the input is a valid JSON representation of a numpy O dtype.
            """
            if zarr_format == 2:
                # TODO: take the entire metadata document in here, and
                # check the compressors / filters for vlen-utf8
                return data == "|O"
            elif zarr_format == 3:
                return data == cls._zarr_v3_name
            raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

        def to_json(self, zarr_format: ZarrFormat) -> JSON:
            if zarr_format == 2:
                return self.to_dtype().str
            elif zarr_format == 3:
                return self._zarr_v3_name
            raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

        @classmethod
        def _from_json_unsafe(cls, data: JSON, zarr_format: ZarrFormat) -> Self:
            return cls()

        def default_value(self) -> str:
            return ""

        def to_json_value(self, data: object, *, zarr_format: ZarrFormat) -> str:
            return data  # type: ignore[return-value]

        def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> str:
            """
            Strings pass through
            """
            if not check_json_str(data):
                raise TypeError(f"Invalid type: {data}. Expected a string.")
            return data

        def check_value(self, data: object) -> bool:
            return isinstance(data, str)

        def _cast_value_unsafe(self, value: object) -> str:
            return str(value)


DateUnit = Literal["Y", "M", "W", "D"]
TimeUnit = Literal["h", "m", "s", "ms", "us", "μs", "ns", "ps", "fs", "as"]


@dataclass(frozen=True, kw_only=True, slots=True)
class DateTime64(ZDType[np.dtypes.DateTime64DType, np.datetime64], HasEndianness):
    dtype_cls = np.dtypes.DateTime64DType  # type: ignore[assignment]
    _zarr_v3_name = "numpy.datetime64"
    unit: DateUnit | TimeUnit

    @classmethod
    def _from_dtype_unsafe(cls, dtype: _BaseDType) -> Self:
        unit: DateUnit | TimeUnit = dtype.name[dtype.name.rfind("[") + 1 : dtype.name.rfind("]")]  # type: ignore[assignment]
        if unit not in get_args(DateUnit) and unit not in get_args(TimeUnit):
            raise DataTypeValidationError('Invalid unit for "numpy.datetime64"')
        byteorder = cast("EndiannessNumpy", dtype.byteorder)
        return cls(unit=unit, endianness=endianness_from_numpy_str(byteorder))

    def to_dtype(self) -> np.dtypes.DateTime64DType:
        # Numpy does not allow creating datetime64 via
        # np.dtypes.DateTime64Dtype()
        return cast(
            "np.dtypes.DateTime64DType",
            np.dtype(f"datetime64[{self.unit}]").newbyteorder(
                endianness_to_numpy_str(self.endianness)
            ),
        )

    @classmethod
    def check_json(cls, data: JSON, zarr_format: ZarrFormat) -> TypeGuard[JSON]:
        if zarr_format == 2:
            # match <M[ns], >M[M], etc
            # consider making this a standalone function
            return (
                isinstance(data, str)
                and len(data) in (6, 7)
                and data[0] in (">", "<")
                and data[1:4] == "M8["
                and data[4:-1] in get_args(TimeUnit) + get_args(DateUnit)
                and data[-1] == "]"
            )
        elif zarr_format == 3:
            return (
                isinstance(data, dict)
                and "name" in data
                and data["name"] == cls._zarr_v3_name
                and "configuration" in data
                and "unit" in data["configuration"]
                and data["configuration"]["unit"] in get_args(DateUnit) + get_args(TimeUnit)
            )
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def default_value(self) -> np.datetime64:
        return np.datetime64("NaT")

    def to_json(self, zarr_format: ZarrFormat) -> JSON:
        if zarr_format == 2:
            return self.to_dtype().str
        elif zarr_format == 3:
            return {"name": self._zarr_v3_name, "configuration": {"unit": self.unit}}
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    @classmethod
    def _from_json_unsafe(cls, data: JSON, zarr_format: ZarrFormat) -> Self:
        if zarr_format == 2:
            return cls.from_dtype(np.dtype(data))  # type: ignore[arg-type]
        elif zarr_format == 3:
            return cls(unit=data["configuration"]["unit"])  # type: ignore[arg-type, index, call-overload]
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.datetime64:
        if check_json_int(data):
            return datetime_from_json(data, self.unit)
        raise TypeError(f"Invalid type: {data}. Expected an integer.")

    def to_json_value(self, data: object, *, zarr_format: ZarrFormat) -> int:
        return datetime_to_json(data)  # type: ignore[arg-type]

    def check_value(self, data: object) -> bool:
        # not sure which values we should accept for structured dtypes.
        try:
            np.array([data], dtype=self.to_dtype())
            return True  # noqa: TRY300
        except ValueError:
            return False

    def _cast_value_unsafe(self, value: object) -> np.datetime64:
        return self.to_dtype().type(value)  # type: ignore[no-any-return, call-overload]


@dataclass(frozen=True, kw_only=True)
class Structured(ZDType[np.dtypes.VoidDType[int], np.void]):
    dtype_cls = np.dtypes.VoidDType  # type: ignore[assignment]
    _zarr_v3_name = "structured"
    fields: tuple[tuple[str, ZDType[_BaseDType, _BaseScalar]], ...]

    def default_value(self) -> np.void:
        return self._cast_value_unsafe(0)

    def _cast_value_unsafe(self, value: object) -> np.void:
        return cast("np.void", np.array([value], dtype=self.to_dtype())[0])

    @classmethod
    def check_dtype(cls, dtype: _BaseDType) -> TypeGuard[np.dtypes.VoidDType[int]]:
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
    def _from_dtype_unsafe(cls, dtype: _BaseDType) -> Self:
        from zarr.core.dtype import get_data_type_from_native_dtype

        fields: list[tuple[str, ZDType[Any, Any]]] = []

        if dtype.fields is None:
            raise ValueError("numpy dtype has no fields")

        # fields of a structured numpy dtype are either 2-tuples or 3-tuples. we only
        # care about the first element in either case.
        for key, (dtype_instance, *_) in dtype.fields.items():
            dtype_wrapped = get_data_type_from_native_dtype(dtype_instance)
            fields.append((key, dtype_wrapped))

        return cls(fields=tuple(fields))

    def to_json(self, zarr_format: ZarrFormat) -> JSON:
        fields = [
            (f_name, f_dtype.to_json(zarr_format=zarr_format)) for f_name, f_dtype in self.fields
        ]
        if zarr_format == 2:
            return fields
        elif zarr_format == 3:
            base_dict = {"name": self._zarr_v3_name}
            base_dict["configuration"] = {"fields": fields}  # type: ignore[assignment]
            return cast("JSON", base_dict)
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    @classmethod
    def check_json(
        cls, data: JSON, zarr_format: ZarrFormat
    ) -> TypeGuard[dict[str, JSON] | list[Any]]:
        # the actual JSON form is recursive and hard to annotate, so we give up and do
        # list[Any] for now
        if zarr_format == 2:
            return (
                not isinstance(data, str)
                and isinstance(data, Sequence)
                and all(
                    not isinstance(field, str) and isinstance(field, Sequence) and len(field) == 2
                    for field in data
                )
            )
        elif zarr_format == 3:
            return (
                isinstance(data, dict)
                and "name" in data
                and "configuration" in data
                and isinstance(data["configuration"], dict)
                and "fields" in data["configuration"]
            )
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    @classmethod
    def _from_json_unsafe(cls, data: JSON, zarr_format: ZarrFormat) -> Self:
        from zarr.core.dtype import get_data_type_from_json

        if cls.check_json(data, zarr_format=zarr_format):
            if zarr_format == 2:
                # structured dtypes are constructed directly from a list of lists
                return cls(
                    fields=tuple(  # type: ignore[misc]
                        (f_name, get_data_type_from_json(f_dtype, zarr_format=zarr_format))
                        for f_name, f_dtype in data
                    )
                )
            elif zarr_format == 3:  # noqa: SIM102
                if isinstance(data, dict) and "configuration" in data:
                    config = data["configuration"]
                    if isinstance(config, dict) and "fields" in config:
                        meta_fields = config["fields"]
                        fields = tuple(
                            (f_name, get_data_type_from_json(f_dtype, zarr_format=zarr_format))
                            for f_name, f_dtype in meta_fields
                        )
                        return cls(fields=fields)
            raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover
        raise DataTypeValidationError(f"Invalid JSON representation of data type {cls}.")

    def to_dtype(self) -> np.dtypes.VoidDType[int]:
        return cast(
            "np.dtypes.VoidDType[int]",
            np.dtype([(key, dtype.to_dtype()) for (key, dtype) in self.fields]),
        )

    def to_json_value(self, data: object, *, zarr_format: ZarrFormat) -> str:
        return bytes_to_json(self.cast_value(data).tobytes(), zarr_format)

    def check_value(self, data: object) -> bool:
        # not sure which values we should accept for structured dtypes.
        try:
            np.array([data], dtype=self.to_dtype())
            return True  # noqa: TRY300
        except ValueError:
            return False

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.void:
        if not check_json_str(data):
            raise TypeError(f"Invalid type: {data}. Expected a string.")
        as_bytes = bytes_from_json(data, zarr_format=zarr_format)
        dtype = self.to_dtype()
        return cast("np.void", np.array([as_bytes], dtype=dtype.str).view(dtype)[0])


def endianness_to_numpy_str(endianness: Endianness | None) -> EndiannessNumpy:
    """
    Convert an endianness literal to its numpy string representation.

    Parameters
    ----------
    endianness : Endianness or None
        The endianness to convert.

    Returns
    -------
    Literal[">", "<", "|"]
        The numpy string representation of the endianness.

    Raises
    ------
    ValueError
        If the endianness is invalid.
    """
    match endianness:
        case "little":
            return "<"
        case "big":
            return ">"
        case None:
            return "|"
    raise ValueError(
        f"Invalid endianness: {endianness}. Expected one of {get_args(Endianness)} or None"
    )


def endianness_from_numpy_str(endianness: EndiannessNumpy) -> Endianness | None:
    """
    Convert a numpy endianness string literal to a human-readable literal value.

    Parameters
    ----------
    endianness : Literal[">", "<", "=", "|"]
        The numpy string representation of the endianness.

    Returns
    -------
    Endianness or None
        The human-readable representation of the endianness.

    Raises
    ------
    ValueError
        If the endianness is invalid.
    """
    match endianness:
        case "=":
            return sys.byteorder
        case "<":
            return "little"
        case ">":
            return "big"
        case "|":
            return None
    raise ValueError(
        f"Invalid endianness: {endianness}. Expected one of {get_args(EndiannessNumpy)}"
    )
