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
    TypeGuard,
    cast,
    get_args,
)

import numpy as np

from zarr.core.dtype.common import (
    DataTypeValidationError,
    Endianness,
    JSONFloat,
    bytes_from_json,
    bytes_to_json,
    check_json_bool,
    check_json_complex_float,
    check_json_float,
    check_json_int,
    check_json_str,
    complex_from_json,
    complex_to_json,
    datetime_from_json,
    datetime_to_json,
    float_from_json,
    float_to_json,
)
from zarr.core.dtype.wrapper import ZDType, _BaseDType, _BaseScalar

if TYPE_CHECKING:
    from zarr.core.common import JSON, ZarrFormat

EndiannessNumpy = Literal[">", "<", "|", "="]


@dataclass(frozen=True, kw_only=True)
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
    def _from_dtype_unsafe(cls, dtype: np.dtypes.BoolDType) -> Self:
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
            return np.bool_(data)
        raise TypeError(f"Invalid type: {data}. Expected a boolean.")


@dataclass(frozen=True, kw_only=True)
class Int8(ZDType[np.dtypes.Int8DType, np.int8]):
    dtype_cls = np.dtypes.Int8DType
    _zarr_v3_name = "int8"
    _zarr_v2_names: ClassVar[tuple[str, ...]] = ("|i1",)

    @classmethod
    def _from_dtype_unsafe(cls, dtype: np.dtypes.Int8DType) -> Self:
        return cls()

    def to_dtype(self: Self) -> np.dtypes.Int8DType:
        return self.dtype_cls()

    @classmethod
    def check_json(cls, data: JSON, zarr_format: ZarrFormat) -> TypeGuard[Literal["int8", "|i1"]]:
        """
        Check that the input is a valid JSON representation of a 8-bit integer.
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

    def default_value(self) -> np.int8:
        """
        Get the default value.

        Returns
        -------
        np.int8
            The default value.
        """
        return np.int8(0)

    def to_json_value(self, data: np.int8, zarr_format: ZarrFormat) -> int:
        """
        Convert a numpy 8-bit int to JSON-serializable format.

        Parameters
        ----------
        data : np.int8
            The value to convert.
        zarr_format : ZarrFormat
            The zarr format version.

        Returns
        -------
        int
            The JSON-serializable form of the scalar.
        """
        return int(data)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.int8:
        """
        Read a JSON-serializable value as a numpy int8 scalar.

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
        if check_json_int(data):
            return np.int8(data)
        raise TypeError(f"Invalid type: {data}. Expected an integer.")


@dataclass(frozen=True, kw_only=True)
class UInt8(ZDType[np.dtypes.UInt8DType, np.uint8]):
    dtype_cls = np.dtypes.UInt8DType
    _zarr_v3_name = "uint8"
    _zarr_v2_names: ClassVar[tuple[str, ...]] = ("|u1",)

    @classmethod
    def _from_dtype_unsafe(cls, dtype: np.dtypes.UInt8DType) -> Self:
        return cls()

    def to_dtype(self: Self) -> np.dtypes.UInt8DType:
        return self.dtype_cls()

    @classmethod
    def check_json(cls, data: JSON, zarr_format: ZarrFormat) -> TypeGuard[Literal["uint8", "|u1"]]:
        """
        Check that the input is a valid JSON representation of an unsigned 8-bit integer.
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

    def default_value(self) -> np.uint8:
        """
        Get the default value for this data type.

        Returns
        -------
        np.uint8
            The default value.
        """
        return np.uint8(0)

    def to_json_value(self, data: np.uint8, zarr_format: ZarrFormat) -> int:
        """
        Convert a numpy unsigned 8-bit integer to JSON-serializable format.

        Parameters
        ----------
        data : np.uint8
            The value to convert.
        zarr_format : ZarrFormat
            The zarr format version.

        Returns
        -------
        int
            The JSON-serializable form of the scalar.
        """
        return int(data)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.uint8:
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
        if check_json_int(data):
            return np.uint8(data)
        raise TypeError(f"Invalid type: {data}. Expected an integer.")


@dataclass(frozen=True, kw_only=True)
class Int16(ZDType[np.dtypes.Int16DType, np.int16]):
    dtype_cls = np.dtypes.Int16DType
    _zarr_v3_name = "int16"
    _zarr_v2_names: ClassVar[tuple[str, ...]] = (">i2", "<i2")
    endianness: Endianness | None = "little"

    @classmethod
    def _from_dtype_unsafe(cls, dtype: np.dtypes.Int16DType) -> Self:
        byte_order = cast("EndiannessNumpy", dtype.byteorder)
        return cls(endianness=endianness_from_numpy_str(byte_order))

    def to_dtype(self) -> np.dtypes.Int16DType:
        byte_order = endianness_to_numpy_str(self.endianness)
        return self.dtype_cls().newbyteorder(byte_order)

    @classmethod
    def check_json(
        cls, data: JSON, zarr_format: ZarrFormat
    ) -> TypeGuard[Literal["int16", ">i2", "<i2"]]:
        """
        Check that the input is a valid JSON representation of a signed 16-bit integer.
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
        if zarr_format == 2:
            return cls.from_dtype(np.dtype(data))  # type: ignore[arg-type]
        elif zarr_format == 3:
            return cls()
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def default_value(self) -> np.int16:
        return self.to_dtype().type(0)

    def to_json_value(self, data: np.int16, zarr_format: ZarrFormat) -> int:
        return int(data)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.int16:
        if check_json_int(data):
            return self.to_dtype().type(data)
        raise TypeError(f"Invalid type: {data}. Expected an integer.")


@dataclass(frozen=True, kw_only=True)
class UInt16(ZDType[np.dtypes.UInt16DType, np.uint16]):
    dtype_cls = np.dtypes.UInt16DType
    _zarr_v3_name = "uint16"
    _zarr_v2_names: ClassVar[tuple[str, ...]] = (">u2", "<u2")
    endianness: Endianness | None = "little"

    @classmethod
    def _from_dtype_unsafe(cls, dtype: np.dtypes.UInt16DType) -> Self:
        byte_order = cast("EndiannessNumpy", dtype.byteorder)
        return cls(endianness=endianness_from_numpy_str(byte_order))

    def to_dtype(self) -> np.dtypes.UInt16DType:
        byte_order = endianness_to_numpy_str(self.endianness)
        return self.dtype_cls().newbyteorder(byte_order)

    @classmethod
    def check_json(
        cls, data: JSON, zarr_format: ZarrFormat
    ) -> TypeGuard[Literal["uint16", ">u2", "<u2"]]:
        """
        Check that the input is a valid JSON representation of an unsigned 16-bit integer.
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
        if zarr_format == 2:
            return cls.from_dtype(np.dtype(data))  # type: ignore[arg-type]
        elif zarr_format == 3:
            return cls()
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def default_value(self) -> np.uint16:
        return self.to_dtype().type(0)

    def to_json_value(self, data: np.uint16, zarr_format: ZarrFormat) -> int:
        return int(data)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.uint16:
        if check_json_int(data):
            return self.to_dtype().type(data)
        raise TypeError(f"Invalid type: {data}. Expected an integer.")


@dataclass(frozen=True, kw_only=True)
class Int32(ZDType[np.dtypes.Int32DType, np.int32]):
    dtype_cls = np.dtypes.Int32DType
    _zarr_v3_name = "int32"
    _zarr_v2_names: ClassVar[tuple[str, ...]] = (">i4", "<i4")
    endianness: Endianness | None = "little"

    @classmethod
    def from_dtype(cls: type[Self], dtype: _BaseDType) -> Self:
        # We override the base implementation to address a windows-specific, pre-numpy 2 issue where
        # ``np.dtype('i')`` is an instance of ``np.dtypes.IntDType`` that acts like `int32` instead of ``np.dtype('int32')``
        if dtype == np.dtypes.Int32DType():
            return cls._from_dtype_unsafe(np.dtypes.Int32DType().newbyteorder(dtype.byteorder))
        else:
            return super().from_dtype(dtype)

    @classmethod
    def _from_dtype_unsafe(cls, dtype: np.dtypes.Int32DType) -> Self:
        byte_order = cast("EndiannessNumpy", dtype.byteorder)
        return cls(endianness=endianness_from_numpy_str(byte_order))

    def to_dtype(self) -> np.dtypes.Int32DType:
        byte_order = endianness_to_numpy_str(self.endianness)
        return self.dtype_cls().newbyteorder(byte_order)

    @classmethod
    def check_json(
        cls, data: JSON, zarr_format: ZarrFormat
    ) -> TypeGuard[Literal["int32", ">i4", "<i4"]]:
        """
        Check that the input is a valid JSON representation of a signed 32-bit integer.
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
        if zarr_format == 2:
            return cls.from_dtype(np.dtype(data))  # type: ignore[arg-type]
        elif zarr_format == 3:
            return cls()
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def default_value(self) -> np.int32:
        return self.to_dtype().type(0)

    def to_json_value(self, data: np.int32, zarr_format: ZarrFormat) -> int:
        return int(data)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.int32:
        if check_json_int(data):
            return self.to_dtype().type(data)
        raise TypeError(f"Invalid type: {data}. Expected an integer.")


@dataclass(frozen=True, kw_only=True)
class UInt32(ZDType[np.dtypes.UInt32DType, np.uint32]):
    dtype_cls = np.dtypes.UInt32DType
    _zarr_v3_name = "uint32"
    _zarr_v2_names: ClassVar[tuple[str, ...]] = (">u4", "<u4")
    endianness: Endianness | None = "little"

    @classmethod
    def _from_dtype_unsafe(cls, dtype: np.dtypes.UInt32DType) -> Self:
        byte_order = cast("EndiannessNumpy", dtype.byteorder)
        return cls(endianness=endianness_from_numpy_str(byte_order))

    def to_dtype(self) -> np.dtypes.UInt32DType:
        byte_order = endianness_to_numpy_str(self.endianness)
        return self.dtype_cls().newbyteorder(byte_order)

    @classmethod
    def check_json(
        cls, data: JSON, zarr_format: ZarrFormat
    ) -> TypeGuard[Literal["uint32", ">u4", "<u4"]]:
        """
        Check that the input is a valid JSON representation of an unsigned 16-bit integer.
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
        if zarr_format == 2:
            return cls.from_dtype(np.dtype(data))  # type: ignore[arg-type]
        elif zarr_format == 3:
            return cls()
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def default_value(self) -> np.uint32:
        return self.to_dtype().type(0)

    def to_json_value(self, data: np.uint32, zarr_format: ZarrFormat) -> int:
        return int(data)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.uint32:
        if check_json_int(data):
            return self.to_dtype().type(data)
        raise TypeError(f"Invalid type: {data}. Expected an integer.")


@dataclass(frozen=True, kw_only=True)
class Int64(ZDType[np.dtypes.Int64DType, np.int64]):
    dtype_cls = np.dtypes.Int64DType
    _zarr_v3_name = "int64"
    _zarr_v2_names: ClassVar[tuple[str, ...]] = (">i8", "<i8")
    endianness: Endianness | None = "little"

    @classmethod
    def _from_dtype_unsafe(cls, dtype: np.dtypes.Int64DType) -> Self:
        byte_order = cast("EndiannessNumpy", dtype.byteorder)
        return cls(endianness=endianness_from_numpy_str(byte_order))

    def to_dtype(self) -> np.dtypes.Int64DType:
        byte_order = endianness_to_numpy_str(self.endianness)
        return self.dtype_cls().newbyteorder(byte_order)

    @classmethod
    def check_json(
        cls, data: JSON, zarr_format: ZarrFormat
    ) -> TypeGuard[Literal["int64", ">i8", "<i8"]]:
        """
        Check that the input is a valid JSON representation of a signed 16-bit integer.
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
        if zarr_format == 2:
            return cls.from_dtype(np.dtype(data))  # type: ignore[arg-type]
        elif zarr_format == 3:
            return cls()
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def default_value(self) -> np.int64:
        return self.to_dtype().type(0)

    def to_json_value(self, data: np.int64, zarr_format: ZarrFormat) -> int:
        return int(data)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.int64:
        if check_json_int(data):
            return self.to_dtype().type(data)
        raise TypeError(f"Invalid type: {data}. Expected an integer.")


@dataclass(frozen=True, kw_only=True)
class UInt64(ZDType[np.dtypes.UInt64DType, np.uint64]):
    dtype_cls = np.dtypes.UInt64DType
    _zarr_v3_name = "uint64"
    _zarr_v2_names: ClassVar[tuple[str, ...]] = (">u8", "<u8")
    endianness: Endianness | None = "little"

    @classmethod
    def _from_dtype_unsafe(cls, dtype: np.dtypes.UInt64DType) -> Self:
        byte_order = cast("EndiannessNumpy", dtype.byteorder)
        return cls(endianness=endianness_from_numpy_str(byte_order))

    def to_dtype(self) -> np.dtypes.UInt64DType:
        byte_order = endianness_to_numpy_str(self.endianness)
        return self.dtype_cls().newbyteorder(byte_order)

    @classmethod
    def check_json(
        cls, data: JSON, zarr_format: ZarrFormat
    ) -> TypeGuard[Literal["uint64", ">u8", "<u8"]]:
        """
        Check that the input is a valid JSON representation of a signed 16-bit integer.
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
        if zarr_format == 2:
            return cls.from_dtype(np.dtype(data))  # type: ignore[arg-type]
        elif zarr_format == 3:
            return cls()
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def default_value(self) -> np.uint64:
        return self.to_dtype().type(0)

    def to_json_value(self, data: np.uint64, zarr_format: ZarrFormat) -> int:
        return int(data)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.uint64:
        if check_json_int(data):
            return self.to_dtype().type(data)
        raise TypeError(f"Invalid type: {data}. Expected an integer.")


@dataclass(frozen=True, kw_only=True)
class Float16(ZDType[np.dtypes.Float16DType, np.float16]):
    dtype_cls = np.dtypes.Float16DType
    _zarr_v3_name = "float16"
    _zarr_v2_names: ClassVar[tuple[str, ...]] = (">f2", "<f2")
    endianness: Endianness | None = "little"

    @classmethod
    def _from_dtype_unsafe(cls, dtype: np.dtypes.Float16DType) -> Self:
        byte_order = cast("EndiannessNumpy", dtype.byteorder)
        return cls(endianness=endianness_from_numpy_str(byte_order))

    def to_dtype(self) -> np.dtypes.Float16DType:
        byte_order = endianness_to_numpy_str(self.endianness)
        return self.dtype_cls().newbyteorder(byte_order)

    @classmethod
    def check_json(
        cls, data: JSON, zarr_format: ZarrFormat
    ) -> TypeGuard[Literal["float", ">f2", "<f2"]]:
        """
        Check that the input is a valid JSON representation of a signed 16-bit integer.
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
        if zarr_format == 2:
            return cls.from_dtype(np.dtype(data))  # type: ignore[arg-type]
        elif zarr_format == 3:
            return cls()
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def default_value(self) -> np.float16:
        return self.to_dtype().type(0)

    def to_json_value(self, data: np.float16, zarr_format: ZarrFormat) -> JSONFloat:
        return float_to_json(data, zarr_format)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.float16:
        if check_json_float(data, zarr_format=zarr_format):
            return self.to_dtype().type(float_from_json(data, zarr_format))
        raise TypeError(f"Invalid type: {data}. Expected a float.")


@dataclass(frozen=True, kw_only=True)
class Float32(ZDType[np.dtypes.Float32DType, np.float32]):
    dtype_cls = np.dtypes.Float32DType
    _zarr_v3_name = "float32"
    _zarr_v2_names: ClassVar[tuple[str, ...]] = (">f4", "<f4")
    endianness: Endianness | None = "little"

    @classmethod
    def _from_dtype_unsafe(cls, dtype: np.dtypes.Float32DType) -> Self:
        byte_order = cast("EndiannessNumpy", dtype.byteorder)
        return cls(endianness=endianness_from_numpy_str(byte_order))

    def to_dtype(self) -> np.dtypes.Float32DType:
        byte_order = endianness_to_numpy_str(self.endianness)
        return self.dtype_cls().newbyteorder(byte_order)

    @classmethod
    def check_json(
        cls, data: JSON, zarr_format: ZarrFormat
    ) -> TypeGuard[Literal["float32", ">f4", "<f4"]]:
        """
        Check that the input is a valid JSON representation of a signed 16-bit integer.
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
        if zarr_format == 2:
            return cls.from_dtype(np.dtype(data))  # type: ignore[arg-type]
        elif zarr_format == 3:
            return cls()
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def default_value(self) -> np.float32:
        return self.to_dtype().type(0)

    def to_json_value(self, data: np.float32, zarr_format: ZarrFormat) -> JSONFloat:
        return float_to_json(data, zarr_format)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.float32:
        if check_json_float(data, zarr_format=zarr_format):
            return self.to_dtype().type(float_from_json(data, zarr_format))
        raise TypeError(f"Invalid type: {data}. Expected a float.")


@dataclass(frozen=True, kw_only=True)
class Float64(ZDType[np.dtypes.Float64DType, np.float64]):
    dtype_cls = np.dtypes.Float64DType
    _zarr_v3_name = "float64"
    _zarr_v2_names: ClassVar[tuple[str, ...]] = (">f8", "<f8")
    endianness: Endianness | None = "little"

    @classmethod
    def _from_dtype_unsafe(cls, dtype: np.dtypes.Float64DType) -> Self:
        byte_order = cast("EndiannessNumpy", dtype.byteorder)
        return cls(endianness=endianness_from_numpy_str(byte_order))

    def to_dtype(self) -> np.dtypes.Float64DType:
        byte_order = endianness_to_numpy_str(self.endianness)
        return self.dtype_cls().newbyteorder(byte_order)

    @classmethod
    def check_json(
        cls, data: JSON, zarr_format: ZarrFormat
    ) -> TypeGuard[Literal["float64", ">f8", "<f8"]]:
        """
        Check that the input is a valid JSON representation of a signed 16-bit integer.
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
        if zarr_format == 2:
            return cls.from_dtype(np.dtype(data))  # type: ignore[arg-type]
        elif zarr_format == 3:
            return cls()
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def default_value(self) -> np.float64:
        return self.to_dtype().type(0)

    def to_json_value(self, data: np.float64, zarr_format: ZarrFormat) -> JSONFloat:
        return float_to_json(data, zarr_format)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.float64:
        if check_json_float(data, zarr_format=zarr_format):
            return self.to_dtype().type(float_from_json(data, zarr_format))
        raise TypeError(f"Invalid type: {data}. Expected a float.")


@dataclass(frozen=True, kw_only=True)
class Complex64(ZDType[np.dtypes.Complex64DType, np.complex64]):
    dtype_cls = np.dtypes.Complex64DType
    _zarr_v3_name = "complex64"
    _zarr_v2_names: ClassVar[tuple[str, ...]] = (">c8", "<c8")
    endianness: Endianness | None = "little"

    @classmethod
    def _from_dtype_unsafe(cls, dtype: np.dtypes.Complex64DType) -> Self:
        byte_order = cast("EndiannessNumpy", dtype.byteorder)
        return cls(endianness=endianness_from_numpy_str(byte_order))

    def to_dtype(self) -> np.dtypes.Complex64DType:
        byte_order = endianness_to_numpy_str(self.endianness)
        return self.dtype_cls().newbyteorder(byte_order)

    @classmethod
    def check_json(
        cls, data: JSON, zarr_format: ZarrFormat
    ) -> TypeGuard[Literal["complex64", ">c8", "<c8"]]:
        """
        Check that the input is a valid JSON representation of a signed 16-bit integer.
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
        if zarr_format == 2:
            return cls.from_dtype(np.dtype(data))  # type: ignore[arg-type]
        elif zarr_format == 3:
            return cls()
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def default_value(self) -> np.complex64:
        return self.to_dtype().type(0)

    def to_json_value(
        self, data: np.complex64, zarr_format: ZarrFormat
    ) -> tuple[JSONFloat, JSONFloat]:
        return complex_to_json(data, zarr_format)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.complex64:
        if check_json_complex_float(data, zarr_format=zarr_format):
            return complex_from_json(data, dtype=self.to_dtype(), zarr_format=zarr_format)
        raise TypeError(f"Invalid type: {data}. Expected a complex float.")


@dataclass(frozen=True, kw_only=True)
class Complex128(ZDType[np.dtypes.Complex128DType, np.complex128]):
    dtype_cls = np.dtypes.Complex128DType
    _zarr_v3_name = "complex128"
    _zarr_v2_names: ClassVar[tuple[str, ...]] = (">c16", "<c16")
    endianness: Endianness | None = "little"

    @classmethod
    def _from_dtype_unsafe(cls, dtype: np.dtypes.Complex128DType) -> Self:
        byte_order = cast("EndiannessNumpy", dtype.byteorder)
        return cls(endianness=endianness_from_numpy_str(byte_order))

    def to_dtype(self) -> np.dtypes.Complex128DType:
        byte_order = endianness_to_numpy_str(self.endianness)
        return self.dtype_cls().newbyteorder(byte_order)

    @classmethod
    def check_json(
        cls, data: JSON, zarr_format: ZarrFormat
    ) -> TypeGuard[Literal["complex128", ">c16", "<c16"]]:
        """
        Check that the input is a valid JSON representation of a signed 16-bit integer.
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
        if zarr_format == 2:
            return cls.from_dtype(np.dtype(data))  # type: ignore[arg-type]
        elif zarr_format == 3:
            return cls()
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def default_value(self) -> np.complex128:
        return self.to_dtype().type(0)

    def to_json_value(
        self, data: np.complex128, zarr_format: ZarrFormat
    ) -> tuple[JSONFloat, JSONFloat]:
        return complex_to_json(data, zarr_format)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.complex128:
        if check_json_complex_float(data, zarr_format=zarr_format):
            return complex_from_json(data, dtype=self.to_dtype(), zarr_format=zarr_format)
        raise TypeError(f"Invalid type: {data}. Expected a complex float.")


@dataclass(frozen=True, kw_only=True)
class FixedLengthAscii(ZDType[np.dtypes.BytesDType[int], np.bytes_]):
    dtype_cls = np.dtypes.BytesDType
    _zarr_v3_name = "numpy.fixed_length_ascii"
    item_size_bits: ClassVar[int] = 8
    length: int = 1

    @classmethod
    def _from_dtype_unsafe(cls, dtype: np.dtypes.BytesDType[int]) -> Self:
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

    def to_json_value(self, data: np.bytes_, *, zarr_format: ZarrFormat) -> str:
        return base64.standard_b64encode(data).decode("ascii")

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.bytes_:
        if check_json_str(data):
            return self.to_dtype().type(base64.standard_b64decode(data.encode("ascii")))
        raise TypeError(f"Invalid type: {data}. Expected a string.")


@dataclass(frozen=True, kw_only=True)
class FixedLengthBytes(ZDType[np.dtypes.VoidDType[int], np.void]):
    # np.dtypes.VoidDType is specified in an odd way in numpy
    # it cannot be used to create instances of the dtype
    # so we have to tell mypy to ignore this here
    dtype_cls = np.dtypes.VoidDType  # type: ignore[assignment]
    _zarr_v3_name = "numpy.void"
    item_size_bits: ClassVar[int] = 8
    length: int = 1

    @classmethod
    def _from_dtype_unsafe(cls, dtype: np.dtypes.VoidDType[int]) -> Self:
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

    def to_json_value(self, data: np.void, *, zarr_format: ZarrFormat) -> str:
        return base64.standard_b64encode(data.tobytes()).decode("ascii")

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.void:
        if check_json_str(data):
            return self.to_dtype().type(base64.standard_b64decode(data))
        raise DataTypeValidationError(f"Invalid type: {data}. Expected a string.")


@dataclass(frozen=True, kw_only=True)
class FixedLengthUnicode(ZDType[np.dtypes.StrDType[int], np.str_]):
    dtype_cls = np.dtypes.StrDType
    _zarr_v3_name = "numpy.fixed_length_ucs4"
    item_size_bits: ClassVar[int] = 32  # UCS4 is 32 bits per code point
    endianness: Endianness | None = "little"
    length: int = 1

    @classmethod
    def _from_dtype_unsafe(cls, dtype: np.dtypes.StrDType[int]) -> Self:
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

    def to_json_value(self, data: np.str_, *, zarr_format: ZarrFormat) -> str:
        return str(data)

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.str_:
        if not check_json_str(data):
            raise TypeError(f"Invalid type: {data}. Expected a string.")
        return self.to_dtype().type(data)


_NUMPY_SUPPORTS_VLEN_STRING = hasattr(np.dtypes, "StringDType")


if _NUMPY_SUPPORTS_VLEN_STRING:

    @dataclass(frozen=True, kw_only=True)
    class VariableLengthString(ZDType[np.dtypes.StringDType, str]):  # type: ignore[type-var]
        dtype_cls = np.dtypes.StringDType
        _zarr_v3_name = "numpy.variable_length_utf8"

        @classmethod
        def _from_dtype_unsafe(cls, dtype: np.dtypes.StringDType) -> Self:
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

else:

    @dataclass(frozen=True, kw_only=True)
    class VariableLengthString(ZDType[np.dtypes.ObjectDType, str]):  # type: ignore[no-redef]
        dtype_cls = np.dtypes.ObjectDType
        _zarr_v3_name = "numpy.variable_length_utf8"

        @classmethod
        def _from_dtype_unsafe(cls, dtype: np.dtypes.ObjectDType) -> Self:
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

        def to_json_value(self, data: str, *, zarr_format: ZarrFormat) -> str:
            return data

        def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> str:
            """
            Strings pass through
            """
            if not check_json_str(data):
                raise TypeError(f"Invalid type: {data}. Expected a string.")
            return data


DateUnit = Literal["Y", "M", "W", "D"]
TimeUnit = Literal["h", "m", "s", "ms", "us", "μs", "ns", "ps", "fs", "as"]


@dataclass(frozen=True, kw_only=True)
class DateTime64(ZDType[np.dtypes.DateTime64DType, np.datetime64]):
    dtype_cls = np.dtypes.DateTime64DType  # type: ignore[assignment]
    _zarr_v3_name = "numpy.datetime64"
    unit: DateUnit | TimeUnit = "s"
    endianness: Endianness | None = "little"

    @classmethod
    def _from_dtype_unsafe(cls, dtype: np.dtypes.DateTime64DType) -> Self:
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

    def to_json_value(self, data: np.datetime64, *, zarr_format: ZarrFormat) -> int:
        return datetime_to_json(data)


@dataclass(frozen=True, kw_only=True)
class Structured(ZDType[np.dtypes.VoidDType[int], np.void]):
    dtype_cls = np.dtypes.VoidDType  # type: ignore[assignment]
    _zarr_v3_name = "structured"
    fields: tuple[tuple[str, ZDType[_BaseDType, _BaseScalar]], ...]

    def default_value(self) -> np.void:
        return self.cast_value(0)

    def cast_value(self, value: object) -> np.void:
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
    def _from_dtype_unsafe(cls, dtype: np.dtypes.VoidDType[int]) -> Self:
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

    def to_json_value(self, data: np.generic, *, zarr_format: ZarrFormat) -> str:
        return bytes_to_json(data.tobytes(), zarr_format)

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
