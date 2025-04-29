from dataclasses import dataclass
from typing import ClassVar, Self, SupportsIndex, SupportsInt, TypeGuard, TypeVar, cast

import numpy as np

from zarr.core.common import JSON, ZarrFormat
from zarr.core.dtype.common import HasEndianness
from zarr.core.dtype.npy.common import (
    EndiannessNumpy,
    check_json_int,
    endianness_from_numpy_str,
    endianness_to_numpy_str,
)
from zarr.core.dtype.wrapper import ZDType, _BaseDType

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
IntLike = SupportsInt | SupportsIndex | bytes | str


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
