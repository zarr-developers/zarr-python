from dataclasses import dataclass
from typing import (
    ClassVar,
    Literal,
    Self,
    SupportsIndex,
    SupportsInt,
    TypeGuard,
    TypeVar,
    overload,
)

import numpy as np

from zarr.core.common import JSON, ZarrFormat
from zarr.core.dtype.common import DataTypeValidationError, HasEndianness, HasItemSize
from zarr.core.dtype.npy.common import (
    check_json_int,
    endianness_to_numpy_str,
    get_endianness_from_numpy_dtype,
)
from zarr.core.dtype.wrapper import TBaseDType, ZDType

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
class BaseInt(ZDType[TIntDType_co, TIntScalar_co], HasItemSize):
    # This attribute holds the possible zarr V2 JSON names for the data type
    _zarr_v2_names: ClassVar[tuple[str, ...]]

    @classmethod
    def _check_json_v2(cls, data: JSON, *, object_codec_id: str | None = None) -> TypeGuard[str]:
        """
        Check that the input is a valid JSON representation of this data type.
        """
        return data in cls._zarr_v2_names

    @classmethod
    def _check_json_v3(cls, data: JSON) -> TypeGuard[str]:
        """
        Check that a JSON value is consistent with the zarr v3 spec for this data type.
        """
        return data == cls._zarr_v3_name

    def _check_scalar(self, data: object) -> TypeGuard[IntLike]:
        """
        Check that a python object is IntLike
        """
        return isinstance(data, IntLike)

    def _cast_scalar_unchecked(self, data: IntLike) -> TIntScalar_co:
        """
        Create an integer without any type checking of the input.
        """
        return self.to_native_dtype().type(data)  # type: ignore[return-value]

    def cast_scalar(self, data: object) -> TIntScalar_co:
        if self._check_scalar(data):
            return self._cast_scalar_unchecked(data)
        msg = f"Cannot convert object with type {type(data)} to a numpy integer."
        raise TypeError(msg)

    def default_scalar(self) -> TIntScalar_co:
        """
        Get the default value, which is 0 cast to this dtype

        Returns
        -------
        Int scalar
            The default value.
        """
        return self._cast_scalar_unchecked(0)

    def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> TIntScalar_co:
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
            return self._cast_scalar_unchecked(data)
        raise TypeError(f"Invalid type: {data}. Expected an integer.")

    def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> int:
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
        return int(self.cast_scalar(data))


@dataclass(frozen=True, kw_only=True)
class Int8(BaseInt[np.dtypes.Int8DType, np.int8]):
    dtype_cls = np.dtypes.Int8DType
    _zarr_v3_name: ClassVar[Literal["int8"]] = "int8"
    _zarr_v2_names: ClassVar[tuple[Literal["|i1"]]] = ("|i1",)

    @classmethod
    def from_native_dtype(cls, dtype: TBaseDType) -> Self:
        """
        Create a Int8 from a np.dtype('int8') instance.
        """
        if cls._check_native_dtype(dtype):
            return cls()
        raise DataTypeValidationError(
            f"Invalid data type: {dtype}. Expected an instance of {cls.dtype_cls}"
        )

    def to_native_dtype(self: Self) -> np.dtypes.Int8DType:
        return self.dtype_cls()

    @classmethod
    def from_json_v2(cls, data: JSON, *, object_codec_id: str | None = None) -> Self:
        if cls._check_json_v2(data, object_codec_id=object_codec_id):
            return cls()
        msg = f"Invalid JSON representation of Int8. Got {data!r}, expected the string {cls._zarr_v2_names[0]!r}"
        raise DataTypeValidationError(msg)

    @classmethod
    def from_json_v3(cls, data: JSON) -> Self:
        if cls._check_json_v3(data):
            return cls()
        msg = f"Invalid JSON representation of Int8. Got {data!r}, expected the string {cls._zarr_v3_name!r}"
        raise DataTypeValidationError(msg)

    @overload
    def to_json(self, zarr_format: Literal[2]) -> Literal["|i1"]: ...

    @overload
    def to_json(self, zarr_format: Literal[3]) -> Literal["int8"]: ...

    def to_json(self, zarr_format: ZarrFormat) -> Literal["int8", "|i1"]:
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
            return self._zarr_v2_names[0]
        elif zarr_format == 3:
            return self._zarr_v3_name
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    @property
    def item_size(self) -> int:
        return 1


@dataclass(frozen=True, kw_only=True)
class UInt8(BaseInt[np.dtypes.UInt8DType, np.uint8]):
    dtype_cls = np.dtypes.UInt8DType
    _zarr_v3_name: ClassVar[Literal["uint8"]] = "uint8"
    _zarr_v2_names: ClassVar[tuple[Literal["|u1"]]] = ("|u1",)

    @classmethod
    def from_native_dtype(cls, dtype: TBaseDType) -> Self:
        """
        Create a Bool from a np.dtype('uint8') instance.
        """
        if cls._check_native_dtype(dtype):
            return cls()
        raise DataTypeValidationError(
            f"Invalid data type: {dtype}. Expected an instance of {cls.dtype_cls}"
        )

    def to_native_dtype(self: Self) -> np.dtypes.UInt8DType:
        return self.dtype_cls()

    @classmethod
    def from_json_v2(cls, data: JSON, *, object_codec_id: str | None = None) -> Self:
        if cls._check_json_v2(data, object_codec_id=object_codec_id):
            return cls()
        msg = f"Invalid JSON representation of UInt8. Got {data!r}, expected the string {cls._zarr_v2_names[0]!r}"
        raise DataTypeValidationError(msg)

    @classmethod
    def from_json_v3(cls, data: JSON) -> Self:
        if cls._check_json_v3(data):
            return cls()
        msg = f"Invalid JSON representation of UInt8. Got {data!r}, expected the string {cls._zarr_v3_name!r}"
        raise DataTypeValidationError(msg)

    @overload
    def to_json(self, zarr_format: Literal[2]) -> Literal["|u1"]: ...

    @overload
    def to_json(self, zarr_format: Literal[3]) -> Literal["uint8"]: ...

    def to_json(self, zarr_format: ZarrFormat) -> Literal["uint8", "|u1"]:
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
            return self._zarr_v2_names[0]
        elif zarr_format == 3:
            return self._zarr_v3_name
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    @property
    def item_size(self) -> int:
        return 1


@dataclass(frozen=True, kw_only=True)
class Int16(BaseInt[np.dtypes.Int16DType, np.int16], HasEndianness):
    dtype_cls = np.dtypes.Int16DType
    _zarr_v3_name: ClassVar[Literal["int16"]] = "int16"
    _zarr_v2_names: ClassVar[tuple[Literal[">i2"], Literal["<i2"]]] = (">i2", "<i2")

    @classmethod
    def from_native_dtype(cls, dtype: TBaseDType) -> Self:
        if cls._check_native_dtype(dtype):
            return cls(endianness=get_endianness_from_numpy_dtype(dtype))
        raise DataTypeValidationError(
            f"Invalid data type: {dtype}. Expected an instance of {cls.dtype_cls}"
        )

    def to_native_dtype(self) -> np.dtypes.Int16DType:
        byte_order = endianness_to_numpy_str(self.endianness)
        return self.dtype_cls().newbyteorder(byte_order)

    @classmethod
    def from_json_v2(cls, data: JSON, *, object_codec_id: str | None = None) -> Self:
        if cls._check_json_v2(data, object_codec_id=object_codec_id):
            # Going via numpy ensures that we get the endianness correct without
            # annoying string parsing.
            return cls.from_native_dtype(np.dtype(data))
        msg = f"Invalid JSON representation of Int16. Got {data!r}, expected one of the strings {cls._zarr_v2_names!r}."
        raise DataTypeValidationError(msg)

    @classmethod
    def from_json_v3(cls, data: JSON) -> Self:
        if cls._check_json_v3(data):
            return cls()
        msg = f"Invalid JSON representation of Int16. Got {data!r}, expected the string {cls._zarr_v3_name!r}"
        raise DataTypeValidationError(msg)

    @overload
    def to_json(self, zarr_format: Literal[2]) -> Literal[">i2", "<i2"]: ...

    @overload
    def to_json(self, zarr_format: Literal[3]) -> Literal["int16"]: ...

    def to_json(self, zarr_format: ZarrFormat) -> Literal["int16", ">i2", "<i2"]:
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
            return self.to_native_dtype().str
        elif zarr_format == 3:
            return self._zarr_v3_name
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    @property
    def item_size(self) -> int:
        return 2


@dataclass(frozen=True, kw_only=True)
class UInt16(BaseInt[np.dtypes.UInt16DType, np.uint16], HasEndianness):
    dtype_cls = np.dtypes.UInt16DType
    _zarr_v3_name: ClassVar[Literal["uint16"]] = "uint16"
    _zarr_v2_names: ClassVar[tuple[Literal[">u2"], Literal["<u2"]]] = (">u2", "<u2")

    @classmethod
    def from_native_dtype(cls, dtype: TBaseDType) -> Self:
        if cls._check_native_dtype(dtype):
            return cls(endianness=get_endianness_from_numpy_dtype(dtype))
        raise DataTypeValidationError(
            f"Invalid data type: {dtype}. Expected an instance of {cls.dtype_cls}"
        )

    def to_native_dtype(self) -> np.dtypes.UInt16DType:
        byte_order = endianness_to_numpy_str(self.endianness)
        return self.dtype_cls().newbyteorder(byte_order)

    @classmethod
    def from_json_v2(cls, data: JSON, *, object_codec_id: str | None = None) -> Self:
        if cls._check_json_v2(data, object_codec_id=object_codec_id):
            # Going via numpy ensures that we get the endianness correct without
            # annoying string parsing.
            return cls.from_native_dtype(np.dtype(data))
        msg = f"Invalid JSON representation of UInt16. Got {data!r}, expected one of the strings {cls._zarr_v2_names}."
        raise DataTypeValidationError(msg)

    @classmethod
    def from_json_v3(cls, data: JSON) -> Self:
        if cls._check_json_v3(data):
            return cls()
        msg = f"Invalid JSON representation of UInt16. Got {data!r}, expected the string {cls._zarr_v3_name!r}"
        raise DataTypeValidationError(msg)

    @overload
    def to_json(self, zarr_format: Literal[2]) -> Literal[">u2", "<u2"]: ...

    @overload
    def to_json(self, zarr_format: Literal[3]) -> Literal["uint16"]: ...

    def to_json(self, zarr_format: ZarrFormat) -> Literal["uint16", ">u2", "<u2"]:
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
            return self.to_native_dtype().str
        elif zarr_format == 3:
            return self._zarr_v3_name
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    @property
    def item_size(self) -> int:
        return 2


@dataclass(frozen=True, kw_only=True)
class Int32(BaseInt[np.dtypes.Int32DType, np.int32], HasEndianness):
    dtype_cls = np.dtypes.Int32DType
    _zarr_v3_name: ClassVar[Literal["int32"]] = "int32"
    _zarr_v2_names: ClassVar[tuple[Literal[">i4"], Literal["<i4"]]] = (">i4", "<i4")

    @classmethod
    def from_native_dtype(cls: type[Self], dtype: TBaseDType) -> Self:
        if cls._check_native_dtype(dtype):
            return cls(endianness=get_endianness_from_numpy_dtype(dtype))
        raise DataTypeValidationError(
            f"Invalid data type: {dtype}. Expected an instance of {cls.dtype_cls}"
        )

    def to_native_dtype(self) -> np.dtypes.Int32DType:
        byte_order = endianness_to_numpy_str(self.endianness)
        return self.dtype_cls().newbyteorder(byte_order)

    @classmethod
    def from_json_v2(cls, data: JSON, *, object_codec_id: str | None = None) -> Self:
        if cls._check_json_v2(data, object_codec_id=object_codec_id):
            # Going via numpy ensures that we get the endianness correct without
            # annoying string parsing.
            return cls.from_native_dtype(np.dtype(data))
        msg = f"Invalid JSON representation of {cls.__name__}. Got {data!r}, expected one of the strings {cls._zarr_v2_names}."
        raise DataTypeValidationError(msg)

    @classmethod
    def from_json_v3(cls, data: JSON) -> Self:
        if cls._check_json_v3(data):
            return cls()
        msg = f"Invalid JSON representation of {cls.__name__}. Got {data!r}, expected the string {cls._zarr_v3_name!r}"
        raise DataTypeValidationError(msg)

    @overload
    def to_json(self, zarr_format: Literal[2]) -> Literal[">i4", "<i4"]: ...

    @overload
    def to_json(self, zarr_format: Literal[3]) -> Literal["int32"]: ...

    def to_json(self, zarr_format: ZarrFormat) -> Literal["int32", ">i4", "<i4"]:
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
            return self.to_native_dtype().str
        elif zarr_format == 3:
            return self._zarr_v3_name
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    @property
    def item_size(self) -> int:
        return 4


@dataclass(frozen=True, kw_only=True)
class UInt32(BaseInt[np.dtypes.UInt32DType, np.uint32], HasEndianness):
    dtype_cls = np.dtypes.UInt32DType
    _zarr_v3_name: ClassVar[Literal["uint32"]] = "uint32"
    _zarr_v2_names: ClassVar[tuple[Literal[">u4"], Literal["<u4"]]] = (">u4", "<u4")

    @classmethod
    def from_native_dtype(cls, dtype: TBaseDType) -> Self:
        if cls._check_native_dtype(dtype):
            return cls(endianness=get_endianness_from_numpy_dtype(dtype))
        raise DataTypeValidationError(
            f"Invalid data type: {dtype}. Expected an instance of {cls.dtype_cls}"
        )

    def to_native_dtype(self) -> np.dtypes.UInt32DType:
        byte_order = endianness_to_numpy_str(self.endianness)
        return self.dtype_cls().newbyteorder(byte_order)

    @classmethod
    def from_json_v2(cls, data: JSON, *, object_codec_id: str | None = None) -> Self:
        if cls._check_json_v2(data, object_codec_id=object_codec_id):
            # Going via numpy ensures that we get the endianness correct without
            # annoying string parsing.
            return cls.from_native_dtype(np.dtype(data))
        msg = f"Invalid JSON representation of {cls.__name__}. Got {data!r}, expected one of the strings {cls._zarr_v2_names}."
        raise DataTypeValidationError(msg)

    @classmethod
    def from_json_v3(cls, data: JSON) -> Self:
        if cls._check_json_v3(data):
            return cls()
        msg = f"Invalid JSON representation of {cls.__name__}. Got {data!r}, expected the string {cls._zarr_v3_name!r}"
        raise DataTypeValidationError(msg)

    @overload
    def to_json(self, zarr_format: Literal[2]) -> Literal[">u4", "<u4"]: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> Literal["uint32"]: ...
    def to_json(self, zarr_format: ZarrFormat) -> Literal["uint32", ">u4", "<u4"]:
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
            return self.to_native_dtype().str
        elif zarr_format == 3:
            return self._zarr_v3_name
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")

    @property
    def item_size(self) -> int:
        return 4


@dataclass(frozen=True, kw_only=True)
class Int64(BaseInt[np.dtypes.Int64DType, np.int64], HasEndianness):
    dtype_cls = np.dtypes.Int64DType
    _zarr_v3_name: ClassVar[Literal["int64"]] = "int64"
    _zarr_v2_names: ClassVar[tuple[Literal[">i8"], Literal["<i8"]]] = (">i8", "<i8")

    @classmethod
    def from_native_dtype(cls, dtype: TBaseDType) -> Self:
        if cls._check_native_dtype(dtype):
            return cls(endianness=get_endianness_from_numpy_dtype(dtype))
        raise DataTypeValidationError(
            f"Invalid data type: {dtype}. Expected an instance of {cls.dtype_cls}"
        )

    def to_native_dtype(self) -> np.dtypes.Int64DType:
        byte_order = endianness_to_numpy_str(self.endianness)
        return self.dtype_cls().newbyteorder(byte_order)

    @classmethod
    def from_json_v2(cls, data: JSON, *, object_codec_id: str | None = None) -> Self:
        if cls._check_json_v2(data, object_codec_id=object_codec_id):
            # Going via numpy ensures that we get the endianness correct without
            # annoying string parsing.
            return cls.from_native_dtype(np.dtype(data))
        msg = f"Invalid JSON representation of {cls.__name__}. Got {data!r}, expected one of the strings {cls._zarr_v2_names}."
        raise DataTypeValidationError(msg)

    @classmethod
    def from_json_v3(cls, data: JSON) -> Self:
        if cls._check_json_v3(data):
            return cls()
        msg = f"Invalid JSON representation of {cls.__name__}. Got {data!r}, expected the string {cls._zarr_v3_name!r}"
        raise DataTypeValidationError(msg)

    @overload
    def to_json(self, zarr_format: Literal[2]) -> Literal[">i8", "<i8"]: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> Literal["int64"]: ...
    def to_json(self, zarr_format: ZarrFormat) -> Literal["int64", ">i8", "<i8"]:
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
            return self.to_native_dtype().str
        elif zarr_format == 3:
            return self._zarr_v3_name
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")

    @property
    def item_size(self) -> int:
        return 8


@dataclass(frozen=True, kw_only=True)
class UInt64(BaseInt[np.dtypes.UInt64DType, np.uint64], HasEndianness):
    dtype_cls = np.dtypes.UInt64DType
    _zarr_v3_name: ClassVar[Literal["uint64"]] = "uint64"
    _zarr_v2_names: ClassVar[tuple[Literal[">u8"], Literal["<u8"]]] = (">u8", "<u8")

    def to_native_dtype(self) -> np.dtypes.UInt64DType:
        byte_order = endianness_to_numpy_str(self.endianness)
        return self.dtype_cls().newbyteorder(byte_order)

    @classmethod
    def from_json_v2(cls, data: JSON, *, object_codec_id: str | None = None) -> Self:
        if cls._check_json_v2(data, object_codec_id=object_codec_id):
            # Going via numpy ensures that we get the endianness correct without
            # annoying string parsing.
            return cls.from_native_dtype(np.dtype(data))
        msg = f"Invalid JSON representation of {cls.__name__}. Got {data!r}, expected one of the strings {cls._zarr_v2_names}."
        raise DataTypeValidationError(msg)

    @classmethod
    def from_json_v3(cls, data: JSON) -> Self:
        if cls._check_json_v3(data):
            return cls()
        msg = f"Invalid JSON representation of {cls.__name__}. Got {data!r}, expected the string {cls._zarr_v3_name!r}"
        raise DataTypeValidationError(msg)

    @overload
    def to_json(self, zarr_format: Literal[2]) -> Literal[">u8", "<u8"]: ...

    @overload
    def to_json(self, zarr_format: Literal[3]) -> Literal["uint64"]: ...

    def to_json(self, zarr_format: ZarrFormat) -> Literal["uint64", ">u8", "<u8"]:
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
            return self.to_native_dtype().str
        elif zarr_format == 3:
            return self._zarr_v3_name
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    @classmethod
    def from_native_dtype(cls, dtype: TBaseDType) -> Self:
        if cls._check_native_dtype(dtype):
            return cls(endianness=get_endianness_from_numpy_dtype(dtype))
        raise DataTypeValidationError(
            f"Invalid data type: {dtype}. Expected an instance of {cls.dtype_cls}"
        )

    @property
    def item_size(self) -> int:
        return 8
