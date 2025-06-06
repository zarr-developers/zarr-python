from dataclasses import dataclass
from typing import ClassVar, Self, TypeGuard, cast

import numpy as np

from zarr.core.common import JSON, ZarrFormat
from zarr.core.dtype.common import HasEndianness, HasItemSize
from zarr.core.dtype.npy.common import (
    EndiannessNumpy,
    FloatLike,
    TFloatDType_co,
    TFloatScalar_co,
    check_json_float_v2,
    check_json_float_v3,
    endianness_from_numpy_str,
    endianness_to_numpy_str,
    float_from_json_v2,
    float_from_json_v3,
    float_to_json_v2,
    float_to_json_v3,
)
from zarr.core.dtype.wrapper import DTypeJSON_V2, DTypeJSON_V3, TBaseDType, ZDType


@dataclass(frozen=True)
class BaseFloat(ZDType[TFloatDType_co, TFloatScalar_co], HasEndianness, HasItemSize):
    # This attribute holds the possible zarr v2 JSON names for the data type
    _zarr_v2_names: ClassVar[tuple[str, ...]]

    @classmethod
    def _from_native_dtype_unchecked(cls, dtype: TBaseDType) -> Self:
        byte_order = cast("EndiannessNumpy", dtype.byteorder)
        return cls(endianness=endianness_from_numpy_str(byte_order))

    def to_native_dtype(self) -> TFloatDType_co:
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
            return self.to_native_dtype().str
        elif zarr_format == 3:
            return self._zarr_v3_name
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    @classmethod
    def _from_json_unchecked(
        cls, data: DTypeJSON_V2 | DTypeJSON_V3, *, zarr_format: ZarrFormat
    ) -> Self:
        if zarr_format == 2:
            return cls.from_native_dtype(np.dtype(data))  # type: ignore[arg-type]
        elif zarr_format == 3:
            return cls()
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    @classmethod
    def _check_json_v2(cls, data: JSON, *, object_codec_id: str | None = None) -> TypeGuard[str]:
        """
        Check that the input is a valid JSON representation of this data type.
        """
        return data in cls._zarr_v2_names

    @classmethod
    def _check_json_v3(cls, data: JSON) -> TypeGuard[str]:
        return data == cls._zarr_v3_name

    def _check_scalar(self, data: object) -> TypeGuard[FloatLike]:
        return isinstance(data, FloatLike)

    def _cast_scalar_unchecked(self, data: object) -> TFloatScalar_co:
        return self.to_native_dtype().type(data)  # type: ignore[return-value, arg-type]

    def default_scalar(self) -> TFloatScalar_co:
        """
        Get the default value, which is 0 cast to this dtype

        Returns
        -------
        Int scalar
            The default value.
        """
        return self._cast_scalar_unchecked(0)

    def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> TFloatScalar_co:
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
        if zarr_format == 2:
            if check_json_float_v2(data):
                return self._cast_scalar_unchecked(float_from_json_v2(data))
            else:
                raise TypeError(
                    f"Invalid type: {data}. Expected a float or a special string encoding of a float."
                )
        elif zarr_format == 3:
            if check_json_float_v3(data):
                return self._cast_scalar_unchecked(float_from_json_v3(data))
            else:
                raise TypeError(
                    f"Invalid type: {data}. Expected a float or a special string encoding of a float."
                )
        else:
            raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> float | str:
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
        if zarr_format == 2:
            return float_to_json_v2(self._cast_scalar_unchecked(data))
        elif zarr_format == 3:
            return float_to_json_v3(self._cast_scalar_unchecked(data))
        else:
            raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover


@dataclass(frozen=True, kw_only=True)
class Float16(BaseFloat[np.dtypes.Float16DType, np.float16]):
    dtype_cls = np.dtypes.Float16DType
    _zarr_v3_name = "float16"
    _zarr_v2_names: ClassVar[tuple[str, ...]] = (">f2", "<f2")

    @property
    def item_size(self) -> int:
        return 2


@dataclass(frozen=True, kw_only=True)
class Float32(BaseFloat[np.dtypes.Float32DType, np.float32]):
    dtype_cls = np.dtypes.Float32DType
    _zarr_v3_name = "float32"
    _zarr_v2_names: ClassVar[tuple[str, ...]] = (">f4", "<f4")

    @property
    def item_size(self) -> int:
        return 4


@dataclass(frozen=True, kw_only=True)
class Float64(BaseFloat[np.dtypes.Float64DType, np.float64]):
    dtype_cls = np.dtypes.Float64DType
    _zarr_v3_name = "float64"
    _zarr_v2_names: ClassVar[tuple[str, ...]] = (">f8", "<f8")

    @property
    def item_size(self) -> int:
        return 8
