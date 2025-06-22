from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal, Self, TypeGuard, overload

import numpy as np

from zarr.core.dtype.common import (
    DataTypeValidationError,
    DTypeConfig_V2,
    DTypeJSON,
    HasItemSize,
    check_dtype_spec_v2,
)
from zarr.core.dtype.wrapper import TBaseDType, ZDType

if TYPE_CHECKING:
    from zarr.core.common import JSON, ZarrFormat


@dataclass(frozen=True, kw_only=True, slots=True)
class Bool(ZDType[np.dtypes.BoolDType, np.bool_], HasItemSize):
    """
    Wrapper for numpy boolean dtype.

    Attributes
    ----------
    name : str
        The name of the dtype.
    dtype_cls : ClassVar[type[np.dtypes.BoolDType]]
        The numpy dtype class.
    """

    _zarr_v3_name: ClassVar[Literal["bool"]] = "bool"
    _zarr_v2_name: ClassVar[Literal["|b1"]] = "|b1"
    dtype_cls = np.dtypes.BoolDType

    @classmethod
    def from_native_dtype(cls, dtype: TBaseDType) -> Self:
        """
        Create a Bool from a np.dtype('bool') instance.
        """
        if cls._check_native_dtype(dtype):
            return cls()
        raise DataTypeValidationError(
            f"Invalid data type: {dtype}. Expected an instance of {cls.dtype_cls}"
        )

    def to_native_dtype(self: Self) -> np.dtypes.BoolDType:
        """
        Create a NumPy boolean dtype instance from this ZDType
        """
        return self.dtype_cls()

    @classmethod
    def _check_json_v2(
        cls,
        data: DTypeJSON,
    ) -> TypeGuard[DTypeConfig_V2[Literal["|b1"], None]]:
        """
        Check that the input is a valid JSON representation of a Bool.
        """
        return (
            check_dtype_spec_v2(data)
            and data["name"] == cls._zarr_v2_name
            and data["object_codec_id"] is None
        )

    @classmethod
    def _check_json_v3(cls, data: DTypeJSON) -> TypeGuard[Literal["bool"]]:
        return data == cls._zarr_v3_name

    @classmethod
    def _from_json_v2(cls, data: DTypeJSON) -> Self:
        if cls._check_json_v2(data):
            return cls()
        msg = f"Invalid JSON representation of {cls.__name__}. Got {data!r}, expected the string {cls._zarr_v2_name!r}"
        raise DataTypeValidationError(msg)

    @classmethod
    def _from_json_v3(cls: type[Self], data: DTypeJSON) -> Self:
        if cls._check_json_v3(data):
            return cls()
        msg = f"Invalid JSON representation of {cls.__name__}. Got {data!r}, expected the string {cls._zarr_v3_name!r}"
        raise DataTypeValidationError(msg)

    @overload  # type: ignore[override]
    def to_json(self, zarr_format: Literal[2]) -> DTypeConfig_V2[Literal["|b1"], None]: ...

    @overload
    def to_json(self, zarr_format: Literal[3]) -> Literal["bool"]: ...

    def to_json(
        self, zarr_format: ZarrFormat
    ) -> DTypeConfig_V2[Literal["|b1"], None] | Literal["bool"]:
        if zarr_format == 2:
            return {"name": self._zarr_v2_name, "object_codec_id": None}
        elif zarr_format == 3:
            return self._zarr_v3_name
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def _check_scalar(self, data: object) -> bool:
        # Anything can become a bool
        return True

    def cast_scalar(self, data: object) -> np.bool_:
        if self._check_scalar(data):
            return np.bool_(data)
        msg = f"Cannot convert object with type {type(data)} to a numpy boolean."
        raise TypeError(msg)

    def default_scalar(self) -> np.bool_:
        """
        Get the default value for the boolean dtype.

        Returns
        -------
        np.bool_
            The default value.
        """
        return np.False_

    def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> bool:
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

    def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> np.bool_:
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
        if self._check_scalar(data):
            return np.bool_(data)
        raise TypeError(f"Invalid type: {data}. Expected a boolean.")  # pragma: no cover

    @property
    def item_size(self) -> int:
        return 1
