from dataclasses import dataclass
from typing import ClassVar, Literal, Self, TypeGuard

import numpy as np

from zarr.core.common import JSON, ZarrFormat
from zarr.core.dtype.npy.common import check_json_bool
from zarr.core.dtype.wrapper import TBaseDType, ZDType


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
    def _from_dtype_unsafe(cls, dtype: TBaseDType) -> Self:
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
