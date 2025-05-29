from dataclasses import dataclass
from typing import ClassVar, Literal, Self, TypeGuard, overload

import numpy as np

from zarr.core.common import JSON, ZarrFormat
from zarr.core.dtype.common import HasItemSize
from zarr.core.dtype.npy.common import check_json_bool
from zarr.core.dtype.wrapper import DTypeJSON_V2, DTypeJSON_V3, TBaseDType, ZDType


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
    _zarr_v2_names: ClassVar[tuple[str, ...]] = ("|b1",)
    dtype_cls = np.dtypes.BoolDType

    @classmethod
    def _from_native_dtype_unsafe(cls, dtype: TBaseDType) -> Self:
        return cls()

    def to_native_dtype(self: Self) -> np.dtypes.BoolDType:
        return self.dtype_cls()

    @classmethod
    def check_json_v2(
        cls, data: JSON, *, object_codec_id: str | None = None
    ) -> TypeGuard[Literal["|b1"]]:
        """
        Check that the input is a valid JSON representation of a bool.
        """
        return data in cls._zarr_v2_names

    @classmethod
    def check_json_v3(cls, data: JSON) -> TypeGuard[Literal["bool"]]:
        return data == cls._zarr_v3_name

    @overload
    def to_json(self, zarr_format: Literal[2]) -> Literal["|b1"]: ...

    @overload
    def to_json(self, zarr_format: Literal[3]) -> Literal["bool"]: ...

    def to_json(self, zarr_format: ZarrFormat) -> Literal["|b1", "bool"]:
        if zarr_format == 2:
            return self.to_native_dtype().str
        elif zarr_format == 3:
            return self._zarr_v3_name
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    @classmethod
    def _from_json_unchecked(
        cls, data: DTypeJSON_V2 | DTypeJSON_V3, *, zarr_format: ZarrFormat
    ) -> Self:
        return cls()

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
        if check_json_bool(data):
            return self._cast_scalar_unchecked(data)
        raise TypeError(f"Invalid type: {data}. Expected a boolean.")  # pragma: no cover

    def check_scalar(self, data: object) -> bool:
        # Anything can become a bool
        return True

    def _cast_scalar_unchecked(self, data: object) -> np.bool_:
        return np.bool_(data)

    @property
    def item_size(self) -> int:
        return 1
