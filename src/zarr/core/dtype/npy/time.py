from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self, TypeGuard, cast, get_args

import numpy as np

from zarr.core.dtype.common import DataTypeValidationError, HasEndianness
from zarr.core.dtype.npy.common import (
    DateUnit,
    EndiannessNumpy,
    TimeUnit,
    check_json_int,
    endianness_from_numpy_str,
    endianness_to_numpy_str,
)
from zarr.core.dtype.wrapper import ZDType, _BaseDType

if TYPE_CHECKING:
    from zarr.core.common import JSON, ZarrFormat


def datetime_from_json(data: int, unit: DateUnit | TimeUnit) -> np.datetime64:
    """
    Convert a JSON integer to a datetime64.

    Parameters
    ----------
    data : int
        The JSON integer to convert.
    unit : DateUnit or TimeUnit
        The unit of the datetime64.

    Returns
    -------
    np.datetime64
        The datetime64 value.
    """
    return cast("np.datetime64", np.int64(data).view(f"datetime64[{unit}]"))


def datetime_to_json(data: np.datetime64) -> int:
    """
    Convert a datetime64 to a JSON integer.

    Parameters
    ----------
    data : np.datetime64
        The datetime64 value to convert.

    Returns
    -------
    int
        The JSON representation of the datetime64.
    """
    return data.view(np.int64).item()


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
        # TODO: decide which values we should accept for datetimes.
        try:
            np.array([data], dtype=self.to_dtype())
            return True  # noqa: TRY300
        except ValueError:
            return False

    def _cast_value_unsafe(self, value: object) -> np.datetime64:
        return self.to_dtype().type(value)  # type: ignore[no-any-return, call-overload]
