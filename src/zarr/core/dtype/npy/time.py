from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Self, TypeGuard, cast, get_args

import numpy as np

from zarr.core.dtype.common import DataTypeValidationError, HasEndianness
from zarr.core.dtype.npy.common import (
    DateTimeUnit,
    EndiannessNumpy,
    check_json_int,
    endianness_from_numpy_str,
    endianness_to_numpy_str,
)
from zarr.core.dtype.wrapper import ZDType, _BaseDType

if TYPE_CHECKING:
    from zarr.core.common import JSON, ZarrFormat

_DTypeName = Literal["datetime64", "timedelta64"]


def parse_timedtype_name(name: str) -> tuple[_DTypeName, DateTimeUnit | None]:
    """
    Parse a string like "datetime64[s]" into a tuple like ("datetime64", "s").
    """
    dtype_name: _DTypeName
    unit: DateTimeUnit | None

    if name.startswith("datetime64"):
        dtype_name = "datetime64"
    elif name.startswith("timedelta64"):
        dtype_name = "timedelta64"
    else:
        msg = (
            f"Invalid dtype name. Expected a string starting with on of {get_args(_DTypeName)}. "
            f"Got {name!r} instead."
        )
        raise ValueError(msg)

    regex = re.search(r"\[(.*?)\]", name)

    if regex is None:
        if dtype_name == "timedelta64":
            unit = None
        else:
            msg = (
                "The name of a datetime64 dtype must end with a specification of a unit. "
                'For example, "datetime64[s].'
                f"Got {name!r}, which does not follow this pattern."
            )
            raise ValueError(msg)
    else:
        maybe_unit = regex.group(1)
        unit_expected = get_args(DateTimeUnit)
        if maybe_unit not in unit_expected:
            msg = f"Invalid unit. Expected one of {unit_expected}. Got {maybe_unit} instead."
            raise ValueError(msg)
        unit = maybe_unit  # type: ignore[assignment]

    return dtype_name, unit


def datetime_from_int(data: int, unit: DateTimeUnit) -> np.datetime64:
    """
    Convert an integer to a datetime64.

    Parameters
    ----------
    data : int
        The integer to convert.
    unit : DateUnit or TimeUnit
        The unit of the datetime64.

    Returns
    -------
    np.datetime64
        The datetime64 value.
    """
    return cast("np.datetime64", np.int64(data).view(f"datetime64[{unit}]"))


def datetimelike_to_int(data: np.datetime64 | np.timedelta64) -> int:
    """
    Convert a datetime64 or a timedelta64 to an integer.

    Parameters
    ----------
    data : np.datetime64 | np.timedelta64
        The value to convert.

    Returns
    -------
    int
        An integer representation of the scalar.
    """
    return data.view(np.int64).item()


def timedelta_from_int(data: int, unit: DateTimeUnit | None) -> np.timedelta64:
    """
    Convert an integer to a timedelta64.

    Parameters
    ----------
    data : int
        The integer to convert.
    unit : DateUnit or TimeUnit
        The unit of the timedelta64.

    Returns
    -------
    np.timedelta64
        The timedelta64 value.
    """
    if unit is not None:
        dtype_name = f"timedelta64[{unit}]"
    else:
        dtype_name = "timedelta64"
    return cast("np.timedelta64", np.int64(data).view(dtype_name))


@dataclass(frozen=True, kw_only=True, slots=True)
class TimeDelta64(ZDType[np.dtypes.TimeDelta64DType, np.timedelta64], HasEndianness):
    """
    A wrapper for the ``TimeDelta64`` data type defined in numpy.
    Scalars of this type can be created by performing arithmetic with ``DateTime64`` scalars.
    Like ``DateTime64``, ``TimeDelta64`` is parametrized by a unit, but unlike ``DateTime64``, the
    unit for ``TimeDelta64`` is optional.
    """

    dtype_cls = np.dtypes.TimeDelta64DType  # type: ignore[assignment]
    _zarr_v3_name = "numpy.timedelta64"
    unit: DateTimeUnit | None = None

    @classmethod
    def _from_dtype_unsafe(cls, dtype: _BaseDType) -> Self:
        _, unit = parse_timedtype_name(dtype.name)
        byteorder = cast("EndiannessNumpy", dtype.byteorder)
        return cls(unit=unit, endianness=endianness_from_numpy_str(byteorder))

    def to_dtype(self) -> np.dtypes.TimeDelta64DType:
        # Numpy does not allow creating timedelta64 via
        # np.dtypes.TimeDelta64DType()
        if self.unit is not None:
            dtype_string = f"timedelta64[{self.unit}]"
        else:
            dtype_string = "timedelta64"
        dt = np.dtype(dtype_string).newbyteorder(endianness_to_numpy_str(self.endianness))
        return cast("np.dtypes.TimeDelta64DType", dt)

    def default_value(self) -> np.timedelta64:
        return np.timedelta64("NaT")

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

    def from_json_value(self, data: JSON, *, zarr_format: ZarrFormat) -> np.timedelta64:
        if check_json_int(data):
            return timedelta_from_int(data, self.unit)
        raise TypeError(f"Invalid type: {data}. Expected an integer.")

    def to_json_value(self, data: object, *, zarr_format: ZarrFormat) -> int:
        return datetimelike_to_int(data)  # type: ignore[arg-type]

    def check_value(self, data: object) -> bool:
        # TODO: decide which values we should accept for datetimes.
        try:
            np.array([data], dtype=self.to_dtype())
            return True  # noqa: TRY300
        except ValueError:
            return False

    def _cast_value_unsafe(self, value: object) -> np.timedelta64:
        return self.to_dtype().type(value)  # type: ignore[arg-type]

    @classmethod
    def check_json(cls, data: JSON, zarr_format: ZarrFormat) -> TypeGuard[JSON]:
        if zarr_format == 2:
            # match <m[ns], >m[M], etc
            # consider making this a standalone function
            if not (isinstance(data, str) and data[0] in (">", "<") and data[1:3] == "m8"):
                return False
            if len(data) == 3:
                # no unit, and
                # we already checked that this string is either <m8 or >m8
                return True
            if len(data) in (6, 7):
                return data[4:-1] in get_args(DateTimeUnit) and data[-1] == "]"
            else:
                return False
        elif zarr_format == 3:
            return (
                isinstance(data, dict)
                and set(data.keys()) == {"name", "configuration"}
                and data["name"] == cls._zarr_v3_name
                and set(data.keys()) == {"name", "configuration"}
                and isinstance(data["configuration"], dict)
                and set(data["configuration"].keys()) in ({"unit"}, {})
                and data["configuration"].get("unit", None) in (*get_args(DateTimeUnit), None)
            )
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover


@dataclass(frozen=True, kw_only=True, slots=True)
class DateTime64(ZDType[np.dtypes.DateTime64DType, np.datetime64], HasEndianness):
    dtype_cls = np.dtypes.DateTime64DType  # type: ignore[assignment]
    _zarr_v3_name = "numpy.datetime64"
    unit: DateTimeUnit

    @classmethod
    def _from_dtype_unsafe(cls, dtype: _BaseDType) -> Self:
        unit: DateTimeUnit = dtype.name[dtype.name.rfind("[") + 1 : dtype.name.rfind("]")]  # type: ignore[assignment]
        if unit not in get_args(DateTimeUnit):
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
            return datetime_from_int(data, self.unit)
        raise TypeError(f"Invalid type: {data}. Expected an integer.")

    def to_json_value(self, data: object, *, zarr_format: ZarrFormat) -> int:
        return datetimelike_to_int(data)  # type: ignore[arg-type]

    def check_value(self, data: object) -> bool:
        # TODO: decide which values we should accept for datetimes.
        try:
            np.array([data], dtype=self.to_dtype())
            return True  # noqa: TRY300
        except ValueError:
            return False

    def _cast_value_unsafe(self, value: object) -> np.datetime64:
        return self.to_dtype().type(value)  # type: ignore[no-any-return, call-overload]

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
                and data[4:-1] in get_args(DateTimeUnit)
                and data[-1] == "]"
            )
        elif zarr_format == 3:
            return (
                isinstance(data, dict)
                and set(data.keys()) == {"name", "configuration"}
                and data["name"] == cls._zarr_v3_name
                and set(data["configuration"].keys()) == {"unit"}
                and data["configuration"]["unit"] in get_args(DateTimeUnit)
            )
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover
