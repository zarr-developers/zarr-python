from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Literal,
    Self,
    TypedDict,
    TypeGuard,
    TypeVar,
    cast,
    get_args,
    overload,
)

import numpy as np

from zarr.core.common import NamedConfig
from zarr.core.dtype.common import DataTypeValidationError, HasEndianness, HasItemSize
from zarr.core.dtype.npy.common import (
    DateTimeUnit,
    check_json_int,
    endianness_to_numpy_str,
    get_endianness_from_numpy_dtype,
)
from zarr.core.dtype.wrapper import TBaseDType, ZDType

if TYPE_CHECKING:
    from zarr.core.common import JSON, ZarrFormat

_DTypeName = Literal["datetime64", "timedelta64"]
TimeDeltaLike = str | int | bytes | np.timedelta64 | timedelta | None
DateTimeLike = str | int | bytes | np.datetime64 | datetime | None


def datetime_from_int(data: int, *, unit: DateTimeUnit, scale_factor: int) -> np.datetime64:
    """
    Convert an integer to a datetime64.

    Parameters
    ----------
    data : int
        The integer to convert.
    unit : DateTimeUnit
        The unit of the datetime64.
    scale_factor : int
        The scale factor of the datetime64.

    Returns
    -------
    np.datetime64
        The datetime64 value.
    """
    dtype_name = f"datetime64[{scale_factor}{unit}]"
    return cast("np.datetime64", np.int64(data).view(dtype_name))


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


def check_json_time(data: JSON) -> TypeGuard[Literal["NaT"] | int]:
    """
    Type guard to check if the input JSON data is the literal string "NaT"
    or an integer.
    """
    return check_json_int(data) or data == "NaT"


BaseTimeDType_co = TypeVar(
    "BaseTimeDType_co",
    bound=np.dtypes.TimeDelta64DType | np.dtypes.DateTime64DType,
    covariant=True,
)
BaseTimeScalar_co = TypeVar(
    "BaseTimeScalar_co", bound=np.timedelta64 | np.datetime64, covariant=True
)


class TimeConfig(TypedDict):
    unit: DateTimeUnit
    scale_factor: int


DateTime64JSONV3 = NamedConfig[Literal["numpy.datetime64"], TimeConfig]
TimeDelta64JSONV3 = NamedConfig[Literal["numpy.timedelta64"], TimeConfig]


@dataclass(frozen=True, kw_only=True, slots=True)
class TimeDTypeBase(ZDType[BaseTimeDType_co, BaseTimeScalar_co], HasEndianness, HasItemSize):
    _zarr_v2_names: ClassVar[tuple[str, ...]]
    # this attribute exists so that we can programmatically create a numpy dtype instance
    # because the particular numpy dtype we are wrapping does not allow direct construction via
    # cls.dtype_cls()
    _numpy_name: ClassVar[_DTypeName]
    scale_factor: int
    unit: DateTimeUnit

    def __post_init__(self) -> None:
        if self.scale_factor < 1:
            raise ValueError(f"scale_factor must be > 0, got {self.scale_factor}.")
        if self.scale_factor >= 2**31:
            raise ValueError(f"scale_factor must be < 2147483648, got {self.scale_factor}.")
        if self.unit not in get_args(DateTimeUnit):
            raise ValueError(f"unit must be one of {get_args(DateTimeUnit)}, got {self.unit!r}.")

    @classmethod
    def from_native_dtype(cls, dtype: TBaseDType) -> Self:
        if cls._check_native_dtype(dtype):
            unit, scale_factor = np.datetime_data(dtype.name)
            unit = cast("DateTimeUnit", unit)
            return cls(
                unit=unit,
                scale_factor=scale_factor,
                endianness=get_endianness_from_numpy_dtype(dtype),
            )
        raise DataTypeValidationError(
            f"Invalid data type: {dtype}. Expected an instance of {cls.dtype_cls}"
        )

    def to_native_dtype(self) -> BaseTimeDType_co:
        # Numpy does not allow creating datetime64 or timedelta64 via
        # np.dtypes.{dtype_name}()
        # so we use np.dtype with a formatted string.
        dtype_string = f"{self._numpy_name}[{self.scale_factor}{self.unit}]"
        return np.dtype(dtype_string).newbyteorder(endianness_to_numpy_str(self.endianness))  # type: ignore[return-value]

    @overload
    def to_json(self, zarr_format: Literal[2]) -> str: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> DateTime64JSONV3 | TimeDelta64JSONV3: ...

    def to_json(self, zarr_format: ZarrFormat) -> str | DateTime64JSONV3 | TimeDelta64JSONV3:
        if zarr_format == 2:
            return cast("str", self.to_native_dtype().str)
        elif zarr_format == 3:
            return cast(
                "DateTime64JSONV3 | TimeDelta64JSONV3",
                {
                    "name": self._zarr_v3_name,
                    "configuration": {"unit": self.unit, "scale_factor": self.scale_factor},
                },
            )
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

    def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> int:
        return datetimelike_to_int(data)  # type: ignore[arg-type]

    @property
    def item_size(self) -> int:
        return 8


@dataclass(frozen=True, kw_only=True, slots=True)
class TimeDelta64(TimeDTypeBase[np.dtypes.TimeDelta64DType, np.timedelta64], HasEndianness):
    """
    A wrapper for the ``TimeDelta64`` data type defined in numpy.
    Scalars of this type can be created by performing arithmetic with ``DateTime64`` scalars.
    Like ``DateTime64``, ``TimeDelta64`` is parametrized by a unit, but unlike ``DateTime64``, the
    unit for ``TimeDelta64`` is optional.
    """

    # mypy infers the type of np.dtypes.TimeDelta64DType to be
    # "Callable[[Literal['Y', 'M', 'W', 'D'] | Literal['h', 'm', 's', 'ms', 'us', 'ns', 'ps', 'fs', 'as']], Never]"
    dtype_cls = np.dtypes.TimeDelta64DType  # type: ignore[assignment]
    _zarr_v3_name: ClassVar[Literal["numpy.timedelta64"]] = "numpy.timedelta64"
    _zarr_v2_names = (">m8", "<m8")
    _numpy_name = "timedelta64"
    scale_factor: int = 1
    unit: DateTimeUnit = "generic"

    @classmethod
    def _check_json_v2(cls, data: JSON, *, object_codec_id: str | None = None) -> TypeGuard[str]:
        # match <m[ns], >m[M], etc
        # consider making this a standalone function
        if not isinstance(data, str):
            return False
        if not data.startswith(cls._zarr_v2_names):
            return False
        if len(data) == 3:
            # no unit, and
            # we already checked that this string is either <m8 or >m8
            return True
        else:
            return data[4:-1].endswith(get_args(DateTimeUnit)) and data[-1] == "]"

    @classmethod
    def _check_json_v3(cls, data: JSON) -> TypeGuard[DateTime64JSONV3]:
        return (
            isinstance(data, dict)
            and set(data.keys()) == {"name", "configuration"}
            and data["name"] == cls._zarr_v3_name
            and isinstance(data["configuration"], dict)
            and set(data["configuration"].keys()) == {"unit", "scale_factor"}
        )

    @classmethod
    def from_json_v2(cls, data: JSON, *, object_codec_id: str | None = None) -> Self:
        if cls._check_json_v2(data):
            return cls.from_native_dtype(np.dtype(data))
        msg = (
            f"Invalid JSON representation of {cls.__name__}. Got {data!r}, expected a string "
            f"representation of an instance of {cls.dtype_cls}"  # type: ignore[has-type]
        )
        raise DataTypeValidationError(msg)

    @classmethod
    def from_json_v3(cls, data: JSON) -> Self:
        if cls._check_json_v3(data):
            unit = data["configuration"]["unit"]
            scale_factor = data["configuration"]["scale_factor"]
            return cls(unit=unit, scale_factor=scale_factor)
        msg = (
            f"Invalid JSON representation of {cls.__name__}. Got {data!r}, expected a dict "
            f"with a 'name' key with the value 'numpy.timedelta64', "
            "and a 'configuration' key with a value of a dict with a 'unit' key and a "
            "'scale_factor' key"
        )
        raise DataTypeValidationError(msg)

    def _check_scalar(self, data: object) -> TypeGuard[TimeDeltaLike]:
        if data is None:
            return True
        return isinstance(data, str | int | bytes | np.timedelta64 | timedelta)

    def _cast_scalar_unchecked(self, data: TimeDeltaLike) -> np.timedelta64:
        return self.to_native_dtype().type(data, f"{self.scale_factor}{self.unit}")

    def cast_scalar(self, data: object) -> np.timedelta64:
        if self._check_scalar(data):
            return self._cast_scalar_unchecked(data)
        msg = f"Cannot convert object with type {type(data)} to a numpy timedelta64 scalar."
        raise TypeError(msg)

    def default_scalar(self) -> np.timedelta64:
        return np.timedelta64("NaT")

    def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> np.timedelta64:
        if check_json_time(data):
            return self.to_native_dtype().type(data, f"{self.scale_factor}{self.unit}")
        raise TypeError(f"Invalid type: {data}. Expected an integer.")  # pragma: no cover


@dataclass(frozen=True, kw_only=True, slots=True)
class DateTime64(TimeDTypeBase[np.dtypes.DateTime64DType, np.datetime64], HasEndianness):
    dtype_cls = np.dtypes.DateTime64DType  # type: ignore[assignment]
    _zarr_v3_name: ClassVar[Literal["numpy.datetime64"]] = "numpy.datetime64"
    _zarr_v2_names = (">M8", "<M8")
    _numpy_name = "datetime64"
    unit: DateTimeUnit = "generic"
    scale_factor: int = 1

    @classmethod
    def _check_json_v3(cls, data: JSON) -> TypeGuard[DateTime64JSONV3]:
        return (
            isinstance(data, dict)
            and set(data.keys()) == {"name", "configuration"}
            and data["name"] == cls._zarr_v3_name
            and isinstance(data["configuration"], dict)
            and set(data["configuration"].keys()) == {"unit", "scale_factor"}
        )

    @classmethod
    def from_json_v2(cls, data: JSON, *, object_codec_id: str | None = None) -> Self:
        if cls._check_json_v2(data):
            return cls.from_native_dtype(np.dtype(data))
        msg = (
            f"Invalid JSON representation of {cls.__name__}. Got {data!r}, expected a string "
            f"representation of an instance of {cls.dtype_cls}"  # type: ignore[has-type]
        )
        raise DataTypeValidationError(msg)

    @classmethod
    def from_json_v3(cls, data: JSON) -> Self:
        if cls._check_json_v3(data):
            unit = data["configuration"]["unit"]
            scale_factor = data["configuration"]["scale_factor"]
            return cls(unit=unit, scale_factor=scale_factor)
        msg = (
            f"Invalid JSON representation of {cls.__name__}. Got {data!r}, expected a dict "
            f"with a 'name' key with the value 'numpy.datetime64', "
            "and a 'configuration' key with a value of a dict with a 'unit' key and a "
            "'scale_factor' key"
        )
        raise DataTypeValidationError(msg)

    def _check_scalar(self, data: object) -> TypeGuard[DateTimeLike]:
        if data is None:
            return True
        return isinstance(data, str | int | bytes | np.datetime64 | datetime)

    def _cast_scalar_unchecked(self, data: DateTimeLike) -> np.datetime64:
        return self.to_native_dtype().type(data, f"{self.scale_factor}{self.unit}")

    def cast_scalar(self, data: object) -> np.datetime64:
        if self._check_scalar(data):
            return self._cast_scalar_unchecked(data)
        msg = f"Cannot convert object with type {type(data)} to a numpy datetime scalar."
        raise TypeError(msg)

    def default_scalar(self) -> np.datetime64:
        return np.datetime64("NaT")

    def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> np.datetime64:
        if check_json_time(data):
            return self._cast_scalar_unchecked(data)
        raise TypeError(f"Invalid type: {data}. Expected an integer.")  # pragma: no cover

    @classmethod
    def _check_json_v2(cls, data: JSON, *, object_codec_id: str | None = None) -> TypeGuard[str]:
        """
        Check that JSON input is a string representation of a NumPy datetime64 data type, like "<M8"
        of ">M8[10s]". This function can be used as a type guard to narrow the type of unknown JSON
        input.
        """
        if not isinstance(data, str):
            return False
        if not data.startswith(cls._zarr_v2_names):
            return False
        if len(data) == 3:
            # no unit, and
            # we already checked that this string is either <M8 or >M8
            return True
        else:
            return data[4:-1].endswith(get_args(DateTimeUnit)) and data[-1] == "]"
