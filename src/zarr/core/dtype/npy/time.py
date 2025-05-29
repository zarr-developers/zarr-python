from __future__ import annotations

from dataclasses import dataclass
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
from zarr.core.dtype.common import HasEndianness, HasItemSize
from zarr.core.dtype.npy.common import (
    DateTimeUnit,
    EndiannessNumpy,
    check_json_int,
    endianness_from_numpy_str,
    endianness_to_numpy_str,
)
from zarr.core.dtype.wrapper import DTypeJSON_V2, DTypeJSON_V3, TBaseDType, ZDType

if TYPE_CHECKING:
    from zarr.core.common import JSON, ZarrFormat

_DTypeName = Literal["datetime64", "timedelta64"]


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


_BaseTimeDType_co = TypeVar(
    "_BaseTimeDType_co",
    bound=np.dtypes.TimeDelta64DType | np.dtypes.DateTime64DType,
    covariant=True,
)
_BaseTimeScalar = TypeVar("_BaseTimeScalar", bound=np.timedelta64 | np.datetime64)


class TimeConfig(TypedDict):
    unit: DateTimeUnit
    interval: int


DateTime64JSONV3 = NamedConfig[Literal["numpy.datetime64"], TimeConfig]
TimeDelta64JSONV3 = NamedConfig[Literal["numpy.timedelta64"], TimeConfig]


@dataclass(frozen=True, kw_only=True, slots=True)
class TimeDTypeBase(ZDType[_BaseTimeDType_co, _BaseTimeScalar], HasEndianness, HasItemSize):
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
    def _from_native_dtype_unsafe(cls, dtype: TBaseDType) -> Self:
        unit, scale_factor = np.datetime_data(dtype.name)
        unit = cast("DateTimeUnit", unit)
        byteorder = cast("EndiannessNumpy", dtype.byteorder)
        return cls(
            unit=unit, scale_factor=scale_factor, endianness=endianness_from_numpy_str(byteorder)
        )

    def to_native_dtype(self) -> _BaseTimeDType_co:
        # Numpy does not allow creating datetime64 or timedelta64 via
        # np.dtypes.{dtype_name}()
        # so we use np.dtype with a formatted string.
        dtype_string = f"{self._numpy_name}[{self.scale_factor}{self.unit}]"
        return np.dtype(dtype_string).newbyteorder(endianness_to_numpy_str(self.endianness))  # type: ignore[return-value]

    @classmethod
    def _from_json_unchecked(
        cls, data: DTypeJSON_V2 | DTypeJSON_V3, *, zarr_format: ZarrFormat
    ) -> Self:
        if zarr_format == 2:
            return cls.from_native_dtype(np.dtype(data))  # type: ignore[arg-type]
        elif zarr_format == 3:
            unit = data["configuration"]["unit"]  # type: ignore[index, call-overload]
            scale_factor = data["configuration"]["scale_factor"]  # type: ignore[index, call-overload]
            return cls(unit=unit, scale_factor=scale_factor)
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover

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

    def check_scalar(self, data: object) -> bool:
        # TODO: decide which values we should accept for datetimes.
        try:
            np.array([data], dtype=self.to_native_dtype())
            return True  # noqa: TRY300
        except ValueError:
            return False

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

    dtype_cls = np.dtypes.TimeDelta64DType  # type: ignore[assignment]
    _zarr_v3_name: ClassVar[Literal["numpy.timedelta64"]] = "numpy.timedelta64"
    _zarr_v2_names = (">m8", "<m8")
    _numpy_name = "timedelta64"
    scale_factor: int = 1
    unit: DateTimeUnit = "generic"

    def default_scalar(self) -> np.timedelta64:
        return np.timedelta64("NaT")

    def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> np.timedelta64:
        if check_json_int(data) or data == "NaT":
            return self.to_native_dtype().type(data, f"{self.scale_factor}{self.unit}")  # type: ignore[arg-type]
        raise TypeError(f"Invalid type: {data}. Expected an integer.")  # pragma: no cover

    def _cast_scalar_unchecked(self, data: object) -> np.timedelta64:
        return self.to_native_dtype().type(data)  # type: ignore[arg-type]

    @classmethod
    def check_json_v2(cls, data: JSON, *, object_codec_id: str | None = None) -> TypeGuard[str]:
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
    def check_json_v3(cls, data: JSON) -> TypeGuard[DateTime64JSONV3]:
        return (
            isinstance(data, dict)
            and set(data.keys()) == {"name", "configuration"}
            and data["name"] == cls._zarr_v3_name
            and isinstance(data["configuration"], dict)
            and set(data["configuration"].keys()) == {"unit", "scale_factor"}
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class DateTime64(TimeDTypeBase[np.dtypes.DateTime64DType, np.datetime64], HasEndianness):
    dtype_cls = np.dtypes.DateTime64DType  # type: ignore[assignment]
    _zarr_v3_name: ClassVar[Literal["numpy.datetime64"]] = "numpy.datetime64"
    _zarr_v2_names = (">M8", "<M8")
    _numpy_name = "datetime64"
    unit: DateTimeUnit = "generic"
    scale_factor: int = 1

    def default_scalar(self) -> np.datetime64:
        return np.datetime64("NaT")

    def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> np.datetime64:
        if check_json_int(data) or data == "NaT":
            return self.to_native_dtype().type(data, f"{self.scale_factor}{self.unit}")  # type: ignore[arg-type]
        raise TypeError(f"Invalid type: {data}. Expected an integer.")  # pragma: no cover

    def _cast_scalar_unchecked(self, data: object) -> np.datetime64:
        return self.to_native_dtype().type(data, f"{self.scale_factor}{self.unit}")  # type: ignore[no-any-return, call-overload]

    @classmethod
    def check_json_v2(cls, data: JSON, *, object_codec_id: str | None = None) -> TypeGuard[str]:
        # match <M[ns], >M[M], etc
        # consider making this a standalone function
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

    @classmethod
    def check_json_v3(cls, data: JSON) -> TypeGuard[DateTime64JSONV3]:
        return (
            isinstance(data, dict)
            and set(data.keys()) == {"name", "configuration"}
            and data["name"] == cls._zarr_v3_name
            and isinstance(data["configuration"], dict)
            and set(data["configuration"].keys()) == {"unit", "scale_factor"}
        )
