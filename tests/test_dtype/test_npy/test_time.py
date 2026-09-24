from __future__ import annotations

import re
from typing import TYPE_CHECKING, get_args

import numpy as np
import pytest

import zarr
from tests.test_dtype.test_wrapper import BaseTestZDType
from zarr.core.dtype.npy.common import DateTimeUnit
from zarr.core.dtype.npy.time import DateTime64, TimeDelta64, datetime_from_int

if TYPE_CHECKING:
    from zarr.core.common import ZarrFormat
    from zarr.core.dtype.common import DTypeSpec_V2, DTypeSpec_V3

# Every unit at the smallest, a small, and the largest scale NumPy accepts, in both byte
# orders. "μs" is left out of the JSON cases because it is written back as "us" (see
# test_time_microsecond_alias_normalized).
_UNITS = get_args(DateTimeUnit)
_SCALES = (1, 2, 2**31 - 1)
_BYTEORDERS = ("<", ">")


def _native_dtypes(kind: str) -> tuple[np.dtype[np.generic], ...]:
    return tuple(
        np.dtype(f"{byteorder}{kind}[{scale}{unit}]")
        for unit in _UNITS
        for scale in _SCALES
        for byteorder in _BYTEORDERS
    )


def _v2_names(kind: str) -> tuple[DTypeSpec_V2, ...]:
    # NumPy's spelling, except that a generic scale is written explicitly.
    def suffix(unit: str, scale: int) -> str:
        if unit == "generic" and scale == 1:
            return ""
        return f"[{unit}]" if scale == 1 else f"[{scale}{unit}]"

    return tuple(
        {"name": f"{byteorder}{kind}{suffix(unit, scale)}", "object_codec_id": None}
        for unit in _UNITS
        if unit != "μs"
        for scale in _SCALES
        for byteorder in _BYTEORDERS
    )


def _v3_configurations(name: str) -> tuple[DTypeSpec_V3, ...]:
    return tuple(
        {"name": name, "configuration": {"unit": unit, "scale_factor": scale}}
        for unit in _UNITS
        if unit != "μs"
        for scale in _SCALES
    )


class _TestTimeBase(BaseTestZDType):
    def json_scalar_equals(self, scalar1: object, scalar2: object) -> bool:
        # This method gets overridden here to support the equivalency between NaT and
        # -9223372036854775808 fill values
        nat_scalars = (-9223372036854775808, "NaT")
        if scalar1 in nat_scalars and scalar2 in nat_scalars:
            return True
        return scalar1 == scalar2

    def scalar_equals(self, scalar1: object, scalar2: object) -> bool:
        if np.isnan(scalar1) and np.isnan(scalar2):  # type: ignore[call-overload]
            return True
        return super().scalar_equals(scalar1, scalar2)


class TestDateTime64(_TestTimeBase):
    test_cls = DateTime64
    valid_dtype = (np.dtype("datetime64"), *_native_dtypes("M8"))
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.float64),
        np.dtype("timedelta64[ns]"),
    )
    valid_json_v2 = _v2_names("M8")
    valid_json_v3 = _v3_configurations("numpy.datetime64")
    invalid_json_v2 = (
        {"name": "datetime64", "object_codec_id": None},
        {"name": "|f8", "object_codec_id": None},
        {"name": "timedelta64[ns]", "object_codec_id": None},
        {"name": "<m8[ns]", "object_codec_id": None},
        {"name": "<M8[ns]", "object_codec_id": "vlen-utf8"},
    )
    invalid_json_v3 = (
        {"name": "datetime64", "configuration": {"unit": "invalid"}},
        {"name": "datetime64", "configuration": {"unit": 123}},
    )

    scalar_v2_params = (
        (DateTime64(unit="ns", scale_factor=1), 1),
        (DateTime64(unit="ns", scale_factor=1), "NaT"),
    )
    scalar_v3_params = (
        (DateTime64(unit="ns", scale_factor=1), 1),
        (DateTime64(unit="ns", scale_factor=1), "NaT"),
    )

    cast_value_params = (
        (DateTime64(unit="Y", scale_factor=1), "1", np.datetime64("1", "Y")),
        (DateTime64(unit="s", scale_factor=1), "2005-02-25", np.datetime64("2005-02-25", "s")),
        (DateTime64(unit="ns", scale_factor=1), "NaT", np.datetime64("NaT", "ns")),
    )
    invalid_scalar_params = (
        (DateTime64(unit="Y", scale_factor=1), 1.3),
        (DateTime64(unit="Y", scale_factor=1), [1.3]),
    )
    item_size_params = (DateTime64(unit="ns", scale_factor=1),)


class TestTimeDelta64(_TestTimeBase):
    test_cls = TimeDelta64
    valid_dtype = _native_dtypes("m8")
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.float64),
        np.dtype("datetime64[ns]"),
    )

    valid_json_v2 = _v2_names("m8")
    valid_json_v3 = _v3_configurations("numpy.timedelta64")
    invalid_json_v2 = (
        {"name": "timedelta64", "object_codec_id": None},
        {"name": "|f8", "object_codec_id": None},
        {"name": "datetime64[ns]", "object_codec_id": None},
        {"name": "<M8[ns]", "object_codec_id": None},
        {"name": "<m8[ns]", "object_codec_id": "vlen-utf8"},
    )
    invalid_json_v3 = (
        {"name": "timedelta64", "configuration": {"unit": 1, "scale_factor": 10}},
        {"name": "timedelta64", "configuration": {"unit": 123}},
    )

    scalar_v2_params = (
        (TimeDelta64(unit="ns", scale_factor=1), 1),
        (TimeDelta64(unit="ns", scale_factor=1), "NaT"),
    )
    scalar_v3_params = (
        (TimeDelta64(unit="ns", scale_factor=1), 1),
        (TimeDelta64(unit="ns", scale_factor=1), "NaT"),
    )

    cast_value_params = (
        (TimeDelta64(unit="ns", scale_factor=1), "1", np.timedelta64(1, "ns")),
        (TimeDelta64(unit="ns", scale_factor=1), "NaT", np.timedelta64("NaT", "ns")),
    )
    invalid_scalar_params = (
        (TimeDelta64(unit="Y", scale_factor=1), 1.3),
        (TimeDelta64(unit="Y", scale_factor=1), [1.3]),
    )
    item_size_params = (TimeDelta64(unit="ns", scale_factor=1),)


def test_time_invalid_unit() -> None:
    """
    Test that an invalid unit raises a ValueError.
    """
    unit = "invalid"
    msg = f"unit must be one of ('Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'μs', 'ns', 'ps', 'fs', 'as', 'generic'), got {unit!r}."
    with pytest.raises(ValueError, match=re.escape(msg)):
        DateTime64(unit=unit)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=re.escape(msg)):
        TimeDelta64(unit=unit)  # type: ignore[arg-type]


def test_time_scale_factor_too_low() -> None:
    """
    Test that an invalid unit raises a ValueError.
    """
    scale_factor = 0
    msg = f"scale_factor must be > 0, got {scale_factor}."
    with pytest.raises(ValueError, match=msg):
        DateTime64(scale_factor=scale_factor)
    with pytest.raises(ValueError, match=msg):
        TimeDelta64(scale_factor=scale_factor)


@pytest.mark.parametrize("cls", [DateTime64, TimeDelta64])
@pytest.mark.parametrize("unit", _UNITS)
@pytest.mark.parametrize("scale_factor", _SCALES)
@pytest.mark.filterwarnings(
    "ignore:The 'generic' unit for NumPy timedelta is deprecated:DeprecationWarning"
)
def test_nat_scalars_carry_scale(
    cls: type[DateTime64 | TimeDelta64], unit: DateTimeUnit, scale_factor: int
) -> None:
    """
    The default scalar and a cast NaT are NaT in this dtype's unit and scale. NumPy < 2.2
    turns a timedelta NaT into a count when casting it between scales.
    """
    zdtype = cls(unit=unit, scale_factor=scale_factor)
    nat = np.timedelta64("NaT", "ns") if cls is TimeDelta64 else np.datetime64("NaT", "ns")
    expected_unit = "us" if unit == "μs" else unit
    for scalar in (zdtype.default_scalar(), zdtype.cast_scalar(nat)):
        assert np.isnat(scalar)
        assert np.datetime_data(scalar.dtype) == (expected_unit, scale_factor)


def test_time_scale_factor_too_high() -> None:
    """
    Test that an invalid unit raises a ValueError.
    """
    scale_factor = 2**31
    msg = f"scale_factor must be < 2147483648, got {scale_factor}."
    with pytest.raises(ValueError, match=msg):
        DateTime64(scale_factor=scale_factor)
    with pytest.raises(ValueError, match=msg):
        TimeDelta64(scale_factor=scale_factor)


@pytest.mark.parametrize("cls", [DateTime64, TimeDelta64])
def test_time_microsecond_alias_normalized(cls: type[DateTime64 | TimeDelta64]) -> None:
    """
    Test that the 'μs' unit is stored as NumPy's 'us' spelling.

    The two spellings are equivalent, but NumPy only ever reports 'us', so an instance
    that kept 'μs' would compare unequal to itself after a trip through NumPy. Stored
    metadata may still spell the unit 'μs' and reads back as the normalized instance.
    """
    zdtype = cls(unit="μs", scale_factor=3)
    assert zdtype.unit == "us"
    assert zdtype == cls(unit="us", scale_factor=3)
    assert cls.from_native_dtype(zdtype.to_native_dtype()) == zdtype
    json_v3 = {"name": cls._zarr_v3_name, "configuration": {"unit": "μs", "scale_factor": 3}}
    assert cls.from_json(json_v3, zarr_format=3) == zdtype
    assert zdtype.to_json(zarr_format=3)["configuration"]["unit"] == "us"


@pytest.mark.parametrize("unit", get_args(DateTimeUnit))
@pytest.mark.parametrize("scale_factor", [1, 10])
@pytest.mark.parametrize("value", [0, 1, 10])
def test_datetime_from_int(unit: DateTimeUnit, scale_factor: int, value: int) -> None:
    """
    Test datetime_from_int.
    """
    expected = np.int64(value).view(f"datetime64[{scale_factor}{unit}]")
    assert datetime_from_int(value, unit=unit, scale_factor=scale_factor) == expected


@pytest.mark.parametrize("unit", ["generic", "us"])
@pytest.mark.parametrize("kind", ["M8", "m8"])
@pytest.mark.parametrize("byteorder", ["<", ">"])
@pytest.mark.parametrize("scale_factor", [1, 2, 2**31 - 1])
@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.filterwarnings("ignore::zarr.errors.UnstableSpecificationWarning")
@pytest.mark.filterwarnings(
    "ignore:The 'generic' unit for NumPy timedelta is deprecated:DeprecationWarning"
)
def test_generic_time_array_roundtrip(
    unit: str,
    kind: str,
    byteorder: str,
    scale_factor: int,
    zarr_format: ZarrFormat,
    structured: bool,
) -> None:
    """Persist counts and generic scale through metadata, chunk IO, and output allocation."""
    leaf = np.dtype(f"{byteorder}{kind}[{scale_factor}{unit}]")
    dtype = np.dtype([("time", leaf)]) if structured else leaf
    counts = np.array([0, 1, -2, 100], dtype=f"{byteorder}i8")
    data = counts.view(dtype)
    array = zarr.create_array(
        store={}, data=data, chunks=2, zarr_format=zarr_format, compressors=None
    )
    array.resize((6,))
    reopened = zarr.open_array(array.store, mode="r")
    result = np.asarray(reopened[:])
    values = result["time"] if structured else result
    assert np.datetime_data(values.dtype) == (unit, scale_factor)
    np.testing.assert_array_equal(values[:4].view(values.dtype.byteorder + "i8"), counts)
    expected_fill = 0 if structured else -(2**63)
    np.testing.assert_array_equal(
        values[4:].view(values.dtype.byteorder + "i8"), [expected_fill, expected_fill]
    )
