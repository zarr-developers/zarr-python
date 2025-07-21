from __future__ import annotations

import re
from typing import get_args

import numpy as np
import pytest

from tests.test_dtype.test_wrapper import BaseTestZDType
from zarr.core.dtype.npy.common import DateTimeUnit
from zarr.core.dtype.npy.time import DateTime64, TimeDelta64, datetime_from_int


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
    valid_dtype = (np.dtype("datetime64[10ns]"), np.dtype("datetime64[us]"), np.dtype("datetime64"))
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.float64),
        np.dtype("timedelta64[ns]"),
    )
    valid_json_v2 = (
        {"name": ">M8", "object_codec_id": None},
        {"name": ">M8[s]", "object_codec_id": None},
        {"name": "<M8[10s]", "object_codec_id": None},
        {"name": "<M8[10us]", "object_codec_id": None},
    )
    valid_json_v3 = (
        {"name": "numpy.datetime64", "configuration": {"unit": "ns", "scale_factor": 10}},
        {"name": "numpy.datetime64", "configuration": {"unit": "us", "scale_factor": 1}},
    )
    invalid_json_v2 = (
        "datetime64",
        "|f8",
        "timedelta64[ns]",
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
        (DateTime64(unit="ns", scale_factor=1), "NaT", np.datetime64("NaT")),
    )
    invalid_scalar_params = (
        (DateTime64(unit="Y", scale_factor=1), 1.3),
        (DateTime64(unit="Y", scale_factor=1), [1.3]),
    )
    item_size_params = (DateTime64(unit="ns", scale_factor=1),)


class TestTimeDelta64(_TestTimeBase):
    test_cls = TimeDelta64
    valid_dtype = (np.dtype("timedelta64[ns]"), np.dtype("timedelta64[us]"))
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.float64),
        np.dtype("datetime64[ns]"),
    )

    valid_json_v2 = (
        {"name": ">m8", "object_codec_id": None},
        {"name": ">m8[s]", "object_codec_id": None},
        {"name": "<m8[10s]", "object_codec_id": None},
        {"name": "<m8[10us]", "object_codec_id": None},
    )
    valid_json_v3 = (
        {"name": "numpy.timedelta64", "configuration": {"unit": "ns", "scale_factor": 10}},
        {"name": "numpy.timedelta64", "configuration": {"unit": "us", "scale_factor": 1}},
    )
    invalid_json_v2 = (
        "timedelta64",
        "|f8",
        "datetime64[ns]",
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
        (TimeDelta64(unit="ns", scale_factor=1), "NaT", np.timedelta64("NaT")),
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


@pytest.mark.parametrize("unit", get_args(DateTimeUnit))
@pytest.mark.parametrize("scale_factor", [1, 10])
@pytest.mark.parametrize("value", [0, 1, 10])
def test_datetime_from_int(unit: DateTimeUnit, scale_factor: int, value: int) -> None:
    """
    Test datetime_from_int.
    """
    expected = np.int64(value).view(f"datetime64[{scale_factor}{unit}]")
    assert datetime_from_int(value, unit=unit, scale_factor=scale_factor) == expected
