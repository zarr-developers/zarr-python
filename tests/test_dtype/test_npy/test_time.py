from __future__ import annotations

import re

import numpy as np
import pytest

from tests.test_dtype.test_wrapper import _TestZDType
from zarr.core.dtype.npy.time import DateTime64, TimeDelta64


class TestDateTime64(_TestZDType):
    test_cls = DateTime64
    valid_dtype = (np.dtype("datetime64[10ns]"), np.dtype("datetime64[us]"), np.dtype("datetime64"))
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.float64),
        np.dtype("timedelta64[ns]"),
    )
    valid_json_v2 = (">M8", ">M8[s]", "<M8[10s]", "<M8[10us]")
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


class TestTimeDelta64(_TestZDType):
    test_cls = TimeDelta64
    valid_dtype = (np.dtype("timedelta64[ns]"), np.dtype("timedelta64[us]"))
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.float64),
        np.dtype("datetime64[ns]"),
    )

    valid_json_v2 = TimeDelta64._zarr_v2_names
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
