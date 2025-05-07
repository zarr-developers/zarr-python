from __future__ import annotations

import numpy as np

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
    valid_json_v3_cases = (
        {"name": "datetime64", "configuration": {"unit": "ns", "scale_factor": 10}},
        {"name": "datetime64", "configuration": {"unit": "us", "scale_factor": 1}},
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
    valid_json_v3_cases = (
        {"name": "timedelta64", "configuration": {"unit": "ns"}},
        {"name": "timedelta64", "configuration": {"unit": "us"}},
    )
    invalid_json_v2 = (
        "timedelta64",
        "|f8",
        "datetime64[ns]",
    )
    invalid_json_v3 = (
        {"name": "timedelta64", "configuration": {"unit": "invalid"}},
        {"name": "timedelta64", "configuration": {"unit": 123}},
    )
