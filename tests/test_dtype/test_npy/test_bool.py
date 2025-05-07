from __future__ import annotations

import numpy as np

from tests.test_dtype.test_wrapper import _TestZDType
from zarr.core.dtype.npy.bool import Bool


class TestBool(_TestZDType):
    test_cls = Bool
    valid_dtype = (np.dtype(np.bool_),)
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.float64),
        np.dtype(np.uint16),
    )
    valid_json_v2 = Bool._zarr_v2_names
    valid_json_v3_cases = (Bool._zarr_v3_name,)
    invalid_json_v2 = (
        "|b1",
        "bool",
        "|f8",
    )
    invalid_json_v3 = (
        "|b1",
        "|f8",
        {"name": "bool", "configuration": {"endianness": "little"}},
    )
