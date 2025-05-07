from __future__ import annotations

import numpy as np

from tests.test_dtype.test_wrapper import _TestZDType
from zarr.core.dtype.npy.int import Int8


class TestInt8(_TestZDType):
    test_cls = Int8
    valid_dtype = (np.dtype(np.int8),)
    invalid_dtype = (
        np.dtype(np.int16),
        np.dtype(np.uint16),
        np.dtype(np.float64),
    )
    valid_json_v2 = ("|i1",)
    valid_json_v3_cases = ("int8",)
    invalid_json_v2 = (
        ">i1",
        "int8",
        "|f8",
    )
    invalid_json_v3 = (
        "|i1",
        "|f8",
        {"name": "int8", "configuration": {"endianness": "little"}},
    )

    def test_check_value(self) -> None:
        assert self.test_cls().check_value(1)
        assert not self.test_cls().check_value(["foo"])
