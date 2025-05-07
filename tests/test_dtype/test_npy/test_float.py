from __future__ import annotations

import numpy as np

from tests.test_dtype.test_wrapper import _TestZDType
from zarr.core.dtype.npy.float import Float16, Float32, Float64


class TestFloat16(_TestZDType):
    test_cls = Float16
    valid_dtype = (np.dtype(">f2"), np.dtype("<f2"))
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.uint16),
        np.dtype(np.float32),
    )
    valid_json_v2 = Float16._zarr_v2_names
    valid_json_v3_cases = (Float16._zarr_v3_name,)
    invalid_json_v2 = (
        "|f2",
        "float16",
        "|i1",
    )
    invalid_json_v3 = (
        "|f2",
        "|i1",
        {"name": "float16", "configuration": {"endianness": "little"}},
    )


class TestFloat32(_TestZDType):
    test_cls = Float32
    valid_dtype = (np.dtype(">f4"), np.dtype("<f4"))
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.uint16),
        np.dtype(np.float64),
    )
    valid_json_v2 = Float32._zarr_v2_names
    valid_json_v3_cases = (Float32._zarr_v3_name,)
    invalid_json_v2 = (
        "|f4",
        "float32",
        "|i1",
    )
    invalid_json_v3 = (
        "|f4",
        "|i1",
        {"name": "float32", "configuration": {"endianness": "little"}},
    )


class TestFloat64(_TestZDType):
    test_cls = Float64
    valid_dtype = (np.dtype(">f8"), np.dtype("<f8"))
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.uint16),
        np.dtype(np.float32),
    )
    valid_json_v2 = Float64._zarr_v2_names
    valid_json_v3_cases = (Float64._zarr_v3_name,)
    invalid_json_v2 = (
        "|f8",
        "float64",
        "|i1",
    )
    invalid_json_v3 = (
        "|f8",
        "|i1",
        {"name": "float64", "configuration": {"endianness": "little"}},
    )
