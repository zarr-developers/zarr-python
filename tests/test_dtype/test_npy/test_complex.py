from __future__ import annotations

import numpy as np

from tests.test_dtype.test_wrapper import _TestZDType
from zarr.core.dtype.npy.complex import Complex64, Complex128


class TestComplex64(_TestZDType):
    test_cls = Complex64
    valid_dtype = (np.dtype(">c8"), np.dtype("<c8"))
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.float64),
        np.dtype(np.complex128),
    )
    valid_json_v2 = Complex64._zarr_v2_names
    valid_json_v3_cases = (Complex64._zarr_v3_name,)
    invalid_json_v2 = (
        "|c8",
        "complex64",
        "|f8",
    )
    invalid_json_v3 = (
        "|c8",
        "|f8",
        {"name": "complex64", "configuration": {"endianness": "little"}},
    )


class TestComplex128(_TestZDType):
    test_cls = Complex128
    valid_dtype = (np.dtype(">c16"), np.dtype("<c16"))
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.float64),
        np.dtype(np.complex64),
    )
    valid_json_v2 = Complex128._zarr_v2_names
    valid_json_v3_cases = (Complex128._zarr_v3_name,)
    invalid_json_v2 = (
        "|c16",
        "complex128",
        "|f8",
    )
    invalid_json_v3 = (
        "|c16",
        "|f8",
        {"name": "complex128", "configuration": {"endianness": "little"}},
    )
