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
    valid_json_v2 = (">c8", ">c8")
    valid_json_v3 = ("complex64",)
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

    scalar_v2_params = ((">c8", (1.0, 1.0)), ("<c8", (-1.0, "Infinity")), (">c8", (0, "NaN")))
    scalar_v3_params = (
        ("complex64", (1.0, 1.0)),
        ("complex64", (-1.0, "Infinity")),
        ("complex64", (0, "NaN")),
    )


class TestComplex128(_TestZDType):
    test_cls = Complex128
    valid_dtype = (np.dtype(">c16"), np.dtype("<c16"))
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.float64),
        np.dtype(np.complex64),
    )
    valid_json_v2 = (">c16", "<c16")
    valid_json_v3 = ("complex128",)
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

    scalar_v2_params = ((">c16", (1.0, 1.0)), ("<c16", (-1.0, "Infinity")), (">c16", (0, "NaN")))
    scalar_v3_params = (
        ("complex128", (1.0, 1.0)),
        ("complex128", (-1.0, "Infinity")),
        ("complex128", (0, "NaN")),
    )
