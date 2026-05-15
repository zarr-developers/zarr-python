from __future__ import annotations

import math

import numpy as np

from tests.test_dtype.test_wrapper import BaseTestZDType
from zarr.core.dtype.npy.complex import Complex64, Complex128


class _BaseTestFloat(BaseTestZDType):
    def scalar_equals(self, scalar1: object, scalar2: object) -> bool:
        if np.isnan(scalar1) and np.isnan(scalar2):  # type: ignore[call-overload]
            return True
        return super().scalar_equals(scalar1, scalar2)


class TestComplex64(_BaseTestFloat):
    test_cls = Complex64
    valid_dtype = (np.dtype(">c8"), np.dtype("<c8"))
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.float64),
        np.dtype(np.complex128),
    )
    valid_json_v2 = (
        {"name": ">c8", "object_codec_id": None},
        {"name": "<c8", "object_codec_id": None},
    )
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

    scalar_v2_params = (
        (Complex64(), (1.0, 1.0)),
        (Complex64(), (-1.0, "Infinity")),
        (Complex64(), (0, "NaN")),
    )
    scalar_v3_params = (
        (Complex64(), (1.0, 1.0)),
        (Complex64(), (-1.0, "Infinity")),
        (Complex64(), (0, "NaN")),
    )
    cast_value_params = (
        (Complex64(), complex(1.0, 1.0), np.complex64(complex(1.0, 1.0))),
        (Complex64(), complex(-1.0, math.inf), np.complex64(complex(-1.0, math.inf))),
        (Complex64(), complex(0, math.nan), np.complex64(complex(0, math.nan))),
    )
    invalid_scalar_params = ((Complex64(), {"type": "dict"}),)
    item_size_params = (Complex64(),)


class TestComplex128(_BaseTestFloat):
    test_cls = Complex128
    valid_dtype = (np.dtype(">c16"), np.dtype("<c16"))
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.float64),
        np.dtype(np.complex64),
    )
    valid_json_v2 = (
        {"name": ">c16", "object_codec_id": None},
        {"name": "<c16", "object_codec_id": None},
    )
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

    scalar_v2_params = (
        (Complex128(), (1.0, 1.0)),
        (Complex128(), (-1.0, "Infinity")),
        (Complex128(), (0, "NaN")),
    )
    scalar_v3_params = (
        (Complex128(), (1.0, 1.0)),
        (Complex128(), (-1.0, "Infinity")),
        (Complex128(), (0, "NaN")),
    )
    cast_value_params = (
        (Complex128(), complex(1.0, 1.0), np.complex128(complex(1.0, 1.0))),
        (Complex128(), complex(-1.0, math.inf), np.complex128(complex(-1.0, math.inf))),
        (Complex128(), complex(0, math.nan), np.complex128(complex(0, math.nan))),
    )
    invalid_scalar_params = ((Complex128(), {"type": "dict"}),)
    item_size_params = (Complex128(),)
