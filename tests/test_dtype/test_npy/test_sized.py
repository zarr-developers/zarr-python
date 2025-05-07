from __future__ import annotations

import numpy as np

from tests.test_dtype.test_wrapper import _TestZDType
from zarr.core.dtype.npy.sized import (
    FixedLengthAscii,
    FixedLengthBytes,
    FixedLengthUnicode,
    Structured,
)


class TestFixedLengthAscii(_TestZDType):
    test_cls = FixedLengthAscii
    valid_dtype = (np.dtype("|S10"), np.dtype("|S4"))
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.float64),
        np.dtype("|U10"),
    )
    valid_json_v2 = ("|S0", "|S2", "|S4")
    valid_json_v3_cases = (
        {"name": "numpy.fixed_length_ascii", "configuration": {"length_bits": 80}},
    )
    invalid_json_v2 = (
        "|S",
        "|U10",
        "|f8",
    )
    invalid_json_v3 = (
        {"name": "numpy.fixed_length_ascii", "configuration": {"length_bits": 0}},
        {"name": "numpy.fixed_length_ascii", "configuration": {"length_bits": "invalid"}},
    )


class TestFixedLengthBytes(_TestZDType):
    test_cls = FixedLengthBytes
    valid_dtype = (np.dtype("|V10"),)
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.float64),
        np.dtype("|S10"),
    )
    valid_json_v2 = ("|V10",)
    valid_json_v3_cases = ({"name": "r80"},)
    invalid_json_v2 = (
        "|V",
        "|S10",
        "|f8",
    )
    invalid_json_v3 = (
        {"name": "r0"},
        {"name": "r-80"},
    )


class TestFixedLengthUnicode(_TestZDType):
    test_cls = FixedLengthUnicode
    valid_dtype = (np.dtype(">U10"), np.dtype("<U10"))
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.float64),
        np.dtype("|S10"),
    )
    valid_json_v2 = (">U10", "<U10")
    valid_json_v3_cases = (
        {"name": "numpy.fixed_length_ucs4", "configuration": {"length_bits": 320}},
    )
    invalid_json_v2 = (
        "|U",
        "|S10",
        "|f8",
    )
    invalid_json_v3 = (
        {"name": "numpy.fixed_length_ucs4", "configuration": {"length_bits": 0}},
        {"name": "numpy.fixed_length_ucs4", "configuration": {"length_bits": "invalid"}},
    )


class TestStructured(_TestZDType):
    test_cls = Structured
    valid_dtype = (
        np.dtype([("field1", np.int32), ("field2", np.float64)]),
        np.dtype([("field1", np.int64), ("field2", np.int32)]),
    )
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.float64),
        np.dtype("|S10"),
    )
    valid_json_v2 = (
        [("field1", ">i4"), ("field2", ">f8")],
        [("field1", ">i8"), ("field2", ">i4")],
    )
    valid_json_v3_cases = (
        {
            "name": "structured",
            "configuration": {
                "fields": [
                    ("field1", {"name": "int32", "configuration": {"endianness": "big"}}),
                    ("field2", {"name": "float64", "configuration": {"endianness": "big"}}),
                ]
            },
        },
        {
            "name": "structured",
            "configuration": {
                "fields": [
                    ("field1", {"name": "int64", "configuration": {"endianness": "big"}}),
                    ("field2", {"name": "int32", "configuration": {"endianness": "big"}}),
                ]
            },
        },
    )
    invalid_json_v2 = (
        [("field1", "|i1"), ("field2", "|f8")],
        [("field1", "|S10"), ("field2", "|f8")],
    )
    invalid_json_v3 = (
        {
            "name": "structured",
            "configuration": {
                "fields": [
                    ("field1", {"name": "int32", "configuration": {"endianness": "invalid"}}),
                    ("field2", {"name": "float64", "configuration": {"endianness": "big"}}),
                ]
            },
        },
        {"name": "invalid_name"},
    )
