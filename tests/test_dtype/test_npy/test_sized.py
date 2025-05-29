from __future__ import annotations

from typing import Any

import numpy as np

from tests.test_dtype.test_wrapper import BaseTestZDType, V2JsonTestParams
from zarr.core.dtype import (
    FixedLengthBytes,
    Float16,
    Float64,
    Int32,
    Int64,
    Structured,
)


class TestFixedLengthBytes(BaseTestZDType):
    test_cls = FixedLengthBytes
    valid_dtype = (np.dtype("|V10"),)
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.float64),
        np.dtype("|S10"),
    )
    valid_json_v2 = (V2JsonTestParams(dtype="|V10"),)
    valid_json_v3 = (
        {"name": "fixed_length_bytes", "configuration": {"length_bytes": 0}},
        {"name": "fixed_length_bytes", "configuration": {"length_bytes": 8}},
    )

    invalid_json_v2 = (
        "|V",
        "|S10",
        "|f8",
    )
    invalid_json_v3 = (
        {"name": "r10"},
        {"name": "r-80"},
    )

    scalar_v2_params = (
        (FixedLengthBytes(length=0), ""),
        (FixedLengthBytes(length=2), "YWI="),
        (FixedLengthBytes(length=4), "YWJjZA=="),
    )
    scalar_v3_params = (
        (FixedLengthBytes(length=0), ""),
        (FixedLengthBytes(length=2), "YWI="),
        (FixedLengthBytes(length=4), "YWJjZA=="),
    )
    cast_value_params = (
        (FixedLengthBytes(length=0), b"", np.void(b"")),
        (FixedLengthBytes(length=2), b"ab", np.void(b"ab")),
        (FixedLengthBytes(length=4), b"abcd", np.void(b"abcd")),
    )
    item_size_params = (
        FixedLengthBytes(length=0),
        FixedLengthBytes(length=4),
        FixedLengthBytes(length=10),
    )


class TestStructured(BaseTestZDType):
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
        V2JsonTestParams(dtype=[("field1", ">i4"), ("field2", ">f8")]),
        V2JsonTestParams(dtype=[("field1", ">i8"), ("field2", ">i4")]),
    )
    valid_json_v3 = (
        {
            "name": "structured",
            "configuration": {
                "fields": [
                    ("field1", "int32"),
                    ("field2", "float64"),
                ]
            },
        },
        {
            "name": "structured",
            "configuration": {
                "fields": [
                    (
                        "field1",
                        {
                            "name": "numpy.datetime64",
                            "configuration": {"unit": "s", "scale_factor": 1},
                        },
                    ),
                    (
                        "field2",
                        {"name": "fixed_length_utf32", "configuration": {"length_bytes": 32}},
                    ),
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

    scalar_v2_params = (
        (Structured(fields=(("field1", Int32()), ("field2", Float64()))), "AQAAAAAAAAAAAPA/"),
        (Structured(fields=(("field1", Float16()), ("field2", Int32()))), "AQAAAAAA"),
    )
    scalar_v3_params = (
        (Structured(fields=(("field1", Int32()), ("field2", Float64()))), "AQAAAAAAAAAAAPA/"),
        (Structured(fields=(("field1", Int64()), ("field2", Int32()))), "AQAAAAAAAAAAAPA/"),
    )

    cast_value_params = (
        (
            Structured(fields=(("field1", Int32()), ("field2", Float64()))),
            (1, 2.0),
            np.array((1, 2.0), dtype=[("field1", np.int32), ("field2", np.float64)]),
        ),
        (
            Structured(fields=(("field1", Int64()), ("field2", Int32()))),
            (3, 4.5),
            np.array((3, 4.5), dtype=[("field1", np.int64), ("field2", np.int32)]),
        ),
    )

    def scalar_equals(self, scalar1: Any, scalar2: Any) -> bool:
        if hasattr(scalar1, "shape") and hasattr(scalar2, "shape"):
            return np.array_equal(scalar1, scalar2)
        return super().scalar_equals(scalar1, scalar2)

    item_size_params = (
        Structured(fields=(("field1", Int32()), ("field2", Float64()))),
        Structured(fields=(("field1", Int64()), ("field2", Int32()))),
    )
