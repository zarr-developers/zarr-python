from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.test_dtype.test_wrapper import BaseTestZDType
from zarr.core.dtype import (
    Float16,
    Float64,
    Int32,
    Int64,
    Structured,
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
        {"name": [["field1", ">i4"], ["field2", ">f8"]], "object_codec_id": None},
        {"name": [["field1", ">i8"], ["field2", ">i4"]], "object_codec_id": None},
    )
    valid_json_v3 = (
        {
            "name": "structured",
            "configuration": {
                "fields": [
                    ["field1", "int32"],
                    ["field2", "float64"],
                ]
            },
        },
        {
            "name": "structured",
            "configuration": {
                "fields": [
                    [
                        "field1",
                        {
                            "name": "numpy.datetime64",
                            "configuration": {"unit": "s", "scale_factor": 1},
                        },
                    ],
                    [
                        "field2",
                        {"name": "fixed_length_utf32", "configuration": {"length_bytes": 32}},
                    ],
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

    item_size_params = (
        Structured(fields=(("field1", Int32()), ("field2", Float64()))),
        Structured(fields=(("field1", Int64()), ("field2", Int32()))),
    )

    invalid_scalar_params = (
        (Structured(fields=(("field1", Int32()), ("field2", Float64()))), "i am a string"),
        (Structured(fields=(("field1", Int32()), ("field2", Float64()))), {"type": "dict"}),
    )

    def scalar_equals(self, scalar1: Any, scalar2: Any) -> bool:
        if hasattr(scalar1, "shape") and hasattr(scalar2, "shape"):
            return np.array_equal(scalar1, scalar2)
        return super().scalar_equals(scalar1, scalar2)


def test_invalid_size() -> None:
    """
    Test that it's impossible to create a data type that has no fields
    """
    fields = ()
    msg = f"must have at least one field. Got {fields!r}"
    with pytest.raises(ValueError, match=msg):
        Structured(fields=fields)
