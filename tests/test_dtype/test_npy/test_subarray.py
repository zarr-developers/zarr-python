from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.test_dtype.test_wrapper import BaseTestZDType
from zarr.core.dtype import (
    Bool,
    DateTime64,
    Float16,
    Float32,
    Float64,
    Int16,
    Int32,
    Int64,
    Structured,
    Subarray,
)


class TestSubarray(BaseTestZDType):
    test_cls = Subarray
    valid_dtype = (
        np.dtype((np.float32, (2, 2))),
        np.dtype((np.int32, (3,))),
        np.dtype((bool, (5, 5, 5, 5))),
    )
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.float64),
        np.dtype("|S10"),
        np.dtype([("abc", np.float32, (2, 2))]),
    )
    valid_json_v2 = ()
    valid_json_v3 = (
        {
            "name": "subarray",
            "configuration": {
                "subdtype": "float32",
                "shape": [2, 2],
            },
        },
        {
            "name": "subarray",
            "configuration": {
                "subdtype": {
                    "name": "numpy.datetime64",
                    "configuration": {"unit": "s", "scale_factor": 1},
                },
                "shape": [5],
            },
        },
    )
    invalid_json_v2 = ()
    invalid_json_v3 = (
        {
            "name": "subarray",
            "configuration": {
                "subdtype": "float32",
                "shape": [],  # empty shape not allowed
            },
        },
        {
            "name": "subarray",
            "configuration": {
                "subdtype": "",  # invalid subdtype
                "shape": [6, 6],
            },
        },
        {"name": "invalid_name"},
    )

    scalar_v2_params = ()
    scalar_v3_params = (
        (Subarray(subdtype=Float16(), shape=(5,)), 0.0),
        (Subarray(subdtype=Float32(), shape=(2, 2)), "NaN"),
        (Subarray(subdtype=Float32(), shape=(2, 2)), "Infinity"),
        (Subarray(subdtype=Int32(), shape=(1, 2)), 42),
        (Subarray(subdtype=Int32(), shape=(1, 2)), "AgAAAAEAAAA="),
        (Subarray(subdtype=Int16(), shape=(1, 3)), "BBBBBBBB"),
        (Subarray(subdtype=Int32(), shape=(2, 2)), "AgAAAAEAAAAQAAAAEAAAAA=="),
        (Subarray(subdtype=Bool(), shape=(1, 2)), True),
        (Subarray(subdtype=Bool(), shape=(1, 2)), False),
        (Subarray(subdtype=DateTime64(unit="s"), shape=(2, 2)), 123),
    )

    cast_value_params = (
        (
            Subarray(subdtype=Float16(), shape=(2,)),
            np.array([1.0, 1.0], dtype=np.float16),
            np.array([1.0, 1.0], dtype=np.float16),
        ),  # identity
        # Scalar Broadcasting
        (Subarray(subdtype=Float16(), shape=(2,)), 1.0, np.array([1.0, 1.0], dtype=np.float16)),
        (
            Subarray(subdtype=Float16(), shape=(2,)),
            np.nan,
            np.array([np.nan, np.nan], dtype=np.float16),
        ),
        (Subarray(subdtype=Bool(), shape=(1, 1)), 1.0, np.array([[True]], dtype=bool)),
        # From bytes
        (Subarray(subdtype=Float16(), shape=(2, 2)), bytes(8), np.zeros((2, 2), dtype=np.float16)),
        (
            Subarray(subdtype=Int16(), shape=(1, 3)),
            bytes([0x01, 0x00, 0x02, 0x00, 0x03, 0x00]),
            np.array([[1, 2, 3]], dtype=np.int16),
        ),
        # From nested lists/tuples
        (
            Subarray(subdtype=Int32(), shape=(2, 3)),
            [[1, 2, 3], [4, 5, 6]],
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
        ),
        (
            Subarray(subdtype=Float32(), shape=(1, 2)),
            [(1, 2)],
            np.array([[1, 2]], dtype=np.float32),
        ),  # mixed list/tuple + dtype conversion
    )

    item_size_params = (
        Subarray(subdtype=Int64(), shape=(4, 8)),
        Subarray(subdtype=Float16(), shape=(5,)),
        Subarray(subdtype=Structured(fields=(("abc", Int16()), ("def", Float32()))), shape=(5, 2)),
    )

    invalid_scalar_params = (
        (Subarray(subdtype=Int32(), shape=(1, 1)), bytes(0)),  # empty bytes
        (Subarray(subdtype=Int64(), shape=(4, 8)), bytes(2)),  # wrong size
        (Subarray(subdtype=Int32(), shape=(1, 2)), [[1]]),  # wrong shape
        (Subarray(subdtype=Int32(), shape=(1, 2)), [[1, 2, 3]]),  # wrong shape
        (Subarray(subdtype=Int32(), shape=(1, 2)), [1, 2]),  # not nested
        (Subarray(subdtype=Float64(), shape=(2, 2)), None),
        # (Subarray(subdtype=Float64(), shape=(2, 2)), "some string"),
        (Subarray(subdtype=Float64(), shape=(2, 2)), {"a": 1}),
    )

    def scalar_equals(self, scalar1: Any, scalar2: Any) -> bool:
        if hasattr(scalar1, "shape") and hasattr(scalar2, "shape"):
            return np.array_equal(scalar1, scalar2, equal_nan=True)
        return super().scalar_equals(scalar1, scalar2)


def test_invalid_size() -> None:
    """
    Test that it's impossible to create a data type that has no fields
    """
    msg = "must have at least one dimension"
    with pytest.raises(ValueError, match=msg):
        Subarray(subdtype=Int32(), shape=())
