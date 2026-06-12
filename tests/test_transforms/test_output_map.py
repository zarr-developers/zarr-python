from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict
from typing import Any

import numpy as np
import pytest

from tests.test_transforms.conftest import Expect, ExpectErr
from zarr.core._transforms.output_map import ArrayMap, ConstantMap, DimensionMap


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            input=ConstantMap(offset=42),
            expected={"offset": 42},
            id="ConstantMap-explicit-offset",
        ),
        Expect(
            input=ConstantMap(),
            expected={"offset": 0},
            id="ConstantMap-default-offset",
        ),
        Expect(
            input=DimensionMap(input_dimension=3, offset=5, stride=2),
            expected={"input_dimension": 3, "offset": 5, "stride": 2},
            id="DimensionMap-all-fields",
        ),
        Expect(
            input=DimensionMap(input_dimension=0),
            expected={"input_dimension": 0, "offset": 0, "stride": 1},
            id="DimensionMap-defaults",
        ),
        Expect(
            input=ArrayMap(
                index_array=np.array([1, 3, 5], dtype=np.intp),
                input_dimensions=(0,),
                offset=10,
                stride=2,
            ),
            expected={
                "index_array": np.array([1, 3, 5], dtype=np.intp),
                "input_dimensions": (0,),
                "offset": 10,
                "stride": 2,
            },
            id="ArrayMap-all-fields",
        ),
        Expect(
            input=ArrayMap(
                index_array=np.array([0, 1], dtype=np.intp),
                input_dimensions=(0,),
            ),
            expected={
                "index_array": np.array([0, 1], dtype=np.intp),
                "input_dimensions": (0,),
                "offset": 0,
                "stride": 1,
            },
            id="ArrayMap-defaults",
        ),
    ],
    ids=lambda c: c.id,
)
def test_construction_success(case: Expect[Any, dict[str, Any]]) -> None:
    """Constructing each map type with explicit and default values yields the expected fields."""
    actual = asdict(case.input)
    assert set(actual) == set(case.expected)
    for field, expected_value in case.expected.items():
        if isinstance(expected_value, np.ndarray):
            np.testing.assert_array_equal(actual[field], expected_value)
        else:
            assert actual[field] == expected_value


@pytest.mark.parametrize(
    "case",
    [
        ExpectErr(
            input=(ConstantMap(offset=5), "offset", 99),
            msg="cannot assign to field 'offset'",
            exception_cls=FrozenInstanceError,
            id="ConstantMap-frozen",
        ),
        ExpectErr(
            input=(DimensionMap(input_dimension=0), "stride", 7),
            msg="cannot assign to field 'stride'",
            exception_cls=FrozenInstanceError,
            id="DimensionMap-frozen",
        ),
        ExpectErr(
            input=(
                ArrayMap(index_array=np.array([0], dtype=np.intp), input_dimensions=(0,)),
                "offset",
                1,
            ),
            msg="cannot assign to field 'offset'",
            exception_cls=FrozenInstanceError,
            id="ArrayMap-frozen",
        ),
    ],
    ids=lambda c: c.id,
)
def test_mutation_errors(case: ExpectErr[tuple[Any, str, Any]]) -> None:
    """Attempting to mutate a frozen output map raises FrozenInstanceError."""
    obj, field, new_value = case.input
    with pytest.raises(case.exception_cls, match=case.msg):
        setattr(obj, field, new_value)
