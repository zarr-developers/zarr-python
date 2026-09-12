"""Cover the `numpy_timedelta64_configuration` validator.

The pydantic-driven fixture tests only check the structural shape of a
configuration; the constraints that tie `unit` and `scale_factor` together
live in the validator function and are covered directly here.
"""

from __future__ import annotations

import pytest

from zarr_metadata.v3.data_type.numpy_timedelta64 import numpy_timedelta64_configuration

# (input, expected normalized output)
VALID = [
    ({"unit": "ns", "scale_factor": 1}, {"unit": "ns", "scale_factor": 1}),
    ({"unit": "s", "scale_factor": 10}, {"unit": "s", "scale_factor": 10}),
    ({"unit": "generic", "scale_factor": 1}, {"unit": "generic", "scale_factor": 1}),
    ({"unit": "us", "scale_factor": 2}, {"unit": "us", "scale_factor": 2}),
    ({"unit": "μs", "scale_factor": 2}, {"unit": "us", "scale_factor": 2}),
    ({"unit": "Y", "scale_factor": 2**31 - 1}, {"unit": "Y", "scale_factor": 2**31 - 1}),
]


@pytest.mark.parametrize(("value", "expected"), VALID, ids=lambda x: str(x))
def test_valid(value: dict[str, object], expected: dict[str, object]) -> None:
    assert numpy_timedelta64_configuration(value) == expected


@pytest.mark.parametrize(
    "value",
    [{}, {"unit": "s"}, {"scale_factor": 1}, {"unit": "s", "scale_factor": 1, "extra": 0}],
    ids=lambda x: str(x),
)
def test_wrong_keys(value: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="Expected exactly the keys"):
        numpy_timedelta64_configuration(value)


@pytest.mark.parametrize("unit", [1, None], ids=str)
def test_unit_not_a_string(unit: object) -> None:
    with pytest.raises(TypeError, match="Expected 'unit' to be a string"):
        numpy_timedelta64_configuration({"unit": unit, "scale_factor": 1})


@pytest.mark.parametrize("unit", ["", "invalid", "US", "seconds"], ids=str)
def test_unknown_unit(unit: str) -> None:
    with pytest.raises(ValueError, match="Expected one of"):
        numpy_timedelta64_configuration({"unit": unit, "scale_factor": 1})


@pytest.mark.parametrize("scale_factor", [1.0, "1", True, None], ids=str)
def test_scale_factor_not_an_integer(scale_factor: object) -> None:
    with pytest.raises(TypeError, match="Expected 'scale_factor' to be an integer"):
        numpy_timedelta64_configuration({"unit": "s", "scale_factor": scale_factor})


@pytest.mark.parametrize("scale_factor", [-1, 0, 2**31], ids=str)
def test_scale_factor_out_of_range(scale_factor: int) -> None:
    with pytest.raises(ValueError, match=r"Expected 'scale_factor' in \[1, 2147483647\]"):
        numpy_timedelta64_configuration({"unit": "s", "scale_factor": scale_factor})


def test_generic_unit_rejects_scale_factor() -> None:
    with pytest.raises(ValueError, match="'generic' unit does not take a scale factor"):
        numpy_timedelta64_configuration({"unit": "generic", "scale_factor": 2})
