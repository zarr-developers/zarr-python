from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

from collections.abc import Sequence

import numpy as np
import pytest

from zarr.metadata import parse_dimension_names
from zarr.metadata import parse_fill_value_v3 as parse_fill_value
from zarr.metadata import parse_zarr_format_v3 as parse_zarr_format

bool_dtypes = ("bool",)

int_dtypes = (
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
)

float_dtypes = (
    "float16",
    "float32",
    "float64",
)

complex_dtypes = ("complex64", "complex128")

dtypes = (*bool_dtypes, *int_dtypes, *float_dtypes, *complex_dtypes)


@pytest.mark.parametrize("data", [None, 1, 2, 4, 5, "3"])
def test_parse_zarr_format_invalid(data: Any) -> None:
    with pytest.raises(ValueError, match=f"Invalid value. Expected 3. Got {data}"):
        parse_zarr_format(data)


def test_parse_zarr_format_valid() -> None:
    assert parse_zarr_format(3) == 3


@pytest.mark.parametrize("data", [(), [1, 2, "a"], {"foo": 10}])
def parse_dimension_names_invalid(data: Any) -> None:
    with pytest.raises(TypeError, match="Expected either None or iterable of str,"):
        parse_dimension_names(data)


@pytest.mark.parametrize("data", [None, ("a", "b", "c"), ["a", "a", "a"]])
def parse_dimension_names_valid(data: Sequence[str] | None) -> None:
    assert parse_dimension_names(data) == data


@pytest.mark.parametrize("dtype_str", dtypes)
def test_parse_auto_fill_value(dtype_str: str) -> None:
    """
    Test that parse_fill_value(None, dtype) results in the 0 value for the given dtype.
    """
    dtype = np.dtype(dtype_str)
    fill_value = None
    assert parse_fill_value(fill_value, dtype) == dtype.type(0)


@pytest.mark.parametrize("fill_value", [0, 1.11, False, True])
@pytest.mark.parametrize("dtype_str", dtypes)
def test_parse_fill_value_valid(fill_value: Any, dtype_str: str) -> None:
    """
    Test that parse_fill_value(fill-value, dtype) casts fill_value to the given dtype.
    """
    dtype = np.dtype(dtype_str)
    assert parse_fill_value(fill_value, dtype) == dtype.type(fill_value)


@pytest.mark.parametrize("fill_value", ["not a valid value"])
@pytest.mark.parametrize("dtype_str", [*int_dtypes, *float_dtypes, *complex_dtypes])
def test_parse_fill_value_invalid_value(fill_value: Any, dtype_str: str) -> None:
    """
    Test that parse_fill_value(fill_value, dtype) raises ValueError for invalid values.
    This test excludes bool because the bool constructor takes anything.
    """
    dtype = np.dtype(dtype_str)
    with pytest.raises(ValueError):
        parse_fill_value(fill_value, dtype)


@pytest.mark.parametrize("fill_value", [[1.0, 0.0], [0, 1], complex(1, 1), np.complex64(0)])
@pytest.mark.parametrize("dtype_str", [*complex_dtypes])
def test_parse_fill_value_complex(fill_value: Any, dtype_str: str) -> None:
    """
    Test that parse_fill_value(fill_value, dtype) correctly handles complex values represented
    as length-2 sequences
    """
    dtype = np.dtype(dtype_str)
    if isinstance(fill_value, list):
        expected = dtype.type(complex(*fill_value))
    else:
        expected = dtype.type(fill_value)
    assert expected == parse_fill_value(fill_value, dtype)


@pytest.mark.parametrize("fill_value", [[1.0, 0.0, 3.0], [0, 1, 3], [1]])
@pytest.mark.parametrize("dtype_str", [*complex_dtypes])
def test_parse_fill_value_complex_invalid(fill_value: Any, dtype_str: str) -> None:
    """
    Test that parse_fill_value(fill_value, dtype) correctly rejects sequences with length not
    equal to 2
    """
    dtype = np.dtype(dtype_str)
    match = (
        f"Got an invalid fill value for complex data type {dtype}."
        f"Expected a sequence with 2 elements, but {fill_value} has "
        f"length {len(fill_value)}."
    )
    with pytest.raises(ValueError, match=re.escape(match)):
        parse_fill_value(fill_value=fill_value, dtype=dtype)


@pytest.mark.parametrize("fill_value", [{"foo": 10}])
@pytest.mark.parametrize("dtype_str", [*int_dtypes, *float_dtypes, *complex_dtypes])
def test_parse_fill_value_invalid_type(fill_value: Any, dtype_str: str) -> None:
    """
    Test that parse_fill_value(fill_value, dtype) raises TypeError for invalid non-sequential types.
    This test excludes bool because the bool constructor takes anything.
    """
    dtype = np.dtype(dtype_str)
    match = "must be"
    with pytest.raises(TypeError, match=match):
        parse_fill_value(fill_value, dtype)


@pytest.mark.parametrize(
    "fill_value",
    [
        [
            1,
        ],
        (1, 23, 4),
    ],
)
@pytest.mark.parametrize("dtype_str", [*int_dtypes, *float_dtypes])
def test_parse_fill_value_invalid_type_sequence(fill_value: Any, dtype_str: str) -> None:
    """
    Test that parse_fill_value(fill-value, dtype) raises TypeError for invalid sequential types.
    This test excludes bool because the bool constructor takes anything.
    """
    dtype = np.dtype(dtype_str)
    match = f"Cannot parse non-string sequence {fill_value} as a scalar with type {dtype}"
    with pytest.raises(TypeError, match=re.escape(match)):
        parse_fill_value(fill_value, dtype)
