from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

import pytest

from zarr.metadata.v3 import (
    parse_dimension_names,
    parse_fill_value,
    parse_zarr_format,
)

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


@pytest.mark.parametrize("data", [None, ("a", "b", "c"), ["a", "a", "a"]])
def parse_dimension_names_valid(data: Sequence[str] | None) -> None:
    assert parse_dimension_names(data) == data


@pytest.mark.parametrize("data", [(), [1, 2, "a"], {"foo": 10}])
def parse_dimension_names_invalid(data: Any) -> None:
    with pytest.raises(TypeError, match="Expected either None or iterable of str,"):
        parse_dimension_names(data)


def test_parse_zarr_format_valid() -> None:
    assert parse_zarr_format(3) == 3


@pytest.mark.parametrize("data", [None, 2, "3", (1,)])
def test_parse_zarr_format_invalid(data: Any) -> None:
    with pytest.raises(ValueError, match=re.escape(f"Invalid value. Expected 3. Got {data}.")):
        parse_zarr_format(data)


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
    Test that parse_fill_value(fill-value, dtype) raises ValueError for invalid values.

    This test excludes bool because the bool constructor takes anything.
    """
    dtype = np.dtype(dtype_str)
    with pytest.raises(ValueError):
        parse_fill_value(fill_value, dtype)


@pytest.mark.parametrize("fill_value", [{"foo": 10}, (1, 23, 4)])
@pytest.mark.parametrize("dtype_str", [*int_dtypes, *float_dtypes, *complex_dtypes])
def test_parse_fill_value_invalid_type(fill_value: Any, dtype_str: str) -> None:
    """
    Test that parse_fill_value(fill-value, dtype) raises TypeError for invalid types.

    This test excludes bool because the bool constructor takes anything.
    """
    dtype = np.dtype(dtype_str)
    with pytest.raises(TypeError):
        parse_fill_value(fill_value, dtype)
