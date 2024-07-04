from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

import pytest

from zarr.metadata.v3 import (
    parse_bool,
    parse_dimension_names,
    parse_fill_value,
    parse_integer,
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


@pytest.mark.parametrize("data", [None, 1, 2, 4, 5, "3"])
def test_parse_zarr_format_invalid(data: Any) -> None:
    with pytest.raises(ValueError, match=f"Invalid value. Expected 3. Got {data}"):
        parse_zarr_format(data)


@pytest.mark.parametrize("dtype_str", dtypes)
def test_parse_auto_fill_value(dtype_str: str) -> None:
    """
    Test that parse_fill_value(None, dtype) results in the 0 value for the given dtype.
    """
    dtype = np.dtype(dtype_str)
    fill_value = None
    assert parse_fill_value(fill_value, dtype) == dtype.type(0)


@pytest.mark.parametrize("fill_value", [1, 1.1, "a", (1,)])
@pytest.mark.parametrize("dtype", bool_dtypes)
def test_parse_fill_value_bool(fill_value: Any, dtype: str) -> None:
    """
    Test that any value is cast to bool
    """
    assert parse_bool(fill_value, dtype=dtype) == np.bool_(fill_value)


@pytest.mark.parametrize("fill_value", [1.0, 100, True, np.uint(10)])
@pytest.mark.parametrize("dtype_str", int_dtypes)
def test_parse_fill_value_valid_int(fill_value: Any, dtype_str: str) -> None:
    """
    Test that integer-like values are cast to `dtype`
    """

    dtype = np.dtype(dtype_str)
    assert parse_integer(fill_value, dtype=dtype) == dtype.type(fill_value)


@pytest.mark.parametrize("fill_value", [1.1, -4.5])
@pytest.mark.parametrize("dtype_str", int_dtypes)
def test_parse_fill_value_invalid_int_float(fill_value: Any, dtype_str: str) -> None:
    """
    Test that floats get rejected by parse_integer
    """
    dtype = np.dtype(dtype_str)

    match = (
        f"Could not interpret {fill_value} as an integer, and so it is incompatible "
        f"with the provided data type {dtype}"
    )
    with pytest.raises(TypeError, match=match):
        parse_integer(fill_value, dtype=dtype)


@pytest.mark.parametrize("fill_value", ["a", (1,)])
@pytest.mark.parametrize("dtype_str", int_dtypes)
def test_parse_fill_value_invalid_int_obj(fill_value: Any, dtype_str: str) -> None:
    """
    Test that non-numeric types get rejected by parse_integer
    """
    dtype = np.dtype(dtype_str)

    match = (
        f"Could not interpret {fill_value} as a float, which is "
        f"required for converting it to the data type {dtype}."
    )
    with pytest.raises(TypeError, match=re.escape(match)):
        parse_integer(fill_value, dtype=dtype)
