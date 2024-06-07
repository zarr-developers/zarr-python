from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Literal

import numpy as np
import pytest

from zarr.common import parse_name, parse_shapelike, product
from zarr.config import parse_indexing_order


@pytest.mark.parametrize("data", [(0, 0, 0, 0), (1, 3, 4, 5, 6), (2, 4)])
def test_product(data: tuple[int, ...]):
    assert product(data) == np.prod(data)


# todo: test
def test_concurrent_map(): ...


# todo: test
def test_to_thread(): ...


# todo: test
def test_enum_names(): ...


# todo: test
def test_parse_enum(): ...


@pytest.mark.parametrize("data", [("foo", "bar"), (10, 11)])
def test_parse_name_invalid(data: tuple[Any, Any]):
    observed, expected = data
    if isinstance(observed, str):
        with pytest.raises(ValueError, match=f"Expected '{expected}'. Got {observed} instead."):
            parse_name(observed, expected)
    else:
        with pytest.raises(
            TypeError, match=f"Expected a string, got an instance of {type(observed)}."
        ):
            parse_name(observed, expected)


@pytest.mark.parametrize("data", [("foo", "foo"), ("10", "10")])
def test_parse_name_valid(data: tuple[Any, Any]):
    observed, expected = data
    assert parse_name(observed, expected) == observed


@pytest.mark.parametrize("data", [0, 1, "hello", "f"])
def test_parse_indexing_order_invalid(data):
    with pytest.raises(ValueError, match="Expected one of"):
        parse_indexing_order(data)


@pytest.mark.parametrize("data", ["C", "F"])
def parse_indexing_order_valid(data: Literal["C", "F"]):
    assert parse_indexing_order(data) == data


@pytest.mark.parametrize("data", [("0", 1, 2, 3), {"0": "0"}, []])
def test_parse_shapelike_invalid(data: Any):
    if isinstance(data, Iterable):
        if len(data) == 0:
            with pytest.raises(ValueError, match="Expected at least one element."):
                parse_shapelike(data)
        else:
            with pytest.raises(TypeError, match="Expected an iterable of integers"):
                parse_shapelike(data)
    else:
        with pytest.raises(TypeError, match="Expected an iterable."):
            parse_shapelike(data)


@pytest.mark.parametrize("data", [range(10), [0, 1, 2, 3], (3, 4, 5)])
def test_parse_shapelike_valid(data: Iterable[Any]):
    assert parse_shapelike(data) == tuple(data)


# todo: more dtypes
@pytest.mark.parametrize("data", [("uint8", np.uint8), ("float64", np.float64)])
def parse_dtype(data: tuple[str, np.dtype]):
    unparsed, parsed = data
    assert parse_dtype(unparsed) == parsed


# todo: figure out what it means to test this
def test_parse_fill_value(): ...
