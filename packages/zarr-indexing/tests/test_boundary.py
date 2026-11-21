"""Positional array selectors retain their values through bounds validation."""

from typing import Literal

import numpy as np
import pytest

from zarr_indexing.boundary import normalize_positional_selection
from zarr_indexing.domain import IndexDomain


@pytest.mark.parametrize("dtype", ["i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8"])
@pytest.mark.parametrize("mode", ["orthogonal", "vectorized"])
@pytest.mark.parametrize("origin", [-3, 0, 10])
def test_positional_integer_arrays(
    dtype: str, mode: Literal["orthogonal", "vectorized"], origin: int
) -> None:
    domain = IndexDomain((origin,), (origin + 4,))
    values = [-4, -1, 0, 3] if np.dtype(dtype).kind == "i" else [0, 3, 0, 3]
    selector = np.array(values, dtype=dtype)
    snapshot = selector.copy()
    normalized = normalize_positional_selection(selector, domain, mode)
    expected = np.array([value % 4 + origin for value in values], dtype=np.intp)
    np.testing.assert_array_equal(normalized[0], expected)
    np.testing.assert_array_equal(selector, snapshot)
    assert normalized[0].dtype == np.dtype(np.intp)


@pytest.mark.parametrize(
    ("dtype", "value"),
    [
        (dtype, value)
        for dtype in ("i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8")
        for value in (
            (int(np.iinfo(dtype).min), -5, 4, int(np.iinfo(dtype).max))
            if np.dtype(dtype).kind == "i"
            else (4, int(np.iinfo(dtype).max))
        )
    ],
)
@pytest.mark.parametrize("mode", ["orthogonal", "vectorized"])
def test_positional_integer_array_out_of_bounds(
    dtype: str, value: int, mode: Literal["orthogonal", "vectorized"]
) -> None:
    selector = np.array([value], dtype=dtype)
    with pytest.raises(IndexError, match="out of bounds"):
        normalize_positional_selection(selector, IndexDomain.from_shape((4,)), mode)
