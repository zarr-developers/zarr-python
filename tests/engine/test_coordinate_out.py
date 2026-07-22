"""`get_coordinate_selection(..., out=...)` must validate `out` against the
selection shape, which may be multi-dimensional.

The engine facade flattens the coordinate arrays into a pointwise index, so the
raw read is 1-d; a naive `out`-shape check against that flattened shape rejected
a multi-dimensional `out` that the pre-engine implementation accepted. These
tests pin both the multi-dimensional and the flat cases.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import zarr
from zarr.core.buffer import default_buffer_prototype
from zarr.storage import MemoryStore


def _filled_array() -> zarr.Array[Any]:
    z = zarr.create_array(MemoryStore(), shape=(5, 5), chunks=(5, 5), dtype="int32")
    z[:] = np.arange(25, dtype="int32").reshape(5, 5)
    return z


def test_coordinate_selection_out_multidim() -> None:
    z = _filled_array()
    coords = (np.array([[0, 1], [2, 3]]), np.array([[0, 1], [2, 3]]))
    out = default_buffer_prototype().nd_buffer.from_numpy_array(np.zeros((2, 2), dtype="int32"))
    result = z.get_coordinate_selection(coords, out=out)
    expected = np.array([[0, 6], [12, 18]], dtype="int32")
    np.testing.assert_array_equal(np.asarray(result), expected)
    np.testing.assert_array_equal(out.as_numpy_array(), expected)


def test_coordinate_selection_out_1d() -> None:
    z = _filled_array()
    coords = (np.array([0, 1, 4]), np.array([0, 1, 4]))
    out = default_buffer_prototype().nd_buffer.from_numpy_array(np.zeros((3,), dtype="int32"))
    result = z.get_coordinate_selection(coords, out=out)
    expected = np.array([0, 6, 24], dtype="int32")
    np.testing.assert_array_equal(np.asarray(result), expected)
    np.testing.assert_array_equal(out.as_numpy_array(), expected)


def test_coordinate_selection_out_shape_mismatch_raises() -> None:
    z = _filled_array()
    coords = (np.array([[0, 1], [2, 3]]), np.array([[0, 1], [2, 3]]))
    # a flat out no longer matches the 2-d selection shape
    out = default_buffer_prototype().nd_buffer.from_numpy_array(np.zeros((4,), dtype="int32"))
    with pytest.raises(ValueError, match="shape of out argument"):
        z.get_coordinate_selection(coords, out=out)
