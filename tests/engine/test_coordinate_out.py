"""`get_coordinate_selection(..., out=...)` must validate `out` against the
shape the *engine* reads, and reshape only its return value.

A coordinate selection is pointwise: the request the engine serves is flat, and
the coordinate arrays' own (possibly multi-dimensional) shape is restored by the
caller afterwards. So `out` is sized by the number of selected points, while the
returned array carries the selection's shape. These tests pin both halves, and
the error when `out` matches neither.
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
    out = default_buffer_prototype().nd_buffer.from_numpy_array(np.zeros((4,), dtype="int32"))
    result = z.get_coordinate_selection(coords, out=out)
    expected = np.array([[0, 6], [12, 18]], dtype="int32")
    # the result carries the coordinate arrays' shape; `out` holds the points
    np.testing.assert_array_equal(np.asarray(result), expected)
    np.testing.assert_array_equal(out.as_numpy_array(), expected.reshape(-1))


async def test_coordinate_selection_out_multidim_async() -> None:
    z = _filled_array()
    coords = (np.array([[0, 1], [2, 3]]), np.array([[0, 1], [2, 3]]))
    out = default_buffer_prototype().nd_buffer.from_numpy_array(np.zeros((4,), dtype="int32"))
    result = await z.async_array.get_coordinate_selection(coords, out=out)
    expected = np.array([[0, 6], [12, 18]], dtype="int32")
    np.testing.assert_array_equal(np.asarray(result), expected)
    np.testing.assert_array_equal(out.as_numpy_array(), expected.reshape(-1))


def test_coordinate_selection_out_1d() -> None:
    z = _filled_array()
    coords = (np.array([0, 1, 4]), np.array([0, 1, 4]))
    out = default_buffer_prototype().nd_buffer.from_numpy_array(np.zeros((3,), dtype="int32"))
    result = z.get_coordinate_selection(coords, out=out)
    expected = np.array([0, 6, 24], dtype="int32")
    np.testing.assert_array_equal(np.asarray(result), expected)
    np.testing.assert_array_equal(out.as_numpy_array(), expected)


@pytest.mark.parametrize("out_shape", [(2, 2), (3,)])
def test_coordinate_selection_out_shape_mismatch_raises(out_shape: tuple[int, ...]) -> None:
    # neither the selection's own 2-d shape nor a wrong-length flat buffer is
    # the shape the engine reads into
    z = _filled_array()
    coords = (np.array([[0, 1], [2, 3]]), np.array([[0, 1], [2, 3]]))
    out = default_buffer_prototype().nd_buffer.from_numpy_array(np.zeros(out_shape, dtype="int32"))
    with pytest.raises(ValueError, match="shape of out argument"):
        z.get_coordinate_selection(coords, out=out)
