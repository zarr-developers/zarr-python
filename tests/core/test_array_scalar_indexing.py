"""Tests for scalar indexing (Issue #3741)."""

import numpy as np

import zarr


class TestScalarIndexing:
    """Test that scalar indexing returns numpy scalars, matching numpy behavior."""

    def test_1d_scalar_indexing(self) -> None:
        """Test scalar indexing on 1-D array returns numpy scalar."""
        arr_zarr = zarr.array([1, 2, 3, 4, 5], dtype="int64")
        arr_numpy = np.array([1, 2, 3, 4, 5], dtype="int64")

        result_zarr = arr_zarr[0]
        result_numpy = arr_numpy[0]

        assert type(result_zarr) is type(result_numpy)
        assert result_zarr == result_numpy
        assert not isinstance(result_zarr, np.ndarray)
        assert isinstance(result_zarr, np.generic)

    def test_2d_scalar_indexing(self) -> None:
        """Test scalar indexing on 2-D array returns numpy scalar."""
        arr_zarr = zarr.array([[1, 2, 3], [4, 5, 6]], dtype="int64")
        arr_numpy = np.array([[1, 2, 3], [4, 5, 6]], dtype="int64")

        result_zarr = arr_zarr[0, 0]
        result_numpy = arr_numpy[0, 0]

        assert type(result_zarr) is type(result_numpy)
        assert result_zarr == result_numpy
        assert not isinstance(result_zarr, np.ndarray)

    def test_slice_indexing_returns_array(self) -> None:
        """Test that slice indexing still returns arrays."""
        arr_zarr = zarr.array([1, 2, 3, 4, 5])
        result = arr_zarr[0:2]

        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert len(result) == 2

    def test_partial_scalar_indexing_on_2d(self) -> None:
        """Test partial scalar indexing on 2-D array returns 1-D array."""
        arr_zarr = zarr.array([[1, 2, 3], [4, 5, 6]], dtype="int64")
        arr_numpy = np.array([[1, 2, 3], [4, 5, 6]], dtype="int64")

        result_zarr = arr_zarr[0]
        result_numpy = arr_numpy[0]

        assert type(result_zarr) is type(result_numpy)
        assert isinstance(result_zarr, np.ndarray)
        assert result_zarr.ndim == 1
        np.testing.assert_array_equal(result_zarr, result_numpy)
