"""Tests for scalar indexing fix (Issue #3741)."""

import numpy as np
import pytest

import zarr


class TestScalarIndexing:
    """Test that scalar indexing returns numpy scalars, matching numpy behavior."""

    def test_1d_scalar_indexing(self):
        """Test scalar indexing on 1-D array returns numpy scalar."""
        arr_zarr = zarr.array([1, 2, 3, 4, 5], dtype='int64')
        arr_numpy = np.array([1, 2, 3, 4, 5], dtype='int64')

        result_zarr = arr_zarr[0]
        result_numpy = arr_numpy[0]

        assert type(result_zarr) == type(result_numpy)
        assert result_zarr == result_numpy
        assert not isinstance(result_zarr, np.ndarray)
        assert isinstance(result_zarr, np.generic)

    def test_2d_scalar_indexing(self):
        """Test scalar indexing on 2-D array returns numpy scalar."""
        arr_zarr = zarr.array([[1, 2, 3], [4, 5, 6]], dtype='int64')
        arr_numpy = np.array([[1, 2, 3], [4, 5, 6]], dtype='int64')

        result_zarr = arr_zarr[0, 0]
        result_numpy = arr_numpy[0, 0]

        assert type(result_zarr) == type(result_numpy)
        assert result_zarr == result_numpy
        assert not isinstance(result_zarr, np.ndarray)

    def test_3d_scalar_indexing(self):
        """Test scalar indexing on 3-D array returns numpy scalar."""
        arr_zarr = zarr.arange(24, dtype='int64').reshape(2, 3, 4)
        arr_numpy = np.arange(24, dtype='int64').reshape(2, 3, 4)

        result_zarr = arr_zarr[0, 1, 2]
        result_numpy = arr_numpy[0, 1, 2]

        assert type(result_zarr) == type(result_numpy)
        assert result_zarr == result_numpy

    def test_slice_indexing_returns_array(self):
        """Test that slice indexing still returns arrays."""
        arr_zarr = zarr.array([1, 2, 3, 4, 5])
        result = arr_zarr[0:2]

        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert len(result) == 2

    def test_partial_scalar_indexing_on_2d(self):
        """Test partial scalar indexing on 2-D array returns 1-D array."""
        arr_zarr = zarr.array([[1, 2, 3], [4, 5, 6]], dtype='int64')
        arr_numpy = np.array([[1, 2, 3], [4, 5, 6]], dtype='int64')

        result_zarr = arr_zarr[0]
        result_numpy = arr_numpy[0]

        assert type(result_zarr) == type(result_numpy)
        assert isinstance(result_zarr, np.ndarray)
        assert result_zarr.ndim == 1
        np.testing.assert_array_equal(result_zarr, result_numpy)

    def test_float_dtype_scalar_indexing(self):
        """Test scalar indexing with float dtype."""
        arr_zarr = zarr.array([1.5, 2.5, 3.5], dtype='float64')
        arr_numpy = np.array([1.5, 2.5, 3.5], dtype='float64')

        result_zarr = arr_zarr[0]
        result_numpy = arr_numpy[0]

        assert type(result_zarr) == type(result_numpy)
        assert result_zarr == result_numpy

    def test_negative_indexing(self):
        """Test scalar indexing with negative indices."""
        arr_zarr = zarr.array([1, 2, 3, 4, 5], dtype='int64')
        arr_numpy = np.array([1, 2, 3, 4, 5], dtype='int64')

        result_zarr = arr_zarr[-1]
        result_numpy = arr_numpy[-1]

        assert type(result_zarr) == type(result_numpy)
        assert result_zarr == result_numpy

    def test_ellipsis_indexing_returns_array(self):
        """Test that ellipsis indexing returns the full array."""
        arr_zarr = zarr.array([1, 2, 3], dtype='int64')
        result = arr_zarr[...]

        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))

    def test_mixed_slice_and_scalar(self):
        """Test mixed slice and scalar indexing."""
        arr_zarr = zarr.arange(24, dtype='int64').reshape(2, 3, 4)
        arr_numpy = np.arange(24, dtype='int64').reshape(2, 3, 4)

        # [0, :, 2] should return 1-D array
        result_zarr = arr_zarr[0, :, 2]
        result_numpy = arr_numpy[0, :, 2]

        assert type(result_zarr) == type(result_numpy)
        assert isinstance(result_zarr, np.ndarray)
        assert result_zarr.ndim == 1
        np.testing.assert_array_equal(result_zarr, result_numpy)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
