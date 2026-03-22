"""Tests for masked array support in Zarr."""

import warnings
from typing import Any

import numpy as np
import numpy.ma as ma

import zarr


class TestMaskedArrays:
    """Test support for numpy masked arrays."""

    def test_create_array_from_masked_array(self) -> None:
        """Test creating a Zarr array from a masked array."""
        masked_array: Any = ma.masked_array([1, 2, 3], mask=[0, 1, 0])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            zarr_array = zarr.array(masked_array)

            # Check that a warning was issued
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "Masked arrays are not fully supported" in str(w[0].message)

        # The data should be stored with fill values
        result: Any = zarr_array[:]
        assert result.shape == (3,)
        # Check that unmasked values are preserved
        assert result[0] == 1
        assert result[2] == 3

    def test_create_array_from_masked_array_with_dtype(self) -> None:
        """Test creating a Zarr array from a masked array with explicit dtype."""
        masked_array: Any = ma.masked_array([1.5, 2.5, 3.5], mask=[0, 1, 0])

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            zarr_array = zarr.array(masked_array, dtype=np.float32)

        result = zarr_array[:]
        assert result.dtype == np.float32
        assert result.shape == (3,)
        assert result[0] == 1.5
        assert result[2] == 3.5

    def test_create_array_from_masked_array_with_chunks(self) -> None:
        """Test creating a Zarr array from a masked array with explicit chunks."""
        masked_array: Any = ma.masked_array(np.arange(10), mask=[i % 2 == 0 for i in range(10)])

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            zarr_array = zarr.array(masked_array, chunks=5)

        assert zarr_array.chunks == (5,)
        result = zarr_array[:]
        assert result.shape == (10,)

    def test_create_array_from_2d_masked_array(self) -> None:
        """Test creating a Zarr array from a 2D masked array."""
        data = np.arange(6).reshape(2, 3)
        mask = [[0, 1, 0], [1, 0, 1]]
        masked_array: Any = ma.masked_array(data, mask=mask)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            zarr_array = zarr.array(masked_array)

        result = zarr_array[:]
        assert result.shape == (2, 3)
        # Check that unmasked values are preserved
        assert result[0, 0] == 0
        assert result[0, 2] == 2
        assert result[1, 1] == 4

    def test_create_array_from_masked_array_all_masked(self) -> None:
        """Test creating a Zarr array where all values are masked."""
        masked_array: Any = ma.masked_array([1, 2, 3], mask=[1, 1, 1])

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            zarr_array = zarr.array(masked_array)

        result = zarr_array[:]
        assert result.shape == (3,)

    def test_create_array_from_masked_array_none_masked(self) -> None:
        """Test creating a Zarr array where no values are masked."""
        masked_array: Any = ma.masked_array([1, 2, 3], mask=[0, 0, 0])

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            zarr_array = zarr.array(masked_array)

        result = zarr_array[:]
        assert result.shape == (3,)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_masked_array_with_zarr_v2_format(self) -> None:
        """Test masked arrays with Zarr v2 format."""
        masked_array: Any = ma.masked_array([1, 2, 3], mask=[0, 1, 0])

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            zarr_array = zarr.array(masked_array, zarr_format=2)

        result = zarr_array[:]
        assert result.shape == (3,)
        assert result[0] == 1
        assert result[2] == 3

    def test_masked_array_with_zarr_v3_format(self) -> None:
        """Test masked arrays with Zarr v3 format."""
        masked_array: Any = ma.masked_array([1, 2, 3], mask=[0, 1, 0])

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            zarr_array = zarr.array(masked_array, zarr_format=3)

        result = zarr_array[:]
        assert result.shape == (3,)
        assert result[0] == 1
        assert result[2] == 3

    def test_masked_array_dtype_preserved(self) -> None:
        """Test that the dtype of the masked array is preserved."""
        for dtype in [np.int32, np.int64, np.float32, np.float64]:
            masked_array: Any = ma.masked_array([1, 2, 3], mask=[0, 1, 0], dtype=dtype)

            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                zarr_array = zarr.array(masked_array)

            assert zarr_array.dtype == dtype

    def test_masked_array_with_fill_value(self) -> None:
        """Test masked arrays with explicit fill value."""
        masked_array: Any = ma.masked_array([1, 2, 3], mask=[0, 1, 0], fill_value=999)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            zarr_array = zarr.array(masked_array)

        result = zarr_array[:]
        # The filled value should be used
        assert result[0] == 1
        assert result[2] == 3

    def test_regular_array_unchanged(self) -> None:
        """Test that regular (non-masked) arrays still work as before."""
        regular_array = np.array([1, 2, 3])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            zarr_array = zarr.array(regular_array)

            # No warning should be issued for regular arrays
            masked_warnings = [warning for warning in w if "Masked arrays" in str(warning.message)]
            assert len(masked_warnings) == 0

        result = zarr_array[:]
        np.testing.assert_array_equal(result, [1, 2, 3])
