"""
Tests for issue #3796: ValueError on dtype matching (Windows-specific issue with generic dtypes).

This test suite verifies that the dtype matching logic correctly handles cases where
numpy's bitwise operations produce generic dtype classes (like UIntDtype, IntDtype)
instead of specific sized types (like UInt32DType, Int32DType), which happens on Windows.
"""

from __future__ import annotations

import numpy as np
import pytest

from zarr.core.dtype.npy.int import Int16, Int32, Int64, UInt16, UInt32, UInt64


class TestDtypeMatching:
    """Test dtype matching for integer types with generic numpy dtypes (Windows issue)."""

    def test_uint32_from_normal_array(self) -> None:
        """Test that UInt32 correctly matches a normal uint32 numpy array."""
        arr = np.array([1, 2], dtype=np.uint32)
        zdtype = UInt32.from_native_dtype(arr.dtype)
        assert isinstance(zdtype, UInt32)
        assert zdtype.to_native_dtype().itemsize == 4

    def test_uint32_from_bitwise_operation(self) -> None:
        """
        Test that UInt32 correctly matches uint32 from bitwise operations.
        
        On Windows, bitwise operations on uint32 can produce UIntDtype instead of UInt32DType.
        This test verifies that our fix handles this case.
        """
        arr = np.array([1, 2], dtype=np.uint32) & 1
        # The dtype might be UInt32DType or UIntDtype depending on OS/numpy version
        assert arr.dtype.itemsize == 4
        assert np.issubdtype(arr.dtype, np.unsignedinteger)
        
        # This should not raise ValueError
        zdtype = UInt32.from_native_dtype(arr.dtype)
        assert isinstance(zdtype, UInt32)

    def test_uint16_from_bitwise_operation(self) -> None:
        """Test that UInt16 correctly matches uint16 from bitwise operations."""
        arr = np.array([1, 2], dtype=np.uint16) & 1
        assert arr.dtype.itemsize == 2
        assert np.issubdtype(arr.dtype, np.unsignedinteger)
        
        zdtype = UInt16.from_native_dtype(arr.dtype)
        assert isinstance(zdtype, UInt16)

    def test_uint64_from_bitwise_operation(self) -> None:
        """Test that UInt64 correctly matches uint64 from bitwise operations."""
        arr = np.array([1, 2], dtype=np.uint64) & 1
        assert arr.dtype.itemsize == 8
        assert np.issubdtype(arr.dtype, np.unsignedinteger)
        
        zdtype = UInt64.from_native_dtype(arr.dtype)
        assert isinstance(zdtype, UInt64)

    def test_int32_from_bitwise_operation(self) -> None:
        """Test that Int32 correctly matches int32 from bitwise operations."""
        arr = np.array([1, 2], dtype=np.int32) & 1
        assert arr.dtype.itemsize == 4
        assert np.issubdtype(arr.dtype, np.signedinteger)
        
        zdtype = Int32.from_native_dtype(arr.dtype)
        assert isinstance(zdtype, Int32)

    def test_int16_from_bitwise_operation(self) -> None:
        """Test that Int16 correctly matches int16 from bitwise operations."""
        arr = np.array([1, 2], dtype=np.int16) & 1
        assert arr.dtype.itemsize == 2
        assert np.issubdtype(arr.dtype, np.signedinteger)
        
        zdtype = Int16.from_native_dtype(arr.dtype)
        assert isinstance(zdtype, Int16)

    def test_int64_from_bitwise_operation(self) -> None:
        """Test that Int64 correctly matches int64 from bitwise operations."""
        arr = np.array([1, 2], dtype=np.int64) & 1
        assert arr.dtype.itemsize == 8
        assert np.issubdtype(arr.dtype, np.signedinteger)
        
        zdtype = Int64.from_native_dtype(arr.dtype)
        assert isinstance(zdtype, Int64)

    def test_uint32_with_different_endianness(self) -> None:
        """Test that UInt32 correctly matches uint32 with different endianness."""
        # Test native endianness
        arr_native = np.array([1, 2], dtype=np.uint32)
        zdtype_native = UInt32.from_native_dtype(arr_native.dtype)
        assert isinstance(zdtype_native, UInt32)

        # Test little-endian
        arr_le = np.array([1, 2], dtype="<u4")
        zdtype_le = UInt32.from_native_dtype(arr_le.dtype)
        assert isinstance(zdtype_le, UInt32)

        # Test big-endian
        arr_be = np.array([1, 2], dtype=">u4")
        zdtype_be = UInt32.from_native_dtype(arr_be.dtype)
        assert isinstance(zdtype_be, UInt32)

    def test_roundtrip_uint32(self) -> None:
        """Test that creating and converting back to native dtype works for UInt32."""
        zdtype = UInt32()
        native_dtype = zdtype.to_native_dtype()
        zdtype_again = UInt32.from_native_dtype(native_dtype)
        assert isinstance(zdtype_again, UInt32)
        assert zdtype_again.to_native_dtype().itemsize == 4


class TestDtypeMatchingWithZarr:
    """Test dtype matching through the zarr.array() API."""

    def test_zarr_array_from_uint32_bitwise(self) -> None:
        """Test that zarr.array() works with uint32 from bitwise operations."""
        import zarr

        arr = np.array([1, 2], dtype=np.uint32) & 1
        # This should not raise ValueError
        z = zarr.array(arr)
        assert z.dtype == np.dtype("uint32")
        assert z.shape == (2,)

    def test_zarr_array_from_uint16_bitwise(self) -> None:
        """Test that zarr.array() works with uint16 from bitwise operations."""
        import zarr

        arr = np.array([1, 2], dtype=np.uint16) & 1
        z = zarr.array(arr)
        assert z.dtype == np.dtype("uint16")
        assert z.shape == (2,)

    def test_zarr_array_from_int32_bitwise(self) -> None:
        """Test that zarr.array() works with int32 from bitwise operations."""
        import zarr

        arr = np.array([1, 2], dtype=np.int32) & 1
        z = zarr.array(arr)
        assert z.dtype == np.dtype("int32")
        assert z.shape == (2,)


class TestErrorCases:
    """Test that invalid dtypes still raise appropriate errors."""

    def test_uint32_rejects_wrong_size(self) -> None:
        """Test that UInt32 rejects dtypes with wrong itemsize."""
        # Create a dtype with wrong size - this is artificial, 
        # as numpy doesn't naturally create such dtypes
        arr_correct = np.array([1, 2], dtype=np.uint32)
        arr_wrong = np.array([1, 2], dtype=np.uint16)

        # This should work
        UInt32.from_native_dtype(arr_correct.dtype)

        # This should raise
        with pytest.raises(Exception):  # Could be DataTypeValidationError or ValueError
            UInt32.from_native_dtype(arr_wrong.dtype)

    def test_uint32_rejects_signed_integer(self) -> None:
        """Test that UInt32 rejects signed integer dtypes."""
        arr_signed = np.array([1, 2], dtype=np.int32)

        with pytest.raises(Exception):  # Could be DataTypeValidationError or ValueError
            UInt32.from_native_dtype(arr_signed.dtype)

    def test_int32_rejects_unsigned_integer(self) -> None:
        """Test that Int32 rejects unsigned integer dtypes."""
        arr_unsigned = np.array([1, 2], dtype=np.uint32)

        with pytest.raises(Exception):  # Could be DataTypeValidationError or ValueError
            Int32.from_native_dtype(arr_unsigned.dtype)
