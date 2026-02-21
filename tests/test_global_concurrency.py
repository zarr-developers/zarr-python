"""
Tests for store-level concurrency limiting through the array API.
"""

import numpy as np

import zarr


class TestStoreConcurrencyThroughArrayAPI:
    """Tests that store-level concurrency limiting works through the array API."""

    def test_array_operations_with_store_concurrency(self) -> None:
        """Test that array read/write works correctly with store-level concurrency limits."""
        store = zarr.storage.MemoryStore()
        arr = zarr.create(
            shape=(20, 20),
            chunks=(10, 10),
            dtype="i4",
            store=store,
            zarr_format=3,
        )
        arr[:] = 42

        data = arr[:]

        assert np.all(data == 42)

    def test_array_operations_with_local_store_concurrency(self, tmp_path: object) -> None:
        """Test that array read/write works correctly with LocalStore concurrency limits."""
        store = zarr.storage.LocalStore(str(tmp_path), concurrency_limit=10)
        arr = zarr.create(
            shape=(20, 20),
            chunks=(10, 10),
            dtype="i4",
            store=store,
            zarr_format=3,
        )
        arr[:] = 42

        data = arr[:]

        assert np.all(data == 42)
