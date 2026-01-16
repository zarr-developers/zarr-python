"""Base test class for store concurrency limiting behavior."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Generic, TypeVar

import pytest

from zarr.core.buffer import Buffer, default_buffer_prototype

if TYPE_CHECKING:
    from zarr.abc.store import Store

__all__ = ["StoreConcurrencyTests"]


S = TypeVar("S", bound="Store")
B = TypeVar("B", bound="Buffer")


class StoreConcurrencyTests(Generic[S, B]):
    """Base class for testing store concurrency limiting behavior.

    This mixin provides tests for verifying that stores correctly implement
    concurrency limiting.

    Subclasses should set:
    - store_cls: The store class being tested
    - buffer_cls: The buffer class to use (e.g., cpu.Buffer)
    - expected_concurrency_limit: Expected default concurrency limit (or None for unlimited)
    """

    store_cls: type[S]
    buffer_cls: type[B]
    expected_concurrency_limit: int | None

    @pytest.fixture
    async def store(self, store_kwargs: dict) -> S:
        """Create and open a store instance."""
        return await self.store_cls.open(**store_kwargs)

    def test_concurrency_limit_default(self, store: S) -> None:
        """Test that store has the expected default concurrency limit."""
        if hasattr(store, "_semaphore"):
            if self.expected_concurrency_limit is None:
                assert store._semaphore is None, "Expected no concurrency limit"
            else:
                assert store._semaphore is not None, "Expected concurrency limit to be set"
                assert store._semaphore._value == self.expected_concurrency_limit, (
                    f"Expected limit {self.expected_concurrency_limit}, got {store._semaphore._value}"
                )

    def test_concurrency_limit_custom(self, store_kwargs: dict) -> None:
        """Test that custom concurrency limits can be set."""
        if "concurrency_limit" not in self.store_cls.__init__.__code__.co_varnames:
            pytest.skip("Store does not support custom concurrency limits")

        # Test with custom limit
        store = self.store_cls(**store_kwargs, concurrency_limit=42)
        if hasattr(store, "_semaphore"):
            assert store._semaphore is not None
            assert store._semaphore._value == 42

        # Test with None (unlimited)
        store = self.store_cls(**store_kwargs, concurrency_limit=None)
        if hasattr(store, "_semaphore"):
            assert store._semaphore is None

    async def test_concurrency_limit_enforced(self, store: S) -> None:
        """Test that the concurrency limit is actually enforced during execution.

        This test verifies that when many operations are submitted concurrently,
        only up to the concurrency limit are actually executing at once.
        """
        if not hasattr(store, "_semaphore") or store._semaphore is None:
            pytest.skip("Store has no concurrency limit")

        limit = store._semaphore._value

        # We'll monitor the semaphore's available count
        # When it reaches 0, that means `limit` operations are running
        min_available = limit

        async def monitored_operation(key: str, value: B) -> None:
            nonlocal min_available
            # Check semaphore state right after we're scheduled
            await asyncio.sleep(0)  # Yield to ensure we're in the queue
            available = store._semaphore._value
            min_available = min(min_available, available)

            # Now do the actual operation (which will acquire the semaphore)
            await store.set(key, value)

        # Launch more operations than the limit to ensure contention
        num_ops = limit * 2
        items = [
            (f"limit_test_key_{i}", self.buffer_cls.from_bytes(f"value_{i}".encode()))
            for i in range(num_ops)
        ]

        await asyncio.gather(*[monitored_operation(k, v) for k, v in items])

        # The semaphore should have been fully utilized (reached 0 or close to it)
        # This indicates that `limit` operations were running concurrently
        assert min_available < limit, (
            f"Semaphore was never fully utilized. "
            f"Min available: {min_available}, Limit: {limit}. "
            f"This suggests operations aren't running concurrently."
        )

        # Ideally it should reach 0, but allow some slack for timing
        assert min_available <= 5, (
            f"Semaphore only reached {min_available} available slots. "
            f"Expected close to 0 with limit {limit}."
        )

    async def test_batch_write_no_deadlock(self, store: S) -> None:
        """Test that batch writes don't deadlock when exceeding concurrency limit."""
        # Create more items than any reasonable concurrency limit
        num_items = 200
        items = [
            (f"test_key_{i}", self.buffer_cls.from_bytes(f"test_value_{i}".encode()))
            for i in range(num_items)
        ]

        # This should complete without deadlock, even if num_items > concurrency_limit
        await asyncio.wait_for(store._set_many(items), timeout=30.0)

        # Verify all items were written correctly
        for key, expected_value in items:
            result = await store.get(key, default_buffer_prototype())
            assert result is not None
            assert result.to_bytes() == expected_value.to_bytes()

    async def test_batch_read_no_deadlock(self, store: S) -> None:
        """Test that batch reads don't deadlock when exceeding concurrency limit."""
        # Write test data
        num_items = 200
        test_data = {
            f"test_key_{i}": self.buffer_cls.from_bytes(f"test_value_{i}".encode())
            for i in range(num_items)
        }

        for key, value in test_data.items():
            await store.set(key, value)

        # Read all items concurrently - should not deadlock
        keys_and_ranges = [(key, None) for key in test_data]
        results = await asyncio.wait_for(
            store.get_partial_values(default_buffer_prototype(), keys_and_ranges),
            timeout=30.0,
        )

        # Verify results
        assert len(results) == num_items
        for result, (key, expected_value) in zip(results, test_data.items()):
            assert result is not None
            assert result.to_bytes() == expected_value.to_bytes()

    async def test_batch_delete_no_deadlock(self, store: S) -> None:
        """Test that batch deletes don't deadlock when exceeding concurrency limit."""
        if not store.supports_deletes:
            pytest.skip("Store does not support deletes")

        # Write test data
        num_items = 200
        keys = [f"test_key_{i}" for i in range(num_items)]
        for key in keys:
            await store.set(key, self.buffer_cls.from_bytes(b"test_value"))

        # Delete all items concurrently - should not deadlock
        await asyncio.wait_for(asyncio.gather(*[store.delete(key) for key in keys]), timeout=30.0)

        # Verify all items were deleted
        for key in keys:
            result = await store.get(key, default_buffer_prototype())
            assert result is None

    async def test_concurrent_operations_correctness(self, store: S) -> None:
        """Test that concurrent operations produce correct results."""
        num_operations = 100

        # Mix of reads and writes
        write_keys = [f"write_key_{i}" for i in range(num_operations)]
        write_values = [
            self.buffer_cls.from_bytes(f"value_{i}".encode()) for i in range(num_operations)
        ]

        # Write all concurrently
        await asyncio.gather(*[store.set(k, v) for k, v in zip(write_keys, write_values)])

        # Read all concurrently
        results = await asyncio.gather(
            *[store.get(k, default_buffer_prototype()) for k in write_keys]
        )

        # Verify correctness
        for result, expected in zip(results, write_values):
            assert result is not None
            assert result.to_bytes() == expected.to_bytes()

    @pytest.mark.parametrize("batch_size", [1, 10, 50, 100])
    async def test_various_batch_sizes(self, store: S, batch_size: int) -> None:
        """Test that various batch sizes work correctly."""
        items = [
            (f"batch_key_{i}", self.buffer_cls.from_bytes(f"batch_value_{i}".encode()))
            for i in range(batch_size)
        ]

        # Should complete without issues for any batch size
        await asyncio.wait_for(store._set_many(items), timeout=10.0)

        # Verify
        for key, expected_value in items:
            result = await store.get(key, default_buffer_prototype())
            assert result is not None
            assert result.to_bytes() == expected_value.to_bytes()

    async def test_empty_batch_operations(self, store: S) -> None:
        """Test that empty batch operations don't cause issues."""
        # Empty batch should not raise
        await store._set_many([])

        # Empty read batch
        results = await store.get_partial_values(default_buffer_prototype(), [])
        assert results == []

    async def test_mixed_success_failure_batch(self, store: S) -> None:
        """Test batch operations with mix of successful and failing items."""
        # Write some initial data
        await store.set("existing_key", self.buffer_cls.from_bytes(b"existing_value"))

        # Try to read mix of existing and non-existing keys
        key_ranges = [
            ("existing_key", None),
            ("non_existing_key_1", None),
            ("non_existing_key_2", None),
        ]

        results = await store.get_partial_values(default_buffer_prototype(), key_ranges)

        # First should exist, others should be None
        assert results[0] is not None
        assert results[0].to_bytes() == b"existing_value"
        assert results[1] is None
        assert results[2] is None
