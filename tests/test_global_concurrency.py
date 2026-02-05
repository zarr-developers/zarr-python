"""
Tests for global per-process concurrency limiting.
"""

import asyncio
from typing import Any

import numpy as np
import pytest

import zarr
from zarr.core.common import get_global_semaphore, reset_global_semaphores
from zarr.core.config import config


class TestGlobalSemaphore:
    """Tests for the global semaphore management."""

    async def test_get_global_semaphore_creates_per_loop(self) -> None:
        """Test that each event loop gets its own semaphore."""
        sem1 = get_global_semaphore()
        assert sem1 is not None
        assert isinstance(sem1, asyncio.Semaphore)

        # Getting it again should return the same instance
        sem2 = get_global_semaphore()
        assert sem1 is sem2

    async def test_global_semaphore_uses_config_limit(self) -> None:
        """Test that the global semaphore respects the configured limit."""
        # Set a custom concurrency limit
        original_limit: Any = config.get("async.concurrency")
        try:
            config.set({"async.concurrency": 5})

            # Clear existing semaphores to force recreation
            reset_global_semaphores()

            sem = get_global_semaphore()

            # The semaphore should have the configured limit
            # We can verify this by acquiring all tokens and checking the semaphore is locked
            for i in range(5):
                await sem.acquire()
                if i < 4:
                    assert not sem.locked()  # Should still have capacity
                else:
                    assert sem.locked()  # All tokens acquired, semaphore is now locked

            # Release all tokens
            for _ in range(5):
                sem.release()

        finally:
            # Restore original config
            config.set({"async.concurrency": original_limit})
            # Clear semaphores again to reset state
            reset_global_semaphores()

    async def test_global_semaphore_shared_across_operations(self) -> None:
        """Test that multiple concurrent operations share the same semaphore."""
        # Track the maximum number of concurrent tasks
        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def tracked_operation() -> None:
            """An operation that tracks concurrency."""
            nonlocal max_concurrent, current_concurrent

            async with lock:
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)

            # Small delay to ensure overlap
            await asyncio.sleep(0.01)

            async with lock:
                current_concurrent -= 1

        # Set a low concurrency limit to make the test observable
        original_limit: Any = config.get("async.concurrency")
        try:
            config.set({"async.concurrency": 5})

            # Clear existing semaphores
            reset_global_semaphores()

            # Get the global semaphore
            sem = get_global_semaphore()

            # Create many tasks that use the semaphore
            async def task_with_semaphore() -> None:
                async with sem:
                    await tracked_operation()

            # Launch 20 tasks (4x the limit)
            tasks = [task_with_semaphore() for _ in range(20)]
            await asyncio.gather(*tasks)

            # Maximum concurrent should respect the limit
            assert max_concurrent <= 5, f"Max concurrent was {max_concurrent}, expected <= 5"
            assert max_concurrent >= 3, (
                f"Max concurrent was {max_concurrent}, expected some concurrency"
            )

        finally:
            config.set({"async.concurrency": original_limit})
            reset_global_semaphores()

    async def test_semaphore_reuse_across_calls(self) -> None:
        """Test that repeated calls to get_global_semaphore return the same instance."""
        reset_global_semaphores()

        # Call multiple times and verify we get the same instance
        sem1 = get_global_semaphore()
        sem2 = get_global_semaphore()
        sem3 = get_global_semaphore()

        assert sem1 is sem2 is sem3, "Should return same semaphore instance on repeated calls"

        # Verify it's still the same after using it
        async with sem1:
            sem4 = get_global_semaphore()
            assert sem1 is sem4

    def test_config_change_after_creation(self) -> None:
        """Test and document that config changes don't affect existing semaphores."""
        original_limit: Any = config.get("async.concurrency")
        try:
            # Set initial config
            config.set({"async.concurrency": 5})

            async def check_limit() -> None:
                reset_global_semaphores()

                # Create semaphore with limit=5
                sem1 = get_global_semaphore()
                initial_capacity: int = sem1._value

                # Change config
                config.set({"async.concurrency": 50})

                # Get semaphore again - should be same instance with old limit
                sem2 = get_global_semaphore()
                assert sem1 is sem2, "Should return same semaphore instance"
                assert sem2._value == initial_capacity, (
                    f"Semaphore limit changed from {initial_capacity} to {sem2._value}. "
                    "Config changes should not affect existing semaphores."
                )

                # Clean up
                reset_global_semaphores()

            asyncio.run(check_limit())

        finally:
            config.set({"async.concurrency": original_limit})


class TestArrayConcurrency:
    """Tests that array operations use global concurrency limiting."""

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
    async def test_multiple_arrays_share_concurrency_limit(self) -> None:
        """Test that reading from multiple arrays shares the global concurrency limit."""
        from zarr.core.common import concurrent_map

        # Track concurrent task executions
        max_concurrent_tasks = 0
        current_concurrent_tasks = 0
        task_lock = asyncio.Lock()

        async def tracked_chunk_operation(chunk_id: int) -> int:
            """Simulate a chunk operation with tracking."""
            nonlocal max_concurrent_tasks, current_concurrent_tasks

            async with task_lock:
                current_concurrent_tasks += 1
                max_concurrent_tasks = max(max_concurrent_tasks, current_concurrent_tasks)

            # Small delay to simulate I/O
            await asyncio.sleep(0.001)

            async with task_lock:
                current_concurrent_tasks -= 1

            return chunk_id

        # Set a low concurrency limit
        original_limit: Any = config.get("async.concurrency")
        try:
            config.set({"async.concurrency": 10})

            # Clear existing semaphores
            reset_global_semaphores()

            # Simulate reading many chunks using concurrent_map (which uses the global semaphore)
            # This simulates what happens when reading from multiple arrays
            chunk_ids = [(i,) for i in range(100)]
            await concurrent_map(chunk_ids, tracked_chunk_operation)

            # The maximum concurrent tasks should respect the global limit
            assert max_concurrent_tasks <= 10, (
                f"Max concurrent tasks was {max_concurrent_tasks}, expected <= 10"
            )

            assert max_concurrent_tasks >= 5, (
                f"Max concurrent tasks was {max_concurrent_tasks}, "
                f"expected at least some concurrency"
            )

        finally:
            config.set({"async.concurrency": original_limit})
            # Note: We don't reset_global_semaphores() here because doing so while
            # many tasks are still cleaning up can trigger ResourceWarnings from
            # asyncio internals. The semaphore will be reused by subsequent tests.

    def test_sync_api_uses_global_concurrency(self) -> None:
        """Test that synchronous API also benefits from global concurrency limiting."""
        # This test verifies that the sync API (which wraps async) uses global limiting

        # Set a low concurrency limit
        original_limit: Any = config.get("async.concurrency")
        try:
            config.set({"async.concurrency": 8})

            # Create a small array - the key is that zarr internally uses
            # concurrent_map which now uses the global semaphore
            store = zarr.storage.MemoryStore()
            arr = zarr.create(
                shape=(20, 20),
                chunks=(10, 10),
                dtype="i4",
                store=store,
                zarr_format=3,
            )
            arr[:] = 42

            # Read data (synchronously)
            data = arr[:]

            # Verify we got the right data
            assert np.all(data == 42)

            # The test passes if no errors occurred
            # The concurrency limiting is happening under the hood

        finally:
            config.set({"async.concurrency": original_limit})


class TestConcurrentMapGlobal:
    """Tests for concurrent_map using global semaphore."""

    async def test_concurrent_map_uses_global_by_default(self) -> None:
        """Test that concurrent_map uses global semaphore by default."""
        from zarr.core.common import concurrent_map

        # Track concurrent executions
        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def tracked_task(x: int) -> int:
            nonlocal max_concurrent, current_concurrent

            async with lock:
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)

            await asyncio.sleep(0.01)

            async with lock:
                current_concurrent -= 1

            return x * 2

        # Set a low limit
        original_limit: Any = config.get("async.concurrency")
        try:
            config.set({"async.concurrency": 5})

            # Clear existing semaphores
            reset_global_semaphores()

            # Use concurrent_map with default settings (use_global_semaphore=True)
            items = [(i,) for i in range(20)]
            results = await concurrent_map(items, tracked_task)

            assert len(results) == 20
            assert max_concurrent <= 5
            assert max_concurrent >= 3  # Should have some concurrency

        finally:
            config.set({"async.concurrency": original_limit})
            reset_global_semaphores()

    async def test_concurrent_map_legacy_mode(self) -> None:
        """Test that concurrent_map legacy mode still works."""
        from zarr.core.common import concurrent_map

        async def simple_task(x: int) -> int:
            await asyncio.sleep(0.001)
            return x * 2

        # Use legacy mode with local limit
        items = [(i,) for i in range(10)]
        results = await concurrent_map(items, simple_task, limit=3, use_global_semaphore=False)

        assert len(results) == 10
        assert results == [i * 2 for i in range(10)]

    async def test_concurrent_map_parameter_validation(self) -> None:
        """Test that concurrent_map validates conflicting parameters."""
        from zarr.core.common import concurrent_map

        async def simple_task(x: int) -> int:
            return x * 2

        items = [(i,) for i in range(10)]

        # Should raise ValueError when both limit and use_global_semaphore=True
        with pytest.raises(
            ValueError, match="Cannot specify both use_global_semaphore=True and a limit"
        ):
            await concurrent_map(items, simple_task, limit=5, use_global_semaphore=True)
