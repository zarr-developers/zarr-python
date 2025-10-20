"""
Tests for the dual-store cache implementation.
"""

import asyncio
import time

import pytest

from zarr.abc.store import Store
from zarr.core.buffer.core import default_buffer_prototype
from zarr.core.buffer.cpu import Buffer as CPUBuffer
from zarr.experimental.cache_store import CacheStore
from zarr.storage import MemoryStore


class TestCacheStore:
    """Test the dual-store cache implementation."""

    @pytest.fixture
    def source_store(self) -> MemoryStore:
        """Create a source store with some test data."""
        return MemoryStore()

    @pytest.fixture
    def cache_store(self) -> MemoryStore:
        """Create an empty cache store."""
        return MemoryStore()

    @pytest.fixture
    def cached_store(self, source_store: Store, cache_store: Store) -> CacheStore:
        """Create a cached store instance."""
        return CacheStore(source_store, cache_store=cache_store, key_insert_times={})

    async def test_basic_caching(self, cached_store: CacheStore, source_store: Store) -> None:
        """Test basic cache functionality."""
        # Store some data
        test_data = CPUBuffer.from_bytes(b"test data")
        await cached_store.set("test_key", test_data)

        # Verify it's in both stores
        assert await source_store.exists("test_key")
        assert await cached_store._cache.exists("test_key")

        # Retrieve and verify caching works
        result = await cached_store.get("test_key", default_buffer_prototype())
        assert result is not None
        assert result.to_bytes() == b"test data"

    async def test_cache_miss_and_population(
        self, cached_store: CacheStore, source_store: Store
    ) -> None:
        """Test cache miss and subsequent population."""
        # Put data directly in source store (bypassing cache)
        test_data = CPUBuffer.from_bytes(b"source data")
        await source_store.set("source_key", test_data)

        # First access should miss cache but populate it
        result = await cached_store.get("source_key", default_buffer_prototype())
        assert result is not None
        assert result.to_bytes() == b"source data"

        # Verify data is now in cache
        assert await cached_store._cache.exists("source_key")

    async def test_cache_expiration(self) -> None:
        """Test cache expiration based on max_age_seconds."""
        source_store = MemoryStore()
        cache_store = MemoryStore()
        cached_store = CacheStore(
            source_store,
            cache_store=cache_store,
            max_age_seconds=1,  # 1 second expiration
            key_insert_times={},
        )

        # Store data
        test_data = CPUBuffer.from_bytes(b"expiring data")
        await cached_store.set("expire_key", test_data)

        # Should be fresh initially (if _is_key_fresh method exists)
        if hasattr(cached_store, "_is_key_fresh"):
            assert cached_store._is_key_fresh("expire_key")

            # Wait for expiration
            await asyncio.sleep(1.1)

            # Should now be stale
            assert not cached_store._is_key_fresh("expire_key")
        else:
            # Skip freshness check if method doesn't exist
            await asyncio.sleep(1.1)
            # Just verify the data is still accessible
            result = await cached_store.get("expire_key", default_buffer_prototype())
            assert result is not None

    async def test_cache_set_data_false(self, source_store: Store, cache_store: Store) -> None:
        """Test behavior when cache_set_data=False."""
        cached_store = CacheStore(
            source_store, cache_store=cache_store, cache_set_data=False, key_insert_times={}
        )

        test_data = CPUBuffer.from_bytes(b"no cache data")
        await cached_store.set("no_cache_key", test_data)

        # Data should be in source but not cache
        assert await source_store.exists("no_cache_key")
        assert not await cache_store.exists("no_cache_key")

    async def test_delete_removes_from_both_stores(self, cached_store: CacheStore) -> None:
        """Test that delete removes from both source and cache."""
        test_data = CPUBuffer.from_bytes(b"delete me")
        await cached_store.set("delete_key", test_data)

        # Verify in both stores
        assert await cached_store._store.exists("delete_key")
        assert await cached_store._cache.exists("delete_key")

        # Delete
        await cached_store.delete("delete_key")

        # Verify removed from both
        assert not await cached_store._store.exists("delete_key")
        assert not await cached_store._cache.exists("delete_key")

    async def test_exists_checks_source_store(
        self, cached_store: CacheStore, source_store: Store
    ) -> None:
        """Test that exists() checks the source store (source of truth)."""
        # Put data directly in source
        test_data = CPUBuffer.from_bytes(b"exists test")
        await source_store.set("exists_key", test_data)

        # Should exist even though not in cache
        assert await cached_store.exists("exists_key")

    async def test_list_operations(self, cached_store: CacheStore, source_store: Store) -> None:
        """Test listing operations delegate to source store."""
        # Add some test data
        test_data = CPUBuffer.from_bytes(b"list test")
        await cached_store.set("list/item1", test_data)
        await cached_store.set("list/item2", test_data)
        await cached_store.set("other/item3", test_data)

        # Test list_dir
        list_items = [key async for key in cached_store.list_dir("list/")]
        assert len(list_items) >= 2  # Should include our items

        # Test list_prefix
        prefix_items = [key async for key in cached_store.list_prefix("list/")]
        assert len(prefix_items) >= 2

    async def test_stale_cache_refresh(self) -> None:
        """Test that stale cache entries are refreshed from source."""
        source_store = MemoryStore()
        cache_store = MemoryStore()
        cached_store = CacheStore(
            source_store, cache_store=cache_store, max_age_seconds=1, key_insert_times={}
        )

        # Store initial data
        old_data = CPUBuffer.from_bytes(b"old data")
        await cached_store.set("refresh_key", old_data)

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Update source store directly (simulating external update)
        new_data = CPUBuffer.from_bytes(b"new data")
        await source_store.set("refresh_key", new_data)

        # Access should refresh from source when cache is stale
        result = await cached_store.get("refresh_key", default_buffer_prototype())
        assert result is not None
        assert result.to_bytes() == b"new data"

    async def test_infinity_max_age(self, cached_store: CacheStore) -> None:
        """Test that 'infinity' max_age means cache never expires."""
        # Skip test if _is_key_fresh method doesn't exist
        if not hasattr(cached_store, "_is_key_fresh"):
            pytest.skip("_is_key_fresh method not implemented")

        test_data = CPUBuffer.from_bytes(b"eternal data")
        await cached_store.set("eternal_key", test_data)

        # Should always be fresh
        assert cached_store._is_key_fresh("eternal_key")

        # Even after time passes
        await asyncio.sleep(0.1)
        assert cached_store._is_key_fresh("eternal_key")

    async def test_cache_returns_cached_data_for_performance(
        self, cached_store: CacheStore, source_store: Store
    ) -> None:
        """Test that cache returns cached data for performance, even if not in source."""
        # Skip test if key_insert_times attribute doesn't exist
        if not hasattr(cached_store, "key_insert_times"):
            pytest.skip("key_insert_times attribute not implemented")

        # Put data in cache but not source (simulates orphaned cache entry)
        test_data = CPUBuffer.from_bytes(b"orphaned data")
        await cached_store._cache.set("orphan_key", test_data)
        cached_store.key_insert_times["orphan_key"] = time.monotonic()

        # Cache should return data for performance (no source verification)
        result = await cached_store.get("orphan_key", default_buffer_prototype())
        assert result is not None
        assert result.to_bytes() == b"orphaned data"

        # Cache entry should remain (performance optimization)
        assert await cached_store._cache.exists("orphan_key")
        assert "orphan_key" in cached_store.key_insert_times

    async def test_cache_coherency_through_expiration(self) -> None:
        """Test that cache coherency is managed through cache expiration, not source verification."""
        source_store = MemoryStore()
        cache_store = MemoryStore()
        cached_store = CacheStore(
            source_store,
            cache_store=cache_store,
            max_age_seconds=1,  # Short expiration for coherency
        )

        # Add data to both stores
        test_data = CPUBuffer.from_bytes(b"original data")
        await cached_store.set("coherency_key", test_data)

        # Remove from source (simulating external deletion)
        await source_store.delete("coherency_key")

        # Cache should still return cached data (performance optimization)
        result = await cached_store.get("coherency_key", default_buffer_prototype())
        assert result is not None
        assert result.to_bytes() == b"original data"

        # Wait for cache expiration
        await asyncio.sleep(1.1)

        # Now stale cache should be refreshed from source
        result = await cached_store.get("coherency_key", default_buffer_prototype())
        assert result is None  # Key no longer exists in source

    async def test_cache_info(self, cached_store: CacheStore) -> None:
        """Test cache_info method returns correct information."""
        # Test initial state
        info = cached_store.cache_info()

        # Check all expected keys are present
        expected_keys = {
            "cache_store_type",
            "max_age_seconds",
            "max_size",
            "current_size",
            "cache_set_data",
            "tracked_keys",
            "cached_keys",
        }
        assert set(info.keys()) == expected_keys

        # Check initial values
        assert info["cache_store_type"] == "MemoryStore"
        assert info["max_age_seconds"] == "infinity"
        assert info["max_size"] is None  # Default unlimited
        assert info["current_size"] == 0
        assert info["cache_set_data"] is True
        assert info["tracked_keys"] == 0
        assert info["cached_keys"] == 0

        # Add some data and verify tracking
        test_data = CPUBuffer.from_bytes(b"test data for cache info")
        await cached_store.set("info_test_key", test_data)

        # Check updated info
        updated_info = cached_store.cache_info()
        assert updated_info["tracked_keys"] == 1
        assert updated_info["cached_keys"] == 1
        assert updated_info["current_size"] > 0  # Should have some size now

    async def test_cache_info_with_max_size(self) -> None:
        """Test cache_info with max_size configuration."""
        source_store = MemoryStore()
        cache_store = MemoryStore()

        # Create cache with specific max_size and max_age
        cached_store = CacheStore(
            source_store,
            cache_store=cache_store,
            max_size=1024,
            max_age_seconds=300,
            key_insert_times={},
        )

        info = cached_store.cache_info()
        assert info["max_size"] == 1024
        assert info["max_age_seconds"] == 300
        assert info["current_size"] == 0

    async def test_clear_cache(self, cached_store: CacheStore) -> None:
        """Test clear_cache method clears all cache data and tracking."""
        # Add some test data
        test_data1 = CPUBuffer.from_bytes(b"test data 1")
        test_data2 = CPUBuffer.from_bytes(b"test data 2")

        await cached_store.set("clear_test_1", test_data1)
        await cached_store.set("clear_test_2", test_data2)

        # Verify data is cached
        info_before = cached_store.cache_info()
        assert info_before["tracked_keys"] == 2
        assert info_before["cached_keys"] == 2
        assert info_before["current_size"] > 0

        # Verify data exists in cache
        assert await cached_store._cache.exists("clear_test_1")
        assert await cached_store._cache.exists("clear_test_2")

        # Clear the cache
        await cached_store.clear_cache()

        # Verify cache is cleared
        info_after = cached_store.cache_info()
        assert info_after["tracked_keys"] == 0
        assert info_after["cached_keys"] == 0
        assert info_after["current_size"] == 0

        # Verify data is removed from cache store (if it supports clear)
        if hasattr(cached_store._cache, "clear"):
            # If cache store supports clear, all data should be gone
            assert not await cached_store._cache.exists("clear_test_1")
            assert not await cached_store._cache.exists("clear_test_2")

        # Verify data still exists in source store
        assert await cached_store._store.exists("clear_test_1")
        assert await cached_store._store.exists("clear_test_2")

    async def test_max_age_infinity(self) -> None:
        """Test cache with infinite max age."""
        source_store = MemoryStore()
        cache_store = MemoryStore()
        cached_store = CacheStore(source_store, cache_store=cache_store, max_age_seconds="infinity")

        # Add data and verify it never expires
        test_data = CPUBuffer.from_bytes(b"test data")
        await cached_store.set("test_key", test_data)

        # Even after time passes, key should be fresh
        assert cached_store._is_key_fresh("test_key")

    async def test_max_age_numeric(self) -> None:
        """Test cache with numeric max age."""
        source_store = MemoryStore()
        cache_store = MemoryStore()
        cached_store = CacheStore(
            source_store,
            cache_store=cache_store,
            max_age_seconds=1,  # 1 second
        )

        # Add data
        test_data = CPUBuffer.from_bytes(b"test data")
        await cached_store.set("test_key", test_data)

        # Key should be fresh initially
        assert cached_store._is_key_fresh("test_key")

        # Manually set old timestamp to test expiration
        cached_store.key_insert_times["test_key"] = time.monotonic() - 2  # 2 seconds ago

        # Key should now be stale
        assert not cached_store._is_key_fresh("test_key")

    async def test_cache_set_data_disabled(self) -> None:
        """Test cache behavior when cache_set_data is False."""
        source_store = MemoryStore()
        cache_store = MemoryStore()
        cached_store = CacheStore(source_store, cache_store=cache_store, cache_set_data=False)

        # Set data
        test_data = CPUBuffer.from_bytes(b"test data")
        await cached_store.set("test_key", test_data)

        # Data should be in source but not in cache
        assert await source_store.exists("test_key")
        assert not await cache_store.exists("test_key")

        # Cache info should show no cached data
        info = cached_store.cache_info()
        assert info["cache_set_data"] is False
        assert info["cached_keys"] == 0

    async def test_eviction_with_max_size(self) -> None:
        """Test LRU eviction when max_size is exceeded."""
        source_store = MemoryStore()
        cache_store = MemoryStore()
        cached_store = CacheStore(
            source_store,
            cache_store=cache_store,
            max_size=100,  # Small cache size
        )

        # Add data that exceeds cache size
        small_data = CPUBuffer.from_bytes(b"a" * 40)  # 40 bytes
        medium_data = CPUBuffer.from_bytes(b"b" * 40)  # 40 bytes
        large_data = CPUBuffer.from_bytes(b"c" * 40)  # 40 bytes (would exceed 100 byte limit)

        # Set first two items
        await cached_store.set("key1", small_data)
        await cached_store.set("key2", medium_data)

        # Cache should have 2 items
        info = cached_store.cache_info()
        assert info["cached_keys"] == 2
        assert info["current_size"] == 80

        # Add third item - should trigger eviction of first item
        await cached_store.set("key3", large_data)

        # Cache should still have items but first one may be evicted
        info = cached_store.cache_info()
        assert info["current_size"] <= 100

    async def test_value_exceeds_max_size(self) -> None:
        """Test behavior when a single value exceeds max_size."""
        source_store = MemoryStore()
        cache_store = MemoryStore()
        cached_store = CacheStore(
            source_store,
            cache_store=cache_store,
            max_size=50,  # Small cache size
        )

        # Try to cache data larger than max_size
        large_data = CPUBuffer.from_bytes(b"x" * 100)  # 100 bytes > 50 byte limit
        await cached_store.set("large_key", large_data)

        # Data should be in source but not cached
        assert await source_store.exists("large_key")
        info = cached_store.cache_info()
        assert info["cached_keys"] == 0
        assert info["current_size"] == 0

    async def test_get_nonexistent_key(self) -> None:
        """Test getting a key that doesn't exist in either store."""
        source_store = MemoryStore()
        cache_store = MemoryStore()
        cached_store = CacheStore(source_store, cache_store=cache_store)

        # Try to get nonexistent key
        result = await cached_store.get("nonexistent", default_buffer_prototype())
        assert result is None

        # Should not create any cache entries
        info = cached_store.cache_info()
        assert info["cached_keys"] == 0

    async def test_delete_both_stores(self) -> None:
        """Test that delete removes from both source and cache stores."""
        source_store = MemoryStore()
        cache_store = MemoryStore()
        cached_store = CacheStore(source_store, cache_store=cache_store)

        # Add data
        test_data = CPUBuffer.from_bytes(b"test data")
        await cached_store.set("test_key", test_data)

        # Verify it's in both stores
        assert await source_store.exists("test_key")
        assert await cache_store.exists("test_key")

        # Delete
        await cached_store.delete("test_key")

        # Verify it's removed from both
        assert not await source_store.exists("test_key")
        assert not await cache_store.exists("test_key")

        # Verify tracking is updated
        info = cached_store.cache_info()
        assert info["cached_keys"] == 0

    async def test_invalid_max_age_seconds(self) -> None:
        """Test that invalid max_age_seconds values raise ValueError."""
        source_store = MemoryStore()
        cache_store = MemoryStore()

        with pytest.raises(ValueError, match="max_age_seconds string value must be 'infinity'"):
            CacheStore(source_store, cache_store=cache_store, max_age_seconds="invalid")

    async def test_unlimited_cache_size(self) -> None:
        """Test behavior when max_size is None (unlimited)."""
        source_store = MemoryStore()
        cache_store = MemoryStore()
        cached_store = CacheStore(
            source_store,
            cache_store=cache_store,
            max_size=None,  # Unlimited cache
        )

        # Add large amounts of data
        for i in range(10):
            large_data = CPUBuffer.from_bytes(b"x" * 1000)  # 1KB each
            await cached_store.set(f"large_key_{i}", large_data)

        # All should be cached since there's no size limit
        info = cached_store.cache_info()
        assert info["cached_keys"] == 10
        assert info["current_size"] == 10000  # 10 * 1000 bytes

    async def test_evict_key_exception_handling(self) -> None:
        """Test exception handling in _evict_key method."""
        source_store = MemoryStore()
        cache_store = MemoryStore()
        cached_store = CacheStore(source_store, cache_store=cache_store, max_size=100)

        # Add some data
        test_data = CPUBuffer.from_bytes(b"test data")
        await cached_store.set("test_key", test_data)

        # Manually corrupt the tracking to trigger exception
        # Remove from one structure but not others to create inconsistency
        del cached_store._cache_order["test_key"]

        # Try to evict - should handle the KeyError gracefully
        await cached_store._evict_key("test_key")

        # Should still work and not crash
        info = cached_store.cache_info()
        assert isinstance(info, dict)

    async def test_get_no_cache_delete_tracking(self) -> None:
        """Test _get_no_cache when key doesn't exist and needs cleanup."""
        source_store = MemoryStore()
        cache_store = MemoryStore()
        cached_store = CacheStore(source_store, cache_store=cache_store)

        # First, add key to cache tracking but not to source
        test_data = CPUBuffer.from_bytes(b"test data")
        await cache_store.set("phantom_key", test_data)
        await cached_store._cache_value("phantom_key", test_data)

        # Verify it's in tracking
        assert "phantom_key" in cached_store._cache_order
        assert "phantom_key" in cached_store.key_insert_times

        # Now try to get it - since it's not in source, should clean up tracking
        result = await cached_store._get_no_cache("phantom_key", default_buffer_prototype())
        assert result is None

        # Should have cleaned up tracking
        assert "phantom_key" not in cached_store._cache_order
        assert "phantom_key" not in cached_store.key_insert_times

    async def test_accommodate_value_no_max_size(self) -> None:
        """Test _accommodate_value early return when max_size is None."""
        source_store = MemoryStore()
        cache_store = MemoryStore()
        cached_store = CacheStore(
            source_store,
            cache_store=cache_store,
            max_size=None,  # No size limit
        )

        # This should return early without doing anything
        await cached_store._accommodate_value(1000000)  # Large value

        # Should not affect anything since max_size is None
        info = cached_store.cache_info()
        assert info["current_size"] == 0

    async def test_concurrent_set_operations(self) -> None:
        """Test that concurrent set operations don't corrupt cache size tracking."""
        source_store = MemoryStore()
        cache_store = MemoryStore()
        cached_store = CacheStore(source_store, cache_store=cache_store, max_size=1000)

        # Create 10 concurrent set operations
        async def set_data(key: str) -> None:
            data = CPUBuffer.from_bytes(b"x" * 50)
            await cached_store.set(key, data)

        # Run concurrently
        await asyncio.gather(*[set_data(f"key_{i}") for i in range(10)])

        info = cached_store.cache_info()
        # Expected: 10 keys * 50 bytes = 500 bytes
        assert info["cached_keys"] == 10
        assert info["current_size"] == 500  # WOULD FAIL due to race condition

    async def test_concurrent_eviction_race(self) -> None:
        """Test concurrent evictions don't corrupt size tracking."""
        source_store = MemoryStore()
        cache_store = MemoryStore()
        cached_store = CacheStore(source_store, cache_store=cache_store, max_size=200)

        # Fill cache to near capacity
        data = CPUBuffer.from_bytes(b"x" * 80)
        await cached_store.set("key1", data)
        await cached_store.set("key2", data)

        # Now trigger two concurrent sets that both need to evict
        async def set_large(key: str) -> None:
            large_data = CPUBuffer.from_bytes(b"y" * 100)
            await cached_store.set(key, large_data)

        await asyncio.gather(set_large("key3"), set_large("key4"))

        info = cached_store.cache_info()
        # Size should be consistent with tracked keys
        assert info["current_size"] <= 200  # Might pass
        # But verify actual cache store size matches tracking
        total_size = sum(cached_store._key_sizes.get(k, 0) for k in cached_store._cache_order)
        assert total_size == info["current_size"]  # WOULD FAIL

    async def test_concurrent_get_and_evict(self) -> None:
        """Test get operations during eviction don't cause corruption."""
        source_store = MemoryStore()
        cache_store = MemoryStore()
        cached_store = CacheStore(source_store, cache_store=cache_store, max_size=100)

        # Setup
        data = CPUBuffer.from_bytes(b"x" * 40)
        await cached_store.set("key1", data)
        await cached_store.set("key2", data)

        # Concurrent: read key1 while adding key3 (triggers eviction)
        async def read_key() -> None:
            for _ in range(100):
                await cached_store.get("key1", default_buffer_prototype())

        async def write_key() -> None:
            for i in range(10):
                new_data = CPUBuffer.from_bytes(b"y" * 40)
                await cached_store.set(f"new_{i}", new_data)

        await asyncio.gather(read_key(), write_key())

        # Verify consistency
        info = cached_store.cache_info()
        assert info["current_size"] <= 100
        assert len(cached_store._cache_order) == len(cached_store._key_sizes)

    async def test_eviction_actually_deletes_from_cache_store(self) -> None:
        """Test that eviction removes keys from cache_store, not just tracking."""
        source_store = MemoryStore()
        cache_store = MemoryStore()
        cached_store = CacheStore(source_store, cache_store=cache_store, max_size=100)

        # Add data that will be evicted
        data1 = CPUBuffer.from_bytes(b"x" * 60)
        data2 = CPUBuffer.from_bytes(b"y" * 60)

        await cached_store.set("key1", data1)

        # Verify key1 is in cache_store
        assert await cache_store.exists("key1")

        # Add key2, which should evict key1
        await cached_store.set("key2", data2)

        # Check tracking - key1 should be removed
        assert "key1" not in cached_store._cache_order
        assert "key1" not in cached_store._key_sizes

        # CRITICAL: key1 should also be removed from cache_store
        assert not await cache_store.exists("key1"), (
            "Evicted key still exists in cache_store! _evict_key doesn't actually delete."
        )

        # But key1 should still exist in source store
        assert await source_store.exists("key1")

    async def test_eviction_no_orphaned_keys(self) -> None:
        """Test that eviction doesn't leave orphaned keys in cache_store."""
        source_store = MemoryStore()
        cache_store = MemoryStore()
        cached_store = CacheStore(source_store, cache_store=cache_store, max_size=150)

        # Add multiple keys that will cause evictions
        for i in range(10):
            data = CPUBuffer.from_bytes(b"x" * 60)
            await cached_store.set(f"key_{i}", data)

        # Check tracking
        info = cached_store.cache_info()
        tracked_keys = info["cached_keys"]

        # Count actual keys in cache_store
        actual_keys = 0
        async for _ in cache_store.list():
            actual_keys += 1

        # Cache store should have same number of keys as tracking
        assert actual_keys == tracked_keys, (
            f"Cache store has {actual_keys} keys but tracking shows {tracked_keys}. "
            f"Eviction doesn't delete from cache_store!"
        )

    async def test_size_accounting_with_key_updates(self) -> None:
        """Test that updating the same key replaces size instead of accumulating."""
        source_store = MemoryStore()
        cache_store = MemoryStore()
        cached_store = CacheStore(source_store, cache_store=cache_store, max_size=500)

        # Set initial value
        data1 = CPUBuffer.from_bytes(b"x" * 100)
        await cached_store.set("same_key", data1)

        info1 = cached_store.cache_info()
        assert info1["current_size"] == 100

        # Update with different size
        data2 = CPUBuffer.from_bytes(b"y" * 200)
        await cached_store.set("same_key", data2)

        info2 = cached_store.cache_info()

        # Should be 200, not 300 (update replaces, doesn't accumulate)
        assert info2["current_size"] == 200, (
            f"Expected size 200 but got {info2['current_size']}. "
            "Updating same key should replace, not accumulate."
        )

    async def test_all_tracked_keys_exist_in_cache_store(self) -> None:
        """Test invariant: all keys in tracking should exist in cache_store."""
        source_store = MemoryStore()
        cache_store = MemoryStore()
        cached_store = CacheStore(source_store, cache_store=cache_store, max_size=500)

        # Add some data
        for i in range(5):
            data = CPUBuffer.from_bytes(b"x" * 50)
            await cached_store.set(f"key_{i}", data)

        # Every key in tracking should exist in cache_store
        for key in cached_store._cache_order:
            assert await cache_store.exists(key), (
                f"Key '{key}' is tracked but doesn't exist in cache_store"
            )

        # Every key in _key_sizes should exist in cache_store
        for key in cached_store._key_sizes:
            assert await cache_store.exists(key), (
                f"Key '{key}' has size tracked but doesn't exist in cache_store"
            )

    # Additional coverage tests for 100% coverage

    async def test_cache_store_requires_delete_support(self) -> None:
        """Test that CacheStore validates cache_store supports deletes."""
        from unittest.mock import MagicMock

        # Create a mock store that doesn't support deletes
        source_store = MemoryStore()
        cache_store = MagicMock()
        cache_store.supports_deletes = False

        with pytest.raises(ValueError, match="does not support deletes"):
            CacheStore(store=source_store, cache_store=cache_store)

    async def test_evict_key_exception_handling_with_real_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test _evict_key exception handling when deletion fails."""
        source_store = MemoryStore()
        cache_store = MemoryStore()
        cached_store = CacheStore(store=source_store, cache_store=cache_store, max_size=100)

        # Set up a key in tracking
        buffer = CPUBuffer.from_bytes(b"test data")
        await cached_store.set("test_key", buffer)

        # Mock the cache delete to raise an exception
        async def failing_delete(key: str) -> None:
            raise RuntimeError("Simulated cache deletion failure")

        monkeypatch.setattr(cache_store, "delete", failing_delete)

        # Attempt to evict should raise the exception
        with pytest.raises(RuntimeError, match="Simulated cache deletion failure"):
            async with cached_store._lock:
                await cached_store._evict_key("test_key")

    async def test_cache_stats_method(self) -> None:
        """Test cache_stats method returns correct statistics."""
        source_store = MemoryStore()
        cache_store = MemoryStore()
        cached_store = CacheStore(store=source_store, cache_store=cache_store, max_size=1000)

        # Initially, stats should be zero
        stats = cached_store.cache_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["evictions"] == 0
        assert stats["total_requests"] == 0
        assert stats["hit_rate"] == 0.0

        # Perform some operations
        buffer = CPUBuffer.from_bytes(b"x" * 100)

        # Write to source store directly to avoid affecting stats
        await source_store.set("key1", buffer)

        # First get is a miss (not in cache yet)
        result1 = await cached_store.get("key1", default_buffer_prototype())
        assert result1 is not None

        # Second get is a hit (now in cache)
        result2 = await cached_store.get("key1", default_buffer_prototype())
        assert result2 is not None

        stats = cached_store.cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["total_requests"] == 2
        assert stats["hit_rate"] == 0.5

    async def test_cache_stats_with_evictions(self) -> None:
        """Test cache_stats tracks evictions correctly."""
        source_store = MemoryStore()
        cache_store = MemoryStore()
        cached_store = CacheStore(
            store=source_store,
            cache_store=cache_store,
            max_size=150,  # Small size to force eviction
        )

        # Add items that will trigger eviction
        buffer1 = CPUBuffer.from_bytes(b"x" * 100)
        buffer2 = CPUBuffer.from_bytes(b"y" * 100)

        await cached_store.set("key1", buffer1)
        await cached_store.set("key2", buffer2)  # Should evict key1

        stats = cached_store.cache_stats()
        assert stats["evictions"] == 1

    def test_repr_method(self) -> None:
        """Test __repr__ returns useful string representation."""
        source_store = MemoryStore()
        cache_store = MemoryStore()
        cached_store = CacheStore(
            store=source_store, cache_store=cache_store, max_age_seconds=60, max_size=1024
        )

        repr_str = repr(cached_store)

        # Check that repr contains key information
        assert "CacheStore" in repr_str
        assert "max_age_seconds=60" in repr_str
        assert "max_size=1024" in repr_str
        assert "current_size=0" in repr_str
        assert "cached_keys=0" in repr_str

    async def test_cache_stats_zero_division_protection(self) -> None:
        """Test cache_stats handles zero requests correctly."""
        source_store = MemoryStore()
        cache_store = MemoryStore()
        cached_store = CacheStore(store=source_store, cache_store=cache_store)

        # With no requests, hit_rate should be 0.0 (not NaN or error)
        stats = cached_store.cache_stats()
        assert stats["hit_rate"] == 0.0
        assert stats["total_requests"] == 0
