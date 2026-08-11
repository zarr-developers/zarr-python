"""
Tests for the dual-store cache implementation.
"""

import asyncio
import random
import time

import pytest

from zarr.abc.store import ByteRequest, RangeByteRequest, Store, SuffixByteRequest
from zarr.core.buffer.core import Buffer, BufferPrototype, default_buffer_prototype
from zarr.core.buffer.cpu import Buffer as CPUBuffer
from zarr.experimental.cache_store import CacheStore, _Entry, _KeyState, _RangeEntry
from zarr.storage import MemoryStore
from zarr.storage._wrapper import WrapperStore


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
        return CacheStore(source_store, cache_store=cache_store)

    async def test_with_read_only_round_trip(self) -> None:
        """
        Ensure that CacheStore.with_read_only returns another CacheStore with
        the requested read_only state, shares cache state, and does not change
        the original store's read_only flag.
        """
        source = MemoryStore()
        cache = MemoryStore()

        # Start from a read-only underlying store
        source_ro = source.with_read_only(read_only=True)
        cached_ro = CacheStore(store=source_ro, cache_store=cache)
        assert cached_ro.read_only

        buf = CPUBuffer.from_bytes(b"0123")

        # Cannot write through the read-only cache store
        with pytest.raises(
            ValueError, match="store was opened in read-only mode and does not support writing"
        ):
            await cached_ro.set("foo", buf)

        # Create a writable cache store from the read-only one
        writer = cached_ro.with_read_only(read_only=False)
        assert isinstance(writer, CacheStore)
        assert not writer.read_only

        # Cache configuration and state are shared
        assert writer._cache is cached_ro._cache
        assert writer._state is cached_ro._state
        assert writer._state.entries is cached_ro._state.entries

        # Writes via the writable cache store succeed and are cached
        await writer.set("foo", buf)
        out = await writer.get("foo", default_buffer_prototype())
        assert out is not None
        assert out.to_bytes() == buf.to_bytes()

        # The original cache store remains read-only
        assert cached_ro.read_only
        with pytest.raises(
            ValueError, match="store was opened in read-only mode and does not support writing"
        ):
            await cached_ro.set("bar", buf)

        # Creating a read-only copy from the writable cache store works and is enforced
        reader = writer.with_read_only(read_only=True)
        assert isinstance(reader, CacheStore)
        assert reader.read_only
        with pytest.raises(
            ValueError, match="store was opened in read-only mode and does not support writing"
        ):
            await reader.set("baz", buf)

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
        )

        # Store data
        test_data = CPUBuffer.from_bytes(b"expiring data")
        await cached_store.set("expire_key", test_data)

        # Should be fresh initially
        assert cached_store._is_fresh("expire_key")

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Should now be stale
        assert not cached_store._is_fresh("expire_key")

    async def test_cache_set_data_false(self, source_store: Store, cache_store: Store) -> None:
        """Test behavior when cache_set_data=False."""
        cached_store = CacheStore(source_store, cache_store=cache_store, cache_set_data=False)

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
        cached_store = CacheStore(source_store, cache_store=cache_store, max_age_seconds=1)

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
        test_data = CPUBuffer.from_bytes(b"eternal data")
        await cached_store.set("eternal_key", test_data)

        # Should always be fresh
        assert cached_store._is_fresh("eternal_key")

        # Even after time passes
        await asyncio.sleep(0.1)
        assert cached_store._is_fresh("eternal_key")

    async def test_cache_returns_cached_data_for_performance(
        self, cached_store: CacheStore, source_store: Store
    ) -> None:
        """Test that cache returns cached data for performance, even if not in source."""
        # Put data in cache but not source (simulates orphaned cache entry)
        test_data = CPUBuffer.from_bytes(b"orphaned data")
        await cached_store._cache.set("orphan_key", test_data)
        cached_store._state.entries["orphan_key"] = _KeyState(
            full=_Entry(insert_time=time.monotonic(), size=len(test_data), present=True)
        )

        # Cache should return data for performance (no source verification)
        result = await cached_store.get("orphan_key", default_buffer_prototype())
        assert result is not None
        assert result.to_bytes() == b"orphaned data"

        # Cache entry should remain (performance optimization)
        assert await cached_store._cache.exists("orphan_key")
        assert "orphan_key" in cached_store._state.entries

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
            "cache_missing",
            "tracked_keys",
            "cached_keys",
            "missing_keys",
        }
        assert set(info.keys()) == expected_keys

        # Check initial values
        assert info["cache_store_type"] == "MemoryStore"
        assert info["max_age_seconds"] == 300  # the default: finite, bounded staleness
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
        assert cached_store._is_fresh("test_key")

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
        assert cached_store._is_fresh("test_key")

        # Manually set old timestamp to test expiration
        slot = cached_store._state.entries["test_key"].full
        assert slot is not None
        slot.insert_time = time.monotonic() - 2  # 2 seconds ago

        # Key should now be stale
        assert not cached_store._is_fresh("test_key")

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

    async def test_evict_slot_exception_handling(self) -> None:
        """Test exception handling in _evict_slot method."""
        source_store = MemoryStore()
        cache_store = MemoryStore()
        cached_store = CacheStore(source_store, cache_store=cache_store, max_size=100)

        # Add some data
        test_data = CPUBuffer.from_bytes(b"test data")
        await cached_store.set("test_key", test_data)

        # Manually corrupt the tracking to trigger exception
        # Remove the tracked entry while leaving the cached value behind
        del cached_store._state.entries["test_key"]

        # Try to evict - should handle the missing record gracefully
        await cached_store._evict_slot("test_key")

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
        async with cached_store._state.lock:
            await cached_store._track_entry("phantom_key", test_data)

        # Verify it's in tracking
        assert "phantom_key" in cached_store._state.entries

        # Now try to get it - since it's not in source, should clean up tracking
        result = await cached_store._get_no_cache("phantom_key", default_buffer_prototype())
        assert result is None

        # Should have cleaned up tracking (the positive entry is gone). With
        # cache_missing on by default, a negative marker replaces it.
        state = cached_store._state.entries.get("phantom_key")
        assert state is None or (state.full is not None and not state.full.present)

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
        total_size = sum(state.tracked_size for state in cached_store._state.entries.values())
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
        # Tracked size accounting stays consistent with all entries (present
        # values plus any negative markers, which each carry a flat overhead).
        total_size = sum(state.tracked_size for state in cached_store._state.entries.values())
        assert total_size == info["current_size"]

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
        assert "key1" not in cached_store._state.entries

        # CRITICAL: key1 should also be removed from cache_store
        assert not await cache_store.exists("key1"), (
            "Evicted key still exists in cache_store! _evict_slot doesn't actually delete."
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

        # Every key tracking a present full value should exist in cache_store.
        # (Byte-range entries are stored in-memory inside the record, not in the
        # Store; absent markers have no stored value.)
        for key, state in cached_store._state.entries.items():
            if state.full is not None and state.full.present:
                assert await cache_store.exists(key), (
                    f"Key '{key}' is tracked but doesn't exist in cache_store"
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

    async def test_evict_slot_exception_handling_with_real_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test _evict_slot exception handling when deletion fails."""
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
            async with cached_store._state.lock:
                await cached_store._evict_slot("test_key")

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

    async def test_byte_range_does_not_corrupt_cache(self) -> None:
        """Test that fetching a byte range does not store partial data under the full key.

        Reproduces https://github.com/zarr-developers/zarr-python/issues/3690:
        when a byte-range read populates the cache, subsequent reads of different
        ranges (or the full key) return wrong data.
        """
        source_store = MemoryStore()
        cache_store = MemoryStore()
        cached_store = CacheStore(store=source_store, cache_store=cache_store)

        full_data = b"bar baz"
        await source_store.set("foo", CPUBuffer.from_bytes(full_data))

        proto = default_buffer_prototype()

        # First read: byte range [0, 3) -> b"bar"
        bar = await cached_store.get("foo", proto, byte_range=RangeByteRequest(0, 3))
        assert bar is not None
        assert bar.to_bytes() == b"bar"

        # Second read: different byte range [4, 7) -> b"baz"
        baz = await cached_store.get("foo", proto, byte_range=RangeByteRequest(4, 7))
        assert baz is not None
        assert baz.to_bytes() == b"baz"

        # Third read: full key -> full data
        full = await cached_store.get("foo", proto)
        assert full is not None
        assert full.to_bytes() == full_data

    async def test_full_read_then_byte_range(self) -> None:
        """Test that a cached full read correctly serves subsequent byte-range requests."""
        source_store = MemoryStore()
        cache_store = MemoryStore()
        cached_store = CacheStore(store=source_store, cache_store=cache_store)

        full_data = b"hello world"
        await source_store.set("key", CPUBuffer.from_bytes(full_data))

        proto = default_buffer_prototype()

        # Full read populates cache
        full = await cached_store.get("key", proto)
        assert full is not None
        assert full.to_bytes() == full_data

        # Byte-range reads should return the correct slices
        part = await cached_store.get("key", proto, byte_range=RangeByteRequest(0, 5))
        assert part is not None
        assert part.to_bytes() == b"hello"

        part2 = await cached_store.get("key", proto, byte_range=RangeByteRequest(6, 11))
        assert part2 is not None
        assert part2.to_bytes() == b"world"

        suffix = await cached_store.get("key", proto, byte_range=SuffixByteRequest(5))
        assert suffix is not None
        assert suffix.to_bytes() == b"world"

    async def test_byte_range_set_then_read(self) -> None:
        """Test that data written via set() can be read back with byte ranges."""
        source_store = MemoryStore()
        cache_store = MemoryStore()
        cached_store = CacheStore(store=source_store, cache_store=cache_store)

        full_data = b"abcdefghij"
        await cached_store.set("key", CPUBuffer.from_bytes(full_data))

        proto = default_buffer_prototype()

        # Byte-range reads from the cached data
        mid = await cached_store.get("key", proto, byte_range=RangeByteRequest(3, 7))
        assert mid is not None
        assert mid.to_bytes() == b"defg"

        # Full read should still work
        full = await cached_store.get("key", proto)
        assert full is not None
        assert full.to_bytes() == full_data

    async def test_set_invalidates_cached_byte_ranges(self) -> None:
        """Test that set() invalidates previously cached byte-range entries."""
        source_store = MemoryStore()
        cache_store = MemoryStore()
        cached_store = CacheStore(store=source_store, cache_store=cache_store)

        proto = default_buffer_prototype()

        # Populate source and cache some byte ranges
        await source_store.set("key", CPUBuffer.from_bytes(b"old data!!"))
        r1 = await cached_store.get("key", proto, byte_range=RangeByteRequest(0, 3))
        assert r1 is not None
        assert r1.to_bytes() == b"old"

        # Byte-range entry should be tracked inside the key's record
        assert RangeByteRequest(0, 3) in cached_store._state.entries["key"].ranges

        # Overwrite via set() — range entries must be invalidated
        await cached_store.set("key", CPUBuffer.from_bytes(b"NEW DATA!!"))

        # The old range entry should be gone from the key's record
        assert not cached_store._state.entries["key"].ranges

        # A fresh byte-range read should return the new data
        r2 = await cached_store.get("key", proto, byte_range=RangeByteRequest(0, 3))
        assert r2 is not None
        assert r2.to_bytes() == b"NEW"

    async def test_delete_invalidates_cached_byte_ranges(self) -> None:
        """Test that delete() removes previously cached byte-range entries."""
        source_store = MemoryStore()
        cache_store = MemoryStore()
        cached_store = CacheStore(store=source_store, cache_store=cache_store)

        proto = default_buffer_prototype()

        # Populate and cache a byte range
        await source_store.set("key", CPUBuffer.from_bytes(b"hello world"))
        r = await cached_store.get("key", proto, byte_range=RangeByteRequest(0, 5))
        assert r is not None
        assert r.to_bytes() == b"hello"

        assert RangeByteRequest(0, 5) in cached_store._state.entries["key"].ranges

        # Delete the key — the whole record must be cleaned up
        await cached_store.delete("key")

        assert "key" not in cached_store._state.entries

        # Key is gone from source
        result = await cached_store.get("key", proto)
        assert result is None


class TestCacheStoreNegativeCaching:
    """Tests for opt-in negative (missing-key) caching (``cache_missing=True``)."""

    async def test_basic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A second get of an absent key is served from the negative cache without a
        source round-trip."""
        source = MemoryStore()
        cs = CacheStore(source, cache_store=MemoryStore(), cache_missing=True)
        proto = default_buffer_prototype()

        calls = {"n": 0}
        orig_get = source.get

        async def counting_get(*args: object, **kwargs: object) -> object:
            calls["n"] += 1
            return await orig_get(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(source, "get", counting_get)

        assert await cs.get("c/0", proto) is None
        assert cs.cache_info()["missing_keys"] == 1
        after_first = calls["n"]

        assert await cs.get("c/0", proto) is None
        assert calls["n"] == after_first  # no further source access
        assert cs.cache_stats()["negative_hits"] == 1

    async def test_enabled_by_default(self) -> None:
        """Negative caching is on by default (opt-out)."""
        cs = CacheStore(MemoryStore(), cache_store=MemoryStore())
        proto = default_buffer_prototype()
        assert cs.cache_missing is True
        assert await cs.get("c/0", proto) is None
        assert await cs.get("c/0", proto) is None
        assert cs.cache_info()["missing_keys"] == 1
        assert cs.cache_stats()["negative_hits"] == 1

    async def test_can_be_disabled(self) -> None:
        """With ``cache_missing=False`` nothing is remembered."""
        cs = CacheStore(MemoryStore(), cache_store=MemoryStore(), cache_missing=False)
        proto = default_buffer_prototype()
        assert await cs.get("c/0", proto) is None
        assert await cs.get("c/0", proto) is None
        assert cs.cache_info()["missing_keys"] == 0
        assert cs.cache_stats()["negative_hits"] == 0

    async def test_evicted_on_set(self) -> None:
        source = MemoryStore()
        cs = CacheStore(source, cache_store=MemoryStore(), cache_missing=True)
        proto = default_buffer_prototype()
        assert await cs.get("c/0", proto) is None
        assert cs.cache_info()["missing_keys"] == 1

        await cs.set("c/0", CPUBuffer.from_bytes(b"value"))
        assert cs.cache_info()["missing_keys"] == 0
        result = await cs.get("c/0", proto)
        assert result is not None
        assert result.to_bytes() == b"value"

    async def test_evicted_on_set_if_not_exists(self) -> None:
        source = MemoryStore()
        cs = CacheStore(source, cache_store=MemoryStore(), cache_missing=True)
        proto = default_buffer_prototype()
        assert await cs.get("c/0", proto) is None
        assert cs.cache_info()["missing_keys"] == 1

        await cs.set_if_not_exists("c/0", CPUBuffer.from_bytes(b"value"))
        assert cs.cache_info()["missing_keys"] == 0
        result = await cs.get("c/0", proto)
        assert result is not None
        assert result.to_bytes() == b"value"

    async def test_respects_ttl(self) -> None:
        """A negative entry expires after ``max_age_seconds`` so a key written to the
        source out-of-band becomes visible again."""
        source = MemoryStore()
        cs = CacheStore(source, cache_store=MemoryStore(), cache_missing=True, max_age_seconds=1)
        proto = default_buffer_prototype()
        assert await cs.get("c/0", proto) is None

        # an external writer adds the key directly to the source store
        await source.set("c/0", CPUBuffer.from_bytes(b"late"))

        # before TTL: still reported missing from the negative cache
        assert await cs.get("c/0", proto) is None
        await asyncio.sleep(1.1)

        # after TTL: the stale negative entry is bypassed, source is consulted
        result = await cs.get("c/0", proto)
        assert result is not None
        assert result.to_bytes() == b"late"
        assert cs.cache_info()["missing_keys"] == 0

    async def test_byte_range_unaffected(self) -> None:
        """Byte-range misses do not populate the negative cache."""
        cs = CacheStore(MemoryStore(), cache_store=MemoryStore(), cache_missing=True)
        proto = default_buffer_prototype()
        assert await cs.get("c/0", proto, byte_range=RangeByteRequest(0, 4)) is None
        assert cs.cache_info()["missing_keys"] == 0

    async def test_stats_and_info(self) -> None:
        """``negative_hits``/``missing_keys``/``cache_missing`` are surfaced and the
        positive ``hit_rate`` is unaffected by negative hits."""
        source = MemoryStore()
        cs = CacheStore(source, cache_store=MemoryStore(), cache_missing=True)
        proto = default_buffer_prototype()

        await cs.set("present", CPUBuffer.from_bytes(b"x"))
        assert (await cs.get("present", proto)) is not None  # positive hit
        assert await cs.get("absent", proto) is None  # records miss
        assert await cs.get("absent", proto) is None  # negative hit

        info = cs.cache_info()
        stats = cs.cache_stats()
        assert info["cache_missing"] is True
        assert info["missing_keys"] == 1
        assert stats["negative_hits"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1  # negative hit counts as neither hit nor miss
        assert stats["hit_rate"] == 0.5

    async def test_delete_does_not_record(self) -> None:
        """Deleting a key does not create a negative entry (deletion != checked-absent)."""
        cs = CacheStore(MemoryStore(), cache_store=MemoryStore(), cache_missing=True)
        await cs.delete("c/0")
        assert cs.cache_info()["missing_keys"] == 0

    async def test_negative_entry_counts_against_max_size(self) -> None:
        """A negative marker is charged against the shared ``max_size`` budget."""
        from zarr.experimental.cache_store import _NEGATIVE_ENTRY_SIZE

        cs = CacheStore(
            MemoryStore(), cache_store=MemoryStore(), cache_missing=True, max_size=10_000
        )
        proto = default_buffer_prototype()
        assert cs.cache_info()["current_size"] == 0
        assert await cs.get("absent", proto) is None
        assert cs.cache_info()["current_size"] == _NEGATIVE_ENTRY_SIZE

    async def test_shared_budget_bounds_negative_entries(self) -> None:
        """Many misses cannot grow the cache past ``max_size`` — old negative
        markers are evicted (LRU) to stay within the shared budget."""
        from zarr.experimental.cache_store import _NEGATIVE_ENTRY_SIZE

        cap = 5
        cs = CacheStore(
            MemoryStore(),
            cache_store=MemoryStore(),
            cache_missing=True,
            max_size=cap * _NEGATIVE_ENTRY_SIZE,
        )
        proto = default_buffer_prototype()
        for i in range(25):
            assert await cs.get(f"absent/{i}", proto) is None

        info = cs.cache_info()
        assert info["missing_keys"] == cap
        assert info["current_size"] <= cap * _NEGATIVE_ENTRY_SIZE
        # The most-recent misses are retained (LRU eviction of the oldest).
        assert await cs.get("absent/24", proto) is None
        assert cs.cache_stats()["negative_hits"] >= 1

    async def test_absent_evicted_before_present(self) -> None:
        """Under memory pressure, miss-markers are evicted before cached values."""
        from zarr.experimental.cache_store import _NEGATIVE_ENTRY_SIZE

        source = MemoryStore()
        # Budget for one value plus a couple of negative markers.
        value = CPUBuffer.from_bytes(b"v" * 64)
        cs = CacheStore(
            source,
            cache_store=MemoryStore(),
            cache_missing=True,
            max_size=len(value) + 2 * _NEGATIVE_ENTRY_SIZE,
        )
        proto = default_buffer_prototype()

        # Cache a present value, then record several misses that exceed the budget.
        await source.set("present", value)
        assert (await cs.get("present", proto)) is not None
        for i in range(5):
            assert await cs.get(f"absent/{i}", proto) is None

        # The present value survives; markers are bounded and never evict it.
        info = cs.cache_info()
        assert info["cached_keys"] == 1
        assert "present" in cs._state.entries
        present_slot = cs._state.entries["present"].full
        assert present_slot is not None
        assert present_slot.present
        # Markers fill only the room left over by the cached value (2 here), proving
        # both that misses were actually recorded and that they were bounded.
        assert info["missing_keys"] == 2
        assert info["current_size"] <= cs.max_size

    async def test_no_size_leak_on_miss_then_write(self) -> None:
        """Recording a miss then writing the key must not leak the marker's charge
        against ``current_size`` (regression for negative-entry accounting)."""
        from zarr.experimental.cache_store import _NEGATIVE_ENTRY_SIZE

        source = MemoryStore()
        value = CPUBuffer.from_bytes(b"v" * 50)
        cs = CacheStore(source, cache_store=MemoryStore(), cache_missing=True, max_size=10_000)
        proto = default_buffer_prototype()

        # Miss → marker charged; then write the same key → marker must be reclaimed.
        assert await cs.get("k", proto) is None
        assert cs.cache_info()["current_size"] == _NEGATIVE_ENTRY_SIZE
        await cs.set("k", value)

        info = cs.cache_info()
        assert info["missing_keys"] == 0
        assert info["cached_keys"] == 1
        # Only the value's bytes remain — no leftover marker overhead.
        assert info["current_size"] == len(value)
        # And the invariant holds: current_size == sum of all tracked entry sizes.
        total = sum(state.tracked_size for state in cs._state.entries.values())
        assert total == info["current_size"]

    async def test_no_size_leak_on_miss_then_set_if_not_exists(self) -> None:
        """``set_if_not_exists`` after a miss reclaims the marker's charge too."""
        source = MemoryStore()
        value = CPUBuffer.from_bytes(b"v" * 50)
        cs = CacheStore(source, cache_store=MemoryStore(), cache_missing=True, max_size=10_000)
        proto = default_buffer_prototype()

        assert await cs.get("k", proto) is None
        await cs.set_if_not_exists("k", value)

        info = cs.cache_info()
        assert info["missing_keys"] == 0
        total = sum(state.tracked_size for state in cs._state.entries.values())
        assert total == info["current_size"]

    async def test_stale_cached_value_becomes_negative_entry(self) -> None:
        """A cached value whose source key later reads absent is replaced by a negative
        marker, reclaiming the value's bytes (no leftover positive accounting)."""
        from zarr.experimental.cache_store import _NEGATIVE_ENTRY_SIZE

        source = MemoryStore()
        value = CPUBuffer.from_bytes(b"v" * 200)
        cs = CacheStore(source, cache_store=MemoryStore(), cache_missing=True, max_age_seconds=1)
        proto = default_buffer_prototype()

        # Cache a present value, then delete the key from the source out-of-band.
        await cs.set("k", value)
        assert cs.cache_info()["current_size"] == len(value)
        await source.delete("k")
        # Force the cached entry stale so the next read consults the (now empty) source.
        stale_slot = cs._state.entries["k"].full
        assert stale_slot is not None
        stale_slot.insert_time = time.monotonic() - 10

        assert await cs.get("k", proto) is None  # source absent -> records a miss
        info = cs.cache_info()
        assert info["cached_keys"] == 0
        assert info["missing_keys"] == 1
        # The 200-byte value was reclaimed; only the marker's overhead remains.
        assert info["current_size"] == _NEGATIVE_ENTRY_SIZE

    async def test_miss_not_recorded_when_budget_full_of_values(self) -> None:
        """When the budget is full of cached values and no markers exist to evict, a
        miss is not recorded (a marker never displaces a cached value)."""
        source = MemoryStore()
        value = CPUBuffer.from_bytes(b"v" * 200)
        # Budget fits exactly one value, with no room for a negative marker.
        cs = CacheStore(source, cache_store=MemoryStore(), cache_missing=True, max_size=200)
        proto = default_buffer_prototype()

        await cs.set("present", value)
        assert cs.cache_info()["cached_keys"] == 1

        assert await cs.get("absent", proto) is None
        info = cs.cache_info()
        assert info["missing_keys"] == 0  # no room -> miss not remembered
        assert info["cached_keys"] == 1  # cached value untouched
        assert info["current_size"] == len(value)

    async def test_caching_value_evicts_absent_markers(self) -> None:
        """Caching a present value reclaims room by evicting negative markers first."""
        from zarr.experimental.cache_store import _NEGATIVE_ENTRY_SIZE

        source = MemoryStore()
        cs = CacheStore(
            source,
            cache_store=MemoryStore(),
            cache_missing=True,
            max_size=3 * _NEGATIVE_ENTRY_SIZE,
        )
        proto = default_buffer_prototype()

        # Fill the budget with three negative markers.
        for i in range(3):
            assert await cs.get(f"absent/{i}", proto) is None
        assert cs.cache_info()["missing_keys"] == 3

        # Caching a value must evict marker(s) to fit — markers go before any value.
        await cs.set("v", CPUBuffer.from_bytes(b"v" * 100))
        info = cs.cache_info()
        assert info["cached_keys"] == 1
        assert info["missing_keys"] < 3  # at least one marker evicted to make room
        assert info["current_size"] <= cs.max_size

    async def test_upgrade_marker_to_value_under_pressure_evicts_other_entry(self) -> None:
        """Upgrading a stale negative marker to a cached value under memory pressure must
        evict a *different* entry, not self-evict (which would under-count current_size
        and breach max_size)."""
        from zarr.experimental.cache_store import _NEGATIVE_ENTRY_SIZE

        source = MemoryStore()
        cs = CacheStore(
            source,
            cache_store=MemoryStore(),
            cache_missing=True,
            max_age_seconds=1000,
            max_size=300,
        )
        proto = default_buffer_prototype()

        # One cached value "a" (128 B) and one negative marker "k" (128 B) → 256 B used.
        await cs.set("a", CPUBuffer.from_bytes(b"a" * (_NEGATIVE_ENTRY_SIZE)))
        assert await cs.get("k", proto) is None
        assert cs.cache_info()["current_size"] == 2 * _NEGATIVE_ENTRY_SIZE

        # "k" now exists in the source with a 200 B value; force the marker stale so the
        # next read fetches it and upgrades the slot to a present value (needs eviction).
        await source.set("k", CPUBuffer.from_bytes(b"k" * 200))
        marker_slot = cs._state.entries["k"].full
        assert marker_slot is not None
        marker_slot.insert_time = time.monotonic() - 5000

        result = await cs.get("k", proto)
        assert result is not None
        assert result.to_bytes() == b"k" * 200

        info = cs.cache_info()
        # Only "k" remains (the other value "a" was evicted to make room); the bound
        # holds and the size accounting matches the actual tracked entries exactly.
        assert info["cached_keys"] == 1
        assert "a" not in cs._state.entries
        assert not await cs._cache.exists("a")
        assert info["current_size"] == 200
        assert info["current_size"] <= cs.max_size
        total = sum(state.tracked_size for state in cs._state.entries.values())
        assert total == info["current_size"]

    async def test_set_if_not_exists_invalidates_stale_byte_range(self) -> None:
        """``set_if_not_exists`` must invalidate cached byte-range entries, not just the
        negative marker, so a later byte-range read does not return stale bytes."""
        source = MemoryStore()
        cs = CacheStore(source, cache_store=MemoryStore(), cache_missing=True)
        proto = default_buffer_prototype()

        await source.set("k", CPUBuffer.from_bytes(b"old data!!"))
        r1 = await cs.get("k", proto, byte_range=RangeByteRequest(0, 3))
        assert r1 is not None
        assert r1.to_bytes() == b"old"
        assert RangeByteRequest(0, 3) in cs._state.entries["k"].ranges

        # Source key removed out-of-band, then re-created via set_if_not_exists.
        await source.delete("k")
        await cs.set_if_not_exists("k", CPUBuffer.from_bytes(b"NEW data!!"))

        # The stale byte-range entry must be gone, and a fresh read returns new bytes.
        assert "k" not in cs._state.entries
        r2 = await cs.get("k", proto, byte_range=RangeByteRequest(0, 3))
        assert r2 is not None
        assert r2.to_bytes() == b"NEW"

    async def test_default_max_age_is_finite(self) -> None:
        """The default TTL is finite so remembered misses (and cached values) are
        re-validated against the source at bounded staleness."""
        cs = CacheStore(MemoryStore(), cache_store=MemoryStore())
        assert cs.max_age_seconds == 300

    async def test_negative_count_capped_without_max_size(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With ``max_size=None`` the marker *count* is capped: a scan over a huge
        sparse key space cannot grow the index without bound."""
        import zarr.experimental.cache_store as mod

        monkeypatch.setattr(mod, "_MAX_NEGATIVE_ENTRIES", 10)
        cs = CacheStore(MemoryStore(), cache_store=MemoryStore(), cache_missing=True)
        proto = default_buffer_prototype()

        for i in range(25):
            assert await cs.get(f"c/{i}", proto) is None

        assert cs.cache_info()["missing_keys"] == 10
        assert cs._state.negative_count == 10
        assert cs.cache_stats()["evictions"] >= 15
        # LRU: the most recent misses are the ones retained.
        assert all(f"c/{i}" in cs._state.entries for i in range(15, 25))

    async def test_concurrent_write_not_shadowed_by_stale_miss(self) -> None:
        """A miss observed *before* a concurrent write completed must not overwrite
        the newly written (present) entry with an absent marker."""
        source = MemoryStore()
        cs = CacheStore(source, cache_store=MemoryStore(), cache_missing=True)
        proto = default_buffer_prototype()

        # No slot exists when the (stale) fetch begins...
        prior_slot = cs._prior_full_slot("k")
        assert prior_slot is None
        # ...a concurrent set() completes during the fetch...
        await cs.set("k", CPUBuffer.from_bytes(b"value"))
        # ...then the stale absent observation lands.
        await cs._cache_miss("k", None, None, prior_slot)

        state = cs._state.entries.get("k")
        assert state is not None
        assert state.full is not None
        assert state.full.present
        assert cs.cache_info()["missing_keys"] == 0
        result = await cs.get("k", proto)
        assert result is not None
        assert result.to_bytes() == b"value"

    async def test_negative_hit_refreshes_lru(self) -> None:
        """A negative-cache hit marks the marker most-recently-used, so a hot absent
        key outlives cold markers under eviction pressure."""
        # Budget fits exactly two 128-byte markers.
        cs = CacheStore(MemoryStore(), cache_store=MemoryStore(), cache_missing=True, max_size=256)
        proto = default_buffer_prototype()

        assert await cs.get("a", proto) is None
        assert await cs.get("b", proto) is None
        # Touch "a": it becomes most-recently-used.
        assert await cs.get("a", proto) is None
        assert cs.cache_stats()["negative_hits"] == 1
        # A third marker must evict "b" (the LRU marker), not the hot "a".
        assert await cs.get("c", proto) is None
        assert "a" in cs._state.entries
        assert "b" not in cs._state.entries
        assert "c" in cs._state.entries

    async def test_oversized_set_rolls_back_backing_cache(self) -> None:
        """A written value larger than ``max_size`` must not linger untracked in the
        backing cache store."""
        cache = MemoryStore()
        cs = CacheStore(MemoryStore(), cache_store=cache, max_size=64, cache_set_data=True)
        proto = default_buffer_prototype()

        await cs.set("big", CPUBuffer.from_bytes(b"x" * 128))
        # The source has the value; the backing cache holds no untracked orphan.
        assert await cs._store.get("big", proto) is not None
        assert await cache.get("big", proto) is None
        assert cs._state.current_size == 0

    async def test_oversized_read_rolls_back_backing_cache(self) -> None:
        """A fetched value larger than ``max_size`` must not linger untracked in the
        backing cache store (mirror of the write-path rollback)."""
        source = MemoryStore()
        cache = MemoryStore()
        cs = CacheStore(source, cache_store=cache, max_size=64)
        proto = default_buffer_prototype()

        await source.set("big", CPUBuffer.from_bytes(b"x" * 128))
        result = await cs.get("big", proto)
        assert result is not None
        assert len(result) == 128
        assert await cache.get("big", proto) is None
        assert cs._state.current_size == 0

    async def test_full_key_miss_invalidates_byte_ranges(self) -> None:
        """Observing a key absent invalidates cached byte-range entries for it, so
        ``get(key)`` and ``get(key, byte_range)`` cannot diverge."""
        source = MemoryStore()
        cs = CacheStore(source, cache_store=MemoryStore(), cache_missing=True)
        proto = default_buffer_prototype()

        await source.set("k", CPUBuffer.from_bytes(b"old data!!"))
        r1 = await cs.get("k", proto, byte_range=RangeByteRequest(0, 3))
        assert r1 is not None
        assert r1.to_bytes() == b"old"
        assert RangeByteRequest(0, 3) in cs._state.entries["k"].ranges

        # Key removed out-of-band; a full-key read observes the absence.
        await source.delete("k")
        assert await cs.get("k", proto) is None

        # The stale byte-range entry is gone (the marker record replaced it), and a
        # fresh range read sees the absence.
        assert not cs._state.entries["k"].ranges
        assert await cs.get("k", proto, byte_range=RangeByteRequest(0, 3)) is None

    async def test_byte_range_success_clears_negative_marker(self) -> None:
        """A successful byte-range read proves the key exists, so it must clear the
        key's negative marker: the same instance cannot answer differently for the
        same key depending on how it is asked (mirror image of the full-key-miss
        range invalidation)."""
        source = MemoryStore()
        cs = CacheStore(source, cache_store=MemoryStore())
        proto = default_buffer_prototype()

        assert await cs.get("k", proto) is None  # marker recorded
        assert cs.cache_info()["missing_keys"] == 1

        # Out-of-band write to the source, then a ranged read observes bytes.
        await source.set("k", CPUBuffer.from_bytes(b"HELLO WORLD"))
        r = await cs.get("k", proto, byte_range=RangeByteRequest(0, 5))
        assert r is not None
        assert r.to_bytes() == b"HELLO"

        # The marker is gone and the counts are per key — no marker+range double
        # count (previously: tracked_keys=2, missing_keys=1 for one key).
        info = cs.cache_info()
        assert info["missing_keys"] == 0
        assert info["tracked_keys"] == 1

        # The full-key read must not serve the stale marker.
        full = await cs.get("k", proto)
        assert full is not None
        assert full.to_bytes() == b"HELLO WORLD"

    async def test_oversized_byte_range_still_clears_negative_marker(self) -> None:
        """Observing range bytes clears the marker even when the range itself is
        too large to cache."""
        from zarr.experimental.cache_store import _NEGATIVE_ENTRY_SIZE

        source = MemoryStore()
        # Budget fits a marker but not the 250-byte range value.
        cs = CacheStore(source, cache_store=MemoryStore(), max_size=2 * _NEGATIVE_ENTRY_SIZE)
        proto = default_buffer_prototype()

        assert await cs.get("k", proto) is None
        assert cs.cache_info()["missing_keys"] == 1

        await source.set("k", CPUBuffer.from_bytes(b"x" * 300))
        r = await cs.get("k", proto, byte_range=RangeByteRequest(0, 250))
        assert r is not None
        assert len(r) == 250

        # The range was not cached (too large), but the stale marker is gone.
        assert cs.cache_info()["missing_keys"] == 0
        full = await cs.get("k", proto)
        assert full is not None
        assert len(full) == 300

    async def test_range_entries_evicted_lru_within_key(self) -> None:
        """Per-range recency lives inside the key's record: under pressure the
        least-recently-used range of the eviction-candidate key is dropped first."""
        source = MemoryStore()
        cs = CacheStore(source, cache_store=MemoryStore(), max_size=100)
        proto = default_buffer_prototype()

        await source.set("k", CPUBuffer.from_bytes(b"x" * 120))
        r1 = RangeByteRequest(0, 40)
        r2 = RangeByteRequest(40, 80)
        r3 = RangeByteRequest(80, 120)
        assert await cs.get("k", proto, byte_range=r1) is not None
        assert await cs.get("k", proto, byte_range=r2) is not None
        # Touch r1 so r2 becomes the key's least-recently-used slot.
        assert await cs.get("k", proto, byte_range=r1) is not None
        assert cs.cache_stats()["hits"] == 1

        # Caching r3 (40 more bytes) exceeds the 100-byte budget: r2 must go.
        assert await cs.get("k", proto, byte_range=r3) is not None
        ranges = cs._state.entries["k"].ranges
        assert r1 in ranges
        assert r2 not in ranges
        assert r3 in ranges
        assert cs._state.current_size <= 100

    async def test_eviction_candidate_prefers_markers_else_lru_key(self) -> None:
        """Candidate selection: with no markers the LRU key is returned (O(1) fast
        path); with markers present, the LRU marker wins over an older cached value."""
        cs = CacheStore(MemoryStore(), cache_store=MemoryStore(), max_size=10_000)
        proto = default_buffer_prototype()

        await cs.set("a", CPUBuffer.from_bytes(b"aa"))
        await cs.set("b", CPUBuffer.from_bytes(b"bb"))
        # Touch "a" so "b" is the LRU key.
        assert await cs.get("a", proto) is not None
        assert cs._next_eviction_candidate() == "b"

        # A marker is preferred over any cached value, even a less recent one.
        assert await cs.get("absent", proto) is None
        assert cs._state.negative_count == 1
        assert cs._next_eviction_candidate() == "absent"


class _GatedGetStore(WrapperStore[Store]):
    """Source store whose ``get`` returns only once ``release`` is set.

    The value is read from the wrapped store *before* the wait, so the caller
    observes the state as of when its fetch began — the shape of a read/write
    race, made deterministic: ``fetched`` fires once the read has happened, the
    test then mutates the store, and ``release`` lets the stale result land.
    """

    def __init__(self, store: Store) -> None:
        super().__init__(store)
        self.fetched = asyncio.Event()
        self.release = asyncio.Event()

    async def get(
        self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        result = await self._store.get(key, prototype, byte_range)
        self.fetched.set()
        await self.release.wait()
        return result


class TestCacheStoreRecordCoherence:
    """One record per key, holding one source generation.

    Every source observation for a key — bytes or absence, full-key or ranged —
    replaces the key's whole record, and a cached full value answers ranged reads
    by slicing rather than by consulting the source. So the two read paths always
    answer from the same observation.
    """

    async def test_ranged_read_served_from_cached_full_value(self) -> None:
        """A fresh cached value serves byte ranges of itself, instead of reaching
        the source and caching a second, newer generation alongside it."""
        source = MemoryStore()
        cs = CacheStore(source, cache_store=MemoryStore())
        proto = default_buffer_prototype()

        await source.set("k", CPUBuffer.from_bytes(b"AAAAA"))
        full = await cs.get("k", proto)
        assert full is not None
        assert full.to_bytes() == b"AAAAA"

        # Out-of-band overwrite: the cached value is now stale, but still fresh by
        # TTL, so it — not the source — must answer both reads.
        await source.set("k", CPUBuffer.from_bytes(b"BBBBB"))
        ranged = await cs.get("k", proto, byte_range=RangeByteRequest(0, 3))
        again = await cs.get("k", proto)
        assert ranged is not None
        assert again is not None
        assert again.to_bytes()[:3] == ranged.to_bytes()
        assert ranged.to_bytes() == b"AAA"

        # Served from the cache, with no range entry (and no extra bytes) recorded.
        assert cs.cache_stats()["hits"] == 2
        assert not cs._state.entries["k"].ranges
        assert cs._state.current_size == 5
        await cs._assert_invariants()

    async def test_full_value_supersedes_stale_byte_ranges(self) -> None:
        """Caching a full value drops the key's byte ranges: they were read from an
        older generation, and the new value can serve those ranges itself."""
        source = MemoryStore()
        cs = CacheStore(source, cache_store=MemoryStore())
        proto = default_buffer_prototype()

        await source.set("k", CPUBuffer.from_bytes(b"AAAAA"))
        assert await cs.get("k", proto, byte_range=RangeByteRequest(0, 3)) is not None
        assert RangeByteRequest(0, 3) in cs._state.entries["k"].ranges

        await source.set("k", CPUBuffer.from_bytes(b"BBBBB"))
        full = await cs.get("k", proto)
        assert full is not None
        assert full.to_bytes() == b"BBBBB"
        assert not cs._state.entries["k"].ranges

        ranged = await cs.get("k", proto, byte_range=RangeByteRequest(0, 3))
        assert ranged is not None
        assert ranged.to_bytes() == b"BBB"
        await cs._assert_invariants()

    @staticmethod
    def _expire_value(cs: CacheStore, key: str) -> None:
        """Age *key*'s cached value past ``max_age_seconds`` without sleeping."""
        slot = cs._state.entries[key].full
        assert slot is not None
        slot.insert_time = time.monotonic() - 5000

    async def test_ranged_bytes_supersede_a_stale_cached_value(self) -> None:
        """When a ranged read does reach the source (the cached value has expired),
        the bytes it brings back replace that value rather than sitting next to it."""
        source = MemoryStore()
        cs = CacheStore(source, cache_store=MemoryStore())
        proto = default_buffer_prototype()

        await source.set("k", CPUBuffer.from_bytes(b"AAAAA"))
        assert await cs.get("k", proto) is not None
        await source.set("k", CPUBuffer.from_bytes(b"BBBBB"))
        self._expire_value(cs, "k")

        ranged = await cs.get("k", proto, byte_range=RangeByteRequest(0, 3))
        assert ranged is not None
        assert ranged.to_bytes() == b"BBB"
        # The stale value is gone from both the record and the backing cache, so no
        # later read can serve it.
        assert cs._state.entries["k"].full is None
        assert await cs._cache.get("k", proto) is None
        await cs._assert_invariants()

    async def test_ranged_miss_invalidates_the_key(self) -> None:
        """A ranged read that comes back empty drops the key's record, so a full
        read cannot serve bytes for a key this instance just reported nothing for.
        The absence is deliberately not recorded as a marker."""
        source = MemoryStore()
        cs = CacheStore(source, cache_store=MemoryStore())
        proto = default_buffer_prototype()

        await source.set("k", CPUBuffer.from_bytes(b"AAAAA"))
        assert await cs.get("k", proto) is not None
        await source.delete("k")
        self._expire_value(cs, "k")  # so the ranged read reaches the source

        assert await cs.get("k", proto, byte_range=RangeByteRequest(0, 3)) is None
        assert "k" not in cs._state.entries
        assert await cs._cache.get("k", proto) is None
        assert cs.cache_info()["missing_keys"] == 0  # invalidated, not marked absent
        assert await cs.get("k", proto) is None
        await cs._assert_invariants()

    async def test_oversized_set_drops_the_previous_entry(self) -> None:
        """An oversized write invalidates what it replaced: the earlier value is
        untracked (not left charging the budget) as well as deleted."""
        cache = MemoryStore()
        cs = CacheStore(MemoryStore(), cache_store=cache, max_size=200)
        proto = default_buffer_prototype()

        await cs.set("k", CPUBuffer.from_bytes(b"a" * 50))
        assert cs._state.current_size == 50

        await cs.set("k", CPUBuffer.from_bytes(b"b" * 500))
        assert cs._state.current_size == 0
        assert cs.cache_info()["cached_keys"] == 0
        assert await cache.get("k", proto) is None
        await cs._assert_invariants()

        # The value is still readable, and the budget is fully available.
        result = await cs.get("k", proto)
        assert result is not None
        assert len(result) == 500

    async def test_oversized_read_drops_the_previous_entry(self) -> None:
        """Mirror of the write path: an oversized fetch untracks the entry it
        replaced instead of leaving a phantom charging the budget."""
        source = MemoryStore()
        cache = MemoryStore()
        cs = CacheStore(source, cache_store=cache, max_size=200)
        proto = default_buffer_prototype()

        await source.set("k", CPUBuffer.from_bytes(b"a" * 50))
        assert await cs.get("k", proto) is not None
        assert cs._state.current_size == 50

        await source.set("k", CPUBuffer.from_bytes(b"b" * 500))
        self._expire_value(cs, "k")  # so the read refetches
        result = await cs.get("k", proto)
        assert result is not None
        assert len(result) == 500

        assert cs._state.current_size == 0
        assert await cache.get("k", proto) is None
        await cs._assert_invariants()

    async def test_concurrent_write_not_overwritten_by_stale_fill(self) -> None:
        """A read that fetched before a concurrent ``set`` must not publish its
        (older) bytes over the newly written value — the positive-path mirror of
        ``test_concurrent_write_not_shadowed_by_stale_miss``."""
        source = MemoryStore()
        gated = _GatedGetStore(source)
        cs = CacheStore(gated, cache_store=MemoryStore())
        proto = default_buffer_prototype()

        await source.set("k", CPUBuffer.from_bytes(b"old"))
        reader = asyncio.create_task(cs.get("k", proto))
        await gated.fetched.wait()  # the fetch has read b"old" but not returned

        await cs.set("k", CPUBuffer.from_bytes(b"new"))
        gated.release.set()
        stale = await reader
        assert stale is not None
        assert stale.to_bytes() == b"old"  # the caller still sees its own fetch

        # ...but the cache holds the newer write, and serves it.
        cached = await cs._cache.get("k", proto)
        assert cached is not None
        assert cached.to_bytes() == b"new"
        assert cs._state.current_size == 3
        await cs._assert_invariants()

    async def test_concurrent_write_not_shadowed_by_stale_miss_through_public_api(self) -> None:
        """The absent-path guard, driven through ``get``/``set`` rather than by
        calling ``_cache_miss`` directly, so it pins the behaviour and not the
        implementation."""
        source = MemoryStore()
        gated = _GatedGetStore(source)
        cs = CacheStore(gated, cache_store=MemoryStore(), cache_missing=True)
        proto = default_buffer_prototype()

        reader = asyncio.create_task(cs.get("k", proto))
        await gated.fetched.wait()  # the fetch has observed the key absent

        await cs.set("k", CPUBuffer.from_bytes(b"value"))
        gated.release.set()
        assert await reader is None  # the caller still sees its own fetch

        assert cs.cache_info()["missing_keys"] == 0
        result = await cs.get("k", proto)
        assert result is not None
        assert result.to_bytes() == b"value"
        await cs._assert_invariants()

    async def test_negative_count_matches_the_records(self) -> None:
        """``negative_count`` gates both the marker cap and the O(1) eviction fast
        path, so check it against a recount rather than against itself."""
        cs = CacheStore(MemoryStore(), cache_store=MemoryStore(), max_size=1000)
        proto = default_buffer_prototype()

        for i in range(20):
            assert await cs.get(f"absent{i}", proto) is None
            await cs.set(f"p{i}", CPUBuffer.from_bytes(b"x" * 30))
            assert await cs.get(f"p{i}", proto, byte_range=RangeByteRequest(0, 5)) is not None
            await cs.delete(f"p{i}")
            await cs.set_if_not_exists(f"q{i}", CPUBuffer.from_bytes(b"y" * 40))
            assert await cs.get(f"absent{i}", proto) is None

        counted = sum(
            1 for s in cs._state.entries.values() if s.full is not None and not s.full.present
        )
        assert cs._state.negative_count == counted
        assert cs.cache_info()["missing_keys"] == counted
        await cs._assert_invariants()

    async def test_assert_coherent_rejects_a_mixed_record(self) -> None:
        """The record invariant is checkable, not just documented: a record holding
        a full slot *and* byte ranges fails it."""
        state = _KeyState(full=_Entry(insert_time=0.0, size=1))
        state.assert_coherent()  # a full slot alone is fine

        state.ranges[RangeByteRequest(0, 1)] = _RangeEntry(
            buffer=CPUBuffer.from_bytes(b"x"), insert_time=0.0, size=1, last_used=0.0
        )
        with pytest.raises(AssertionError, match="cache incoherent"):
            state.assert_coherent()

    async def test_invariants_hold_under_randomised_operations(self) -> None:
        """Seeded fuzz over the whole operation surface, checking the tracking state
        against the backing cache after every step."""
        proto = default_buffer_prototype()
        byte_ranges: list[ByteRequest | None] = [
            RangeByteRequest(0, 5),
            RangeByteRequest(5, 10),
            SuffixByteRequest(4),
            None,
        ]
        max_sizes: list[int | None] = [None, 200, 600]
        max_ages: list[int | str] = [300, "infinity"]
        for seed in range(4):
            rng = random.Random(seed)
            source = MemoryStore()
            cs = CacheStore(
                source,
                cache_store=MemoryStore(),
                max_size=rng.choice(max_sizes),
                max_age_seconds=rng.choice(max_ages),
                cache_set_data=rng.choice([True, False]),
                cache_missing=rng.choice([True, False]),
            )
            for _ in range(120):
                key = f"k{rng.randrange(6)}"
                match rng.randrange(6):
                    case 0:
                        await cs.get(key, proto, byte_range=rng.choice(byte_ranges))
                    case 1:
                        size = rng.choice([5, 50, 300])
                        await cs.set(key, CPUBuffer.from_bytes(b"v" * size))
                    case 2:
                        await cs.delete(key)
                    case 3:
                        await cs.set_if_not_exists(key, CPUBuffer.from_bytes(b"z" * 20))
                    case 4:
                        # Out-of-band write: the source moves on without the cache.
                        await source.set(key, CPUBuffer.from_bytes(b"o" * 30))
                    case _:
                        await cs.delete_dir("k")
                await cs._assert_invariants()


def test_cache_store_opts_out_of_sync_io() -> None:
    """`CacheStore` must not advertise sync IO capability.

    Its caching logic lives only in the async `get`/`set`/`delete` overrides,
    while the inherited `WrapperStore` sync methods delegate straight to the
    source store. If the fused codec pipeline took the sync fast path, writes
    and deletes would bypass the cache and later async reads would serve stale
    entries. The opt-out forces sync-capable consumers onto the async path,
    which keeps the cache coherent.
    """
    from zarr.abc.store import _store_supports_sync_io
    from zarr.storage import MemoryStore

    cached = CacheStore(MemoryStore(), cache_store=MemoryStore())
    assert _store_supports_sync_io(cached) is False


async def test_cache_coherent_after_fused_pipeline_write() -> None:
    """Writing through the fused pipeline must not leave stale cache entries."""
    import numpy as np

    import zarr
    from zarr.core.config import config as zarr_config
    from zarr.storage import MemoryStore

    source = MemoryStore()
    cached = CacheStore(source, cache_store=MemoryStore())
    with zarr_config.set({"codec_pipeline.path": "zarr.core.codec_pipeline.FusedCodecPipeline"}):
        arr = zarr.create_array(cached, shape=(8,), chunks=(8,), dtype="int32", fill_value=0)
        arr[:] = np.arange(8, dtype="int32")
        np.testing.assert_array_equal(arr[:], np.arange(8))
        # Overwrite, then read back through the same cached handle: the read
        # must observe the overwrite, not a cached copy of the first write.
        arr[:] = np.arange(100, 108, dtype="int32")
        np.testing.assert_array_equal(arr[:], np.arange(100, 108))
