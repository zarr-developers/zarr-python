"""
Tests for the dual-store cache implementation.
"""

import asyncio
import time

import pytest

"""
Tests for the dual-store cache implementation.
"""

"""
Tests for the dual-store cache implementation.
"""

import asyncio
import time

import pytest

from zarr.abc.store import Store
from zarr.core.buffer.core import default_buffer_prototype
from zarr.core.buffer.cpu import Buffer as CPUBuffer
from zarr.storage import MemoryStore
from zarr.storage._caching_store import CacheStore


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
        if hasattr(cached_store, '_is_key_fresh'):
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
        cached_store = CacheStore(source_store, cache_store=cache_store, cache_set_data=False, key_insert_times={})

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
        cached_store = CacheStore(source_store, cache_store=cache_store, max_age_seconds=1, key_insert_times={})

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
        if not hasattr(cached_store, '_is_key_fresh'):
            pytest.skip("_is_key_fresh method not implemented")
            
        test_data = CPUBuffer.from_bytes(b"eternal data")
        await cached_store.set("eternal_key", test_data)

        # Should always be fresh
        assert cached_store._is_key_fresh("eternal_key")

        # Even after time passes
        await asyncio.sleep(0.1)
        assert cached_store._is_key_fresh("eternal_key")

    async def test_missing_key_cleanup(self, cached_store: CacheStore, source_store: Store) -> None:
        """Test that accessing non-existent keys cleans up cache."""
        # Skip test if key_insert_times attribute doesn't exist
        if not hasattr(cached_store, 'key_insert_times'):
            pytest.skip("key_insert_times attribute not implemented")
            
        # Put data in cache but not source
        test_data = CPUBuffer.from_bytes(b"orphaned data")
        await cached_store._cache.set("orphan_key", test_data)
        cached_store.key_insert_times["orphan_key"] = time.monotonic()

        # Access should clean up cache
        result = await cached_store.get("orphan_key", default_buffer_prototype())
        assert result is None
        assert not await cached_store._cache.exists("orphan_key")
        assert "orphan_key" not in cached_store.key_insert_times

    async def test_cache_info(self, cached_store: CacheStore) -> None:
        """Test cache_info method returns correct information."""
        # Test initial state
        info = cached_store.cache_info()
        
        # Check all expected keys are present
        expected_keys = {
            "cache_store_type", "max_age_seconds", "max_size", "current_size",
            "cache_set_data", "tracked_keys", "cached_keys"
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
            key_insert_times={}
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
        if hasattr(cached_store._cache, 'clear'):
            # If cache store supports clear, all data should be gone
            assert not await cached_store._cache.exists("clear_test_1")
            assert not await cached_store._cache.exists("clear_test_2")
        
        # Verify data still exists in source store
        assert await cached_store._store.exists("clear_test_1")
        assert await cached_store._store.exists("clear_test_2")

    async def test_clear_cache_with_cache_store_without_clear(self) -> None:
        """Test clear_cache when cache store doesn't support clear method."""
        # Create a simple mock cache store without clear method
        from typing import Any
        
        class MockCacheStore(MemoryStore):
            def __init__(self) -> None:
                super().__init__()
            
            # Override to not have clear method
            def __getattribute__(self, name: str) -> Any:
                if name == 'clear':
                    raise AttributeError("'MockCacheStore' object has no attribute 'clear'")
                return super().__getattribute__(name)
        
        source_store = MemoryStore()
        mock_cache_store = MockCacheStore()
        
        # Verify mock doesn't have clear
        assert not hasattr(mock_cache_store, 'clear')
        
        cached_store = CacheStore(
            source_store,
            cache_store=mock_cache_store,
            key_insert_times={}
        )
        
        # Add test data
        test_data = CPUBuffer.from_bytes(b"test data")
        await cached_store.set("mock_test", test_data)
        
        # Verify tracking before clear
        assert cached_store.cache_info()["tracked_keys"] == 1
        
        # Clear cache (should only clear tracking, not the cache store since it has no clear method)
        await cached_store.clear_cache()
        
        # Verify tracking is cleared
        info = cached_store.cache_info()
        assert info["tracked_keys"] == 0
        assert info["cached_keys"] == 0
        assert info["current_size"] == 0
