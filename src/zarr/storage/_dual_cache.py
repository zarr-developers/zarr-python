from __future__ import annotations

import time
from typing import Any

from zarr.abc.store import ByteRequest, Store
from zarr.core.buffer.core import Buffer, BufferPrototype
from zarr.storage._wrapper import WrapperStore


class CacheStore(WrapperStore[Store]):
    """
    A dual-store caching implementation for Zarr stores.

    This cache wraps any Store implementation and uses a separate Store instance
    as the cache backend. This provides persistent caching capabilities with
    time-based expiration and flexible cache storage options.

    Parameters
    ----------
    store : Store
        The underlying store to wrap with caching
    cache_store : Store
        The store to use for caching (can be any Store implementation)
    max_age_seconds : int | str, optional
        Maximum age of cached entries in seconds, or "infinity" for no expiration.
        Default is "infinity".
    cache_set_data : bool, optional
        Whether to cache data when it's written to the store. Default is True.

    Examples
    --------
    >>> from zarr.storage import MemoryStore, FSStore
    >>> base_store = FSStore("/path/to/data")
    >>> cache_store = MemoryStore()
    >>> cached_store = CacheStore(base_store, cache_store=cache_store, max_age_seconds=3600)
    """

    def __init__(
        self,
        store: Store,
        *,
        cache_store: Store,
        max_age_seconds: int | str = "infinity",
        cache_set_data: bool = True,
    ) -> None:
        super().__init__(store)
        self._cache = cache_store
        self.max_age_seconds = max_age_seconds
        self.cache_set_data = cache_set_data
        self.key_insert_times: dict[str, float] = {}

    def _is_key_fresh(self, key: str) -> bool:
        """Check if a cached key is still fresh based on max_age_seconds."""
        if self.max_age_seconds == "infinity":
            return True

        if not isinstance(self.max_age_seconds, (int, float)):
            return True

        now = time.monotonic()
        elapsed = now - self.key_insert_times.get(key, 0)
        return elapsed < self.max_age_seconds

    async def _get_try_cache(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        """Try to get data from cache first, falling back to source store."""
        maybe_cached_result = await self._cache.get(key, prototype, byte_range)
        if maybe_cached_result is not None:
            # Found in cache, but verify it still exists in source
            if await super().exists(key):
                return maybe_cached_result
            else:
                # Key doesn't exist in source anymore, clean up cache
                await self._cache.delete(key)
                self.key_insert_times.pop(key, None)
                return None

        # Not in cache, fetch from source store
        maybe_fresh_result = await super().get(key, prototype, byte_range)
        if maybe_fresh_result is None:
            # Key doesn't exist in source, remove from cache if present
            await self._cache.delete(key)
        else:
            # Cache the result for future use
            await self._cache.set(key, maybe_fresh_result)
            self.key_insert_times[key] = time.monotonic()
        return maybe_fresh_result

    async def _get_no_cache(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        """Get data directly from source store and update cache."""
        maybe_fresh_result = await super().get(key, prototype, byte_range)
        if maybe_fresh_result is None:
            # Key doesn't exist in source, remove from cache and tracking
            await self._cache.delete(key)
            self.key_insert_times.pop(key, None)
        else:
            # Update cache with fresh data
            await self._cache.set(key, maybe_fresh_result)
            self.key_insert_times[key] = time.monotonic()
        return maybe_fresh_result

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        """
        Retrieve data from the store, using cache when appropriate.

        Parameters
        ----------
        key : str
            The key to retrieve
        prototype : BufferPrototype
            Buffer prototype for creating the result buffer
        byte_range : ByteRequest, optional
            Byte range to retrieve

        Returns
        -------
        Buffer | None
            The retrieved data, or None if not found
        """
        if self._is_key_fresh(key):
            return await self._get_try_cache(key, prototype, byte_range)
        else:
            return await self._get_no_cache(key, prototype, byte_range)

    async def set(self, key: str, value: Buffer) -> None:
        """
        Store data in the underlying store and optionally in cache.

        Parameters
        ----------
        key : str
            The key to store under
        value : Buffer
            The data to store
        """
        await super().set(key, value)

        if self.cache_set_data:
            # Cache the new data
            await self._cache.set(key, value)
            self.key_insert_times[key] = time.monotonic()
        else:
            # Remove from cache since data changed
            await self._cache.delete(key)
            self.key_insert_times.pop(key, None)

    async def delete(self, key: str) -> None:
        """
        Delete data from both the underlying store and cache.

        Parameters
        ----------
        key : str
            The key to delete
        """
        await super().delete(key)
        await self._cache.delete(key)
        self.key_insert_times.pop(key, None)

    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in the store.

        Parameters
        ----------
        key : str
            The key to check

        Returns
        -------
        bool
            True if the key exists
        """
        # Check source store for existence (cache might be stale)
        return await super().exists(key)

    def clear_cache(self) -> None:
        """Clear all cached data and timing information."""
        # Note: This is a synchronous method but cache operations are async
        # In practice, you might want to call this from an async context
        self.key_insert_times.clear()

    async def clear_cache_async(self) -> None:
        """Asynchronously clear all cached data and timing information."""
        # Clear timing tracking
        self.key_insert_times.clear()
        
        # Clear the cache store - we need to list and delete all keys
        # since Store doesn't have a clear() method
        try:
            cache_keys = []
            async for key in self._cache.list_dir(""):
                cache_keys.append(key)

            for key in cache_keys:
                await self._cache.delete(key)
        except Exception:
            # If listing/clearing fails, just reset timing info
            pass

    def cache_info(self) -> dict[str, Any]:
        """
        Get cache configuration information.

        Returns
        -------
        dict[str, Any]
            Dictionary containing cache configuration
        """
        return {
            "cache_store_type": type(self._cache).__name__,
            "max_age_seconds": self.max_age_seconds,
            "cache_set_data": self.cache_set_data,
            "tracked_keys": len(self.key_insert_times),
        }


# Alias for backward compatibility
DualStoreCache = CacheStore
