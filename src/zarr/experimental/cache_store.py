from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Literal

from zarr.abc.store import ByteRequest, Store
from zarr.storage._wrapper import WrapperStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zarr.core.buffer.core import Buffer, BufferPrototype


def buffer_size(v: Any) -> int:
    """Calculate the size in bytes of a value, handling Buffer objects properly."""
    if hasattr(v, "__len__") and hasattr(v, "nbytes"):
        # This is likely a Buffer object
        return int(v.nbytes)
    elif hasattr(v, "to_bytes"):
        # This is a Buffer object, get its bytes representation
        return len(v.to_bytes())
    elif isinstance(v, (bytes, bytearray, memoryview)):
        return len(v)
    else:
        # Fallback to numpy if available
        try:
            import numpy as np

            return int(np.asarray(v).nbytes)
        except ImportError:
            # If numpy not available, estimate size
            return len(str(v).encode("utf-8"))


class CacheStore(WrapperStore[Store]):
    """
    A dual-store caching implementation for Zarr stores.

    This cache wraps any Store implementation and uses a separate Store instance
    as the cache backend. This provides persistent caching capabilities with
    time-based expiration, size-based eviction, and flexible cache storage options.

    Parameters
    ----------
    store : Store
        The underlying store to wrap with caching
    cache_store : Store
        The store to use for caching (can be any Store implementation)
    max_age_seconds : int | str, optional
        Maximum age of cached entries in seconds, or "infinity" for no expiration.
        Default is "infinity".
    max_size : int | None, optional
        Maximum size of the cache in bytes. When exceeded, least recently used
        items are evicted. None means unlimited size. Default is None.
    cache_set_data : bool, optional
        Whether to cache data when it's written to the store. Default is True.

    Examples
    --------
    >>> from zarr.storage._memory import MemoryStore
    >>> store_a = MemoryStore({})
    >>> store_b = MemoryStore({})
    >>> cached_store = CacheStore(store=store_a, cache_store=store_b, max_age_seconds=10, max_size=1024*1024)

    """

    _cache: Store
    max_age_seconds: int | Literal["infinity"]
    max_size: int | None
    key_insert_times: dict[str, float]
    cache_set_data: bool
    _cache_order: OrderedDict[str, None]  # Track access order for LRU
    _current_size: int  # Track current cache size
    _key_sizes: dict[str, int]  # Track size of each cached key

    def __init__(
        self,
        store: Store,
        *,
        cache_store: Store,
        max_age_seconds: int | str = "infinity",
        max_size: int | None = None,
        key_insert_times: dict[str, float] | None = None,
        cache_set_data: bool = True,
    ) -> None:
        super().__init__(store)
        self._cache = cache_store
        # Validate and convert max_age_seconds
        if isinstance(max_age_seconds, str):
            if max_age_seconds != "infinity":
                raise ValueError("max_age_seconds string value must be 'infinity'")
            self.max_age_seconds = "infinity"
        else:
            self.max_age_seconds = max_age_seconds
        self.max_size = max_size
        if key_insert_times is None:
            self.key_insert_times = {}
        else:
            self.key_insert_times = key_insert_times
        self.cache_set_data = cache_set_data
        self._cache_order = OrderedDict()
        self._current_size = 0
        self._key_sizes = {}

    def _is_key_fresh(self, key: str) -> bool:
        """Check if a cached key is still fresh based on max_age_seconds."""
        if self.max_age_seconds == "infinity":
            return True
        else:
            now = time.monotonic()
            elapsed = now - self.key_insert_times.get(key, 0)
            return elapsed < self.max_age_seconds

    def _get_cache_size(self, key: str) -> int:
        """Get the size of a cached item."""
        # For now, we'll estimate by getting the data when we cache it
        return 0  # Will be properly set when caching

    async def _accommodate_value(self, value_size: int) -> None:
        """Ensure there is enough space in the cache for a new value."""
        if self.max_size is None:
            return

        # Remove least recently used items until we have enough space
        while self._current_size + value_size > self.max_size and self._cache_order:
            # Get the least recently used key (first in OrderedDict)
            lru_key = next(iter(self._cache_order))
            await self._evict_key(lru_key)

    async def _evict_key(self, key: str) -> None:
        """Remove a key from cache and update size tracking."""
        try:
            # Get the size of the key being evicted
            key_size = self._key_sizes.get(key, 0)

            # Remove from tracking structures
            if key in self._cache_order:
                del self._cache_order[key]
            if key in self.key_insert_times:
                del self.key_insert_times[key]
            if key in self._key_sizes:
                del self._key_sizes[key]

            # Update current size
            self._current_size = max(0, self._current_size - key_size)

            # Actually delete from cache store
            await self._cache.delete(key)

            logger.info("_evict_key: evicted key %s from cache, size %d", key, key_size)
        except Exception as e:
            logger.warning("_evict_key: failed to evict key %s: %s", key, e)

    async def _cache_value(self, key: str, value: Any) -> None:
        """Cache a value with size tracking."""
        value_size = buffer_size(value)

        # Check if value exceeds max size
        if self.max_size is not None and value_size > self.max_size:
            logger.warning(
                "_cache_value: value size %d exceeds max_size %d, not caching",
                value_size,
                self.max_size,
            )
            return

        # If key already exists, subtract old size first (Bug fix #3)
        if key in self._key_sizes:
            old_size = self._key_sizes[key]
            self._current_size -= old_size
            logger.info("_cache_value: updating existing key %s, old size %d", key, old_size)

        # Make room for the new value
        await self._accommodate_value(value_size)

        # Update tracking
        self._cache_order[key] = None  # OrderedDict to track access order
        self._current_size += value_size
        self._key_sizes[key] = value_size
        self.key_insert_times[key] = time.monotonic()

        logger.info("_cache_value: cached key %s with size %d bytes", key, value_size)

    def _update_access_order(self, key: str) -> None:
        """Update the access order for LRU tracking."""
        if key in self._cache_order:
            # Move to end (most recently used)
            self._cache_order.move_to_end(key)

    def _remove_from_tracking(self, key: str) -> None:
        """Remove a key from all tracking structures."""
        if key in self._cache_order:
            del self._cache_order[key]
        if key in self.key_insert_times:
            del self.key_insert_times[key]
        if key in self._key_sizes:
            del self._key_sizes[key]

    async def _get_try_cache(
        self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        """Try to get data from cache first, falling back to source store."""
        maybe_cached_result = await self._cache.get(key, prototype, byte_range)
        if maybe_cached_result is not None:
            logger.info("_get_try_cache: key %s found in cache", key)
            # Update access order for LRU
            self._update_access_order(key)
            # Verify the key still exists in source store before returning cached data
            if await super().exists(key):
                return maybe_cached_result
            else:
                # Key no longer exists in source, clean up cache
                logger.info(
                    "_get_try_cache: key %s no longer exists in source, cleaning up cache", key
                )
                await self._cache.delete(key)
                self._remove_from_tracking(key)
                return None
        else:
            logger.info("_get_try_cache: key %s not found in cache, fetching from store", key)
            maybe_fresh_result = await super().get(key, prototype, byte_range)
            if maybe_fresh_result is None:
                await self._cache.delete(key)
                self._remove_from_tracking(key)
            else:
                await self._cache.set(key, maybe_fresh_result)
                await self._cache_value(key, maybe_fresh_result)
            return maybe_fresh_result

    async def _get_no_cache(
        self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        """Get data directly from source store and update cache."""
        maybe_fresh_result = await super().get(key, prototype, byte_range)
        if maybe_fresh_result is None:
            # Key doesn't exist in source, remove from cache and tracking
            await self._cache.delete(key)
            self._remove_from_tracking(key)
        else:
            logger.info("_get_no_cache: key %s found in store, setting in cache", key)
            await self._cache.set(key, maybe_fresh_result)
            await self._cache_value(key, maybe_fresh_result)
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
        if not self._is_key_fresh(key):
            logger.info("get: key %s is not fresh, fetching from store", key)
            return await self._get_no_cache(key, prototype, byte_range)
        else:
            logger.info("get: key %s is fresh, trying cache", key)
            return await self._get_try_cache(key, prototype, byte_range)

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
        logger.info("set: setting key %s in store", key)
        await super().set(key, value)
        if self.cache_set_data:
            logger.info("set: setting key %s in cache", key)
            await self._cache.set(key, value)
            await self._cache_value(key, value)
        else:
            logger.info("set: deleting key %s from cache", key)
            await self._cache.delete(key)
            self._remove_from_tracking(key)

    async def delete(self, key: str) -> None:
        """
        Delete data from both the underlying store and cache.

        Parameters
        ----------
        key : str
            The key to delete
        """
        logger.info("delete: deleting key %s from store", key)
        await super().delete(key)
        logger.info("delete: deleting key %s from cache", key)
        await self._cache.delete(key)
        self._remove_from_tracking(key)

    def cache_info(self) -> dict[str, Any]:
        """Return information about the cache state."""
        return {
            "cache_store_type": type(self._cache).__name__,
            "max_age_seconds": "infinity"
            if self.max_age_seconds == "infinity"
            else self.max_age_seconds,
            "max_size": self.max_size,
            "current_size": self._current_size,
            "cache_set_data": self.cache_set_data,
            "tracked_keys": len(self.key_insert_times),
            "cached_keys": len(self._cache_order),
        }

    async def clear_cache(self) -> None:
        """Clear all cached data and tracking information."""
        # Clear the cache store if it supports clear
        if hasattr(self._cache, "clear"):
            await self._cache.clear()

        # Reset tracking
        self.key_insert_times.clear()
        self._cache_order.clear()
        self._key_sizes.clear()
        self._current_size = 0
        logger.info("clear_cache: cleared all cache data")
