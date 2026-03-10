from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Self

from zarr.abc.store import ByteRequest, Store
from zarr.storage._wrapper import WrapperStore

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zarr.core.buffer.core import Buffer, BufferPrototype

# A cache entry identifier.  Plain ``str`` for full-key entries that live in
# the Store-backed cache; ``(str, ByteRequest)`` for byte-range entries that
# live in the in-memory range cache.
_CacheEntryKey = str | tuple[str, ByteRequest]


@dataclass(slots=True)
class _CacheState:
    cache_order: OrderedDict[_CacheEntryKey, None] = field(default_factory=OrderedDict)
    current_size: int = 0
    key_sizes: dict[_CacheEntryKey, int] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    key_insert_times: dict[_CacheEntryKey, float] = field(default_factory=dict)
    range_cache: dict[str, dict[ByteRequest, Buffer]] = field(default_factory=dict)


class CacheStore(WrapperStore[Store]):
    """
    A dual-store caching implementation for Zarr stores.

    This cache wraps any Store implementation and uses a separate Store instance
    as the cache backend. This provides persistent caching capabilities with
    time-based expiration, size-based eviction, and flexible cache storage options.

    Full-key reads are cached in the Store-backed cache.  Byte-range reads are
    cached in a separate in-memory dictionary so that partial reads never
    pollute the filesystem (or other persistent backend).  Both caches share
    the same ``max_size`` budget and LRU eviction policy.

    Parameters
    ----------
    store : Store
        The underlying store to wrap with caching
    cache_store : Store
        The store to use for caching (can be any Store implementation)
    max_age_seconds : int | None, optional
        Maximum age of cached entries in seconds. None means no expiration.
        Default is None.
    max_size : int | None, optional
        Maximum size of the cache in bytes. When exceeded, least recently used
        items are evicted. None means unlimited size. Default is None.
        Note: Individual values larger than max_size will not be cached.
    cache_set_data : bool, optional
        Whether to cache data when it's written to the store. Default is True.

    Examples
    --------
    ```python
    import zarr
    from zarr.storage import MemoryStore
    from zarr.experimental.cache_store import CacheStore

    # Create a cached store
    source_store = MemoryStore()
    cache_store = MemoryStore()
    cached_store = CacheStore(
        store=source_store,
        cache_store=cache_store,
        max_age_seconds=60,
        max_size=1024*1024
    )

    # Use it like any other store
    array = zarr.create(shape=(100,), store=cached_store)
    array[:] = 42
    ```

    """

    _cache: Store
    max_age_seconds: int | Literal["infinity"]
    max_size: int | None
    cache_set_data: bool
    _state: _CacheState

    def __init__(
        self,
        store: Store,
        *,
        cache_store: Store,
        max_age_seconds: int | str = "infinity",
        max_size: int | None = None,
        cache_set_data: bool = True,
    ) -> None:
        super().__init__(store)

        if not cache_store.supports_deletes:
            msg = (
                f"The provided cache store {cache_store} does not support deletes. "
                "The cache_store must support deletes for CacheStore to function properly."
            )
            raise ValueError(msg)

        self._cache = cache_store
        # Validate and set max_age_seconds
        if isinstance(max_age_seconds, str):
            if max_age_seconds != "infinity":
                raise ValueError("max_age_seconds string value must be 'infinity'")
            self.max_age_seconds = "infinity"
        else:
            self.max_age_seconds = max_age_seconds
        self.max_size = max_size
        self.cache_set_data = cache_set_data
        self._state = _CacheState()

    def _with_store(self, store: Store) -> Self:
        # Cannot support this operation because it would share a cache, but have a new store
        # So cache keys would conflict
        raise NotImplementedError("CacheStore does not support this operation.")

    def with_read_only(self, read_only: bool = False) -> Self:
        # Create a new cache store that shares the same cache and mutable state
        store = type(self)(
            store=self._store.with_read_only(read_only),
            cache_store=self._cache,
            max_age_seconds=self.max_age_seconds,
            max_size=self.max_size,
            cache_set_data=self.cache_set_data,
        )
        store._state = self._state
        return store

    def _is_key_fresh(self, entry_key: _CacheEntryKey) -> bool:
        """Check if a cached entry is still fresh based on max_age_seconds.

        Uses monotonic time for accurate elapsed time measurement.
        """
        if self.max_age_seconds == "infinity":
            return True
        now = time.monotonic()
        elapsed = now - self._state.key_insert_times.get(entry_key, 0)
        return elapsed < self.max_age_seconds

    async def _accommodate_value(self, value_size: int) -> None:
        """Ensure there is enough space in the cache for a new value.

        Must be called while holding self._state.lock.
        """
        if self.max_size is None:
            return

        # Remove least recently used items until we have enough space
        while self._state.current_size + value_size > self.max_size and self._state.cache_order:
            # Get the least recently used key (first in OrderedDict)
            lru_key = next(iter(self._state.cache_order))
            await self._evict_key(lru_key)

    async def _evict_key(self, entry_key: _CacheEntryKey) -> None:
        """Evict a cache entry.

        Must be called while holding self._state.lock.

        For ``str`` keys the entry is deleted from the Store-backed cache.
        For ``(str, ByteRequest)`` keys the entry is removed from the
        in-memory range cache.
        """
        key_size = self._state.key_sizes.get(entry_key, 0)

        if isinstance(entry_key, str):
            await self._cache.delete(entry_key)
        else:
            base_key, byte_range = entry_key
            per_key = self._state.range_cache.get(base_key)
            if per_key is not None:
                per_key.pop(byte_range, None)
                if not per_key:
                    del self._state.range_cache[base_key]

        self._state.cache_order.pop(entry_key, None)
        self._state.key_insert_times.pop(entry_key, None)
        self._state.key_sizes.pop(entry_key, None)
        self._state.current_size = max(0, self._state.current_size - key_size)
        self._state.evictions += 1

    async def _track_entry(self, entry_key: _CacheEntryKey, value: Buffer) -> bool:
        """Register *entry_key* in the shared size / LRU tracking.

        Returns ``True`` if the entry was tracked, ``False`` if the value
        exceeds ``max_size`` and was skipped.  Callers should roll back any
        data they already stored when this returns ``False``.

        This method holds the lock for the entire operation to ensure atomicity.
        """
        value_size = len(value)

        # Check if value exceeds max size
        if self.max_size is not None and value_size > self.max_size:
            return False

        async with self._state.lock:
            # If key already exists, subtract old size first
            if entry_key in self._state.key_sizes:
                old_size = self._state.key_sizes[entry_key]
                self._state.current_size -= old_size

            # Make room for the new value
            await self._accommodate_value(value_size)

            # Update tracking atomically
            self._state.cache_order[entry_key] = None
            self._state.current_size += value_size
            self._state.key_sizes[entry_key] = value_size
            self._state.key_insert_times[entry_key] = time.monotonic()

        return True

    async def _update_access_order(self, entry_key: _CacheEntryKey) -> None:
        """Update the access order for LRU tracking."""
        if entry_key in self._state.cache_order:
            async with self._state.lock:
                self._state.cache_order.move_to_end(entry_key)

    def _remove_from_tracking(self, entry_key: _CacheEntryKey) -> None:
        """Remove an entry from all tracking structures.

        Must be called while holding self._state.lock.
        """
        self._state.cache_order.pop(entry_key, None)
        self._state.key_insert_times.pop(entry_key, None)
        self._state.key_sizes.pop(entry_key, None)

    def _invalidate_range_entries(self, key: str) -> None:
        """Remove all byte-range entries for *key* from the range cache and tracking.

        Must be called while holding self._state.lock.
        """
        per_key = self._state.range_cache.pop(key, None)
        if per_key is not None:
            for byte_range in per_key:
                entry_key: _CacheEntryKey = (key, byte_range)
                entry_size = self._state.key_sizes.pop(entry_key, 0)
                self._state.cache_order.pop(entry_key, None)
                self._state.key_insert_times.pop(entry_key, None)
                self._state.current_size = max(0, self._state.current_size - entry_size)

    # ------------------------------------------------------------------
    # get helpers
    # ------------------------------------------------------------------

    async def _cache_miss(
        self, key: str, byte_range: ByteRequest | None, result: Buffer | None
    ) -> None:
        """Handle a cache miss by storing or cleaning up after a source-store fetch."""
        if result is None:
            if byte_range is None:
                await self._cache.delete(key)
                async with self._state.lock:
                    self._remove_from_tracking(key)
            else:
                entry_key: _CacheEntryKey = (key, byte_range)
                async with self._state.lock:
                    per_key = self._state.range_cache.get(key)
                    if per_key is not None:
                        per_key.pop(byte_range, None)
                        if not per_key:
                            del self._state.range_cache[key]
                    self._remove_from_tracking(entry_key)
        else:
            if byte_range is None:
                await self._cache.set(key, result)
                await self._track_entry(key, result)
            else:
                entry_key = (key, byte_range)
                self._state.range_cache.setdefault(key, {})[byte_range] = result
                tracked = await self._track_entry(entry_key, result)
                if not tracked:
                    # Value too large for the cache — roll back the insertion
                    per_key = self._state.range_cache.get(key)
                    if per_key is not None:
                        per_key.pop(byte_range, None)
                        if not per_key:
                            del self._state.range_cache[key]

    async def _get_try_cache(
        self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        """Try to get data from cache first, falling back to source store."""
        if byte_range is None:
            # Full-key read — use Store-backed cache
            maybe_cached = await self._cache.get(key, prototype)
            if maybe_cached is not None:
                self._state.hits += 1
                await self._update_access_order(key)
                return maybe_cached
        else:
            # Byte-range read — use in-memory range cache
            entry_key: _CacheEntryKey = (key, byte_range)
            per_key = self._state.range_cache.get(key)
            if per_key is not None:
                cached_buf = per_key.get(byte_range)
                if cached_buf is not None:
                    self._state.hits += 1
                    await self._update_access_order(entry_key)
                    return cached_buf

        # Cache miss — fetch from source store
        self._state.misses += 1
        result = await super().get(key, prototype, byte_range)
        await self._cache_miss(key, byte_range, result)
        return result

    async def _get_no_cache(
        self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        """Get data directly from source store and update cache."""
        self._state.misses += 1
        result = await super().get(key, prototype, byte_range)
        await self._cache_miss(key, byte_range, result)
        return result

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
        entry_key: _CacheEntryKey = (key, byte_range) if byte_range is not None else key
        if not self._is_key_fresh(entry_key):
            return await self._get_no_cache(key, prototype, byte_range)
        else:
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
        await super().set(key, value)
        # Invalidate all cached byte-range entries (source data changed)
        async with self._state.lock:
            self._invalidate_range_entries(key)
        if self.cache_set_data:
            await self._cache.set(key, value)
            await self._track_entry(key, value)
        else:
            await self._cache.delete(key)
            async with self._state.lock:
                self._remove_from_tracking(key)

    async def delete(self, key: str) -> None:
        """
        Delete data from both the underlying store and cache.

        Parameters
        ----------
        key : str
            The key to delete
        """
        await super().delete(key)
        # Invalidate all cached byte-range entries
        async with self._state.lock:
            self._invalidate_range_entries(key)
        await self._cache.delete(key)
        async with self._state.lock:
            self._remove_from_tracking(key)

    def cache_info(self) -> dict[str, Any]:
        """Return information about the cache state."""
        return {
            "cache_store_type": type(self._cache).__name__,
            "max_age_seconds": "infinity"
            if self.max_age_seconds == "infinity"
            else self.max_age_seconds,
            "max_size": self.max_size,
            "current_size": self._state.current_size,
            "cache_set_data": self.cache_set_data,
            "tracked_keys": len(self._state.key_insert_times),
            "cached_keys": len(self._state.cache_order),
        }

    def cache_stats(self) -> dict[str, Any]:
        """Return cache performance statistics."""
        total_requests = self._state.hits + self._state.misses
        hit_rate = self._state.hits / total_requests if total_requests > 0 else 0.0
        return {
            "hits": self._state.hits,
            "misses": self._state.misses,
            "evictions": self._state.evictions,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
        }

    async def clear_cache(self) -> None:
        """Clear all cached data and tracking information."""
        # Clear the cache store if it supports clear
        if hasattr(self._cache, "clear"):
            await self._cache.clear()

        # Reset tracking
        async with self._state.lock:
            self._state.key_insert_times.clear()
            self._state.cache_order.clear()
            self._state.key_sizes.clear()
            self._state.range_cache.clear()
            self._state.current_size = 0

    def __repr__(self) -> str:
        """Return string representation of the cache store."""
        return (
            f"{self.__class__.__name__}("
            f"store={self._store!r}, "
            f"cache_store={self._cache!r}, "
            f"max_age_seconds={self.max_age_seconds}, "
            f"max_size={self.max_size}, "
            f"current_size={self._state.current_size}, "
            f"cached_keys={len(self._state.cache_order)})"
        )
