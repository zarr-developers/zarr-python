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

# Nominal byte cost charged to ``max_size`` for a negative (known-absent) entry.
# Such entries carry no data, but each one occupies an index slot (the key plus a
# small ``_Entry`` record), so it is charged a flat overhead.  This lets a single
# ``max_size`` budget bound *total* cache memory — cached values and miss-markers
# together — rather than letting negative entries grow without limit.
_NEGATIVE_ENTRY_SIZE = 128


@dataclass(slots=True)
class _Entry:
    """A single cache slot, tracked in :attr:`_CacheState.entries`.

    ``present=True`` (the default): a value is cached for this key — in the
    Store-backed cache for full keys, or the in-memory range cache for
    byte-range keys — occupying ``size`` bytes.

    ``present=False``: the full key is known-*absent* in the source store (a
    negative-cache entry). It carries no data, but is charged a flat
    ``_NEGATIVE_ENTRY_SIZE`` against ``max_size`` for the index slot it occupies,
    so cached values and miss-markers share one memory budget. Its staleness is
    bounded by ``max_age_seconds``.

    Because every key maps to exactly one ``_Entry``, "present" and "absent" are
    mutually exclusive by construction: a key cannot simultaneously be cached and
    marked missing.
    """

    insert_time: float
    size: int = 0
    present: bool = True


@dataclass(slots=True)
class _CacheState:
    # Single source of truth for every tracked key (full-key and byte-range,
    # present and absent).  Ordered for LRU eviction; ``move_to_end`` marks a key
    # most-recently-used.  Replaces the former parallel cache_order / key_sizes /
    # key_insert_times / missing_keys structures so a key has one unambiguous state.
    entries: OrderedDict[_CacheEntryKey, _Entry] = field(default_factory=OrderedDict)
    current_size: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    negative_hits: int = 0
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
    cache_missing : bool, optional
        Whether to remember full-key misses (negative caching). When True, a full-key
        ``get`` that finds the key absent in the source store records that absence, so
        subsequent ``get``s for the same key return ``None`` without a source round-trip.
        This benefits repeated reads of sparse arrays (most chunks absent). Negative
        entries respect ``max_age_seconds`` and are evicted when the key is written
        (``set``/``set_if_not_exists``). Only full-key reads are affected (not byte-range
        reads or ``exists``). Default is True.

        Notes:

        - With ``max_age_seconds="infinity"`` (the default) a remembered miss never
          expires, so a key written to the source by another process stays invisible
          through this cache. Pair ``cache_missing=True`` with a finite
          ``max_age_seconds`` if the source may be written concurrently.
        - Negative entries share the ``max_size`` budget with cached values: each is
          charged a small flat overhead, and under memory pressure miss-markers are
          evicted (least-recently-used first) before any cached value. A single
          ``max_size`` therefore bounds *total* cache memory. When ``max_size is None``
          both caches are unbounded, so a scan over a very large sparse key space will
          accumulate one small entry per absent key; set ``max_size`` (and/or a finite
          ``max_age_seconds``, or ``cache_missing=False``) for such workloads.

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
    cache_missing: bool
    _state: _CacheState

    def __init__(
        self,
        store: Store,
        *,
        cache_store: Store,
        max_age_seconds: int | str = "infinity",
        max_size: int | None = None,
        cache_set_data: bool = True,
        cache_missing: bool = True,
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
        self.cache_missing = cache_missing
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
            cache_missing=self.cache_missing,
        )
        store._state = self._state
        return store

    def _is_fresh(self, entry_key: _CacheEntryKey) -> bool:
        """Check if a tracked entry (present or absent) is still fresh.

        Uses monotonic time for accurate elapsed time measurement.  A key with no
        entry is treated as not fresh (except under an infinite TTL, matching the
        previous behaviour of routing unseen keys through the cache path).
        """
        if self.max_age_seconds == "infinity":
            return True
        entry = self._state.entries.get(entry_key)
        if entry is None:
            return False
        elapsed = time.monotonic() - entry.insert_time
        return elapsed < self.max_age_seconds

    async def _record_missing(self, key: str) -> None:
        """Record *key* as known-missing (absent in the source store).

        Overwrites any existing slot for *key*, so a key cannot be both cached and
        marked missing.  The marker is charged ``_NEGATIVE_ENTRY_SIZE`` against the
        shared ``max_size`` budget, then the budget is re-enforced (evicting
        absent entries first).  Must be called while holding ``self._state.lock``.
        Staleness is bounded by ``max_age_seconds`` via ``_is_fresh``.
        """
        old = self._state.entries.get(key)
        if old is not None:
            self._state.current_size = max(0, self._state.current_size - old.size)
        self._state.entries[key] = _Entry(
            insert_time=time.monotonic(), size=_NEGATIVE_ENTRY_SIZE, present=False
        )
        self._state.entries.move_to_end(key)
        self._state.current_size += _NEGATIVE_ENTRY_SIZE
        # Re-enforce the shared budget (no further incoming bytes to reserve).
        await self._accommodate_value(0)

    def _evict_missing(self, key: str) -> None:
        """Drop any negative entry for *key* (it is now present or being written).

        Only removes an *absent* slot — a present (cached) value for the same key is
        left untouched.  Must be called while holding ``self._state.lock``.
        """
        entry = self._state.entries.get(key)
        if entry is not None and not entry.present:
            del self._state.entries[key]

    async def _accommodate_value(self, value_size: int) -> None:
        """Evict until ``value_size`` more bytes fit within ``max_size``.

        Eviction is *absent-first*: least-recently-used negative markers are
        dropped before any cached value, because miss-markers are cheap to
        regenerate (just re-probe the source) while cached data is not.  A cached
        value is only evicted once no negative markers remain.  Must be called
        while holding self._state.lock.
        """
        if self.max_size is None:
            return

        while self._state.current_size + value_size > self.max_size:
            lru_key = self._next_eviction_candidate()
            if lru_key is None:
                break
            await self._evict_key(lru_key)

    def _next_eviction_candidate(self) -> _CacheEntryKey | None:
        """Return the next entry to evict, preferring absent markers (LRU-first).

        Walks entries in LRU order: the first absent entry found is returned; if
        none are absent, the least-recently-used present entry is returned.  Must
        be called while holding self._state.lock.
        """
        lru_present: _CacheEntryKey | None = None
        for entry_key, entry in self._state.entries.items():
            if not entry.present:
                return entry_key
            if lru_present is None:
                lru_present = entry_key
        return lru_present

    async def _evict_key(self, entry_key: _CacheEntryKey) -> None:
        """Evict a cache entry.

        Must be called while holding self._state.lock.

        For ``str`` keys the entry is deleted from the Store-backed cache.
        For ``(str, ByteRequest)`` keys the entry is removed from the
        in-memory range cache.
        """
        entry = self._state.entries.pop(entry_key, None)
        key_size = entry.size if entry is not None else 0

        if isinstance(entry_key, str):
            # Absent markers store no value in the backing cache — skip the delete.
            if entry is None or entry.present:
                await self._cache.delete(entry_key)
        else:
            base_key, byte_range = entry_key
            per_key = self._state.range_cache.get(base_key)
            if per_key is not None:
                per_key.pop(byte_range, None)
                if not per_key:
                    del self._state.range_cache[base_key]

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
            # If key already exists, subtract old size first (an absent slot has
            # size 0, so this also cleanly upgrades a negative entry to present).
            old = self._state.entries.get(entry_key)
            if old is not None:
                self._state.current_size -= old.size

            # Make room for the new value
            await self._accommodate_value(value_size)

            # Update tracking atomically.  Assigning to an existing key preserves
            # its LRU position, matching the previous behaviour.
            self._state.entries[entry_key] = _Entry(
                insert_time=time.monotonic(), size=value_size, present=True
            )
            self._state.current_size += value_size

        return True

    async def _update_access_order(self, entry_key: _CacheEntryKey) -> None:
        """Update the access order for LRU tracking."""
        if entry_key in self._state.entries:
            async with self._state.lock:
                self._state.entries.move_to_end(entry_key)

    def _remove_from_tracking(self, entry_key: _CacheEntryKey) -> None:
        """Remove an entry from tracking, reclaiming any bytes it accounted for.

        Must be called while holding self._state.lock.
        """
        entry = self._state.entries.pop(entry_key, None)
        if entry is not None:
            self._state.current_size = max(0, self._state.current_size - entry.size)

    def _invalidate_range_entries(self, key: str) -> None:
        """Remove all byte-range entries for *key* from the range cache and tracking.

        Must be called while holding self._state.lock.
        """
        per_key = self._state.range_cache.pop(key, None)
        if per_key is not None:
            for byte_range in per_key:
                entry_key: _CacheEntryKey = (key, byte_range)
                entry = self._state.entries.pop(entry_key, None)
                if entry is not None:
                    self._state.current_size = max(0, self._state.current_size - entry.size)

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
                    # The key is absent in the source: remember the miss so a repeat
                    # read can short-circuit without a source round-trip.
                    if self.cache_missing:
                        await self._record_missing(key)
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
                # ``_track_entry`` overwrites the key's single slot with a present
                # entry, so any prior negative marker is structurally replaced —
                # no separate negative-cache eviction is needed here.
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
        # Negative cache fast-path (full-key reads only): a fresh "known absent" record
        # short-circuits to None without consulting the positive cache or the source.
        # Checked here, before the positive-entry freshness gate, because a negative-only
        # key has no positive entry and would otherwise be routed straight to the source.
        if self.cache_missing and byte_range is None:
            async with self._state.lock:
                entry = self._state.entries.get(key)
                if entry is not None and not entry.present and self._is_fresh(key):
                    self._state.negative_hits += 1
                    return None

        entry_key: _CacheEntryKey = (key, byte_range) if byte_range is not None else key
        if not self._is_fresh(entry_key):
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
        # Invalidate all cached byte-range entries (source data changed) and drop any
        # negative entry — the key now has a value.
        async with self._state.lock:
            self._invalidate_range_entries(key)
            if self.cache_missing:
                self._evict_missing(key)
        if self.cache_set_data:
            await self._cache.set(key, value)
            await self._track_entry(key, value)
        else:
            await self._cache.delete(key)
            async with self._state.lock:
                self._remove_from_tracking(key)

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        """
        Store data only if the key does not already exist in the source store.

        Parameters
        ----------
        key : str
            The key to store under
        value : Buffer
            The data to store
        """
        await super().set_if_not_exists(key, value)
        # Whether or not the write happened, any negative entry is now unsafe: either
        # we just wrote the key, or it already existed (so the record was already
        # wrong). Evicting unconditionally is always safe. We do not populate the
        # positive cache here — there is no guaranteed-fresh value to store.
        if self.cache_missing:
            async with self._state.lock:
                self._evict_missing(key)

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
        present = sum(1 for entry in self._state.entries.values() if entry.present)
        missing = len(self._state.entries) - present
        return {
            "cache_store_type": type(self._cache).__name__,
            "max_age_seconds": "infinity"
            if self.max_age_seconds == "infinity"
            else self.max_age_seconds,
            "max_size": self.max_size,
            "current_size": self._state.current_size,
            "cache_set_data": self.cache_set_data,
            "cache_missing": self.cache_missing,
            "tracked_keys": len(self._state.entries),
            "cached_keys": present,
            "missing_keys": missing,
        }

    def cache_stats(self) -> dict[str, Any]:
        """Return cache performance statistics.

        ``hit_rate`` reflects positive-cache hits over positive lookups only; a
        negative-cache hit (an absent key served from the negative cache) is reported
        separately as ``negative_hits`` and is counted as neither a hit nor a miss.
        """
        total_requests = self._state.hits + self._state.misses
        hit_rate = self._state.hits / total_requests if total_requests > 0 else 0.0
        return {
            "hits": self._state.hits,
            "misses": self._state.misses,
            "evictions": self._state.evictions,
            "negative_hits": self._state.negative_hits,
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
            self._state.entries.clear()
            self._state.range_cache.clear()
            self._state.current_size = 0
            self._state.negative_hits = 0

    def __repr__(self) -> str:
        """Return string representation of the cache store."""
        cached_keys = sum(1 for entry in self._state.entries.values() if entry.present)
        return (
            f"{self.__class__.__name__}("
            f"store={self._store!r}, "
            f"cache_store={self._cache!r}, "
            f"max_age_seconds={self.max_age_seconds}, "
            f"max_size={self.max_size}, "
            f"current_size={self._state.current_size}, "
            f"cached_keys={cached_keys})"
        )
