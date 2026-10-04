from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Self

from zarr.abc.store import ByteRequest, Store
from zarr.core.common import concurrent_map
from zarr.storage._wrapper import WrapperStore

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Iterable, Sequence

    from zarr.core.buffer.core import Buffer, BufferPrototype

# Nominal byte cost charged to ``max_size`` for a negative (known-absent) entry.
# Such entries carry no data, but each one occupies an index slot (the key plus a
# small ``_Entry`` record), so it is charged a flat overhead.  This lets a single
# ``max_size`` budget bound *total* cache memory — cached values and miss-markers
# together — rather than letting negative entries grow without limit.
_NEGATIVE_ENTRY_SIZE = 128

# When ``max_size is None`` the shared byte budget cannot bound anything, so the
# number of negative (known-absent) markers is capped here instead: recording a
# marker beyond this count evicts the least-recently-used marker first. This keeps
# a scan over a very large sparse key space from accumulating one entry per absent
# key without bound (at the default charge this cap is ~13 MB of index overhead).
_MAX_NEGATIVE_ENTRIES = 100_000


@dataclass(slots=True)
class _Entry:
    """The full-key slot of a ``_KeyState``.

    ``present=True`` (the default): the key's full value is cached in the
    Store-backed cache, occupying ``size`` bytes.

    ``present=False``: the key is known-*absent* in the source store (a
    negative-cache entry). It carries no data, but is charged a flat
    ``_NEGATIVE_ENTRY_SIZE`` against ``max_size`` for the index slot it occupies,
    so cached values and miss-markers share one memory budget. Its staleness is
    bounded by ``max_age_seconds``.
    """

    insert_time: float
    size: int = 0
    present: bool = True
    last_used: float = 0.0


@dataclass(slots=True)
class _RangeEntry:
    """A cached byte-range of a key's value.

    Held in memory inside the key's ``_KeyState`` so partial reads never touch
    the persistent backend.  ``insert_time`` bounds staleness via
    ``max_age_seconds``; ``last_used`` orders eviction within the key (a monotonic use-tick, not wall time — coarse clocks tie).
    """

    buffer: Buffer
    insert_time: float
    size: int
    last_used: float


@dataclass(slots=True)
class _KeyState:
    """All cached knowledge about a single source key.

    ``full`` is the full-key slot: a cached value, a known-absent marker, or
    ``None`` when nothing is known about the key as a whole.  ``ranges`` holds
    the key's cached byte-range reads.

    Because everything known about a key lives in this one record, every source
    observation for the key — bytes or absence, full-key or ranged — replaces
    the whole record in one locked mutation, so the record only ever holds
    knowledge from a single source generation.  The two halves are therefore
    *mutually exclusive*: a record holds a full slot **or** byte ranges, never
    both (``assert_coherent``).  A cached full value answers ranged reads by
    slicing, so keeping ranges alongside it would be redundant as well as
    divergence-prone, and an absent marker can never sit next to cached bytes.
    """

    full: _Entry | None = None
    ranges: dict[ByteRequest, _RangeEntry] = field(default_factory=dict)

    def assert_coherent(self) -> None:
        """Assert the record invariant: full slot and byte ranges are exclusive.

        Called at the end of the locked sections that fill either half, where it
        fires if that half was recorded without superseding the other.  Note that
        ``python -O`` strips it, so it is a development check, not a guarantee;
        ``CacheStore._assert_invariants`` sweeps the whole state for tests.
        """
        assert self.full is None or not self.ranges, (
            "cache incoherent: key holds a full slot and byte-range data at once"
        )

    @property
    def is_empty(self) -> bool:
        return self.full is None and not self.ranges

    @property
    def tracked_size(self) -> int:
        """Total bytes this record charges against the shared budget."""
        full_size = self.full.size if self.full is not None else 0
        return full_size + sum(r.size for r in self.ranges.values())


@dataclass(slots=True)
class _CacheState:
    # Single source of truth: one record per source key, holding the full-key
    # state (value / absent marker) *and* the key's byte-range buffers.  Ordered
    # for key-level LRU eviction; ``move_to_end`` marks a key most-recently-used.
    # Replaces the former flat entry dict (tuple keys for byte ranges) plus the
    # separate range cache, so a key has one unambiguous, atomically-mutated state.
    entries: OrderedDict[str, _KeyState] = field(default_factory=OrderedDict)
    current_size: int = 0
    # Number of keys whose full slot is an absent marker, maintained incrementally
    # so the negative-marker cap and the eviction-candidate fast path are O(1).
    negative_count: int = 0
    # Monotonic use-counter for within-key LRU ordering. Wall-clock recency ties
    # on coarse clocks (Windows monotonic granularity ~15.6 ms) and mis-evicts;
    # real time is kept only where TTL needs it (``insert_time``).
    use_tick: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    negative_hits: int = 0


class CacheStore(WrapperStore[Store]):
    """
    A dual-store caching implementation for Zarr stores.

    This cache wraps any Store implementation and uses a separate Store instance
    as the cache backend. This provides persistent caching capabilities with
    time-based expiration, size-based eviction, and flexible cache storage options.

    Full-key reads are cached in the Store-backed cache.  A cached full value also
    answers byte-range reads of that key, by asking the cache store for the range
    instead of the source.  Ranges of a key with no cached value are cached in
    memory, inside the key's tracking record, so that partial reads never pollute
    the filesystem (or other persistent backend); ``get_ranges`` and
    ``get_partial_values`` are routed through ``get``, so they use the cache too.

    Both halves share the same ``max_size`` budget, and eviction is LRU over
    *keys*: the least-recently-used key is chosen first (absent markers before
    cached data), then its least-recently-used slot within the record.  Touching
    any of a key's slots therefore keeps the rest of that key resident.

    Parameters
    ----------
    store : Store
        The underlying store to wrap with caching
    cache_store : Store
        The store to use for caching (can be any Store implementation that
        supports deletes)
    max_age_seconds : int or "infinity", optional
        Maximum age of cached entries in seconds. The string "infinity" means
        entries never expire. Default is 300 (five minutes), so that both cached
        values and remembered misses are re-validated against the source at a
        bounded staleness; pass "infinity" to opt out of expiration entirely.
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
        entries respect ``max_age_seconds`` and are dropped when the key is written
        (``set``/``set_if_not_exists``) or when a byte-range read observes data for
        the key. Only full-key reads consult the negative cache (not byte-range reads
        or ``exists``). Default is True.

        Notes:

        - With ``max_age_seconds="infinity"`` a remembered miss never expires, so a
          key written to the source by another process stays invisible through this
          cache until it is written through this instance. The default finite
          ``max_age_seconds`` bounds that staleness; keep it finite if the source
          may be written concurrently.
        - Negative entries share the ``max_size`` budget with cached values: each is
          charged a small flat overhead, and under memory pressure miss-markers are
          evicted (least-recently-used first) before any cached value. A single
          ``max_size`` therefore bounds *total* cache memory. When ``max_size is
          None``, the number of negative markers is instead capped at an internal
          limit (100,000), evicting the least-recently-used marker first.
        - This is store-level, per-key negative caching aimed at the stock ``arr[:]``
          path, which probes every chunk. For very large sparse arrays, prefer the
          array-level sparse-read primitives ``zarr.shards_initialized`` and
          ``zarr.read_regions`` (PR #4028), which touch only populated chunks and so
          never issue the empty-chunk reads this cache would otherwise remember.

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
        max_age_seconds: int | str = 300,
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

    def _slot_is_fresh(self, slot: _Entry | _RangeEntry) -> bool:
        """Check whether an already-looked-up slot is still within ``max_age_seconds``.

        Uses monotonic time for accurate elapsed time measurement.
        """
        if self.max_age_seconds == "infinity":
            return True
        return time.monotonic() - slot.insert_time < self.max_age_seconds

    def _is_fresh(self, key: str, byte_range: ByteRequest | None = None) -> bool:
        """Check if a tracked slot (full-key, or one byte range) is still fresh.

        A key with no tracked slot for the request is treated as not fresh (except
        under an infinite TTL, matching the previous behaviour of routing unseen
        keys through the cache path).
        """
        if self.max_age_seconds == "infinity":
            return True
        state = self._state.entries.get(key)
        if state is None:
            return False
        slot: _Entry | _RangeEntry | None
        slot = state.full if byte_range is None else state.ranges.get(byte_range)
        return slot is not None and self._slot_is_fresh(slot)

    def _has_fresh_value(self, key: str) -> bool:
        """Is a fresh full value for *key* cached (as opposed to absent/unknown)?

        Such a value is the newest thing known about the key, and can serve any
        byte range of it by slicing — see ``_get_try_cache``.
        """
        state = self._state.entries.get(key)
        return (
            state is not None
            and state.full is not None
            and state.full.present
            and self._slot_is_fresh(state.full)
        )

    # ------------------------------------------------------------------
    # tracking-state mutation helpers (all require ``self._state.lock``)
    # ------------------------------------------------------------------

    def _reclaim_full(self, state: _KeyState) -> None:
        """Drop *state*'s full slot, reclaiming its charged bytes.

        Must be called while holding ``self._state.lock``.
        """
        if state.full is not None:
            self._state.current_size = max(0, self._state.current_size - state.full.size)
            if not state.full.present:
                self._state.negative_count -= 1
            state.full = None

    def _reclaim_ranges(self, state: _KeyState) -> None:
        """Drop all of *state*'s byte-range entries, reclaiming their bytes.

        Must be called while holding ``self._state.lock``.
        """
        for range_entry in state.ranges.values():
            self._state.current_size = max(0, self._state.current_size - range_entry.size)
        state.ranges.clear()

    def _drop_key(self, key: str) -> None:
        """Remove everything tracked for *key* (full slot and byte ranges).

        Must be called while holding ``self._state.lock``.
        """
        state = self._state.entries.pop(key, None)
        if state is not None:
            self._reclaim_full(state)
            self._reclaim_ranges(state)

    async def _record_missing(self, key: str) -> None:
        """Record *key* as known-missing (absent in the source store).

        Replaces the key's whole record: any previously cached value bytes and
        byte-range buffers are reclaimed in the same mutation, so the record holds
        only this observation.  The caller has already removed any backing-store
        value for *key*.

        Charges a flat ``_NEGATIVE_ENTRY_SIZE`` against the shared ``max_size``
        budget.  A negative marker is strictly lower priority than cached data: it
        may only displace *other* (older) absent markers to fit, never a cached
        value, and is skipped entirely if the budget is full of cached values
        (detected in O(1) via ``negative_count``).  Must be called while holding
        ``self._state.lock``.  Staleness is bounded by ``max_age_seconds``.
        """
        self._drop_key(key)

        if self.max_size is not None:
            # Make room by evicting older absent markers only — never cached values.
            while self._state.current_size + _NEGATIVE_ENTRY_SIZE > self.max_size:
                if self._state.negative_count == 0:
                    return  # only cached values fill the budget — don't record the miss
                lru_absent = self._lru_absent_key()
                if lru_absent is None:  # pragma: no cover - count implies one exists
                    return
                await self._evict_slot(lru_absent)
        else:
            # No byte budget to share: bound the marker *count* instead, so a scan
            # over a huge sparse key space cannot grow the index without limit.
            while self._state.negative_count >= _MAX_NEGATIVE_ENTRIES:
                lru_absent = self._lru_absent_key()
                if lru_absent is None:  # pragma: no cover - count implies one exists
                    break
                await self._evict_slot(lru_absent)

        now = time.monotonic()
        self._state.use_tick += 1
        state = _KeyState(
            full=_Entry(
                insert_time=now,
                size=_NEGATIVE_ENTRY_SIZE,
                present=False,
                last_used=self._state.use_tick,
            )
        )
        state.assert_coherent()
        # The key was popped above, so this assignment appends it as most-recent.
        self._state.entries[key] = state
        self._state.current_size += _NEGATIVE_ENTRY_SIZE
        self._state.negative_count += 1

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
                # Defensive: the callers guarantee ``value_size <= max_size``,
                # so an empty cache always has room.
                break  # pragma: no cover
            await self._evict_slot(lru_key)

    def _lru_absent_key(self) -> str | None:
        """Return the least-recently-used key marked absent, or ``None``.

        Must be called while holding self._state.lock.
        """
        return next(
            (
                key
                for key, state in self._state.entries.items()
                if state.full is not None and not state.full.present
            ),
            None,
        )

    def _next_eviction_candidate(self) -> str | None:
        """Return the key to evict from next, preferring absent markers (LRU-first).

        When no absent markers exist (``negative_count == 0``) the least-recently-
        used key is the candidate in O(1) — the common all-positive case never scans.
        Otherwise the entries are walked in LRU order for the first marker.  Must be
        called while holding self._state.lock.
        """
        if self._state.negative_count == 0:
            return next(iter(self._state.entries), None)
        candidate = self._lru_absent_key()
        if candidate is None:  # pragma: no cover - negative_count > 0 implies one exists
            return next(iter(self._state.entries), None)
        return candidate

    async def _evict_slot(self, key: str) -> None:
        """Evict *key*'s least-recently-used slot, removing an emptied record.

        Must be called while holding self._state.lock.

        The full-key slot competes with the key's byte-range entries on
        ``last_used``.  Evicting a present full slot deletes the value from the
        Store-backed cache; absent markers and byte-range entries live only in
        memory.
        """
        state = self._state.entries.get(key)
        if state is None:
            # Tracking already dropped (e.g. by a concurrent operation) — make
            # sure no orphaned value lingers in the backing cache.
            await self._cache.delete(key)
            return

        lru_range: ByteRequest | None = None
        lru_range_used = float("inf")
        for byte_range, range_entry in state.ranges.items():
            if range_entry.last_used < lru_range_used:
                lru_range = byte_range
                lru_range_used = range_entry.last_used

        if state.full is not None and (lru_range is None or state.full.last_used <= lru_range_used):
            entry = state.full
            self._reclaim_full(state)
            self._state.evictions += 1
            if entry.present:
                await self._cache.delete(key)
        elif lru_range is not None:
            range_entry = state.ranges.pop(lru_range)
            self._state.current_size = max(0, self._state.current_size - range_entry.size)
            self._state.evictions += 1

        if state.is_empty:
            self._state.entries.pop(key, None)

    async def _track_entry(self, key: str, value: Buffer) -> bool:
        """Register a full-key value in the shared size / LRU tracking.

        Returns ``True`` if the entry was tracked, ``False`` if the value exceeds
        ``max_size`` and was skipped (the key's previous state is dropped either
        way).  Callers should roll back any data they already stored when this
        returns ``False``.

        Must be called while holding ``self._state.lock``, which the callers hold
        across the backing-store write as well, so the value and its record are
        published together.
        """
        value_size = len(value)

        # Drop everything previously known about the key, reclaiming its bytes.
        # The caller observed the key's full value at the source, which supersedes
        # both the old full slot and any byte ranges cached from an older
        # generation (once the new value is tracked, ranged reads are served by
        # slicing it). Dropping the old full slot first also removes it from the
        # eviction candidates so ``_accommodate_value`` cannot select the very slot
        # being (re)tracked — which would double-subtract its size, stop the
        # eviction loop early, and (for a present overwrite) delete the value the
        # caller just wrote to the backing store. The caller has already written
        # the new value, so the backing store is not touched here.
        state = self._state.entries.get(key)
        if state is not None:
            self._reclaim_full(state)
            self._reclaim_ranges(state)
            del self._state.entries[key]

        # The observation is recorded even when the value itself is too large to
        # cache, so the checks above are not skipped by an early return.
        if self.max_size is not None and value_size > self.max_size:
            return False

        # Make room for the new value, then track it (the record is appended
        # as most-recently-used).
        await self._accommodate_value(value_size)
        now = time.monotonic()
        self._state.use_tick += 1
        state = _KeyState(
            full=_Entry(
                insert_time=now,
                size=value_size,
                present=True,
                last_used=self._state.use_tick,
            )
        )
        state.assert_coherent()
        self._state.entries[key] = state
        self._state.current_size += value_size
        return True

    async def _track_range(self, key: str, byte_range: ByteRequest, value: Buffer) -> bool:
        """Register a byte-range read in the key's record.

        Returns ``True`` if the range was cached, ``False`` if the value exceeds
        ``max_size`` (nothing is stored in that case, so there is nothing to roll
        back).

        These bytes were just read from the source, so they supersede the key's
        full slot: an absent marker is disproved by them, and a cached value can
        only still be here if it was stale or unreadable (a fresh one would have
        served this range by slicing instead of reaching the source). Either way
        the full slot is dropped in the same locked mutation — deleting the
        backing value with it, so nothing untracked is left for the cache path to
        serve — and this is the single point where ranges are recorded, so a
        record can never hold both a full slot and range data
        (``assert_coherent``), even when the range itself is too large to cache.

        Must be called while holding ``self._state.lock``.
        """
        value_size = len(value)

        state = self._state.entries.get(key)
        if state is not None:
            if state.full is not None:
                had_value = state.full.present
                self._reclaim_full(state)
                if had_value:
                    await self._cache.delete(key)
            # Drop any slot being replaced so accommodation cannot select it.
            old = state.ranges.pop(byte_range, None)
            if old is not None:
                self._state.current_size = max(0, self._state.current_size - old.size)
            if state.is_empty:
                del self._state.entries[key]

        if self.max_size is not None and value_size > self.max_size:
            return False

        await self._accommodate_value(value_size)
        state = self._state.entries.pop(key, None)
        if state is None:
            state = _KeyState()
        now = time.monotonic()
        self._state.use_tick += 1
        state.ranges[byte_range] = _RangeEntry(
            buffer=value, insert_time=now, size=value_size, last_used=self._state.use_tick
        )
        state.assert_coherent()
        self._state.entries[key] = state
        self._state.current_size += value_size
        return True

    async def _update_access_order(self, key: str, byte_range: ByteRequest | None = None) -> None:
        """Mark a slot — and its key's record — most-recently-used for LRU tracking."""
        async with self._state.lock:
            # Re-check membership under the lock: the record may have been evicted
            # by a concurrent operation between the call and acquiring the lock.
            state = self._state.entries.get(key)
            if state is None:
                return
            self._state.use_tick += 1
            tick = self._state.use_tick
            if byte_range is None:
                if state.full is not None:
                    state.full.last_used = tick
            else:
                range_entry = state.ranges.get(byte_range)
                if range_entry is not None:
                    range_entry.last_used = tick
            self._state.entries.move_to_end(key)

    # ------------------------------------------------------------------
    # get helpers
    # ------------------------------------------------------------------

    async def _cache_miss(
        self,
        key: str,
        byte_range: ByteRequest | None,
        result: Buffer | None,
        prior_slot: _Entry | None,
    ) -> None:
        """Handle a cache miss by storing or cleaning up after a source-store fetch.

        ``prior_slot`` is the key's full slot as observed just *before* the source
        fetch began (``None`` if there was none). It guards both full-key paths
        against a write/miss race: if the key's slot now holds a *different* present
        entry, a concurrent ``set`` completed during the fetch, so neither the stale
        "absent" result nor the stale fetched bytes may overwrite the new value.
        Identity (not timestamps) is used so the check is immune to coarse clocks,
        and the guard, the backing-store write and the tracking mutation share one
        locked section. Known blind spot: a concurrent mutation that leaves no present
        slot behind cannot be detected here — ``cache_set_data=False``,
        ``set_if_not_exists`` (whose override drops tracking rather than inserting a
        present slot), a ``set`` whose value exceeds ``max_size`` (not cached, so no
        slot), and ``delete`` (which removes the record, leaving no identity to
        compare against) — so its effect may be shadowed by this older observation
        until the finite default ``max_age_seconds`` expires the record.

        Byte-range fetches carry no such guard: they never write the backing store,
        and both outcomes (bytes, or absence) only invalidate the key's record, so a
        lost race costs at most one extra source round-trip.
        """
        if result is None:
            if byte_range is None:
                async with self._state.lock:
                    state = self._state.entries.get(key)
                    current = state.full if state is not None else None
                    if current is not None and current.present and current is not prior_slot:
                        # A concurrent write completed during this fetch — the key now
                        # has a (cached) value. Recording the miss would shadow it, so
                        # drop the stale "absent" observation instead.
                        return
                    # The key is absent in the source: drop any (stale) cached value
                    # for it, then either remember the miss (so a repeat read
                    # short-circuits without a source round-trip) or just drop the
                    # tracking record. Either way the key's byte-range entries go in
                    # the same mutation, so full-key and ranged reads cannot diverge.
                    await self._cache.delete(key)
                    if self.cache_missing:
                        await self._record_missing(key)
                    else:
                        self._drop_key(key)
            else:
                # A ranged read came back empty. That is evidence about the key
                # itself, so the key's whole record — cached value included — is
                # dropped: keeping it would let ``get(key)`` serve bytes for a key
                # this same instance just reported nothing for. Deliberately *not*
                # recorded as an absent marker: whether ``get(key, byte_range) is
                # None`` implies the key is absent (rather than the range being
                # unsatisfiable) is a store-contract question left to the follow-up
                # interval work, and invalidating needs no answer to it — worst case
                # it costs one extra source round-trip.
                async with self._state.lock:
                    self._drop_key(key)
                    await self._cache.delete(key)
        else:
            if byte_range is None:
                async with self._state.lock:
                    state = self._state.entries.get(key)
                    current = state.full if state is not None else None
                    if current is not None and current.present and current is not prior_slot:
                        # A concurrent write completed during this fetch — these
                        # bytes are from before it, so leave the newer value alone.
                        return
                    await self._cache.set(key, result)
                    # ``_track_entry`` replaces the key's whole record with a present
                    # entry, so any prior negative marker or stale byte range is
                    # structurally superseded in the same locked section.
                    if not await self._track_entry(key, result):
                        # Value too large for the cache — roll back so the backing
                        # cache holds no untracked (uncounted, unevictable) orphan.
                        await self._cache.delete(key)
            else:
                # ``_track_range`` stores the buffer inside the key's record and
                # drops the key's full slot in the same locked mutation, so a
                # successful ranged read can leave neither an absent marker nor a
                # superseded value behind.
                async with self._state.lock:
                    await self._track_range(key, byte_range, result)

    def _prior_full_slot(self, key: str) -> _Entry | None:
        """Snapshot the key's full slot for ``_cache_miss``'s write/miss race guard."""
        state = self._state.entries.get(key)
        return state.full if state is not None else None

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
        elif self._has_fresh_value(key):
            # A fresh full value is cached: slice the range out of it rather than
            # asking the source. Both answers then come from the one observation,
            # so they cannot disagree, and the round-trip is saved outright. The
            # Store-backed cache does the slicing, so the whole value is never
            # materialised for a small range.
            maybe_cached = await self._cache.get(key, prototype, byte_range)
            if maybe_cached is not None:
                self._state.hits += 1
                await self._update_access_order(key)
                return maybe_cached
        else:
            # Byte-range read — served from the key's in-memory record
            state = self._state.entries.get(key)
            if state is not None:
                range_entry = state.ranges.get(byte_range)
                if range_entry is not None:
                    self._state.hits += 1
                    await self._update_access_order(key, byte_range)
                    return range_entry.buffer

        # Cache miss — fetch from source store
        self._state.misses += 1
        prior_slot = self._prior_full_slot(key) if byte_range is None else None
        result = await super().get(key, prototype, byte_range)
        await self._cache_miss(key, byte_range, result, prior_slot)
        return result

    async def _get_no_cache(
        self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        """Get data directly from source store and update cache."""
        self._state.misses += 1
        prior_slot = self._prior_full_slot(key) if byte_range is None else None
        result = await super().get(key, prototype, byte_range)
        await self._cache_miss(key, byte_range, result, prior_slot)
        return result

    @property
    def _supports_sync_io(self) -> bool:
        # The caching logic lives only in the async get/set/delete overrides;
        # the sync methods inherited from `WrapperStore` delegate straight to
        # the source store, so a sync-capable consumer (the fused codec
        # pipeline) would write and delete around the cache, leaving stale
        # entries that later async reads serve as current data. Opt out of
        # sync IO until the sync surface is cache-aware.
        return False

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
                state = self._state.entries.get(key)
                slot = state.full if state is not None else None
                if slot is not None and not slot.present and self._slot_is_fresh(slot):
                    self._state.negative_hits += 1
                    # Mark the marker most-recently-used so eviction stays LRU:
                    # a frequently-probed absent key should outlive cold markers.
                    self._state.use_tick += 1
                    slot.last_used = self._state.use_tick
                    self._state.entries.move_to_end(key)
                    return None

        # A ranged read also goes through the cache path when the key's *full*
        # value is cached and fresh, because that value can serve any range of it.
        if self._is_fresh(key, byte_range) or (
            byte_range is not None and self._has_fresh_value(key)
        ):
            return await self._get_try_cache(key, prototype, byte_range)
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
        # Either way the key's whole record is replaced in one locked section: the
        # value just written supersedes any cached byte ranges, any stale value and
        # any negative marker. (No ``cache_missing`` gate on the marker: one
        # recorded before the flag was flipped off must still be cleared.)
        if self.cache_set_data:
            async with self._state.lock:
                await self._cache.set(key, value)
                if not await self._track_entry(key, value):
                    # Value too large for the cache — roll back so the backing cache
                    # holds no untracked (uncounted, unevictable) orphan, and no
                    # record of the value it replaced survives either.
                    await self._cache.delete(key)
        else:
            async with self._state.lock:
                self._drop_key(key)
                await self._cache.delete(key)

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
        # Whether or not the write happened, any cached state for this key may now be
        # stale (we may have just written a new value, or it already existed). Drop
        # all of it — byte-range entries, any positive value, and any negative marker
        # — so the next read reflects the source. Invalidating unconditionally is
        # always safe. We do not populate the positive cache here: there is no
        # guaranteed-fresh value to store (the write may have been a no-op).
        async with self._state.lock:
            self._drop_key(key)
        await self._cache.delete(key)

    async def delete(self, key: str) -> None:
        """
        Delete data from both the underlying store and cache.

        Parameters
        ----------
        key : str
            The key to delete
        """
        await super().delete(key)
        # Drop the key's whole record: full slot and byte-range entries together.
        async with self._state.lock:
            self._drop_key(key)
        await self._cache.delete(key)

    async def delete_dir(self, prefix: str) -> None:
        """
        Delete a prefix from the underlying store and drop its cached keys.

        ``WrapperStore.delete_dir`` delegates straight to the source store, so
        without this override no ``delete`` runs for the keys under *prefix* and
        the cache keeps serving them (for the whole ``max_age_seconds`` window)
        after e.g. an ``overwrite=True`` array creation.
        """
        await super().delete_dir(prefix)
        if prefix != "" and not prefix.endswith("/"):
            prefix += "/"
        await self._cache.delete_dir(prefix)
        async with self._state.lock:
            for key in [k for k in self._state.entries if k.startswith(prefix)]:
                self._drop_key(key)

    async def clear(self) -> None:
        """Clear the underlying store, and with it everything cached from it.

        Same bypass as ``delete_dir``: ``WrapperStore.clear`` delegates to the
        source store, which would leave the cache serving values for keys that no
        longer exist anywhere.
        """
        await super().clear()
        await self.clear_cache()

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        """Partial-value reads routed through ``self.get``, so they use the cache.

        ``WrapperStore`` forwards this straight to the source store, which would
        read around the cache: the reads would neither be served from it nor
        recorded in it, and their observations would not supersede the keys'
        records.
        """

        async def _get(key: str, byte_range: ByteRequest | None) -> Buffer | None:
            return await self.get(key, prototype=prototype, byte_range=byte_range)

        return await concurrent_map(key_ranges, _get, limit=None)

    async def get_ranges(
        self,
        key: str,
        byte_ranges: Sequence[ByteRequest | None],
        *,
        prototype: BufferPrototype,
        max_concurrency: int | None = None,
        max_gap_bytes: int | None = None,
        max_coalesced_bytes: int | None = None,
    ) -> AsyncIterator[Sequence[tuple[int, Buffer | None]]]:
        """Byte-range reads routed through the coalescing ``Store.get_ranges``.

        The ``WrapperStore`` delegation would bypass this store's ``get`` — and so
        the cache — for the sharded partial-read path, which is exactly the mixed
        full/range workload this cache targets: the shard index read goes through
        ``get`` while the chunk reads would not. Routing through the ``Store``
        default runs the same coalescer over ``self.get`` instead, so a cached full
        value serves every range of it by slicing and uncached ranges are recorded.
        ``None`` for a coalescing kwarg means "use the ``Store`` default".
        """
        kwargs: dict[str, int] = {}
        if max_concurrency is not None:
            kwargs["max_concurrency"] = max_concurrency
        if max_gap_bytes is not None:
            kwargs["max_gap_bytes"] = max_gap_bytes
        if max_coalesced_bytes is not None:
            kwargs["max_coalesced_bytes"] = max_coalesced_bytes
        async for group in Store.get_ranges(self, key, byte_ranges, prototype=prototype, **kwargs):
            yield group

    async def _get_many(
        self, requests: Iterable[tuple[str, BufferPrototype, ByteRequest | None]]
    ) -> AsyncGenerator[tuple[str, Buffer | None], None]:
        """Batch reads routed through ``self.get`` (see ``get_partial_values``)."""
        async for req in Store._get_many(self, requests):
            yield req

    def cache_info(self) -> dict[str, Any]:
        """Return information about the cache state.

        Counts are per source key: ``tracked_keys`` is the number of keys with any
        tracked state, ``missing_keys`` the number marked absent, and
        ``cached_keys`` the number holding cached data (a full value and/or
        byte-range entries).  A key is never counted as both cached and missing.
        """
        tracked = len(self._state.entries)
        missing = self._state.negative_count
        return {
            "cache_store_type": type(self._cache).__name__,
            "max_age_seconds": "infinity"
            if self.max_age_seconds == "infinity"
            else self.max_age_seconds,
            "max_size": self.max_size,
            "current_size": self._state.current_size,
            "cache_set_data": self.cache_set_data,
            "cache_missing": self.cache_missing,
            "tracked_keys": tracked,
            "cached_keys": tracked - missing,
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
        await self._cache.clear()

        # Reset tracking. Cumulative performance counters (hits/misses/evictions/
        # negative_hits) are lifetime stats and intentionally survive a clear.
        async with self._state.lock:
            self._state.entries.clear()
            self._state.current_size = 0
            self._state.negative_count = 0

    async def _assert_invariants(self) -> None:
        """Check the whole tracking state against the backing cache (test helper).

        ``_KeyState.assert_coherent`` only sees the record being mutated, so this
        sweeps every record and cross-checks the incrementally maintained
        aggregates — ``negative_count`` (which gates both the marker cap and the
        O(1) eviction fast path) and ``current_size`` — against a recount, plus the
        two cross-store facts the records claim: a cached value has a backing value
        of the tracked length, and nothing untracked is left in the backing store.
        It is O(number of tracked keys) and issues one backing read per cached
        value, so it is for tests and debugging, not for the hot path.
        """
        # Local import: only this debug helper needs a prototype of its own.
        from zarr.core.buffer import default_buffer_prototype

        state = self._state
        negative = sum(
            1 for s in state.entries.values() if s.full is not None and not s.full.present
        )
        assert negative == state.negative_count, (
            f"negative_count drift: {state.negative_count} tracked, {negative} counted"
        )
        size = sum(s.tracked_size for s in state.entries.values())
        assert size == state.current_size, (
            f"current_size drift: {state.current_size} tracked, {size} counted"
        )
        assert self.max_size is None or size <= self.max_size, (
            f"cache over budget: {size} > {self.max_size}"
        )
        for key, record in state.entries.items():
            record.assert_coherent()
            assert not record.is_empty, f"empty record retained for {key}"
            if record.full is not None and record.full.present:
                cached = await self._cache.get(key, default_buffer_prototype())
                assert cached is not None, (
                    f"{key} tracked as cached but absent from the cache store"
                )
                assert len(cached) == record.full.size, (
                    f"{key} tracked as {record.full.size} bytes, cache store holds {len(cached)}"
                )
        async for cached_key in self._cache.list():
            tracked = state.entries.get(cached_key)
            full = tracked.full if tracked is not None else None
            assert full is not None, f"untracked value for {cached_key} left in the cache store"
            assert full.present, f"{cached_key} marked absent but holds a value in the cache store"

    def __repr__(self) -> str:
        """Return string representation of the cache store."""
        cached_keys = len(self._state.entries) - self._state.negative_count
        return (
            f"{self.__class__.__name__}("
            f"store={self._store!r}, "
            f"cache_store={self._cache!r}, "
            f"max_age_seconds={self.max_age_seconds}, "
            f"max_size={self.max_size}, "
            f"current_size={self._state.current_size}, "
            f"cached_keys={cached_keys})"
        )
