from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any
from typing_extensions import Literal

from zarr.abc.store import ByteRequest, Store
from zarr.storage._wrapper import WrapperStore
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zarr.core.buffer.core import Buffer, BufferPrototype


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
    >>> from zarr.storage._memory import MemoryStore
    >>> store_a = MemoryStore({})
    >>> store_b = MemoryStore({})
    >>> cached_store = CacheStore(store=store_a, cache_store=store_b, max_age_seconds=10, key_insert_times={})

    """

    _cache: Store
    max_age_seconds: int | Literal["infinity"]
    key_insert_times: dict[str, float]
    cache_set_data: bool

    def __init__(
        self,
        store: Store,
        *,
        cache_store: Store,
        max_age_seconds: int | str = "infinity",
        key_insert_times: dict[str, float]  | None = None,
        cache_set_data: bool = True
    ) -> None:
        super().__init__(store)
        self._cache = cache_store
        self.max_age_seconds = max_age_seconds
        if key_insert_times is None:
            key_insert_times = {}
        else:
            self.key_insert_times = key_insert_times
        self.cache_set_data = cache_set_data

    def _is_key_fresh(self, key: str) -> bool:
        """Check if a cached key is still fresh based on max_age_seconds."""
        if self.max_age_seconds == "infinity":
            return True
        else:
            now = time.monotonic()
            elapsed = now - self.key_insert_times.get(key, 0)
            return elapsed < self.max_age_seconds

    async def _get_try_cache(
        self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        """Try to get data from cache first, falling back to source store."""
        maybe_cached_result = await self._cache.get(key, prototype, byte_range)
        if maybe_cached_result is not None:
            logger.info('_get_try_cache: key %s found in cache', key)
            # Verify the key still exists in source store before returning cached data
            if await super().exists(key):
                return maybe_cached_result
            else:
                # Key no longer exists in source, clean up cache
                logger.info('_get_try_cache: key %s no longer exists in source, cleaning up cache', key)
                await self._cache.delete(key)
                self.key_insert_times.pop(key, None)
                return None
        else:
            logger.info('_get_try_cache: key %s not found in cache, fetching from store', key)
            maybe_fresh_result = await super().get(key, prototype, byte_range)
            if maybe_fresh_result is None:
                await self._cache.delete(key)
            else:
                await self._cache.set(key, maybe_fresh_result)
                self.key_insert_times[key] = time.monotonic()
            return maybe_fresh_result

    async def _get_no_cache(
        self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        """Get data directly from source store and update cache."""
        maybe_fresh_result = await super().get(key, prototype, byte_range)
        if maybe_fresh_result is None:
            # Key doesn't exist in source, remove from cache and tracking
            await self._cache.delete(key)
            self.key_insert_times.pop(key, None)
        else:
            logger.info('_get_no_cache: key %s found in store, setting in cache', key)
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
        if not self._is_key_fresh(key):
            logger.info('get: key %s is not fresh, fetching from store', key)
            return await self._get_no_cache(key, prototype, byte_range)
        else:
            logger.info('get: key %s is fresh, trying cache', key)
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
        logger.info('set: setting key %s in store', key)
        await super().set(key, value)
        if self.cache_set_data:
            logger.info('set: setting key %s in cache', key)
            await self._cache.set(key, value)
            self.key_insert_times[key] = time.monotonic()
        else:
            logger.info('set: deleting key %s from cache', key)
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
        logger.info('delete: deleting key %s from store', key)
        await super().delete(key)
        logger.info('delete: deleting key %s from cache', key)
        await self._cache.delete(key)
        self.key_insert_times.pop(key, None)
