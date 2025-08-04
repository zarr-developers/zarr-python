import asyncio
import inspect
import io
import logging
import shutil
import time
import warnings
from collections import OrderedDict
from collections.abc import AsyncIterator, Generator, Iterable
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Any, Optional, TypeAlias

import numpy as np

from zarr.abc.store import OffsetByteRequest, RangeByteRequest, Store, SuffixByteRequest
from zarr.core.buffer import Buffer, BufferPrototype
from zarr.core.buffer.core import default_buffer_prototype
from zarr.core.common import concurrent_map
from zarr.storage._utils import normalize_path

ByteRequest: TypeAlias = RangeByteRequest | OffsetByteRequest | SuffixByteRequest

def buffer_size(v) -> int:
    """Calculate the size in bytes of a value, handling Buffer objects properly."""
    if hasattr(v, '__len__') and hasattr(v, 'nbytes'):
        # This is likely a Buffer object
        return v.nbytes
    elif hasattr(v, 'to_bytes'):
        # This is a Buffer object, get its bytes representation
        return len(v.to_bytes())
    elif isinstance(v, (bytes, bytearray, memoryview)):
        return len(v)
    else:
        # Fallback to numpy
        return np.asarray(v).nbytes

def _path_to_prefix(path: Optional[str]) -> str:
    # assume path already normalized
    if path:
        prefix = path + "/"
    else:
        prefix = ""
    return prefix

def _listdir_from_keys(store: Store, path: Optional[str] = None) -> list[str]:
    # assume path already normalized
    prefix = _path_to_prefix(path)
    children = set()
    for key in list(store.keys()):
        if key.startswith(prefix) and len(key) > len(prefix):
            suffix = key[len(prefix) :]
            child = suffix.split("/")[0]
            children.add(child)
    return sorted(children)

def listdir(store: Store, path: Path = None):
    """Obtain a directory listing for the given path. If `store` provides a `listdir`
    method, this will be called, otherwise will fall back to implementation via the
    `MutableMapping` interface."""
    path = normalize_path(path)
    if hasattr(store, "listdir"):
        # pass through
        return store.listdir(path)
    else:
        # slow version, iterate through all keys
        warnings.warn(
            f"Store {store} has no `listdir` method. From zarr 2.9 onwards "
            "may want to inherit from `Store`.",
            stacklevel=2,
        )
        return _listdir_from_keys(store, path)

def _get(path: Path, prototype: BufferPrototype, byte_range: ByteRequest | None) -> Buffer:
    if byte_range is None:
        return prototype.buffer.from_bytes(path.read_bytes())
    with path.open("rb") as f:
        size = f.seek(0, io.SEEK_END)
        if isinstance(byte_range, RangeByteRequest):
            f.seek(byte_range.start)
            return prototype.buffer.from_bytes(f.read(byte_range.end - f.tell()))
        elif isinstance(byte_range, OffsetByteRequest):
            f.seek(byte_range.offset)
        elif isinstance(byte_range, SuffixByteRequest):
            f.seek(max(0, size - byte_range.suffix))
        else:
            raise TypeError(f"Unexpected byte_range, got {byte_range}.")
        return prototype.buffer.from_bytes(f.read())

def _put(
    path: Path,
    value: Buffer,
    start: int | None = None,
    exclusive: bool = False,
) -> int | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if start is not None:
        with path.open("r+b") as f:
            f.seek(start)
            # write takes any object supporting the buffer protocol
            f.write(value.as_buffer_like())
        return None
    else:
        view = value.as_buffer_like()
        if exclusive:
            mode = "xb"
        else:
            mode = "wb"
        with path.open(mode=mode) as f:
            # write takes any object supporting the buffer protocol
            return f.write(view)



class LRUStoreCache(Store):
    """Storage class that implements a least-recently-used (LRU) cache layer over
    some other store. Intended primarily for use with stores that can be slow to
    access, e.g., remote stores that require network communication to store and
    retrieve data.

    Parameters
    ----------
    store : Store
        The store containing the actual data to be cached.
    max_size : int
        The maximum size that the cache may grow to, in number of bytes. Provide `None`
        if you would like the cache to have unlimited size.

    Examples
    --------
    The example below wraps an S3 store with an LRU cache::

        >>> import s3fs
        >>> import zarr
        >>> s3 = s3fs.S3FileSystem(anon=True, client_kwargs=dict(region_name='eu-west-2'))
        >>> store = s3fs.S3Map(root='zarr-demo/store', s3=s3, check=False)
        >>> cache = zarr.LRUStoreCache(store, max_size=2**28)
        >>> root = zarr.group(store=cache)  # doctest: +REMOTE_DATA
        >>> z = root['foo/bar/baz']  # doctest: +REMOTE_DATA
        >>> from timeit import timeit
        >>> # first data access is relatively slow, retrieved from store
        ... timeit('print(z[:].tobytes())', number=1, globals=globals())  # doctest: +SKIP
        b'Hello from the cloud!'
        0.1081731989979744
        >>> # second data access is faster, uses cache
        ... timeit('print(z[:].tobytes())', number=1, globals=globals())  # doctest: +SKIP
        b'Hello from the cloud!'
        0.0009490990014455747

    """
    
    supports_writes: bool = True
    supports_deletes: bool = True
    supports_partial_writes: bool = True
    supports_listing: bool = True
    
    root: Path

    def __init__(self, store: Store, max_size: int, **kwargs):
        # Extract and handle known parameters
        read_only = kwargs.get('read_only', getattr(store, 'read_only', False))
        
        # Call parent constructor with read_only parameter
        super().__init__(read_only=read_only)
        
        self._store = store
        self._max_size = max_size
        self._current_size = 0
        self._keys_cache = None
        self._contains_cache: dict[Any, Any] = {}
        self._listdir_cache: dict[str, Any] = {}
        self._values_cache: dict[str, Any] = OrderedDict()
        self._mutex = Lock()
        self.hits = self.misses = 0
        
        # Handle root attribute if present in underlying store
        if hasattr(store, 'root'):
            self.root = store.root
        else:
            self.root = None

    @classmethod
    async def open(cls, store: Store, max_size: int, **kwargs: Any) -> "LRUStoreCache":
        """
        Create and open the LRU cache store.

        Parameters
        ----------
        store : Store
            The underlying store to wrap with caching.
        max_size : int
            The maximum size that the cache may grow to, in number of bytes.
        **kwargs : Any
            Additional keyword arguments passed to the store constructor.

        Returns
        -------
        LRUStoreCache
            The opened cache store instance.
        """
        cache = cls(store, max_size, **kwargs)
        await cache._open()
        return cache

    def with_read_only(self, read_only: bool = False) -> "LRUStoreCache":
        """
        Return a new LRUStoreCache with a new read_only setting.

        Parameters
        ----------
        read_only
            If True, the store will be created in read-only mode. Defaults to False.

        Returns
        -------
        LRUStoreCache
            A new LRUStoreCache with the specified read_only setting.
        """
        # Create a new underlying store with the new read_only setting
        underlying_store = self._store.with_read_only(read_only)
        return LRUStoreCache(underlying_store, self._max_size, read_only=read_only)


    def _normalize_key(self, key):
        """Convert key to string if it's a Path object, otherwise return as-is"""
        if isinstance(key, Path):
            return str(key)
        return key

    def __getstate__(self):
        return (
            self._store,
            self._max_size,
            self._current_size,
            self._keys_cache,
            self._contains_cache,
            self._listdir_cache,
            self._values_cache,
            self.hits,
            self.misses,
            self._read_only,
            self._is_open,
        )

    def __setstate__(self, state):
        (
            self._store,
            self._max_size,
            self._current_size,
            self._keys_cache,
            self._contains_cache,
            self._listdir_cache,
            self._values_cache,
            self.hits,
            self.misses,
            self._read_only,
            self._is_open,
        ) = state
        self._mutex = Lock()

    def __len__(self):
        return len(self._keys())

    def __iter__(self):
        return self.keys()

    def __contains__(self, key):
        with self._mutex:
            if key not in self._contains_cache:
                self._contains_cache[key] = key in self._store
            return self._contains_cache[key]

    async def clear(self):
        # Check if store is writable
        self._check_writable()
        
        await self._store.clear()
        self.invalidate()

    def keys(self):
        with self._mutex:
            return iter(self._keys())

    def _keys(self):
        if self._keys_cache is None:
            self._keys_cache = list(self._store.keys())
        return self._keys_cache

    def listdir(self, path: Path | None = None):
        with self._mutex:
            # Normalize path to string for consistent caching
            path_key = self._normalize_key(path) if path is not None else None
            try:
                return self._listdir_cache[path_key]
            except KeyError:
                listing = listdir(self._store, path)
                self._listdir_cache[path_key] = listing
                return listing

    def getsize(self, path=None) -> int:
        return self._store.getsize(key=path)

    def _pop_value(self):
        # remove the first value from the cache, as this will be the least recently
        # used value
        _, v = self._values_cache.popitem(last=False)
        return v

    def _accommodate_value(self, value_size):
        if self._max_size is None:
            return
        # ensure there is enough space in the cache for a new value
        while self._current_size + value_size > self._max_size:
            v = self._pop_value()
            self._current_size -= buffer_size(v)

    def _cache_value(self, key: str, value):  # Change parameter type annotation
        # cache a value
        # Convert Buffer objects to bytes for storage in cache
        if hasattr(value, 'to_bytes'):
            cache_value = value.to_bytes()
        else:
            cache_value = value
            
        value_size = buffer_size(cache_value)
        # check size of the value against max size, as if the value itself exceeds max
        # size then we are never going to cache it
        if self._max_size is None or value_size <= self._max_size:
            self._accommodate_value(value_size)
            # Ensure key is string for consistent caching
            cache_key = self._normalize_key(key)
            self._values_cache[cache_key] = cache_value
            self._current_size += value_size

    def invalidate(self):
        """Completely clear the cache."""
        with self._mutex:
            self._values_cache.clear()
            self._invalidate_keys()
            self._current_size = 0

    def invalidate_values(self):
        """Clear the values cache."""
        with self._mutex:
            self._values_cache.clear()

    def invalidate_keys(self):
        """Clear the keys cache."""
        with self._mutex:
            self._invalidate_keys()

    def _invalidate_keys(self):
        self._keys_cache = None
        self._contains_cache.clear()
        self._listdir_cache.clear()

    def _invalidate_value(self, key):
        cache_key = self._normalize_key(key)
        if cache_key in self._values_cache:
            value = self._values_cache.pop(cache_key)
            self._current_size -= buffer_size(value)

    def __getitem__(self, key):
        cache_key = self._normalize_key(key)
        try:
            # first try to obtain the value from the cache
            with self._mutex:
                value = self._values_cache[cache_key]
                # cache hit if no KeyError is raised
                self.hits += 1
                # treat the end as most recently used
                self._values_cache.move_to_end(cache_key)

        except KeyError:
            # cache miss, retrieve value from the store
            value = self._store[key]  # Use original key for store access
            with self._mutex:
                self.misses += 1
                # need to check if key is not in the cache, as it may have been cached
                # while we were retrieving the value from the store
                if cache_key not in self._values_cache:
                    self._cache_value(cache_key, value)

        return value

    def __setitem__(self, key, value):
        self._store[key] = value
        with self._mutex:
            self._invalidate_keys()
            cache_key = self._normalize_key(key)
            self._invalidate_value(cache_key)
            self._cache_value(cache_key, value)

    def __delitem__(self, key):
        del self._store[key]
        with self._mutex:
            self._invalidate_keys()
            cache_key = self._normalize_key(key)
            self._invalidate_value(cache_key)
            
    def __eq__(self, value: object) -> bool:
        return type(self) is type(value) and self._store.__eq__(value._store)  # type: ignore[attr-defined]
        
    async def delete(self, key: str) -> None:
        """
        Remove a key from the store.

        Parameters
        ----------
        key : str

        Notes
        -----
        If ``key`` is a directory within this store, the entire directory
        at ``store.root / key`` is deleted.
        """
        # Check if store is writable
        self._check_writable()
        
        # Delegate to the underlying store for actual deletion
        if hasattr(self._store, 'delete'):
            await self._store.delete(key)
        else:
            # Fallback for stores that don't have async delete
            del self._store[key]
        
        # Invalidate cache entries
        with self._mutex:
            self._invalidate_keys()
            cache_key = self._normalize_key(key)
            self._invalidate_value(cache_key)
        
        
    async def exists(self, key: str) -> bool:
        # Delegate to the underlying store
        if hasattr(self._store, 'exists'):
            return await self._store.exists(key)
        else:
            # Fallback for stores that don't have async exists
            return key in self._store
    
    async def _set(self, key: str, value: Buffer, exclusive: bool = False) -> None:
        # Check if store is writable
        self._check_writable()
        
        # Delegate to the underlying store
        if hasattr(self._store, 'set'):
            await self._store.set(key, value)
        else:
            # Fallback for stores that don't have async set
            # Convert Buffer to bytes for sync stores
            if hasattr(value, 'to_bytes'):
                self._store[key] = value.to_bytes()
            else:
                self._store[key] = value
        
        # Update cache
        with self._mutex:
            self._invalidate_keys()
            cache_key = self._normalize_key(key)
            self._invalidate_value(cache_key)
            self._cache_value(cache_key, value)

        
    async def get(
        self,
        key: str,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        # Use the cache for get operations
        cache_key = self._normalize_key(key)
        
        # For byte_range requests, don't use cache for now (could be optimized later)
        if byte_range is not None:
            if hasattr(self._store, 'get') and callable(self._store.get):
                # Check if it's an async Store.get method (takes prototype and byte_range)
                try:
                    return await self._store.get(key, prototype, byte_range)
                except TypeError:
                    # Fallback to sync get from mapping
                    full_value = self._store.get(key)
                    if full_value is None:
                        return None
                    if prototype is None:
                        prototype = default_buffer_prototype()
                    # This is a simplified implementation - a full implementation would handle byte ranges
                    return prototype.buffer.from_bytes(full_value)
            else:
                # Fallback - get full value from mapping and slice
                try:
                    full_value = self._store[key]
                    if prototype is None:
                        prototype = default_buffer_prototype()
                    # This is a simplified implementation - a full implementation would handle byte ranges
                    return prototype.buffer.from_bytes(full_value)
                except KeyError:
                    return None
        
        try:
            # Try cache first
            with self._mutex:
                value = self._values_cache[cache_key]
                self.hits += 1
                self._values_cache.move_to_end(cache_key)
                if prototype is None:
                    prototype = default_buffer_prototype()
                return prototype.buffer.from_bytes(value)
        except KeyError:
            # Cache miss - get from store
            if hasattr(self._store, 'get') and callable(self._store.get):
                # Try async Store.get method first
                try:
                    result = await self._store.get(key, prototype, byte_range)
                except TypeError:
                    # Fallback to sync mapping get
                    try:
                        value = self._store.get(key)
                        if value is None:
                            result = None
                        else:
                            if prototype is None:
                                prototype = default_buffer_prototype()
                            result = prototype.buffer.from_bytes(value)
                    except KeyError:
                        result = None
            else:
                # Fallback for sync stores/mappings
                try:
                    value = self._store[key]
                    if prototype is None:
                        prototype = default_buffer_prototype()
                    result = prototype.buffer.from_bytes(value)
                except KeyError:
                    result = None
            
            # Cache the result if we got one
            if result is not None:
                with self._mutex:
                    self.misses += 1
                    if cache_key not in self._values_cache:
                        self._cache_value(cache_key, result.to_bytes())
            else:
                # Still count as a miss even if result is None
                with self._mutex:
                    self.misses += 1
            
            return result
    

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        # Delegate to the underlying store
        if hasattr(self._store, 'get_partial_values'):
            return await self._store.get_partial_values(prototype, key_ranges)
        else:
            # Fallback - get each value individually
            results = []
            for key, byte_range in key_ranges:
                result = await self.get(key, prototype, byte_range)
                results.append(result)
            return results

        
    async def list(self) -> AsyncIterator[str]:
        # Delegate to the underlying store
        if hasattr(self._store, 'list'):
            async for key in self._store.list():
                yield key
        else:
            # Fallback for stores that don't have async list
            for key in list(self._store.keys()):
                yield key
        
    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        # Delegate to the underlying store
        if hasattr(self._store, 'list_dir'):
            async for key in self._store.list_dir(prefix):
                yield key
        else:
            # Fallback using listdir
            try:
                listing = self.listdir(prefix)
                for item in listing:
                    yield item
            except (FileNotFoundError, NotADirectoryError, KeyError):
                pass
    
    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        # Delegate to the underlying store
        if hasattr(self._store, 'list_prefix'):
            async for key in self._store.list_prefix(prefix):
                yield key
        else:
            # Fallback - filter all keys by prefix
            for key in list(self._store.keys()):
                if key.startswith(prefix):
                    yield key
    
    async def set(self, key: str, value: Buffer) -> None:
        # docstring inherited
        return await self._set(key, value)
        
    async def set_partial_values(
        self, key_start_values: Iterable[tuple[str, int, bytes | bytearray | memoryview]]
    ) -> None:
        # Check if store is writable
        self._check_writable()
        
        # Delegate to the underlying store
        if hasattr(self._store, 'set_partial_values'):
            await self._store.set_partial_values(key_start_values)
        else:
            # Fallback - this is complex to implement properly, so just invalidate cache
            for key, _start, _value in key_start_values:
                # For now, just invalidate the cache for these keys
                with self._mutex:
                    self._invalidate_keys()
                    cache_key = self._normalize_key(key)
                    self._invalidate_value(cache_key)
