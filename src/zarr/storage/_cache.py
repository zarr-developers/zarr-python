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

    def __init__(self, store: Store, max_size: int):
        super().__init__(read_only=store.read_only)  # Initialize parent with store's read_only state
        self._store = store
        self._max_size = max_size
        self._current_size = 0
        self._keys_cache = None
        self._contains_cache: dict[Any, Any] = {}
        self._listdir_cache: dict[Path, Any] = dict()
        self._values_cache: dict[Path, Any] = OrderedDict()
        self._mutex = Lock()
        self.hits = self.misses = 0
        # self.log_level = log_level
        # self.log_handler = log_handler
        # self._configure_logger(log_level, log_handler)

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

    def clear(self):
        self._store.clear()
        self.invalidate()

    def keys(self):
        with self._mutex:
            return iter(self._keys())

    def _keys(self):
        if self._keys_cache is None:
            self._keys_cache = list(self._store.keys())
        return self._keys_cache

    def listdir(self, path: Path = None):
        with self._mutex:
            try:
                return self._listdir_cache[path]
            except KeyError:
                listing = listdir(self._store, path)
                self._listdir_cache[path] = listing
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

    def _cache_value(self, key: Path, value):
        # cache a value
        value_size = buffer_size(value)
        # check size of the value against max size, as if the value itself exceeds max
        # size then we are never going to cache it
        if self._max_size is None or value_size <= self._max_size:
            self._accommodate_value(value_size)
            self._values_cache[key] = value
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
        if key in self._values_cache:
            value = self._values_cache.pop(key)
            self._current_size -= buffer_size(value)

    def __getitem__(self, key):
        try:
            # first try to obtain the value from the cache
            with self._mutex:
                value = self._values_cache[key]
                # cache hit if no KeyError is raised
                self.hits += 1
                # treat the end as most recently used
                self._values_cache.move_to_end(key)

        except KeyError:
            # cache miss, retrieve value from the store
            value = self._store[key]
            with self._mutex:
                self.misses += 1
                # need to check if key is not in the cache, as it may have been cached
                # while we were retrieving the value from the store
                if key not in self._values_cache:
                    self._cache_value(key, value)

        return value

    def __setitem__(self, key, value):
        self._store[key] = value
        with self._mutex:
            self._invalidate_keys()
            self._invalidate_value(key)
            self._cache_value(key, value)

    def __delitem__(self, key):
        del self._store[key]
        with self._mutex:
            self._invalidate_keys()
            self._invalidate_value(key)
            
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
        # docstring inherited
        self._check_writable()
        path = self.root / key
        if path.is_dir():  # TODO: support deleting directories? shutil.rmtree?
            shutil.rmtree(path)
        else:
            await asyncio.to_thread(path.unlink, True)  # Q: we may want to raise if path is missing
        
        
    async def exists(self, key: str) -> bool:
        # docstring inherited
        path = self.root / key
        return await asyncio.to_thread(path.is_file)
        
    async def get(
        self,
        key: str,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        # docstring inherited
        if prototype is None:
            prototype = default_buffer_prototype()
        if not self._is_open:
            await self._open()
        assert isinstance(key, str)
        path = self.root / key

        try:
            return await asyncio.to_thread(_get, path, prototype, byte_range)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return None
    

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        # docstring inherited
        args = []
        for key, byte_range in key_ranges:
            assert isinstance(key, str)
            path = self.root / key
            args.append((_get, path, prototype, byte_range))
        return await concurrent_map(args, asyncio.to_thread, limit=None)  # TODO: fix limit

        
    async def list(self) -> AsyncIterator[str]:
        # docstring inherited
        to_strip = self.root.as_posix() + "/"
        for p in list(self.root.rglob("*")):
            if p.is_file():
                yield p.as_posix().replace(to_strip, "")
        # This method should be async, like overridden methods in child classes.
        # However, that's not straightforward:
        # https://stackoverflow.com/questions/68905848
        
    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        # docstring inherited
        base = self.root / prefix
        try:
            key_iter = base.iterdir()
            for key in key_iter:
                yield key.relative_to(base).as_posix()
        except (FileNotFoundError, NotADirectoryError):
            pass
        # This method should be async, like overridden methods in child classes.
        # However, that's not straightforward:
        # https://stackoverflow.com/questions/68905848
    
    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        # docstring inherited
        to_strip = self.root.as_posix() + "/"
        prefix = prefix.rstrip("/")
        for p in (self.root / prefix).rglob("*"):
            if p.is_file():
                yield p.as_posix().replace(to_strip, "")
        # This method should be async, like overridden methods in child classes.
        # However, that's not straightforward:
        # https://stackoverflow.com/questions/68905848
    
    async def set(self, key: str, value: Buffer) -> None:
        # docstring inherited
        return await self._set(key, value)
        
    async def set_partial_values(
        self, key_start_values: Iterable[tuple[str, int, bytes | bytearray | memoryview]]
    ) -> None:
        # docstring inherited
        self._check_writable()
        args = []
        for key, start, value in key_start_values:
            assert isinstance(key, str)
            path = self.root / key
            args.append((_put, path, value, start))
        await concurrent_map(args, asyncio.to_thread, limit=None)  # TODO: fix limit
    
    # def _configure_logger(
    #     self, log_level: str = "DEBUG", log_handler: logging.Handler | None = None
    # ) -> None:
    #     self.log_level = log_level
    #     self.logger = logging.getLogger(f"LoggingStore({self._store})")
    #     self.logger.setLevel(log_level)

    #     if not self.logger.hasHandlers():
    #         if not log_handler:
    #             log_handler = self._default_handler()
    #         # Add handler to logger
    #         self.logger.addHandler(log_handler)

    # def _default_handler(self) -> logging.Handler:
    #     """Define a default log handler"""
    #     handler = logging.StreamHandler(stream=sys.stdout)
    #     handler.setLevel(self.log_level)
    #     handler.setFormatter(
    #         logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    #     )
    #     return handler
    
    # @contextmanager
    # def log(self, hint: Any = "") -> Generator[None, None, None]:
    #     """Context manager to log method calls

    #     Each call to the wrapped store is logged to the configured logger and added to
    #     the counter dict.
    #     """
    #     method = inspect.stack()[2].function
    #     op = f"{type(self._store).__name__}.{method}"
    #     if hint:
    #         op = f"{op}({hint})"
    #     self.logger.info(" Calling %s", op)
    #     start_time = time.time()
    #     try:
    #         self.counter[method] += 1
    #         yield
    #     finally:
    #         end_time = time.time()
    #         self.logger.info("Finished %s [%.2f s]", op, end_time - start_time)

    # @property
    # def supports_writes(self) -> bool:
    #     with self.log():
    #         return self._store.supports_writes

    # @property
    # def supports_deletes(self) -> bool:
    #     with self.log():
    #         return self._store.supports_deletes

    # @property
    # def supports_partial_writes(self) -> bool:
    #     with self.log():
    #         return self._store.supports_partial_writes

    # @property
    # def supports_listing(self) -> bool:
    #     with self.log():
    #         return self._store.supports_listing
    
    