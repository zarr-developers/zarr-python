from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from itertools import starmap
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from zarr.core.sync import sync

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Iterable, Sequence
    from types import TracebackType
    from typing import Any, Self

    from zarr.core.buffer import Buffer, BufferPrototype

__all__ = [
    "ByteGetter",
    "ByteSetter",
    "Store",
    "SupportsDeleteSync",
    "SupportsGetSync",
    "SupportsSetRange",
    "SupportsSetSync",
    "SupportsSyncStore",
    "set_or_delete",
]


@dataclass(frozen=True, slots=True)
class RangeByteRequest:
    """Request a specific byte range"""

    start: int
    """The start of the byte range request (inclusive)."""
    end: int
    """The end of the byte range request (exclusive)."""


@dataclass(frozen=True, slots=True)
class OffsetByteRequest:
    """Request all bytes starting from a given byte offset"""

    offset: int
    """The byte offset for the offset range request."""


@dataclass(frozen=True, slots=True)
class SuffixByteRequest:
    """Request up to the last `n` bytes"""

    suffix: int
    """The number of bytes from the suffix to request."""


type ByteRequest = RangeByteRequest | OffsetByteRequest | SuffixByteRequest


class Store(ABC):
    """
    Abstract base class for Zarr stores.
    """

    _read_only: bool
    _is_open: bool

    def __init__(self, *, read_only: bool = False) -> None:
        self._is_open = False
        self._read_only = read_only

    @classmethod
    async def open(cls, *args: Any, **kwargs: Any) -> Self:
        """
        Create and open the store.

        Parameters
        ----------
        *args : Any
            Positional arguments to pass to the store constructor.
        **kwargs : Any
            Keyword arguments to pass to the store constructor.

        Returns
        -------
        Store
            The opened store instance.
        """
        store = cls(*args, **kwargs)
        await store._open()
        return store

    def with_read_only(self, read_only: bool = False) -> Store:
        """
        Return a new store with a new read_only setting.

        The new store points to the same location with the specified new read_only state.
        The returned Store is not automatically opened, and this store is
        not automatically closed.

        Parameters
        ----------
        read_only
            If True, the store will be created in read-only mode. Defaults to False.

        Returns
        -------
            A new store of the same type with the new read only attribute.
        """
        raise NotImplementedError(
            f"with_read_only is not implemented for the {type(self)} store type."
        )

    def __enter__(self) -> Self:
        """Enter a context manager that will close the store upon exiting."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the store."""
        self.close()

    async def _open(self) -> None:
        """
        Open the store.

        Raises
        ------
        ValueError
            If the store is already open.
        """
        if self._is_open:
            raise ValueError("store is already open")
        self._is_open = True

    async def _ensure_open(self) -> None:
        """Open the store if it is not already open."""
        if not self._is_open:
            await self._open()

    async def is_empty(self, prefix: str) -> bool:
        """
        Check if the directory is empty.

        Parameters
        ----------
        prefix : str
            Prefix of keys to check.

        Returns
        -------
        bool
            True if the store is empty, False otherwise.
        """
        if not self.supports_listing:
            raise NotImplementedError
        if prefix != "" and not prefix.endswith("/"):
            prefix += "/"
        async for _ in self.list_prefix(prefix):
            return False
        return True

    async def clear(self) -> None:
        """
        Clear the store.

        Remove all keys and values from the store.
        """
        if not self.supports_deletes:
            raise NotImplementedError
        if not self.supports_listing:
            raise NotImplementedError
        self._check_writable()
        await self.delete_dir("")

    @property
    def read_only(self) -> bool:
        """Is the store read-only?"""
        return self._read_only

    def _check_writable(self) -> None:
        """Raise an exception if the store is not writable."""
        if self.read_only:
            raise ValueError("store was opened in read-only mode and does not support writing")

    @abstractmethod
    def __eq__(self, value: object) -> bool:
        """Equality comparison."""
        ...

    @abstractmethod
    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        """Retrieve the value associated with a given key.

        Parameters
        ----------
        key : str
        prototype : BufferPrototype
            The prototype of the output buffer. Stores may support a default buffer prototype.
        byte_range : ByteRequest, optional
            ByteRequest may be one of the following. If not provided, all data associated with the key is retrieved.
            - RangeByteRequest(int, int): Request a specific range of bytes in the form (start, end). The end is exclusive. If the given range is zero-length or starts after the end of the object, an error will be returned. Additionally, if the range ends after the end of the object, the entire remainder of the object will be returned. Otherwise, the exact requested range will be returned.
            - OffsetByteRequest(int): Request all bytes starting from a given byte offset. This is equivalent to bytes={int}- as an HTTP header.
            - SuffixByteRequest(int): Request the last int bytes. Note that here, int is the size of the request, not the byte offset. This is equivalent to bytes=-{int} as an HTTP header.

        Returns
        -------
        Buffer
        """
        ...

    async def _get_bytes(
        self, key: str, *, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> bytes:
        """
        Retrieve raw bytes from the store asynchronously.

        This is a convenience method that wraps ``get()`` and converts the result
        to bytes. Use this when you need the raw byte content of a stored value.

        Parameters
        ----------
        key : str
            The key identifying the data to retrieve.
        prototype : BufferPrototype
            The buffer prototype to use for reading the data.
        byte_range : ByteRequest, optional
            If specified, only retrieve a portion of the stored data.
            Can be a ``RangeByteRequest``, ``OffsetByteRequest``, or ``SuffixByteRequest``.

        Returns
        -------
        bytes
            The raw bytes stored at the given key.

        Raises
        ------
        FileNotFoundError
            If the key does not exist in the store.

        See Also
        --------
        get : Lower-level method that returns a Buffer object.
        get_bytes : Synchronous version of this method.
        get_json : Asynchronous method for retrieving and parsing JSON data.

        Examples
        --------
        >>> async def example():
        ...     from zarr.core.buffer.cpu import Buffer
        ...     from zarr.storage import MemoryStore
        ...
        ...     store = await MemoryStore.open()
        ...     await store.set("data", Buffer.from_bytes(b"hello world"))
        ...     # No need to specify prototype for MemoryStore
        ...     return await store._get_bytes("data")

        >>> import asyncio
        >>> asyncio.run(example())
        b'hello world'
        """

        buffer = await self.get(key, prototype, byte_range)
        if buffer is None:
            raise FileNotFoundError(key)
        return buffer.to_bytes()

    def _get_bytes_sync(
        self, key: str = "", *, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> bytes:
        """
        Retrieve raw bytes from the store synchronously.

        This is a synchronous wrapper around ``get_bytes()``. It should only
        be called from non-async code. For async contexts, use ``get_bytes()``
        instead.

        Parameters
        ----------
        key : str, optional
            The key identifying the data to retrieve. Defaults to an empty string.
        prototype : BufferPrototype
            The buffer prototype to use for reading the data.
        byte_range : ByteRequest, optional
            If specified, only retrieve a portion of the stored data.
            Can be a ``RangeByteRequest``, ``OffsetByteRequest``, or ``SuffixByteRequest``.

        Returns
        -------
        bytes
            The raw bytes stored at the given key.

        Raises
        ------
        FileNotFoundError
            If the key does not exist in the store.

        Warnings
        --------
        Do not call this method from async functions. Use ``get_bytes()`` instead
        to avoid blocking the event loop.

        See Also
        --------
        get_bytes : Asynchronous version of this method.
        get_json_sync : Synchronous method for retrieving and parsing JSON data.

        Examples
        --------
        >>> from zarr.core.buffer.cpu import Buffer
        >>> from zarr.storage import MemoryStore
        >>> store = MemoryStore()
        >>> store.set_sync("data", Buffer.from_bytes(b"hello world"))
        >>> store._get_bytes_sync("data")  # No need to specify prototype for MemoryStore
        b'hello world'
        """

        return sync(self._get_bytes(key, prototype=prototype, byte_range=byte_range))

    async def _get_json(
        self, key: str, *, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Any:
        """
        Retrieve and parse JSON data from the store asynchronously.

        This is a convenience method that retrieves bytes from the store and
        parses them as JSON.

        Parameters
        ----------
        key : str
            The key identifying the JSON data to retrieve.
        prototype : BufferPrototype
            The buffer prototype to use for reading the data.
        byte_range : ByteRequest, optional
            If specified, only retrieve a portion of the stored data.
            Can be a ``RangeByteRequest``, ``OffsetByteRequest``, or ``SuffixByteRequest``.
            Note: Using byte ranges with JSON may result in invalid JSON.

        Returns
        -------
        Any
            The parsed JSON data. This follows the behavior of ``json.loads()`` and
            can be any JSON-serializable type: dict, list, str, int, float, bool, or None.

        Raises
        ------
        FileNotFoundError
            If the key does not exist in the store.
        json.JSONDecodeError
            If the stored data is not valid JSON.

        See Also
        --------
        get_bytes : Method for retrieving raw bytes.
        get_json_sync : Synchronous version of this method.

        Examples
        --------
        >>> async def example():
        ...     from zarr.core.buffer.cpu import Buffer
        ...     from zarr.storage import MemoryStore
        ...
        ...     store = await MemoryStore.open()
        ...     metadata = {"zarr_format": 3, "node_type": "array"}
        ...     await store.set("zarr.json", Buffer.from_bytes(json.dumps(metadata).encode()))
        ...     # No need to specify prototype for MemoryStore
        ...     return await store._get_json("zarr.json")

        >>> import asyncio
        >>> asyncio.run(example())
        {'zarr_format': 3, 'node_type': 'array'}
        """

        return json.loads(await self._get_bytes(key, prototype=prototype, byte_range=byte_range))

    def _get_json_sync(
        self, key: str = "", *, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Any:
        """
        Retrieve and parse JSON data from the store synchronously.

        This is a synchronous wrapper around ``get_json()``. It should only
        be called from non-async code. For async contexts, use ``get_json()``
        instead.

        Parameters
        ----------
        key : str, optional
            The key identifying the JSON data to retrieve. Defaults to an empty string.
        prototype : BufferPrototype
            The buffer prototype to use for reading the data.
        byte_range : ByteRequest, optional
            If specified, only retrieve a portion of the stored data.
            Can be a ``RangeByteRequest``, ``OffsetByteRequest``, or ``SuffixByteRequest``.
            Note: Using byte ranges with JSON may result in invalid JSON.

        Returns
        -------
        Any
            The parsed JSON data. This follows the behavior of ``json.loads()`` and
            can be any JSON-serializable type: dict, list, str, int, float, bool, or None.

        Raises
        ------
        FileNotFoundError
            If the key does not exist in the store.
        json.JSONDecodeError
            If the stored data is not valid JSON.

        Warnings
        --------
        Do not call this method from async functions. Use ``get_json()`` instead
        to avoid blocking the event loop.

        See Also
        --------
        get_json : Asynchronous version of this method.
        get_bytes_sync : Synchronous method for retrieving raw bytes without parsing.

        Examples
        --------
        >>> from zarr.core.buffer.cpu import Buffer
        >>> from zarr.storage import MemoryStore
        >>> store = MemoryStore()
        >>> metadata = {"zarr_format": 3, "node_type": "array"}
        >>> store.set_sync("zarr.json", Buffer.from_bytes(json.dumps(metadata).encode()))
        >>> store._get_json_sync("zarr.json")  # No need to specify prototype for MemoryStore
        {'zarr_format': 3, 'node_type': 'array'}
        """

        return sync(self._get_json(key, prototype=prototype, byte_range=byte_range))

    @abstractmethod
    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        """Retrieve possibly partial values from given key_ranges.

        Parameters
        ----------
        prototype : BufferPrototype
            The prototype of the output buffer. Stores may support a default buffer prototype.
        key_ranges : Iterable[tuple[str, tuple[int | None, int | None]]]
            Ordered set of key, range pairs, a key may occur multiple times with different ranges

        Returns
        -------
        list of values, in the order of the key_ranges, may contain null/none for missing keys
        """
        ...

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if a key exists in the store.

        Parameters
        ----------
        key : str

        Returns
        -------
        bool
        """
        ...

    @property
    @abstractmethod
    def supports_writes(self) -> bool:
        """Does the store support writes?"""
        ...

    @abstractmethod
    async def set(self, key: str, value: Buffer) -> None:
        """Store a (key, value) pair.

        Parameters
        ----------
        key : str
        value : Buffer
        """
        ...

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        """
        Store a key to ``value`` if the key is not already present.

        Parameters
        ----------
        key : str
        value : Buffer
        """
        # Note for implementers: the default implementation provided here
        # is not safe for concurrent writers. There's a race condition between
        # the `exists` check and the `set` where another writer could set some
        # value at `key` or delete `key`.
        if not await self.exists(key):
            await self.set(key, value)

    async def _set_many(self, values: Iterable[tuple[str, Buffer]]) -> None:
        """
        Insert multiple (key, value) pairs into storage.
        """
        await asyncio.gather(*starmap(self.set, values))

    @property
    def supports_consolidated_metadata(self) -> bool:
        """
        Does the store support consolidated metadata?.

        If it doesn't an error will be raised on requests to consolidate the metadata.
        Returning `False` can be useful for stores which implement their own
        consolidation mechanism outside of the zarr-python implementation.
        """

        return True

    @property
    @abstractmethod
    def supports_deletes(self) -> bool:
        """Does the store support deletes?"""
        ...

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Remove a key from the store

        Parameters
        ----------
        key : str
        """
        ...

    @property
    def supports_partial_writes(self) -> Literal[False]:
        """Does the store support partial writes?

        Partial writes are no longer used by Zarr, so this is always false.
        """
        return False

    @property
    @abstractmethod
    def supports_listing(self) -> bool:
        """Does the store support listing?"""
        ...

    @abstractmethod
    def list(self) -> AsyncIterator[str]:
        """Retrieve all keys in the store.

        Returns
        -------
        AsyncIterator[str]
        """
        # This method should be async, like overridden methods in child classes.
        # However, that's not straightforward:
        # https://stackoverflow.com/questions/68905848

    @abstractmethod
    def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        """
        Retrieve all keys in the store that begin with a given prefix. Keys are returned relative
        to the root of the store.

        Parameters
        ----------
        prefix : str

        Returns
        -------
        AsyncIterator[str]
        """
        # This method should be async, like overridden methods in child classes.
        # However, that's not straightforward:
        # https://stackoverflow.com/questions/68905848

    @abstractmethod
    def list_dir(self, prefix: str) -> AsyncIterator[str]:
        """
        Retrieve all keys and prefixes with a given prefix and which do not contain the character
        “/” after the given prefix.

        Parameters
        ----------
        prefix : str

        Returns
        -------
        AsyncIterator[str]
        """
        # This method should be async, like overridden methods in child classes.
        # However, that's not straightforward:
        # https://stackoverflow.com/questions/68905848

    async def delete_dir(self, prefix: str) -> None:
        """
        Remove all keys and prefixes in the store that begin with a given prefix.
        """
        if not self.supports_deletes:
            raise NotImplementedError
        if not self.supports_listing:
            raise NotImplementedError
        self._check_writable()
        if prefix != "" and not prefix.endswith("/"):
            prefix += "/"
        async for key in self.list_prefix(prefix):
            await self.delete(key)

    def close(self) -> None:
        """Close the store."""
        self._is_open = False

    async def _get_many(
        self, requests: Iterable[tuple[str, BufferPrototype, ByteRequest | None]]
    ) -> AsyncGenerator[tuple[str, Buffer | None], None]:
        """
        Retrieve a collection of objects from storage. In general this method does not guarantee
        that objects will be retrieved in the order in which they were requested, so this method
        yields tuple[str, Buffer | None] instead of just Buffer | None
        """
        for req in requests:
            yield (req[0], await self.get(*req))

    async def get_ranges(
        self,
        key: str,
        byte_ranges: Sequence[ByteRequest | None],
        *,
        prototype: BufferPrototype,
        max_concurrency: int = 10,
        max_gap_bytes: int = 1 << 20,  # 1 MiB
        max_coalesced_bytes: int = 16 << 20,  # 16 MiB
    ) -> AsyncIterator[Sequence[tuple[int, Buffer | None]]]:
        """Read many byte ranges from `key`.

        Yields one batch per underlying I/O operation, each a sequence of
        `(input_index, Buffer | None)` tuples. Batches across yields arrive in
        completion order, not input order. The default implementation built
        into `Store` runs the coalescer over `self.get`, so subclasses get a
        working implementation for free; stores that have a more efficient
        backend (e.g. ranged HTTP, S3 byte-range fetches) should override.

        Parameters
        ----------
        key
            Storage key to read from.
        byte_ranges
            Input ranges. `None` means "the whole value".
        prototype
            Buffer prototype, forwarded to `self.get`.
        max_concurrency
            Maximum number of merged fetches in flight at once.
        max_gap_bytes
            Two `RangeByteRequest`s separated by at most this many bytes may
            be merged into one fetch.
        max_coalesced_bytes
            Upper bound on the size of a single merged fetch.

        Raises
        ------
        BaseExceptionGroup
            Failures from underlying fetches are reported as a
            `BaseExceptionGroup` (PEP 654) and should be handled with
            `except*`. Inner exceptions include `FileNotFoundError` if any
            fetch returns `None` (i.e. `key` is absent), and any exception
            raised by `self.get` for the corresponding range. Pending
            fetches are cancelled as soon as one task fails, so the group
            typically contains a single non-`CancelledError` exception even
            under high concurrency.
        """
        # Local import: zarr.core._coalesce imports symbols from this module.
        from zarr.core._coalesce import coalesced_get

        fetch = partial(self.get, key, prototype)
        async for group in coalesced_get(
            fetch,
            byte_ranges,
            max_concurrency=max_concurrency,
            max_gap_bytes=max_gap_bytes,
            max_coalesced_bytes=max_coalesced_bytes,
        ):
            yield group

    async def getsize(self, key: str) -> int:
        """
        Return the size, in bytes, of a value in a Store.

        Parameters
        ----------
        key : str

        Returns
        -------
        nbytes : int
            The size of the value (in bytes).

        Raises
        ------
        FileNotFoundError
            When the given key does not exist in the store.
        """
        # Note to implementers: this default implementation is very inefficient since
        # it requires reading the entire object. Many systems will have ways to get the
        # size of an object without reading it.
        # avoid circular import
        from zarr.core.buffer.core import default_buffer_prototype

        value = await self.get(key, prototype=default_buffer_prototype())
        if value is None:
            raise FileNotFoundError(key)
        return len(value)

    async def getsize_prefix(self, prefix: str) -> int:
        """
        Return the size, in bytes, of all values under a prefix.

        Parameters
        ----------
        prefix : str
            The prefix of the directory to measure.

        Returns
        -------
        nbytes : int
            The sum of the sizes of the values in the directory (in bytes).

        See Also
        --------
        zarr.Array.nbytes_stored
        Store.getsize

        Notes
        -----
        ``getsize_prefix`` is just provided as a potentially faster alternative to
        listing all the keys under a prefix calling [`Store.getsize`][zarr.abc.store.Store.getsize] on each.

        In general, ``prefix`` should be the path of an Array or Group in the Store.
        Implementations may differ on the behavior when some other ``prefix``
        is provided.
        """
        # TODO: Overlap listing keys with getsize calls.
        # Currently, we load the list of keys into memory and only then move
        # on to getting sizes. Ideally we would overlap those two, which should
        # improve tail latency and might reduce memory pressure (since not all keys
        # would be in memory at once).

        # avoid circular import
        from zarr.core.common import concurrent_map
        from zarr.core.config import config

        keys = [(x,) async for x in self.list_prefix(prefix)]
        limit = config.get("async.concurrency")
        sizes = await concurrent_map(keys, self.getsize, limit=limit)
        return sum(sizes)


@runtime_checkable
class ByteGetter(Protocol):
    async def get(
        self, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Buffer | None: ...


@runtime_checkable
class ByteSetter(Protocol):
    async def get(
        self, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Buffer | None: ...

    async def set(self, value: Buffer) -> None: ...

    async def delete(self) -> None: ...

    async def set_if_not_exists(self, default: Buffer) -> None: ...


# Design note: byte-range writes are exposed as an opt-in protocol rather than a
# method on the `Store` ABC. Only a few stores can do them natively (`LocalStore`,
# `MemoryStore`); most (cloud, zip, read-only) cannot. A universal `Store.set_range`
# with a read-modify-write fallback (as in the Rust `zarrs` crate's
# `WritableStorageTraits::set_partial` + `supports_set_partial`) would let every
# store participate, but for our motivating use case — writing one subchunk without
# rewriting the whole shard — that fallback is a footgun: it would silently rewrite
# an entire (possibly multi-GB) shard, defeating the purpose while appearing to
# succeed. An opt-in protocol instead keeps the cost model honest (a store either
# supports cheap ranged writes or doesn't advertise the capability at all) and keeps
# `set_range` out of the signatures of stores that will never support it.
#
# Stores satisfy this protocol *structurally* (by defining the methods), not by
# nominal inheritance, so a subclass can disclaim it by setting the methods to `None`
# (see `GpuMemoryStore`). Any read-modify-write fallback strategy belongs in the
# caller (the sharding codec), which already has to decide between in-place and
# buffer-and-rewrite — mirroring the zarrs layering (storage writes bytes, codec owns
# strategy) without making every store carry the method. If broad-backend partial
# encoding is wanted later, adding a `supports_set_range()` capability flag plus a
# codec-level fallback is an additive change that does not require retrofitting stores.
@runtime_checkable
class SupportsSetRange(Protocol):
    """Protocol for stores that support writing to a byte range within an existing value.

    Overwrites `len(value)` bytes starting at byte offset `start` within the
    existing stored value for `key`. The key must already exist and the write
    must fit within the existing value (i.e., `start + len(value) <= len(existing)`);
    a write that does not fit raises `ValueError`.

    Concurrency and atomicity
    -------------------------
    **It is entirely the caller's responsibility to ensure consistency.** Any
    coordination needed to keep stored values consistent must be arranged by the
    caller. In particular:

    - Concurrent `set_range` calls that write to **disjoint** byte ranges of the
      same key are safe.
    - Concurrent `set_range` calls that write to **overlapping** ranges of the same
      key have order-dependent, unspecified results. The caller must serialize them.
    - A `set_range` racing against a `set` or `delete` on the same key is a race
      condition, just as concurrent `set` calls are. The caller must serialize these.
    - Writes are **not** guaranteed to be atomic with respect to a process crash:
      a crash mid-write may leave the value partially updated. The caller is
      responsible for any durability or recovery guarantees it requires.

    What an implementation does to honor (or fall short of) this contract — locking,
    atomic replacement, and so on — is documented on the implementing store, not here.
    The intended consumer (the sharding codec writing inner chunks of deterministic
    compressed size) coordinates writes so that they target disjoint ranges.
    """

    async def set_range(self, key: str, value: Buffer, start: int) -> None: ...

    def set_range_sync(self, key: str, value: Buffer, start: int) -> None: ...


@runtime_checkable
class SupportsGetSync(Protocol):
    def get_sync(
        self,
        key: str,
        *,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None: ...


@runtime_checkable
class SupportsSetSync(Protocol):
    def set_sync(self, key: str, value: Buffer) -> None: ...


@runtime_checkable
class SupportsDeleteSync(Protocol):
    def delete_sync(self, key: str) -> None: ...


@runtime_checkable
class SupportsSyncStore(SupportsGetSync, SupportsSetSync, SupportsDeleteSync, Protocol): ...


async def set_or_delete(byte_setter: ByteSetter, value: Buffer | None) -> None:
    """Set or delete a value in a byte setter

    Parameters
    ----------
    byte_setter : ByteSetter
    value : Buffer | None

    Notes
    -----
    If value is None, the key will be deleted.
    """
    if value is None:
        await byte_setter.delete()
    else:
        await byte_setter.set(value)
