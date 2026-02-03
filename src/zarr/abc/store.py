from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import starmap
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from zarr.core.buffer import Buffer, BufferPrototype
from zarr.core.sync import sync
from zarr.registry import get_buffer_class

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Iterable
    from types import TracebackType
    from typing import Any, Self, TypeAlias

__all__ = ["BufferClassLike", "ByteGetter", "ByteSetter", "Store", "set_or_delete"]

BufferClassLike = type[Buffer] | BufferPrototype
"""An object that is or contains a Buffer class"""


@dataclass
class RangeByteRequest:
    """Request a specific byte range"""

    start: int
    """The start of the byte range request (inclusive)."""
    end: int
    """The end of the byte range request (exclusive)."""


@dataclass
class OffsetByteRequest:
    """Request all bytes starting from a given byte offset"""

    offset: int
    """The byte offset for the offset range request."""


@dataclass
class SuffixByteRequest:
    """Request up to the last `n` bytes"""

    suffix: int
    """The number of bytes from the suffix to request."""


ByteRequest: TypeAlias = RangeByteRequest | OffsetByteRequest | SuffixByteRequest


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

    def _get_default_buffer_class(self) -> type[Buffer]:
        """
        Get the default buffer class.
        """
        return get_buffer_class()

    @abstractmethod
    async def get(
        self,
        key: str,
        prototype: BufferClassLike | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        """Retrieve the value associated with a given key.

        Parameters
        ----------
        key : str
        prototype : BufferLike | None, optional
            The prototype of the output buffer.
            Can be either a Buffer class or an instance of `BufferPrototype`, in which the
            `buffer` attribute will be used.
            If `None`, the default buffer class for this store will be retrieved via the
            ``_get_default_buffer_class`` method.
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
        self,
        key: str,
        *,
        prototype: BufferClassLike | None = None,
        byte_range: ByteRequest | None = None,
    ) -> bytes:
        """
        Retrieve raw bytes from the store asynchronously.

        This is a convenience method that wraps ``get()`` and converts the result
        to bytes. Use this when you need the raw byte content of a stored value.

        Parameters
        ----------
        key : str
            The key identifying the data to retrieve.
        prototype : BufferLike | None, optional
            The prototype of the output buffer.
            Can be either a Buffer class or an instance of `BufferPrototype`, in which the
            `buffer` attribute will be used.
            If `None`, the default buffer prototype for this store will be retrieved via the
            ``_get_default_buffer_class`` method.
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
        >>> store = await MemoryStore.open()
        >>> await store.set("data", Buffer.from_bytes(b"hello world"))
        >>> data = await store._get_bytes("data")
        >>> print(data)
        b'hello world'
        """
        buffer = await self.get(key, prototype, byte_range)
        if buffer is None:
            raise FileNotFoundError(key)
        return buffer.to_bytes()

    def _get_bytes_sync(
        self,
        key: str = "",
        *,
        prototype: BufferClassLike | None = None,
        byte_range: ByteRequest | None = None,
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
        prototype : BufferLike | None, optional
            The prototype of the output buffer.
            Can be either a Buffer class or an instance of `BufferPrototype`, in which the
            `buffer` attribute will be used.
            If `None`, the default buffer prototype for this store will be retrieved via the
            ``_get_default_buffer_class`` method.
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
        >>> store = MemoryStore()
        >>> await store.set("data", Buffer.from_bytes(b"hello world"))
        >>> data = store._get_bytes_sync("data")
        >>> print(data)
        b'hello world'
        """

        return sync(self._get_bytes(key, prototype=prototype, byte_range=byte_range))

    async def _get_json(
        self,
        key: str,
        *,
        prototype: BufferClassLike | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Any:
        """
        Retrieve and parse JSON data from the store asynchronously.

        This is a convenience method that retrieves bytes from the store and
        parses them as JSON.

        Parameters
        ----------
        key : str
            The key identifying the JSON data to retrieve.
        prototype : BufferLike | None, optional
            The prototype of the output buffer.
            Can be either a Buffer class or an instance of `BufferPrototype`, in which the
            `buffer` attribute will be used.
            If `None`, the default buffer prototype for this store will be retrieved via the
            ``_get_default_buffer_class`` method.
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
        >>> store = await MemoryStore.open()
        >>> metadata = {"zarr_format": 3, "node_type": "array"}
        >>> await store.set("zarr.json", Buffer.from_bytes(json.dumps(metadata).encode()))
        >>> data = await store._get_json("zarr.json")
        >>> print(data)
        {'zarr_format': 3, 'node_type': 'array'}
        """

        return json.loads(await self._get_bytes(key, prototype=prototype, byte_range=byte_range))

    def _get_json_sync(
        self,
        key: str = "",
        *,
        prototype: BufferClassLike | None = None,
        byte_range: ByteRequest | None = None,
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
        prototype : BufferLike | None, optional
            The prototype of the output buffer.
            Can be either a Buffer class or an instance of `BufferPrototype`, in which the
            `buffer` attribute will be used.
            If `None`, the default buffer prototype for this store will be retrieved via the
            ``_get_default_buffer_class`` method.
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
        >>> store = MemoryStore()
        >>> metadata = {"zarr_format": 3, "node_type": "array"}
        >>> store.set("zarr.json", Buffer.from_bytes(json.dumps(metadata).encode()))
        >>> data = store._get_json_sync("zarr.json")
        >>> print(data)
        {'zarr_format': 3, 'node_type': 'array'}
        """

        return sync(self._get_json(key, prototype=prototype, byte_range=byte_range))

    @abstractmethod
    async def get_partial_values(
        self,
        prototype: BufferClassLike | None,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        """Retrieve possibly partial values from given key_ranges.

        Parameters
        ----------
        prototype : BufferLike | None
            The prototype of the output buffer.
            Can be either a Buffer class or an instance of `BufferPrototype`, in which the
            `buffer` attribute will be used.
            If `None`, the default buffer prototype for this store will be retrieved via the
            ``_get_default_buffer_class`` method.
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
        self, requests: Iterable[tuple[str, BufferClassLike | None, ByteRequest | None]]
    ) -> AsyncGenerator[tuple[str, Buffer | None], None]:
        """
        Retrieve a collection of objects from storage. In general this method does not guarantee
        that objects will be retrieved in the order in which they were requested, so this method
        yields tuple[str, Buffer | None] instead of just Buffer | None
        """
        for req in requests:
            yield (req[0], await self.get(*req))

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

        value = await self.get(key)
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
