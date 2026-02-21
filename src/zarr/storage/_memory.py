from __future__ import annotations

import os
import weakref
from logging import getLogger
from typing import TYPE_CHECKING, Any, Self

from zarr.abc.store import ByteRequest, Store
from zarr.core.buffer import Buffer, gpu
from zarr.core.buffer.core import default_buffer_prototype
from zarr.core.common import concurrent_map
from zarr.storage._utils import (
    _dereference_path,
    _normalize_byte_range_index,
    normalize_path,
    parse_store_url,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable, MutableMapping

    from zarr.core.buffer import BufferPrototype


logger = getLogger(__name__)


class MemoryStore(Store):
    """
    Store for local memory.

    Parameters
    ----------
    store_dict : dict
        Initial data
    read_only : bool
        Whether the store is read-only

    Attributes
    ----------
    supports_writes
    supports_deletes
    supports_listing
    """

    supports_writes: bool = True
    supports_deletes: bool = True
    supports_listing: bool = True

    _store_dict: MutableMapping[str, Buffer]

    def __init__(
        self,
        store_dict: MutableMapping[str, Buffer] | None = None,
        *,
        read_only: bool = False,
    ) -> None:
        super().__init__(read_only=read_only)
        if store_dict is None:
            store_dict = {}
        self._store_dict = store_dict

    def with_read_only(self, read_only: bool = False) -> MemoryStore:
        # docstring inherited
        return type(self)(
            store_dict=self._store_dict,
            read_only=read_only,
        )

    async def clear(self) -> None:
        # docstring inherited
        self._store_dict.clear()

    def __str__(self) -> str:
        return f"memory://{id(self._store_dict)}"

    def __repr__(self) -> str:
        return f"MemoryStore('{self}')"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self._store_dict == other._store_dict
            and self.read_only == other.read_only
        )

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
        try:
            value = self._store_dict[key]
            start, stop = _normalize_byte_range_index(value, byte_range)
            return prototype.buffer.from_buffer(value[start:stop])
        except KeyError:
            return None

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        # docstring inherited

        # All the key-ranges arguments goes with the same prototype
        async def _get(key: str, byte_range: ByteRequest | None) -> Buffer | None:
            return await self.get(key, prototype=prototype, byte_range=byte_range)

        return await concurrent_map(key_ranges, _get, limit=None)

    async def exists(self, key: str) -> bool:
        # docstring inherited
        return key in self._store_dict

    async def set(self, key: str, value: Buffer, byte_range: tuple[int, int] | None = None) -> None:
        # docstring inherited
        self._check_writable()
        await self._ensure_open()
        assert isinstance(key, str)
        if not isinstance(value, Buffer):
            raise TypeError(
                f"MemoryStore.set(): `value` must be a Buffer instance. Got an instance of {type(value)} instead."
            )

        if byte_range is not None:
            buf = self._store_dict[key]
            buf[byte_range[0] : byte_range[1]] = value
            self._store_dict[key] = buf
        else:
            self._store_dict[key] = value

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        # docstring inherited
        self._check_writable()
        await self._ensure_open()
        self._store_dict.setdefault(key, value)

    async def delete(self, key: str) -> None:
        # docstring inherited
        self._check_writable()
        try:
            del self._store_dict[key]
        except KeyError:
            logger.debug("Key %s does not exist.", key)

    async def list(self) -> AsyncIterator[str]:
        # docstring inherited
        for key in self._store_dict:
            yield key

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        # docstring inherited
        # note: we materialize all dict keys into a list here so we can mutate the dict in-place (e.g. in delete_prefix)
        for key in list(self._store_dict):
            if key.startswith(prefix):
                yield key

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        # docstring inherited
        prefix = prefix.rstrip("/")

        if prefix == "":
            keys_unique = {k.split("/")[0] for k in self._store_dict}
        else:
            # Our dictionary doesn't contain directory markers, but we want to include
            # a pseudo directory when there's a nested item and we're listing an
            # intermediate level.
            keys_unique = {
                key.removeprefix(prefix + "/").split("/")[0]
                for key in self._store_dict
                if key.startswith(prefix + "/") and key != prefix
            }

        for key in keys_unique:
            yield key

    async def _get_bytes(
        self,
        key: str = "",
        *,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> bytes:
        """
        Retrieve raw bytes from the memory store asynchronously.

        This is a convenience override that makes the ``prototype`` parameter optional
        by defaulting to the standard buffer prototype. See the base ``Store.get_bytes``
        for full documentation.

        Parameters
        ----------
        key : str, optional
            The key identifying the data to retrieve. Defaults to an empty string.
        prototype : BufferPrototype, optional
            The buffer prototype to use for reading the data. If None, uses
            ``default_buffer_prototype()``.
        byte_range : ByteRequest, optional
            If specified, only retrieve a portion of the stored data.

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
        Store.get_bytes : Base implementation with full documentation.
        get_bytes_sync : Synchronous version of this method.

        Examples
        --------
        >>> store = await MemoryStore.open()
        >>> await store.set("data", Buffer.from_bytes(b"hello"))
        >>> # No need to specify prototype for MemoryStore
        >>> data = await store.get_bytes("data")
        >>> print(data)
        b'hello'
        """
        if prototype is None:
            prototype = default_buffer_prototype()
        return await super()._get_bytes(key, prototype=prototype, byte_range=byte_range)

    def _get_bytes_sync(
        self,
        key: str = "",
        *,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> bytes:
        """
        Retrieve raw bytes from the memory store synchronously.

        This is a convenience override that makes the ``prototype`` parameter optional
        by defaulting to the standard buffer prototype. See the base ``Store.get_bytes``
        for full documentation.

        Parameters
        ----------
        key : str, optional
            The key identifying the data to retrieve. Defaults to an empty string.
        prototype : BufferPrototype, optional
            The buffer prototype to use for reading the data. If None, uses
            ``default_buffer_prototype()``.
        byte_range : ByteRequest, optional
            If specified, only retrieve a portion of the stored data.

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
        Do not call this method from async functions. Use ``get_bytes()`` instead.

        See Also
        --------
        Store.get_bytes_sync : Base implementation with full documentation.
        get_bytes : Asynchronous version of this method.

        Examples
        --------
        >>> store = MemoryStore()
        >>> store.set("data", Buffer.from_bytes(b"hello"))
        >>> # No need to specify prototype for MemoryStore
        >>> data = store.get_bytes("data")
        >>> print(data)
        b'hello'
        """
        if prototype is None:
            prototype = default_buffer_prototype()
        return super()._get_bytes_sync(key, prototype=prototype, byte_range=byte_range)

    async def _get_json(
        self,
        key: str = "",
        *,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Any:
        """
        Retrieve and parse JSON data from the memory store asynchronously.

        This is a convenience override that makes the ``prototype`` parameter optional
        by defaulting to the standard buffer prototype. See the base ``Store.get_json``
        for full documentation.

        Parameters
        ----------
        key : str, optional
            The key identifying the JSON data to retrieve. Defaults to an empty string.
        prototype : BufferPrototype, optional
            The buffer prototype to use for reading the data. If None, uses
            ``default_buffer_prototype()``.
        byte_range : ByteRequest, optional
            If specified, only retrieve a portion of the stored data.
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
        Store.get_json : Base implementation with full documentation.
        get_json_sync : Synchronous version of this method.
        get_bytes : Method for retrieving raw bytes without parsing.

        Examples
        --------
        >>> store = await MemoryStore.open()
        >>> import json
        >>> metadata = {"zarr_format": 3, "node_type": "array"}
        >>> await store.set("zarr.json", Buffer.from_bytes(json.dumps(metadata).encode()))
        >>> # No need to specify prototype for MemoryStore
        >>> data = await store.get_json("zarr.json")
        >>> print(data)
        {'zarr_format': 3, 'node_type': 'array'}
        """
        if prototype is None:
            prototype = default_buffer_prototype()
        return await super()._get_json(key, prototype=prototype, byte_range=byte_range)

    def _get_json_sync(
        self,
        key: str = "",
        *,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Any:
        """
        Retrieve and parse JSON data from the memory store synchronously.

        This is a convenience override that makes the ``prototype`` parameter optional
        by defaulting to the standard buffer prototype. See the base ``Store.get_json``
        for full documentation.

        Parameters
        ----------
        key : str, optional
            The key identifying the JSON data to retrieve. Defaults to an empty string.
        prototype : BufferPrototype, optional
            The buffer prototype to use for reading the data. If None, uses
            ``default_buffer_prototype()``.
        byte_range : ByteRequest, optional
            If specified, only retrieve a portion of the stored data.
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
        Do not call this method from async functions. Use ``get_json()`` instead.

        See Also
        --------
        Store.get_json_sync : Base implementation with full documentation.
        get_json : Asynchronous version of this method.
        get_bytes_sync : Method for retrieving raw bytes without parsing.

        Examples
        --------
        >>> store = MemoryStore()
        >>> import json
        >>> metadata = {"zarr_format": 3, "node_type": "array"}
        >>> store.set("zarr.json", Buffer.from_bytes(json.dumps(metadata).encode()))
        >>> # No need to specify prototype for MemoryStore
        >>> data = store.get_json("zarr.json")
        >>> print(data)
        {'zarr_format': 3, 'node_type': 'array'}
        """
        if prototype is None:
            prototype = default_buffer_prototype()
        return super()._get_json_sync(key, prototype=prototype, byte_range=byte_range)


class GpuMemoryStore(MemoryStore):
    """
    Store for GPU memory.

    Stores every chunk in GPU memory irrespective of the original location.

    The dictionary of buffers to initialize this memory store with *must* be
    GPU Buffers.

    Writing data to this store through ``.set`` will move the buffer to the GPU
    if necessary.

    Parameters
    ----------
    store_dict : MutableMapping, optional
        A mutable mapping with string keys and [zarr.core.buffer.gpu.Buffer][]
        values.
    read_only : bool
        Whether to open the store in read-only mode.
    """

    _store_dict: MutableMapping[str, gpu.Buffer]  # type: ignore[assignment]

    def __init__(
        self,
        store_dict: MutableMapping[str, gpu.Buffer] | None = None,
        *,
        read_only: bool = False,
    ) -> None:
        super().__init__(store_dict=store_dict, read_only=read_only)  # type: ignore[arg-type]

    def __str__(self) -> str:
        return f"gpumemory://{id(self._store_dict)}"

    def __repr__(self) -> str:
        return f"GpuMemoryStore('{self}')"

    @classmethod
    def from_dict(cls, store_dict: MutableMapping[str, Buffer]) -> Self:
        """
        Create a GpuMemoryStore from a dictionary of buffers at any location.

        The dictionary backing the newly created ``GpuMemoryStore`` will not be
        the same as ``store_dict``.

        Parameters
        ----------
        store_dict : mapping
            A mapping of strings keys to arbitrary Buffers. The buffer data
            will be moved into a [`gpu.Buffer`][zarr.core.buffer.gpu.Buffer].

        Returns
        -------
        GpuMemoryStore
        """
        gpu_store_dict = {k: gpu.Buffer.from_buffer(v) for k, v in store_dict.items()}
        return cls(gpu_store_dict)

    async def set(self, key: str, value: Buffer, byte_range: tuple[int, int] | None = None) -> None:
        # docstring inherited
        self._check_writable()
        assert isinstance(key, str)
        if not isinstance(value, Buffer):
            raise TypeError(
                f"GpuMemoryStore.set(): `value` must be a Buffer instance. Got an instance of {type(value)} instead."
            )
        # Convert to gpu.Buffer
        gpu_value = value if isinstance(value, gpu.Buffer) else gpu.Buffer.from_buffer(value)
        await super().set(key, gpu_value, byte_range=byte_range)


# -----------------------------------------------------------------------------
# ManagedMemoryStore and its registry
# -----------------------------------------------------------------------------
# ManagedMemoryStore owns the lifecycle of its backing dict, enabling proper
# weakref-based tracking. This allows memory:// URLs to be resolved back to
# the store's dict within the same process.


class _ManagedStoreDict(dict[str, Buffer]):
    """
    A dict subclass that supports weak references.

    Regular dicts don't support weakrefs, but we need to track managed store dicts
    in a WeakValueDictionary so they can be garbage collected when no longer
    referenced. This subclass adds the necessary __weakref__ slot.
    """

    __slots__ = ("__weakref__",)


class _ManagedStoreDictRegistry:
    """
    Registry for managed store dicts.

    This registry is the source of truth for managed store dicts. It creates
    new dicts, tracks them via weak references, and looks them up by name.
    """

    def __init__(self) -> None:
        self._registry: weakref.WeakValueDictionary[str, _ManagedStoreDict] = (
            weakref.WeakValueDictionary()
        )
        self._counter = 0

    def _generate_name(self) -> str:
        """Generate a unique name for a store."""
        name = str(self._counter)
        self._counter += 1
        return name

    def get_or_create(self, name: str | None = None) -> tuple[_ManagedStoreDict, str]:
        """
        Get an existing managed dict by name, or create a new one.

        Parameters
        ----------
        name : str | None
            The name for the store. If None, a unique name is auto-generated.
            If a store with this name already exists, returns the existing store.
            Names cannot contain '/' characters.

        Returns
        -------
        tuple[_ManagedStoreDict, str]
            The store dict and its name.

        Raises
        ------
        ValueError
            If the name contains '/' characters.
        """
        if name is None:
            name = self._generate_name()
        elif "/" in name:
            raise ValueError(
                f"Store name cannot contain '/': {name!r}. "
                "Use the 'path' parameter to specify a path within the store."
            )

        existing = self._registry.get(name)
        if existing is not None:
            return existing, name

        store_dict = _ManagedStoreDict()
        self._registry[name] = store_dict
        return store_dict, name

    def get(self, name: str) -> _ManagedStoreDict | None:
        """
        Look up a managed store dict by name.

        Parameters
        ----------
        name : str
            The name of the store.

        Returns
        -------
        _ManagedStoreDict | None
            The store dict if found, None otherwise.
        """
        return self._registry.get(name)


_managed_store_dict_registry = _ManagedStoreDictRegistry()


class ManagedMemoryStore(MemoryStore):
    """
    A memory store that owns and manages the lifecycle of its backing dict.

    Unlike ``MemoryStore`` which accepts any ``MutableMapping``, this store
    creates and owns its backing dict internally. This enables proper lifecycle
    management and allows the store to be looked up by its ``memory://`` URL
    within the same process.

    Parameters
    ----------
    name : str | None
        The name for this store, used in the ``memory://`` URL. If None, a unique
        name is auto-generated. If a store with this name already exists, the
        new store will share the same backing dict.
    path : str
        The root path for this store. All keys will be prefixed with this path.
    read_only : bool
        Whether the store is read-only.

    Attributes
    ----------
    name : str
        The name of this store.
    path : str
        The root path of this store.

    Notes
    -----
    The backing dict is tracked via weak references and will be garbage collected
    when no ``ManagedMemoryStore`` instances reference it. URLs pointing to a
    garbage-collected store will fail to resolve.

    See Also
    --------
    MemoryStore : A memory store that accepts any MutableMapping.

    Examples
    --------
    >>> store = ManagedMemoryStore(name="my-data")
    >>> str(store)
    'memory://my-data'
    >>> # Later, resolve the URL back to the store's dict
    >>> store2 = ManagedMemoryStore.from_url("memory://my-data")
    >>> store2._store_dict is store._store_dict
    True
    >>> # Create a store with a path prefix
    >>> store3 = ManagedMemoryStore.from_url("memory://my-data/subdir")
    >>> store3.path
    'subdir'
    """

    _store_dict: _ManagedStoreDict
    _name: str
    path: str

    def __init__(self, name: str | None = None, *, path: str = "", read_only: bool = False) -> None:
        # Skip MemoryStore.__init__ and call Store.__init__ directly
        # because we need to set up _store_dict differently
        Store.__init__(self, read_only=read_only)

        # Get or create a managed dict from the registry
        self._store_dict, self._name = _managed_store_dict_registry.get_or_create(name)
        self.path = normalize_path(path)

    def __str__(self) -> str:
        return _dereference_path(f"memory://{self._name}", self.path)

    def __repr__(self) -> str:
        return f"ManagedMemoryStore('{self}')"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self._store_dict is other._store_dict
            and self.path == other.path
            and self.read_only == other.read_only
        )

    @property
    def name(self) -> str:
        """The name of this store, used in the memory:// URL."""
        return self._name

    @classmethod
    def _from_managed_dict(
        cls,
        managed_dict: _ManagedStoreDict,
        name: str,
        *,
        path: str = "",
        read_only: bool = False,
    ) -> ManagedMemoryStore:
        """Internal: create a store from an existing managed dict."""
        store = object.__new__(cls)
        Store.__init__(store, read_only=read_only)
        store._store_dict = managed_dict
        store._name = name
        store.path = normalize_path(path)
        return store

    def with_read_only(self, read_only: bool = False) -> ManagedMemoryStore:
        # docstring inherited
        return type(self)._from_managed_dict(
            self._store_dict, self._name, path=self.path, read_only=read_only
        )

    @classmethod
    def from_url(cls, url: str, *, read_only: bool = False) -> ManagedMemoryStore:
        """
        Create a ManagedMemoryStore from a memory:// URL.

        This looks up the backing dict in the process-wide registry and creates
        a new store instance that shares the same dict.

        Parameters
        ----------
        url : str
            A URL like "memory://my-store" or "memory://my-store/path/to/data"
            identifying the store and optional path prefix.
        read_only : bool
            Whether the new store should be read-only.

        Returns
        -------
        ManagedMemoryStore
            A store sharing the same backing dict as the original.

        Raises
        ------
        ValueError
            If the URL is not a valid memory:// URL or the store has been
            garbage collected.
        """
        parsed = parse_store_url(url)
        if parsed.scheme != "memory":
            raise ValueError(
                f"Memory store not found for URL '{url}'. "
                "The store may have been garbage collected or the URL is invalid."
            )
        name = parsed.name or ""
        managed_dict = _managed_store_dict_registry.get(name)
        if managed_dict is None:
            raise ValueError(
                f"Memory store not found for URL '{url}'. "
                "The store may have been garbage collected or the URL is invalid."
            )
        return cls._from_managed_dict(managed_dict, name, path=parsed.path, read_only=read_only)

    # Override MemoryStore methods to use path prefix and check process

    async def get(
        self,
        key: str,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        # docstring inherited
        return await super().get(
            _dereference_path(self.path, key), prototype=prototype, byte_range=byte_range
        )

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        # docstring inherited
        key_ranges = [
            (_dereference_path(self.path, key), byte_range) for key, byte_range in key_ranges
        ]
        return await super().get_partial_values(prototype, key_ranges)

    async def exists(self, key: str) -> bool:
        # docstring inherited
        return await super().exists(_dereference_path(self.path, key))

    async def set(self, key: str, value: Buffer, byte_range: tuple[int, int] | None = None) -> None:
        # docstring inherited
        return await super().set(_dereference_path(self.path, key), value, byte_range=byte_range)

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        # docstring inherited
        return await super().set_if_not_exists(_dereference_path(self.path, key), value)

    async def delete(self, key: str) -> None:
        # docstring inherited
        return await super().delete(_dereference_path(self.path, key))

    async def list(self) -> AsyncIterator[str]:
        # docstring inherited
        prefix = self.path + "/" if self.path else ""
        async for key in super().list():
            if key.startswith(prefix):
                yield key.removeprefix(prefix)

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        # docstring inherited
        # Don't use _dereference_path here because it strips trailing slashes,
        # which would break prefix matching (e.g., "fo/" vs "foo/")
        full_prefix = f"{self.path}/{prefix}" if self.path else prefix
        path_prefix = self.path + "/" if self.path else ""
        async for key in super().list_prefix(full_prefix):
            yield key.removeprefix(path_prefix)

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        # docstring inherited
        full_prefix = _dereference_path(self.path, prefix)
        async for key in super().list_dir(full_prefix):
            yield key

    def __reduce__(
        self,
    ) -> tuple[type[ManagedMemoryStore], tuple[str | None], dict[str, Any]]:
        """
        Support pickling of ManagedMemoryStore.

        On unpickle, the store will reconnect to an existing store with the same
        name if one exists in the registry, or create a new empty store otherwise.

        Note that the backing dict data is NOT serialized - only the store's
        identity (name, path, read_only) is preserved. If the original store has
        been garbage collected, the unpickled store will have an empty dict.

        The current process ID is preserved so that cross-process unpickling can be
        detected and will raise an error at unpickle time.
        """
        return (
            self.__class__,
            (self._name,),
            {
                "path": self.path,
                "read_only": self.read_only,
                "created_pid": os.getpid(),
            },
        )

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state after unpickling."""
        # The __reduce__ method returns (cls, (name,), state)
        # Python calls cls(name) then __setstate__(state)
        # But __init__ already set up _store_dict and _name from the registry
        # We just need to restore path and read_only
        self.path = normalize_path(state.get("path", ""))
        self._read_only = state.get("read_only", False)

        # Check for cross-process usage - fail fast at unpickle time
        created_pid = state.get("created_pid")
        if created_pid is not None and created_pid != os.getpid():
            raise RuntimeError(
                f"ManagedMemoryStore '{self._name}' was created in process {created_pid} "
                f"but is being unpickled in process {os.getpid()}. "
                "ManagedMemoryStore instances cannot be shared across processes because "
                "their backing dict is not serialized. Use a persistent store (e.g., "
                "LocalStore, ZipStore) for cross-process data sharing."
            )
