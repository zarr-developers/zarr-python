from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self, TypeAlias

from zarr.abc.store import ByteRequest, Store
from zarr.core.buffer import Buffer, default_buffer_prototype
from zarr.core.common import (
    ANY_ACCESS_MODE,
    ZARR_JSON,
    ZARRAY_JSON,
    ZGROUP_JSON,
    AccessModeLiteral,
    ZarrFormat,
)
from zarr.errors import ContainsArrayAndGroupError, ContainsArrayError, ContainsGroupError
from zarr.storage._local import LocalStore
from zarr.storage._memory import MemoryStore
from zarr.storage._utils import normalize_path

_has_fsspec = importlib.util.find_spec("fsspec")
if _has_fsspec:
    from fsspec.mapping import FSMap
else:
    FSMap = None

if TYPE_CHECKING:
    from zarr.core.buffer import BufferPrototype


def _dereference_path(root: str, path: str) -> str:
    assert isinstance(root, str)
    assert isinstance(path, str)
    root = root.rstrip("/")
    path = f"{root}/{path}" if root else path
    return path.rstrip("/")


class StorePath:
    """
    Path-like interface for a Store.

    Parameters
    ----------
    store : Store
        The store to use.
    path : str
        The path within the store.
    """

    store: Store
    path: str

    def __init__(self, store: Store, path: str = "") -> None:
        self.store = store
        self.path = normalize_path(path)

    @property
    def read_only(self) -> bool:
        return self.store.read_only

    @classmethod
    async def _create_open_instance(cls, store: Store, path: str) -> Self:
        """Helper to create and return a StorePath instance."""
        await store._ensure_open()
        return cls(store, path)

    @classmethod
    async def open(cls, store: Store, path: str, mode: AccessModeLiteral | None = None) -> Self:
        """
        Open StorePath based on the provided mode.

        * If the mode is None, return an opened version of the store with no changes.
        * If the mode is 'r+', 'w-', 'w', or 'a' and the store is read-only, raise a ValueError.
        * If the mode is 'r' and the store is not read-only, return a copy of the store with read_only set to True.
        * If the mode is 'w-' and the store is not read-only and the StorePath contains keys, raise a FileExistsError.
        * If the mode is 'w'  and the store is not read-only, delete all keys nested within the StorePath.

        Parameters
        ----------
        mode : AccessModeLiteral
            The mode to use when initializing the store path.

            The accepted values are:

            - ``'r'``: read only (must exist)
            - ``'r+'``: read/write (must exist)
            - ``'a'``: read/write (create if doesn't exist)
            - ``'w'``: read/write (overwrite if exists)
            - ``'w-'``: read/write (create if doesn't exist).

        Raises
        ------
        FileExistsError
            If the mode is 'w-' and the store path already exists.
        ValueError
            If the mode is not "r" and the store is read-only, or
        """

        # fastpath if mode is None
        if mode is None:
            return await cls._create_open_instance(store, path)

        if mode not in ANY_ACCESS_MODE:
            raise ValueError(f"Invalid mode: {mode}, expected one of {ANY_ACCESS_MODE}")

        if store.read_only:
            # Don't allow write operations on a read-only store
            if mode != "r":
                raise ValueError(
                    f"Store is read-only but mode is {mode!r}. Create a writable store or use 'r' mode."
                )
            self = await cls._create_open_instance(store, path)
        elif mode == "r":
            # Create read-only copy for read mode on writable store
            try:
                read_only_store = store.with_read_only(True)
            except NotImplementedError as e:
                raise ValueError(
                    "Store is not read-only but mode is 'r'. Unable to create a read-only copy of the store. "
                    "Please use a read-only store or a storage class that implements .with_read_only()."
                ) from e
            self = await cls._create_open_instance(read_only_store, path)
        else:
            # writable store and writable mode
            self = await cls._create_open_instance(store, path)

        # Handle mode-specific operations
        match mode:
            case "w-":
                if not await self.is_empty():
                    raise FileExistsError(
                        f"Cannot create '{path}' with mode 'w-' because it already contains data. "
                        f"Use mode 'w' to overwrite or 'a' to append."
                    )
            case "w":
                await self.delete_dir()
        return self

    async def get(
        self,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        """
        Read bytes from the store.

        Parameters
        ----------
        prototype : BufferPrototype, optional
            The buffer prototype to use when reading the bytes.
        byte_range : ByteRequest, optional
            The range of bytes to read.

        Returns
        -------
        buffer : Buffer or None
            The read bytes, or None if the key does not exist.
        """
        if prototype is None:
            prototype = default_buffer_prototype()
        return await self.store.get(self.path, prototype=prototype, byte_range=byte_range)

    async def set(self, value: Buffer, byte_range: ByteRequest | None = None) -> None:
        """
        Write bytes to the store.

        Parameters
        ----------
        value : Buffer
            The buffer to write.
        byte_range : ByteRequest, optional
            The range of bytes to write. If None, the entire buffer is written.

        Raises
        ------
        NotImplementedError
            If `byte_range` is not None, because Store.set does not support partial writes yet.
        """
        if byte_range is not None:
            raise NotImplementedError("Store.set does not have partial writes yet")
        await self.store.set(self.path, value)

    async def delete(self) -> None:
        """
        Delete the key from the store.

        Raises
        ------
        NotImplementedError
            If the store does not support deletion.
        """
        await self.store.delete(self.path)

    async def delete_dir(self) -> None:
        """
        Delete all keys with the given prefix from the store.
        """
        await self.store.delete_dir(self.path)

    async def set_if_not_exists(self, default: Buffer) -> None:
        """
        Store a key to ``value`` if the key is not already present.

        Parameters
        ----------
        default : Buffer
            The buffer to store if the key is not already present.
        """
        await self.store.set_if_not_exists(self.path, default)

    async def exists(self) -> bool:
        """
        Check if the key exists in the store.

        Returns
        -------
        bool
            True if the key exists in the store, False otherwise.
        """
        return await self.store.exists(self.path)

    async def is_empty(self) -> bool:
        """
        Check if any keys exist in the store with the given prefix.

        Returns
        -------
        bool
            True if no keys exist in the store with the given prefix, False otherwise.
        """
        return await self.store.is_empty(self.path)

    def __truediv__(self, other: str) -> StorePath:
        """Combine this store path with another path"""
        return self.__class__(self.store, _dereference_path(self.path, other))

    def __str__(self) -> str:
        return _dereference_path(str(self.store), self.path)

    def __repr__(self) -> str:
        return f"StorePath({self.store.__class__.__name__}, '{self}')"

    def __eq__(self, other: object) -> bool:
        """
        Check if two StorePath objects are equal.

        Returns
        -------
        bool
            True if the two objects are equal, False otherwise.

        Notes
        -----
        Two StorePath objects are considered equal if their stores are equal
        and their paths are equal.
        """
        try:
            return self.store == other.store and self.path == other.path  # type: ignore[attr-defined, no-any-return]
        except Exception:
            pass
        return False


StoreLike: TypeAlias = Store | StorePath | FSMap | Path | str | dict[str, Buffer]


async def make_store_path(
    store_like: StoreLike | None,
    *,
    path: str | None = "",
    mode: AccessModeLiteral | None = None,
    storage_options: dict[str, Any] | None = None,
) -> StorePath:
    """
    Convert a `StoreLike` object into a StorePath object.

    This function takes a `StoreLike` object and returns a `StorePath` object.  The
    `StoreLike` object can be a `Store`, `StorePath`, `Path`, `str`, or `dict[str, Buffer]`.
    If the `StoreLike` object is a Store or `StorePath`, it is converted to a
    `StorePath` object.  If the `StoreLike` object is a Path or str, it is converted
    to a LocalStore object and then to a `StorePath` object.  If the `StoreLike`
    object is a dict[str, Buffer], it is converted to a `MemoryStore` object and
    then to a `StorePath` object.

    If the `StoreLike` object is None, a `MemoryStore` object is created and
    converted to a `StorePath` object.

    If the `StoreLike` object is a str and starts with a protocol, it is
    converted to a RemoteStore object and then to a `StorePath` object.

    If the `StoreLike` object is a dict[str, Buffer] and the mode is not None,
    the `MemoryStore` object is created with the given mode.

    If the `StoreLike` object is a str and starts with a protocol, the
    RemoteStore object is created with the given mode and storage options.

    Parameters
    ----------
    store_like : StoreLike | None
        The object to convert to a `StorePath` object.
    path : str | None, optional
        The path to use when creating the `StorePath` object.  If None, the
        default path is the empty string.
    mode : StoreAccessMode | None, optional
        The mode to use when creating the `StorePath` object.  If None, the
        default mode is 'r'.
    storage_options : dict[str, Any] | None, optional
        The storage options to use when creating the `RemoteStore` object.  If
        None, the default storage options are used.

    Returns
    -------
    StorePath
        The converted StorePath object.

    Raises
    ------
    TypeError
        If the StoreLike object is not one of the supported types.
    """
    from zarr.storage._fsspec import FsspecStore  # circular import

    used_storage_options = False
    path_normalized = normalize_path(path)
    if isinstance(store_like, StorePath):
        result = store_like / path_normalized
    else:
        assert mode in (None, "r", "r+", "a", "w", "w-")
        # if mode 'r' was provided, we'll open any new stores as read-only
        _read_only = mode == "r"
        if isinstance(store_like, Store):
            store = store_like
        elif store_like is None:
            store = await MemoryStore.open(read_only=_read_only)
        elif isinstance(store_like, Path):
            store = await LocalStore.open(root=store_like, read_only=_read_only)
        elif isinstance(store_like, str):
            storage_options = storage_options or {}

            if _is_fsspec_uri(store_like):
                used_storage_options = True
                store = FsspecStore.from_url(
                    store_like, storage_options=storage_options, read_only=_read_only
                )
            else:
                store = await LocalStore.open(root=Path(store_like), read_only=_read_only)
        elif isinstance(store_like, dict):
            # We deliberate only consider dict[str, Buffer] here, and not arbitrary mutable mappings.
            # By only allowing dictionaries, which are in-memory, we know that MemoryStore appropriate.
            store = await MemoryStore.open(store_dict=store_like, read_only=_read_only)
        elif _has_fsspec and isinstance(store_like, FSMap):
            if path:
                raise ValueError(
                    "'path' was provided but is not used for FSMap store_like objects. Specify the path when creating the FSMap instance instead."
                )
            if storage_options:
                raise ValueError(
                    "'storage_options was provided but is not used for FSMap store_like objects. Specify the storage options when creating the FSMap instance instead."
                )
            store = FsspecStore.from_mapper(store_like, read_only=_read_only)
        else:
            raise TypeError(f"Unsupported type for store_like: '{type(store_like).__name__}'")

        result = await StorePath.open(store, path=path_normalized, mode=mode)

    if storage_options and not used_storage_options:
        msg = "'storage_options' was provided but unused. 'storage_options' is only used for fsspec filesystem stores."
        raise TypeError(msg)

    return result


def _is_fsspec_uri(uri: str) -> bool:
    """
    Check if a URI looks like a non-local fsspec URI.

    Examples
    --------
    >>> _is_fsspec_uri("s3://bucket")
    True
    >>> _is_fsspec_uri("my-directory")
    False
    >>> _is_fsspec_uri("local://my-directory")
    False
    """
    return "://" in uri or ("::" in uri and "local://" not in uri)


async def ensure_no_existing_node(store_path: StorePath, zarr_format: ZarrFormat) -> None:
    """
    Check if a store_path is safe for array / group creation.
    Returns `None` or raises an exception.

    Parameters
    ----------
    store_path : StorePath
        The storage location to check.
    zarr_format : ZarrFormat
        The Zarr format to check.

    Raises
    ------
    ContainsArrayError, ContainsGroupError, ContainsArrayAndGroupError
    """
    if zarr_format == 2:
        extant_node = await _contains_node_v2(store_path)
    elif zarr_format == 3:
        extant_node = await _contains_node_v3(store_path)

    if extant_node == "array":
        raise ContainsArrayError(store_path.store, store_path.path)
    elif extant_node == "group":
        raise ContainsGroupError(store_path.store, store_path.path)
    elif extant_node == "nothing":
        return
    msg = f"Invalid value for extant_node: {extant_node}"  # type: ignore[unreachable]
    raise ValueError(msg)


async def _contains_node_v3(store_path: StorePath) -> Literal["array", "group", "nothing"]:
    """
    Check if a store_path contains nothing, an array, or a group. This function
    returns the string "array", "group", or "nothing" to denote containing an array, a group, or
    nothing.

    Parameters
    ----------
    store_path : StorePath
        The location in storage to check.

    Returns
    -------
    Literal["array", "group", "nothing"]
        A string representing the zarr node found at store_path.
    """
    result: Literal["array", "group", "nothing"] = "nothing"
    extant_meta_bytes = await (store_path / ZARR_JSON).get()
    # if no metadata document could be loaded, then we just return "nothing"
    if extant_meta_bytes is not None:
        try:
            extant_meta_json = json.loads(extant_meta_bytes.to_bytes())
            # avoid constructing a full metadata document here in the name of speed.
            if extant_meta_json["node_type"] == "array":
                result = "array"
            elif extant_meta_json["node_type"] == "group":
                result = "group"
        except (KeyError, json.JSONDecodeError):
            # either of these errors is consistent with no array or group present.
            pass
    return result


async def _contains_node_v2(store_path: StorePath) -> Literal["array", "group", "nothing"]:
    """
    Check if a store_path contains nothing, an array, a group, or both. If both an array and a
    group are detected, a `ContainsArrayAndGroup` exception is raised. Otherwise, this function
    returns the string "array", "group", or "nothing" to denote containing an array, a group, or
    nothing.

    Parameters
    ----------
    store_path : StorePath
        The location in storage to check.

    Returns
    -------
    Literal["array", "group", "nothing"]
        A string representing the zarr node found at store_path.
    """
    _array = await contains_array(store_path=store_path, zarr_format=2)
    _group = await contains_group(store_path=store_path, zarr_format=2)

    if _array and _group:
        raise ContainsArrayAndGroupError(store_path.store, store_path.path)
    elif _array:
        return "array"
    elif _group:
        return "group"
    else:
        return "nothing"


async def contains_array(store_path: StorePath, zarr_format: ZarrFormat) -> bool:
    """
    Check if an array exists at a given StorePath.

    Parameters
    ----------
    store_path : StorePath
        The StorePath to check for an existing group.
    zarr_format :
        The zarr format to check for.

    Returns
    -------
    bool
        True if the StorePath contains a group, False otherwise.

    """
    if zarr_format == 3:
        extant_meta_bytes = await (store_path / ZARR_JSON).get()
        if extant_meta_bytes is None:
            return False
        else:
            try:
                extant_meta_json = json.loads(extant_meta_bytes.to_bytes())
                # we avoid constructing a full metadata document here in the name of speed.
                if extant_meta_json["node_type"] == "array":
                    return True
            except (ValueError, KeyError):
                return False
    elif zarr_format == 2:
        return await (store_path / ZARRAY_JSON).exists()
    msg = f"Invalid zarr_format provided. Got {zarr_format}, expected 2 or 3"
    raise ValueError(msg)


async def contains_group(store_path: StorePath, zarr_format: ZarrFormat) -> bool:
    """
    Check if a group exists at a given StorePath.

    Parameters
    ----------

    store_path : StorePath
        The StorePath to check for an existing group.
    zarr_format :
        The zarr format to check for.

    Returns
    -------

    bool
        True if the StorePath contains a group, False otherwise

    """
    if zarr_format == 3:
        extant_meta_bytes = await (store_path / ZARR_JSON).get()
        if extant_meta_bytes is None:
            return False
        else:
            try:
                extant_meta_json = json.loads(extant_meta_bytes.to_bytes())
                # we avoid constructing a full metadata document here in the name of speed.
                result: bool = extant_meta_json["node_type"] == "group"
            except (ValueError, KeyError):
                return False
            else:
                return result
    elif zarr_format == 2:
        return await (store_path / ZGROUP_JSON).exists()
    msg = f"Invalid zarr_format provided. Got {zarr_format}, expected 2 or 3"  # type: ignore[unreachable]
    raise ValueError(msg)
