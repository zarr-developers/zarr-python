from __future__ import annotations

import asyncio
import dataclasses
import warnings
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import numpy.typing as npt
from typing_extensions import deprecated

from zarr.abc.store import Store
from zarr.core.array import (
    DEFAULT_FILL_VALUE,
    Array,
    AsyncArray,
    CompressorLike,
    create_array,
    from_array,
    get_array_metadata,
)
from zarr.core.array_spec import ArrayConfigLike, parse_array_config
from zarr.core.buffer import NDArrayLike
from zarr.core.common import (
    JSON,
    AccessModeLiteral,
    ChunkCoords,
    DimensionNames,
    MemoryOrder,
    ZarrFormat,
    _default_zarr_format,
    _warn_write_empty_chunks_kwarg,
)
from zarr.core.dtype import ZDTypeLike, get_data_type_from_native_dtype
from zarr.core.group import (
    AsyncGroup,
    ConsolidatedMetadata,
    GroupMetadata,
    create_hierarchy,
)
from zarr.core.metadata import ArrayMetadataDict, ArrayV2Metadata, ArrayV3Metadata
from zarr.errors import GroupNotFoundError, NodeTypeValidationError
from zarr.storage import StorePath
from zarr.storage._common import make_store_path

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numcodecs.abc

    from zarr.abc.codec import Codec
    from zarr.core.buffer import NDArrayLikeOrScalar
    from zarr.core.chunk_key_encodings import ChunkKeyEncoding
    from zarr.storage import StoreLike

    # TODO: this type could use some more thought
    ArrayLike = AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata] | Array | npt.NDArray[Any]
    PathLike = str

__all__ = [
    "array",
    "consolidate_metadata",
    "copy",
    "copy_all",
    "copy_store",
    "create",
    "create_array",
    "create_hierarchy",
    "empty",
    "empty_like",
    "from_array",
    "full",
    "full_like",
    "group",
    "load",
    "ones",
    "ones_like",
    "open",
    "open_array",
    "open_consolidated",
    "open_group",
    "open_like",
    "save",
    "save_array",
    "save_group",
    "tree",
    "zeros",
    "zeros_like",
]


_READ_MODES: tuple[AccessModeLiteral, ...] = ("r", "r+", "a")
_CREATE_MODES: tuple[AccessModeLiteral, ...] = ("a", "w", "w-")
_OVERWRITE_MODES: tuple[AccessModeLiteral, ...] = ("w",)


def _infer_overwrite(mode: AccessModeLiteral) -> bool:
    """
    Check that an ``AccessModeLiteral`` is compatible with overwriting an existing Zarr node.
    """
    return mode in _OVERWRITE_MODES


def _get_shape_chunks(a: ArrayLike | Any) -> tuple[ChunkCoords | None, ChunkCoords | None]:
    """Helper function to get the shape and chunks from an array-like object"""
    shape = None
    chunks = None

    if hasattr(a, "shape") and isinstance(a.shape, tuple):
        shape = a.shape

        if hasattr(a, "chunks") and isinstance(a.chunks, tuple) and (len(a.chunks) == len(a.shape)):
            chunks = a.chunks

        elif hasattr(a, "chunklen"):
            # bcolz carray
            chunks = (a.chunklen,) + a.shape[1:]

    return shape, chunks


def _like_args(a: ArrayLike, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Set default values for shape and chunks if they are not present in the array-like object"""

    new = kwargs.copy()

    shape, chunks = _get_shape_chunks(a)
    if shape is not None:
        new["shape"] = shape
    if chunks is not None:
        new["chunks"] = chunks

    if hasattr(a, "dtype"):
        new["dtype"] = a.dtype

    if isinstance(a, AsyncArray):
        new["order"] = a.order
        if isinstance(a.metadata, ArrayV2Metadata):
            new["compressor"] = a.metadata.compressor
            new["filters"] = a.metadata.filters
        else:
            # TODO: Remove type: ignore statement when type inference improves.
            # mypy cannot correctly infer the type of a.metadata here for some reason.
            new["codecs"] = a.metadata.codecs  # type: ignore[unreachable]

    else:
        # TODO: set default values compressor/codecs
        # to do this, we may need to evaluate if this is a v2 or v3 array
        # new["compressor"] = "default"
        pass

    return new


def _handle_zarr_version_or_format(
    *, zarr_version: ZarrFormat | None, zarr_format: ZarrFormat | None
) -> ZarrFormat | None:
    """Handle the deprecated zarr_version kwarg and return zarr_format"""
    if zarr_format is not None and zarr_version is not None and zarr_format != zarr_version:
        raise ValueError(
            f"zarr_format {zarr_format} does not match zarr_version {zarr_version}, please only set one"
        )
    if zarr_version is not None:
        warnings.warn(
            "zarr_version is deprecated, use zarr_format", DeprecationWarning, stacklevel=2
        )
        return zarr_version
    return zarr_format


async def consolidate_metadata(
    store: StoreLike,
    path: str | None = None,
    zarr_format: ZarrFormat | None = None,
) -> AsyncGroup:
    """
    Consolidate the metadata of all nodes in a hierarchy.

    Upon completion, the metadata of the root node in the Zarr hierarchy will be
    updated to include all the metadata of child nodes. For Stores that do
    not support consolidated metadata, this operation raises a ``TypeError``.

    Parameters
    ----------
    store : StoreLike
        The store-like object whose metadata you wish to consolidate.
    path : str, optional
        A path to a group in the store to consolidate at. Only children
        below that group will be consolidated.

        By default, the root node is used so all the metadata in the
        store is consolidated.
    zarr_format : {2, 3, None}, optional
        The zarr format of the hierarchy. By default the zarr format
        is inferred.

    Returns
    -------
    group: AsyncGroup
        The group, with the ``consolidated_metadata`` field set to include
        the metadata of each child node. If the Store doesn't support
        consolidated metadata, this function raises a `TypeError`.
        See ``Store.supports_consolidated_metadata``.
    """
    store_path = await make_store_path(store, path=path)

    if not store_path.store.supports_consolidated_metadata:
        store_name = type(store_path.store).__name__
        raise TypeError(
            f"The Zarr Store in use ({store_name}) doesn't support consolidated metadata",
        )

    group = await AsyncGroup.open(store_path, zarr_format=zarr_format, use_consolidated=False)
    group.store_path.store._check_writable()

    members_metadata = {
        k: v.metadata
        async for k, v in group.members(max_depth=None, use_consolidated_for_children=False)
    }
    # While consolidating, we want to be explicit about when child groups
    # are empty by inserting an empty dict for consolidated_metadata.metadata
    for k, v in members_metadata.items():
        if isinstance(v, GroupMetadata) and v.consolidated_metadata is None:
            v = dataclasses.replace(v, consolidated_metadata=ConsolidatedMetadata(metadata={}))
            members_metadata[k] = v

    if any(m.zarr_format == 3 for m in members_metadata.values()):
        warnings.warn(
            "Consolidated metadata is currently not part in the Zarr format 3 specification. It "
            "may not be supported by other zarr implementations and may change in the future.",
            category=UserWarning,
            stacklevel=1,
        )

    ConsolidatedMetadata._flat_to_nested(members_metadata)

    consolidated_metadata = ConsolidatedMetadata(metadata=members_metadata)
    metadata = dataclasses.replace(group.metadata, consolidated_metadata=consolidated_metadata)
    group = dataclasses.replace(
        group,
        metadata=metadata,
    )
    await group._save_metadata()
    return group


async def copy(*args: Any, **kwargs: Any) -> tuple[int, int, int]:
    raise NotImplementedError


async def copy_all(*args: Any, **kwargs: Any) -> tuple[int, int, int]:
    raise NotImplementedError


async def copy_store(*args: Any, **kwargs: Any) -> tuple[int, int, int]:
    raise NotImplementedError


async def load(
    *,
    store: StoreLike,
    path: str | None = None,
    zarr_format: ZarrFormat | None = None,
    zarr_version: ZarrFormat | None = None,
) -> NDArrayLikeOrScalar | dict[str, NDArrayLikeOrScalar]:
    """Load data from an array or group into memory.

    Parameters
    ----------
    store : Store or str
        Store or path to directory in file system or name of zip file.
    path : str or None, optional
        The path within the store from which to load.

    Returns
    -------
    out
        If the path contains an array, out will be a numpy array. If the path contains
        a group, out will be a dict-like object where keys are array names and values
        are numpy arrays.

    See Also
    --------
    save, savez

    Notes
    -----
    If loading data from a group of arrays, data will not be immediately loaded into
    memory. Rather, arrays will be loaded into memory as they are requested.
    """
    zarr_format = _handle_zarr_version_or_format(zarr_version=zarr_version, zarr_format=zarr_format)

    obj = await open(store=store, path=path, zarr_format=zarr_format)
    if isinstance(obj, AsyncArray):
        return await obj.getitem(slice(None))
    else:
        raise NotImplementedError("loading groups not yet supported")


async def open(
    *,
    store: StoreLike | None = None,
    mode: AccessModeLiteral | None = None,
    zarr_version: ZarrFormat | None = None,  # deprecated
    zarr_format: ZarrFormat | None = None,
    path: str | None = None,
    storage_options: dict[str, Any] | None = None,
    **kwargs: Any,  # TODO: type kwargs as valid args to open_array
) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata] | AsyncGroup:
    """Convenience function to open a group or array using file-mode-like semantics.

    Parameters
    ----------
    store : Store or str, optional
        Store or path to directory in file system or name of zip file.
    mode : {'r', 'r+', 'a', 'w', 'w-'}, optional
        Persistence mode: 'r' means read only (must exist); 'r+' means
        read/write (must exist); 'a' means read/write (create if doesn't
        exist); 'w' means create (overwrite if exists); 'w-' means create
        (fail if exists).
        If the store is read-only, the default is 'r'; otherwise, it is 'a'.
    zarr_format : {2, 3, None}, optional
        The zarr format to use when saving.
    path : str or None, optional
        The path within the store to open.
    storage_options : dict
        If the store is backed by an fsspec-based implementation, then this dict will be passed to
        the Store constructor for that implementation. Ignored otherwise.
    **kwargs
        Additional parameters are passed through to :func:`zarr.creation.open_array` or
        :func:`zarr.hierarchy.open_group`.

    Returns
    -------
    z : array or group
        Return type depends on what exists in the given store.
    """
    zarr_format = _handle_zarr_version_or_format(zarr_version=zarr_version, zarr_format=zarr_format)
    if mode is None:
        if isinstance(store, (Store, StorePath)) and store.read_only:
            mode = "r"
        else:
            mode = "a"
    store_path = await make_store_path(store, mode=mode, path=path, storage_options=storage_options)

    # TODO: the mode check below seems wrong!
    if "shape" not in kwargs and mode in {"a", "r", "r+", "w"}:
        try:
            metadata_dict = await get_array_metadata(store_path, zarr_format=zarr_format)
            # TODO: remove this cast when we fix typing for array metadata dicts
            _metadata_dict = cast("ArrayMetadataDict", metadata_dict)
            # for v2, the above would already have raised an exception if not an array
            zarr_format = _metadata_dict["zarr_format"]
            is_v3_array = zarr_format == 3 and _metadata_dict.get("node_type") == "array"
            if is_v3_array or zarr_format == 2:
                return AsyncArray(store_path=store_path, metadata=_metadata_dict)
        except (AssertionError, FileNotFoundError, NodeTypeValidationError):
            pass
        return await open_group(store=store_path, zarr_format=zarr_format, mode=mode, **kwargs)

    try:
        return await open_array(store=store_path, zarr_format=zarr_format, mode=mode, **kwargs)
    except (KeyError, NodeTypeValidationError):
        # KeyError for a missing key
        # NodeTypeValidationError for failing to parse node metadata as an array when it's
        # actually a group
        return await open_group(store=store_path, zarr_format=zarr_format, mode=mode, **kwargs)


async def open_consolidated(
    *args: Any, use_consolidated: Literal[True] = True, **kwargs: Any
) -> AsyncGroup:
    """
    Alias for :func:`open_group` with ``use_consolidated=True``.
    """
    if use_consolidated is not True:
        raise TypeError(
            "'use_consolidated' must be 'True' in 'open_consolidated'. Use 'open' with "
            "'use_consolidated=False' to bypass consolidated metadata."
        )
    return await open_group(*args, use_consolidated=use_consolidated, **kwargs)


async def save(
    store: StoreLike,
    *args: NDArrayLike,
    zarr_version: ZarrFormat | None = None,  # deprecated
    zarr_format: ZarrFormat | None = None,
    path: str | None = None,
    **kwargs: Any,  # TODO: type kwargs as valid args to save
) -> None:
    """Convenience function to save an array or group of arrays to the local file system.

    Parameters
    ----------
    store : Store or str
        Store or path to directory in file system or name of zip file.
    *args : ndarray
        NumPy arrays with data to save.
    zarr_format : {2, 3, None}, optional
        The zarr format to use when saving.
    path : str or None, optional
        The path within the group where the arrays will be saved.
    **kwargs
        NumPy arrays with data to save.
    """
    zarr_format = _handle_zarr_version_or_format(zarr_version=zarr_version, zarr_format=zarr_format)

    if len(args) == 0 and len(kwargs) == 0:
        raise ValueError("at least one array must be provided")
    if len(args) == 1 and len(kwargs) == 0:
        await save_array(store, args[0], zarr_format=zarr_format, path=path)
    else:
        await save_group(store, *args, zarr_format=zarr_format, path=path, **kwargs)


async def save_array(
    store: StoreLike,
    arr: NDArrayLike,
    *,
    zarr_version: ZarrFormat | None = None,  # deprecated
    zarr_format: ZarrFormat | None = None,
    path: str | None = None,
    storage_options: dict[str, Any] | None = None,
    **kwargs: Any,  # TODO: type kwargs as valid args to create
) -> None:
    """Convenience function to save a NumPy array to the local file system, following a
    similar API to the NumPy save() function.

    Parameters
    ----------
    store : Store or str
        Store or path to directory in file system or name of zip file.
    arr : ndarray
        NumPy array with data to save.
    zarr_format : {2, 3, None}, optional
        The zarr format to use when saving (default is 3 if not specified).
    path : str or None, optional
        The path within the store where the array will be saved.
    storage_options : dict
        If using an fsspec URL to create the store, these will be passed to
        the backend implementation. Ignored otherwise.
    **kwargs
        Passed through to :func:`create`, e.g., compressor.
    """
    zarr_format = (
        _handle_zarr_version_or_format(zarr_version=zarr_version, zarr_format=zarr_format)
        or _default_zarr_format()
    )
    if not isinstance(arr, NDArrayLike):
        raise TypeError("arr argument must be numpy or other NDArrayLike array")

    mode = kwargs.pop("mode", "a")
    store_path = await make_store_path(store, path=path, mode=mode, storage_options=storage_options)
    if np.isscalar(arr):
        arr = np.array(arr)
    shape = arr.shape
    chunks = getattr(arr, "chunks", None)  # for array-likes with chunks attribute
    overwrite = kwargs.pop("overwrite", None) or _infer_overwrite(mode)
    zarr_dtype = get_data_type_from_native_dtype(arr.dtype)
    new = await AsyncArray._create(
        store_path,
        zarr_format=zarr_format,
        shape=shape,
        dtype=zarr_dtype,
        chunks=chunks,
        overwrite=overwrite,
        **kwargs,
    )
    await new.setitem(slice(None), arr)


async def save_group(
    store: StoreLike,
    *args: NDArrayLike,
    zarr_version: ZarrFormat | None = None,  # deprecated
    zarr_format: ZarrFormat | None = None,
    path: str | None = None,
    storage_options: dict[str, Any] | None = None,
    **kwargs: NDArrayLike,
) -> None:
    """Convenience function to save several NumPy arrays to the local file system, following a
    similar API to the NumPy savez()/savez_compressed() functions.

    Parameters
    ----------
    store : Store or str
        Store or path to directory in file system or name of zip file.
    *args : ndarray
        NumPy arrays with data to save.
    zarr_format : {2, 3, None}, optional
        The zarr format to use when saving.
    path : str or None, optional
        Path within the store where the group will be saved.
    storage_options : dict
        If using an fsspec URL to create the store, these will be passed to
        the backend implementation. Ignored otherwise.
    **kwargs
        NumPy arrays with data to save.
    """

    store_path = await make_store_path(store, path=path, mode="w", storage_options=storage_options)

    zarr_format = (
        _handle_zarr_version_or_format(
            zarr_version=zarr_version,
            zarr_format=zarr_format,
        )
        or _default_zarr_format()
    )

    for arg in args:
        if not isinstance(arg, NDArrayLike):
            raise TypeError(
                "All arguments must be numpy or other NDArrayLike arrays (except store, path, storage_options, and zarr_format)"
            )
    for k, v in kwargs.items():
        if not isinstance(v, NDArrayLike):
            raise TypeError(f"Keyword argument '{k}' must be a numpy or other NDArrayLike array")

    if len(args) == 0 and len(kwargs) == 0:
        raise ValueError("at least one array must be provided")
    aws = []
    for i, arr in enumerate(args):
        aws.append(
            save_array(
                store_path,
                arr,
                zarr_format=zarr_format,
                path=f"arr_{i}",
                storage_options=storage_options,
            )
        )
    for k, arr in kwargs.items():
        aws.append(save_array(store_path, arr, zarr_format=zarr_format, path=k))
    await asyncio.gather(*aws)


@deprecated("Use AsyncGroup.tree instead.")
async def tree(grp: AsyncGroup, expand: bool | None = None, level: int | None = None) -> Any:
    """Provide a rich display of the hierarchy.

    .. deprecated:: 3.0.0
        `zarr.tree()` is deprecated and will be removed in a future release.
        Use `group.tree()` instead.

    Parameters
    ----------
    grp : Group
        Zarr or h5py group.
    expand : bool, optional
        Only relevant for HTML representation. If True, tree will be fully expanded.
    level : int, optional
        Maximum depth to descend into hierarchy.

    Returns
    -------
    TreeRepr
        A pretty-printable object displaying the hierarchy.
    """
    return await grp.tree(expand=expand, level=level)


async def array(
    data: npt.ArrayLike | Array, **kwargs: Any
) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]:
    """Create an array filled with `data`.

    Parameters
    ----------
    data : array_like
        The data to fill the array with.
    **kwargs
        Passed through to :func:`create`.

    Returns
    -------
    array : array
        The new array.
    """

    if isinstance(data, Array):
        return await from_array(data=data, **kwargs)

    # ensure data is array-like
    if not hasattr(data, "shape") or not hasattr(data, "dtype"):
        data = np.asanyarray(data)

    # setup dtype
    kw_dtype = kwargs.get("dtype")
    if kw_dtype is None and hasattr(data, "dtype"):
        kwargs["dtype"] = data.dtype
    else:
        kwargs["dtype"] = kw_dtype

    # setup shape and chunks
    data_shape, data_chunks = _get_shape_chunks(data)
    kwargs["shape"] = data_shape
    kw_chunks = kwargs.get("chunks")
    if kw_chunks is None:
        kwargs["chunks"] = data_chunks
    else:
        kwargs["chunks"] = kw_chunks

    read_only = kwargs.pop("read_only", False)
    if read_only:
        raise ValueError("read_only=True is no longer supported when creating new arrays")

    # instantiate array
    z = await create(**kwargs)

    # fill with data
    await z.setitem(Ellipsis, data)

    return z


async def group(
    *,  # Note: this is a change from v2
    store: StoreLike | None = None,
    overwrite: bool = False,
    chunk_store: StoreLike | None = None,  # not used
    cache_attrs: bool | None = None,  # not used, default changed
    synchronizer: Any | None = None,  # not used
    path: str | None = None,
    zarr_version: ZarrFormat | None = None,  # deprecated
    zarr_format: ZarrFormat | None = None,
    meta_array: Any | None = None,  # not used
    attributes: dict[str, JSON] | None = None,
    storage_options: dict[str, Any] | None = None,
) -> AsyncGroup:
    """Create a group.

    Parameters
    ----------
    store : Store or str, optional
        Store or path to directory in file system.
    overwrite : bool, optional
        If True, delete any pre-existing data in `store` at `path` before
        creating the group.
    chunk_store : Store, optional
        Separate storage for chunks. If not provided, `store` will be used
        for storage of both chunks and metadata.
    cache_attrs : bool, optional
        If True (default), user attributes will be cached for attribute read
        operations. If False, user attributes are reloaded from the store prior
        to all attribute read operations.
    synchronizer : object, optional
        Array synchronizer.
    path : str, optional
        Group path within store.
    meta_array : array-like, optional
        An array instance to use for determining arrays to create and return
        to users. Use `numpy.empty(())` by default.
    zarr_format : {2, 3, None}, optional
        The zarr format to use when saving.
    storage_options : dict
        If using an fsspec URL to create the store, these will be passed to
        the backend implementation. Ignored otherwise.

    Returns
    -------
    g : group
        The new group.
    """

    zarr_format = _handle_zarr_version_or_format(zarr_version=zarr_version, zarr_format=zarr_format)

    mode: AccessModeLiteral
    if overwrite:
        mode = "w"
    else:
        mode = "r+"
    store_path = await make_store_path(store, path=path, mode=mode, storage_options=storage_options)

    if chunk_store is not None:
        warnings.warn("chunk_store is not yet implemented", RuntimeWarning, stacklevel=2)
    if cache_attrs is not None:
        warnings.warn("cache_attrs is not yet implemented", RuntimeWarning, stacklevel=2)
    if synchronizer is not None:
        warnings.warn("synchronizer is not yet implemented", RuntimeWarning, stacklevel=2)
    if meta_array is not None:
        warnings.warn("meta_array is not yet implemented", RuntimeWarning, stacklevel=2)

    if attributes is None:
        attributes = {}

    try:
        return await AsyncGroup.open(store=store_path, zarr_format=zarr_format)
    except (KeyError, FileNotFoundError):
        _zarr_format = zarr_format or _default_zarr_format()
        return await AsyncGroup.from_store(
            store=store_path,
            zarr_format=_zarr_format,
            overwrite=overwrite,
            attributes=attributes,
        )


async def create_group(
    *,
    store: StoreLike,
    path: str | None = None,
    overwrite: bool = False,
    zarr_format: ZarrFormat | None = None,
    attributes: dict[str, Any] | None = None,
    storage_options: dict[str, Any] | None = None,
) -> AsyncGroup:
    """Create a group.

    Parameters
    ----------
    store : Store or str
        Store or path to directory in file system.
    path : str, optional
        Group path within store.
    overwrite : bool, optional
        If True, pre-existing data at ``path`` will be deleted before
        creating the group.
    zarr_format : {2, 3, None}, optional
        The zarr format to use when saving.
        If no ``zarr_format`` is provided, the default format will be used.
        This default can be changed by modifying the value of ``default_zarr_format``
        in :mod:`zarr.core.config`.
    storage_options : dict
        If using an fsspec URL to create the store, these will be passed to
        the backend implementation. Ignored otherwise.

    Returns
    -------
    AsyncGroup
        The new group.
    """

    if zarr_format is None:
        zarr_format = _default_zarr_format()

    mode: Literal["a"] = "a"

    store_path = await make_store_path(store, path=path, mode=mode, storage_options=storage_options)

    return await AsyncGroup.from_store(
        store=store_path,
        zarr_format=zarr_format,
        overwrite=overwrite,
        attributes=attributes,
    )


async def open_group(
    store: StoreLike | None = None,
    *,  # Note: this is a change from v2
    mode: AccessModeLiteral = "a",
    cache_attrs: bool | None = None,  # not used, default changed
    synchronizer: Any = None,  # not used
    path: str | None = None,
    chunk_store: StoreLike | None = None,  # not used
    storage_options: dict[str, Any] | None = None,
    zarr_version: ZarrFormat | None = None,  # deprecated
    zarr_format: ZarrFormat | None = None,
    meta_array: Any | None = None,  # not used
    attributes: dict[str, JSON] | None = None,
    use_consolidated: bool | str | None = None,
) -> AsyncGroup:
    """Open a group using file-mode-like semantics.

    Parameters
    ----------
    store : Store, str, or mapping, optional
        Store or path to directory in file system or name of zip file.

        Strings are interpreted as paths on the local file system
        and used as the ``root`` argument to :class:`zarr.storage.LocalStore`.

        Dictionaries are used as the ``store_dict`` argument in
        :class:`zarr.storage.MemoryStore``.

        By default (``store=None``) a new :class:`zarr.storage.MemoryStore`
        is created.

    mode : {'r', 'r+', 'a', 'w', 'w-'}, optional
        Persistence mode: 'r' means read only (must exist); 'r+' means
        read/write (must exist); 'a' means read/write (create if doesn't
        exist); 'w' means create (overwrite if exists); 'w-' means create
        (fail if exists).
    cache_attrs : bool, optional
        If True (default), user attributes will be cached for attribute read
        operations. If False, user attributes are reloaded from the store prior
        to all attribute read operations.
    synchronizer : object, optional
        Array synchronizer.
    path : str, optional
        Group path within store.
    chunk_store : Store or str, optional
        Store or path to directory in file system or name of zip file.
    storage_options : dict
        If using an fsspec URL to create the store, these will be passed to
        the backend implementation. Ignored otherwise.
    meta_array : array-like, optional
        An array instance to use for determining arrays to create and return
        to users. Use `numpy.empty(())` by default.
    attributes : dict
        A dictionary of JSON-serializable values with user-defined attributes.
    use_consolidated : bool or str, default None
        Whether to use consolidated metadata.

        By default, consolidated metadata is used if it's present in the
        store (in the ``zarr.json`` for Zarr format 3 and in the ``.zmetadata`` file
        for Zarr format 2).

        To explicitly require consolidated metadata, set ``use_consolidated=True``,
        which will raise an exception if consolidated metadata is not found.

        To explicitly *not* use consolidated metadata, set ``use_consolidated=False``,
        which will fall back to using the regular, non consolidated metadata.

        Zarr format 2 allowed configuring the key storing the consolidated metadata
        (``.zmetadata`` by default). Specify the custom key as ``use_consolidated``
        to load consolidated metadata from a non-default key.

    Returns
    -------
    g : group
        The new group.
    """

    zarr_format = _handle_zarr_version_or_format(zarr_version=zarr_version, zarr_format=zarr_format)

    if cache_attrs is not None:
        warnings.warn("cache_attrs is not yet implemented", RuntimeWarning, stacklevel=2)
    if synchronizer is not None:
        warnings.warn("synchronizer is not yet implemented", RuntimeWarning, stacklevel=2)
    if meta_array is not None:
        warnings.warn("meta_array is not yet implemented", RuntimeWarning, stacklevel=2)
    if chunk_store is not None:
        warnings.warn("chunk_store is not yet implemented", RuntimeWarning, stacklevel=2)

    store_path = await make_store_path(store, mode=mode, storage_options=storage_options, path=path)
    if attributes is None:
        attributes = {}

    try:
        if mode in _READ_MODES:
            return await AsyncGroup.open(
                store_path, zarr_format=zarr_format, use_consolidated=use_consolidated
            )
    except (KeyError, FileNotFoundError):
        pass
    if mode in _CREATE_MODES:
        overwrite = _infer_overwrite(mode)
        _zarr_format = zarr_format or _default_zarr_format()
        return await AsyncGroup.from_store(
            store_path,
            zarr_format=_zarr_format,
            overwrite=overwrite,
            attributes=attributes,
        )
    raise GroupNotFoundError(store, store_path.path)


async def create(
    shape: ChunkCoords | int,
    *,  # Note: this is a change from v2
    chunks: ChunkCoords | int | bool | None = None,
    dtype: ZDTypeLike | None = None,
    compressor: CompressorLike = "auto",
    fill_value: Any | None = DEFAULT_FILL_VALUE,
    order: MemoryOrder | None = None,
    store: str | StoreLike | None = None,
    synchronizer: Any | None = None,
    overwrite: bool = False,
    path: PathLike | None = None,
    chunk_store: StoreLike | None = None,
    filters: Iterable[dict[str, JSON] | numcodecs.abc.Codec] | None = None,
    cache_metadata: bool | None = None,
    cache_attrs: bool | None = None,
    read_only: bool | None = None,
    object_codec: Codec | None = None,  # TODO: type has changed
    dimension_separator: Literal[".", "/"] | None = None,
    write_empty_chunks: bool | None = None,
    zarr_version: ZarrFormat | None = None,  # deprecated
    zarr_format: ZarrFormat | None = None,
    meta_array: Any | None = None,  # TODO: need type
    attributes: dict[str, JSON] | None = None,
    # v3 only
    chunk_shape: ChunkCoords | int | None = None,
    chunk_key_encoding: (
        ChunkKeyEncoding
        | tuple[Literal["default"], Literal[".", "/"]]
        | tuple[Literal["v2"], Literal[".", "/"]]
        | None
    ) = None,
    codecs: Iterable[Codec | dict[str, JSON]] | None = None,
    dimension_names: DimensionNames = None,
    storage_options: dict[str, Any] | None = None,
    config: ArrayConfigLike | None = None,
    **kwargs: Any,
) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]:
    """Create an array.

    Parameters
    ----------
    shape : int or tuple of ints
        Array shape.
    chunks : int or tuple of ints, optional
        The shape of the array's chunks.
        Zarr format 2 only. Zarr format 3 arrays should use `chunk_shape` instead.
        If not specified, default values are guessed based on the shape and dtype.
    dtype : str or dtype, optional
        NumPy dtype.
    chunk_shape : int or tuple of ints, optional
        The shape of the Array's chunks (default is None).
        Zarr format 3 only. Zarr format 2 arrays should use `chunks` instead.
    chunk_key_encoding : ChunkKeyEncoding, optional
        A specification of how the chunk keys are represented in storage.
        Zarr format 3 only. Zarr format 2 arrays should use `dimension_separator` instead.
        Default is ``("default", "/")``.
    codecs : Sequence of Codecs or dicts, optional
        An iterable of Codec or dict serializations of Codecs. The elements of
        this collection specify the transformation from array values to stored bytes.
        Zarr format 3 only. Zarr format 2 arrays should use ``filters`` and ``compressor`` instead.

        If no codecs are provided, default codecs will be used:

        - For numeric arrays, the default is ``BytesCodec`` and ``ZstdCodec``.
        - For Unicode strings, the default is ``VLenUTF8Codec`` and ``ZstdCodec``.
        - For bytes or objects, the default is ``VLenBytesCodec`` and ``ZstdCodec``.

        These defaults can be changed by modifying the value of ``array.v3_default_filters``,
        ``array.v3_default_serializer`` and ``array.v3_default_compressors`` in :mod:`zarr.core.config`.
    compressor : Codec, optional
        Primary compressor to compress chunk data.
        Zarr format 2 only. Zarr format 3 arrays should use ``codecs`` instead.

        If neither ``compressor`` nor ``filters`` are provided, a default compressor will be used:

        - For numeric arrays, the default is ``ZstdCodec``.
        - For Unicode strings, the default is ``VLenUTF8Codec``.
        - For bytes or objects, the default is ``VLenBytesCodec``.

        These defaults can be changed by modifying the value of ``array.v2_default_compressor`` in :mod:`zarr.core.config`.
    fill_value : object
        Default value to use for uninitialized portions of the array.
    order : {'C', 'F'}, optional
        Deprecated in favor of the ``config`` keyword argument.
        Pass ``{'order': <value>}`` to ``create`` instead of using this parameter.
        Memory layout to be used within each chunk.
        If not specified, the ``array.order`` parameter in the global config will be used.
    store : Store or str
        Store or path to directory in file system or name of zip file.
    synchronizer : object, optional
        Array synchronizer.
    overwrite : bool, optional
        If True, delete all pre-existing data in `store` at `path` before
        creating the array.
    path : str, optional
        Path under which array is stored.
    chunk_store : MutableMapping, optional
        Separate storage for chunks. If not provided, `store` will be used
        for storage of both chunks and metadata.
    filters : sequence of Codecs, optional
        Sequence of filters to use to encode chunk data prior to compression.
        Zarr format 2 only. If no ``filters`` are provided, a default set of filters will be used.
        These defaults can be changed by modifying the value of ``array.v2_default_filters`` in :mod:`zarr.core.config`.
    cache_metadata : bool, optional
        If True, array configuration metadata will be cached for the
        lifetime of the object. If False, array metadata will be reloaded
        prior to all data access and modification operations (may incur
        overhead depending on storage and data access pattern).
    cache_attrs : bool, optional
        If True (default), user attributes will be cached for attribute read
        operations. If False, user attributes are reloaded from the store prior
        to all attribute read operations.
    read_only : bool, optional
        True if array should be protected against modification.
    object_codec : Codec, optional
        A codec to encode object arrays, only needed if dtype=object.
    dimension_separator : {'.', '/'}, optional
        Separator placed between the dimensions of a chunk.
        Zarr format 2 only. Zarr format 3 arrays should use ``chunk_key_encoding`` instead.
        Default is ".".
    write_empty_chunks : bool, optional
        Deprecated in favor of the ``config`` keyword argument.
        Pass ``{'write_empty_chunks': <value>}`` to ``create`` instead of using this parameter.
        If True, all chunks will be stored regardless of their
        contents. If False, each chunk is compared to the array's fill value
        prior to storing. If a chunk is uniformly equal to the fill value, then
        that chunk is not be stored, and the store entry for that chunk's key
        is deleted.
    zarr_format : {2, 3, None}, optional
        The zarr format to use when saving.
        Default is 3.
    meta_array : array-like, optional
        An array instance to use for determining arrays to create and return
        to users. Use `numpy.empty(())` by default.
    storage_options : dict
        If using an fsspec URL to create the store, these will be passed to
        the backend implementation. Ignored otherwise.
    config : ArrayConfig or ArrayConfigLike, optional
        Runtime configuration of the array. If provided, will override the
        default values from `zarr.config.array`.

    Returns
    -------
    z : array
        The array.
    """
    zarr_format = (
        _handle_zarr_version_or_format(zarr_version=zarr_version, zarr_format=zarr_format)
        or _default_zarr_format()
    )

    if synchronizer is not None:
        warnings.warn("synchronizer is not yet implemented", RuntimeWarning, stacklevel=2)
    if chunk_store is not None:
        warnings.warn("chunk_store is not yet implemented", RuntimeWarning, stacklevel=2)
    if cache_metadata is not None:
        warnings.warn("cache_metadata is not yet implemented", RuntimeWarning, stacklevel=2)
    if cache_attrs is not None:
        warnings.warn("cache_attrs is not yet implemented", RuntimeWarning, stacklevel=2)
    if object_codec is not None:
        warnings.warn("object_codec is not yet implemented", RuntimeWarning, stacklevel=2)
    if read_only is not None:
        warnings.warn("read_only is not yet implemented", RuntimeWarning, stacklevel=2)
    if meta_array is not None:
        warnings.warn("meta_array is not yet implemented", RuntimeWarning, stacklevel=2)

    if write_empty_chunks is not None:
        _warn_write_empty_chunks_kwarg()

    mode = kwargs.pop("mode", None)
    if mode is None:
        mode = "a"
    store_path = await make_store_path(store, path=path, mode=mode, storage_options=storage_options)

    config_parsed = parse_array_config(config)

    if write_empty_chunks is not None:
        if config is not None:
            msg = (
                "Both write_empty_chunks and config keyword arguments are set. "
                "This is redundant. When both are set, write_empty_chunks will be used instead "
                "of the value in config."
            )
            warnings.warn(UserWarning(msg), stacklevel=1)
        config_parsed = dataclasses.replace(config_parsed, write_empty_chunks=write_empty_chunks)

    return await AsyncArray._create(
        store_path,
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        compressor=compressor,
        fill_value=fill_value,
        overwrite=overwrite,
        filters=filters,
        dimension_separator=dimension_separator,
        order=order,
        zarr_format=zarr_format,
        chunk_shape=chunk_shape,
        chunk_key_encoding=chunk_key_encoding,
        codecs=codecs,
        dimension_names=dimension_names,
        attributes=attributes,
        config=config_parsed,
        **kwargs,
    )


async def empty(
    shape: ChunkCoords, **kwargs: Any
) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]:
    """Create an empty array with the specified shape. The contents will be filled with the
    array's fill value or zeros if no fill value is provided.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the empty array.
    **kwargs
        Keyword arguments passed to :func:`zarr.api.asynchronous.create`.

    Notes
    -----
    The contents of an empty Zarr array are not defined. On attempting to
    retrieve data from an empty Zarr array, any values may be returned,
    and these are not guaranteed to be stable from one access to the next.
    """

    return await create(shape=shape, fill_value=None, **kwargs)


async def empty_like(
    a: ArrayLike, **kwargs: Any
) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]:
    """Create an empty array like `a`. The contents will be filled with the
    array's fill value or zeros if no fill value is provided.

    Parameters
    ----------
    a : array-like
        The array to create an empty array like.
    **kwargs
        Keyword arguments passed to :func:`zarr.api.asynchronous.create`.

    Returns
    -------
    Array
        The new array.

    Notes
    -----
    The contents of an empty Zarr array are not defined. On attempting to
    retrieve data from an empty Zarr array, any values may be returned,
    and these are not guaranteed to be stable from one access to the next.
    """
    like_kwargs = _like_args(a, kwargs)
    return await empty(**like_kwargs)


# TODO: add type annotations for fill_value and kwargs
async def full(
    shape: ChunkCoords, fill_value: Any, **kwargs: Any
) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]:
    """Create an array, with `fill_value` being used as the default value for
    uninitialized portions of the array.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the empty array.
    fill_value : scalar
        Fill value.
    **kwargs
        Keyword arguments passed to :func:`zarr.api.asynchronous.create`.

    Returns
    -------
    Array
        The new array.
    """
    return await create(shape=shape, fill_value=fill_value, **kwargs)


# TODO: add type annotations for kwargs
async def full_like(
    a: ArrayLike, **kwargs: Any
) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]:
    """Create a filled array like `a`.

    Parameters
    ----------
    a : array-like
        The array to create an empty array like.
    **kwargs
        Keyword arguments passed to :func:`zarr.api.asynchronous.create`.

    Returns
    -------
    Array
        The new array.
    """
    like_kwargs = _like_args(a, kwargs)
    if isinstance(a, AsyncArray):
        like_kwargs.setdefault("fill_value", a.metadata.fill_value)
    return await full(**like_kwargs)


async def ones(
    shape: ChunkCoords, **kwargs: Any
) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]:
    """Create an array, with one being used as the default value for
    uninitialized portions of the array.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the empty array.
    **kwargs
        Keyword arguments passed to :func:`zarr.api.asynchronous.create`.

    Returns
    -------
    Array
        The new array.
    """
    return await create(shape=shape, fill_value=1, **kwargs)


async def ones_like(
    a: ArrayLike, **kwargs: Any
) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]:
    """Create an array of ones like `a`.

    Parameters
    ----------
    a : array-like
        The array to create an empty array like.
    **kwargs
        Keyword arguments passed to :func:`zarr.api.asynchronous.create`.

    Returns
    -------
    Array
        The new array.
    """
    like_kwargs = _like_args(a, kwargs)
    return await ones(**like_kwargs)


async def open_array(
    *,  # note: this is a change from v2
    store: StoreLike | None = None,
    zarr_version: ZarrFormat | None = None,  # deprecated
    zarr_format: ZarrFormat | None = None,
    path: PathLike = "",
    storage_options: dict[str, Any] | None = None,
    **kwargs: Any,  # TODO: type kwargs as valid args to save
) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]:
    """Open an array using file-mode-like semantics.

    Parameters
    ----------
    store : Store or str
        Store or path to directory in file system or name of zip file.
    zarr_version : {2, 3, None}, optional
        The zarr format to use when saving. Deprecated in favor of zarr_format.
    zarr_format : {2, 3, None}, optional
        The zarr format to use when saving.
    path : str, optional
        Path in store to array.
    storage_options : dict
        If using an fsspec URL to create the store, these will be passed to
        the backend implementation. Ignored otherwise.
    **kwargs
        Any keyword arguments to pass to :func:`create`.

    Returns
    -------
    AsyncArray
        The opened array.
    """

    mode = kwargs.pop("mode", None)
    store_path = await make_store_path(store, path=path, mode=mode, storage_options=storage_options)

    zarr_format = _handle_zarr_version_or_format(zarr_version=zarr_version, zarr_format=zarr_format)

    if "write_empty_chunks" in kwargs:
        _warn_write_empty_chunks_kwarg()

    try:
        return await AsyncArray.open(store_path, zarr_format=zarr_format)
    except FileNotFoundError:
        if not store_path.read_only and mode in _CREATE_MODES:
            overwrite = _infer_overwrite(mode)
            _zarr_format = zarr_format or _default_zarr_format()
            return await create(
                store=store_path,
                zarr_format=_zarr_format,
                overwrite=overwrite,
                **kwargs,
            )
        raise


async def open_like(
    a: ArrayLike, path: str, **kwargs: Any
) -> AsyncArray[ArrayV3Metadata] | AsyncArray[ArrayV2Metadata]:
    """Open a persistent array like `a`.

    Parameters
    ----------
    a : Array
        The shape and data-type of a define these same attributes of the returned array.
    path : str
        The path to the new array.
    **kwargs
        Any keyword arguments to pass to the array constructor.

    Returns
    -------
    AsyncArray
        The opened array.
    """
    like_kwargs = _like_args(a, kwargs)
    if isinstance(a, (AsyncArray | Array)):
        kwargs.setdefault("fill_value", a.metadata.fill_value)
    return await open_array(path=path, **like_kwargs)


async def zeros(
    shape: ChunkCoords, **kwargs: Any
) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]:
    """Create an array, with zero being used as the default value for
    uninitialized portions of the array.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the empty array.
    **kwargs
        Keyword arguments passed to :func:`zarr.api.asynchronous.create`.

    Returns
    -------
    Array
        The new array.
    """
    return await create(shape=shape, fill_value=0, **kwargs)


async def zeros_like(
    a: ArrayLike, **kwargs: Any
) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]:
    """Create an array of zeros like `a`.

    Parameters
    ----------
    a : array-like
        The array to create an empty array like.
    **kwargs
        Keyword arguments passed to :func:`zarr.api.asynchronous.create`.

    Returns
    -------
    Array
        The new array.
    """
    like_kwargs = _like_args(a, kwargs)
    return await zeros(**like_kwargs)
