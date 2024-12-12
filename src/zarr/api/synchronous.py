from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from typing_extensions import deprecated

import zarr.api.asynchronous as async_api
from zarr._compat import _deprecate_positional_args
from zarr.core.array import Array, AsyncArray
from zarr.core.group import Group
from zarr.core.sync import sync

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy.typing as npt

    from zarr.abc.codec import Codec
    from zarr.api.asynchronous import ArrayLike, PathLike
    from zarr.core.buffer import NDArrayLike
    from zarr.core.chunk_key_encodings import ChunkKeyEncoding
    from zarr.core.common import JSON, AccessModeLiteral, ChunkCoords, MemoryOrder, ZarrFormat
    from zarr.storage import StoreLike

__all__ = [
    "array",
    "consolidate_metadata",
    "copy",
    "copy_all",
    "copy_store",
    "create",
    "empty",
    "empty_like",
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


def consolidate_metadata(
    store: StoreLike,
    path: str | None = None,
    zarr_format: ZarrFormat | None = None,
) -> Group:
    """
    Consolidate the metadata of all nodes in a hierarchy.

    Upon completion, the metadata of the root node in the Zarr hierarchy will be
    updated to include all the metadata of child nodes.

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
        the metadata of each child node.
    """
    return Group(sync(async_api.consolidate_metadata(store, path=path, zarr_format=zarr_format)))


def copy(*args: Any, **kwargs: Any) -> tuple[int, int, int]:
    return sync(async_api.copy(*args, **kwargs))


def copy_all(*args: Any, **kwargs: Any) -> tuple[int, int, int]:
    return sync(async_api.copy_all(*args, **kwargs))


def copy_store(*args: Any, **kwargs: Any) -> tuple[int, int, int]:
    return sync(async_api.copy_store(*args, **kwargs))


def load(
    store: StoreLike,
    path: str | None = None,
    zarr_format: ZarrFormat | None = None,
    zarr_version: ZarrFormat | None = None,
) -> NDArrayLike | dict[str, NDArrayLike]:
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
    return sync(
        async_api.load(store=store, zarr_version=zarr_version, zarr_format=zarr_format, path=path)
    )


@_deprecate_positional_args
def open(
    store: StoreLike | None = None,
    *,
    mode: AccessModeLiteral = "a",
    zarr_version: ZarrFormat | None = None,  # deprecated
    zarr_format: ZarrFormat | None = None,
    path: str | None = None,
    storage_options: dict[str, Any] | None = None,
    **kwargs: Any,  # TODO: type kwargs as valid args to async_api.open
) -> Array | Group:
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
    zarr_format : {2, 3, None}, optional
        The zarr format to use when saving.
    path : str or None, optional
        The path within the store to open.
    storage_options : dict
        If using an fsspec URL to create the store, these will be passed to
        the backend implementation. Ignored otherwise.
    **kwargs
        Additional parameters are passed through to :func:`zarr.creation.open_array` or
        :func:`zarr.hierarchy.open_group`.

    Returns
    -------
    z : array or group
        Return type depends on what exists in the given store.
    """
    obj = sync(
        async_api.open(
            store=store,
            mode=mode,
            zarr_version=zarr_version,
            zarr_format=zarr_format,
            path=path,
            storage_options=storage_options,
            **kwargs,
        )
    )
    if isinstance(obj, AsyncArray):
        return Array(obj)
    else:
        return Group(obj)


def open_consolidated(*args: Any, use_consolidated: Literal[True] = True, **kwargs: Any) -> Group:
    """
    Alias for :func:`open_group` with ``use_consolidated=True``.
    """
    return Group(
        sync(async_api.open_consolidated(*args, use_consolidated=use_consolidated, **kwargs))
    )


def save(
    store: StoreLike,
    *args: NDArrayLike,
    zarr_version: ZarrFormat | None = None,  # deprecated
    zarr_format: ZarrFormat | None = None,
    path: str | None = None,
    **kwargs: Any,  # TODO: type kwargs as valid args to async_api.save
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
    return sync(
        async_api.save(
            store, *args, zarr_version=zarr_version, zarr_format=zarr_format, path=path, **kwargs
        )
    )


@_deprecate_positional_args
def save_array(
    store: StoreLike,
    arr: NDArrayLike,
    *,
    zarr_version: ZarrFormat | None = None,  # deprecated
    zarr_format: ZarrFormat | None = None,
    path: str | None = None,
    storage_options: dict[str, Any] | None = None,
    **kwargs: Any,  # TODO: type kwargs as valid args to async_api.save_array
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
        The zarr format to use when saving.
    path : str or None, optional
        The path within the store where the array will be saved.
    storage_options : dict
        If using an fsspec URL to create the store, these will be passed to
        the backend implementation. Ignored otherwise.
    **kwargs
        Passed through to :func:`create`, e.g., compressor.
    """
    return sync(
        async_api.save_array(
            store=store,
            arr=arr,
            zarr_version=zarr_version,
            zarr_format=zarr_format,
            path=path,
            storage_options=storage_options,
            **kwargs,
        )
    )


def save_group(
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

    return sync(
        async_api.save_group(
            store,
            *args,
            zarr_version=zarr_version,
            zarr_format=zarr_format,
            path=path,
            storage_options=storage_options,
            **kwargs,
        )
    )


@deprecated("Use Group.tree instead.")
def tree(grp: Group, expand: bool | None = None, level: int | None = None) -> Any:
    """Provide a rich display of the hierarchy.

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

    .. deprecated:: 3.0.0
        `zarr.tree()` is deprecated and will be removed in a future release.
        Use `group.tree()` instead.
    """
    return sync(async_api.tree(grp._async_group, expand=expand, level=level))


# TODO: add type annotations for kwargs
def array(data: npt.ArrayLike, **kwargs: Any) -> Array:
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

    return Array(sync(async_api.array(data=data, **kwargs)))


@_deprecate_positional_args
def group(
    store: StoreLike | None = None,
    *,
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
) -> Group:
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
    return Group(
        sync(
            async_api.group(
                store=store,
                overwrite=overwrite,
                chunk_store=chunk_store,
                cache_attrs=cache_attrs,
                synchronizer=synchronizer,
                path=path,
                zarr_version=zarr_version,
                zarr_format=zarr_format,
                meta_array=meta_array,
                attributes=attributes,
                storage_options=storage_options,
            )
        )
    )


@_deprecate_positional_args
def open_group(
    store: StoreLike | None = None,
    *,
    mode: AccessModeLiteral = "a",
    cache_attrs: bool | None = None,  # default changed, not used in async api
    synchronizer: Any = None,  # not used in async api
    path: str | None = None,
    chunk_store: StoreLike | None = None,  # not used in async api
    storage_options: dict[str, Any] | None = None,  # not used in async api
    zarr_version: ZarrFormat | None = None,  # deprecated
    zarr_format: ZarrFormat | None = None,
    meta_array: Any | None = None,  # not used in async api
    attributes: dict[str, JSON] | None = None,
    use_consolidated: bool | str | None = None,
) -> Group:
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
        store (in the ``zarr.json`` for Zarr v3 and in the ``.zmetadata`` file
        for Zarr v2).

        To explicitly require consolidated metadata, set ``use_consolidated=True``,
        which will raise an exception if consolidated metadata is not found.

        To explicitly *not* use consolidated metadata, set ``use_consolidated=False``,
        which will fall back to using the regular, non consolidated metadata.

        Zarr v2 allowed configuring the key storing the consolidated metadata
        (``.zmetadata`` by default). Specify the custom key as ``use_consolidated``
        to load consolidated metadata from a non-default key.

    Returns
    -------
    g : group
        The new group.
    """
    return Group(
        sync(
            async_api.open_group(
                store=store,
                mode=mode,
                cache_attrs=cache_attrs,
                synchronizer=synchronizer,
                path=path,
                chunk_store=chunk_store,
                storage_options=storage_options,
                zarr_version=zarr_version,
                zarr_format=zarr_format,
                meta_array=meta_array,
                attributes=attributes,
                use_consolidated=use_consolidated,
            )
        )
    )


# TODO: add type annotations for kwargs
def create(
    shape: ChunkCoords | int,
    *,  # Note: this is a change from v2
    chunks: ChunkCoords | int | None = None,  # TODO: v2 allowed chunks=True
    dtype: npt.DTypeLike | None = None,
    compressor: dict[str, JSON] | None = None,  # TODO: default and type change
    fill_value: Any | None = 0,  # TODO: need type
    order: MemoryOrder | None = None,
    store: str | StoreLike | None = None,
    synchronizer: Any | None = None,
    overwrite: bool = False,
    path: PathLike | None = None,
    chunk_store: StoreLike | None = None,
    filters: list[dict[str, JSON]] | None = None,  # TODO: type has changed
    cache_metadata: bool | None = None,
    cache_attrs: bool | None = None,
    read_only: bool | None = None,
    object_codec: Codec | None = None,  # TODO: type has changed
    dimension_separator: Literal[".", "/"] | None = None,
    write_empty_chunks: bool = False,  # TODO: default has changed
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
    dimension_names: Iterable[str] | None = None,
    storage_options: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Array:
    """Create an array.

    Parameters
    ----------
    shape : int or tuple of ints
        Array shape.
    chunks : int or tuple of ints, optional
        Chunk shape. If True, will be guessed from `shape` and `dtype`. If
        False, will be set to `shape`, i.e., single chunk for the whole array.
        If an int, the chunk size in each dimension will be given by the value
        of `chunks`. Default is True.
    dtype : str or dtype, optional
        NumPy dtype.
    compressor : Codec, optional
        Primary compressor.
    fill_value : object
        Default value to use for uninitialized portions of the array.
    order : {'C', 'F'}, optional
        Memory layout to be used within each chunk.
        Default is set in Zarr's config (`array.order`).
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

        .. versionadded:: 2.8

    write_empty_chunks : bool, optional
        If True (default), all chunks will be stored regardless of their
        contents. If False, each chunk is compared to the array's fill value
        prior to storing. If a chunk is uniformly equal to the fill value, then
        that chunk is not be stored, and the store entry for that chunk's key
        is deleted. This setting enables sparser storage, as only chunks with
        non-fill-value data are stored, at the expense of overhead associated
        with checking the data of each chunk.

        .. versionadded:: 2.11

    zarr_format : {2, 3, None}, optional
        The zarr format to use when saving.
    meta_array : array-like, optional
        An array instance to use for determining arrays to create and return
        to users. Use `numpy.empty(())` by default.

        .. versionadded:: 2.13
    storage_options : dict
        If using an fsspec URL to create the store, these will be passed to
        the backend implementation. Ignored otherwise.

    Returns
    -------
    z : array
        The array.
    """
    return Array(
        sync(
            async_api.create(
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                compressor=compressor,
                fill_value=fill_value,
                order=order,
                store=store,
                synchronizer=synchronizer,
                overwrite=overwrite,
                path=path,
                chunk_store=chunk_store,
                filters=filters,
                cache_metadata=cache_metadata,
                cache_attrs=cache_attrs,
                read_only=read_only,
                object_codec=object_codec,
                dimension_separator=dimension_separator,
                write_empty_chunks=write_empty_chunks,
                zarr_version=zarr_version,
                zarr_format=zarr_format,
                meta_array=meta_array,
                attributes=attributes,
                chunk_shape=chunk_shape,
                chunk_key_encoding=chunk_key_encoding,
                codecs=codecs,
                dimension_names=dimension_names,
                storage_options=storage_options,
                **kwargs,
            )
        )
    )


# TODO: add type annotations for kwargs
def empty(shape: ChunkCoords, **kwargs: Any) -> Array:
    """Create an empty array.

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
    return Array(sync(async_api.empty(shape, **kwargs)))


# TODO: move ArrayLike to common module
# TODO: add type annotations for kwargs
def empty_like(a: ArrayLike, **kwargs: Any) -> Array:
    """Create an empty array like `a`.

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
    return Array(sync(async_api.empty_like(a, **kwargs)))


# TODO: add type annotations for kwargs and fill_value
def full(shape: ChunkCoords, fill_value: Any, **kwargs: Any) -> Array:
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
    return Array(sync(async_api.full(shape=shape, fill_value=fill_value, **kwargs)))


# TODO: move ArrayLike to common module
# TODO: add type annotations for kwargs
def full_like(a: ArrayLike, **kwargs: Any) -> Array:
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
    return Array(sync(async_api.full_like(a, **kwargs)))


# TODO: add type annotations for kwargs
def ones(shape: ChunkCoords, **kwargs: Any) -> Array:
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
    return Array(sync(async_api.ones(shape, **kwargs)))


# TODO: add type annotations for kwargs
def ones_like(a: ArrayLike, **kwargs: Any) -> Array:
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
    return Array(sync(async_api.ones_like(a, **kwargs)))


# TODO: update this once async_api.open_array is fully implemented
def open_array(
    store: StoreLike | None = None,
    *,
    zarr_version: ZarrFormat | None = None,
    path: PathLike = "",
    storage_options: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Array:
    """Open an array using file-mode-like semantics.

    Parameters
    ----------
    store : Store or str
        Store or path to directory in file system or name of zip file.
    zarr_version : {2, 3, None}, optional
        The zarr format to use when saving.
    path : str, optional
        Path in store to array.
    storage_options : dict
        If using an fsspec URL to create the store, these will be passed to
        the backend implementation. Ignored otherwise.
    **kwargs
        Any keyword arguments to pass to ``create``.

    Returns
    -------
    AsyncArray
        The opened array.
    """
    return Array(
        sync(
            async_api.open_array(
                store=store,
                zarr_version=zarr_version,
                path=path,
                storage_options=storage_options,
                **kwargs,
            )
        )
    )


# TODO: add type annotations for kwargs
def open_like(a: ArrayLike, path: str, **kwargs: Any) -> Array:
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
    return Array(sync(async_api.open_like(a, path=path, **kwargs)))


# TODO: add type annotations for kwargs
def zeros(shape: ChunkCoords, **kwargs: Any) -> Array:
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
    return Array(sync(async_api.zeros(shape=shape, **kwargs)))


# TODO: add type annotations for kwargs
def zeros_like(a: ArrayLike, **kwargs: Any) -> Array:
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
    return Array(sync(async_api.zeros_like(a, **kwargs)))
