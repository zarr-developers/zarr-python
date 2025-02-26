from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from typing_extensions import deprecated

import zarr.api.asynchronous as async_api
import zarr.core.array
from zarr._compat import _deprecate_positional_args
from zarr.core.array import Array, AsyncArray
from zarr.core.group import Group
from zarr.core.sync import sync
from zarr.core.sync_group import create_hierarchy

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np
    import numpy.typing as npt

    from zarr.abc.codec import Codec
    from zarr.api.asynchronous import ArrayLike, PathLike
    from zarr.core.array import (
        CompressorsLike,
        FiltersLike,
        SerializerLike,
        ShardsLike,
    )
    from zarr.core.array_spec import ArrayConfigLike
    from zarr.core.buffer import NDArrayLike
    from zarr.core.chunk_key_encodings import ChunkKeyEncoding, ChunkKeyEncodingLike
    from zarr.core.common import (
        JSON,
        AccessModeLiteral,
        ChunkCoords,
        MemoryOrder,
        ShapeLike,
        ZarrFormat,
    )
    from zarr.storage import StoreLike

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
    group: Group
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
    """Open a group or array using file-mode-like semantics.

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
        Additional parameters are passed through to :func:`zarr.api.asynchronous.open_array` or
        :func:`zarr.api.asynchronous.open_group`.

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
    """Save an array or group of arrays to the local file system.

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
    """Save a NumPy array to the local file system.

    Follows a similar API to the NumPy save() function.

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
    """Save several NumPy arrays to the local file system.

    Follows a similar API to the NumPy savez()/savez_compressed() functions.

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
    array : Array
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
    g : Group
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
        store (in the ``zarr.json`` for Zarr format 3 and in the ``.zmetadata`` file
        for Zarr format 2).

        To explicitly require consolidated metadata, set ``use_consolidated=True``,
        which will raise an exception if consolidated metadata is not found.

        To explicitly *not* use consolidated metadata, set ``use_consolidated=False``,
        which will fall back to using the regular, non consolidated metadata.

        Zarr format 2 allows configuring the key storing the consolidated metadata
        (``.zmetadata`` by default). Specify the custom key as ``use_consolidated``
        to load consolidated metadata from a non-default key.

    Returns
    -------
    g : Group
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


def create_group(
    store: StoreLike,
    *,
    path: str | None = None,
    zarr_format: ZarrFormat | None = None,
    overwrite: bool = False,
    attributes: dict[str, Any] | None = None,
    storage_options: dict[str, Any] | None = None,
) -> Group:
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
    Group
        The new group.
    """
    return Group(
        sync(
            async_api.create_group(
                store=store,
                path=path,
                overwrite=overwrite,
                storage_options=storage_options,
                zarr_format=zarr_format,
                attributes=attributes,
            )
        )
    )


# TODO: add type annotations for kwargs
def create(
    shape: ChunkCoords | int,
    *,  # Note: this is a change from v2
    chunks: ChunkCoords | int | bool | None = None,
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
    write_empty_chunks: bool | None = None,  # TODO: default has changed
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
    config: ArrayConfigLike | None = None,
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
    meta_array : array-like, optional
        An array instance to use for determining arrays to create and return
        to users. Use `numpy.empty(())` by default.
    storage_options : dict
        If using an fsspec URL to create the store, these will be passed to
        the backend implementation. Ignored otherwise.
    config : ArrayConfigLike, optional
        Runtime configuration of the array. If provided, will override the
        default values from `zarr.config.array`.

    Returns
    -------
    z : Array
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
                config=config,
                **kwargs,
            )
        )
    )


def create_array(
    store: str | StoreLike,
    *,
    name: str | None = None,
    shape: ShapeLike | None = None,
    dtype: npt.DTypeLike | None = None,
    data: np.ndarray[Any, np.dtype[Any]] | None = None,
    chunks: ChunkCoords | Literal["auto"] = "auto",
    shards: ShardsLike | None = None,
    filters: FiltersLike = "auto",
    compressors: CompressorsLike = "auto",
    serializer: SerializerLike = "auto",
    fill_value: Any | None = None,
    order: MemoryOrder | None = None,
    zarr_format: ZarrFormat | None = 3,
    attributes: dict[str, JSON] | None = None,
    chunk_key_encoding: ChunkKeyEncoding | ChunkKeyEncodingLike | None = None,
    dimension_names: Iterable[str] | None = None,
    storage_options: dict[str, Any] | None = None,
    overwrite: bool = False,
    config: ArrayConfigLike | None = None,
) -> Array:
    """Create an array.

    This function wraps :func:`zarr.core.array.create_array`.

    Parameters
    ----------
    store : str or Store
        Store or path to directory in file system or name of zip file.
    name : str or None, optional
        The name of the array within the store. If ``name`` is ``None``, the array will be located
        at the root of the store.
    shape : ChunkCoords, optional
        Shape of the array. Can be ``None`` if ``data`` is provided.
    dtype : npt.DTypeLike, optional
        Data type of the array. Can be ``None`` if ``data`` is provided.
    data : np.ndarray, optional
        Array-like data to use for initializing the array. If this parameter is provided, the
        ``shape`` and ``dtype`` parameters must be identical to ``data.shape`` and ``data.dtype``,
        or ``None``.
    chunks : ChunkCoords, optional
        Chunk shape of the array.
        If not specified, default are guessed based on the shape and dtype.
    shards : ChunkCoords, optional
        Shard shape of the array. The default value of ``None`` results in no sharding at all.
    filters : Iterable[Codec], optional
        Iterable of filters to apply to each chunk of the array, in order, before serializing that
        chunk to bytes.

        For Zarr format 3, a "filter" is a codec that takes an array and returns an array,
        and these values must be instances of ``ArrayArrayCodec``, or dict representations
        of ``ArrayArrayCodec``.
        If no ``filters`` are provided, a default set of filters will be used.
        These defaults can be changed by modifying the value of ``array.v3_default_filters``
        in :mod:`zarr.core.config`.
        Use ``None`` to omit default filters.

        For Zarr format 2, a "filter" can be any numcodecs codec; you should ensure that the
        the order if your filters is consistent with the behavior of each filter.
        If no ``filters`` are provided, a default set of filters will be used.
        These defaults can be changed by modifying the value of ``array.v2_default_filters``
        in :mod:`zarr.core.config`.
        Use ``None`` to omit default filters.
    compressors : Iterable[Codec], optional
        List of compressors to apply to the array. Compressors are applied in order, and after any
        filters are applied (if any are specified) and the data is serialized into bytes.

        For Zarr format 3, a "compressor" is a codec that takes a bytestream, and
        returns another bytestream. Multiple compressors my be provided for Zarr format 3.
        If no ``compressors`` are provided, a default set of compressors will be used.
        These defaults can be changed by modifying the value of ``array.v3_default_compressors``
        in :mod:`zarr.core.config`.
        Use ``None`` to omit default compressors.

        For Zarr format 2, a "compressor" can be any numcodecs codec. Only a single compressor may
        be provided for Zarr format 2.
        If no ``compressor`` is provided, a default compressor will be used.
        in :mod:`zarr.core.config`.
        Use ``None`` to omit the default compressor.
    serializer : dict[str, JSON] | ArrayBytesCodec, optional
        Array-to-bytes codec to use for encoding the array data.
        Zarr format 3 only. Zarr format 2 arrays use implicit array-to-bytes conversion.
        If no ``serializer`` is provided, a default serializer will be used.
        These defaults can be changed by modifying the value of ``array.v3_default_serializer``
        in :mod:`zarr.core.config`.
    fill_value : Any, optional
        Fill value for the array.
    order : {"C", "F"}, optional
        The memory of the array (default is "C").
        For Zarr format 2, this parameter sets the memory order of the array.
        For Zarr format 3, this parameter is deprecated, because memory order
        is a runtime parameter for Zarr format 3 arrays. The recommended way to specify the memory
        order for Zarr format 3 arrays is via the ``config`` parameter, e.g. ``{'config': 'C'}``.
        If no ``order`` is provided, a default order will be used.
        This default can be changed by modifying the value of ``array.order`` in :mod:`zarr.core.config`.
    zarr_format : {2, 3}, optional
        The zarr format to use when saving.
    attributes : dict, optional
        Attributes for the array.
    chunk_key_encoding : ChunkKeyEncoding, optional
        A specification of how the chunk keys are represented in storage.
        For Zarr format 3, the default is ``{"name": "default", "separator": "/"}}``.
        For Zarr format 2, the default is ``{"name": "v2", "separator": "."}}``.
    dimension_names : Iterable[str], optional
        The names of the dimensions (default is None).
        Zarr format 3 only. Zarr format 2 arrays should not use this parameter.
    storage_options : dict, optional
        If using an fsspec URL to create the store, these will be passed to the backend implementation.
        Ignored otherwise.
    overwrite : bool, default False
        Whether to overwrite an array with the same name in the store, if one exists.
    config : ArrayConfigLike, optional
        Runtime configuration for the array.

    Returns
    -------
    Array
        The array.

    Examples
    --------
    >>> import zarr
    >>> store = zarr.storage.MemoryStore(mode='w')
    >>> arr = await zarr.create_array(
    >>>     store=store,
    >>>     shape=(100,100),
    >>>     chunks=(10,10),
    >>>     dtype='i4',
    >>>     fill_value=0)
    <Array memory://140349042942400 shape=(100, 100) dtype=int32>
    """
    return Array(
        sync(
            zarr.core.array.create_array(
                store,
                name=name,
                shape=shape,
                dtype=dtype,
                data=data,
                chunks=chunks,
                shards=shards,
                filters=filters,
                compressors=compressors,
                serializer=serializer,
                fill_value=fill_value,
                order=order,
                zarr_format=zarr_format,
                attributes=attributes,
                chunk_key_encoding=chunk_key_encoding,
                dimension_names=dimension_names,
                storage_options=storage_options,
                overwrite=overwrite,
                config=config,
            )
        )
    )


# TODO: add type annotations for kwargs
def empty(shape: ChunkCoords, **kwargs: Any) -> Array:
    """Create an empty array with the specified shape. The contents will be filled with the
    array's fill value or zeros if no fill value is provided.

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
    """Create an empty array like another array. The contents will be filled with the
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
    return Array(sync(async_api.empty_like(a, **kwargs)))


# TODO: add type annotations for kwargs and fill_value
def full(shape: ChunkCoords, fill_value: Any, **kwargs: Any) -> Array:
    """Create an array with a default fill value.

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
    """Create a filled array like another array.

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
    """Create an array with a fill value of one.

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
    """Create an array of ones like another array.

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
    """Open a persistent array like another array.

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
    """Create an array with a fill value of zero.

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
    """Create an array of zeros like another array.

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
