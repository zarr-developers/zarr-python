from __future__ import annotations

import asyncio
import warnings
from collections.abc import Iterable
from typing import Any, Literal, Union, cast

import numpy as np
import numpy.typing as npt

from zarr.abc.codec import Codec
from zarr.array import Array, AsyncArray
from zarr.buffer import NDArrayLike
from zarr.chunk_key_encodings import ChunkKeyEncoding
from zarr.common import JSON, ChunkCoords, MemoryOrder, OpenMode, ZarrFormat
from zarr.group import AsyncGroup
from zarr.metadata import ArrayV2Metadata, ArrayV3Metadata
from zarr.store import (
    StoreLike,
    make_store_path,
)

# TODO: this type could use some more thought, noqa to avoid "Variable "asynchronous.ArrayLike" is not valid as a type"
ArrayLike = Union[AsyncArray | Array | npt.NDArray[Any]]  # noqa
PathLike = str


def _get_shape_chunks(a: ArrayLike | Any) -> tuple[ChunkCoords | None, ChunkCoords | None]:
    """helper function to get the shape and chunks from an array-like object"""
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
    """set default values for shape and chunks if they are not present in the array-like object"""

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

        if isinstance(a.metadata, ArrayV3Metadata):
            new["codecs"] = a.metadata.codecs
        else:
            raise ValueError(f"Unsupported zarr format: {a.metadata.zarr_format}")
    else:
        # TODO: set default values compressor/codecs
        # to do this, we may need to evaluate if this is a v2 or v3 array
        # new["compressor"] = "default"
        pass

    return new


def _handle_zarr_version_or_format(
    *, zarr_version: ZarrFormat | None, zarr_format: ZarrFormat | None
) -> ZarrFormat | None:
    """handle the deprecated zarr_version kwarg and return zarr_format"""
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


def _default_zarr_version() -> ZarrFormat:
    """return the default zarr_version"""
    # TODO: set default value from config
    return 3


async def consolidate_metadata(*args: Any, **kwargs: Any) -> AsyncGroup:
    raise NotImplementedError


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
) -> NDArrayLike | dict[str, NDArrayLike]:
    """Load data from an array or group into memory.

    Parameters
    ----------
    store : Store or string
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
    mode: OpenMode | None = None,  # type and value changed
    zarr_version: ZarrFormat | None = None,  # deprecated
    zarr_format: ZarrFormat | None = None,
    path: str | None = None,
    **kwargs: Any,  # TODO: type kwargs as valid args to open_array
) -> AsyncArray | AsyncGroup:
    """Convenience function to open a group or array using file-mode-like semantics.

    Parameters
    ----------
    store : Store or string, optional
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
    **kwargs
        Additional parameters are passed through to :func:`zarr.creation.open_array` or
        :func:`zarr.hierarchy.open_group`.

    Returns
    -------
    z : array or group
        Return type depends on what exists in the given store.
    """
    zarr_format = _handle_zarr_version_or_format(zarr_version=zarr_version, zarr_format=zarr_format)
    store_path = make_store_path(store, mode=mode)

    if path is not None:
        store_path = store_path / path

    try:
        return await open_array(store=store_path, zarr_format=zarr_format, **kwargs)
    except KeyError:
        return await open_group(store=store_path, zarr_format=zarr_format, **kwargs)


async def open_consolidated(*args: Any, **kwargs: Any) -> AsyncGroup:
    raise NotImplementedError


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
    store : Store or string
        Store or path to directory in file system or name of zip file.
    args : ndarray
        NumPy arrays with data to save.
    zarr_format : {2, 3, None}, optional
        The zarr format to use when saving.
    path : str or None, optional
        The path within the group where the arrays will be saved.
    kwargs
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
    **kwargs: Any,  # TODO: type kwargs as valid args to create
) -> None:
    """Convenience function to save a NumPy array to the local file system, following a
    similar API to the NumPy save() function.

    Parameters
    ----------
    store : Store or string
        Store or path to directory in file system or name of zip file.
    arr : ndarray
        NumPy array with data to save.
    zarr_format : {2, 3, None}, optional
        The zarr format to use when saving.
    path : str or None, optional
        The path within the store where the array will be saved.
    kwargs
        Passed through to :func:`create`, e.g., compressor.
    """
    zarr_format = (
        _handle_zarr_version_or_format(zarr_version=zarr_version, zarr_format=zarr_format)
        or _default_zarr_version()
    )

    store_path = make_store_path(store, mode="w")
    if path is not None:
        store_path = store_path / path
    new = await AsyncArray.create(
        store_path,
        zarr_format=zarr_format,
        shape=arr.shape,
        dtype=arr.dtype,
        chunks=arr.shape,
        **kwargs,
    )
    await new.setitem(slice(None), arr)


async def save_group(
    store: StoreLike,
    *args: NDArrayLike,
    zarr_version: ZarrFormat | None = None,  # deprecated
    zarr_format: ZarrFormat | None = None,
    path: str | None = None,
    **kwargs: NDArrayLike,
) -> None:
    """Convenience function to save several NumPy arrays to the local file system, following a
    similar API to the NumPy savez()/savez_compressed() functions.

    Parameters
    ----------
    store : Store or string
        Store or path to directory in file system or name of zip file.
    args : ndarray
        NumPy arrays with data to save.
    zarr_format : {2, 3, None}, optional
        The zarr format to use when saving.
    path : str or None, optional
        Path within the store where the group will be saved.
    kwargs
        NumPy arrays with data to save.
    """
    zarr_format = _handle_zarr_version_or_format(zarr_version=zarr_version, zarr_format=zarr_format)

    if len(args) == 0 and len(kwargs) == 0:
        raise ValueError("at least one array must be provided")
    aws = []
    for i, arr in enumerate(args):
        aws.append(save_array(store, arr, zarr_format=zarr_format, path=f"{path}/arr_{i}"))
    for k, arr in kwargs.items():
        _path = f"{path}/{k}" if path is not None else k
        aws.append(save_array(store, arr, zarr_format=zarr_format, path=_path))
    await asyncio.gather(*aws)


async def tree(*args: Any, **kwargs: Any) -> None:
    raise NotImplementedError


async def array(data: npt.ArrayLike, **kwargs: Any) -> AsyncArray:
    """Create an array filled with `data`.

    Parameters
    ----------
    data : array_like
        The data to fill the array with.
    kwargs
        Passed through to :func:`create`.

    Returns
    -------
    array : array
        The new array.
    """

    # ensure data is array-like
    if not hasattr(data, "shape") or not hasattr(data, "dtype"):
        data = np.asanyarray(data)

    # setup dtype
    kw_dtype = kwargs.get("dtype")
    if kw_dtype is None:
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
    await z.setitem(slice(None), data)

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
) -> AsyncGroup:
    """Create a group.

    Parameters
    ----------
    store : Store or string, optional
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
    path : string, optional
        Group path within store.
    meta_array : array-like, optional
        An array instance to use for determining arrays to create and return
        to users. Use `numpy.empty(())` by default.
    zarr_format : {2, 3, None}, optional
        The zarr format to use when saving.

    Returns
    -------
    g : group
        The new group.
    """

    zarr_format = (
        _handle_zarr_version_or_format(zarr_version=zarr_version, zarr_format=zarr_format)
        or _default_zarr_version()
    )

    store_path = make_store_path(store)
    if path is not None:
        store_path = store_path / path

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
        return await AsyncGroup.create(
            store=store_path,
            zarr_format=zarr_format,
            exists_ok=overwrite,
            attributes=attributes,
        )


async def open_group(
    *,  # Note: this is a change from v2
    store: StoreLike | None = None,
    mode: OpenMode | None = None,  # not used
    cache_attrs: bool | None = None,  # not used, default changed
    synchronizer: Any = None,  # not used
    path: str | None = None,
    chunk_store: StoreLike | None = None,  # not used
    storage_options: dict[str, Any] | None = None,  # not used
    zarr_version: ZarrFormat | None = None,  # deprecated
    zarr_format: ZarrFormat | None = None,
    meta_array: Any | None = None,  # not used
    attributes: dict[str, JSON] | None = None,
) -> AsyncGroup:
    """Open a group using file-mode-like semantics.

    Parameters
    ----------
    store : Store or string, optional
        Store or path to directory in file system or name of zip file.
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
    path : string, optional
        Group path within store.
    chunk_store : Store or string, optional
        Store or path to directory in file system or name of zip file.
    storage_options : dict
        If using an fsspec URL to create the store, these will be passed to
        the backend implementation. Ignored otherwise.
    meta_array : array-like, optional
        An array instance to use for determining arrays to create and return
        to users. Use `numpy.empty(())` by default.

    Returns
    -------
    g : group
        The new group.
    """

    zarr_format = (
        _handle_zarr_version_or_format(zarr_version=zarr_version, zarr_format=zarr_format)
        or _default_zarr_version()
    )

    if cache_attrs is not None:
        warnings.warn("cache_attrs is not yet implemented", RuntimeWarning, stacklevel=2)
    if synchronizer is not None:
        warnings.warn("synchronizer is not yet implemented", RuntimeWarning, stacklevel=2)
    if meta_array is not None:
        warnings.warn("meta_array is not yet implemented", RuntimeWarning, stacklevel=2)
    if chunk_store is not None:
        warnings.warn("chunk_store is not yet implemented", RuntimeWarning, stacklevel=2)
    if storage_options is not None:
        warnings.warn("storage_options is not yet implemented", RuntimeWarning, stacklevel=2)

    store_path = make_store_path(store, mode=mode)
    if path is not None:
        store_path = store_path / path

    if attributes is None:
        attributes = {}

    try:
        return await AsyncGroup.open(store_path, zarr_format=zarr_format)
    except (KeyError, FileNotFoundError):
        return await AsyncGroup.create(
            store_path, zarr_format=zarr_format, exists_ok=True, attributes=attributes
        )


async def create(
    shape: ChunkCoords,
    *,  # Note: this is a change from v2
    chunks: ChunkCoords | None = None,  # TODO: v2 allowed chunks=True
    dtype: npt.DTypeLike | None = None,
    compressor: dict[str, JSON] | None = None,  # TODO: default and type change
    fill_value: Any = 0,  # TODO: need type
    order: MemoryOrder | None = None,  # TODO: default change
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
    chunk_shape: ChunkCoords | None = None,
    chunk_key_encoding: (
        ChunkKeyEncoding
        | tuple[Literal["default"], Literal[".", "/"]]
        | tuple[Literal["v2"], Literal[".", "/"]]
        | None
    ) = None,
    codecs: Iterable[Codec | dict[str, JSON]] | None = None,
    dimension_names: Iterable[str] | None = None,
    **kwargs: Any,
) -> AsyncArray:
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
    dtype : string or dtype, optional
        NumPy dtype.
    compressor : Codec, optional
        Primary compressor.
    fill_value : object
        Default value to use for uninitialized portions of the array.
    order : {'C', 'F'}, optional
        Memory layout to be used within each chunk.
    store : Store or string
        Store or path to directory in file system or name of zip file.
    synchronizer : object, optional
        Array synchronizer.
    overwrite : bool, optional
        If True, delete all pre-existing data in `store` at `path` before
        creating the array.
    path : string, optional
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

    Returns
    -------
    z : array
        The array.
    """
    zarr_format = (
        _handle_zarr_version_or_format(zarr_version=zarr_version, zarr_format=zarr_format)
        or _default_zarr_version()
    )

    if zarr_format == 2 and chunks is None:
        chunks = shape
    if zarr_format == 3 and chunk_shape is None:
        if chunks is not None:
            chunk_shape = chunks
            chunks = None
        else:
            chunk_shape = shape

    if order is not None:
        warnings.warn(
            "order is deprecated, use config `array.order` instead",
            DeprecationWarning,
            stacklevel=2,
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
    if dimension_separator is not None:
        if zarr_format == 3:
            raise ValueError(
                "dimension_separator is not supported for zarr format 3, use chunk_key_encoding instead"
            )
        else:
            warnings.warn(
                "dimension_separator is not yet implemented", RuntimeWarning, stacklevel=2
            )
    if write_empty_chunks:
        warnings.warn("write_empty_chunks is not yet implemented", RuntimeWarning, stacklevel=2)
    if meta_array is not None:
        warnings.warn("meta_array is not yet implemented", RuntimeWarning, stacklevel=2)

    mode = cast(OpenMode, "r" if read_only else "w")
    store_path = make_store_path(store, mode=mode)
    if path is not None:
        store_path = store_path / path

    return await AsyncArray.create(
        store_path,
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        compressor=compressor,
        fill_value=fill_value,
        exists_ok=overwrite,  # TODO: name change
        filters=filters,
        dimension_separator=dimension_separator,
        zarr_format=zarr_format,
        chunk_shape=chunk_shape,
        chunk_key_encoding=chunk_key_encoding,
        codecs=codecs,
        dimension_names=dimension_names,
        attributes=attributes,
        **kwargs,
    )


async def empty(shape: ChunkCoords, **kwargs: Any) -> AsyncArray:
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
    return await create(shape=shape, fill_value=None, **kwargs)


async def empty_like(a: ArrayLike, **kwargs: Any) -> AsyncArray:
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
    like_kwargs = _like_args(a, kwargs)
    return await empty(**like_kwargs)


# TODO: add type annotations for fill_value and kwargs
async def full(shape: ChunkCoords, fill_value: Any, **kwargs: Any) -> AsyncArray:
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
async def full_like(a: ArrayLike, **kwargs: Any) -> AsyncArray:
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
        kwargs.setdefault("fill_value", a.metadata.fill_value)
    return await full(**like_kwargs)


async def ones(shape: ChunkCoords, **kwargs: Any) -> AsyncArray:
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


async def ones_like(a: ArrayLike, **kwargs: Any) -> AsyncArray:
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
    path: PathLike | None = None,
    **kwargs: Any,  # TODO: type kwargs as valid args to save
) -> AsyncArray:
    """Open an array using file-mode-like semantics.

    Parameters
    ----------
    store : Store or string
        Store or path to directory in file system or name of zip file.
    zarr_format : {2, 3, None}, optional
        The zarr format to use when saving.
    path : string, optional
        Path in store to array.
    **kwargs
        Any keyword arguments to pass to the array constructor.

    Returns
    -------
    AsyncArray
        The opened array.
    """

    store_path = make_store_path(store)
    if path is not None:
        store_path = store_path / path

    zarr_format = _handle_zarr_version_or_format(zarr_version=zarr_version, zarr_format=zarr_format)

    try:
        return await AsyncArray.open(store_path, zarr_format=zarr_format)
    except KeyError as e:
        if store_path.store.writeable:
            pass
        else:
            raise e

    # if array was not found, create it
    return await create(store=store, path=path, zarr_format=zarr_format, **kwargs)


async def open_like(a: ArrayLike, path: str, **kwargs: Any) -> AsyncArray:
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


async def zeros(shape: ChunkCoords, **kwargs: Any) -> AsyncArray:
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


async def zeros_like(a: ArrayLike, **kwargs: Any) -> AsyncArray:
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
