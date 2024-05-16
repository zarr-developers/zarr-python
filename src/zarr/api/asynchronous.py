from __future__ import annotations

import asyncio
import warnings
from typing import Union, Any, Literal, Iterable

import numpy as np
import numpy.typing as npt

from zarr.abc.codec import Codec
from zarr.array import AsyncArray, Array
from zarr.common import ZarrFormat, MEMORY_ORDER, JSON, ChunkCoords
from zarr.group import AsyncGroup
from zarr.metadata import ChunkKeyEncoding
from zarr.store import (
    StoreLike,
    make_store_path,
)

ShapeLike = Union[int, tuple[int, ...]]
ArrayLike = Union[AsyncArray, Array, npt.NDArray[Any]]


def _get_shape_chunks(a: ArrayLike | Any) -> tuple[ShapeLike | None, ChunkCoords | None]:
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


def _like_args(a: ArrayLike, kwargs: dict[str, Any]) -> None:
    shape, chunks = _get_shape_chunks(a)
    if shape is not None:
        kwargs.setdefault("shape", shape)
    if chunks is not None:
        kwargs.setdefault("chunks", chunks)

    if hasattr(a, "dtype"):
        kwargs.setdefault("dtype", a.dtype)

    if isinstance(a, AsyncArray):
        kwargs.setdefault("order", a.order)
        if a.metadata.zarr_format == 2:
            # TODO: make this v2/v3 aware
            kwargs.setdefault("compressor", a.metadata.compressor)
            kwargs.setdefault("filters", a.metadata.filters)

        elif a.metadata.zarr_format == 3:
            kwargs.setdefault("codecs", a.codecs)
        else:
            raise ValueError(f"Unsupported zarr format: {a.metadata.zarr_format}")
    else:
        # TODO: set default values compressor/codecs
        # to do this, we may need to evaluate if this is a v2 or v3 array
        # kwargs.setdefault("compressor", "default")
        pass


async def consolidate_metadata(*args: Any, **kwargs: Any) -> AsyncGroup:
    raise NotImplementedError


async def copy(*args: Any, **kwargs: Any) -> tuple[int, int, int]:
    raise NotImplementedError


async def copy_all(*args: Any, **kwargs: Any) -> tuple[int, int, int]:
    raise NotImplementedError


async def copy_store(*args: Any, **kwargs: Any) -> tuple[int, int, int]:
    raise NotImplementedError


async def load(
    store: StoreLike, zarr_version: ZarrFormat | None = None, path: str | None = None
) -> Union[AsyncArray, AsyncGroup]:
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
    if zarr_version is not None:
        warnings.warn(
            "zarr_version is deprecated and no longer required in load", DeprecationWarning
        )
    obj = await open(store, path=path)
    if isinstance(obj, AsyncArray):
        return await obj.getitem(slice(None))
    else:
        raise NotImplementedError("loading groups not yet supported")


async def open(
    store: StoreLike | None = None,
    mode: str = "a",
    *,
    zarr_version: ZarrFormat | None = None,
    zarr_format: ZarrFormat | None = None,
    path: str | None = None,
    **kwargs: Any,  # TODO: type kwargs as valid args to open_array
) -> Union[AsyncArray, AsyncGroup]:
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
    z : AsyncArray or AsyncGroup
        Array or group, depending on what exists in the given store.
    """
    if zarr_version is not None:
        warnings.warn("zarr_version is deprecated, use zarr_format", DeprecationWarning)
        zarr_format = zarr_version

    store_path = make_store_path(store)

    if path is not None:
        store_path = store_path / path

    warnings.warn("TODO: mode is ignored", RuntimeWarning)

    try:
        return await AsyncArray.open(store_path, zarr_format=zarr_format, **kwargs)
    except KeyError:
        return await AsyncGroup.open(store_path, zarr_format=zarr_format, **kwargs)


async def open_consolidated(*args: Any, **kwargs: Any) -> AsyncGroup:
    raise NotImplementedError


async def save(
    store: StoreLike,
    *args: npt.ArrayLike,
    zarr_version: ZarrFormat | None = None,
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
    if zarr_version is not None:
        warnings.warn("zarr_version is deprecated, use zarr_format", DeprecationWarning)
        zarr_format = zarr_version
    if len(args) == 0 and len(kwargs) == 0:
        raise ValueError("at least one array must be provided")
    if len(args) == 1 and len(kwargs) == 0:
        await save_array(store, args[0], zarr_format=zarr_format, path=path)
    else:
        await save_group(store, *args, zarr_format=zarr_format, path=path, **kwargs)


async def save_array(
    store: StoreLike,
    arr: npt.ArrayLike,
    *,
    zarr_version: ZarrFormat | None = None,
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
    if zarr_version is not None:
        warnings.warn("zarr_version is deprecated, use zarr_format", DeprecationWarning)
        zarr_format = zarr_version

    if zarr_format is None:
        zarr_format = 3  # TODO: perhaps this default should be set via config?

    store_path = make_store_path(store)
    if path is not None:
        store_path = store_path / path
    new = await AsyncArray.create(store_path, zarr_format=zarr_format, **kwargs)
    await new.setitem(slice(None), arr)


async def save_group(
    store: StoreLike,
    *args: npt.ArrayLike,
    zarr_version: ZarrFormat | None = None,
    zarr_format: ZarrFormat | None = None,
    path: str | None = None,
    **kwargs: npt.ArrayLike,
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
    if zarr_version is not None:
        warnings.warn("zarr_version is deprecated, use zarr_format", DeprecationWarning)
        zarr_format = zarr_version

    if len(args) == 0 and len(kwargs) == 0:
        raise ValueError("at least one array must be provided")
    aws = []
    for i, arr in enumerate(args):
        aws.append(save_array(store, arr, zarr_format=zarr_format, path=f"{path}/arr_{i}"))
    for k, arr in kwargs.items():
        aws.append(save_array(store, arr, zarr_format=zarr_format, path=f"{path}/{k}"))
    await asyncio.gather(*aws)


# async def tree(*args: Any, **kwargs: Any) -> "TreeViewer":
#     raise NotImplementedError


async def array(data: npt.ArrayLike, **kwargs: Any) -> AsyncArray:
    """Create an array filled with `data`.

    The `data` argument should be a array-like object. For
    other parameter definitions see :func:`zarr.api.asynchronous.create`.
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

    # pop read-only to apply after storing the data
    # read_only = kwargs.pop("read_only", False)

    # instantiate array
    z = await create(**kwargs)

    # fill with data
    await z.setitem(slice(None), data)

    # set read_only property afterwards
    # z.read_only = read_only

    return z


async def group(
    store: StoreLike | None = None,
    overwrite: bool = False,
    chunk_store: StoreLike | None = None,  # not used
    cache_attrs: bool = True,  # not used
    synchronizer: Any | None = None,  # not used
    path: str | None = None,
    *,
    zarr_version: ZarrFormat | None = None,
    zarr_format: ZarrFormat | None = None,
    meta_array: Any | None = None,  # not used
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
    g : AsyncGroup
    """

    if zarr_version is not None:
        zarr_format = zarr_version
        warnings.warn("zarr_format is deprecated, use zarr_format instead", DeprecationWarning)

    if zarr_format is None:
        zarr_format = 3  # TODO: perhaps this default should be set via config?

    store_path = make_store_path(store)
    if path is not None:
        store_path = store_path / path

    # requires_init = None
    # if zarr_version == 2:
    #     requires_init = overwrite or not contains_group(store)
    # elif zarr_version == 3:
    #     requires_init = overwrite or not contains_group(store, path)

    # if requires_init:
    #     init_group(store, overwrite=overwrite, chunk_store=chunk_store, path=path)

    try:
        return await AsyncGroup.open(store_path, zarr_format=zarr_format)
    except KeyError:
        # TODO: pass attributes here
        attributes: dict[str, Any] = {}
        return await AsyncGroup.create(
            store_path, zarr_format=zarr_format, exists_ok=overwrite, attributes=attributes
        )


async def open_group(
    store: StoreLike | None = None,
    mode: str = "a",  # not used
    cache_attrs: bool = True,  # not used
    synchronizer: Any = None,  # not used
    path: str | None = None,
    chunk_store: StoreLike | None = None,  # not used
    storage_options: dict[str, Any] | None = None,  # not used
    *,
    zarr_version: ZarrFormat | None = None,
    zarr_format: ZarrFormat | None = None,
    meta_array: Any | None = None,  # not used
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
    g : AsyncGroup
    """

    if zarr_version is not None:
        zarr_format = zarr_version
        warnings.warn("zarr_format is deprecated, use zarr_format instead", DeprecationWarning)
    if zarr_format is None:
        zarr_format = 3  # TODO: perhaps this default should be set via config?

    store_path = make_store_path(store)
    if path is not None:
        store_path = store_path / path

    return await AsyncGroup.open(store_path, zarr_format=zarr_format)


# TODO: require kwargs
async def create(
    shape: ShapeLike,
    chunks: Union[int, tuple[int, ...], bool] = True,
    dtype: npt.DTypeLike | None = None,
    compressor: str = "default",
    fill_value: int | None = 0,
    order: MEMORY_ORDER = "C",
    store: StoreLike | None = None,
    synchronizer: Any | None = None,
    overwrite: bool = False,
    path: str | None = None,
    # chunk_store: StoreLike | None = None,
    # filters: Sequence[Codec] | None = None,
    # cache_metadata: bool = True,
    # cache_attrs: bool = True,
    # read_only: bool = False,
    # object_codec: Codec | None = None,
    # dimension_separator: DIMENSION_SEPARATOR | None = None,
    # write_empty_chunks: bool = True,
    *,
    zarr_version: ZarrFormat | None = None,
    # meta_array: MetaArray | None = None,
    # storage_transformers: Sequence[StorageTransformer] = (),
    **kwargs: Any,  # TODO: type kwargs as valid args to AsyncArray.Create
) -> AsyncArray:
    store_path = make_store_path(store)
    if path is not None:
        store_path = store_path / path

    raise NotImplementedError
    # TODO: finish when Norman's PR goes in
    # return await AsyncArray.create(store_path, chunks=chunks, dtype=dtype, zarr_version=zarr_version, **kwargs)


async def empty(shape: ShapeLike, **kwargs: Any) -> AsyncArray:
    """Create an empty array.

    For parameter definitions see :func:`zarr.api.asynchronous.create`.

    Notes
    -----
    The contents of an empty Zarr array are not defined. On attempting to
    retrieve data from an empty Zarr array, any values may be returned,
    and these are not guaranteed to be stable from one access to the next.
    """
    return await create(shape=shape, fill_value=None, **kwargs)


async def empty_like(a: ArrayLike, **kwargs: Any) -> AsyncArray:
    """Create an empty array like `a`."""
    _like_args(a, kwargs)
    return await empty(**kwargs)


# TODO: add type annotations for fill_value and kwargs
async def full(shape: ShapeLike, fill_value: Any, **kwargs: Any) -> AsyncArray:
    """Create an array, with `fill_value` being used as the default value for
    uninitialized portions of the array.

    For parameter definitions see :func:`zarr.api.asynchronous.create`.
    """
    return await create(shape=shape, fill_value=fill_value, **kwargs)


# TODO: add type annotations for kwargs
async def full_like(a: ArrayLike, **kwargs: Any) -> AsyncArray:
    """Create a filled array like `a`."""
    _like_args(a, kwargs)
    if isinstance(a, AsyncArray):
        kwargs.setdefault("fill_value", a.metadata.fill_value)
    return await full(**kwargs)


async def ones(shape: ShapeLike, **kwargs: Any) -> AsyncArray:
    """Create an array, with one being used as the default value for
    uninitialized portions of the array.

    For parameter definitions see :func:`zarr.creation.create`.

    Returns
    -------
    Array
        The new array.
    """
    return await create(shape=shape, fill_value=1, **kwargs)


async def ones_like(a: ArrayLike, **kwargs: Any) -> AsyncArray:
    """Create an array of ones like `a`."""
    _like_args(a, kwargs)
    return await ones(**kwargs)


async def open_array(
    store: StoreLike | None = None,
    mode: str = "a",
    shape: ShapeLike | None = None,
    chunks: Union[int, tuple[int, ...], bool] = True,  # v2 only
    dtype: npt.DTypeLike | None = None,
    compressor: dict[str, JSON] | None = None,  # v2 only
    fill_value: Any | None = 0,  # note: default is 0 here and None on Array.create
    order: Literal["C", "F"] | None = "C",  # deprecate in favor of runtime config?
    synchronizer: Any = None,  # deprecate and catch
    filters: list[dict[str, JSON]] | None = None,  # v2 only
    cache_metadata: bool = True,  # not implemented
    cache_attrs: bool = True,  # not implemented
    path: str | None = None,
    object_codec: Any = None,  # not implemented
    chunk_store: StoreLike | None = None,  # not implemented
    storage_options: dict[str, Any] | None = None,  # not implemented
    partial_decompress: bool = False,  # not implemented
    write_empty_chunks: bool = True,  # not implemented
    *,
    zarr_version: ZarrFormat | None = None,  # deprecate in favor of zarr_format
    zarr_format: ZarrFormat | None = None,
    dimension_separator: Literal[".", "/"] | None = None,  # v2 only
    meta_array: Any | None = None,  # not implemented
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
    **kwargs: Any,  # TODO: type kwargs as valid args to save
) -> AsyncArray:
    """Open an array using file-mode-like semantics.

    Parameters
    ----------
    TODO

    Returns
    -------
    AsyncArray
        The opened array.
    """

    store_path = make_store_path(store)
    if path is not None:
        store_path = store_path / path

    try:
        return await AsyncArray.open(store_path)
    except KeyError:
        pass

    warnings.warn("mode is ignored", RuntimeWarning)

    if zarr_version is not None:
        zarr_format = zarr_version
        warnings.warn("zarr_format is deprecated, use zarr_format instead", DeprecationWarning)
    if zarr_format is None:
        zarr_format = 3  # TODO: perhaps this default should be set via config?

    # TODO: finish when Norman's PR goes in
    return await AsyncArray.create(store_path, zarr_format=zarr_format, **kwargs)


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
    _like_args(a, kwargs)
    if isinstance(a, (AsyncArray, Array)):
        kwargs.setdefault("fill_value", a.metadata.fill_value)
    return await open_array(path, **kwargs)


async def zeros(shape: ShapeLike, **kwargs: Any) -> AsyncArray:
    """Create an array, with zero being used as the default value for
    uninitialized portions of the array.
    """
    return await create(shape=shape, fill_value=0, **kwargs)


async def zeros_like(a: ArrayLike, **kwargs: Any) -> AsyncArray:
    """Create an array of zeros like `a`."""
    _like_args(a, kwargs)
    return await zeros(**kwargs)
