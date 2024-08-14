from collections.abc import MutableMapping
from typing import Optional, Tuple, Union, Sequence
from warnings import warn

import numpy as np
import numpy.typing as npt
from numcodecs.abc import Codec
from numcodecs.registry import codec_registry

from zarr._storage.store import DEFAULT_ZARR_VERSION
from zarr.core import Array
from zarr.errors import (
    ArrayNotFoundError,
    ContainsArrayError,
    ContainsGroupError,
)
from zarr.storage import (
    contains_array,
    contains_group,
    default_compressor,
    init_array,
    normalize_storage_path,
    normalize_store_arg,
)
from zarr._storage.store import StorageTransformer
from zarr.sync import Synchronizer
from zarr.types import ZARR_VERSION, DIMENSION_SEPARATOR, MEMORY_ORDER, MetaArray, PathLike
from zarr.util import normalize_dimension_separator


def create(
    shape: Union[int, Tuple[int, ...]],
    chunks: Union[int, Tuple[int, ...], bool] = True,
    dtype: Optional[npt.DTypeLike] = None,
    compressor="default",
    fill_value: Optional[int] = 0,
    order: MEMORY_ORDER = "C",
    store: Optional[Union[str, MutableMapping]] = None,
    synchronizer: Optional[Synchronizer] = None,
    overwrite: bool = False,
    path: Optional[PathLike] = None,
    chunk_store: Optional[MutableMapping] = None,
    filters: Optional[Sequence[Codec]] = None,
    cache_metadata: bool = True,
    cache_attrs: bool = True,
    read_only: bool = False,
    object_codec: Optional[Codec] = None,
    dimension_separator: Optional[DIMENSION_SEPARATOR] = None,
    write_empty_chunks: bool = True,
    *,
    zarr_version: Optional[ZARR_VERSION] = None,
    meta_array: Optional[MetaArray] = None,
    storage_transformers: Sequence[StorageTransformer] = (),
    **kwargs,
):
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
    store : MutableMapping or string
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

    storage_transformers : sequence of StorageTransformers, optional
        Setting storage transformers, changes the storage structure and behaviour
        of data coming from the underlying store. The transformers are applied in the
        order of the given sequence. Supplying an empty sequence is the same as omitting
        the argument or setting it to None. May only be set when using zarr_version 3.

        .. versionadded:: 2.13

    zarr_version : {None, 2, 3}, optional
        The zarr protocol version of the created array. If None, it will be
        inferred from ``store`` or ``chunk_store`` if they are provided,
        otherwise defaulting to 2.

        .. versionadded:: 2.12

    meta_array : array-like, optional
        An array instance to use for determining arrays to create and return
        to users. Use `numpy.empty(())` by default.

        .. versionadded:: 2.13

    Returns
    -------
    z : zarr.core.Array

    Examples
    --------

    Create an array with default settings::

        >>> import zarr
        >>> z = zarr.create((10000, 10000), chunks=(1000, 1000))
        >>> z
        <zarr.core.Array (10000, 10000) float64>

    Create an array with different some different configuration options::

        >>> from numcodecs import Blosc
        >>> compressor = Blosc(cname='zstd', clevel=1, shuffle=Blosc.BITSHUFFLE)
        >>> z = zarr.create((10000, 10000), chunks=(1000, 1000), dtype='i1', order='F',
        ...                 compressor=compressor)
        >>> z
        <zarr.core.Array (10000, 10000) int8>

    To create an array with object dtype requires a filter that can handle Python object
    encoding, e.g., `MsgPack` or `Pickle` from `numcodecs`::

        >>> from numcodecs import MsgPack
        >>> z = zarr.create((10000, 10000), chunks=(1000, 1000), dtype=object,
        ...                 object_codec=MsgPack())
        >>> z
        <zarr.core.Array (10000, 10000) object>

    Example with some filters, and also storing chunks separately from metadata::

        >>> from numcodecs import Quantize, Adler32
        >>> store, chunk_store = dict(), dict()
        >>> z = zarr.create((10000, 10000), chunks=(1000, 1000), dtype='f8',
        ...                 filters=[Quantize(digits=2, dtype='f8'), Adler32()],
        ...                 store=store, chunk_store=chunk_store)
        >>> z
        <zarr.core.Array (10000, 10000) float64>

    """
    if zarr_version is None and store is None:
        zarr_version = getattr(chunk_store, "_store_version", DEFAULT_ZARR_VERSION)

    # handle polymorphic store arg
    store = normalize_store_arg(store, zarr_version=zarr_version, mode="w")
    zarr_version = getattr(store, "_store_version", DEFAULT_ZARR_VERSION)

    # API compatibility with h5py
    compressor, fill_value = _kwargs_compat(compressor, fill_value, kwargs)

    # optional array metadata
    if dimension_separator is None:
        dimension_separator = getattr(store, "_dimension_separator", None)
    else:
        store_separator = getattr(store, "_dimension_separator", None)
        if store_separator not in (None, dimension_separator):
            raise ValueError(
                f"Specified dimension_separator: {dimension_separator}"
                f"conflicts with store's separator: "
                f"{store_separator}"
            )
    dimension_separator = normalize_dimension_separator(dimension_separator)

    if zarr_version > 2 and path is None:
        path = "/"

    # initialize array metadata
    init_array(
        store,
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        compressor=compressor,
        fill_value=fill_value,
        order=order,
        overwrite=overwrite,
        path=path,
        chunk_store=chunk_store,
        filters=filters,
        object_codec=object_codec,
        dimension_separator=dimension_separator,
        storage_transformers=storage_transformers,
    )

    # instantiate array
    z = Array(
        store,
        path=path,
        chunk_store=chunk_store,
        synchronizer=synchronizer,
        cache_metadata=cache_metadata,
        cache_attrs=cache_attrs,
        read_only=read_only,
        write_empty_chunks=write_empty_chunks,
        meta_array=meta_array,
    )

    return z


def _kwargs_compat(compressor, fill_value, kwargs):
    # to be compatible with h5py, as well as backwards-compatible with Zarr
    # 1.x, accept 'compression' and 'compression_opts' keyword arguments

    if compressor != "default":
        # 'compressor' overrides 'compression'
        if "compression" in kwargs:
            warn(
                "'compression' keyword argument overridden by 'compressor'",
                stacklevel=3,
            )
            del kwargs["compression"]
        if "compression_opts" in kwargs:
            warn(
                "'compression_opts' keyword argument overridden by 'compressor'",
                stacklevel=3,
            )
            del kwargs["compression_opts"]

    elif "compression" in kwargs:
        compression = kwargs.pop("compression")
        compression_opts = kwargs.pop("compression_opts", None)

        if compression is None or compression == "none":
            compressor = None

        elif compression == "default":
            compressor = default_compressor

        elif isinstance(compression, str):
            codec_cls = codec_registry[compression]

            # handle compression_opts
            if isinstance(compression_opts, dict):
                compressor = codec_cls(**compression_opts)
            elif isinstance(compression_opts, (list, tuple)):
                compressor = codec_cls(*compression_opts)
            elif compression_opts is None:
                compressor = codec_cls()
            else:
                # assume single argument, e.g., int
                compressor = codec_cls(compression_opts)

        # be lenient here if user gives compressor as 'compression'
        elif hasattr(compression, "get_config"):
            compressor = compression

        else:
            raise ValueError(f"bad value for compression: {compression!r}")

    # handle 'fillvalue'
    if "fillvalue" in kwargs:
        # to be compatible with h5py, accept 'fillvalue' instead of
        # 'fill_value'
        fill_value = kwargs.pop("fillvalue")

    # ignore other keyword arguments
    for k in kwargs:
        warn(f"ignoring keyword argument {k!r}", stacklevel=2)

    return compressor, fill_value


def empty(shape, **kwargs):
    """Create an empty array.

    For parameter definitions see :func:`zarr.creation.create`.

    Notes
    -----
    The contents of an empty Zarr array are not defined. On attempting to
    retrieve data from an empty Zarr array, any values may be returned,
    and these are not guaranteed to be stable from one access to the next.

    """
    return create(shape=shape, fill_value=None, **kwargs)


def zeros(shape, **kwargs):
    """Create an array, with zero being used as the default value for
    uninitialized portions of the array.

    For parameter definitions see :func:`zarr.creation.create`.

    Examples
    --------
    >>> import zarr
    >>> z = zarr.zeros((10000, 10000), chunks=(1000, 1000))
    >>> z
    <zarr.core.Array (10000, 10000) float64>
    >>> z[:2, :2]
    array([[0., 0.],
           [0., 0.]])

    """

    return create(shape=shape, fill_value=0, **kwargs)


def ones(shape, **kwargs):
    """Create an array, with one being used as the default value for
    uninitialized portions of the array.

    For parameter definitions see :func:`zarr.creation.create`.

    Examples
    --------
    >>> import zarr
    >>> z = zarr.ones((10000, 10000), chunks=(1000, 1000))
    >>> z
    <zarr.core.Array (10000, 10000) float64>
    >>> z[:2, :2]
    array([[1., 1.],
           [1., 1.]])

    """

    return create(shape=shape, fill_value=1, **kwargs)


def full(shape, fill_value, **kwargs):
    """Create an array, with `fill_value` being used as the default value for
    uninitialized portions of the array.

    For parameter definitions see :func:`zarr.creation.create`.

    Examples
    --------
    >>> import zarr
    >>> z = zarr.full((10000, 10000), chunks=(1000, 1000), fill_value=42)
    >>> z
    <zarr.core.Array (10000, 10000) float64>
    >>> z[:2, :2]
    array([[42., 42.],
           [42., 42.]])

    """

    return create(shape=shape, fill_value=fill_value, **kwargs)


def _get_shape_chunks(a):
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


def array(data, **kwargs):
    """Create an array filled with `data`.

    The `data` argument should be a NumPy array or array-like object. For
    other parameter definitions see :func:`zarr.creation.create`.

    Examples
    --------
    >>> import numpy as np
    >>> import zarr
    >>> a = np.arange(100000000).reshape(10000, 10000)
    >>> z = zarr.array(a, chunks=(1000, 1000))
    >>> z
    <zarr.core.Array (10000, 10000) int64>

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
    read_only = kwargs.pop("read_only", False)

    # instantiate array
    z = create(**kwargs)

    # fill with data
    z[...] = data

    # set read_only property afterwards
    z.read_only = read_only

    return z


def open_array(
    store=None,
    mode="a",
    shape=None,
    chunks=True,
    dtype=None,
    compressor="default",
    fill_value=0,
    order="C",
    synchronizer=None,
    filters=None,
    cache_metadata=True,
    cache_attrs=True,
    path=None,
    object_codec=None,
    chunk_store=None,
    storage_options=None,
    partial_decompress=False,
    write_empty_chunks=True,
    *,
    zarr_version=None,
    dimension_separator: Optional[DIMENSION_SEPARATOR] = None,
    meta_array=None,
    **kwargs,
):
    """Open an array using file-mode-like semantics.

    Parameters
    ----------
    store : MutableMapping or string, optional
        Store or path to directory in file system or name of zip file.
    mode : {'r', 'r+', 'a', 'w', 'w-'}, optional
        Persistence mode: 'r' means read only (must exist); 'r+' means
        read/write (must exist); 'a' means read/write (create if doesn't
        exist); 'w' means create (overwrite if exists); 'w-' means create
        (fail if exists).
    shape : int or tuple of ints, optional
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
    fill_value : object, optional
        Default value to use for uninitialized portions of the array.
    order : {'C', 'F'}, optional
        Memory layout to be used within each chunk.
    synchronizer : object, optional
        Array synchronizer.
    filters : sequence, optional
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
    path : string, optional
        Array path within store.
    object_codec : Codec, optional
        A codec to encode object arrays, only needed if dtype=object.
    chunk_store : MutableMapping or string, optional
        Store or path to directory in file system or name of zip file.
    storage_options : dict
        If using an fsspec URL to create the store, these will be passed to
        the backend implementation. Ignored otherwise.
    partial_decompress : bool, optional
        If True and while the chunk_store is a FSStore and the compression used
        is Blosc, when getting data from the array chunks will be partially
        read and decompressed when possible.
    write_empty_chunks : bool, optional
        If True (default), all chunks will be stored regardless of their
        contents. If False, each chunk is compared to the array's fill value
        prior to storing. If a chunk is uniformly equal to the fill value, then
        that chunk is not be stored, and the store entry for that chunk's key
        is deleted. This setting enables sparser storage, as only chunks with
        non-fill-value data are stored, at the expense of overhead associated
        with checking the data of each chunk.

        .. versionadded:: 2.11

    zarr_version : {None, 2, 3}, optional
        The zarr protocol version of the array to be opened. If None, it will
        be inferred from ``store`` or ``chunk_store`` if they are provided,
        otherwise defaulting to 2.
    dimension_separator : {None, '.', '/'}, optional
        Can be used to specify whether the array is in a flat ('.') or nested
        ('/') format. If None, the appropriate value will be read from `store`
        when present. Otherwise, defaults to '.' when ``zarr_version == 2``
        and `/` otherwise.
    meta_array : array-like, optional
        An array instance to use for determining arrays to create and return
        to users. Use `numpy.empty(())` by default.

        .. versionadded:: 2.15

    Returns
    -------
    z : zarr.core.Array

    Examples
    --------
    >>> import numpy as np
    >>> import zarr
    >>> z1 = zarr.open_array('data/example.zarr', mode='w', shape=(10000, 10000),
    ...                      chunks=(1000, 1000), fill_value=0)
    >>> z1[:] = np.arange(100000000).reshape(10000, 10000)
    >>> z1
    <zarr.core.Array (10000, 10000) float64>
    >>> z2 = zarr.open_array('data/example.zarr', mode='r')
    >>> z2
    <zarr.core.Array (10000, 10000) float64 read-only>
    >>> np.all(z1[:] == z2[:])
    np.True_

    Notes
    -----
    There is no need to close an array. Data are automatically flushed to the
    file system.

    """

    # use same mode semantics as h5py
    # r : read only, must exist
    # r+ : read/write, must exist
    # w : create, delete if exists
    # w- or x : create, fail if exists
    # a : read/write if exists, create otherwise (default)

    if zarr_version is None and store is None:
        zarr_version = getattr(chunk_store, "_store_version", DEFAULT_ZARR_VERSION)

    # handle polymorphic store arg
    store = normalize_store_arg(
        store, storage_options=storage_options, mode=mode, zarr_version=zarr_version
    )
    zarr_version = getattr(store, "_store_version", DEFAULT_ZARR_VERSION)
    if chunk_store is not None:
        chunk_store = normalize_store_arg(
            chunk_store, storage_options=storage_options, mode=mode, zarr_version=zarr_version
        )

    # respect the dimension separator specified in a store, if present
    if dimension_separator is None:
        if hasattr(store, "_dimension_separator"):
            dimension_separator = store._dimension_separator
        else:
            dimension_separator = "." if zarr_version == 2 else "/"

    if zarr_version == 3 and path is None:
        path = "array"  # TODO: raise ValueError instead?

    path = normalize_storage_path(path)

    # API compatibility with h5py
    compressor, fill_value = _kwargs_compat(compressor, fill_value, kwargs)

    # ensure fill_value of correct type
    if fill_value is not None:
        fill_value = np.array(fill_value, dtype=dtype)[()]

    # ensure store is initialized

    if mode in ["r", "r+"]:
        if not contains_array(store, path=path):
            if contains_group(store, path=path):
                raise ContainsGroupError(path)
            raise ArrayNotFoundError(path)

    elif mode == "w":
        init_array(
            store,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            compressor=compressor,
            fill_value=fill_value,
            order=order,
            filters=filters,
            overwrite=True,
            path=path,
            object_codec=object_codec,
            chunk_store=chunk_store,
            dimension_separator=dimension_separator,
        )

    elif mode == "a":
        if not contains_array(store, path=path):
            if contains_group(store, path=path):
                raise ContainsGroupError(path)
            init_array(
                store,
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                compressor=compressor,
                fill_value=fill_value,
                order=order,
                filters=filters,
                path=path,
                object_codec=object_codec,
                chunk_store=chunk_store,
                dimension_separator=dimension_separator,
            )

    elif mode in ["w-", "x"]:
        if contains_group(store, path=path):
            raise ContainsGroupError(path)
        elif contains_array(store, path=path):
            raise ContainsArrayError(path)
        else:
            init_array(
                store,
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                compressor=compressor,
                fill_value=fill_value,
                order=order,
                filters=filters,
                path=path,
                object_codec=object_codec,
                chunk_store=chunk_store,
                dimension_separator=dimension_separator,
            )

    # determine read only status
    read_only = mode == "r"

    # instantiate array
    z = Array(
        store,
        read_only=read_only,
        synchronizer=synchronizer,
        cache_metadata=cache_metadata,
        cache_attrs=cache_attrs,
        path=path,
        chunk_store=chunk_store,
        write_empty_chunks=write_empty_chunks,
        meta_array=meta_array,
    )

    return z


def _like_args(a, kwargs):
    shape, chunks = _get_shape_chunks(a)
    if shape is not None:
        kwargs.setdefault("shape", shape)
    if chunks is not None:
        kwargs.setdefault("chunks", chunks)

    if hasattr(a, "dtype"):
        kwargs.setdefault("dtype", a.dtype)

    if isinstance(a, Array):
        kwargs.setdefault("compressor", a.compressor)
        kwargs.setdefault("order", a.order)
        kwargs.setdefault("filters", a.filters)
        kwargs.setdefault("zarr_version", a._version)
    else:
        kwargs.setdefault("compressor", "default")
        kwargs.setdefault("order", "C")


def empty_like(a, **kwargs):
    """Create an empty array like `a`."""
    _like_args(a, kwargs)
    return empty(**kwargs)


def zeros_like(a, **kwargs):
    """Create an array of zeros like `a`."""
    _like_args(a, kwargs)
    return zeros(**kwargs)


def ones_like(a, **kwargs):
    """Create an array of ones like `a`."""
    _like_args(a, kwargs)
    return ones(**kwargs)


def full_like(a, **kwargs):
    """Create a filled array like `a`."""
    _like_args(a, kwargs)
    if isinstance(a, Array):
        kwargs.setdefault("fill_value", a.fill_value)
    return full(**kwargs)


def open_like(a, path, **kwargs):
    """Open a persistent array like `a`."""
    _like_args(a, kwargs)
    if isinstance(a, Array):
        kwargs.setdefault("fill_value", a.fill_value)
    return open_array(path, **kwargs)
