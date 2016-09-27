# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from warnings import warn


import numpy as np


from zarr.core import Array
from zarr.storage import DirectoryStore, init_array, contains_array, \
    contains_group, default_compressor, normalize_storage_path
from zarr.codecs import codec_registry
from zarr.errors import err_contains_array, err_contains_group, \
    err_array_not_found


def create(shape, chunks=None, dtype=None, compressor='default',
           fill_value=0, order='C', store=None, synchronizer=None,
           overwrite=False, path=None, chunk_store=None, filters=None,
           cache_metadata=True, **kwargs):
    """Create an array.

    Parameters
    ----------
    shape : int or tuple of ints
        Array shape.
    chunks : int or tuple of ints, optional
        Chunk shape. If not provided, will be guessed from `shape` and `dtype`.
    dtype : string or dtype, optional
        NumPy dtype.
    compressor : Codec, optional
        Primary compressor.
    fill_value : object
        Default value to use for uninitialized portions of the array.
    order : {'C', 'F'}, optional
        Memory layout to be used within each chunk.
    store : MutableMapping or string
        Store or path to directory in file system.
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

    Returns
    -------
    z : zarr.core.Array

    Examples
    --------

    Create an array with default settings::

        >>> import zarr
        >>> z = zarr.create((10000, 10000), chunks=(1000, 1000))
        >>> z
        Array((10000, 10000), float64, chunks=(1000, 1000), order=C)
          nbytes: 762.9M; nbytes_stored: 323; ratio: 2476780.2; initialized: 0/100
          compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
          store: dict

    """  # flake8: noqa

    # handle polymorphic store arg
    store = _handle_store_arg(store)

    # compatibility
    compressor, fill_value = _handle_kwargs(compressor, fill_value, kwargs)

    # initialize array metadata
    init_array(store, shape=shape, chunks=chunks, dtype=dtype,
               compressor=compressor, fill_value=fill_value, order=order, 
               overwrite=overwrite, path=path, chunk_store=chunk_store, 
               filters=filters)

    # instantiate array
    z = Array(store, path=path, chunk_store=chunk_store,
              synchronizer=synchronizer, cache_metadata=cache_metadata)

    return z


def _handle_store_arg(store):
    if store is None:
        return dict()
    elif isinstance(store, str):
        return DirectoryStore(store)
    else:
        return store


def _handle_kwargs(compressor, fill_value, kwargs):

    # to be compatible with h5py, as well as backwards-compatible with Zarr
    # 1.x, accept 'compression' and 'compression_opts' keyword arguments

    if compressor != 'default':
        # 'compressor' overrides 'compression'
        if 'compression' in kwargs:
            warn("'compression' keyword argument overridden by 'compressor'")
        if 'compression_opts' in kwargs:
            warn("ignoring keyword argument 'compression_opts'")

    elif 'compression' in kwargs:
        compression = kwargs.pop('compression')
        compression_opts = kwargs.pop('compression_opts', None)

        if compression in {None, 'none'}:
            compressor = None

        elif compression == 'default':
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
        elif hasattr(compression, 'get_config'):
            compressor = compression

        else:
            raise ValueError('bad value for compression: %r' % compression)

    # handle 'fillvalue'
    if 'fillvalue' in kwargs:
        # to be compatible with h5py, accept 'fillvalue' instead of
        # 'fill_value'
        fill_value = kwargs.pop('fillvalue')

    # ignore other keyword arguments
    for k in kwargs:
        warn('ignoring keyword argument %r' % k)

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
    Array((10000, 10000), float64, chunks=(1000, 1000), order=C)
      nbytes: 762.9M; nbytes_stored: 323; ratio: 2476780.2; initialized: 0/100
      compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
      store: dict
    >>> z[:2, :2]
    array([[ 0.,  0.],
           [ 0.,  0.]])

    """  # flake8: noqa

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
    Array((10000, 10000), float64, chunks=(1000, 1000), order=C)
      nbytes: 762.9M; nbytes_stored: 323; ratio: 2476780.2; initialized: 0/100
      compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
      store: dict
    >>> z[:2, :2]
    array([[ 1.,  1.],
           [ 1.,  1.]])

    """  # flake8: noqa

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
    Array((10000, 10000), float64, chunks=(1000, 1000), order=C)
      nbytes: 762.9M; nbytes_stored: 324; ratio: 2469135.8; initialized: 0/100
      compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
      store: dict
    >>> z[:2, :2]
    array([[ 42.,  42.],
           [ 42.,  42.]])

    """  # flake8: noqa

    return create(shape=shape, fill_value=fill_value, **kwargs)


def _get_shape_chunks(a):
    shape = None
    chunks = None

    if hasattr(a, 'shape') and \
            isinstance(a.shape, tuple):
        shape = a.shape

        if hasattr(a, 'chunks') and \
                isinstance(a.chunks, tuple) and \
                (len(a.chunks) == len(a.shape)):
            chunks = a.chunks

        elif hasattr(a, 'chunklen'):
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
    Array((10000, 10000), int64, chunks=(1000, 1000), order=C)
      nbytes: 762.9M; nbytes_stored: 15.2M; ratio: 50.2; initialized: 100/100
      compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
      store: dict

    """  # flake8: noqa

    # ensure data is array-like
    if not hasattr(data, 'shape') or not hasattr(data, 'dtype'):
        data = np.asanyarray(data)

    # setup dtype
    kw_dtype = kwargs.get('dtype', None)
    if kw_dtype is None:
        kwargs['dtype'] = data.dtype
    else:
        kwargs['dtype'] = kw_dtype

    # setup shape and chunks
    data_shape, data_chunks = _get_shape_chunks(data)
    kwargs['shape'] = data_shape
    kw_chunks = kwargs.get('chunks', None)
    if kw_chunks is None:
        kwargs['chunks'] = data_chunks
    else:
        kwargs['chunks'] = kw_chunks

    # instantiate array
    z = create(**kwargs)

    # fill with data
    z[:] = data

    return z


def open_array(store=None, mode='a', shape=None, chunks=None, dtype=None,
               compressor='default', fill_value=0, order='C',
               synchronizer=None, filters=None, cache_metadata=True,
               path=None, **kwargs):
    """Open array using mode-like semantics.

    Parameters
    ----------
    store : MutableMapping or string
        Store or path to directory in file system.
    mode : {'r', 'r+', 'a', 'w', 'w-'}
        Persistence mode: 'r' means read only (must exist); 'r+' means
        read/write (must exist); 'a' means read/write (create if doesn't
        exist); 'w' means create (overwrite if exists); 'w-' means create
        (fail if exists).
    shape : int or tuple of ints
        Array shape.
    chunks : int or tuple of ints, optional
        Chunk shape. If not provided, will be guessed from `shape` and `dtype`.
    dtype : string or dtype, optional
        NumPy dtype.
    compressor : Codec, optional
        Primary compressor.
    fill_value : object
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
    path : string, optional
        Array path.

    Returns
    -------
    z : zarr.core.Array

    Examples
    --------
    >>> import numpy as np
    >>> import zarr
    >>> z1 = zarr.open_array('example.zarr', mode='w', shape=(10000, 10000),
    ...                      chunks=(1000, 1000), fill_value=0)
    >>> z1[:] = np.arange(100000000).reshape(10000, 10000)
    >>> z1
    Array((10000, 10000), float64, chunks=(1000, 1000), order=C)
      nbytes: 762.9M; nbytes_stored: 23.0M; ratio: 33.2; initialized: 100/100
      compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
      store: DirectoryStore
    >>> z2 = zarr.open_array('example.zarr', mode='r')
    >>> z2
    Array((10000, 10000), float64, chunks=(1000, 1000), order=C)
      nbytes: 762.9M; nbytes_stored: 23.0M; ratio: 33.2; initialized: 100/100
      compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
      store: DirectoryStore
    >>> np.all(z1[:] == z2[:])
    True

    Notes
    -----
    There is no need to close an array. Data are automatically flushed to the
    file system.

    """  # flake8: noqa

    # use same mode semantics as h5py
    # r : read only, must exist
    # r+ : read/write, must exist
    # w : create, delete if exists
    # w- or x : create, fail if exists
    # a : read/write if exists, create otherwise (default)

    # handle polymorphic store arg
    store = _handle_store_arg(store)
    path = normalize_storage_path(path)

    # compatibility
    compressor, fill_value = _handle_kwargs(compressor, fill_value, kwargs)

    # ensure store is initialized

    if mode in ['r', 'r+']:
        if contains_group(store, path=path):
            err_contains_group(path)
        elif not contains_array(store, path=path):
            err_array_not_found(path)

    elif mode == 'w':
        init_array(store, shape=shape, chunks=chunks, dtype=dtype,
                   compressor=compressor, fill_value=fill_value,
                   order=order, filters=filters, overwrite=True, path=path)

    elif mode == 'a':
        if contains_group(store, path=path):
            err_contains_group(path)
        elif not contains_array(store, path=path):
            init_array(store, shape=shape, chunks=chunks, dtype=dtype,
                       compressor=compressor, fill_value=fill_value,
                       order=order, filters=filters, path=path)

    elif mode in ['w-', 'x']:
        if contains_group(store, path=path):
            err_contains_group(path)
        elif contains_array(store, path=path):
            err_contains_array(path)
        else:
            init_array(store, shape=shape, chunks=chunks, dtype=dtype,
                       compressor=compressor, fill_value=fill_value,
                       order=order, filters=filters, path=path)

    # determine read only status
    read_only = mode == 'r'

    # instantiate array
    z = Array(store, read_only=read_only, synchronizer=synchronizer,
              cache_metadata=cache_metadata, path=path)

    return z


# backwards compatibility
open = open_array


def _like_args(a, kwargs):

    shape, chunks = _get_shape_chunks(a)
    if shape is not None:
        kwargs.setdefault('shape', shape)
    if chunks is not None:
        kwargs.setdefault('chunks', chunks)

    if hasattr(a, 'dtype'):
        kwargs.setdefault('dtype', a.dtype)

    if isinstance(a, Array):
        kwargs.setdefault('compressor', a.compressor)
        kwargs.setdefault('order', a.order)
        kwargs.setdefault('filters', a.filters)
    else:
        kwargs.setdefault('compressor', 'default')
        kwargs.setdefault('order', 'C')


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
        kwargs.setdefault('fill_value', a.fill_value)
    return full(**kwargs)


def open_like(a, path, **kwargs):
    """Open a persistent array like `a`."""
    _like_args(a, kwargs)
    if isinstance(a, Array):
        kwargs.setdefault('fill_value', a.fill_value)
    return open_array(path, **kwargs)
