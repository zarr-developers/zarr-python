# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import os


import numpy as np


from zarr.core import Array
from zarr.sync import SynchronizedArray
from zarr.storage import DirectoryStore, init_store


def create(shape, chunks, dtype=None, compression='default',
           compression_opts=None, fill_value=None, order='C', store=None,
           synchronizer=None, overwrite=False):
    """Create an array.

    Parameters
    ----------
    shape : int or tuple of ints
        Array shape.
    chunks : int or tuple of ints
        Chunk shape.
    dtype : string or dtype, optional
        NumPy dtype.
    compression : string, optional
        Name of primary compression library, e.g., 'blosc', 'zlib', 'bz2',
        'lzma'.
    compression_opts : object, optional
        Options to primary compressor. E.g., for blosc, provide a dictionary
        with keys 'cname', 'clevel' and 'shuffle'.
    fill_value : object
        Default value to use for uninitialised portions of the array.
    order : {'C', 'F'}, optional
        Memory layout to be used within each chunk.
    store : MutableMapping, optional
        Array storage. If not provided, a Python dict will be used, meaning
        array data will be stored in memory.
    synchronizer : zarr.sync.ArraySynchronizer, optional
        Array synchronizer.
    overwrite : bool, optional
        If True, delete all pre-existing data in `store` before creating the
        array.

    Returns
    -------
    z : zarr.core.Array

    Examples
    --------

    Create an array with default settings::

        >>> import zarr
        >>> z = zarr.create((10000, 10000), chunks=(1000, 1000))
        >>> z
        zarr.core.Array((10000, 10000), float64, chunks=(1000, 1000), order=C)
          compression: blosc; compression_opts: {'clevel': 5, 'cname': 'blosclz', 'shuffle': 1}
          nbytes: 762.9M; nbytes_stored: 320; ratio: 2500000.0; initialized: 0/100
          store: builtins.dict

    """  # flake8: noqa

    # initialise store
    if store is None:
        store = dict()
    init_store(store, shape=shape, chunks=chunks, dtype=dtype,
               compression=compression, compression_opts=compression_opts,
               fill_value=fill_value, order=order, overwrite=overwrite)

    # instantiate array
    if synchronizer is not None:
        z = SynchronizedArray(store, synchronizer)
    else:
        z = Array(store)

    return z


def empty(shape, chunks, dtype=None, compression='default',
          compression_opts=None, order='C', store=None, synchronizer=None):
    """Create an empty array.

    For parameter definitions see :func:`zarr.creation.create`.

    Notes
    -----
    The contents of an empty Zarr array are not defined. On attempting to
    retrieve data from an empty Zarr array, any values may be returned,
    and these are not guaranteed to be stable from one access to the next.

    """
    return create(shape=shape, chunks=chunks, dtype=dtype,
                  compression=compression, compression_opts=compression_opts,
                  fill_value=None, order=order, store=store,
                  synchronizer=synchronizer)


def zeros(shape, chunks, dtype=None, compression='default',
          compression_opts=None, order='C', store=None, synchronizer=None):
    """Create an array, with zero being used as the default value for
    uninitialised portions of the array.

    For parameter definitions see :func:`zarr.creation.create`.

    Examples
    --------
    >>> import zarr
    >>> z = zarr.zeros((10000, 10000), chunks=(1000, 1000))
    >>> z
    zarr.core.Array((10000, 10000), float64, chunks=(1000, 1000), order=C)
      compression: blosc; compression_opts: {'clevel': 5, 'cname': 'blosclz', 'shuffle': 1}
      nbytes: 762.9M; nbytes_stored: 317; ratio: 2523659.3; initialized: 0/100
      store: builtins.dict
    >>> z[:2, :2]
    array([[ 0.,  0.],
           [ 0.,  0.]])

    """  # flake8: noqa

    return create(shape=shape, chunks=chunks, dtype=dtype,
                  compression=compression,
                  compression_opts=compression_opts, fill_value=0, order=order,
                  store=store, synchronizer=synchronizer)


def ones(shape, chunks, dtype=None, compression='default',
         compression_opts=None, order='C', store=None, synchronizer=None):
    """Create an array, with one being used as the default value for
    uninitialised portions of the array.

    For parameter definitions see :func:`zarr.creation.create`.

    Examples
    --------
    >>> import zarr
    >>> z = zarr.ones((10000, 10000), chunks=(1000, 1000))
    >>> z
    zarr.core.Array((10000, 10000), float64, chunks=(1000, 1000), order=C)
      compression: blosc; compression_opts: {'clevel': 5, 'cname': 'blosclz', 'shuffle': 1}
      nbytes: 762.9M; nbytes_stored: 317; ratio: 2523659.3; initialized: 0/100
      store: builtins.dict
    >>> z[:2, :2]
    array([[ 1.,  1.],
           [ 1.,  1.]])

    """  # flake8: noqa

    return create(shape=shape, chunks=chunks, dtype=dtype,
                  compression=compression, compression_opts=compression_opts,
                  fill_value=1, order=order, store=store,
                  synchronizer=synchronizer)


def full(shape, chunks, fill_value, dtype=None, compression='default',
         compression_opts=None, order='C', store=None, synchronizer=None):
    """Create an array, with `fill_value` being used as the default value for
    uninitialised portions of the array.

    For parameter definitions see :func:`zarr.creation.create`.

    Examples
    --------
    >>> import zarr
    >>> z = zarr.full((10000, 10000), chunks=(1000, 1000), fill_value=42)
    >>> z
    zarr.core.Array((10000, 10000), float64, chunks=(1000, 1000), order=C)
      compression: blosc; compression_opts: {'clevel': 5, 'cname': 'blosclz', 'shuffle': 1}
      nbytes: 762.9M; nbytes_stored: 318; ratio: 2515723.3; initialized: 0/100
      store: builtins.dict
    >>> z[:2, :2]
    array([[ 42.,  42.],
           [ 42.,  42.]])

    """  # flake8: noqa

    return create(shape=shape, chunks=chunks, dtype=dtype,
                  compression=compression, compression_opts=compression_opts,
                  fill_value=fill_value, order=order, store=store,
                  synchronizer=synchronizer)


def array(data, chunks=None, dtype=None, compression='default',
          compression_opts=None, fill_value=None, order='C', store=None,
          synchronizer=None):
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
    zarr.core.Array((10000, 10000), int64, chunks=(1000, 1000), order=C)
      compression: blosc; compression_opts: {'clevel': 5, 'cname': 'blosclz', 'shuffle': 1}
      nbytes: 762.9M; nbytes_stored: 16.8M; ratio: 45.5; initialized: 100/100
      store: builtins.dict

    """  # flake8: noqa

    # ensure data is array-like
    if not hasattr(data, 'shape') or not hasattr(data, 'dtype'):
        data = np.asanyarray(data)

    # setup dtype
    if dtype is None:
        dtype = data.dtype

    # setup shape
    shape = data.shape

    # setup chunks
    if chunks is None:
        # try to use same chunks as data
        if hasattr(data, 'chunklen'):
            # bcolz carray
            chunks = (data.chunklen,) + shape[1:]
        elif hasattr(data, 'chunks') and len(data.chunks) == len(data.shape):
            # h5py dataset or zarr array
            chunks = data.chunks
        else:
            raise ValueError('chunks must be specified')

    # instantiate array
    z = create(shape=shape, chunks=chunks, dtype=dtype,
               compression=compression, compression_opts=compression_opts,
               fill_value=fill_value, order=order, store=store,
               synchronizer=synchronizer)

    # fill with data
    z[:] = data

    return z


# noinspection PyShadowingBuiltins
def open(path, mode='a', shape=None, chunks=None, dtype=None,
         compression='default', compression_opts=None, fill_value=0, order='C',
         synchronizer=None):
    """Open an array stored in a directory on the file system.

    Parameters
    ----------
    path : string
        Path to directory in which to store the array.
    mode : {'r', 'r+', 'a', 'w', 'w-'}
        Persistence mode: 'r' means readonly (must exist); 'r+' means
        read/write (must exist); 'a' means read/write (create if doesn't
        exist); 'w' means create (overwrite if exists); 'w-' means create
        (fail if exists).
    shape : int or tuple of ints
        Array shape.
    chunks : int or tuple of ints
        Chunk shape.
    dtype : string or dtype, optional
        NumPy dtype.
    compression : string, optional
        Name of primary compression library, e.g., 'blosc', 'zlib', 'bz2',
        'lzma'.
    compression_opts : object, optional
        Options to primary compressor. E.g., for blosc, provide a dictionary
        with keys 'cname', 'clevel' and 'shuffle'.
    fill_value : object
        Default value to use for uninitialised portions of the array.
    order : {'C', 'F'}, optional
        Memory layout to be used within each chunk.
    synchronizer : zarr.sync.ArraySynchronizer, optional
        Array synchronizer.

    Returns
    -------
    z : zarr.core.Array

    Examples
    --------
    >>> import numpy as np
    >>> import zarr
    >>> z1 = zarr.open('example.zarr', mode='w', shape=(10000, 10000),
    ...                chunks=(1000, 1000), fill_value=0)
    >>> z1[:] = np.arange(100000000).reshape(10000, 10000)
    >>> z1
    zarr.core.Array((10000, 10000), float64, chunks=(1000, 1000), order=C)
      compression: blosc; compression_opts: {'clevel': 5, 'cname': 'blosclz', 'shuffle': 1}
      nbytes: 762.9M; nbytes_stored: 24.6M; ratio: 31.0; initialized: 100/100
      store: zarr.storage.DirectoryStore
    >>> z2 = zarr.open('example.zarr', mode='r')
    >>> z2
    zarr.core.Array((10000, 10000), float64, chunks=(1000, 1000), order=C)
      compression: blosc; compression_opts: {'clevel': 5, 'cname': 'blosclz', 'shuffle': 1}
      nbytes: 762.9M; nbytes_stored: 24.6M; ratio: 31.0; initialized: 100/100
      store: zarr.storage.DirectoryStore
    >>> np.all(z1[:] == z2[:])
    True

    Notes
    -----
    There is no need to close an array. Data are automatically flushed to the
    file system.

    """  # flake8: noqa

    # use same mode semantics as h5py, although N.B., here `path` is a
    # directory:
    # r : readonly, must exist
    # r+ : read/write, must exist
    # w : create, delete if exists
    # w- or x : create, fail if exists
    # a : read/write if exists, create otherwise (default)

    # ensure directory exists
    if not os.path.exists(path):
        if mode in ['w', 'w-', 'x', 'a']:
            os.makedirs(path)
        elif mode in ['r', 'r+']:
            raise ValueError('path does not exist: %r' % path)

    # setup store
    store = DirectoryStore(path)
    exists = 'meta' in store  # use metadata key as indicator of existence

    # ensure store is initialized
    if mode in ['r', 'r+'] and not exists:
        raise ValueError('array does not exist')
    elif mode in ['w-', 'x'] and exists:
        raise ValueError('array exists')
    elif mode == 'w' or (mode in ['a', 'w-', 'x'] and not exists):
        init_store(store, shape=shape, chunks=chunks, dtype=dtype,
                   compression=compression, compression_opts=compression_opts,
                   fill_value=fill_value, order=order, overwrite=True)

    # determine readonly status
    readonly = mode == 'r'

    # handle optional synchronizer
    if synchronizer is not None:
        z = SynchronizedArray(store, synchronizer, readonly=readonly)
    else:
        z = Array(store, readonly=readonly)

    return z


def _like_args(a, shape, chunks, dtype, compression, compression_opts, order):
    if shape is None:
        shape = a.shape
    if chunks is None:
        try:
            chunks = a.chunks
        except AttributeError:
            raise ValueError('chunks must be specified')
    if dtype is None:
        dtype = a.dtype
    if compression is None:
        try:
            compression = a.compression
        except AttributeError:
            compression = 'blosc'
    if compression_opts is None:
        try:
            compression_opts = a.compression_opts
        except AttributeError:
            pass
    if order is None:
        try:
            order = a.order
        except AttributeError:
            order = 'C'
    return shape, chunks, dtype, compression, compression_opts, order


def empty_like(a, shape=None, chunks=None, dtype=None, compression=None,
               compression_opts=None, order=None, store=None,
               synchronizer=None):
    """Create an empty array like `a`."""
    shape, chunks, dtype, compression, compression_opts, order = \
        _like_args(a, shape, chunks, dtype, compression, compression_opts,
                   order)
    return empty(shape, chunks, dtype=dtype, compression=compression,
                 compression_opts=compression_opts, order=order,
                 store=store, synchronizer=synchronizer)


def zeros_like(a, shape=None, chunks=None, dtype=None, compression=None,
               compression_opts=None, order=None, store=None,
               synchronizer=None):
    """Create an array of zeros like `a`."""
    shape, chunks, dtype, compression, compression_opts, order = \
        _like_args(a, shape, chunks, dtype, compression, compression_opts,
                   order)
    return zeros(shape, chunks, dtype=dtype, compression=compression,
                 compression_opts=compression_opts, order=order,
                 store=store, synchronizer=synchronizer)


def ones_like(a, shape=None, chunks=None, dtype=None, compression=None,
              compression_opts=None, order=None, store=None,
              synchronizer=None):
    """Create an array of ones like `a`."""
    shape, chunks, dtype, compression, compression_opts, order = \
        _like_args(a, shape, chunks, dtype, compression, compression_opts,
                   order)
    return ones(shape, chunks, dtype=dtype, compression=compression,
                compression_opts=compression_opts, order=order,
                store=store, synchronizer=synchronizer)


def full_like(a, shape=None, chunks=None, fill_value=None, dtype=None,
              compression=None, compression_opts=None, order=None,
              store=None, synchronizer=None):
    """Create a filled array like `a`."""
    shape, chunks, dtype, compression, compression_opts, order = \
        _like_args(a, shape, chunks, dtype, compression, compression_opts,
                   order)
    if fill_value is None:
        try:
            fill_value = a.fill_value
        except AttributeError:
            raise ValueError('fill_value must be specified')
    return full(shape, chunks, fill_value, dtype=dtype,
                compression=compression, compression_opts=compression_opts,
                order=order, store=store, synchronizer=synchronizer)


def open_like(a, path, mode='a', shape=None, chunks=None, dtype=None,
              compression=None, compression_opts=None, fill_value=None,
              order=None, synchronizer=None):
    """Open a persistent array like `a`."""
    shape, chunks, dtype, compression, compression_opts, order = \
        _like_args(a, shape, chunks, dtype, compression, compression_opts,
                   order)
    if fill_value is None:
        try:
            fill_value = a.fill_value
        except AttributeError:
            # leave empty
            pass
    return open(path, mode=mode, shape=shape, chunks=chunks, dtype=dtype,
                compression=compression, compression_opts=compression_opts,
                fill_value=fill_value, order=order, synchronizer=synchronizer)
