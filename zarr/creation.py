# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import os


import numpy as np


from zarr.core import Array, SynchronizedArray, init_store
from zarr.mappings import DirectoryMap


def create(shape, chunks, dtype=None, compression='blosc',
           compression_opts=None, fill_value=None, store=None,
           synchronizer=None, overwrite=False):
    """Create an array

    Parameters
    ----------
    shape : int or tuple of ints
        Array shape.
    chunks : int or tuple of ints
        Chunk shape.
    dtype : string or dtype, optional
        NumPy dtype.
    compression : string, optional
        Name of primary compression library, e.g., 'blosc', 'zlib'.
    compression_opts : object, optional
        Options to primary compressor. E.g., for blosc, provide a dictionary
        with keys 'cname', 'clevel' and 'shuffle'.
    fill_value : object
        Default value to use for uninitialised portions of the array.
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

    """

    # initialise store
    if store is None:
        store = dict()
    init_store(store, shape=shape, chunks=chunks, dtype=dtype,
               compression=compression, compression_opts=compression_opts,
               fill_value=fill_value, overwrite=overwrite)

    # instantiate array
    if synchronizer is not None:
        z = SynchronizedArray(store, synchronizer)
    else:
        z = Array(store)

    return z


def empty(shape, chunks, dtype=None, compression='blosc',
          compression_opts=None, store=None, synchronizer=None):
    """Create an empty array.

    Parameters
    ----------
    shape : int or tuple of ints
        Array shape.
    chunks : int or tuple of ints
        Chunk shape.
    dtype : string or dtype, optional
        NumPy dtype.
    compression : string, optional
        Name of primary compression library, e.g., 'blosc', 'zlib'.
    compression_opts : object, optional
        Options to primary compressor. E.g., for blosc, provide a dictionary
        with keys 'cname', 'clevel' and 'shuffle'.
    store : MutableMapping, optional
        Array storage. If not provided, a Python dict will be used, meaning
        array data will be stored in memory.
    synchronizer : zarr.sync.ArraySynchronizer, optional
        Array synchronizer.

    Returns
    -------
    z : zarr.core.Array

    """

    return create(shape=shape, chunks=chunks, dtype=dtype,
                  compression=compression, compression_opts=compression_opts,
                  fill_value=None, store=store, synchronizer=synchronizer)


def zeros(shape, chunks, dtype=None, compression='blosc',
          compression_opts=None, store=None, synchronizer=None):
    """Create an array, with zero being used as the default value for
    uninitialised portions of the array.

    Parameters
    ----------
    shape : int or tuple of ints
        Array shape.
    chunks : int or tuple of ints
        Chunk shape.
    dtype : string or dtype, optional
        NumPy dtype.
    compression : string, optional
        Name of primary compression library, e.g., 'blosc', 'zlib'.
    compression_opts : object, optional
        Options to primary compressor. E.g., for blosc, provide a dictionary
        with keys 'cname', 'clevel' and 'shuffle'.
    store : MutableMapping, optional
        Array storage. If not provided, a Python dict will be used, meaning
        array data will be stored in memory.
    synchronizer : zarr.sync.ArraySynchronizer, optional
        Array synchronizer.

    Returns
    -------
    z : zarr.core.Array

    """

    return create(shape=shape, chunks=chunks, dtype=dtype,
                  compression=compression,
                  compression_opts=compression_opts, fill_value=0,
                  store=store, synchronizer=synchronizer)


def ones(shape, chunks, dtype=None, compression='blosc',
         compression_opts=None, store=None, synchronizer=None):
    """Create an array, with one being used as the default value for
    uninitialised portions of the array.

    Parameters
    ----------
    shape : int or tuple of ints
        Array shape.
    chunks : int or tuple of ints
        Chunk shape.
    dtype : string or dtype, optional
        NumPy dtype.
    compression : string, optional
        Name of primary compression library, e.g., 'blosc', 'zlib'.
    compression_opts : object, optional
        Options to primary compressor. E.g., for blosc, provide a dictionary
        with keys 'cname', 'clevel' and 'shuffle'.
    store : MutableMapping, optional
        Array storage. If not provided, a Python dict will be used, meaning
        array data will be stored in memory.
    synchronizer : zarr.sync.ArraySynchronizer, optional
        Array synchronizer.

    Returns
    -------
    z : zarr.core.Array

    """

    return create(shape=shape, chunks=chunks, dtype=dtype,
                  compression=compression, compression_opts=compression_opts,
                  fill_value=1, store=store, synchronizer=synchronizer)


def full(shape, chunks, fill_value, dtype=None, compression='blosc',
         compression_opts=None, store=None, synchronizer=None):
    """Create an array, with `fill_value` being used as the default value for
    uninitialised portions of the array.

    Parameters
    ----------
    shape : int or tuple of ints
        Array shape.
    chunks : int or tuple of ints
        Chunk shape.
    fill_value : object
        Default value to use for uninitialised portions of the array.
    dtype : string or dtype, optional
        NumPy dtype.
    compression : string, optional
        Name of primary compression library, e.g., 'blosc', 'zlib'.
    compression_opts : object, optional
        Options to primary compressor. E.g., for blosc, provide a dictionary
        with keys 'cname', 'clevel' and 'shuffle'.
    store : MutableMapping, optional
        Array storage. If not provided, a Python dict will be used, meaning
        array data will be stored in memory.
    synchronizer : zarr.sync.ArraySynchronizer, optional
        Array synchronizer.

    Returns
    -------
    z : zarr.core.Array

    """

    return create(shape=shape, chunks=chunks, dtype=dtype,
                  compression=compression, compression_opts=compression_opts,
                  fill_value=fill_value, store=store,
                  synchronizer=synchronizer)


def array(data, chunks=None, dtype=None, compression='blosc',
          compression_opts=None, fill_value=None, store=None,
          synchronizer=None):
    """Create an array filled with `data`.

    Parameters
    ----------
    data : array_like
        Data to store.
    chunks : int or tuple of ints
        Chunk shape.
    dtype : string or dtype, optional
        NumPy dtype.
    compression : string, optional
        Name of primary compression library, e.g., 'blosc', 'zlib'.
    compression_opts : object, optional
        Options to primary compressor. E.g., for blosc, provide a dictionary
        with keys 'cname', 'clevel' and 'shuffle'.
    fill_value : object
        Default value to use for uninitialised portions of the array.
    store : MutableMapping, optional
        Array storage. If not provided, a Python dict will be used, meaning
        array data will be stored in memory.
    synchronizer : zarr.sync.ArraySynchronizer, optional
        Array synchronizer.

    Returns
    -------
    z : zarr.core.Array

    """

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
               fill_value=fill_value, store=store, synchronizer=synchronizer)

    # fill with data
    z[:] = data

    return z


# noinspection PyShadowingBuiltins
def open(path, mode='a', shape=None, chunks=None, dtype=None,
         compression='blosc', compression_opts=None, fill_value=0,
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
        Name of primary compression library, e.g., 'blosc', 'zlib'.
    compression_opts : object, optional
        Options to primary compressor. E.g., for blosc, provide a dictionary
        with keys 'cname', 'clevel' and 'shuffle'.
    fill_value : object
        Default value to use for uninitialised portions of the array.
    synchronizer : zarr.sync.ArraySynchronizer, optional
        Array synchronizer.

    Returns
    -------
    z : zarr.core.Array

    """

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
    store = DirectoryMap(path)
    exists = 'meta' in store  # use metadata key as indicator of existence

    # ensure store is initialized
    if mode in ['r', 'r+'] and not exists:
        raise ValueError('array does not exist')
    elif mode in ['w-', 'x'] and exists:
        raise ValueError('array exists')
    elif mode == 'w' or (mode in ['a', 'w-', 'x'] and not exists):
        init_store(store, shape=shape, chunks=chunks, dtype=dtype,
                   compression=compression, compression_opts=compression_opts,
                   fill_value=fill_value, overwrite=True)

    # determine readonly status
    readonly = mode == 'r'

    # handle optional synchronizer
    if synchronizer is not None:
        z = SynchronizedArray(store, synchronizer, readonly=readonly)
    else:
        z = Array(store, readonly=readonly)

    return z


def empty_like(z, shape=None, chunks=None, dtype=None, compression='blosc',
               compression_opts=None, store=None, synchronizer=None):
    """Create an empty array like 'z'."""

    shape = shape if shape is not None else z.shape
    chunks = chunks if chunks is not None else z.chunks
    dtype = dtype if dtype is not None else z.dtype
    compression = compression if compression is not None else z.compression
    compression_opts = compression_opts if compression_opts is not None else \
        z.compression_opts
    return empty(shape, chunks, dtype=dtype, compression=compression,
                 compression_opts=compression_opts,
                 store=store, synchronizer=synchronizer)


def zeros_like(z, shape=None, chunks=None, dtype=None, compression='blosc',
               compression_opts=None, store=None, synchronizer=None):
    """Create an array of zeros like 'z'."""

    shape = shape if shape is not None else z.shape
    chunks = chunks if chunks is not None else z.chunks
    dtype = dtype if dtype is not None else z.dtype
    compression = compression if compression is not None else z.compression
    compression_opts = compression_opts if compression_opts is not None else \
        z.compression_opts
    return zeros(shape, chunks, dtype=dtype, compression=compression,
                 compression_opts=compression_opts,
                 store=store, synchronizer=synchronizer)


def ones_like(z, shape=None, chunks=None, dtype=None, compression='blosc',
              compression_opts=None, store=None, synchronizer=None):
    """Create an array of ones like 'z'."""

    shape = shape if shape is not None else z.shape
    chunks = chunks if chunks is not None else z.chunks
    dtype = dtype if dtype is not None else z.dtype
    compression = compression if compression is not None else z.compression
    compression_opts = compression_opts if compression_opts is not None else \
        z.compression_opts
    return ones(shape, chunks, dtype=dtype, compression=compression,
                compression_opts=compression_opts,
                store=store, synchronizer=synchronizer)


def full_like(z, shape=None, chunks=None, fill_value=None, dtype=None,
              compression='blosc', compression_opts=None,
              store=None, synchronizer=None):
    """Create a filled array like 'z'."""

    shape = shape if shape is not None else z.shape
    chunks = chunks if chunks is not None else z.chunks
    dtype = dtype if dtype is not None else z.dtype
    compression = compression if compression is not None else z.compression
    compression_opts = compression_opts if compression_opts is not None else \
        z.compression_opts
    fill_value = fill_value if fill_value is not None else z.fill_value
    return full(shape, chunks, fill_value, dtype=dtype,
                compression=compression, compression_opts=compression_opts,
                store=store, synchronizer=synchronizer)


def open_like(z, path, mode='a', shape=None, chunks=None, dtype=None,
              compression='blosc', compression_opts=None, fill_value=None,
              synchronizer=None):
    """Open a persistent array like 'z'."""

    shape = shape if shape is not None else z.shape
    chunks = chunks if chunks is not None else z.chunks
    dtype = dtype if dtype is not None else z.dtype
    compression = compression if compression is not None else z.compression
    compression_opts = compression_opts if compression_opts is not None else \
        z.compression_opts
    fill_value = fill_value if fill_value is not None else z.fill_value
    return open(path, mode=mode, shape=shape, chunks=chunks, dtype=dtype,
                compression=compression, compression_opts=compression_opts,
                fill_value=fill_value, synchronizer=synchronizer)
