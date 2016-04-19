# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np


from zarr.array import Array, SynchronizedArray
from zarr.store.memory import MemoryStore
from zarr.store.directory import DirectoryStore


def _create(shape, chunks, dtype, cname, clevel, shuffle, fill_value,
            synchronizer):

    # setup memory store
    store = MemoryStore(shape=shape, chunks=chunks, dtype=dtype, cname=cname,
                        clevel=clevel, shuffle=shuffle, fill_value=fill_value)

    # handle optional synchronizer
    if synchronizer is not None:
        z = SynchronizedArray(store, synchronizer)
    else:
        z = Array(store)

    return z


def empty(shape, chunks, dtype=None, cname=None, clevel=None, shuffle=None,
          synchronizer=None):
    """Create an empty array.

    Parameters
    ----------
    shape : int or tuple of ints
        Array shape.
    chunks : int or tuple of ints
        Chunk shape.
    dtype : string or dtype, optional
        NumPy dtype.
    cname : string, optional
        Name of compression library to use, e.g., 'blosclz', 'lz4', 'zlib',
        'snappy'.
    clevel : int, optional
        Compression level, 0 means no compression.
    shuffle : int, optional
        Shuffle filter, 0 means no shuffle, 1 means byte shuffle, 2 means
        bit shuffle.
    synchronizer : zarr.sync.ArraySynchronizer, optional
        Array synchronizer.

    Returns
    -------
    z : zarr.array.Array

    """

    return _create(shape=shape, chunks=chunks, dtype=dtype, cname=cname,
                   clevel=clevel, shuffle=shuffle, fill_value=None,
                   synchronizer=synchronizer)


def zeros(shape, chunks, dtype=None, cname=None, clevel=None, shuffle=None,
          synchronizer=None):
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
    cname : string, optional
        Name of compression library to use, e.g., 'blosclz', 'lz4', 'zlib',
        'snappy'.
    clevel : int, optional
        Compression level, 0 means no compression.
    shuffle : int, optional
        Shuffle filter, 0 means no shuffle, 1 means byte shuffle, 2 means
        bit shuffle.
    synchronizer : zarr.sync.ArraySynchronizer, optional
        Array synchronizer.

    Returns
    -------
    z : zarr.array.Array

    """

    return _create(shape=shape, chunks=chunks, dtype=dtype, cname=cname,
                   clevel=clevel, shuffle=shuffle, fill_value=0,
                   synchronizer=synchronizer)


def ones(shape, chunks, dtype=None, cname=None, clevel=None, shuffle=None,
         synchronizer=None):
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
    cname : string, optional
        Name of compression library to use, e.g., 'blosclz', 'lz4', 'zlib',
        'snappy'.
    clevel : int, optional
        Compression level, 0 means no compression.
    shuffle : int, optional
        Shuffle filter, 0 means no shuffle, 1 means byte shuffle, 2 means
        bit shuffle.
    synchronizer : zarr.sync.ArraySynchronizer, optional
        Array synchronizer.

    Returns
    -------
    z : zarr.array.Array

    """

    return _create(shape=shape, chunks=chunks, dtype=dtype, cname=cname,
                   clevel=clevel, shuffle=shuffle, fill_value=1,
                   synchronizer=synchronizer)


def full(shape, chunks, fill_value, dtype=None, cname=None, clevel=None,
         shuffle=None, synchronizer=None):
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
    cname : string, optional
        Name of compression library to use, e.g., 'blosclz', 'lz4', 'zlib',
        'snappy'.
    clevel : int, optional
        Compression level, 0 means no compression.
    shuffle : int, optional
        Shuffle filter, 0 means no shuffle, 1 means byte shuffle, 2 means
        bit shuffle.
    synchronizer : zarr.sync.ArraySynchronizer, optional
        Array synchronizer.

    Returns
    -------
    z : zarr.array.Array

    """

    return _create(shape=shape, chunks=chunks, dtype=dtype, cname=cname,
                   clevel=clevel, shuffle=shuffle, fill_value=fill_value,
                   synchronizer=synchronizer)


def array(data, chunks=None, dtype=None, cname=None, clevel=None,
          shuffle=None, fill_value=None, synchronizer=None):
    """Create an array filled with `data`.

    Parameters
    ----------
    data : array_like
        Data to store.
    chunks : int or tuple of ints
        Chunk shape.
    dtype : string or dtype, optional
        NumPy dtype.
    cname : string, optional
        Name of compression library to use, e.g., 'blosclz', 'lz4', 'zlib',
        'snappy'.
    clevel : int, optional
        Compression level, 0 means no compression.
    shuffle : int, optional
        Shuffle filter, 0 means no shuffle, 1 means byte shuffle, 2 means
        bit shuffle.
    fill_value : object
        Default value to use for uninitialised portions of the array.
    synchronizer : zarr.sync.ArraySynchronizer, optional
        Array synchronizer.

    Returns
    -------
    z : zarr.array.Array

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
    z = _create(shape=shape, chunks=chunks, dtype=dtype, cname=cname,
                clevel=clevel, shuffle=shuffle, fill_value=fill_value,
                synchronizer=synchronizer)

    # fill with data
    z[:] = data

    return z


# noinspection PyShadowingBuiltins
def open(path, mode='a', shape=None, chunks=None, dtype=None, cname=None,
         clevel=None, shuffle=None, fill_value=0, synchronizer=None):
    """Open a persistent array.

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
    cname : string, optional
        Name of compression library to use, e.g., 'blosclz', 'lz4', 'zlib',
        'snappy'.
    clevel : int, optional
        Compression level, 0 means no compression.
    shuffle : int, optional
        Shuffle filter, 0 means no shuffle, 1 means byte shuffle, 2 means
        bit shuffle.
    fill_value : object
        Default value to use for uninitialised portions of the array.
    synchronizer : zarr.sync.ArraySynchronizer, optional
        Array synchronizer.

    Returns
    -------
    z : zarr.array.Array

    """

    # setup directory store
    store = DirectoryStore(path=path, mode=mode, shape=shape, chunks=chunks,
                           dtype=dtype, cname=cname, clevel=clevel,
                           shuffle=shuffle, fill_value=fill_value)

    # handle optional synchronizer
    if synchronizer is not None:
        z = SynchronizedArray(store, synchronizer)
    else:
        z = Array(store)

    return z


def empty_like(z, shape=None, chunks=None, dtype=None, cname=None, clevel=None,
               shuffle=None, synchronizer=None):
    """Create an empty array like 'z'."""

    shape = shape if shape is not None else z.shape
    chunks = chunks if chunks is not None else z.chunks
    dtype = dtype if dtype is not None else z.dtype
    cname = cname if cname is not None else z.cname
    clevel = clevel if clevel is not None else z.clevel
    shuffle = shuffle if shuffle is not None else z.shuffle
    return empty(shape, chunks, dtype=dtype, cname=cname, clevel=clevel,
                 shuffle=shuffle, synchronizer=synchronizer)


def zeros_like(z, shape=None, chunks=None, dtype=None, cname=None, clevel=None,
               shuffle=None, synchronizer=None):
    """Create an array of zeros like 'z'."""

    shape = shape if shape is not None else z.shape
    chunks = chunks if chunks is not None else z.chunks
    dtype = dtype if dtype is not None else z.dtype
    cname = cname if cname is not None else z.cname
    clevel = clevel if clevel is not None else z.clevel
    shuffle = shuffle if shuffle is not None else z.shuffle
    return zeros(shape, chunks, dtype=dtype, cname=cname, clevel=clevel,
                 shuffle=shuffle, synchronizer=synchronizer)


def ones_like(z, shape=None, chunks=None, dtype=None, cname=None, clevel=None,
              shuffle=None, synchronizer=None):
    """Create an array of ones like 'z'."""

    shape = shape if shape is not None else z.shape
    chunks = chunks if chunks is not None else z.chunks
    dtype = dtype if dtype is not None else z.dtype
    cname = cname if cname is not None else z.cname
    clevel = clevel if clevel is not None else z.clevel
    shuffle = shuffle if shuffle is not None else z.shuffle
    return ones(shape, chunks, dtype=dtype, cname=cname, clevel=clevel,
                shuffle=shuffle, synchronizer=synchronizer)


def full_like(z, shape=None, chunks=None, fill_value=None, dtype=None,
              cname=None, clevel=None, shuffle=None, synchronizer=None):
    """Create a filled array like 'z'."""

    shape = shape if shape is not None else z.shape
    chunks = chunks if chunks is not None else z.chunks
    dtype = dtype if dtype is not None else z.dtype
    cname = cname if cname is not None else z.cname
    clevel = clevel if clevel is not None else z.clevel
    shuffle = shuffle if shuffle is not None else z.shuffle
    fill_value = fill_value if fill_value is not None else z.fill_value
    return full(shape, chunks, fill_value, dtype=dtype, cname=cname,
                clevel=clevel, shuffle=shuffle, synchronizer=synchronizer)


def open_like(z, path, mode='a', shape=None, chunks=None, dtype=None,
              cname=None, clevel=None, shuffle=None, fill_value=None,
              synchronizer=None):
    """Open a persistent array like 'z'."""

    shape = shape if shape is not None else z.shape
    chunks = chunks if chunks is not None else z.chunks
    dtype = dtype if dtype is not None else z.dtype
    cname = cname if cname is not None else z.cname
    clevel = clevel if clevel is not None else z.clevel
    shuffle = shuffle if shuffle is not None else z.shuffle
    fill_value = fill_value if fill_value is not None else z.fill_value
    return open(path, mode=mode, shape=shape, chunks=chunks, dtype=dtype,
                cname=cname, clevel=clevel, shuffle=shuffle,
                fill_value=fill_value, synchronizer=synchronizer)
