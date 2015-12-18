# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np


import zarr.ext as _ext


def empty(shape, chunks, dtype=None, cname=None, clevel=None, shuffle=None,
          synchronized=True):
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
    synchronized : bool, optional
        If True, each chunk will be protected with a lock to prevent data
        collision during write operations.

    Returns
    -------
    z : zarr.ext.Array

    """

    return _ext.Array(shape, chunks=chunks, dtype=dtype, cname=cname,
                      clevel=clevel, shuffle=shuffle,
                      synchronized=synchronized)


def zeros(shape, chunks, dtype=None, cname=None, clevel=None, shuffle=None,
          synchronized=True):
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
    synchronized : bool, optional
        If True, each chunk will be protected with a lock to prevent data
        collision during write operations.

    Returns
    -------
    z : zarr.ext.Array

    """

    return _ext.Array(shape, chunks=chunks, dtype=dtype, cname=cname,
                      clevel=clevel, shuffle=shuffle, fill_value=0,
                      synchronized=synchronized)


def ones(shape, chunks, dtype=None, cname=None, clevel=None, shuffle=None,
         synchronized=True):
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
    synchronized : bool, optional
        If True, each chunk will be protected with a lock to prevent data
        collision during write operations.

    Returns
    -------
    z : zarr.ext.Array

    """


    return _ext.Array(shape, chunks=chunks, dtype=dtype, cname=cname,
                      clevel=clevel, shuffle=shuffle, fill_value=1,
                      synchronized=synchronized)


def full(shape, chunks, fill_value, dtype=None, cname=None, clevel=None,
         shuffle=None, synchronized=True):
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
    synchronized : bool, optional
        If True, each chunk will be protected with a lock to prevent data
        collision during write operations.

    Returns
    -------
    z : zarr.ext.Array

    """

    return _ext.Array(shape, chunks=chunks, dtype=dtype, cname=cname,
                      clevel=clevel, shuffle=shuffle, fill_value=fill_value,
                      synchronized=synchronized)


def array(data, chunks=None, dtype=None, cname=None, clevel=None,
          shuffle=None, synchronized=True, fill_value=None):
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
    synchronized : bool, optional
        If True, each chunk will be protected with a lock to prevent data
        collision during write operations.
    fill_value : object
        Default value to use for uninitialised portions of the array.

    Returns
    -------
    z : zarr.ext.Array

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
        if hasattr(data, 'chunklen'):
            # bcolz carray
            chunks = (data.chunklen,) + shape[1:]
        elif hasattr(data, 'chunks') and len(data.chunks) == len(data.shape):
            # h5py dataset or zarr array
            chunks = data.chunks
        else:
            raise ValueError('chunks must be specified')

    # create array
    z = _ext.Array(shape, chunks=chunks, dtype=dtype, cname=cname,
                   clevel=clevel, shuffle=shuffle,
                   synchronized=synchronized, fill_value=fill_value)

    # fill with data
    z[:] = data

    return z
