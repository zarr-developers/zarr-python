# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np


from zarr import ext as _ext


def empty(shape, chunks, dtype=None, cname=None, clevel=None, shuffle=None,
          synchronized=True, lazy=False):
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
        collision during concurrent write operations.
    lazy : bool, optional
        If True, an alternative array class is used which instantiates chunk
        objects only on demand. This may reduce overhead when working with
        small regions of very large arrays with a large number of chunks.

    Returns
    -------
    z : zarr Array

    """

    if synchronized:
        if lazy:
            cls = _ext.SynchronizedLazyArray
        else:
            cls = _ext.SynchronizedArray
    else:
        if lazy:
            cls = _ext.LazyArray
        else:
            cls = _ext.Array
    return cls(shape=shape, chunks=chunks, dtype=dtype, cname=cname,
               clevel=clevel, shuffle=shuffle, fill_value=None)


def zeros(shape, chunks, dtype=None, cname=None, clevel=None, shuffle=None,
          synchronized=True, lazy=False):
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
        collision during concurrent write operations.
    lazy : bool, optional
        If True, an alternative array class is used which instantiates chunk
        objects only on demand. This may reduce overhead when working with
        small regions of very large arrays with a large number of chunks.

    Returns
    -------
    z : zarr Array

    """

    if synchronized:
        if lazy:
            cls = _ext.SynchronizedLazyArray
        else:
            cls = _ext.SynchronizedArray
    else:
        if lazy:
            cls = _ext.LazyArray
        else:
            cls = _ext.Array
    return cls(shape=shape, chunks=chunks, dtype=dtype, cname=cname,
               clevel=clevel, shuffle=shuffle, fill_value=0)


def ones(shape, chunks, dtype=None, cname=None, clevel=None, shuffle=None,
         synchronized=True, lazy=False):
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
    lazy : bool, optional
        If True, an alternative array class is used which instantiates chunk
        objects only on demand. This may reduce overhead when working with
        small regions of very large arrays with a large number of chunks.

    Returns
    -------
    z : zarr Array

    """

    if synchronized:
        if lazy:
            cls = _ext.SynchronizedLazyArray
        else:
            cls = _ext.SynchronizedArray
    else:
        if lazy:
            cls = _ext.LazyArray
        else:
            cls = _ext.Array
    return cls(shape=shape, chunks=chunks, dtype=dtype, cname=cname,
               clevel=clevel, shuffle=shuffle, fill_value=1)


def full(shape, chunks, fill_value, dtype=None, cname=None, clevel=None,
         shuffle=None, synchronized=True, lazy=False):
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
    lazy : bool, optional
        If True, an alternative array class is used which instantiates chunk
        objects only on demand. This may reduce overhead when working with
        small regions of very large arrays with a large number of chunks.

    Returns
    -------
    z : zarr Array

    """

    if synchronized:
        if lazy:
            cls = _ext.SynchronizedLazyArray
        else:
            cls = _ext.SynchronizedArray
    else:
        if lazy:
            cls = _ext.LazyArray
        else:
            cls = _ext.Array
    return cls(shape=shape, chunks=chunks, dtype=dtype, cname=cname,
               clevel=clevel, shuffle=shuffle, fill_value=fill_value)


def array(data, chunks=None, dtype=None, cname=None, clevel=None,
          shuffle=None, fill_value=None, synchronized=True, lazy=False):
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
    synchronized : bool, optional
        If True, each chunk will be protected with a lock to prevent data
        collision during write operations.
    lazy : bool, optional
        If True, an alternative array class is used which instantiates chunk
        objects only on demand. This may reduce overhead when working with
        small regions of very large arrays with a large number of chunks.

    Returns
    -------
    z : zarr Array

    """

    # determine array class to use
    if synchronized:
        if lazy:
            cls = _ext.SynchronizedLazyArray
        else:
            cls = _ext.SynchronizedArray
    else:
        if lazy:
            cls = _ext.LazyArray
        else:
            cls = _ext.Array

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

    # instantiate array
    z = cls(shape=shape, chunks=chunks, dtype=dtype, cname=cname,
            clevel=clevel, shuffle=shuffle, fill_value=fill_value)

    # fill with data
    z[:] = data

    return z


# noinspection PyShadowingBuiltins
def open(path, mode='a', shape=None, chunks=None, dtype=None, cname=None,
         clevel=None, shuffle=None, fill_value=0, synchronized=True,
         lazy=False):
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
    synchronized : bool, optional
        If True, each chunk will be protected with a lock to prevent data
        collision during write operations.
    lazy : bool, optional
        If True, an alternative array class is used which instantiates chunk
        objects only on demand. This may reduce overhead when working with
        small regions of very large arrays with a large number of chunks.

    Returns
    -------
    z : zarr Array

    """

    if synchronized:
        if lazy:
            cls = _ext.SynchronizedLazyPersistentArray
        else:
            cls = _ext.SynchronizedPersistentArray
    else:
        if lazy:
            cls = _ext.LazyPersistentArray
        else:
            cls = _ext.PersistentArray
    return cls(path=path, mode=mode, shape=shape, chunks=chunks, dtype=dtype,
               cname=cname, clevel=clevel, shuffle=shuffle,
               fill_value=fill_value)


def empty_like(z, shape=None, chunks=None, dtype=None, cname=None, clevel=None,
               shuffle=None, synchronized=None, lazy=None):
    """Create an empty array like 'z'."""
    
    shape = shape if shape is not None else z.shape
    chunks = chunks if chunks is not None else z.chunks
    dtype = dtype if dtype is not None else z.dtype
    cname = cname if cname is not None else z.cname
    clevel = clevel if clevel is not None else z.clevel
    shuffle = shuffle if shuffle is not None else z.shuffle
    synchronized = synchronized if synchronized is not None \
        else z.is_synchronized
    lazy = lazy if lazy is not None else z.is_lazy
    return empty(shape, chunks, dtype=dtype, cname=cname, clevel=clevel, 
                 shuffle=shuffle, synchronized=synchronized, lazy=lazy)


def zeros_like(z, shape=None, chunks=None, dtype=None, cname=None, clevel=None,
               shuffle=None, synchronized=None, lazy=None):
    """Create an array of zeros like 'z'."""
    
    shape = shape if shape is not None else z.shape
    chunks = chunks if chunks is not None else z.chunks
    dtype = dtype if dtype is not None else z.dtype
    cname = cname if cname is not None else z.cname
    clevel = clevel if clevel is not None else z.clevel
    shuffle = shuffle if shuffle is not None else z.shuffle
    synchronized = synchronized if synchronized is not None \
        else z.is_synchronized
    lazy = lazy if lazy is not None else z.is_lazy
    return zeros(shape, chunks, dtype=dtype, cname=cname, clevel=clevel, 
                 shuffle=shuffle, synchronized=synchronized, lazy=lazy)


def ones_like(z, shape=None, chunks=None, dtype=None, cname=None, clevel=None,
              shuffle=None, synchronized=None, lazy=None):
    """Create an array of ones like 'z'."""
    
    shape = shape if shape is not None else z.shape
    chunks = chunks if chunks is not None else z.chunks
    dtype = dtype if dtype is not None else z.dtype
    cname = cname if cname is not None else z.cname
    clevel = clevel if clevel is not None else z.clevel
    shuffle = shuffle if shuffle is not None else z.shuffle
    synchronized = synchronized if synchronized is not None \
        else z.is_synchronized
    lazy = lazy if lazy is not None else z.is_lazy
    return ones(shape, chunks, dtype=dtype, cname=cname, clevel=clevel, 
                shuffle=shuffle, synchronized=synchronized, lazy=lazy)


def full_like(z, shape=None, chunks=None, fill_value=None, dtype=None, 
              cname=None, clevel=None, shuffle=None, synchronized=None, 
              lazy=None):
    """Create a filled array like 'z'."""
    
    shape = shape if shape is not None else z.shape
    chunks = chunks if chunks is not None else z.chunks
    dtype = dtype if dtype is not None else z.dtype
    cname = cname if cname is not None else z.cname
    clevel = clevel if clevel is not None else z.clevel
    shuffle = shuffle if shuffle is not None else z.shuffle
    fill_value = fill_value if fill_value is not None else z.fill_value
    synchronized = synchronized if synchronized is not None \
        else z.is_synchronized
    lazy = lazy if lazy is not None else z.is_lazy
    return full(shape, chunks, fill_value, dtype=dtype, cname=cname,
                clevel=clevel, shuffle=shuffle, synchronized=synchronized,
                lazy=lazy)


def open_like(z, path, mode='a', shape=None, chunks=None, dtype=None,
              cname=None, clevel=None, shuffle=None, fill_value=None,
              synchronized=None, lazy=None):
    """Open a persistent array like 'z'."""

    shape = shape if shape is not None else z.shape
    chunks = chunks if chunks is not None else z.chunks
    dtype = dtype if dtype is not None else z.dtype
    cname = cname if cname is not None else z.cname
    clevel = clevel if clevel is not None else z.clevel
    shuffle = shuffle if shuffle is not None else z.shuffle
    fill_value = fill_value if fill_value is not None else z.fill_value
    synchronized = synchronized if synchronized is not None \
        else z.is_synchronized
    lazy = lazy if lazy is not None else z.is_lazy
    return open(path, mode=mode, shape=shape, chunks=chunks, dtype=dtype,
                cname=cname, clevel=clevel, shuffle=shuffle,
                fill_value=fill_value, synchronized=synchronized, lazy=lazy)
