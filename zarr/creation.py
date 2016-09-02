# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np


from zarr.core import Array
from zarr.storage import DirectoryStore, init_array, contains_array, \
    contains_group


def create(shape, chunks=None, dtype=None, compressor='default',
           fill_value=None, order='C', store=None, synchronizer=None,
           overwrite=False, path=None, chunk_store=None, filters=None):
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
    store : MutableMapping, optional
        Array storage. If not provided, a Python dict will be used, meaning
        array data will be stored in memory.
    synchronizer : object, optional
        Array synchronizer.
    overwrite : bool, optional
        If True, delete all pre-existing data in `store` before creating the
        array.
    path : string, optional
        Path under which array is stored.
    chunk_store : MutableMapping, optional
        Separate storage for chunks. If not provided, `store` will be used
        for storage of both chunks and metadata.
    filters : sequence of Codecs, optional
        Sequence of filters to use to encode chunk data prior to compression.

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
          nbytes: 762.9M; nbytes_stored: 326; ratio: 2453987.7; initialized: 0/100
          compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
          store: dict

    """  # flake8: noqa

    # initialize store
    if store is None:
        store = dict()

    # initialize array metadata
    init_array(store, shape=shape, chunks=chunks, dtype=dtype,
               compressor=compressor, fill_value=fill_value, order=order, 
               overwrite=overwrite, path=path, chunk_store=chunk_store, 
               filters=filters)

    # instantiate array
    z = Array(store, path=path, chunk_store=chunk_store,
              synchronizer=synchronizer)

    return z


def empty(shape, chunks=None, dtype=None, compressor='default',
          order='C', store=None, synchronizer=None, path=None, 
          overwrite=False, chunk_store=None, filters=None):
    """Create an empty array.

    For parameter definitions see :func:`zarr.creation.create`.

    Notes
    -----
    The contents of an empty Zarr array are not defined. On attempting to
    retrieve data from an empty Zarr array, any values may be returned,
    and these are not guaranteed to be stable from one access to the next.

    """
    return create(shape=shape, chunks=chunks, dtype=dtype,
                  compressor=compressor, fill_value=None, order=order, 
                  store=store, synchronizer=synchronizer, path=path, 
                  overwrite=overwrite, chunk_store=chunk_store, 
                  filters=filters)


def zeros(shape, chunks=None, dtype=None, compressor='default',
          order='C', store=None, synchronizer=None, path=None,
          overwrite=False, chunk_store=None, filters=None):
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

    return create(shape=shape, chunks=chunks, dtype=dtype,
                  compressor=compressor, fill_value=0, order=order,
                  store=store, synchronizer=synchronizer, path=path,
                  overwrite=overwrite, chunk_store=chunk_store, filters=filters)


def ones(shape, chunks=None, dtype=None, compressor='default',
         order='C', store=None, synchronizer=None, path=None,
         overwrite=False, chunk_store=None, filters=None):
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

    return create(shape=shape, chunks=chunks, dtype=dtype,
                  compressor=compressor, fill_value=1, order=order,
                  store=store, synchronizer=synchronizer, path=path,
                  overwrite=overwrite, chunk_store=chunk_store,
                  filters=filters)


def full(shape, fill_value, chunks=None, dtype=None, compressor='default',
         order='C', store=None, synchronizer=None, path=None,
         overwrite=False, chunk_store=None, filters=None):
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

    return create(shape=shape, chunks=chunks, dtype=dtype,
                  compressor=compressor, fill_value=fill_value, order=order,
                  store=store, synchronizer=synchronizer, path=path,
                  overwrite=overwrite, chunk_store=chunk_store,
                  filters=filters)


def array(data, chunks=None, dtype=None, compressor='default',
          fill_value=None, order='C', store=None, synchronizer=None,
          path=None, overwrite=False, chunk_store=None, filters=None):
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
      nbytes: 762.9M; nbytes_stored: 17.1M; ratio: 44.7; initialized: 100/100
      compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
      store: dict

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
               compressor=compressor, fill_value=fill_value, order=order,
               store=store, synchronizer=synchronizer, path=path,
               overwrite=overwrite, chunk_store=chunk_store, filters=filters)

    # fill with data
    z[:] = data

    return z


def open_array(path, mode='a', shape=None, chunks=None, dtype=None,
               compressor='default', fill_value=0, order='C',
               synchronizer=None, filters=None):
    """Convenience function to instantiate an array stored in a
    directory on the file system.

    Parameters
    ----------
    path : string
        Path to directory in file system in which to store the array.
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
      nbytes: 762.9M; nbytes_stored: 24.8M; ratio: 30.8; initialized: 100/100
      compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
      store: DirectoryStore
    >>> z2 = zarr.open_array('example.zarr', mode='r')
    >>> z2
    Array((10000, 10000), float64, chunks=(1000, 1000), order=C)
      nbytes: 762.9M; nbytes_stored: 24.8M; ratio: 30.8; initialized: 100/100
      compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
      store: DirectoryStore
    >>> np.all(z1[:] == z2[:])
    True

    Notes
    -----
    There is no need to close an array. Data are automatically flushed to the
    file system.

    """  # flake8: noqa

    # use same mode semantics as h5py, although N.B., here `path` is a
    # directory:
    # r : read only, must exist
    # r+ : read/write, must exist
    # w : create, delete if exists
    # w- or x : create, fail if exists
    # a : read/write if exists, create otherwise (default)

    # setup store
    store = DirectoryStore(path)

    # ensure store is initialized

    if mode in ['r', 'r+']:
        if contains_group(store):
            raise ValueError('store contains group')
        elif not contains_array(store):
            raise ValueError('array does not exist')

    elif mode == 'w':
        init_array(store, shape=shape, chunks=chunks, dtype=dtype,
                   compressor=compressor, fill_value=fill_value,
                   order=order, filters=filters, overwrite=True)

    elif mode == 'a':
        if contains_group(store):
            raise ValueError('store contains group')
        elif not contains_array(store):
            init_array(store, shape=shape, chunks=chunks, dtype=dtype,
                       compressor=compressor, fill_value=fill_value,
                       order=order, filters=filters)

    elif mode in ['w-', 'x']:
        if contains_group(store):
            raise ValueError('store contains group')
        elif contains_array(store):
            raise ValueError('store contains array')
        else:
            init_array(store, shape=shape, chunks=chunks, dtype=dtype,
                       compressor=compressor, fill_value=fill_value,
                       order=order, filters=filters)

    # determine read only status
    read_only = mode == 'r'

    # instantiate array
    z = Array(store, read_only=read_only, synchronizer=synchronizer)

    return z


# backwards compatibility
open = open_array


def _like_args(a, shape, chunks, dtype, compressor, order, filters):

    # handle shape
    if shape is None:
        shape = a.shape

    # handle chunks
    if chunks is None:
        try:
            chunks = a.chunks
        except AttributeError:
            # use auto-chunking
            pass

    # handle dtype
    if dtype is None:
        try:
            dtype = a.dtype
        except AttributeError:
            pass

    # handle compressor
    if compressor is None:
        if isinstance(a, Array):
            compressor = a.compressor
        else:
            compressor = 'default'

    # handle order
    if order is None:
        if isinstance(a, Array):
            order = a.order
        else:
            order = 'C'

    # handle filters
    if filters is None:
        if isinstance(a, Array):
            filters = a.filters

    return shape, chunks, dtype, compressor, order, filters


def empty_like(a, shape=None, chunks=None, dtype=None, compressor=None,
               order=None, store=None, synchronizer=None, path=None,
               overwrite=False, chunk_store=None, filters=None):
    """Create an empty array like `a`."""
    shape, chunks, dtype, compressor, order, filters = \
        _like_args(a, shape, chunks, dtype, compressor, order, filters)
    return empty(shape, chunks=chunks, dtype=dtype, compressor=compressor,
                 order=order, store=store, synchronizer=synchronizer,
                 path=path, overwrite=overwrite, chunk_store=chunk_store,
                 filters=filters)


def zeros_like(a, shape=None, chunks=None, dtype=None, compressor=None,
               order=None, store=None, synchronizer=None, path=None,
               overwrite=False, chunk_store=None, filters=None):
    """Create an array of zeros like `a`."""
    shape, chunks, dtype, compressor, order, filters = \
        _like_args(a, shape, chunks, dtype, compressor, order, filters)
    return zeros(shape, chunks=chunks, dtype=dtype, compressor=compressor,
                 order=order, store=store, synchronizer=synchronizer,
                 path=path, overwrite=overwrite, chunk_store=chunk_store,
                 filters=filters)


def ones_like(a, shape=None, chunks=None, dtype=None, compressor=None,
              order=None, store=None, synchronizer=None, path=None,
              overwrite=False, chunk_store=None, filters=None):
    """Create an array of ones like `a`."""
    shape, chunks, dtype, compressor, order, filters = \
        _like_args(a, shape, chunks, dtype, compressor, order, filters)
    return ones(shape, chunks=chunks, dtype=dtype, compressor=compressor,
                order=order, store=store, synchronizer=synchronizer, path=path,
                overwrite=overwrite, chunk_store=chunk_store, filters=filters)


def full_like(a, shape=None, chunks=None, fill_value=None, dtype=None,
              compressor=None, order=None, store=None, synchronizer=None,
              path=None, overwrite=False, chunk_store=None, filters=None):
    """Create a filled array like `a`."""
    shape, chunks, dtype, compressor, order, filters = \
        _like_args(a, shape, chunks, dtype, compressor, order, filters)
    if fill_value is None:
        try:
            fill_value = a.fill_value
        except AttributeError:
            raise ValueError('fill_value must be specified')
    return full(shape, chunks=chunks, fill_value=fill_value, dtype=dtype,
                compressor=compressor, order=order, store=store,
                synchronizer=synchronizer, path=path, overwrite=overwrite,
                chunk_store=chunk_store, filters=filters)


def open_like(a, path, mode='a', shape=None, chunks=None, dtype=None,
              compressor=None, fill_value=None, order=None,
              synchronizer=None, filters=None):
    """Open a persistent array like `a`."""
    shape, chunks, dtype, compressor, order, filters = \
        _like_args(a, shape, chunks, dtype, compressor, order, filters)
    if fill_value is None:
        try:
            fill_value = a.fill_value
        except AttributeError:
            # leave empty
            pass
    return open_array(path, mode=mode, shape=shape, chunks=chunks, dtype=dtype,
                      compressor=compressor, fill_value=fill_value,
                      order=order, synchronizer=synchronizer, filters=filters)
