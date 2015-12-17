# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np


import zarr.ext as _ext


def empty(shape, chunks, dtype=None, cname=None, clevel=None, shuffle=None,
          synchronized=True):
    """TODO"""

    return _ext.Array(shape, chunks=chunks, dtype=dtype, cname=cname,
                      clevel=clevel, shuffle=shuffle,
                      synchronized=synchronized)


def zeros(shape, chunks, dtype=None, cname=None, clevel=None, shuffle=None,
          synchronized=True):
    """TODO"""

    return _ext.Array(shape, chunks=chunks, dtype=dtype, cname=cname,
                      clevel=clevel, shuffle=shuffle, fill_value=0,
                      synchronized=synchronized)


def ones(shape, chunks, dtype=None, cname=None, clevel=None, shuffle=None,
          synchronized=True):
    """TODO"""

    return _ext.Array(shape, chunks=chunks, dtype=dtype, cname=cname,
                      clevel=clevel, shuffle=shuffle, fill_value=1,
                      synchronized=synchronized)


def full(shape, chunks, fill_value, dtype=None, cname=None, clevel=None,
         shuffle=None, synchronized=True):
    """TODO"""

    return _ext.Array(shape, chunks=chunks, dtype=dtype, cname=cname,
                      clevel=clevel, shuffle=shuffle, fill_value=fill_value,
                      synchronized=synchronized)


def array(data, chunks=None, dtype=None, cname=None, clevel=None,
          shuffle=None, synchronized=True):
    """TODO"""

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
                   clevel=clevel, shuffle=shuffle, synchronized=synchronized)

    # fill with data
    z[:] = data

    return z
