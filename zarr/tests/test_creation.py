# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import tempfile
import shutil
import atexit
import os


import numpy as np
from nose.tools import eq_ as eq, assert_is_none, assert_is_instance, \
    assert_raises
from numpy.testing import assert_array_equal


from zarr.creation import array, empty, zeros, ones, full, open, empty_like, \
    zeros_like, ones_like, full_like, open_like, create
from zarr.sync import ThreadSynchronizer, SynchronizedArray
from zarr.core import Array
from zarr.storage import DirectoryStore, init_store


def test_array():

    # with numpy array
    a = np.arange(100)
    z = array(a, chunks=10)
    eq(a.shape, z.shape)
    eq(a.dtype, z.dtype)
    assert_array_equal(a, z[:])

    # with array-like
    a = list(range(100))
    z = array(a, chunks=10)
    eq((100,), z.shape)
    eq(np.asarray(a).dtype, z.dtype)
    assert_array_equal(np.asarray(a), z[:])

    # with another zarr array
    z2 = array(z)
    eq(z.shape, z2.shape)
    eq(z.chunks, z2.chunks)
    eq(z.dtype, z2.dtype)
    assert_array_equal(z[:], z2[:])

    # with something bcolz-like
    class MockBcolzArray(object):

        def __init__(self, data, chunklen):
            self.data = data
            self.chunklen = chunklen

        def __getattr__(self, item):
            return getattr(self.data, item)

        def __getitem__(self, item):
            return self.data[item]

    b = np.arange(1000).reshape(100, 10)
    c = MockBcolzArray(b, 10)
    z3 = array(c)
    eq(c.shape, z3.shape)
    eq((10, 10), z3.chunks)

    # chunks not specified
    with assert_raises(ValueError):
        z = array(np.arange(100))


def test_empty():
    z = empty(100, 10)
    eq((100,), z.shape)
    eq((10,), z.chunks)


def test_zeros():
    z = zeros(100, 10)
    eq((100,), z.shape)
    eq((10,), z.chunks)
    assert_array_equal(np.zeros(100), z[:])


def test_ones():
    z = ones(100, 10)
    eq((100,), z.shape)
    eq((10,), z.chunks)
    assert_array_equal(np.ones(100), z[:])


def test_full():
    z = full(100, 10, fill_value=42, dtype='i4')
    eq((100,), z.shape)
    eq((10,), z.chunks)
    assert_array_equal(np.full(100, fill_value=42, dtype='i4'), z[:])


def test_open():

    path = tempfile.mktemp()
    atexit.register(
        lambda: shutil.rmtree(path) if os.path.exists(path) else None
    )
    z = open(path, mode='w', shape=100, chunks=10, dtype='i4')
    z[:] = 42
    eq((100,), z.shape)
    eq((10,), z.chunks)
    assert_array_equal(np.full(100, fill_value=42, dtype='i4'), z[:])
    z2 = open(path, mode='r')
    eq((100,), z2.shape)
    eq((10,), z2.chunks)
    assert_array_equal(z[:], z2[:])

    # path does not exist
    path = 'doesnotexist'
    with assert_raises(ValueError):
        open(path, mode='r')

    # path exists but store not initialised
    path = tempfile.mkdtemp()
    atexit.register(shutil.rmtree, path)
    with assert_raises(ValueError):
        open(path, mode='r')
    with assert_raises(ValueError):
        open(path, mode='r+')

    # store initialised, mode w-
    store = DirectoryStore(path)
    init_store(store, shape=100, chunks=10)
    with assert_raises(ValueError):
        open(path, mode='w-')
    with assert_raises(ValueError):
        open(path, mode='x')

    # with synchronizer
    z = open(path, synchronizer=ThreadSynchronizer())
    assert_is_instance(z, SynchronizedArray)


def test_empty_like():
    # zarr array
    z = empty(100, 10, dtype='f4', compression='zlib',
              compression_opts=5, order='F')
    z2 = empty_like(z)
    eq(z.shape, z2.shape)
    eq(z.chunks, z2.chunks)
    eq(z.dtype, z2.dtype)
    eq(z.compression, z2.compression)
    eq(z.compression_opts, z2.compression_opts)
    eq(z.fill_value, z2.fill_value)
    eq(z.order, z2.order)
    # numpy array
    a = np.empty(100, dtype='f4')
    z3 = empty_like(a, chunks=10)
    eq(a.shape, z3.shape)
    eq((10,), z3.chunks)
    eq(a.dtype, z3.dtype)
    assert_is_none(z3.fill_value)
    with assert_raises(ValueError):
        # chunks missing
        empty_like(a)


def test_zeros_like():
    # zarr array
    z = zeros(100, 10, dtype='f4', compression='zlib',
              compression_opts=5, order='F')
    z2 = zeros_like(z)
    eq(z.shape, z2.shape)
    eq(z.chunks, z2.chunks)
    eq(z.dtype, z2.dtype)
    eq(z.compression, z2.compression)
    eq(z.compression_opts, z2.compression_opts)
    eq(z.fill_value, z2.fill_value)
    eq(z.order, z2.order)
    # numpy array
    a = np.empty(100, dtype='f4')
    z3 = zeros_like(a, chunks=10)
    eq(a.shape, z3.shape)
    eq((10,), z3.chunks)
    eq(a.dtype, z3.dtype)
    eq(0, z3.fill_value)
    with assert_raises(ValueError):
        # chunks missing
        zeros_like(a)


def test_ones_like():
    # zarr array
    z = ones(100, 10, dtype='f4', compression='zlib',
             compression_opts=5, order='F')
    z2 = ones_like(z)
    eq(z.shape, z2.shape)
    eq(z.chunks, z2.chunks)
    eq(z.dtype, z2.dtype)
    eq(z.compression, z2.compression)
    eq(z.compression_opts, z2.compression_opts)
    eq(z.fill_value, z2.fill_value)
    eq(z.order, z2.order)
    # numpy array
    a = np.empty(100, dtype='f4')
    z3 = ones_like(a, chunks=10)
    eq(a.shape, z3.shape)
    eq((10,), z3.chunks)
    eq(a.dtype, z3.dtype)
    eq(1, z3.fill_value)
    with assert_raises(ValueError):
        # chunks missing
        ones_like(a)


def test_full_like():
    z = full(100, 10, dtype='f4', compression='zlib',
             compression_opts=5, fill_value=42, order='F')
    z2 = full_like(z)
    eq(z.shape, z2.shape)
    eq(z.chunks, z2.chunks)
    eq(z.dtype, z2.dtype)
    eq(z.compression, z2.compression)
    eq(z.compression_opts, z2.compression_opts)
    eq(z.fill_value, z2.fill_value)
    eq(z.order, z2.order)
    # numpy array
    a = np.empty(100, dtype='f4')
    z3 = full_like(a, chunks=10, fill_value=42)
    eq(a.shape, z3.shape)
    eq((10,), z3.chunks)
    eq(a.dtype, z3.dtype)
    eq(42, z3.fill_value)
    with assert_raises(ValueError):
        # chunks missing
        full_like(a)
    with assert_raises(ValueError):
        # fill_value missing
        full_like(a, chunks=10)


def test_open_like():
    # zarr array
    path = tempfile.mktemp()
    atexit.register(shutil.rmtree, path)
    z = full(100, 10, dtype='f4', compression='zlib',
             compression_opts=5, fill_value=42, order='F')
    z2 = open_like(z, path)
    eq(z.shape, z2.shape)
    eq(z.chunks, z2.chunks)
    eq(z.dtype, z2.dtype)
    eq(z.compression, z2.compression)
    eq(z.compression_opts, z2.compression_opts)
    eq(z.fill_value, z2.fill_value)
    eq(z.order, z2.order)
    # numpy array
    path = tempfile.mktemp()
    atexit.register(shutil.rmtree, path)
    a = np.empty(100, dtype='f4')
    z3 = open_like(a, path, chunks=10)
    eq(a.shape, z3.shape)
    eq((10,), z3.chunks)
    eq(a.dtype, z3.dtype)
    assert_is_none(z3.fill_value)
    with assert_raises(ValueError):
        # chunks missing
        open_like(a, path)


def test_create():

    # defaults
    z = create(100, 10)
    assert_is_instance(z, Array)
    eq((100,), z.shape)
    eq((10,), z.chunks)
    eq(np.dtype(None), z.dtype)
    eq('blosc', z.compression)
    assert_is_none(z.fill_value)

    # all specified
    z = create(100, 10, dtype='i4', compression='zlib', compression_opts=1,
               fill_value=42, order='F')
    assert_is_instance(z, Array)
    eq((100,), z.shape)
    eq((10,), z.chunks)
    eq(np.dtype('i4'), z.dtype)
    eq('zlib', z.compression)
    eq(1, z.compression_opts)
    eq(42, z.fill_value)
    eq('F', z.order)

    # with synchronizer
    synchronizer = ThreadSynchronizer()
    z = create(100, 10, synchronizer=synchronizer)
    assert_is_instance(z, SynchronizedArray)
    eq((100,), z.shape)
    eq((10,), z.chunks)
    assert synchronizer is z.synchronizer
