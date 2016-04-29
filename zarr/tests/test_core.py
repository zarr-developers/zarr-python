# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import json
import unittest
from tempfile import NamedTemporaryFile, mkdtemp
import atexit
import shutil


import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import eq_ as eq, assert_is_instance, assert_is_none, \
    assert_raises
import zict


from zarr.core import Array, SynchronizedArray, init_store
from zarr.sync import ThreadSynchronizer, ProcessSynchronizer
from zarr.compat import text_type
from zarr.meta import decode_metadata
from zarr.errors import ReadOnlyError


def test_init_store():

    store = dict()
    init_store(store, shape=1000, chunks=100)

    # check metadata
    assert 'meta' in store
    meta = decode_metadata(store['meta'])
    eq((1000,), meta['shape'])
    eq((100,), meta['chunks'])
    eq(np.dtype(None), meta['dtype'])
    eq('blosc', meta['compression'])
    assert 'compression_opts' in meta
    assert_is_none(meta['fill_value'])

    # check attributes
    assert 'attrs' in store
    eq(dict(), json.loads(text_type(store['attrs'], 'ascii')))


def test_init_store_overwrite():

    store = dict(shape=(2000,), chunks=(200,))

    # overwrite
    init_store(store, shape=1000, chunks=100, overwrite=True)
    assert 'meta' in store
    meta = decode_metadata(store['meta'])
    eq((1000,), meta['shape'])
    eq((100,), meta['chunks'])

    # don't overwrite
    with assert_raises(ValueError):
        init_store(store, shape=1000, chunks=100, overwrite=False)


def test_array_init():

    store = dict()  # store not initialised
    with assert_raises(ValueError):
        Array(store)


def test_nbytes_stored():

    store = dict()
    init_store(store, shape=1000, chunks=100)
    z = Array(store)
    eq(sum(len(v) for v in store.values()), z.nbytes_stored)
    z[:] = 42
    eq(sum(len(v) for v in store.values()), z.nbytes_stored)

    # custom store, doesn't support size determination
    with NamedTemporaryFile() as f:
        store = zict.Zip(f.name, mode='w')
        init_store(store, shape=1000, chunks=100)
        z = Array(store)
        eq(-1, z.nbytes_stored)
        z[:] = 42
        eq(-1, z.nbytes_stored)


class TestArray(unittest.TestCase):

    def create_array(self, store=None, readonly=False, **kwargs):
        if store is None:
            store = dict()
        init_store(store, **kwargs)
        return Array(store, readonly=readonly)

    def test_1d(self):

        a = np.arange(1050)
        z = self.create_array(shape=a.shape, chunks=100, dtype=a.dtype)

        # check properties
        eq(a.shape, z.shape)
        eq(a.dtype, z.dtype)
        eq((100,), z.chunks)
        eq(a.nbytes, z.nbytes)
        eq(sum(len(v) for v in z.store.values()), z.nbytes_stored)
        eq(0, z.initialized)
        eq((11,), z.cdata_shape)

        # check empty
        b = z[:]
        assert_is_instance(b, np.ndarray)
        eq(a.shape, b.shape)
        eq(a.dtype, b.dtype)

        # set data
        z[:] = a

        # check properties
        eq(a.nbytes, z.nbytes)
        eq(sum(len(v) for v in z.store.values()), z.nbytes_stored)
        eq(11, z.initialized)

        # check slicing
        assert_array_equal(a, np.array(z))
        assert_array_equal(a, z[:])
        assert_array_equal(a, z[...])
        # noinspection PyTypeChecker
        assert_array_equal(a, z[slice(None)])
        assert_array_equal(a[:10], z[:10])
        assert_array_equal(a[10:20], z[10:20])
        assert_array_equal(a[-10:], z[-10:])
        # ...across chunk boundaries...
        assert_array_equal(a[:110], z[:110])
        assert_array_equal(a[190:310], z[190:310])
        assert_array_equal(a[-110:], z[-110:])

        # check partial assignment
        b = np.arange(1e5, 2e5)
        z[190:310] = b[190:310]
        assert_array_equal(a[:190], z[:190])
        assert_array_equal(b[190:310], z[190:310])
        assert_array_equal(a[310:], z[310:])

    def test_array_1d_fill_value(self):

        for fill_value in -1, 0, 1, 10:

            a = np.arange(1050)
            f = np.empty_like(a)
            f.fill(fill_value)
            z = self.create_array(shape=a.shape, chunks=100, dtype=a.dtype,
                                  fill_value=fill_value)
            z[190:310] = a[190:310]

            assert_array_equal(f[:190], z[:190])
            assert_array_equal(a[190:310], z[190:310])
            assert_array_equal(f[310:], z[310:])

    def test_array_1d_set_scalar(self):

        # setup
        a = np.empty(100)
        z = self.create_array(shape=a.shape, chunks=10, dtype=a.dtype)
        z[:] = a
        assert_array_equal(a, z[:])

        for value in -1, 0, 1, 10:
            a[15:35] = value
            z[15:35] = value
            assert_array_equal(a, z[:])
            a[:] = value
            z[:] = value
            assert_array_equal(a, z[:])

    def test_array_2d(self):

        a = np.arange(10000).reshape((1000, 10))
        z = self.create_array(shape=a.shape, chunks=(100, 2), dtype=a.dtype)

        # check properties
        eq(a.shape, z.shape)
        eq(a.dtype, z.dtype)
        eq((100, 2), z.chunks)
        eq(sum(len(v) for v in z.store.values()), z.nbytes_stored)
        eq(0, z.initialized)
        eq((10, 5), z.cdata_shape)

        # set data
        z[:] = a

        # check properties
        eq(a.nbytes, z.nbytes)
        eq(sum(len(v) for v in z.store.values()), z.nbytes_stored)
        eq(50, z.initialized)

        # check slicing
        assert_array_equal(a, np.array(z))
        assert_array_equal(a, z[:])
        assert_array_equal(a, z[...])
        # noinspection PyTypeChecker
        assert_array_equal(a, z[slice(None)])
        assert_array_equal(a[:10], z[:10])
        assert_array_equal(a[10:20], z[10:20])
        assert_array_equal(a[-10:], z[-10:])
        assert_array_equal(a[:, :2], z[:, :2])
        assert_array_equal(a[:, 2:4], z[:, 2:4])
        assert_array_equal(a[:, -2:], z[:, -2:])
        assert_array_equal(a[:10, :2], z[:10, :2])
        assert_array_equal(a[10:20, 2:4], z[10:20, 2:4])
        assert_array_equal(a[-10:, -2:], z[-10:, -2:])
        # ...across chunk boundaries...
        assert_array_equal(a[:110], z[:110])
        assert_array_equal(a[190:310], z[190:310])
        assert_array_equal(a[-110:], z[-110:])
        assert_array_equal(a[:, :3], z[:, :3])
        assert_array_equal(a[:, 3:7], z[:, 3:7])
        assert_array_equal(a[:, -3:], z[:, -3:])
        assert_array_equal(a[:110, :3], z[:110, :3])
        assert_array_equal(a[190:310, 3:7], z[190:310, 3:7])
        assert_array_equal(a[-110:, -3:], z[-110:, -3:])

        # check partial assignment
        b = np.arange(10000, 20000).reshape((1000, 10))
        z[190:310, 3:7] = b[190:310, 3:7]
        assert_array_equal(a[:190], z[:190])
        assert_array_equal(a[:, :3], z[:, :3])
        assert_array_equal(b[190:310, 3:7], z[190:310, 3:7])
        assert_array_equal(a[310:], z[310:])
        assert_array_equal(a[:, 7:], z[:, 7:])

    def test_resize_1d(self):

        z = self.create_array(shape=105, chunks=10, dtype='i4', fill_value=0)
        a = np.arange(105, dtype='i4')
        z[:] = a
        eq((105,), z.shape)
        eq((105,), z[:].shape)
        eq(np.dtype('i4'), z.dtype)
        eq(np.dtype('i4'), z[:].dtype)
        eq((10,), z.chunks)
        assert_array_equal(a, z[:])

        z.resize(205)
        eq((205,), z.shape)
        eq((205,), z[:].shape)
        eq(np.dtype('i4'), z.dtype)
        eq(np.dtype('i4'), z[:].dtype)
        eq((10,), z.chunks)
        assert_array_equal(a, z[:105])
        assert_array_equal(np.zeros(100, dtype='i4'), z[105:])

        z.resize(55)
        eq((55,), z.shape)
        eq((55,), z[:].shape)
        eq(np.dtype('i4'), z.dtype)
        eq(np.dtype('i4'), z[:].dtype)
        eq((10,), z.chunks)
        assert_array_equal(a[:55], z[:])

    def test_resize_2d(self):

        z = self.create_array(shape=(105, 105), chunks=(10, 10), dtype='i4',
                              fill_value=0)
        a = np.arange(105*105, dtype='i4').reshape((105, 105))
        z[:] = a
        eq((105, 105), z.shape)
        eq((105, 105), z[:].shape)
        eq(np.dtype('i4'), z.dtype)
        eq(np.dtype('i4'), z[:].dtype)
        eq((10, 10), z.chunks)
        assert_array_equal(a, z[:])

        z.resize((205, 205))
        eq((205, 205), z.shape)
        eq((205, 205), z[:].shape)
        eq(np.dtype('i4'), z.dtype)
        eq(np.dtype('i4'), z[:].dtype)
        eq((10, 10), z.chunks)
        assert_array_equal(a, z[:105, :105])
        assert_array_equal(np.zeros((100, 205), dtype='i4'), z[105:, :])
        assert_array_equal(np.zeros((205, 100), dtype='i4'), z[:, 105:])

        z.resize((55, 55))
        eq((55, 55), z.shape)
        eq((55, 55), z[:].shape)
        eq(np.dtype('i4'), z.dtype)
        eq(np.dtype('i4'), z[:].dtype)
        eq((10, 10), z.chunks)
        assert_array_equal(a[:55, :55], z[:])

        z.resize((55, 1))
        eq((55, 1), z.shape)
        eq((55, 1), z[:].shape)
        eq(np.dtype('i4'), z.dtype)
        eq(np.dtype('i4'), z[:].dtype)
        eq((10, 10), z.chunks)
        assert_array_equal(a[:55, :1], z[:])

    def test_append_1d(self):

        a = np.arange(105)
        z = self.create_array(shape=a.shape, chunks=10, dtype=a.dtype)
        z[:] = a
        eq(a.shape, z.shape)
        eq(a.dtype, z.dtype)
        eq((10,), z.chunks)
        assert_array_equal(a, z[:])

        b = np.arange(105, 205)
        e = np.append(a, b)
        z.append(b)
        eq(e.shape, z.shape)
        eq(e.dtype, z.dtype)
        eq((10,), z.chunks)
        assert_array_equal(e, z[:])

        # check append handles array-like
        c = [1, 2, 3]
        f = np.append(e, c)
        z.append(c)
        eq(f.shape, z.shape)
        eq(f.dtype, z.dtype)
        eq((10,), z.chunks)
        assert_array_equal(f, z[:])

    def test_append_2d(self):

        a = np.arange(105*105, dtype='i4').reshape((105, 105))
        z = self.create_array(shape=a.shape, chunks=(10, 10), dtype=a.dtype)
        z[:] = a
        eq(a.shape, z.shape)
        eq(a.dtype, z.dtype)
        eq((10, 10), z.chunks)
        assert_array_equal(a, z[:])

        b = np.arange(105*105, 2*105*105, dtype='i4').reshape((105, 105))
        e = np.append(a, b, axis=0)
        z.append(b)
        eq(e.shape, z.shape)
        eq(e.dtype, z.dtype)
        eq((10, 10), z.chunks)
        assert_array_equal(e, z[:])

    def test_append_2d_axis(self):

        a = np.arange(105*105, dtype='i4').reshape((105, 105))
        z = self.create_array(shape=a.shape, chunks=(10, 10), dtype=a.dtype)
        z[:] = a
        eq(a.shape, z.shape)
        eq(a.dtype, z.dtype)
        eq((10, 10), z.chunks)
        assert_array_equal(a, z[:])

        b = np.arange(105*105, 2*105*105, dtype='i4').reshape((105, 105))
        e = np.append(a, b, axis=1)
        z.append(b, axis=1)
        eq(e.shape, z.shape)
        eq(e.dtype, z.dtype)
        eq((10, 10), z.chunks)
        assert_array_equal(e, z[:])

    def test_readonly(self):

        z = self.create_array(shape=1000, chunks=100, readonly=True)
        with assert_raises(ReadOnlyError):
            z[:] = 42
        with assert_raises(ReadOnlyError):
            z.resize(2000)
        with assert_raises(ReadOnlyError):
            z.append(np.arange(1000))


class TestThreadSynchronizedArray(TestArray):

    def create_array(self, store=None, readonly=False, **kwargs):
        if store is None:
            store = dict()
        init_store(store, **kwargs)
        return SynchronizedArray(store, synchronizer=ThreadSynchronizer(),
                                 readonly=readonly)


class TestProcessSynchronizedArray(TestArray):

    def create_array(self, store=None, readonly=False, **kwargs):
        if store is None:
            store = dict()
        init_store(store, **kwargs)
        sync_path = mkdtemp()
        atexit.register(shutil.rmtree, sync_path)
        synchronizer = ProcessSynchronizer(sync_path)
        return SynchronizedArray(store, synchronizer=synchronizer,
                                 readonly=readonly)
