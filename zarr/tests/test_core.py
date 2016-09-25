# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import unittest
from tempfile import mkdtemp
import atexit
import shutil
import pickle
from collections import MutableMapping


import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import eq_ as eq, assert_is_instance, \
    assert_raises, assert_true, assert_false, assert_is, assert_is_none


from zarr.storage import DirectoryStore, ZipStore, init_array, init_group
from zarr.core import Array
from zarr.errors import PermissionError
from zarr.compat import PY2
from zarr.util import buffer_size
from zarr.codecs import Delta, FixedScaleOffset, Zlib,\
    Blosc, BZ2


class TestArray(unittest.TestCase):

    def test_array_init(self):

        # normal initialization
        store = dict()
        init_array(store, shape=100, chunks=10)
        a = Array(store)
        assert_is_instance(a, Array)
        eq((100,), a.shape)
        eq((10,), a.chunks)
        eq('', a.path)
        assert_is_none(a.name)
        assert_is(store, a.store)

        # initialize at path
        store = dict()
        init_array(store, shape=100, chunks=10, path='foo/bar')
        a = Array(store, path='foo/bar')
        assert_is_instance(a, Array)
        eq((100,), a.shape)
        eq((10,), a.chunks)
        eq('foo/bar', a.path)
        eq('/foo/bar', a.name)
        assert_is(store, a.store)

        # store not initialized
        store = dict()
        with assert_raises(KeyError):
            Array(store)

        # group is in the way
        store = dict()
        init_group(store, path='baz')
        with assert_raises(KeyError):
            Array(store, path='baz')

    def create_array(self, read_only=False, **kwargs):
        store = dict()
        kwargs.setdefault('compressor', Zlib(level=1))
        init_array(store, **kwargs)
        return Array(store, read_only=read_only)

    def test_nbytes_stored(self):

        # dict as store
        z = self.create_array(shape=1000, chunks=100)
        expect_nbytes_stored = sum(buffer_size(v) for v in z.store.values())
        eq(expect_nbytes_stored, z.nbytes_stored)
        z[:] = 42
        expect_nbytes_stored = sum(buffer_size(v) for v in z.store.values())
        eq(expect_nbytes_stored, z.nbytes_stored)

        # mess with store
        try:
            z.store[z._key_prefix + 'foo'] = list(range(10))
            eq(-1, z.nbytes_stored)
        except TypeError:
            pass

    def test_array_1d(self):
        a = np.arange(1050)
        z = self.create_array(shape=a.shape, chunks=100, dtype=a.dtype)

        # check properties
        eq(len(a), len(z))
        eq(a.ndim, z.ndim)
        eq(a.shape, z.shape)
        eq(a.dtype, z.dtype)
        eq((100,), z.chunks)
        eq(a.nbytes, z.nbytes)
        eq(11, z.nchunks)
        eq(0, z.nchunks_initialized)
        eq((11,), z.cdata_shape)

        # check empty
        b = z[:]
        assert_is_instance(b, np.ndarray)
        eq(a.shape, b.shape)
        eq(a.dtype, b.dtype)

        # check attributes
        z.attrs['foo'] = 'bar'
        eq('bar', z.attrs['foo'])

        # set data
        z[:] = a

        # check properties
        eq(a.nbytes, z.nbytes)
        eq(11, z.nchunks)
        eq(11, z.nchunks_initialized)

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
        # single item
        eq(a[0], z[0])
        eq(a[-1], z[-1])

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
        a = np.zeros(100)
        z = self.create_array(shape=a.shape, chunks=10, dtype=a.dtype)
        z[:] = a
        assert_array_equal(a, z[:])

        for value in -1, 0, 1, 10:
            print(value)
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
        eq(len(a), len(z))
        eq(a.ndim, z.ndim)
        eq(a.shape, z.shape)
        eq(a.dtype, z.dtype)
        eq((100, 2), z.chunks)
        eq(0, z.nchunks_initialized)
        eq((10, 5), z.cdata_shape)

        # set data
        z[:] = a

        # check properties
        eq(a.nbytes, z.nbytes)
        eq(50, z.nchunks_initialized)

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
        # single item
        assert_array_equal(a[0], z[0])
        assert_array_equal(a[-1], z[-1])
        eq(a[0, 0], z[0, 0])
        eq(a[-1, -1], z[-1, -1])

        # check partial assignment
        b = np.arange(10000, 20000).reshape((1000, 10))
        z[190:310, 3:7] = b[190:310, 3:7]
        assert_array_equal(a[:190], z[:190])
        assert_array_equal(a[:, :3], z[:, :3])
        assert_array_equal(b[190:310, 3:7], z[190:310, 3:7])
        assert_array_equal(a[310:], z[310:])
        assert_array_equal(a[:, 7:], z[:, 7:])

    def test_array_2d_partial(self):
        z = self.create_array(shape=(1000, 10), chunks=(100, 2), dtype='i4',
                              fill_value=0)

        # check partial assignment, single row
        c = np.arange(z.shape[1])
        z[0, :] = c
        with assert_raises(ValueError):
            # N.B., NumPy allows this, but we'll be strict for now
            z[2:3] = c
        with assert_raises(ValueError):
            # N.B., NumPy allows this, but we'll be strict for now
            z[-1:] = c
        z[2:3] = c[None, :]
        z[-1:] = c[None, :]
        assert_array_equal(c, z[0, :])
        assert_array_equal(c, z[2, :])
        assert_array_equal(c, z[-1, :])

        # check partial assignment, single column
        d = np.arange(z.shape[0])
        z[:, 0] = d
        with assert_raises(ValueError):
            z[:, 2:3] = d
        with assert_raises(ValueError):
            z[:, -1:] = d
        z[:, 2:3] = d[:, None]
        z[:, -1:] = d[:, None]
        assert_array_equal(d, z[:, 0])
        assert_array_equal(d, z[:, 2])
        assert_array_equal(d, z[:, -1])

        # check single item assignment
        z[0, 0] = -1
        z[2, 2] = -1
        z[-1, -1] = -1
        eq(-1, z[0, 0])
        eq(-1, z[2, 2])
        eq(-1, z[-1, -1])

    def test_array_order(self):

        # 1D
        a = np.arange(1050)
        for order in 'C', 'F':
            z = self.create_array(shape=a.shape, chunks=100, dtype=a.dtype,
                                  order=order)
            eq(order, z.order)
            if order == 'F':
                assert_true(z[:].flags.f_contiguous)
            else:
                assert_true(z[:].flags.c_contiguous)
            z[:] = a
            assert_array_equal(a, z[:])

        # 2D
        a = np.arange(10000).reshape((100, 100))
        for order in 'C', 'F':
            z = self.create_array(shape=a.shape, chunks=(10, 10),
                                  dtype=a.dtype, order=order)
            eq(order, z.order)
            if order == 'F':
                assert_true(z[:].flags.f_contiguous)
            else:
                assert_true(z[:].flags.c_contiguous)
            z[:] = a
            actual = z[:]
            assert_array_equal(a, actual)

    def test_setitem_data_not_shared(self):
        # check that data don't end up being shared with another array
        # https://github.com/alimanfoo/zarr/issues/79
        z = self.create_array(shape=20, chunks=10, dtype='i4')
        a = np.arange(20, dtype='i4')
        z[:] = a
        assert_array_equal(z[:], np.arange(20, dtype='i4'))
        a[:] = 0
        assert_array_equal(z[:], np.arange(20, dtype='i4'))

    def test_resize_1d(self):

        z = self.create_array(shape=105, chunks=10, dtype='i4',
                              fill_value=0)
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

        # via shape setter
        z.shape = (105,)
        eq((105,), z.shape)
        eq((105,), z[:].shape)

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

        # via shape setter
        z.shape = (105, 105)
        eq((105, 105), z.shape)
        eq((105, 105), z[:].shape)

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
        actual = z[:]
        assert_array_equal(a, actual)

        b = np.arange(105*105, 2*105*105, dtype='i4').reshape((105, 105))
        e = np.append(a, b, axis=0)
        z.append(b)
        eq(e.shape, z.shape)
        eq(e.dtype, z.dtype)
        eq((10, 10), z.chunks)
        actual = z[:]
        assert_array_equal(e, actual)

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

    def test_append_bad_shape(self):
        a = np.arange(100)
        z = self.create_array(shape=a.shape, chunks=10, dtype=a.dtype)
        z[:] = a
        b = a.reshape(10, 10)
        with assert_raises(ValueError):
            z.append(b)

    def test_read_only(self):

        z = self.create_array(shape=1000, chunks=100)
        assert_false(z.read_only)

        z = self.create_array(shape=1000, chunks=100, read_only=True)
        assert_true(z.read_only)
        with assert_raises(PermissionError):
            z[:] = 42
        with assert_raises(PermissionError):
            z.resize(2000)
        with assert_raises(PermissionError):
            z.append(np.arange(1000))

    def test_pickle(self):

        z = self.create_array(shape=1000, chunks=100, dtype=int)
        z[:] = np.random.randint(0, 1000, 1000)
        z2 = pickle.loads(pickle.dumps(z))
        eq(z.shape, z2.shape)
        eq(z.chunks, z2.chunks)
        eq(z.dtype, z2.dtype)
        if z.compressor:
            eq(z.compressor.get_config(), z2.compressor.get_config())
        eq(z.fill_value, z2.fill_value)
        assert_array_equal(z[:], z2[:])

    def test_repr(self):
        if not PY2:
            z = self.create_array(shape=100, chunks=10, dtype='f4')
            expect = """Array((100,), float32, chunks=(10,), order=C)
  nbytes: 400; nbytes_stored: 245; ratio: 1.6; initialized: 0/10
  compressor: Zlib(level=1)
  store: dict
"""
            actual = repr(z)
            for l1, l2 in zip(expect.split('\n'), actual.split('\n')):
                eq(l1, l2)

    def test_np_ufuncs(self):
        z = self.create_array(shape=(100, 100), chunks=(10, 10))
        a = np.arange(10000).reshape(100, 100)
        z[:] = a

        eq(np.sum(a), np.sum(z))
        assert_array_equal(np.sum(a, axis=0), np.sum(z, axis=0))
        eq(np.mean(a), np.mean(z))
        assert_array_equal(np.mean(a, axis=1), np.mean(z, axis=1))
        condition = np.random.randint(0, 2, size=100, dtype=bool)
        assert_array_equal(np.compress(condition, a, axis=0),
                           np.compress(condition, z, axis=0))
        indices = np.random.choice(100, size=50, replace=True)
        assert_array_equal(np.take(a, indices, axis=1),
                           np.take(z, indices, axis=1))

        # use zarr array as indices or condition
        zc = self.create_array(shape=condition.shape, dtype=condition.dtype,
                               chunks=10, filters=None)
        zc[:] = condition
        assert_array_equal(np.compress(condition, a, axis=0),
                           np.compress(zc, a, axis=0))
        zi = self.create_array(shape=indices.shape, dtype=indices.dtype,
                               chunks=10, filters=None)
        zi[:] = indices
        # this triggers __array__() call with dtype argument
        assert_array_equal(np.take(a, indices, axis=1),
                           np.take(a, zi, axis=1))


class TestArrayWithPath(TestArray):

    @staticmethod
    def create_array(read_only=False, **kwargs):
        store = dict()
        init_array(store, path='foo/bar', **kwargs)
        return Array(store, path='foo/bar', read_only=read_only)

    def test_nbytes_stored(self):

        # dict as store
        z = self.create_array(shape=1000, chunks=100)
        expect_nbytes_stored = sum(buffer_size(v)
                                   for k, v in z.store.items()
                                   if k.startswith('foo/bar/'))
        eq(expect_nbytes_stored, z.nbytes_stored)
        z[:] = 42
        expect_nbytes_stored = sum(buffer_size(v)
                                   for k, v in z.store.items()
                                   if k.startswith('foo/bar/'))
        eq(expect_nbytes_stored, z.nbytes_stored)

        # mess with store
        z.store[z._key_prefix + 'foo'] = list(range(10))
        eq(-1, z.nbytes_stored)

    def test_repr(self):
        if not PY2:
            z = self.create_array(shape=100, chunks=10, dtype='f4')
            # flake8: noqa
            expect = """Array(/foo/bar, (100,), float32, chunks=(10,), order=C)
  nbytes: 400; nbytes_stored: 293; ratio: 1.4; initialized: 0/10
  compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
  store: dict
"""
            actual = repr(z)
            for l1, l2 in zip(expect.split('\n'), actual.split('\n')):
                eq(l1, l2)


class TestArrayWithChunkStore(TestArray):

    @staticmethod
    def create_array(read_only=False, **kwargs):
        store = dict()
        # separate chunk store
        chunk_store = dict()
        init_array(store, chunk_store=chunk_store, **kwargs)
        return Array(store, read_only=read_only, chunk_store=chunk_store)

    def test_nbytes_stored(self):

        z = self.create_array(shape=1000, chunks=100)
        expect_nbytes_stored = sum(buffer_size(v) for v in z.store.values())
        expect_nbytes_stored += sum(buffer_size(v)
                                    for v in z.chunk_store.values())
        eq(expect_nbytes_stored, z.nbytes_stored)
        z[:] = 42
        expect_nbytes_stored = sum(buffer_size(v) for v in z.store.values())
        expect_nbytes_stored += sum(buffer_size(v)
                                    for v in z.chunk_store.values())
        eq(expect_nbytes_stored, z.nbytes_stored)

        # mess with store
        z.chunk_store[z._key_prefix + 'foo'] = list(range(10))
        eq(-1, z.nbytes_stored)

    def test_repr(self):
        if not PY2:
            z = self.create_array(shape=100, chunks=10, dtype='f4')
            # flake8: noqa
            expect = """Array((100,), float32, chunks=(10,), order=C)
  nbytes: 400; nbytes_stored: 293; ratio: 1.4; initialized: 0/10
  compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
  store: dict; chunk_store: dict
"""
            actual = repr(z)
            for l1, l2 in zip(expect.split('\n'), actual.split('\n')):
                eq(l1, l2)


class TestArrayWithDirectoryStore(TestArray):

    @staticmethod
    def create_array(read_only=False, **kwargs):
        path = mkdtemp()
        atexit.register(shutil.rmtree, path)
        store = DirectoryStore(path)
        kwargs.setdefault('compressor', Zlib(1))
        init_array(store, **kwargs)
        return Array(store, read_only=read_only)

    def test_nbytes_stored(self):

        # dict as store
        z = self.create_array(shape=1000, chunks=100)
        expect_nbytes_stored = sum(buffer_size(v) for v in z.store.values())
        eq(expect_nbytes_stored, z.nbytes_stored)
        z[:] = 42
        expect_nbytes_stored = sum(buffer_size(v) for v in z.store.values())
        eq(expect_nbytes_stored, z.nbytes_stored)

    def test_repr(self):
        if not PY2:
            z = self.create_array(shape=100, chunks=10, dtype='f4')
            # flake8: noqa
            expect = """Array((100,), float32, chunks=(10,), order=C)
  nbytes: 400; nbytes_stored: 245; ratio: 1.6; initialized: 0/10
  compressor: Zlib(level=1)
  store: DirectoryStore
"""
            actual = repr(z)
            for l1, l2 in zip(expect.split('\n'), actual.split('\n')):
                eq(l1, l2)


class TestArrayWithNoCompressor(TestArray):

    def create_array(self, read_only=False, **kwargs):
        store = dict()
        kwargs.setdefault('compressor', None)
        init_array(store, **kwargs)
        return Array(store, read_only=read_only)

    def test_repr(self):
        if not PY2:
            z = self.create_array(shape=100, chunks=10, dtype='f4')
            expect = """Array((100,), float32, chunks=(10,), order=C)
  nbytes: 400; nbytes_stored: 201; ratio: 2.0; initialized: 0/10
  store: dict
"""
            actual = repr(z)
            for l1, l2 in zip(expect.split('\n'), actual.split('\n')):
                eq(l1, l2)


class TestArrayWithBZ2Compressor(TestArray):

    def create_array(self, read_only=False, **kwargs):
        store = dict()
        compressor = BZ2(level=1)
        kwargs.setdefault('compressor', compressor)
        init_array(store, **kwargs)
        return Array(store, read_only=read_only)

    def test_repr(self):
        if not PY2:
            z = self.create_array(shape=100, chunks=10, dtype='f4')
            expect = """Array((100,), float32, chunks=(10,), order=C)
  nbytes: 400; nbytes_stored: 244; ratio: 1.6; initialized: 0/10
  compressor: BZ2(level=1)
  store: dict
"""
            actual = repr(z)
            for l1, l2 in zip(expect.split('\n'), actual.split('\n')):
                eq(l1, l2)


class TestArrayWithBloscCompressor(TestArray):

    def create_array(self, read_only=False, **kwargs):
        store = dict()
        compressor = Blosc(cname='zstd', clevel=1, shuffle=1)
        kwargs.setdefault('compressor', compressor)
        init_array(store, **kwargs)
        return Array(store, read_only=read_only)

    def test_repr(self):
        if not PY2:
            z = self.create_array(shape=100, chunks=10, dtype='f4')
            expect = """Array((100,), float32, chunks=(10,), order=C)
  nbytes: 400; nbytes_stored: 294; ratio: 1.4; initialized: 0/10
  compressor: Blosc(cname='zstd', clevel=1, shuffle=1)
  store: dict
"""
            actual = repr(z)
            for l1, l2 in zip(expect.split('\n'), actual.split('\n')):
                eq(l1, l2)


if not PY2:

    from zarr.codecs import LZMA

    class TestArrayWithLZMACompressor(TestArray):

        def create_array(self, read_only=False, **kwargs):
            store = dict()
            compressor = LZMA(preset=1)
            kwargs.setdefault('compressor', compressor)
            init_array(store, **kwargs)
            return Array(store, read_only=read_only)

        def test_repr(self):
            z = self.create_array(shape=100, chunks=10, dtype='f4')
            expect = """Array((100,), float32, chunks=(10,), order=C)
  nbytes: 400; nbytes_stored: 313; ratio: 1.3; initialized: 0/10
  compressor: LZMA(format=1, check=-1, preset=1, filters=None)
  store: dict
"""
            actual = repr(z)
            for l1, l2 in zip(expect.split('\n'), actual.split('\n')):
                eq(l1, l2)


class TestArrayWithFilters(TestArray):

    @staticmethod
    def create_array(read_only=False, **kwargs):
        store = dict()
        dtype = kwargs.get('dtype', None)
        filters = [
            Delta(dtype=dtype),
            FixedScaleOffset(dtype=dtype, scale=1, offset=0),
        ]
        kwargs.setdefault('filters', filters)
        compressor = Zlib(1)
        kwargs.setdefault('compressor', compressor)
        init_array(store, **kwargs)
        return Array(store, read_only=read_only)

    def test_repr(self):
        if not PY2:
            z = self.create_array(shape=100, chunks=10, dtype='f4')
            # flake8: noqa
            expect = """Array((100,), float32, chunks=(10,), order=C)
  nbytes: 400; nbytes_stored: 515; ratio: 0.8; initialized: 0/10
  filters: Delta(dtype=float32)
           FixedScaleOffset(scale=1, offset=0, dtype=float32)
  compressor: Zlib(level=1)
  store: dict
"""
            actual = repr(z)
            for l1, l2 in zip(expect.split('\n'), actual.split('\n')):
                eq(l1, l2)


# custom store, does not support getsize()
class CustomMapping(object):

    def __init__(self):
        self.inner = dict()

    def keys(self):
        return self.inner.keys()

    def __getitem__(self, item):
        return self.inner[item]

    def __setitem__(self, item, value):
        self.inner[item] = value

    def __delitem__(self, key):
        del self.inner[key]

    def __contains__(self, item):
        return item in self.inner


class TestArrayWithCustomMapping(TestArray):

    @staticmethod
    def create_array(read_only=False, **kwargs):
        store = CustomMapping()
        kwargs.setdefault('compressor', Zlib(1))
        init_array(store, **kwargs)
        return Array(store, read_only=read_only)

    def test_nbytes_stored(self):
        z = self.create_array(shape=1000, chunks=100)
        eq(-1, z.nbytes_stored)
        z[:] = 42
        eq(-1, z.nbytes_stored)

    def test_repr(self):
        if not PY2:
            z = self.create_array(shape=100, chunks=10, dtype='f4')
            # flake8: noqa
            expect = """Array((100,), float32, chunks=(10,), order=C)
  nbytes: 400; initialized: 0/10
  compressor: Zlib(level=1)
  store: CustomMapping
"""
            actual = repr(z)
            for l1, l2 in zip(expect.split('\n'), actual.split('\n')):
                eq(l1, l2)


class TestArrayNoCacheMetadata(TestArray):

    @staticmethod
    def create_array(read_only=False, **kwargs):
        store = dict()
        kwargs.setdefault('compressor', Zlib(level=1))
        init_array(store, **kwargs)
        return Array(store, read_only=read_only, cache_metadata=False)

    def test_cache_metadata(self):
        a1 = self.create_array(shape=100, chunks=10, dtype='i1')
        a2 = Array(a1.store, cache_metadata=True)
        eq(a1.shape, a2.shape)
        eq(a1.size, a2.size)
        eq(a1.nbytes, a2.nbytes)
        eq(a1.nchunks, a2.nchunks)

        a2.resize(200)
        eq((200,), a2.shape)
        eq(200, a2.size)
        eq(200, a2.nbytes)
        eq(20, a2.nchunks)
        eq(a1.shape, a2.shape)
        eq(a1.size, a2.size)
        eq(a1.nbytes, a2.nbytes)
        eq(a1.nchunks, a2.nchunks)

        a2.append(np.zeros(100))
        eq((300,), a2.shape)
        eq(300, a2.size)
        eq(300, a2.nbytes)
        eq(30, a2.nchunks)
        eq(a1.shape, a2.shape)
        eq(a1.size, a2.size)
        eq(a1.nbytes, a2.nbytes)
        eq(a1.nchunks, a2.nchunks)

        a1.resize(400)
        eq((400,), a1.shape)
        eq(400, a1.size)
        eq(400, a1.nbytes)
        eq(40, a1.nchunks)
        eq((300,), a2.shape)
        eq(300, a2.size)
        eq(300, a2.nbytes)
        eq(30, a2.nchunks)
