# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import unittest
from tempfile import mkdtemp
import atexit
import shutil
import pickle
import os


import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import eq_ as eq, assert_is_instance, \
    assert_raises, assert_true, assert_false, assert_is, assert_is_none


from zarr.storage import DirectoryStore, ZipStore, init_array, init_group
from zarr.core import Array
from zarr.errors import ReadOnlyError
from zarr.compat import PY2
from zarr.util import buffer_size
from zarr.filters import DeltaFilter, FixedScaleOffsetFilter


compression_configs = [
    ('none', None),
    ('zlib', None),
    ('bz2', None),
    ('blosc', None)
]
if not PY2:
    compression_configs.append(('lzma', None))


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
        with assert_raises(ValueError):
            Array(store)

        # group is in the way
        store = dict()
        init_group(store, path='baz')
        with assert_raises(ValueError):
            Array(store, path='baz')

    def create_array(self, store=None, path=None, read_only=False,
                     chunk_store=None, **kwargs):
        if store is None:
            store = dict()
        init_array(store, path=path, chunk_store=chunk_store, **kwargs)
        return Array(store, path=path, read_only=read_only,
                     chunk_store=chunk_store)

    def test_nbytes_stored(self):

        # custom store, does not implement getsize()
        class CustomMapping(object):
            def __init__(self):
                self.inner = dict()

            def __getitem__(self, item):
                return self.inner[item]

            def __setitem__(self, item, value):
                self.inner[item] = value

            def __contains__(self, item):
                return item in self.inner

        store = CustomMapping()
        z = self.create_array(store=store, shape=1000, chunks=100)
        eq(-1, z.nbytes_stored)
        z[:] = 42
        eq(-1, z.nbytes_stored)

        store = dict()
        chunk_store = CustomMapping()
        z = self.create_array(store=store, chunk_store=chunk_store,
                              shape=1000, chunks=100)
        eq(-1, z.nbytes_stored)
        z[:] = 42
        eq(-1, z.nbytes_stored)

        # dict as store
        store = dict()
        z = self.create_array(store=store, shape=1000, chunks=100)
        expect_nbytes_stored = sum(buffer_size(v) for v in z.store.values())
        if z.store != z.chunk_store:
            expect_nbytes_stored += sum(buffer_size(v) for v in
                                        z.chunk_store.values())
        eq(expect_nbytes_stored, z.nbytes_stored)
        z[:] = 42
        expect_nbytes_stored = sum(buffer_size(v) for v in z.store.values())
        if z.store != z.chunk_store:
            expect_nbytes_stored += sum(buffer_size(v) for v in
                                        z.chunk_store.values())
        eq(expect_nbytes_stored, z.nbytes_stored)
        # mess with store
        store[z._key_prefix + 'foo'] = list(range(10))
        eq(-1, z.nbytes_stored)

        # for comparison
        z = self.create_array(store=dict(), shape=1000, chunks=100,
                              compression='zlib', compression_opts=1)
        z[:] = 42

        # DirectoryStore
        path = mkdtemp()
        atexit.register(shutil.rmtree, path)
        store = DirectoryStore(path)
        zz = self.create_array(store=store, shape=1000, chunks=100,
                               compression='zlib', compression_opts=1)
        zz[:] = 42
        eq(z.nbytes_stored, zz.nbytes_stored)

        # ZipStore
        if os.path.exists('test.zip'):
            os.remove('test.zip')
        store = ZipStore('test.zip')
        zz = self.create_array(store=store, shape=1000, chunks=100,
                               compression='zlib', compression_opts=1)
        zz[:] = 42
        eq(z.nbytes_stored, zz.nbytes_stored)

    def test_array_1d(self):
        for compression, compression_opts in compression_configs:
            print(compression, compression_opts)

            a = np.arange(1050)
            z = self.create_array(shape=a.shape, chunks=100, dtype=a.dtype,
                                  compression=compression,
                                  compression_opts=compression_opts)

            # check properties
            eq(len(a), len(z))
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

            # check attributes
            z.attrs['foo'] = 'bar'
            eq('bar', z.attrs['foo'])

            # set data
            z[:] = a

            # check properties
            eq(a.nbytes, z.nbytes)
            expect_nbytes_stored = sum(buffer_size(v) for v in z.store.values())
            if z.store != z.chunk_store:
                expect_nbytes_stored += sum(buffer_size(v) for v in
                                            z.chunk_store.values())
            eq(expect_nbytes_stored, z.nbytes_stored)
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
        for compression, compression_opts in compression_configs:
            print(compression, compression_opts)

            for fill_value in -1, 0, 1, 10:

                a = np.arange(1050)
                f = np.empty_like(a)
                f.fill(fill_value)
                z = self.create_array(shape=a.shape, chunks=100, dtype=a.dtype,
                                      fill_value=fill_value,
                                      compression=compression,
                                      compression_opts=compression_opts)
                z[190:310] = a[190:310]

                assert_array_equal(f[:190], z[:190])
                assert_array_equal(a[190:310], z[190:310])
                assert_array_equal(f[310:], z[310:])

    def test_array_1d_set_scalar(self):
        for compression, compression_opts in compression_configs:
            print(compression, compression_opts)

            # setup
            a = np.zeros(100)
            z = self.create_array(shape=a.shape, chunks=10, dtype=a.dtype,
                                  compression=compression,
                                  compression_opts=compression_opts)
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
        for compression, compression_opts in compression_configs:
            print(compression, compression_opts)

            a = np.arange(10000).reshape((1000, 10))
            z = self.create_array(shape=a.shape, chunks=(100, 2),
                                  dtype=a.dtype, compression=compression,
                                  compression_opts=compression_opts)

            # check properties
            eq(len(a), len(z))
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
            expect_nbytes_stored = sum(buffer_size(v) for v in z.store.values())
            if z.store != z.chunk_store:
                expect_nbytes_stored += sum(buffer_size(v) for v in
                                            z.chunk_store.values())
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
        for compression, compression_opts in compression_configs:
            print(compression, compression_opts)

            z = self.create_array(shape=(1000, 10), chunks=(100, 2), dtype='i4',
                                  fill_value=0, compression=compression,
                                  compression_opts=compression_opts)

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
        for compression, compression_opts in compression_configs:
            print(compression, compression_opts)

            # 1D
            a = np.arange(1050)
            for order in 'C', 'F':
                z = self.create_array(shape=a.shape, chunks=100, dtype=a.dtype,
                                      order=order, compression=compression,
                                      compression_opts=compression_opts)
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
                                      dtype=a.dtype, order=order,
                                      compression=compression,
                                      compression_opts=compression_opts)
                eq(order, z.order)
                if order == 'F':
                    assert_true(z[:].flags.f_contiguous)
                else:
                    assert_true(z[:].flags.c_contiguous)
                z[:] = a
                actual = z[:]
                assert_array_equal(a, actual)

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
        with assert_raises(ReadOnlyError):
            z[:] = 42
        with assert_raises(ReadOnlyError):
            z.resize(2000)
        with assert_raises(ReadOnlyError):
            z.append(np.arange(1000))

    def test_pickle(self):

        memory_store = dict()
        path = mkdtemp()
        atexit.register(shutil.rmtree, path)
        directory_store = DirectoryStore(path)

        for store in memory_store, directory_store:
            z = self.create_array(shape=1000, chunks=100, dtype=int,
                                  store=store)
            z[:] = np.random.randint(0, 1000, 1000)
            z2 = pickle.loads(pickle.dumps(z))
            eq(z.shape, z2.shape)
            eq(z.chunks, z2.chunks)
            eq(z.dtype, z2.dtype)
            eq(z.compression, z2.compression)
            eq(z.compression_opts, z2.compression_opts)
            eq(z.fill_value, z2.fill_value)
            assert_array_equal(z[:], z2[:])

    def test_repr(self):
        if not PY2:

            z = self.create_array(shape=100, chunks=10, dtype='f4',
                                  compression='zlib', compression_opts=1)
            expect = """zarr.core.Array((100,), float32, chunks=(10,), order=C)
  compression: zlib; compression_opts: 1
  nbytes: 400; nbytes_stored: 231; ratio: 1.7; initialized: 0/10
  store: builtins.dict
"""
            actual = repr(z)
            for l1, l2 in zip(expect.split('\n'), actual.split('\n')):
                eq(l1, l2)


class TestArrayWithPath(TestArray):

    @staticmethod
    def create_array(store=None, read_only=False, chunk_store=None, **kwargs):
        if store is None:
            store = dict()
        init_array(store, path='foo/bar', chunk_store=chunk_store, **kwargs)
        return Array(store, path='foo/bar', read_only=read_only,
                     chunk_store=chunk_store)

    def test_repr(self):
        if not PY2:

            z = self.create_array(shape=100, chunks=10, dtype='f4',
                                  compression='zlib', compression_opts=1)
            # flake8: noqa
            expect = """zarr.core.Array(/foo/bar, (100,), float32, chunks=(10,), order=C)
  compression: zlib; compression_opts: 1
  nbytes: 400; nbytes_stored: 231; ratio: 1.7; initialized: 0/10
  store: builtins.dict
"""
            actual = repr(z)
            for l1, l2 in zip(expect.split('\n'), actual.split('\n')):
                eq(l1, l2)


class TestArrayWithChunkStore(TestArray):

    @staticmethod
    def create_array(store=None, read_only=False, chunk_store=None, **kwargs):
        if store is None:
            store = dict()
        if chunk_store is None:
            # separate chunk store
            chunk_store = dict()
        init_array(store, path='foo/bar', chunk_store=chunk_store, **kwargs)
        return Array(store, path='foo/bar', read_only=read_only,
                     chunk_store=chunk_store)

    def test_repr(self):
        if not PY2:

            z = self.create_array(shape=100, chunks=10, dtype='f4',
                                  compression='zlib', compression_opts=1)
            # flake8: noqa
            expect = """zarr.core.Array(/foo/bar, (100,), float32, chunks=(10,), order=C)
  compression: zlib; compression_opts: 1
  nbytes: 400; nbytes_stored: 231; ratio: 1.7; initialized: 0/10
  store: builtins.dict
  chunk_store: builtins.dict
"""
            actual = repr(z)
            for l1, l2 in zip(expect.split('\n'), actual.split('\n')):
                eq(l1, l2)


class TestArrayWithFilters(TestArray):

    @staticmethod
    def create_array(store=None, read_only=False, chunk_store=None, **kwargs):
        if store is None:
            store = dict()
        if chunk_store is None:
            chunk_store = store
        dtype = kwargs.get('dtype', None)
        filters = [
            DeltaFilter(dtype=dtype),
            FixedScaleOffsetFilter(dtype=dtype, scale=1, offset=0),
        ]
        kwargs.setdefault('filters', filters)
        init_array(store, chunk_store=chunk_store, **kwargs)
        return Array(store, read_only=read_only,
                     chunk_store=chunk_store)

    def test_repr(self):
        if not PY2:

            z = self.create_array(shape=100, chunks=10, dtype='f4',
                                  compression='zlib', compression_opts=1)
            # flake8: noqa
            expect = """zarr.core.Array((100,), float32, chunks=(10,), order=C)
  compression: zlib; compression_opts: 1
  nbytes: 400; nbytes_stored: 505; ratio: 0.8; initialized: 0/10
  filters: delta, fixedscaleoffset
  store: builtins.dict
  chunk_store: builtins.dict
"""
            actual = repr(z)
            for l1, l2 in zip(expect.split('\n'), actual.split('\n')):
                eq(l1, l2)
