# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import unittest
from tempfile import mkdtemp
import atexit
import shutil
import pickle


import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import eq_ as eq, assert_is_instance, \
    assert_raises, assert_true, assert_false, assert_is, assert_is_none


from zarr.storage import DirectoryStore, init_array, init_group, NestedDirectoryStore
from zarr.core import Array
from zarr.errors import PermissionError
from zarr.compat import PY2
from zarr.util import buffer_size
from numcodecs import Delta, FixedScaleOffset, Zlib, Blosc, BZ2


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
        assert_array_equal(a[:10, ...], z[:10, ...])
        assert_array_equal(a[10:20, ...], z[10:20, ...])
        assert_array_equal(a[-10:, ...], z[-10:, ...])
        assert_array_equal(a[..., :10], z[..., :10])
        assert_array_equal(a[..., 10:20], z[..., 10:20])
        assert_array_equal(a[..., -10:], z[..., -10:])
        # ...across chunk boundaries...
        assert_array_equal(a[:110], z[:110])
        assert_array_equal(a[190:310], z[190:310])
        assert_array_equal(a[-110:], z[-110:])
        # single item
        eq(a[0], z[0])
        eq(a[-1], z[-1])
        # unusual integer items
        eq(a[42], z[np.int64(42)])
        eq(a[42], z[np.int32(42)])
        eq(a[42], z[np.uint64(42)])
        eq(a[42], z[np.uint32(42)])
        # too many indices
        with assert_raises(IndexError):
            z[:, :]
        with assert_raises(IndexError):
            z[0, :]
        with assert_raises(IndexError):
            z[:, 0]
        with assert_raises(IndexError):
            z[0, 0]
        # only single ellipsis allowed
        with assert_raises(IndexError):
            z[..., ...]

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
        # test setting the contents of an array with a scalar value

        # setup
        a = np.zeros(100)
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

        # check array-like
        assert_array_equal(a, np.array(z))

        # check slicing

        # total slice
        assert_array_equal(a, z[:])
        assert_array_equal(a, z[...])
        # noinspection PyTypeChecker
        assert_array_equal(a, z[slice(None)])

        # slice first dimension
        assert_array_equal(a[:10], z[:10])
        assert_array_equal(a[10:20], z[10:20])
        assert_array_equal(a[-10:], z[-10:])
        assert_array_equal(a[:10, :], z[:10, :])
        assert_array_equal(a[10:20, :], z[10:20, :])
        assert_array_equal(a[-10:, :], z[-10:, :])
        assert_array_equal(a[:10, ...], z[:10, ...])
        assert_array_equal(a[10:20, ...], z[10:20, ...])
        assert_array_equal(a[-10:, ...], z[-10:, ...])
        assert_array_equal(a[:10, :, ...], z[:10, :, ...])
        assert_array_equal(a[10:20, :, ...], z[10:20, :, ...])
        assert_array_equal(a[-10:, :, ...], z[-10:, :, ...])

        # slice second dimension
        assert_array_equal(a[:, :2], z[:, :2])
        assert_array_equal(a[:, 2:4], z[:, 2:4])
        assert_array_equal(a[:, -2:], z[:, -2:])
        assert_array_equal(a[..., :2], z[..., :2])
        assert_array_equal(a[..., 2:4], z[..., 2:4])
        assert_array_equal(a[..., -2:], z[..., -2:])
        assert_array_equal(a[:, ..., :2], z[:, ..., :2])
        assert_array_equal(a[:, ..., 2:4], z[:, ..., 2:4])
        assert_array_equal(a[:, ..., -2:], z[:, ..., -2:])

        # slice both dimensions
        assert_array_equal(a[:10, :2], z[:10, :2])
        assert_array_equal(a[10:20, 2:4], z[10:20, 2:4])
        assert_array_equal(a[-10:, -2:], z[-10:, -2:])

        # slicing across chunk boundaries
        assert_array_equal(a[:110], z[:110])
        assert_array_equal(a[190:310], z[190:310])
        assert_array_equal(a[-110:], z[-110:])
        assert_array_equal(a[:110, :], z[:110, :])
        assert_array_equal(a[190:310, :], z[190:310, :])
        assert_array_equal(a[-110:, :], z[-110:, :])
        assert_array_equal(a[:, :3], z[:, :3])
        assert_array_equal(a[:, 3:7], z[:, 3:7])
        assert_array_equal(a[:, -3:], z[:, -3:])
        assert_array_equal(a[:110, :3], z[:110, :3])
        assert_array_equal(a[190:310, 3:7], z[190:310, 3:7])
        assert_array_equal(a[-110:, -3:], z[-110:, -3:])

        # single row/col/item
        assert_array_equal(a[0], z[0])
        assert_array_equal(a[-1], z[-1])
        assert_array_equal(a[:, 0], z[:, 0])
        assert_array_equal(a[:, -1], z[:, -1])
        eq(a[0, 0], z[0, 0])
        eq(a[-1, -1], z[-1, -1])

        # too many indices
        with assert_raises(IndexError):
            z[:, :, :]
        with assert_raises(IndexError):
            z[0, :, :]
        with assert_raises(IndexError):
            z[:, 0, :]
        with assert_raises(IndexError):
            z[:, :, 0]
        with assert_raises(IndexError):
            z[0, 0, 0]
        # only single ellipsis allowed
        with assert_raises(IndexError):
            z[..., ...]

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

    def test_0len_dim_1d(self):
        # Test behaviour for 1D array with zero-length dimension.

        z = self.create_array(shape=0, fill_value=0)
        a = np.zeros(0)
        eq(a.ndim, z.ndim)
        eq(a.shape, z.shape)
        eq(a.dtype, z.dtype)
        eq(a.size, z.size)
        eq(0, z.nchunks)

        # cannot make a good decision when auto-chunking if a dimension has zero length,
        # fall back to 1 for now
        eq((1,), z.chunks)

        # check __getitem__
        assert_is_instance(z[:], np.ndarray)
        assert_array_equal(a, np.array(z))
        assert_array_equal(a, z[:])
        assert_array_equal(a, z[...])
        assert_array_equal(a[0:0], z[0:0])
        with assert_raises(IndexError):
            z[0]

        # check __setitem__
        # these should succeed but do nothing
        z[:] = 42
        z[...] = 42
        # this should error
        with assert_raises(IndexError):
            z[0] = 42

    def test_0len_dim_2d(self):
        # Test behavioud for 2D array with a zero-length dimension.

        z = self.create_array(shape=(10, 0), fill_value=0)
        a = np.zeros((10, 0))
        eq(a.ndim, z.ndim)
        eq(a.shape, z.shape)
        eq(a.dtype, z.dtype)
        eq(a.size, z.size)
        eq(0, z.nchunks)

        # cannot make a good decision when auto-chunking if a dimension has zero length,
        # fall back to 1 for now
        eq((10, 1), z.chunks)

        # check __getitem__
        assert_is_instance(z[:], np.ndarray)
        assert_array_equal(a, np.array(z))
        assert_array_equal(a, z[:])
        assert_array_equal(a, z[...])
        assert_array_equal(a[0], z[0])
        assert_array_equal(a[0, 0:0], z[0, 0:0])
        assert_array_equal(a[0, :], z[0, :])
        assert_array_equal(a[0, 0:0], z[0, 0:0])
        with assert_raises(IndexError):
            z[:, 0]

        # check __setitem__
        # these should succeed but do nothing
        z[:] = 42
        z[...] = 42
        z[0, :] = 42
        # this should error
        with assert_raises(IndexError):
            z[:, 0] = 42

    def test_array_0d(self):
        # test behaviour for array with 0 dimensions

        # setup
        a = np.zeros(())
        z = self.create_array(shape=(), dtype=a.dtype, fill_value=0)

        # check properties
        eq(a.ndim, z.ndim)
        eq(a.shape, z.shape)
        eq(a.size, z.size)
        eq(a.dtype, z.dtype)
        eq(a.nbytes, z.nbytes)
        with assert_raises(TypeError):
            len(z)
        eq((), z.chunks)
        eq(1, z.nchunks)
        eq((1,), z.cdata_shape)
        # compressor always None - no point in compressing a single value
        assert_is_none(z.compressor)

        # check __getitem__
        b = z[...]
        assert_is_instance(b, np.ndarray)
        eq(a.shape, b.shape)
        eq(a.dtype, b.dtype)
        assert_array_equal(a, np.array(z))
        assert_array_equal(a, z[...])
        eq(a[()], z[()])
        with assert_raises(IndexError):
            z[0]
        with assert_raises(IndexError):
            z[:]

        # check __setitem__
        z[...] = 42
        eq(42, z[()])
        z[()] = 43
        eq(43, z[()])
        with assert_raises(IndexError):
            z[0] = 42
        with assert_raises(IndexError):
            z[:] = 42
        with assert_raises(ValueError):
            z[...] = np.array([1, 2, 3])

    def test_nchunks_initialized(self):

        z = self.create_array(shape=100, chunks=10)
        eq(0, z.nchunks_initialized)
        # manually put something into the store to confuse matters
        z.store['foo'] = b'bar'
        eq(0, z.nchunks_initialized)
        z[:] = 42
        eq(10, z.nchunks_initialized)

    def test_advanced_indexing_1d_bool(self):

        # setup
        a = np.arange(1050, dtype=int)
        z = self.create_array(shape=a.shape, chunks=100, dtype=a.dtype)
        z[:] = a

        np.random.seed(42)
        # test with different degrees of sparseness
        for p in 0.9, 0.5, 0.1, 0.01:
            ix = np.random.binomial(1, p, size=a.shape[0]).astype(bool)
            expect = a[ix]
            actual = z[ix]
            assert_array_equal(expect, actual)

        # TODO test errors

    def test_advanced_indexing_1d_int(self):

        # setup
        a = np.arange(1050, dtype=int)
        z = self.create_array(shape=a.shape, chunks=100, dtype=a.dtype)
        z[:] = a

        np.random.seed(42)
        # test with different degrees of sparseness
        for p in 0.9, 0.5, 0.1, 0.01:
            ix = np.random.binomial(1, p, size=a.shape[0]).astype(bool)
            ix = np.nonzero(ix)[0]
            expect = a[ix]
            actual = z[ix]
            assert_array_equal(expect, actual)

        # TODO test errors

    def test_advanced_indexing_2d_bool(self):

        # setup
        a = np.arange(10000, dtype=int).reshape(100, 100)
        z = self.create_array(shape=a.shape, chunks=(10, 10), dtype=a.dtype)
        z[:] = a

        np.random.seed(42)
        # test with different degrees of sparseness
        for p in 0.9, 0.5, 0.1, 0.01:
            ix0 = np.random.binomial(1, p, size=a.shape[0]).astype(bool)
            ix1 = np.random.binomial(1, p, size=a.shape[1]).astype(bool)

            # index both axes with bool array
            expect = a[np.ix_(ix0, ix1)]
            actual = z[ix0, ix1]
            assert_array_equal(expect, actual)

            # mixed indexing with bool array / slice
            expect = a[ix0, 15:35]
            actual = z[ix0, 15:35]
            assert_array_equal(expect, actual)
            expect = a[15:35, ix1]
            actual = z[15:35, ix1]
            assert_array_equal(expect, actual)

            # mixed indexing with bool array / single index
            expect = a[ix0, 42]
            actual = z[ix0, 42]
            assert_array_equal(expect, actual)
            expect = a[42, ix1]
            actual = z[42, ix1]
            assert_array_equal(expect, actual)

        # TODO test errors

    def test_advanced_indexing_2d_int(self):

        # setup
        a = np.arange(10000, dtype=int).reshape(100, 100)
        z = self.create_array(shape=a.shape, chunks=(10, 10), dtype=a.dtype)
        z[:] = a

        np.random.seed(42)
        # test with different degrees of sparseness
        for p in 0.9, 0.5, 0.1, 0.01:
            ix0 = np.random.binomial(1, p, size=a.shape[0]).astype(bool)
            ix0 = np.nonzero(ix0)[0]
            ix1 = np.random.binomial(1, p, size=a.shape[1]).astype(bool)
            ix1 = np.nonzero(ix1)[0]

            # index both axes with int array
            expect = a[np.ix_(ix0, ix1)]
            actual = z[ix0, ix1]
            assert_array_equal(expect, actual)

            # mixed indexing with int array / slice
            expect = a[ix0, 15:35]
            actual = z[ix0, 15:35]
            assert_array_equal(expect, actual)
            expect = a[15:35, ix1]
            actual = z[15:35, ix1]
            assert_array_equal(expect, actual)

            # mixed indexing with int array / single index
            expect = a[ix0, 42]
            actual = z[ix0, 42]
            assert_array_equal(expect, actual)
            expect = a[42, ix1]
            actual = z[42, ix1]
            assert_array_equal(expect, actual)

        # TODO test errors

    def test_advanced_indexing_3d_bool(self):

        # setup
        a = np.arange(1000000, dtype=int).reshape(100, 100, 100)
        z = self.create_array(shape=a.shape, chunks=(10, 10, 10), dtype=a.dtype)
        z[:] = a

        np.random.seed(42)
        # test with different degrees of sparseness
        for p in 0.9, 0.5, 0.1, 0.01:
            ix0 = np.random.binomial(1, p, size=a.shape[0]).astype(bool)
            ix1 = np.random.binomial(1, p, size=a.shape[1]).astype(bool)
            ix2 = np.random.binomial(1, p, size=a.shape[2]).astype(bool)

            # index all axes with bool array
            expect = a[np.ix_(ix0, ix1, ix2)]
            actual = z[ix0, ix1, ix2]
            assert_array_equal(expect, actual)

            # mixed indexing with single bool array / slices
            expect = a[ix0, 15:35, 25:45]
            actual = z[ix0, 15:35, 25:45]
            assert_array_equal(expect, actual)
            expect = a[15:35, ix1, 25:45]
            actual = z[15:35, ix1, 25:45]
            assert_array_equal(expect, actual)
            expect = a[15:35, 25:45, ix2]
            actual = z[15:35, 25:45, ix2]
            assert_array_equal(expect, actual)

            # mixed indexing with single bool array / single index
            expect = a[ix0, 42, 84]
            actual = z[ix0, 42, 84]
            assert_array_equal(expect, actual)
            expect = a[42, ix1, 84]
            actual = z[42, ix1, 84]
            assert_array_equal(expect, actual)
            expect = a[42, 84, ix2]
            actual = z[42, 84, ix2]
            assert_array_equal(expect, actual)

            # mixed indexing with single bool array / slice / single index
            expect = a[ix0, 15:35, 42]
            actual = z[ix0, 15:35, 42]
            assert_array_equal(expect, actual)
            expect = a[42, ix1, 25:45]
            actual = z[42, ix1, 25:45]
            assert_array_equal(expect, actual)
            expect = a[15:35, 42, ix2]
            actual = z[15:35, 42, ix2]
            assert_array_equal(expect, actual)

            # mixed indexing with two bool array / slice
            expect = a[np.ix_(ix0, ix1, range(25, 45))]
            actual = z[ix0, ix1, 25:45]
            assert_array_equal(expect, actual)
            expect = a[np.ix_(range(15, 35), ix1, ix2)]
            actual = z[15:35, ix1, ix2]
            assert_array_equal(expect, actual)
            expect = a[np.ix_(ix0, range(25, 45), ix2)]
            actual = z[ix0, 25:45, ix2]
            assert_array_equal(expect, actual)

            # mixed indexing with two bool array / integer
            expect = a[np.ix_(ix0, ix1, [42])].squeeze(axis=2)
            actual = z[ix0, ix1, 42]
            assert_array_equal(expect, actual)
            expect = a[np.ix_([42], ix1, ix2)].squeeze(axis=0)
            actual = z[42, ix1, ix2]
            assert_array_equal(expect, actual)
            expect = a[np.ix_(ix0, [42], ix2)].squeeze(axis=1)
            actual = z[ix0, 42, ix2]
            assert_array_equal(expect, actual)

    def test_advanced_indexing_3d_int(self):

        # setup
        a = np.arange(1000000, dtype=int).reshape(100, 100, 100)
        z = self.create_array(shape=a.shape, chunks=(10, 10, 10), dtype=a.dtype)
        z[:] = a

        np.random.seed(42)
        # test with different degrees of sparseness
        for p in 0.9, 0.5, 0.1, 0.01:
            ix0 = np.random.binomial(1, p, size=a.shape[0]).astype(bool)
            ix0 = np.nonzero(ix0)[0]
            ix1 = np.random.binomial(1, p, size=a.shape[1]).astype(bool)
            ix1 = np.nonzero(ix1)[0]
            ix2 = np.random.binomial(1, p, size=a.shape[2]).astype(bool)
            ix2 = np.nonzero(ix2)[0]

            # index all axes with int array
            expect = a[np.ix_(ix0, ix1, ix2)]
            actual = z[ix0, ix1, ix2]
            assert_array_equal(expect, actual)

            # mixed indexing with single int array / slices
            expect = a[ix0, 15:35, 25:45]
            actual = z[ix0, 15:35, 25:45]
            assert_array_equal(expect, actual)
            expect = a[15:35, ix1, 25:45]
            actual = z[15:35, ix1, 25:45]
            assert_array_equal(expect, actual)
            expect = a[15:35, 25:45, ix2]
            actual = z[15:35, 25:45, ix2]
            assert_array_equal(expect, actual)

            # mixed indexing with single int array / single index
            expect = a[ix0, 42, 84]
            actual = z[ix0, 42, 84]
            assert_array_equal(expect, actual)
            expect = a[42, ix1, 84]
            actual = z[42, ix1, 84]
            assert_array_equal(expect, actual)
            expect = a[42, 84, ix2]
            actual = z[42, 84, ix2]
            assert_array_equal(expect, actual)

            # mixed indexing with single int array / slice / single index
            expect = a[ix0, 15:35, 42]
            actual = z[ix0, 15:35, 42]
            assert_array_equal(expect, actual)
            expect = a[42, ix1, 25:45]
            actual = z[42, ix1, 25:45]
            assert_array_equal(expect, actual)
            expect = a[15:35, 42, ix2]
            actual = z[15:35, 42, ix2]
            assert_array_equal(expect, actual)

            # mixed indexing with two int array / slice
            expect = a[np.ix_(ix0, ix1, range(25, 45))]
            actual = z[ix0, ix1, 25:45]
            assert_array_equal(expect, actual)
            expect = a[np.ix_(range(15, 35), ix1, ix2)]
            actual = z[15:35, ix1, ix2]
            assert_array_equal(expect, actual)
            expect = a[np.ix_(ix0, range(25, 45), ix2)]
            actual = z[ix0, 25:45, ix2]
            assert_array_equal(expect, actual)

            # mixed indexing with two int array / integer
            expect = a[np.ix_(ix0, ix1, [42])].squeeze(axis=2)
            actual = z[ix0, ix1, 42]
            assert_array_equal(expect, actual)
            expect = a[np.ix_([42], ix1, ix2)].squeeze(axis=0)
            actual = z[42, ix1, ix2]
            assert_array_equal(expect, actual)
            expect = a[np.ix_(ix0, [42], ix2)].squeeze(axis=1)
            actual = z[ix0, 42, ix2]
            assert_array_equal(expect, actual)

    # TODO test advanced indexing with __setitem__


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


class TestArrayWithNestedDirectoryStore(TestArrayWithDirectoryStore):

    @staticmethod
    def create_array(read_only=False, **kwargs):
        path = mkdtemp()
        atexit.register(shutil.rmtree, path)
        store = NestedDirectoryStore(path)
        kwargs.setdefault('compressor', Zlib(1))
        init_array(store, **kwargs)
        return Array(store, read_only=read_only)


class TestArrayWithNoCompressor(TestArray):

    def create_array(self, read_only=False, **kwargs):
        store = dict()
        kwargs.setdefault('compressor', None)
        init_array(store, **kwargs)
        return Array(store, read_only=read_only)


class TestArrayWithBZ2Compressor(TestArray):

    def create_array(self, read_only=False, **kwargs):
        store = dict()
        compressor = BZ2(level=1)
        kwargs.setdefault('compressor', compressor)
        init_array(store, **kwargs)
        return Array(store, read_only=read_only)


class TestArrayWithBloscCompressor(TestArray):

    def create_array(self, read_only=False, **kwargs):
        store = dict()
        compressor = Blosc(cname='zstd', clevel=1, shuffle=1)
        kwargs.setdefault('compressor', compressor)
        init_array(store, **kwargs)
        return Array(store, read_only=read_only)


# TODO can we rely on backports and remove the PY2 exclusion?
if not PY2:  # pragma: py2 no cover

    from zarr.codecs import LZMA

    class TestArrayWithLZMACompressor(TestArray):

        def create_array(self, read_only=False, **kwargs):
            store = dict()
            compressor = LZMA(preset=1)
            kwargs.setdefault('compressor', compressor)
            init_array(store, **kwargs)
            return Array(store, read_only=read_only)


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

    def test_astype_no_filters(self):
        shape = (100,)
        dtype = np.dtype(np.int8)
        astype = np.dtype(np.float32)

        store = dict()
        init_array(store, shape=shape, chunks=10, dtype=dtype)

        data = np.arange(np.prod(shape), dtype=dtype).reshape(shape)

        z1 = Array(store)
        z1[...] = data
        z2 = z1.astype(astype)

        expected = data.astype(astype)
        assert_array_equal(expected, z2)
        eq(z2.read_only, True)

    def test_astype(self):
        shape = (100,)
        chunks = (10,)

        dtype = np.dtype(np.int8)
        astype = np.dtype(np.float32)

        data = np.arange(np.prod(shape), dtype=dtype).reshape(shape)

        z1 = self.create_array(shape=shape, chunks=chunks, dtype=dtype)
        z1[...] = data
        z2 = z1.astype(astype)

        expected = data.astype(astype)
        assert_array_equal(expected, z2)


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
