# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import unittest
from tempfile import mkdtemp, mktemp
import atexit
import shutil
import pickle
import os
import warnings


import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import (eq_ as eq, assert_is_instance, assert_raises, assert_true,
                        assert_false, assert_is, assert_is_none)
from nose import SkipTest
import pytest


from zarr.storage import (DirectoryStore, init_array, init_group, NestedDirectoryStore,
                          DBMStore, LMDBStore, atexit_rmtree, atexit_rmglob)
from zarr.core import Array
from zarr.errors import PermissionError
from zarr.compat import PY2
from zarr.util import buffer_size
from numcodecs import (Delta, FixedScaleOffset, Zlib, Blosc, BZ2, MsgPack, Pickle,
                       Categorize, JSON)


# needed for PY2/PY3 consistent behaviour
if PY2:  # pragma: py3 no cover
    warnings.resetwarnings()
    warnings.simplefilter('always')


# noinspection PyMethodMayBeStatic
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
        assert_is_none(a.basename)
        assert_is(store, a.store)
        eq("8fecb7a17ea1493d9c1430d04437b4f5b0b34985", a.hexdigest())

        # initialize at path
        store = dict()
        init_array(store, shape=100, chunks=10, path='foo/bar')
        a = Array(store, path='foo/bar')
        assert_is_instance(a, Array)
        eq((100,), a.shape)
        eq((10,), a.chunks)
        eq('foo/bar', a.path)
        eq('/foo/bar', a.name)
        eq('bar', a.basename)
        assert_is(store, a.store)
        eq("8fecb7a17ea1493d9c1430d04437b4f5b0b34985", a.hexdigest())

        # store not initialized
        store = dict()
        with assert_raises(ValueError):
            Array(store)

        # group is in the way
        store = dict()
        init_group(store, path='baz')
        with assert_raises(ValueError):
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

    # noinspection PyStatementEffect
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

    def test_array_1d_selections(self):
        # light test here, full tests in test_indexing

        # setup
        a = np.arange(1050)
        z = self.create_array(shape=a.shape, chunks=100, dtype=a.dtype)
        z[:] = a

        # get
        assert_array_equal(a[50:150], z.get_orthogonal_selection(slice(50, 150)))
        assert_array_equal(a[50:150], z.oindex[50: 150])
        ix = [99, 100, 101]
        bix = np.zeros_like(a, dtype=bool)
        bix[ix] = True
        assert_array_equal(a[ix], z.get_orthogonal_selection(ix))
        assert_array_equal(a[ix], z.oindex[ix])
        assert_array_equal(a[ix], z.get_coordinate_selection(ix))
        assert_array_equal(a[ix], z.vindex[ix])
        assert_array_equal(a[bix], z.get_mask_selection(bix))
        assert_array_equal(a[bix], z.oindex[bix])
        assert_array_equal(a[bix], z.vindex[bix])

        # set
        z.set_orthogonal_selection(slice(50, 150), 1)
        assert_array_equal(1, z[50:150])
        z.oindex[50:150] = 2
        assert_array_equal(2, z[50:150])
        z.set_orthogonal_selection(ix, 3)
        assert_array_equal(3, z.get_coordinate_selection(ix))
        z.oindex[ix] = 4
        assert_array_equal(4, z.oindex[ix])
        z.set_coordinate_selection(ix, 5)
        assert_array_equal(5, z.get_coordinate_selection(ix))
        z.vindex[ix] = 6
        assert_array_equal(6, z.vindex[ix])
        z.set_mask_selection(bix, 7)
        assert_array_equal(7, z.get_mask_selection(bix))
        z.vindex[bix] = 8
        assert_array_equal(8, z.vindex[bix])
        z.oindex[bix] = 9
        assert_array_equal(9, z.oindex[bix])

    # noinspection PyStatementEffect
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

    def test_array_2d_edge_case(self):
        # this fails with filters - chunks extend beyond edge of array, messes with delta
        # filter if no fill value?
        shape = 1000, 10
        chunks = 300, 30
        dtype = 'i8'
        z = self.create_array(shape=shape, dtype=dtype, chunks=chunks)
        z[:] = 0
        expect = np.zeros(shape, dtype=dtype)
        actual = z[:]
        assert_array_equal(expect, actual)

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

    def test_hexdigest(self):
        # Check basic 1-D array
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        eq('063b02ff8d9d3bab6da932ad5828b506ef0a6578', z.hexdigest())

        # Check basic 1-D array with different type
        z = self.create_array(shape=(1050,), chunks=100, dtype='f4')
        eq('f97b84dc9ffac807415f750100108764e837bb82', z.hexdigest())

        # Check basic 2-D array
        z = self.create_array(shape=(20, 35,), chunks=10, dtype='i4')
        eq('4f797d7bdad0fa1c9fa8c80832efb891a68de104', z.hexdigest())

        # Check basic 1-D array with some data
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        z[200:400] = np.arange(200, 400, dtype='i4')
        eq('14470724dca6c1837edddedc490571b6a7f270bc', z.hexdigest())

        # Check basic 1-D array with attributes
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        z.attrs['foo'] = 'bar'
        eq('2a1046dd99b914459b3e86be9dde05027a07d209', z.hexdigest())

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
        with assert_raises(PermissionError):
            z.set_basic_selection(Ellipsis, 42)
        with assert_raises(PermissionError):
            z.set_orthogonal_selection([0, 1, 2], 42)
        with assert_raises(PermissionError):
            z.oindex[[0, 1, 2]] = 42
        with assert_raises(PermissionError):
            z.set_coordinate_selection([0, 1, 2], 42)
        with assert_raises(PermissionError):
            z.vindex[[0, 1, 2]] = 42
        with assert_raises(PermissionError):
            z.set_mask_selection(np.ones(z.shape, dtype=bool), 42)

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

    # noinspection PyStatementEffect
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

    # noinspection PyStatementEffect
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

    # noinspection PyStatementEffect
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

    def test_structured_array(self):

        # setup some data
        a = np.array([(b'aaa', 1, 4.2),
                      (b'bbb', 2, 8.4),
                      (b'ccc', 3, 12.6)],
                     dtype=[('foo', 'S3'), ('bar', 'i4'), ('baz', 'f8')])
        for fill_value in None, b'', (b'zzz', 0, 0.0):
            z = self.create_array(shape=a.shape, chunks=2, dtype=a.dtype,
                                  fill_value=fill_value)
            eq(3, len(z))
            if fill_value is not None:
                np_fill_value = np.array(fill_value, dtype=a.dtype)[()]
                eq(np_fill_value, z.fill_value)
                eq(np_fill_value, z[0])
                eq(np_fill_value, z[-1])
            z[...] = a
            eq(a[0], z[0])
            assert_array_equal(a, z[...])
            assert_array_equal(a['foo'], z['foo'])
            assert_array_equal(a['bar'], z['bar'])
            assert_array_equal(a['baz'], z['baz'])

        with assert_raises(ValueError):
            # dodgy fill value
            self.create_array(shape=a.shape, chunks=2, dtype=a.dtype, fill_value=42)

    def test_dtypes(self):

        # integers
        for t in 'u1', 'u2', 'u4', 'u8', 'i1', 'i2', 'i4', 'i8':
            z = self.create_array(shape=10, chunks=3, dtype=t)
            assert z.dtype == np.dtype(t)
            a = np.arange(z.shape[0], dtype=t)
            z[:] = a
            assert_array_equal(a, z[:])

        # floats
        for t in 'f2', 'f4', 'f8':
            z = self.create_array(shape=10, chunks=3, dtype=t)
            assert z.dtype == np.dtype(t)
            a = np.linspace(0, 1, z.shape[0], dtype=t)
            z[:] = a
            assert_array_almost_equal(a, z[:])

        # datetime, timedelta are not supported for the time being
        for resolution in 'D', 'us', 'ns':
            with assert_raises(ValueError):
                self.create_array(shape=10, dtype='datetime64[{}]'.format(resolution))
            with assert_raises(ValueError):
                self.create_array(shape=10, dtype='timedelta64[{}]'.format(resolution))

    def test_object_arrays(self):

        # an object_codec is required for object arrays
        with pytest.raises(ValueError):
            self.create_array(shape=10, chunks=3, dtype=object)

        # an object_codec is required for object arrays, but allow to be provided via
        # filters to maintain API backwards compatibility
        with pytest.warns(FutureWarning):
            self.create_array(shape=10, chunks=3, dtype=object, filters=[MsgPack()])

        # create an object array using msgpack
        z = self.create_array(shape=10, chunks=3, dtype=object, object_codec=MsgPack())
        z[0] = 'foo'
        assert z[0] == 'foo'
        z[1] = b'bar'
        assert z[1] == 'bar'  # msgpack gets this wrong
        z[2] = 1
        assert z[2] == 1
        z[3] = [2, 4, 6, 'baz']
        assert z[3] == [2, 4, 6, 'baz']
        z[4] = {'a': 'b', 'c': 'd'}
        assert z[4] == {'a': 'b', 'c': 'd'}
        a = z[:]
        assert a.dtype == object

        # create an object array using pickle
        z = self.create_array(shape=10, chunks=3, dtype=object, object_codec=Pickle())
        z[0] = 'foo'
        assert z[0] == 'foo'
        z[1] = b'bar'
        assert z[1] == b'bar'
        z[2] = 1
        assert z[2] == 1
        z[3] = [2, 4, 6, 'baz']
        assert z[3] == [2, 4, 6, 'baz']
        z[4] = {'a': 'b', 'c': 'd'}
        assert z[4] == {'a': 'b', 'c': 'd'}
        a = z[:]
        assert a.dtype == object

        # create an object array using JSON
        z = self.create_array(shape=10, chunks=3, dtype=object, object_codec=JSON())
        z[0] = 'foo'
        assert z[0] == 'foo'
        # z[1] = b'bar'
        # assert z[1] == b'bar'  # not supported for JSON
        z[2] = 1
        assert z[2] == 1
        z[3] = [2, 4, 6, 'baz']
        assert z[3] == [2, 4, 6, 'baz']
        z[4] = {'a': 'b', 'c': 'd'}
        assert z[4] == {'a': 'b', 'c': 'd'}
        a = z[:]
        assert a.dtype == object

    def test_object_arrays_text(self):

        from numcodecs.tests.common import greetings
        data = np.array(greetings * 1000, dtype=object)

        z = self.create_array(shape=data.shape, dtype=object, object_codec=MsgPack())
        z[:] = data
        assert_array_equal(data, z[:])

        z = self.create_array(shape=data.shape, dtype=object, object_codec=JSON())
        z[:] = data
        assert_array_equal(data, z[:])

        z = self.create_array(shape=data.shape, dtype=object, object_codec=Pickle())
        z[:] = data
        assert_array_equal(data, z[:])

        z = self.create_array(shape=data.shape, dtype=object,
                              object_codec=Categorize(greetings, dtype=object))
        z[:] = data
        assert_array_equal(data, z[:])

    def test_object_arrays_danger(self):

        # do something dangerous - manually force an object array with no object codec
        z = self.create_array(shape=5, chunks=2, dtype=object, fill_value=0,
                              object_codec=MsgPack())
        z._filters = None  # wipe filters
        with assert_raises(RuntimeError):
            z[0] = 'foo'
        with assert_raises(RuntimeError):
            z[:] = 42

        # do something else dangerous
        labels = [
            '¡Hola mundo!',
            'Hej Världen!',
            'Servus Woid!',
            'Hei maailma!',
            'Xin chào thế giới',
            'Njatjeta Botë!',
            'Γεια σου κόσμε!',
            'こんにちは世界',
            '世界，你好！',
            'Helló, világ!',
            'Zdravo svete!',
            'เฮลโลเวิลด์'
        ]
        data = labels * 10
        for compressor in Zlib(1), Blosc():
            z = self.create_array(shape=len(data), chunks=30, dtype=object,
                                  object_codec=Categorize(labels, dtype=object),
                                  compressor=compressor)
            z[:] = data
            v = z.view(filters=[])
            with assert_raises(RuntimeError):
                # noinspection PyStatementEffect
                v[:]

    def test_object_codec_warnings(self):

        with pytest.warns(UserWarning):
            # provide object_codec, but not object dtype
            self.create_array(shape=10, chunks=5, dtype='i4', object_codec=JSON())


class TestArrayWithPath(TestArray):

    @staticmethod
    def create_array(read_only=False, **kwargs):
        store = dict()
        init_array(store, path='foo/bar', **kwargs)
        return Array(store, path='foo/bar', read_only=read_only)

    def test_hexdigest(self):
        # Check basic 1-D array
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        eq('f710da18d45d38d4aaf2afd7fb822fdd73d02957', z.hexdigest())

        # Check basic 1-D array with different type
        z = self.create_array(shape=(1050,), chunks=100, dtype='f4')
        eq('1437428e69754b1e1a38bd7fc9e43669577620db', z.hexdigest())

        # Check basic 2-D array
        z = self.create_array(shape=(20, 35,), chunks=10, dtype='i4')
        eq('dde44c72cc530bd6aae39b629eb15a2da627e5f9', z.hexdigest())

        # Check basic 1-D array with some data
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        z[200:400] = np.arange(200, 400, dtype='i4')
        eq('4c0a76fb1222498e09dcd92f7f9221d6cea8b40e', z.hexdigest())

        # Check basic 1-D array with attributes
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        z.attrs['foo'] = 'bar'
        eq('05b0663ffe1785f38d3a459dec17e57a18f254af', z.hexdigest())

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

    def test_hexdigest(self):
        # Check basic 1-D array
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        eq('f710da18d45d38d4aaf2afd7fb822fdd73d02957', z.hexdigest())

        # Check basic 1-D array with different type
        z = self.create_array(shape=(1050,), chunks=100, dtype='f4')
        eq('1437428e69754b1e1a38bd7fc9e43669577620db', z.hexdigest())

        # Check basic 2-D array
        z = self.create_array(shape=(20, 35,), chunks=10, dtype='i4')
        eq('dde44c72cc530bd6aae39b629eb15a2da627e5f9', z.hexdigest())

        # Check basic 1-D array with some data
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        z[200:400] = np.arange(200, 400, dtype='i4')
        eq('4c0a76fb1222498e09dcd92f7f9221d6cea8b40e', z.hexdigest())

        # Check basic 1-D array with attributes
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        z.attrs['foo'] = 'bar'
        eq('05b0663ffe1785f38d3a459dec17e57a18f254af', z.hexdigest())

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


class TestArrayWithDBMStore(TestArray):

    @staticmethod
    def create_array(read_only=False, **kwargs):
        path = mktemp(suffix='.anydbm')
        atexit.register(atexit_rmglob, path + '*')
        store = DBMStore(path, flag='n')
        kwargs.setdefault('compressor', Zlib(1))
        init_array(store, **kwargs)
        return Array(store, read_only=read_only)

    def test_nbytes_stored(self):
        pass  # not implemented


try:
    import bsddb3

    class TestArrayWithDBMStoreBerkeleyDB(TestArray):

        @staticmethod
        def create_array(read_only=False, **kwargs):
            path = mktemp(suffix='.dbm')
            atexit.register(os.remove, path)
            store = DBMStore(path, flag='n', open=bsddb3.btopen)
            kwargs.setdefault('compressor', Zlib(1))
            init_array(store, **kwargs)
            return Array(store, read_only=read_only)

        def test_nbytes_stored(self):
            pass  # not implemented

except ImportError:  # pragma: no cover
    pass


class TestArrayWithLMDBStore(TestArray):

    @staticmethod
    def create_array(read_only=False, **kwargs):
        path = mktemp(suffix='.lmdb')
        atexit.register(atexit_rmtree, path)
        try:
            store = LMDBStore(path, buffers=True)
        except ImportError:  # pragma: no cover
            raise SkipTest('lmdb not installed')
        kwargs.setdefault('compressor', Zlib(1))
        init_array(store, **kwargs)
        return Array(store, read_only=read_only)

    def test_nbytes_stored(self):
        pass  # not implemented


class TestArrayWithLMDBStoreNoBuffers(TestArray):

    @staticmethod
    def create_array(read_only=False, **kwargs):
        path = mktemp(suffix='.lmdb')
        atexit.register(atexit_rmtree, path)
        try:
            store = LMDBStore(path, buffers=False)
        except ImportError:  # pragma: no cover
            raise SkipTest('lmdb not installed')
        kwargs.setdefault('compressor', Zlib(1))
        init_array(store, **kwargs)
        return Array(store, read_only=read_only)

    def test_nbytes_stored(self):
        pass  # not implemented


class TestArrayWithNoCompressor(TestArray):

    def create_array(self, read_only=False, **kwargs):
        store = dict()
        kwargs.setdefault('compressor', None)
        init_array(store, **kwargs)
        return Array(store, read_only=read_only)

    def test_hexdigest(self):
        # Check basic 1-D array
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        eq('d3da3d485de4a5fcc6d91f9dfc6a7cba9720c561', z.hexdigest())

        # Check basic 1-D array with different type
        z = self.create_array(shape=(1050,), chunks=100, dtype='f4')
        eq('443b8dee512e42946cb63ff01d28e9bee8105a5f', z.hexdigest())

        # Check basic 2-D array
        z = self.create_array(shape=(20, 35,), chunks=10, dtype='i4')
        eq('de841ca276042993da53985de1e7769f5d0fc54d', z.hexdigest())

        # Check basic 1-D array with some data
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        z[200:400] = np.arange(200, 400, dtype='i4')
        eq('42b6ae0d50ec361628736ab7e68fe5fefca22136', z.hexdigest())

        # Check basic 1-D array with attributes
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        z.attrs['foo'] = 'bar'
        eq('a0535f31c130f5e5ac66ba0713d1c1ceaebd089b', z.hexdigest())


class TestArrayWithBZ2Compressor(TestArray):

    def create_array(self, read_only=False, **kwargs):
        store = dict()
        compressor = BZ2(level=1)
        kwargs.setdefault('compressor', compressor)
        init_array(store, **kwargs)
        return Array(store, read_only=read_only)

    def test_hexdigest(self):
        # Check basic 1-D array
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        eq('33141032439fb1df5e24ad9891a7d845b6c668c8', z.hexdigest())

        # Check basic 1-D array with different type
        z = self.create_array(shape=(1050,), chunks=100, dtype='f4')
        eq('44d719da065c88a412d609a5500ff41e07b331d6', z.hexdigest())

        # Check basic 2-D array
        z = self.create_array(shape=(20, 35,), chunks=10, dtype='i4')
        eq('f57a9a73a4004490fe1b871688651b8a298a5db7', z.hexdigest())

        # Check basic 1-D array with some data
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        z[200:400] = np.arange(200, 400, dtype='i4')
        eq('1e1bcaac63e4ef3c4a68f11672537131c627f168', z.hexdigest())

        # Check basic 1-D array with attributes
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        z.attrs['foo'] = 'bar'
        eq('86d7b9bf22dccbeaa22f340f38be506b55e76ff2', z.hexdigest())


class TestArrayWithBloscCompressor(TestArray):

    def create_array(self, read_only=False, **kwargs):
        store = dict()
        compressor = Blosc(cname='zstd', clevel=1, shuffle=1)
        kwargs.setdefault('compressor', compressor)
        init_array(store, **kwargs)
        return Array(store, read_only=read_only)

    def test_hexdigest(self):
        # Check basic 1-D array
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        eq('7ff2ae8511eac915fad311647c168ccfe943e788', z.hexdigest())

        # Check basic 1-D array with different type
        z = self.create_array(shape=(1050,), chunks=100, dtype='f4')
        eq('962705c861863495e9ccb7be7735907aa15e85b5', z.hexdigest())

        # Check basic 2-D array
        z = self.create_array(shape=(20, 35,), chunks=10, dtype='i4')
        eq('deb675ff91dd26dba11b65aab5f19a1f21a5645b', z.hexdigest())

        # Check basic 1-D array with some data
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        z[200:400] = np.arange(200, 400, dtype='i4')
        eq('90e30bdab745a9641cd0eb605356f531bc8ec1c3', z.hexdigest())

        # Check basic 1-D array with attributes
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        z.attrs['foo'] = 'bar'
        eq('95d40c391f167db8b1290e3c39d9bf741edacdf6', z.hexdigest())


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

        def test_hexdigest(self):
            # Check basic 1-D array
            z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
            eq('93ecaa530a1162a9d48a3c1dcee4586ccfc59bae', z.hexdigest())

            # Check basic 1-D array with different type
            z = self.create_array(shape=(1050,), chunks=100, dtype='f4')
            eq('04a9755a0cd638683531b7816c7fa4fbb6f577f2', z.hexdigest())

            # Check basic 2-D array
            z = self.create_array(shape=(20, 35,), chunks=10, dtype='i4')
            eq('b93b163a21e8500519250a6defb821d03eb5d9e0', z.hexdigest())

            # Check basic 1-D array with some data
            z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
            z[200:400] = np.arange(200, 400, dtype='i4')
            eq('cde499f3dc945b4e97197ff8e3cf8188a1262c35', z.hexdigest())

            # Check basic 1-D array with attributes
            z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
            z.attrs['foo'] = 'bar'
            eq('e2cf3afbf66ad0e28a2b6b68b1b07817c69aaee2', z.hexdigest())


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

    def test_hexdigest(self):
        # Check basic 1-D array
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        eq('b80367c5599d47110d42bd8886240c2f46620dba', z.hexdigest())

        # Check basic 1-D array with different type
        z = self.create_array(shape=(1050,), chunks=100, dtype='f4')
        eq('95a7b2471225e73199c9716d21e8d3dd6e5f6f2a', z.hexdigest())

        # Check basic 2-D array
        z = self.create_array(shape=(20, 35,), chunks=10, dtype='i4')
        eq('9abf3ad54413ab11855d88a5e0087cd416657e02', z.hexdigest())

        # Check basic 1-D array with some data
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        z[200:400] = np.arange(200, 400, dtype='i4')
        eq('c649ad229bc5720258b934ea958570c2f354c2eb', z.hexdigest())

        # Check basic 1-D array with attributes
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        z.attrs['foo'] = 'bar'
        eq('62fc9236d78af18a5ec26c12eea1d33bce52501e', z.hexdigest())

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

    def test_structured_array(self):
        # skip this one, cannot do delta on structured array
        pass

    def test_dtypes(self):
        # skip this one, delta messes up floats
        pass

    def test_object_arrays(self):
        # skip this one, cannot use delta with objects
        pass

    def test_object_arrays_text(self):
        # skip this one, cannot use delta with objects
        pass

    def test_object_arrays_danger(self):
        # skip this one, cannot use delta with objects
        pass


# custom store, does not support getsize()
class CustomMapping(object):

    def __init__(self):
        self.inner = dict()

    def keys(self):
        return self.inner.keys()

    def get(self, item, default=None):
        try:
            return self.inner[item]
        except KeyError:
            return default

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

    def test_object_arrays_danger(self):
        # skip this one as it only works if metadata are cached
        pass
