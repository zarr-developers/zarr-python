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
import pytest


from zarr.storage import (DirectoryStore, init_array, init_group, NestedDirectoryStore,
                          DBMStore, LMDBStore, atexit_rmtree, atexit_rmglob,
                          LRUStoreCache)
from zarr.core import Array
from zarr.errors import PermissionError
from zarr.compat import PY2, text_type, binary_type
from zarr.util import buffer_size
from numcodecs import (Delta, FixedScaleOffset, Zlib, Blosc, BZ2, MsgPack, Pickle,
                       Categorize, JSON, VLenUTF8, VLenBytes, VLenArray)
from numcodecs.tests.common import greetings


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
        assert isinstance(a, Array)
        assert (100,) == a.shape
        assert (10,) == a.chunks
        assert '' == a.path
        assert a.name is None
        assert a.basename is None
        assert store is a.store
        assert "8fecb7a17ea1493d9c1430d04437b4f5b0b34985" == a.hexdigest()

        # initialize at path
        store = dict()
        init_array(store, shape=100, chunks=10, path='foo/bar')
        a = Array(store, path='foo/bar')
        assert isinstance(a, Array)
        assert (100,) == a.shape
        assert (10,) == a.chunks
        assert 'foo/bar' == a.path
        assert '/foo/bar' == a.name
        assert 'bar' == a.basename
        assert store is a.store
        assert "8fecb7a17ea1493d9c1430d04437b4f5b0b34985" == a.hexdigest()

        # store not initialized
        store = dict()
        with pytest.raises(ValueError):
            Array(store)

        # group is in the way
        store = dict()
        init_group(store, path='baz')
        with pytest.raises(ValueError):
            Array(store, path='baz')

    def create_array(self, read_only=False, **kwargs):
        store = dict()
        kwargs.setdefault('compressor', Zlib(level=1))
        cache_metadata = kwargs.pop('cache_metadata', True)
        cache_attrs = kwargs.pop('cache_attrs', True)
        init_array(store, **kwargs)
        return Array(store, read_only=read_only, cache_metadata=cache_metadata,
                     cache_attrs=cache_attrs)

    def test_nbytes_stored(self):

        # dict as store
        z = self.create_array(shape=1000, chunks=100)
        expect_nbytes_stored = sum(buffer_size(v) for v in z.store.values())
        assert expect_nbytes_stored == z.nbytes_stored
        z[:] = 42
        expect_nbytes_stored = sum(buffer_size(v) for v in z.store.values())
        assert expect_nbytes_stored == z.nbytes_stored

        # mess with store
        try:
            z.store[z._key_prefix + 'foo'] = list(range(10))
            assert -1 == z.nbytes_stored
        except TypeError:
            pass

    # noinspection PyStatementEffect
    def test_array_1d(self):
        a = np.arange(1050)
        z = self.create_array(shape=a.shape, chunks=100, dtype=a.dtype)

        # check properties
        assert len(a) == len(z)
        assert a.ndim == z.ndim
        assert a.shape == z.shape
        assert a.dtype == z.dtype
        assert (100,) == z.chunks
        assert a.nbytes == z.nbytes
        assert 11 == z.nchunks
        assert 0 == z.nchunks_initialized
        assert (11,) == z.cdata_shape

        # check empty
        b = z[:]
        assert isinstance(b, np.ndarray)
        assert a.shape == b.shape
        assert a.dtype == b.dtype

        # check attributes
        z.attrs['foo'] = 'bar'
        assert 'bar' == z.attrs['foo']

        # set data
        z[:] = a

        # check properties
        assert a.nbytes == z.nbytes
        assert 11 == z.nchunks
        assert 11 == z.nchunks_initialized

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
        assert a[0] == z[0]
        assert a[-1] == z[-1]
        # unusual integer items
        assert a[42] == z[np.int64(42)]
        assert a[42] == z[np.int32(42)]
        assert a[42] == z[np.uint64(42)]
        assert a[42] == z[np.uint32(42)]
        # too many indices
        with pytest.raises(IndexError):
            z[:, :]
        with pytest.raises(IndexError):
            z[0, :]
        with pytest.raises(IndexError):
            z[:, 0]
        with pytest.raises(IndexError):
            z[0, 0]
        # only single ellipsis allowed
        with pytest.raises(IndexError):
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
        assert len(a) == len(z)
        assert a.ndim == z.ndim
        assert a.shape == z.shape
        assert a.dtype == z.dtype
        assert (100, 2) == z.chunks
        assert 0 == z.nchunks_initialized
        assert (10, 5) == z.cdata_shape

        # set data
        z[:] = a

        # check properties
        assert a.nbytes == z.nbytes
        assert 50 == z.nchunks_initialized

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
        assert a[0, 0] == z[0, 0]
        assert a[-1, -1] == z[-1, -1]

        # too many indices
        with pytest.raises(IndexError):
            z[:, :, :]
        with pytest.raises(IndexError):
            z[0, :, :]
        with pytest.raises(IndexError):
            z[:, 0, :]
        with pytest.raises(IndexError):
            z[:, :, 0]
        with pytest.raises(IndexError):
            z[0, 0, 0]
        # only single ellipsis allowed
        with pytest.raises(IndexError):
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
        with pytest.raises(ValueError):
            # N.B., NumPy allows this, but we'll be strict for now
            z[2:3] = c
        with pytest.raises(ValueError):
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
        with pytest.raises(ValueError):
            z[:, 2:3] = d
        with pytest.raises(ValueError):
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
        assert -1 == z[0, 0]
        assert -1 == z[2, 2]
        assert -1 == z[-1, -1]

    def test_array_order(self):

        # 1D
        a = np.arange(1050)
        for order in 'C', 'F':
            z = self.create_array(shape=a.shape, chunks=100, dtype=a.dtype,
                                  order=order)
            assert order == z.order
            if order == 'F':
                assert z[:].flags.f_contiguous
            else:
                assert z[:].flags.c_contiguous
            z[:] = a
            assert_array_equal(a, z[:])

        # 2D
        a = np.arange(10000).reshape((100, 100))
        for order in 'C', 'F':
            z = self.create_array(shape=a.shape, chunks=(10, 10),
                                  dtype=a.dtype, order=order)
            assert order == z.order
            if order == 'F':
                assert z[:].flags.f_contiguous
            else:
                assert z[:].flags.c_contiguous
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
        assert '063b02ff8d9d3bab6da932ad5828b506ef0a6578' == z.hexdigest()

        # Check basic 1-D array with different type
        z = self.create_array(shape=(1050,), chunks=100, dtype='f4')
        assert 'f97b84dc9ffac807415f750100108764e837bb82' == z.hexdigest()

        # Check basic 2-D array
        z = self.create_array(shape=(20, 35,), chunks=10, dtype='i4')
        assert '4f797d7bdad0fa1c9fa8c80832efb891a68de104' == z.hexdigest()

        # Check basic 1-D array with some data
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        z[200:400] = np.arange(200, 400, dtype='i4')
        assert '14470724dca6c1837edddedc490571b6a7f270bc' == z.hexdigest()

        # Check basic 1-D array with attributes
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        z.attrs['foo'] = 'bar'
        assert '2a1046dd99b914459b3e86be9dde05027a07d209' == z.hexdigest()

    def test_resize_1d(self):

        z = self.create_array(shape=105, chunks=10, dtype='i4',
                              fill_value=0)
        a = np.arange(105, dtype='i4')
        z[:] = a
        assert (105,) == z.shape
        assert (105,) == z[:].shape
        assert np.dtype('i4') == z.dtype
        assert np.dtype('i4') == z[:].dtype
        assert (10,) == z.chunks
        assert_array_equal(a, z[:])

        z.resize(205)
        assert (205,) == z.shape
        assert (205,) == z[:].shape
        assert np.dtype('i4') == z.dtype
        assert np.dtype('i4') == z[:].dtype
        assert (10,) == z.chunks
        assert_array_equal(a, z[:105])
        assert_array_equal(np.zeros(100, dtype='i4'), z[105:])

        z.resize(55)
        assert (55,) == z.shape
        assert (55,) == z[:].shape
        assert np.dtype('i4') == z.dtype
        assert np.dtype('i4') == z[:].dtype
        assert (10,) == z.chunks
        assert_array_equal(a[:55], z[:])

        # via shape setter
        z.shape = (105,)
        assert (105,) == z.shape
        assert (105,) == z[:].shape

    def test_resize_2d(self):

        z = self.create_array(shape=(105, 105), chunks=(10, 10), dtype='i4',
                              fill_value=0)
        a = np.arange(105*105, dtype='i4').reshape((105, 105))
        z[:] = a
        assert (105, 105) == z.shape
        assert (105, 105) == z[:].shape
        assert np.dtype('i4') == z.dtype
        assert np.dtype('i4') == z[:].dtype
        assert (10, 10) == z.chunks
        assert_array_equal(a, z[:])

        z.resize((205, 205))
        assert (205, 205) == z.shape
        assert (205, 205) == z[:].shape
        assert np.dtype('i4') == z.dtype
        assert np.dtype('i4') == z[:].dtype
        assert (10, 10) == z.chunks
        assert_array_equal(a, z[:105, :105])
        assert_array_equal(np.zeros((100, 205), dtype='i4'), z[105:, :])
        assert_array_equal(np.zeros((205, 100), dtype='i4'), z[:, 105:])

        z.resize((55, 55))
        assert (55, 55) == z.shape
        assert (55, 55) == z[:].shape
        assert np.dtype('i4') == z.dtype
        assert np.dtype('i4') == z[:].dtype
        assert (10, 10) == z.chunks
        assert_array_equal(a[:55, :55], z[:])

        z.resize((55, 1))
        assert (55, 1) == z.shape
        assert (55, 1) == z[:].shape
        assert np.dtype('i4') == z.dtype
        assert np.dtype('i4') == z[:].dtype
        assert (10, 10) == z.chunks
        assert_array_equal(a[:55, :1], z[:])

        # via shape setter
        z.shape = (105, 105)
        assert (105, 105) == z.shape
        assert (105, 105) == z[:].shape

    def test_append_1d(self):

        a = np.arange(105)
        z = self.create_array(shape=a.shape, chunks=10, dtype=a.dtype)
        z[:] = a
        assert a.shape == z.shape
        assert a.dtype == z.dtype
        assert (10,) == z.chunks
        assert_array_equal(a, z[:])

        b = np.arange(105, 205)
        e = np.append(a, b)
        z.append(b)
        assert e.shape == z.shape
        assert e.dtype == z.dtype
        assert (10,) == z.chunks
        assert_array_equal(e, z[:])

        # check append handles array-like
        c = [1, 2, 3]
        f = np.append(e, c)
        z.append(c)
        assert f.shape == z.shape
        assert f.dtype == z.dtype
        assert (10,) == z.chunks
        assert_array_equal(f, z[:])

    def test_append_2d(self):

        a = np.arange(105*105, dtype='i4').reshape((105, 105))
        z = self.create_array(shape=a.shape, chunks=(10, 10), dtype=a.dtype)
        z[:] = a
        assert a.shape == z.shape
        assert a.dtype == z.dtype
        assert (10, 10) == z.chunks
        actual = z[:]
        assert_array_equal(a, actual)

        b = np.arange(105*105, 2*105*105, dtype='i4').reshape((105, 105))
        e = np.append(a, b, axis=0)
        z.append(b)
        assert e.shape == z.shape
        assert e.dtype == z.dtype
        assert (10, 10) == z.chunks
        actual = z[:]
        assert_array_equal(e, actual)

    def test_append_2d_axis(self):

        a = np.arange(105*105, dtype='i4').reshape((105, 105))
        z = self.create_array(shape=a.shape, chunks=(10, 10), dtype=a.dtype)
        z[:] = a
        assert a.shape == z.shape
        assert a.dtype == z.dtype
        assert (10, 10) == z.chunks
        assert_array_equal(a, z[:])

        b = np.arange(105*105, 2*105*105, dtype='i4').reshape((105, 105))
        e = np.append(a, b, axis=1)
        z.append(b, axis=1)
        assert e.shape == z.shape
        assert e.dtype == z.dtype
        assert (10, 10) == z.chunks
        assert_array_equal(e, z[:])

    def test_append_bad_shape(self):
        a = np.arange(100)
        z = self.create_array(shape=a.shape, chunks=10, dtype=a.dtype)
        z[:] = a
        b = a.reshape(10, 10)
        with pytest.raises(ValueError):
            z.append(b)

    def test_read_only(self):

        z = self.create_array(shape=1000, chunks=100)
        assert not z.read_only

        z = self.create_array(shape=1000, chunks=100, read_only=True)
        assert z.read_only
        with pytest.raises(PermissionError):
            z[:] = 42
        with pytest.raises(PermissionError):
            z.resize(2000)
        with pytest.raises(PermissionError):
            z.append(np.arange(1000))
        with pytest.raises(PermissionError):
            z.set_basic_selection(Ellipsis, 42)
        with pytest.raises(PermissionError):
            z.set_orthogonal_selection([0, 1, 2], 42)
        with pytest.raises(PermissionError):
            z.oindex[[0, 1, 2]] = 42
        with pytest.raises(PermissionError):
            z.set_coordinate_selection([0, 1, 2], 42)
        with pytest.raises(PermissionError):
            z.vindex[[0, 1, 2]] = 42
        with pytest.raises(PermissionError):
            z.set_mask_selection(np.ones(z.shape, dtype=bool), 42)

    def test_pickle(self):

        z = self.create_array(shape=1000, chunks=100, dtype=int, cache_metadata=False,
                              cache_attrs=False)
        z[:] = np.random.randint(0, 1000, 1000)
        z2 = pickle.loads(pickle.dumps(z))
        assert z.shape == z2.shape
        assert z.chunks == z2.chunks
        assert z.dtype == z2.dtype
        if z.compressor:
            assert z.compressor.get_config() == z2.compressor.get_config()
        assert z.fill_value == z2.fill_value
        assert z._cache_metadata == z2._cache_metadata
        assert z.attrs.cache == z2.attrs.cache
        assert_array_equal(z[:], z2[:])

    def test_np_ufuncs(self):
        z = self.create_array(shape=(100, 100), chunks=(10, 10))
        a = np.arange(10000).reshape(100, 100)
        z[:] = a

        assert np.sum(a) == np.sum(z)
        assert_array_equal(np.sum(a, axis=0), np.sum(z, axis=0))
        assert np.mean(a) == np.mean(z)
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
        assert a.ndim == z.ndim
        assert a.shape == z.shape
        assert a.dtype == z.dtype
        assert a.size == z.size
        assert 0 == z.nchunks

        # cannot make a good decision when auto-chunking if a dimension has zero length,
        # fall back to 1 for now
        assert (1,) == z.chunks

        # check __getitem__
        assert isinstance(z[:], np.ndarray)
        assert_array_equal(a, np.array(z))
        assert_array_equal(a, z[:])
        assert_array_equal(a, z[...])
        assert_array_equal(a[0:0], z[0:0])
        with pytest.raises(IndexError):
            z[0]

        # check __setitem__
        # these should succeed but do nothing
        z[:] = 42
        z[...] = 42
        # this should error
        with pytest.raises(IndexError):
            z[0] = 42

    # noinspection PyStatementEffect
    def test_0len_dim_2d(self):
        # Test behavioud for 2D array with a zero-length dimension.

        z = self.create_array(shape=(10, 0), fill_value=0)
        a = np.zeros((10, 0))
        assert a.ndim == z.ndim
        assert a.shape == z.shape
        assert a.dtype == z.dtype
        assert a.size == z.size
        assert 0 == z.nchunks

        # cannot make a good decision when auto-chunking if a dimension has zero length,
        # fall back to 1 for now
        assert (10, 1) == z.chunks

        # check __getitem__
        assert isinstance(z[:], np.ndarray)
        assert_array_equal(a, np.array(z))
        assert_array_equal(a, z[:])
        assert_array_equal(a, z[...])
        assert_array_equal(a[0], z[0])
        assert_array_equal(a[0, 0:0], z[0, 0:0])
        assert_array_equal(a[0, :], z[0, :])
        assert_array_equal(a[0, 0:0], z[0, 0:0])
        with pytest.raises(IndexError):
            z[:, 0]

        # check __setitem__
        # these should succeed but do nothing
        z[:] = 42
        z[...] = 42
        z[0, :] = 42
        # this should error
        with pytest.raises(IndexError):
            z[:, 0] = 42

    # noinspection PyStatementEffect
    def test_array_0d(self):
        # test behaviour for array with 0 dimensions

        # setup
        a = np.zeros(())
        z = self.create_array(shape=(), dtype=a.dtype, fill_value=0)

        # check properties
        assert a.ndim == z.ndim
        assert a.shape == z.shape
        assert a.size == z.size
        assert a.dtype == z.dtype
        assert a.nbytes == z.nbytes
        with pytest.raises(TypeError):
            len(z)
        assert () == z.chunks
        assert 1 == z.nchunks
        assert (1,) == z.cdata_shape
        # compressor always None - no point in compressing a single value
        assert z.compressor is None

        # check __getitem__
        b = z[...]
        assert isinstance(b, np.ndarray)
        assert a.shape == b.shape
        assert a.dtype == b.dtype
        assert_array_equal(a, np.array(z))
        assert_array_equal(a, z[...])
        assert a[()] == z[()]
        with pytest.raises(IndexError):
            z[0]
        with pytest.raises(IndexError):
            z[:]

        # check __setitem__
        z[...] = 42
        assert 42 == z[()]
        z[()] = 43
        assert 43 == z[()]
        with pytest.raises(IndexError):
            z[0] = 42
        with pytest.raises(IndexError):
            z[:] = 42
        with pytest.raises(ValueError):
            z[...] = np.array([1, 2, 3])

    def test_nchunks_initialized(self):

        z = self.create_array(shape=100, chunks=10)
        assert 0 == z.nchunks_initialized
        # manually put something into the store to confuse matters
        z.store['foo'] = b'bar'
        assert 0 == z.nchunks_initialized
        z[:] = 42
        assert 10 == z.nchunks_initialized

    def test_structured_array(self):

        # setup some data
        d = np.array([(b'aaa', 1, 4.2),
                      (b'bbb', 2, 8.4),
                      (b'ccc', 3, 12.6)],
                     dtype=[('foo', 'S3'), ('bar', 'i4'), ('baz', 'f8')])
        for a in (d, d[:0]):
            for fill_value in None, b'', (b'zzz', 42, 16.8):
                z = self.create_array(shape=a.shape, chunks=2, dtype=a.dtype, fill_value=fill_value)
                assert len(a) == len(z)
                if fill_value is not None:
                    if fill_value == b'':
                        # numpy 1.14 compatibility
                        np_fill_value = np.array(fill_value, dtype=a.dtype.str).view(a.dtype)[()]
                    else:
                        np_fill_value = np.array(fill_value, dtype=a.dtype)[()]
                    assert np_fill_value == z.fill_value
                    if len(z):
                        assert np_fill_value == z[0]
                        assert np_fill_value == z[-1]
                z[...] = a
                if len(a):
                    assert a[0] == z[0]
                assert_array_equal(a, z[...])
                assert_array_equal(a['foo'], z['foo'])
                assert_array_equal(a['bar'], z['bar'])
                assert_array_equal(a['baz'], z['baz'])

    def test_dtypes(self):

        # integers
        for dtype in 'u1', 'u2', 'u4', 'u8', 'i1', 'i2', 'i4', 'i8':
            z = self.create_array(shape=10, chunks=3, dtype=dtype)
            assert z.dtype == np.dtype(dtype)
            a = np.arange(z.shape[0], dtype=dtype)
            z[:] = a
            assert_array_equal(a, z[:])

        # floats
        for dtype in 'f2', 'f4', 'f8':
            z = self.create_array(shape=10, chunks=3, dtype=dtype)
            assert z.dtype == np.dtype(dtype)
            a = np.linspace(0, 1, z.shape[0], dtype=dtype)
            z[:] = a
            assert_array_almost_equal(a, z[:])

        # datetime, timedelta
        for base_type in 'Mm':
            for resolution in 'D', 'us', 'ns':
                dtype = '{}8[{}]'.format(base_type, resolution)
                z = self.create_array(shape=100, dtype=dtype, fill_value=0)
                assert z.dtype == np.dtype(dtype)
                a = np.random.randint(0, np.iinfo('u8').max, size=z.shape[0],
                                      dtype='u8').view(dtype)
                z[:] = a
                assert_array_equal(a, z[:])

        # check that datetime generic units are not allowed
        with pytest.raises(ValueError):
            self.create_array(shape=100, dtype='M8')
        with pytest.raises(ValueError):
            self.create_array(shape=100, dtype='m8')

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

    def test_object_arrays_vlen_text(self):

        data = np.array(greetings * 1000, dtype=object)

        z = self.create_array(shape=data.shape, dtype=object, object_codec=VLenUTF8())
        z[0] = u'foo'
        assert z[0] == u'foo'
        z[1] = u'bar'
        assert z[1] == u'bar'
        z[2] = u'baz'
        assert z[2] == u'baz'
        z[:] = data
        a = z[:]
        assert a.dtype == object
        assert_array_equal(data, a)

        # convenience API
        z = self.create_array(shape=data.shape, dtype=text_type)
        assert z.dtype == object
        assert isinstance(z.filters[0], VLenUTF8)
        z[:] = data
        assert_array_equal(data, z[:])

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

    def test_object_arrays_vlen_bytes(self):

        greetings_bytes = [g.encode('utf8') for g in greetings]
        data = np.array(greetings_bytes * 1000, dtype=object)

        z = self.create_array(shape=data.shape, dtype=object, object_codec=VLenBytes())
        z[0] = b'foo'
        assert z[0] == b'foo'
        z[1] = b'bar'
        assert z[1] == b'bar'
        z[2] = b'baz'
        assert z[2] == b'baz'
        z[:] = data
        a = z[:]
        assert a.dtype == object
        assert_array_equal(data, a)

        # convenience API
        z = self.create_array(shape=data.shape, dtype=binary_type)
        assert z.dtype == object
        assert isinstance(z.filters[0], VLenBytes)
        z[:] = data
        assert_array_equal(data, z[:])

        z = self.create_array(shape=data.shape, dtype=object, object_codec=Pickle())
        z[:] = data
        assert_array_equal(data, z[:])

    def test_object_arrays_vlen_array(self):

        data = np.array([np.array([1, 3, 7]),
                         np.array([5]),
                         np.array([2, 8, 12])] * 1000, dtype=object)

        def compare_arrays(expected, actual, item_dtype):
            assert isinstance(actual, np.ndarray)
            assert actual.dtype == object
            assert actual.shape == expected.shape
            for ev, av in zip(expected.flat, actual.flat):
                assert isinstance(av, np.ndarray)
                assert_array_equal(ev, av)
                assert av.dtype == item_dtype

        codecs = VLenArray(int), VLenArray('<u4')
        for codec in codecs:
            z = self.create_array(shape=data.shape, dtype=object, object_codec=codec)
            z[0] = np.array([4, 7])
            assert_array_equal(np.array([4, 7]), z[0])
            z[:] = data
            a = z[:]
            assert a.dtype == object
            compare_arrays(data, a, codec.dtype)

        # convenience API
        for item_type in 'int', '<u4':
            z = self.create_array(shape=data.shape, dtype='array:{}'.format(item_type))
            assert z.dtype == object
            assert isinstance(z.filters[0], VLenArray)
            assert z.filters[0].dtype == np.dtype(item_type)
            z[:] = data
            compare_arrays(data, z[:], np.dtype(item_type))

    def test_object_arrays_danger(self):

        # do something dangerous - manually force an object array with no object codec
        z = self.create_array(shape=5, chunks=2, dtype=object, fill_value=0,
                              object_codec=MsgPack())
        z._filters = None  # wipe filters
        with pytest.raises(RuntimeError):
            z[0] = 'foo'
        with pytest.raises(RuntimeError):
            z[:] = 42

        # do something else dangerous
        data = greetings * 10
        for compressor in Zlib(1), Blosc():
            z = self.create_array(shape=len(data), chunks=30, dtype=object,
                                  object_codec=Categorize(greetings,
                                                          dtype=object),
                                  compressor=compressor)
            z[:] = data
            v = z.view(filters=[])
            with pytest.raises(RuntimeError):
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
        cache_metadata = kwargs.pop('cache_metadata', True)
        cache_attrs = kwargs.pop('cache_attrs', True)
        init_array(store, path='foo/bar', **kwargs)
        return Array(store, path='foo/bar', read_only=read_only,
                     cache_metadata=cache_metadata, cache_attrs=cache_attrs)

    def test_hexdigest(self):
        # Check basic 1-D array
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        assert 'f710da18d45d38d4aaf2afd7fb822fdd73d02957' == z.hexdigest()

        # Check basic 1-D array with different type
        z = self.create_array(shape=(1050,), chunks=100, dtype='f4')
        assert '1437428e69754b1e1a38bd7fc9e43669577620db' == z.hexdigest()

        # Check basic 2-D array
        z = self.create_array(shape=(20, 35,), chunks=10, dtype='i4')
        assert 'dde44c72cc530bd6aae39b629eb15a2da627e5f9' == z.hexdigest()

        # Check basic 1-D array with some data
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        z[200:400] = np.arange(200, 400, dtype='i4')
        assert '4c0a76fb1222498e09dcd92f7f9221d6cea8b40e' == z.hexdigest()

        # Check basic 1-D array with attributes
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        z.attrs['foo'] = 'bar'
        assert '05b0663ffe1785f38d3a459dec17e57a18f254af' == z.hexdigest()

    def test_nbytes_stored(self):

        # dict as store
        z = self.create_array(shape=1000, chunks=100)
        expect_nbytes_stored = sum(buffer_size(v)
                                   for k, v in z.store.items()
                                   if k.startswith('foo/bar/'))
        assert expect_nbytes_stored == z.nbytes_stored
        z[:] = 42
        expect_nbytes_stored = sum(buffer_size(v)
                                   for k, v in z.store.items()
                                   if k.startswith('foo/bar/'))
        assert expect_nbytes_stored == z.nbytes_stored

        # mess with store
        z.store[z._key_prefix + 'foo'] = list(range(10))
        assert -1 == z.nbytes_stored


class TestArrayWithChunkStore(TestArray):

    @staticmethod
    def create_array(read_only=False, **kwargs):
        store = dict()
        # separate chunk store
        chunk_store = dict()
        cache_metadata = kwargs.pop('cache_metadata', True)
        cache_attrs = kwargs.pop('cache_attrs', True)
        init_array(store, chunk_store=chunk_store, **kwargs)
        return Array(store, read_only=read_only, chunk_store=chunk_store,
                     cache_metadata=cache_metadata, cache_attrs=cache_attrs)

    def test_hexdigest(self):
        # Check basic 1-D array
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        assert 'f710da18d45d38d4aaf2afd7fb822fdd73d02957' == z.hexdigest()

        # Check basic 1-D array with different type
        z = self.create_array(shape=(1050,), chunks=100, dtype='f4')
        assert '1437428e69754b1e1a38bd7fc9e43669577620db' == z.hexdigest()

        # Check basic 2-D array
        z = self.create_array(shape=(20, 35,), chunks=10, dtype='i4')
        assert 'dde44c72cc530bd6aae39b629eb15a2da627e5f9' == z.hexdigest()

        # Check basic 1-D array with some data
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        z[200:400] = np.arange(200, 400, dtype='i4')
        assert '4c0a76fb1222498e09dcd92f7f9221d6cea8b40e' == z.hexdigest()

        # Check basic 1-D array with attributes
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        z.attrs['foo'] = 'bar'
        assert '05b0663ffe1785f38d3a459dec17e57a18f254af' == z.hexdigest()

    def test_nbytes_stored(self):

        z = self.create_array(shape=1000, chunks=100)
        expect_nbytes_stored = sum(buffer_size(v) for v in z.store.values())
        expect_nbytes_stored += sum(buffer_size(v)
                                    for v in z.chunk_store.values())
        assert expect_nbytes_stored == z.nbytes_stored
        z[:] = 42
        expect_nbytes_stored = sum(buffer_size(v) for v in z.store.values())
        expect_nbytes_stored += sum(buffer_size(v)
                                    for v in z.chunk_store.values())
        assert expect_nbytes_stored == z.nbytes_stored

        # mess with store
        z.chunk_store[z._key_prefix + 'foo'] = list(range(10))
        assert -1 == z.nbytes_stored


class TestArrayWithDirectoryStore(TestArray):

    @staticmethod
    def create_array(read_only=False, **kwargs):
        path = mkdtemp()
        atexit.register(shutil.rmtree, path)
        store = DirectoryStore(path)
        cache_metadata = kwargs.pop('cache_metadata', True)
        cache_attrs = kwargs.pop('cache_attrs', True)
        kwargs.setdefault('compressor', Zlib(1))
        init_array(store, **kwargs)
        return Array(store, read_only=read_only, cache_metadata=cache_metadata,
                     cache_attrs=cache_attrs)

    def test_nbytes_stored(self):

        # dict as store
        z = self.create_array(shape=1000, chunks=100)
        expect_nbytes_stored = sum(buffer_size(v) for v in z.store.values())
        assert expect_nbytes_stored == z.nbytes_stored
        z[:] = 42
        expect_nbytes_stored = sum(buffer_size(v) for v in z.store.values())
        assert expect_nbytes_stored == z.nbytes_stored


class TestArrayWithNestedDirectoryStore(TestArrayWithDirectoryStore):

    @staticmethod
    def create_array(read_only=False, **kwargs):
        path = mkdtemp()
        atexit.register(shutil.rmtree, path)
        store = NestedDirectoryStore(path)
        cache_metadata = kwargs.pop('cache_metadata', True)
        cache_attrs = kwargs.pop('cache_attrs', True)
        kwargs.setdefault('compressor', Zlib(1))
        init_array(store, **kwargs)
        return Array(store, read_only=read_only, cache_metadata=cache_metadata,
                     cache_attrs=cache_attrs)


class TestArrayWithDBMStore(TestArray):

    @staticmethod
    def create_array(read_only=False, **kwargs):
        path = mktemp(suffix='.anydbm')
        atexit.register(atexit_rmglob, path + '*')
        store = DBMStore(path, flag='n')
        cache_metadata = kwargs.pop('cache_metadata', True)
        cache_attrs = kwargs.pop('cache_attrs', True)
        kwargs.setdefault('compressor', Zlib(1))
        init_array(store, **kwargs)
        return Array(store, read_only=read_only, cache_attrs=cache_attrs,
                     cache_metadata=cache_metadata)

    def test_nbytes_stored(self):
        pass  # not implemented


try:
    import bsddb3
except ImportError:  # pragma: no cover
    bsddb3 = None


@unittest.skipIf(bsddb3 is None, 'bsddb3 is not installed')
class TestArrayWithDBMStoreBerkeleyDB(TestArray):

    @staticmethod
    def create_array(read_only=False, **kwargs):
        path = mktemp(suffix='.dbm')
        atexit.register(os.remove, path)
        store = DBMStore(path, flag='n', open=bsddb3.btopen)
        cache_metadata = kwargs.pop('cache_metadata', True)
        cache_attrs = kwargs.pop('cache_attrs', True)
        kwargs.setdefault('compressor', Zlib(1))
        init_array(store, **kwargs)
        return Array(store, read_only=read_only, cache_metadata=cache_metadata,
                     cache_attrs=cache_attrs)

    def test_nbytes_stored(self):
        pass  # not implemented


try:
    import lmdb
except ImportError:  # pragma: no cover
    lmdb = None


@unittest.skipIf(lmdb is None, 'lmdb is not installed')
class TestArrayWithLMDBStore(TestArray):

    @staticmethod
    def create_array(read_only=False, **kwargs):
        path = mktemp(suffix='.lmdb')
        atexit.register(atexit_rmtree, path)
        store = LMDBStore(path, buffers=True)
        cache_metadata = kwargs.pop('cache_metadata', True)
        cache_attrs = kwargs.pop('cache_attrs', True)
        kwargs.setdefault('compressor', Zlib(1))
        init_array(store, **kwargs)
        return Array(store, read_only=read_only, cache_metadata=cache_metadata,
                     cache_attrs=cache_attrs)

    def test_nbytes_stored(self):
        pass  # not implemented


@unittest.skipIf(lmdb is None, 'lmdb is not installed')
class TestArrayWithLMDBStoreNoBuffers(TestArray):

    @staticmethod
    def create_array(read_only=False, **kwargs):
        path = mktemp(suffix='.lmdb')
        atexit.register(atexit_rmtree, path)
        store = LMDBStore(path, buffers=False)
        cache_metadata = kwargs.pop('cache_metadata', True)
        cache_attrs = kwargs.pop('cache_attrs', True)
        kwargs.setdefault('compressor', Zlib(1))
        init_array(store, **kwargs)
        return Array(store, read_only=read_only, cache_metadata=cache_metadata,
                     cache_attrs=cache_attrs)

    def test_nbytes_stored(self):
        pass  # not implemented


class TestArrayWithNoCompressor(TestArray):

    def create_array(self, read_only=False, **kwargs):
        store = dict()
        kwargs.setdefault('compressor', None)
        cache_metadata = kwargs.pop('cache_metadata', True)
        cache_attrs = kwargs.pop('cache_attrs', True)
        init_array(store, **kwargs)
        return Array(store, read_only=read_only, cache_metadata=cache_metadata,
                     cache_attrs=cache_attrs)

    def test_hexdigest(self):
        # Check basic 1-D array
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        assert 'd3da3d485de4a5fcc6d91f9dfc6a7cba9720c561' == z.hexdigest()

        # Check basic 1-D array with different type
        z = self.create_array(shape=(1050,), chunks=100, dtype='f4')
        assert '443b8dee512e42946cb63ff01d28e9bee8105a5f' == z.hexdigest()

        # Check basic 2-D array
        z = self.create_array(shape=(20, 35,), chunks=10, dtype='i4')
        assert 'de841ca276042993da53985de1e7769f5d0fc54d' == z.hexdigest()

        # Check basic 1-D array with some data
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        z[200:400] = np.arange(200, 400, dtype='i4')
        assert '42b6ae0d50ec361628736ab7e68fe5fefca22136' == z.hexdigest()

        # Check basic 1-D array with attributes
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        z.attrs['foo'] = 'bar'
        assert 'a0535f31c130f5e5ac66ba0713d1c1ceaebd089b' == z.hexdigest()


class TestArrayWithBZ2Compressor(TestArray):

    def create_array(self, read_only=False, **kwargs):
        store = dict()
        compressor = BZ2(level=1)
        kwargs.setdefault('compressor', compressor)
        cache_metadata = kwargs.pop('cache_metadata', True)
        cache_attrs = kwargs.pop('cache_attrs', True)
        init_array(store, **kwargs)
        return Array(store, read_only=read_only, cache_metadata=cache_metadata,
                     cache_attrs=cache_attrs)

    def test_hexdigest(self):
        # Check basic 1-D array
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        assert '33141032439fb1df5e24ad9891a7d845b6c668c8' == z.hexdigest()

        # Check basic 1-D array with different type
        z = self.create_array(shape=(1050,), chunks=100, dtype='f4')
        assert '44d719da065c88a412d609a5500ff41e07b331d6' == z.hexdigest()

        # Check basic 2-D array
        z = self.create_array(shape=(20, 35,), chunks=10, dtype='i4')
        assert 'f57a9a73a4004490fe1b871688651b8a298a5db7' == z.hexdigest()

        # Check basic 1-D array with some data
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        z[200:400] = np.arange(200, 400, dtype='i4')
        assert '1e1bcaac63e4ef3c4a68f11672537131c627f168' == z.hexdigest()

        # Check basic 1-D array with attributes
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        z.attrs['foo'] = 'bar'
        assert '86d7b9bf22dccbeaa22f340f38be506b55e76ff2' == z.hexdigest()


class TestArrayWithBloscCompressor(TestArray):

    def create_array(self, read_only=False, **kwargs):
        store = dict()
        compressor = Blosc(cname='zstd', clevel=1, shuffle=1)
        kwargs.setdefault('compressor', compressor)
        cache_metadata = kwargs.pop('cache_metadata', True)
        cache_attrs = kwargs.pop('cache_attrs', True)
        init_array(store, **kwargs)
        return Array(store, read_only=read_only, cache_metadata=cache_metadata,
                     cache_attrs=cache_attrs)

    def test_hexdigest(self):
        # Check basic 1-D array
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        assert '7ff2ae8511eac915fad311647c168ccfe943e788' == z.hexdigest()

        # Check basic 1-D array with different type
        z = self.create_array(shape=(1050,), chunks=100, dtype='f4')
        assert '962705c861863495e9ccb7be7735907aa15e85b5' == z.hexdigest()

        # Check basic 2-D array
        z = self.create_array(shape=(20, 35,), chunks=10, dtype='i4')
        assert 'deb675ff91dd26dba11b65aab5f19a1f21a5645b' == z.hexdigest()

        # Check basic 1-D array with some data
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        z[200:400] = np.arange(200, 400, dtype='i4')
        assert '90e30bdab745a9641cd0eb605356f531bc8ec1c3' == z.hexdigest()

        # Check basic 1-D array with attributes
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        z.attrs['foo'] = 'bar'
        assert '95d40c391f167db8b1290e3c39d9bf741edacdf6' == z.hexdigest()


try:
    from numcodecs import LZMA
except ImportError:  # pragma: no cover
    LZMA = None


@unittest.skipIf(LZMA is None, 'LZMA codec not available')
class TestArrayWithLZMACompressor(TestArray):

    def create_array(self, read_only=False, **kwargs):
        store = dict()
        compressor = LZMA(preset=1)
        kwargs.setdefault('compressor', compressor)
        cache_metadata = kwargs.pop('cache_metadata', True)
        cache_attrs = kwargs.pop('cache_attrs', True)
        init_array(store, **kwargs)
        return Array(store, read_only=read_only, cache_metadata=cache_metadata,
                     cache_attrs=cache_attrs)

    def test_hexdigest(self):
        # Check basic 1-D array
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        assert '93ecaa530a1162a9d48a3c1dcee4586ccfc59bae' == z.hexdigest()

        # Check basic 1-D array with different type
        z = self.create_array(shape=(1050,), chunks=100, dtype='f4')
        assert '04a9755a0cd638683531b7816c7fa4fbb6f577f2' == z.hexdigest()

        # Check basic 2-D array
        z = self.create_array(shape=(20, 35,), chunks=10, dtype='i4')
        assert 'b93b163a21e8500519250a6defb821d03eb5d9e0' == z.hexdigest()

        # Check basic 1-D array with some data
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        z[200:400] = np.arange(200, 400, dtype='i4')
        assert 'cde499f3dc945b4e97197ff8e3cf8188a1262c35' == z.hexdigest()

        # Check basic 1-D array with attributes
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        z.attrs['foo'] = 'bar'
        assert 'e2cf3afbf66ad0e28a2b6b68b1b07817c69aaee2' == z.hexdigest()


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
        cache_metadata = kwargs.pop('cache_metadata', True)
        cache_attrs = kwargs.pop('cache_attrs', True)
        init_array(store, **kwargs)
        return Array(store, read_only=read_only, cache_attrs=cache_attrs,
                     cache_metadata=cache_metadata)

    def test_hexdigest(self):
        # Check basic 1-D array
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        assert 'b80367c5599d47110d42bd8886240c2f46620dba' == z.hexdigest()

        # Check basic 1-D array with different type
        z = self.create_array(shape=(1050,), chunks=100, dtype='f4')
        assert '95a7b2471225e73199c9716d21e8d3dd6e5f6f2a' == z.hexdigest()

        # Check basic 2-D array
        z = self.create_array(shape=(20, 35,), chunks=10, dtype='i4')
        assert '9abf3ad54413ab11855d88a5e0087cd416657e02' == z.hexdigest()

        # Check basic 1-D array with some data
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        z[200:400] = np.arange(200, 400, dtype='i4')
        assert 'c649ad229bc5720258b934ea958570c2f354c2eb' == z.hexdigest()

        # Check basic 1-D array with attributes
        z = self.create_array(shape=(1050,), chunks=100, dtype='i4')
        z.attrs['foo'] = 'bar'
        assert '62fc9236d78af18a5ec26c12eea1d33bce52501e' == z.hexdigest()

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
        assert z2.read_only

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

    def test_object_arrays_vlen_text(self):
        # skip this one, cannot use delta with objects
        pass

    def test_object_arrays_vlen_bytes(self):
        # skip this one, cannot use delta with objects
        pass

    def test_object_arrays_vlen_array(self):
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
        cache_metadata = kwargs.pop('cache_metadata', True)
        cache_attrs = kwargs.pop('cache_attrs', True)
        init_array(store, **kwargs)
        return Array(store, read_only=read_only, cache_metadata=cache_metadata,
                     cache_attrs=cache_attrs)

    def test_nbytes_stored(self):
        z = self.create_array(shape=1000, chunks=100)
        assert -1 == z.nbytes_stored
        z[:] = 42
        assert -1 == z.nbytes_stored


class TestArrayNoCache(TestArray):

    @staticmethod
    def create_array(read_only=False, **kwargs):
        store = dict()
        kwargs.setdefault('compressor', Zlib(level=1))
        cache_metadata = kwargs.pop('cache_metadata', True)
        cache_attrs = kwargs.pop('cache_attrs', True)
        init_array(store, **kwargs)
        return Array(store, read_only=read_only, cache_metadata=cache_metadata,
                     cache_attrs=cache_attrs)

    def test_cache_metadata(self):
        a1 = self.create_array(shape=100, chunks=10, dtype='i1', cache_metadata=False)
        a2 = Array(a1.store, cache_metadata=True)
        assert a1.shape == a2.shape
        assert a1.size == a2.size
        assert a1.nbytes == a2.nbytes
        assert a1.nchunks == a2.nchunks

        # a1 is not caching so *will* see updates made via other objects
        a2.resize(200)
        assert (200,) == a2.shape
        assert 200 == a2.size
        assert 200 == a2.nbytes
        assert 20 == a2.nchunks
        assert a1.shape == a2.shape
        assert a1.size == a2.size
        assert a1.nbytes == a2.nbytes
        assert a1.nchunks == a2.nchunks

        a2.append(np.zeros(100))
        assert (300,) == a2.shape
        assert 300 == a2.size
        assert 300 == a2.nbytes
        assert 30 == a2.nchunks
        assert a1.shape == a2.shape
        assert a1.size == a2.size
        assert a1.nbytes == a2.nbytes
        assert a1.nchunks == a2.nchunks

        # a2 is caching so *will not* see updates made via other objects
        a1.resize(400)
        assert (400,) == a1.shape
        assert 400 == a1.size
        assert 400 == a1.nbytes
        assert 40 == a1.nchunks
        assert (300,) == a2.shape
        assert 300 == a2.size
        assert 300 == a2.nbytes
        assert 30 == a2.nchunks

    def test_cache_attrs(self):
        a1 = self.create_array(shape=100, chunks=10, dtype='i1', cache_attrs=False)
        a2 = Array(a1.store, cache_attrs=True)
        assert a1.attrs.asdict() == a2.attrs.asdict()

        # a1 is not caching so *will* see updates made via other objects
        a2.attrs['foo'] = 'xxx'
        a2.attrs['bar'] = 42
        assert a1.attrs.asdict() == a2.attrs.asdict()

        # a2 is caching so *will not* see updates made via other objects
        a1.attrs['foo'] = 'yyy'
        assert 'yyy' == a1.attrs['foo']
        assert 'xxx' == a2.attrs['foo']

    def test_object_arrays_danger(self):
        # skip this one as it only works if metadata are cached
        pass


class TestArrayWithStoreCache(TestArray):

    @staticmethod
    def create_array(read_only=False, **kwargs):
        store = LRUStoreCache(dict(), max_size=None)
        kwargs.setdefault('compressor', Zlib(level=1))
        cache_metadata = kwargs.pop('cache_metadata', True)
        cache_attrs = kwargs.pop('cache_attrs', True)
        init_array(store, **kwargs)
        return Array(store, read_only=read_only, cache_metadata=cache_metadata,
                     cache_attrs=cache_attrs)
