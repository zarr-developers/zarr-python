# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import tempfile
from unittest import TestCase
import atexit
import os
import shutil


from nose.tools import eq_ as eq, assert_false, assert_true, assert_raises
import numpy as np
from numpy.testing import assert_array_equal
from zarr.ext import Array, SynchronizedArray, PersistentArray, \
    SynchronizedPersistentArray, LazyArray, SynchronizedLazyArray, \
    LazyPersistentArray, SynchronizedLazyPersistentArray
from zarr import defaults


class ArrayTests(object):

    def create_array(self, **kwargs):
        raise NotImplementedError()

    def test_1d(self):

        a = np.arange(1050)
        z = self.create_array(shape=a.shape, chunks=100, dtype=a.dtype)

        # check properties
        eq(a.shape, z.shape)
        eq(a.dtype, z.dtype)
        eq((100,), z.chunks)
        eq((11,), z.cdata_shape)
        eq(defaults.cname, z.cname)
        eq(defaults.clevel, z.clevel)
        eq(defaults.shuffle, z.shuffle)
        eq(a.nbytes, z.nbytes)
        eq(0, z.cbytes)
        eq(0, np.count_nonzero(z.is_initialized))

        # set data
        z[:] = a

        # check properties
        eq(a.nbytes, z.nbytes)
        eq(sum(c.cbytes for c in z.iter_chunks()), z.cbytes)
        eq(11, np.count_nonzero(z.is_initialized))

        # check round-trip
        assert_array_equal(a, z[:])
        assert_array_equal(a, z[...])

        # check slicing
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
        eq((10, 5), z.cdata_shape)
        eq(defaults.cname, z.cname)
        eq(defaults.clevel, z.clevel)
        eq(defaults.shuffle, z.shuffle)
        eq(0, np.count_nonzero(z.is_initialized))

        # set data
        z[:] = a

        # check properties
        eq(a.nbytes, z.nbytes)
        eq(sum(c.cbytes for c in z.iter_chunks()), z.cbytes)
        eq(50, np.count_nonzero(z.is_initialized))

        # check round-trip
        assert_array_equal(a, z[:])
        assert_array_equal(a, z[...])

        # check slicing
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
        assert_array_equal(a[:55], z[:55])

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

        a = np.arange(105, dtype='i4')
        z = self.create_array(shape=a.shape, chunks=10, dtype=a.dtype)
        z[:] = a
        eq(a.shape, z.shape)
        eq(a.dtype, z.dtype)
        eq((10,), z.chunks)
        assert_array_equal(a, z[:])

        b = np.arange(105, 205, dtype='i4')
        e = np.append(a, b)
        z.append(b)
        eq(e.shape, z.shape)
        eq(e.dtype, z.dtype)
        eq((10,), z.chunks)
        assert_array_equal(e, z[:])

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


class TestArray(TestCase, ArrayTests):

    def create_array(self, **kwargs):
        return Array(**kwargs)


class TestSynchronizedArray(TestCase, ArrayTests):

    def create_array(self, **kwargs):
        return SynchronizedArray(**kwargs)


class TestPersistentArray(TestCase, ArrayTests):

    def create_array(self, **kwargs):
        path = kwargs.get('path', tempfile.mktemp())
        kwargs['path'] = path
        # tidy up
        atexit.register(
            lambda: shutil.rmtree(path) if os.path.exists(path) else None
        )
        return PersistentArray(**kwargs)

    def _test_persistence(self, a, chunks):

        # setup path
        path = tempfile.mktemp()
        assert_false(os.path.exists(path))

        # open for reading (does not exist)
        with assert_raises(ValueError):
            self.create_array(path=path, mode='r')
        with assert_raises(ValueError):
            self.create_array(path=path, mode='r+')

        # open for appending
        z = self.create_array(path=path, mode='a', shape=a.shape,
                              chunks=chunks, dtype=a.dtype)

        # check directory creation
        assert_true(os.path.exists(path))

        # set data
        z[:] = a

        # open for reading
        z2 = self.create_array(path=path, mode='r')
        eq(a.shape, z2.shape)
        eq(a.dtype, z2.dtype)
        eq(chunks, z2.chunks)
        eq(defaults.cname, z2.cname)
        eq(defaults.clevel, z2.clevel)
        eq(defaults.shuffle, z2.shuffle)
        eq(a.nbytes, z2.nbytes)
        eq(z.cbytes, z2.cbytes)
        assert_array_equal(z.is_initialized, z2.is_initialized)
        assert_true(np.count_nonzero(z2.is_initialized) > 0)
        assert_array_equal(a, z2[:])

        # check read-only
        with assert_raises(ValueError):
            z2[:] = 0
        with assert_raises(ValueError):
            z2.resize(100)

        # open for read/write if exists
        z3 = self.create_array(path=path, mode='r+')
        eq(a.shape, z3.shape)
        eq(a.dtype, z3.dtype)
        eq(chunks, z3.chunks)
        eq(defaults.cname, z3.cname)
        eq(defaults.clevel, z3.clevel)
        eq(defaults.shuffle, z3.shuffle)
        eq(a.nbytes, z3.nbytes)
        eq(z.cbytes, z3.cbytes)
        assert_array_equal(z.is_initialized, z3.is_initialized)
        assert_true(np.count_nonzero(z3.is_initialized) > 0)
        assert_array_equal(a, z3[:])

        # check can write
        z3[:] = 0

        # check effect of write
        expect = np.empty_like(a)
        expect[:] = 0
        assert_array_equal(expect, z3[:])
        assert_array_equal(expect, z2[:])
        assert_array_equal(expect, z[:])

        # open for writing (must not exist)
        with assert_raises(ValueError):
            self.create_array(path=path, mode='w-')
            
        # tidy up
        shutil.rmtree(path)

    def test_persistence_1d(self):
        # simple dtype
        self._test_persistence(np.arange(1050), chunks=(100,))
        # structured dtype
        dtype = np.dtype([('a', 'i4'), ('b', 'S10')])
        self._test_persistence(np.empty(10000, dtype=dtype), chunks=(100,))

    def test_persistence_2d(self):
        chunks = (100, 2)
        # simple dtype
        a = np.arange(10000).reshape((1000, 10))
        self._test_persistence(a, chunks=chunks)
        # structured dtype
        dtype = np.dtype([('a', 'i4'), ('b', 'S10')])
        self._test_persistence(np.empty((1000, 10), dtype=dtype),
                               chunks=chunks)

    def test_resize_persistence(self):

        # setup path
        path = tempfile.mktemp()
        assert_false(os.path.exists(path))

        # setup array
        a = np.arange(1050, dtype='i4')
        z = self.create_array(path=path, shape=a.shape,
                              dtype=a.dtype, chunks=100,
                              fill_value=0)
        z[:] = a

        # resize
        z.resize(2100)
        z[1050:] = a
        
        # re-open
        z2 = self.create_array(path=path, mode='r')
        
        # check resize is persistent
        eq((2100,), z2.shape)
        eq(a.dtype, z2.dtype)
        eq((100,), z2.chunks)
        eq(defaults.cname, z2.cname)
        eq(defaults.clevel, z2.clevel)
        eq(defaults.shuffle, z2.shuffle)
        eq(2 * a.nbytes,
           z2.nbytes)
        eq(z.cbytes, z2.cbytes)
        assert_array_equal(a, z2[:1050])
        assert_array_equal(a, z2[1050:])
        
        # resize again
        z.resize(525)
        
        # re-open
        z3 = self.create_array(path=path, mode='r')
        
        # check resize is persistent
        eq((525,), z3.shape)
        eq(a.dtype, z3.dtype)
        eq((100,), z3.chunks)
        eq(defaults.cname, z3.cname)
        eq(defaults.clevel, z3.clevel)
        eq(defaults.shuffle, z3.shuffle)
        eq(a.nbytes/2, z3.nbytes)
        eq(z.cbytes, z3.cbytes)
        assert_array_equal(a[:525], z3[:])
        
        # tidy up
        shutil.rmtree(path)


class TestSynchronizedPersistentArray(TestPersistentArray):

    def create_array(self, **kwargs):
        path = kwargs.get('path', tempfile.mktemp())
        kwargs['path'] = path
        # tidy up
        atexit.register(
            lambda: shutil.rmtree(path) if os.path.exists(path) else None
        )
        return SynchronizedPersistentArray(**kwargs)


class TestLazyArray(TestCase, ArrayTests):

    def create_array(self, **kwargs):
        return LazyArray(**kwargs)


class TestSynchronizedLazyArray(TestCase, ArrayTests):

    def create_array(self, **kwargs):
        return SynchronizedLazyArray(**kwargs)


class TestLazyPersistentArray(TestPersistentArray):

    def create_array(self, **kwargs):
        path = kwargs.get('path', tempfile.mktemp())
        kwargs['path'] = path
        # tidy up
        atexit.register(
            lambda: shutil.rmtree(path) if os.path.exists(path) else None
        )
        return LazyPersistentArray(**kwargs)


class TestSynchronizedLazyPersistentArray(TestPersistentArray):

    def create_array(self, **kwargs):
        path = kwargs.get('path', tempfile.mktemp())
        kwargs['path'] = path
        # tidy up
        atexit.register(
            lambda: shutil.rmtree(path) if os.path.exists(path) else None
        )
        return SynchronizedLazyPersistentArray(**kwargs)
