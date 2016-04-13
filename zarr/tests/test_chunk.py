# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import tempfile
from unittest import TestCase
import atexit
import os


from nose.tools import eq_ as eq, assert_false, assert_true, assert_raises
import numpy as np
from numpy.testing import assert_array_equal
from zarr.ext import Chunk, PersistentChunk, SynchronizedChunk, \
    SynchronizedPersistentChunk
from zarr import defaults


class ChunkTests(object):

    def create_chunk(self, **kwargs):
        # override in sub-classes
        raise NotImplementedError()

    def _test_create_chunk_cparams(self, a, cname, clevel, shuffle):

        # setup
        c = self.create_chunk(shape=a.shape, dtype=a.dtype, cname=cname,
                              clevel=clevel, shuffle=shuffle)

        # check basic properties
        eq(a.shape, c.shape)
        eq(a.dtype, c.dtype)
        eq(a.size, c.size)
        eq(a.itemsize, c.itemsize)
        eq(a.nbytes, c.nbytes)

        # check compression properties
        if cname is None:
            eq(defaults.cname, c.cname)
        else:
            eq(cname, c.cname)
        if clevel is None:
            eq(defaults.clevel, c.clevel)
        else:
            eq(clevel, c.clevel)
        if shuffle is None:
            eq(defaults.shuffle, c.shuffle)
        else:
            eq(shuffle, c.shuffle)
        eq(0, c.cbytes)
        assert_false(c.is_initialized)

        # store data
        c[:] = a

        # check properties after storing data
        eq(a.shape, c.shape)
        eq(a.dtype, c.dtype)
        eq(a.size, c.size)
        eq(a.itemsize, c.itemsize)
        eq(a.nbytes, c.nbytes)
        assert_true(c.cbytes > 0)
        assert_true(c.is_initialized)

        # check round-trip
        assert_array_equal(a, c[:])
        assert_array_equal(a, c[...])

    def _test_create_chunk(self, a):
        for cname in None, b'blosclz', b'lz4', b'snappy', b'zlib':
            for clevel in None, 0, 5:
                for shuffle in None, 0, 1, 2:
                    print(cname, clevel, shuffle)
                    self._test_create_chunk_cparams(a, cname, clevel, shuffle)

    def test_create_chunk(self):

        # arange
        for dtype in 'u1', 'u4', 'u8', 'i1', 'i4', 'i8':
            print(dtype)
            print('1-dimensional')
            self._test_create_chunk(np.arange(1e4, dtype=dtype))
            print('2-dimensional')
            self._test_create_chunk(np.arange(1e4, dtype=dtype)
                                    .reshape(100, -1))

        # linspace
        for dtype in 'f2', 'f4', 'f8':
            print(dtype)
            print('1-dimensional')
            self._test_create_chunk(np.linspace(-1, 1, 1e4, dtype=dtype))
            print('2-dimensional')
            self._test_create_chunk(np.linspace(-1, 1, 1e4, dtype=dtype)
                                    .reshape(100, -1))

        # structured dtype
        dtype = np.dtype([('a', 'i4'), ('b', 'S10')])
        print(dtype)
        print('1-dimensional')
        self._test_create_chunk(np.empty(10000, dtype=dtype))
        print('2-dimensional')
        self._test_create_chunk(np.empty((100, 100), dtype=dtype))

    def test_create_chunk_fill_value(self):

        for shape in 100, (100, 100):

            # default dtype and fill_value
            c = self.create_chunk(shape=shape)
            a = c[:]
            e = np.empty(shape)
            eq(e.shape, a.shape)
            eq(e.dtype, a.dtype)

            # specified dtype and fill_value
            for dtype in 'i4', 'f8':
                for fill_value in 1, -1:
                    c = self.create_chunk(shape=shape, dtype=dtype,
                                          fill_value=fill_value)
                    e = np.empty(shape, dtype=dtype)
                    e.fill(fill_value)
                    assert_array_equal(e, c[:])


class TestChunk(TestCase, ChunkTests):

    def create_chunk(self, **kwargs):
        return Chunk(**kwargs)


class TestSynchronizedChunk(TestCase, ChunkTests):

    def create_chunk(self, **kwargs):
        return SynchronizedChunk(**kwargs)


class TestPersistentChunk(TestCase, ChunkTests):

    def create_chunk(self, **kwargs):
        path = kwargs.get('path', tempfile.mktemp())
        kwargs['path'] = path
        # tidy up
        atexit.register(
            lambda: os.remove(path) if os.path.exists(path) else None
        )
        return PersistentChunk(**kwargs)

    def _test_persistence_cparams(self, a, cname, clevel, shuffle):

        # setup file
        path = tempfile.mktemp()
        assert_false(os.path.exists(path))

        # create chunk
        c = self.create_chunk(path=path, shape=a.shape, dtype=a.dtype,
                              cname=cname, clevel=clevel, shuffle=shuffle)

        # check state
        assert_false(os.path.exists(path))
        assert_false(c.is_initialized)

        # store some data
        c[:] = a

        # check state
        assert_true(os.path.exists(path))
        assert_true(c.is_initialized)

        # check persistence
        c2 = self.create_chunk(path=path, shape=a.shape, dtype=a.dtype,
                               cname=cname, clevel=clevel, shuffle=shuffle)

        # check state
        eq(a.shape, c2.shape)
        eq(a.dtype, c2.dtype)
        if cname is None:
            eq(defaults.cname, c2.cname)
        else:
            eq(cname, c2.cname)
        if clevel is None:
            eq(defaults.clevel, c2.clevel)
        else:
            eq(clevel, c2.clevel)
        if shuffle is None:
            eq(defaults.shuffle, c2.shuffle)
        else:
            eq(shuffle, c2.shuffle)
        eq(a.nbytes, c2.nbytes)
        assert_true(c.nbytes, c2.nbytes)
        assert_true(c2.is_initialized)

        # check data
        assert_array_equal(a, c2[:])
        assert_array_equal(a, c2[...])

        # check what happens if you do something stupid
        with assert_raises(ValueError):
            self.create_chunk(path=path, shape=[2*i for i in a.shape],
                              dtype=a.dtype, cname=cname, clevel=clevel,
                              shuffle=shuffle)
        with assert_raises(ValueError):
            self.create_chunk(path=path, shape=a.shape, dtype='S7',
                              cname=cname, clevel=clevel, shuffle=shuffle)

    def _test_persistence(self, a):
        for cname in None, b'blosclz', b'lz4', b'snappy', b'zlib':
            for clevel in None, 0, 1, 5:
                for shuffle in None, 0, 1, 2:
                    print(cname, clevel, shuffle)
                    self._test_persistence_cparams(a, cname, clevel, shuffle)

    def test_persistence(self):

        # arange
        for dtype in 'u1', 'u4', 'u8', 'i1', 'i4', 'i8':
            print(dtype)
            print('1-dimensional')
            self._test_persistence(np.arange(1e4, dtype=dtype))
            print('2-dimensional')
            self._test_persistence(np.arange(1e4, dtype=dtype)
                                   .reshape(100, -1))

        # linspace
        for dtype in 'f2', 'f4', 'f8':
            print(dtype)
            print('1-dimensional')
            self._test_persistence(np.linspace(-1, 1, 1e4, dtype=dtype))
            print('2-dimensional')
            self._test_persistence(np.linspace(-1, 1, 1e4, dtype=dtype)
                                   .reshape(100, -1))


class TestSynchronizedPersistentChunk(TestPersistentChunk):

    def create_chunk(self, **kwargs):
        path = kwargs.get('path', tempfile.mktemp())
        kwargs['path'] = path
        # tidy up
        atexit.register(
            lambda: os.remove(path) if os.path.exists(path) else None
        )
        return SynchronizedPersistentChunk(**kwargs)


def test_shuffles():

    # setup
    a = np.arange(256, dtype='u1')
    # no shuffle
    c0 = Chunk(shape=a.shape, dtype=a.dtype, shuffle=0)
    c0[:] = a
    # byte shuffle
    c1 = Chunk(shape=a.shape, dtype=a.dtype, shuffle=1)
    c1[:] = a
    # bit shuffle
    c2 = Chunk(shape=a.shape, dtype=a.dtype, shuffle=2)
    c2[:] = a

    # expect no effect of byte shuffle because using single byte dtype
    assert c0.cbytes == c1.cbytes

    # expect improvement from bitshuffle
    assert c2.cbytes < c1.cbytes
