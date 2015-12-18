# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import tempfile
import os


import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import eq_ as eq, assert_false, assert_true


from zarr.ext import PersistentChunk
from zarr import defaults


def _test_create_chunk_cparams(a, cname, clevel, shuffle):
    
    # setup file
    path = tempfile.mktemp()
    assert_false(os.path.exists(path))
    
    # instantiate a persistent chunk
    c = PersistentChunk(path, a.shape, a.dtype, cname, clevel, shuffle)
    
    # check properties
    eq(a.shape, c.shape)
    eq(a.dtype, c.dtype)
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
    eq(0, c.nbytes)
    eq(0, c.cbytes)
    assert_false(c.is_initialised)
    
    # store some data
    c[:] = a
    
    # check properties after storing data
    eq(a.shape, c.shape)
    eq(a.dtype, c.dtype)
    eq(a.nbytes, c.nbytes)
    if c.clevel > 0 and c.shuffle > 0:
        # N.B., for some arrays, shuffle is required to achieve any compression
        assert c.cbytes < c.nbytes, (c.nbytes, c.cbytes)
    assert_true(c.is_initialised)
    assert_true(os.path.exists(path))

    # check round trip
    assert_array_equal(a, c[:])
    assert_array_equal(a, c[...])
    
    # check persistence
    c2 = PersistentChunk(path, a.shape, a.dtype, cname, clevel, shuffle)
    eq(a.shape, c2.shape)
    eq(a.dtype, c2.dtype)
    eq(a.nbytes, c2.nbytes)
    if c.clevel > 0 and c.shuffle > 0:
        # N.B., for some arrays, shuffle is required to achieve any compression
        assert c2.cbytes < c2.nbytes, (c2.nbytes, c2.cbytes)
    assert_true(c2.is_initialised)
    assert_array_equal(a, c2[:])
    assert_array_equal(a, c2[...])

    # tidy up    
    os.remove(path)


def _test_create_chunk(a):
    for cname in b'blosclz', b'lz4', b'snappy', b'zlib':
        for clevel in 0, 1, 5:
            for shuffle in 0, 1, 2:
                print(cname, clevel, shuffle)
                _test_create_chunk_cparams(a, cname, clevel, shuffle)


def test_create_chunk():

    # arange
    for dtype in 'u1', 'u4', 'u8', 'i1', 'i4', 'i8':
        print(dtype)
        print('1-dimensional')
        _test_create_chunk(np.arange(1e5, dtype=dtype))
        print('2-dimensional')
        _test_create_chunk(np.arange(1e5, dtype=dtype).reshape((100, -1)))

    # linspace
    for dtype in 'f2', 'f4', 'f8':
        print(dtype)
        print('1-dimensional')
        _test_create_chunk(np.linspace(-1, 1, 1e5, dtype=dtype))
        print('2-dimensional')
        _test_create_chunk(np.linspace(-1, 1, 1e5, dtype=dtype).reshape((100, -1)))


def test_create_chunk_fill_value():

    for shape in 100, (100, 100):

        path = tempfile.mktemp()
        assert_false(os.path.exists(path))

        # default dtype and fill_value
        c = PersistentChunk(path, shape)
        a = c[:]
        e = np.empty(shape)
        eq(e.shape, a.shape)
        eq(e.dtype, a.dtype)
        assert_false(c.is_initialised)
        assert_false(os.path.exists(path))

        # specified dtype and fill_value
        for dtype in 'i4', 'f8':
            for fill_value in 1, -1:
                c = PersistentChunk(path, shape, dtype=dtype,
                                    fill_value=fill_value)
                e = np.empty(shape, dtype=dtype)
                e.fill(fill_value)
                assert_array_equal(e, c[:])
                assert_false(c.is_initialised)
                assert_false(os.path.exists(path))
