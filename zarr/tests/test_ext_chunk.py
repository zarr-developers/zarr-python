# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


from nose.tools import eq_ as eq
import numpy as np
from numpy.testing import assert_array_equal
from zarr.ext import Chunk
from zarr import defaults


def _test_create_chunk_default(a):
    c = Chunk(a.shape, a.dtype)
    c[:] = a

    # check properties
    eq(a.shape, c.shape)
    eq(a.dtype, c.dtype)
    eq(a.nbytes, c.nbytes)
    eq(defaults.cname, c.cname)
    eq(defaults.clevel, c.clevel)
    eq(defaults.shuffle, c.shuffle)

    # check compression is sane
    assert c.cbytes < c.nbytes
    assert c.blocksize <= c.nbytes

    # check round-trip
    assert_array_equal(a, c[:])
    assert_array_equal(a, c[...])


def _test_create_chunk_cparams(a, cname, clevel, shuffle):
    c = Chunk(a.shape, a.dtype, cname, clevel, shuffle)
    c[:] = a

    # check properties
    eq(a.shape, c.shape)
    eq(a.dtype, c.dtype)
    eq(a.nbytes, c.nbytes)
    eq(cname, c.cname)
    eq(clevel, c.clevel)
    eq(shuffle, c.shuffle)

    # check compression is sane
    assert c.blocksize <= c.nbytes
    if clevel > 0 and shuffle > 0:
        # N.B., for some arrays, shuffle is required to achieve any compression
        assert c.cbytes < c.nbytes, (c.nbytes, c.cbytes)

    # check round-trip
    assert_array_equal(a, c[:])
    assert_array_equal(a, c[...])


def _test_create_chunk(a):
    _test_create_chunk_default(a)
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

        # default dtype and fill_value
        c = Chunk(shape)
        a = c[:]
        e = np.empty(shape)
        eq(e.shape, a.shape)
        eq(e.dtype, a.dtype)

        # specified dtype and fill_value
        for dtype in 'i4', 'f8':
            for fill_value in 1, -1:
                c = Chunk(shape, dtype=dtype, fill_value=fill_value)
                e = np.empty(shape, dtype=dtype)
                e.fill(fill_value)
                assert_array_equal(e, c[:])
