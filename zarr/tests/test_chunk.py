# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


from nose.tools import eq_ as eq
import numpy as np
import zarr


def _test_create_default(a):
    c = zarr.Chunk(a)

    # check properties
    eq(a.shape, c.shape)
    eq(a.dtype, c.dtype)
    eq(a.size, c.size)
    eq(a.size * a.dtype.itemsize, c.nbytes)
    eq(zarr.defaults.cname, c.cname)
    eq(zarr.defaults.clevel, c.clevel)
    eq(zarr.defaults.shuffle, c.shuffle)

    # check compression is sane
    assert c.cbytes < c.nbytes
    assert c.blocksize < c.nbytes

    # check round-trip
    assert np.array_equal(a, np.array(c))


def _test_create_cparams(a, cname, clevel, shuffle):
    c = zarr.Chunk(a, cname, clevel, shuffle)
    # check properties
    eq(a.shape, c.shape)
    eq(a.dtype, c.dtype)
    eq(a.size, c.size)
    eq(a.size * a.dtype.itemsize, c.nbytes)
    eq(cname, c.cname)
    eq(clevel, c.clevel)
    eq(shuffle, c.shuffle)

    # check compression is sane
    assert c.blocksize <= c.nbytes
    if clevel > 0 and shuffle > 0:
        # N.B., for some arrays, shuffle is required to achieve any compression
        assert c.cbytes < c.nbytes, (c.nbytes, c.cbytes)

    # check round-trip
    assert np.array_equal(a, np.array(c))


def _test_create(a):
    _test_create_default(a)
    for cname in b'blosclz', b'lz4', b'snappy', b'zlib':
        for clevel in 0, 1, 5, 9:
            for shuffle in 0, 1, 2:
                print(cname, clevel, shuffle)
                _test_create_cparams(a, cname, clevel, shuffle)


def test_create():

    # arange
    for dtype in 'u1', 'u4', 'u8', 'i1', 'i4', 'i8':
        print(dtype)
        print('1-dimensional')
        _test_create(np.arange(1e5, dtype=dtype))
        print('2-dimensional')
        _test_create(np.arange(1e5, dtype=dtype).reshape((100, -1)))

    # linspace
    for dtype in 'f2', 'f4', 'f8':
        print(dtype)
        print('1-dimensional')
        _test_create(np.linspace(-1, 1, 1e5, dtype=dtype))
        print('2-dimensional')
        _test_create(np.linspace(-1, 1, 1e5, dtype=dtype).reshape((100, -1)))
