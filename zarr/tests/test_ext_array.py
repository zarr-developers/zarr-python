# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


from nose.tools import eq_ as eq
import numpy as np
from numpy.testing import assert_array_equal
from zarr import defaults
from zarr.ext import Array


def test_array_1d():

    a = np.arange(1050)
    z = Array(a.shape, chunks=100, dtype=a.dtype)

    # check properties
    eq(a.shape, z.shape)
    eq(a.dtype, z.dtype)
    eq((100,), z.chunks)
    eq((11,), z.cdata.shape)
    eq(defaults.cname, z.cname)
    eq(defaults.clevel, z.clevel)
    eq(defaults.shuffle, z.shuffle)

    # set data
    z[:] = a

    # check properties
    eq(a.nbytes, z.nbytes)
    eq(sum(c.cbytes for c in z.cdata.flat), z.cbytes)

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
    z = Array(a.shape, chunks=100)
    z[:] = a
    assert_array_equal(a, z[:])
    z[190:310] = b[190:310]
    assert_array_equal(a[:190], z[:190])
    assert_array_equal(b[190:310], z[190:310])
    assert_array_equal(a[310:], z[310:])


def test_array_1d_fill_value():

    for fill_value in -1, 0, 1, 10:

        a = np.arange(1050)
        f = np.empty_like(a)
        f.fill(fill_value)
        z = Array(a.shape, chunks=100, fill_value=fill_value)
        z[190:310] = a[190:310]

        assert_array_equal(f[:190], z[:190])
        assert_array_equal(a[190:310], z[190:310])
        assert_array_equal(f[310:], z[310:])


def test_array_1d_set_scalar():

    # setup
    a = np.empty(100)
    z = Array(a.shape, chunks=10, dtype=a.dtype)
    z[:] = a
    assert_array_equal(a, z[:])

    for value in -1, 0, 1, 10:
        a[15:35] = value
        z[15:35] = value
        assert_array_equal(a, z[:])
        a[:] = value
        z[:] = value
        assert_array_equal(a, z[:])


def test_array_2d():

    a = np.arange(10000).reshape((1000, 10))
    z = Array(a.shape, chunks=(100, 2), dtype=a.dtype)

    # check properties
    eq(a.shape, z.shape)
    eq(a.dtype, z.dtype)
    eq((100, 2), z.chunks)
    eq((10, 5), z.cdata.shape)
    eq(defaults.cname, z.cname)
    eq(defaults.clevel, z.clevel)
    eq(defaults.shuffle, z.shuffle)

    # set data
    z[:] = a

    # check properties
    eq(a.nbytes, z.nbytes)
    eq(sum(c.cbytes for c in z.cdata.flat), z.cbytes)

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
    z = Array(a.shape, chunks=(100, 2), dtype=a.dtype)
    z[:] = a
    assert_array_equal(a, z[:])
    z[190:310, 3:7] = b[190:310, 3:7]
    assert_array_equal(a[:190], z[:190])
    assert_array_equal(a[:, :3], z[:, :3])
    assert_array_equal(b[190:310, 3:7], z[190:310, 3:7])
    assert_array_equal(a[310:], z[310:])
    assert_array_equal(a[:, 7:], z[:, 7:])


def test_resize_1d():

    z = Array(105, chunks=10, dtype='i4', fill_value=0)
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


def test_resize_2d():

    z = Array((105, 105), chunks=(10, 10), dtype='i4', fill_value=0)
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


def test_append_1d():

    a = np.arange(105, dtype='i4')
    z = Array(a.shape, chunks=10, dtype=a.dtype)
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


def test_append_2d():

    a = np.arange(105*105, dtype='i4').reshape((105, 105))
    z = Array(a.shape, chunks=(10, 10), dtype=a.dtype)
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


def test_append_2d_axis():

    a = np.arange(105*105, dtype='i4').reshape((105, 105))
    z = Array(a.shape, chunks=(10, 10), dtype=a.dtype)
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
