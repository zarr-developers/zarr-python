# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import tempfile
import os
import shutil


from nose.tools import eq_ as eq, assert_false, assert_raises, assert_true
import numpy as np
from numpy.testing import assert_array_equal
from zarr import defaults
from zarr.ext import PersistentArray


def test_array_1d():
    a = np.arange(1050)

    path = tempfile.mktemp()
    assert_false(os.path.exists(path))
    
    # open for reading (does not exist)
    with assert_raises(ValueError):
        z = PersistentArray(path)
    with assert_raises(ValueError):
        z = PersistentArray(path, mode='r')
    # open for appending (does not exist)
    with assert_raises(ValueError):
        z = PersistentArray(path, mode='a')
        
    # open for writing
    z = PersistentArray(path, mode='w', shape=a.shape, chunks=100, 
                        dtype=a.dtype)
    
    # check directory creation
    assert_true(os.path.exists(path))
    
    # check properties
    eq(a.shape, z.shape)
    eq(a.dtype, z.dtype)
    eq((100,), z.chunks)
    eq((11,), z.cdata.shape)
    eq(defaults.cname, z.cname)
    eq(defaults.clevel, z.clevel)
    eq(defaults.shuffle, z.shuffle)
    eq(a.nbytes, z.nbytes)
    eq(0, z.cbytes)

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

    # open for reading
    z2 = PersistentArray(path, mode='r')
    eq(a.shape, z2.shape)
    eq(a.dtype, z2.dtype)
    eq((100,), z2.chunks)
    eq((11,), z2.cdata.shape)
    eq(defaults.cname, z2.cname)
    eq(defaults.clevel, z2.clevel)
    eq(defaults.shuffle, z2.shuffle)
    eq(a.nbytes, z2.nbytes)
    eq(z.cbytes, z2.cbytes)

    # check data
    assert_array_equal(a, z2[:])

    # check read-only
    with assert_raises(ValueError):
        z2[:] = 0
        
    # open for appending
    z3 = PersistentArray(path, mode='a')
    eq(a.shape, z3.shape)
    eq(a.dtype, z3.dtype)
    eq((100,), z3.chunks)
    eq((11,), z3.cdata.shape)
    eq(defaults.cname, z3.cname)
    eq(defaults.clevel, z3.clevel)
    eq(defaults.shuffle, z3.shuffle)
    eq(a.nbytes, z3.nbytes)
    eq(z.cbytes, z3.cbytes)

    # check can write
    z3[:] = 0

    # check effect of write
    assert_array_equal(np.zeros_like(a), z3[:])
    assert_array_equal(np.zeros_like(a), z2[:])
    assert_array_equal(np.zeros_like(a), z[:])

    # tidy up
    shutil.rmtree(path)


def test_array_1d_fill_value():

    for fill_value in -1, 0, 1, 10:

        path = tempfile.mktemp()

        a = np.arange(1050)
        f = np.empty_like(a)
        f.fill(fill_value)
        z = PersistentArray(path, mode='w', shape=a.shape, chunks=100,
                            fill_value=fill_value)
        z[190:310] = a[190:310]

        assert_array_equal(f[:190], z[:190])
        assert_array_equal(a[190:310], z[190:310])
        assert_array_equal(f[310:], z[310:])

        z2 = PersistentArray(path, mode='r')

        assert_array_equal(f[:190], z2[:190])
        assert_array_equal(a[190:310], z2[190:310])
        assert_array_equal(f[310:], z2[310:])

        shutil.rmtree(path)


def test_array_2d():

    a = np.arange(10000).reshape((1000, 10))
    path = tempfile.mktemp()
    
    z = PersistentArray(path, mode='w', shape=a.shape, chunks=(100, 2), 
                        dtype=a.dtype)

    # check properties
    eq(a.shape, z.shape)
    eq(a.dtype, z.dtype)
    eq((100, 2), z.chunks)
    eq((10, 5), z.cdata.shape)
    eq(defaults.cname, z.cname)
    eq(defaults.clevel, z.clevel)
    eq(defaults.shuffle, z.shuffle)
    eq(a.nbytes, z.nbytes)
    eq(0, z.cbytes)

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
    
    # check open for reading
    z2 = PersistentArray(path, mode='r')

    # check properties
    eq(a.shape, z2.shape)
    eq(a.dtype, z2.dtype)
    eq((100, 2), z2.chunks)
    eq((10, 5), z2.cdata.shape)
    eq(defaults.cname, z2.cname)
    eq(defaults.clevel, z2.clevel)
    eq(defaults.shuffle, z2.shuffle)
    eq(a.nbytes, z2.nbytes)
    eq(z.cbytes, z2.cbytes)

    # check data    
    assert_array_equal(a, z2[:])
    assert_array_equal(a, z2[...])

    # check slicing
    assert_array_equal(a[:10], z2[:10])
    assert_array_equal(a[10:20], z2[10:20])
    assert_array_equal(a[-10:], z2[-10:])
    assert_array_equal(a[:, :2], z2[:, :2])
    assert_array_equal(a[:, 2:4], z2[:, 2:4])
    assert_array_equal(a[:, -2:], z2[:, -2:])
    assert_array_equal(a[:10, :2], z2[:10, :2])
    assert_array_equal(a[10:20, 2:4], z2[10:20, 2:4])
    assert_array_equal(a[-10:, -2:], z2[-10:, -2:])
    # ...across chunk boundaries...
    assert_array_equal(a[:110], z2[:110])
    assert_array_equal(a[190:310], z2[190:310])
    assert_array_equal(a[-110:], z2[-110:])
    assert_array_equal(a[:, :3], z2[:, :3])
    assert_array_equal(a[:, 3:7], z2[:, 3:7])
    assert_array_equal(a[:, -3:], z2[:, -3:])
    assert_array_equal(a[:110, :3], z2[:110, :3])
    assert_array_equal(a[190:310, 3:7], z2[190:310, 3:7])
    assert_array_equal(a[-110:, -3:], z2[-110:, -3:])
    

# TODO test resize
# TODO test append
