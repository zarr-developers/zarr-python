# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import tempfile
import shutil
import atexit
import os
import numpy as np
from nose.tools import eq_ as eq
from numpy.testing import assert_array_equal


from zarr.create import array, empty, zeros, ones, full, open, empty_like, \
    zeros_like, ones_like, full_like, open_like


def test_array():
    a = np.arange(100)
    z = array(a, chunks=10)
    eq(a.shape, z.shape)
    eq(a.dtype, z.dtype)
    assert_array_equal(a, z[:])


def test_empty():
    z = empty(100, 10)
    eq((100,), z.shape)
    eq((10,), z.chunks)


def test_zeros():
    z = zeros(100, 10)
    eq((100,), z.shape)
    eq((10,), z.chunks)
    assert_array_equal(np.zeros(100), z[:])


def test_ones():
    z = ones(100, 10)
    eq((100,), z.shape)
    eq((10,), z.chunks)
    assert_array_equal(np.ones(100), z[:])


def test_full():
    z = full(100, 10, fill_value=42, dtype='i4')
    eq((100,), z.shape)
    eq((10,), z.chunks)
    assert_array_equal(np.full(100, fill_value=42, dtype='i4'), z[:])


def test_open():
    path = tempfile.mktemp()
    atexit.register(
        lambda: shutil.rmtree(path) if os.path.exists(path) else None
    )
    z = open(path, mode='w', shape=100, chunks=10, dtype='i4')
    z[:] = 42
    eq((100,), z.shape)
    eq((10,), z.chunks)
    assert_array_equal(np.full(100, fill_value=42, dtype='i4'), z[:])
    z2 = open(path, mode='r')
    eq((100,), z2.shape)
    eq((10,), z2.chunks)
    assert_array_equal(z[:], z2[:])


def test_empty_like():
    z = empty(100, 10)
    z2 = empty_like(z)
    eq(z.shape, z2.shape)
    eq(z.chunks, z2.chunks)
    eq(z.dtype, z2.dtype)
    eq(z.cname, z2.cname)
    eq(z.clevel, z2.clevel)
    eq(z.shuffle, z2.shuffle)
    eq(z.fill_value, z2.fill_value)


def test_zeros_like():
    z = zeros(100, 10)
    z2 = zeros_like(z)
    eq(z.shape, z2.shape)
    eq(z.chunks, z2.chunks)
    eq(z.dtype, z2.dtype)
    eq(z.cname, z2.cname)
    eq(z.clevel, z2.clevel)
    eq(z.shuffle, z2.shuffle)
    eq(z.fill_value, z2.fill_value)


def test_ones_like():
    z = ones(100, 10)
    z2 = ones_like(z)
    eq(z.shape, z2.shape)
    eq(z.chunks, z2.chunks)
    eq(z.dtype, z2.dtype)
    eq(z.cname, z2.cname)
    eq(z.clevel, z2.clevel)
    eq(z.shuffle, z2.shuffle)
    eq(z.fill_value, z2.fill_value)


def test_full_like():
    z = full(100, 10, fill_value=42)
    z2 = full_like(z)
    eq(z.shape, z2.shape)
    eq(z.chunks, z2.chunks)
    eq(z.dtype, z2.dtype)
    eq(z.cname, z2.cname)
    eq(z.clevel, z2.clevel)
    eq(z.shuffle, z2.shuffle)
    eq(z.fill_value, z2.fill_value)


def test_open_like():
    path = tempfile.mktemp()
    atexit.register(
        lambda: shutil.rmtree(path) if os.path.exists(path) else None
    )
    z = full(100, 10, fill_value=42)
    z2 = open_like(z, path)
    eq(z.shape, z2.shape)
    eq(z.chunks, z2.chunks)
    eq(z.dtype, z2.dtype)
    eq(z.cname, z2.cname)
    eq(z.clevel, z2.clevel)
    eq(z.shuffle, z2.shuffle)
    eq(z.fill_value, z2.fill_value)
