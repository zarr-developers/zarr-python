# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import unittest
from tempfile import mkdtemp
import atexit
import shutil
import pickle
from collections import MutableMapping


import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
from nose.tools import eq_ as eq, assert_is_instance, \
    assert_raises, assert_true, assert_false, assert_is, assert_is_none

from zarr.storage import (DirectoryStore, ZipStore,
                          init_array, init_frame, init_group)
from zarr.core import Array
from zarr.frame import Frame
from zarr.errors import PermissionError
from zarr.compat import PY2
from zarr.util import buffer_size
from zarr.codecs import Delta, FixedScaleOffset, Zlib,\
    Blosc, BZ2


class TestFrame(unittest.TestCase):

    def test_frame_init(self):

        # normal initialization
        store = dict()
        init_frame(store, nrows=100, columns=['float', 'int'], dtypes=[np.float64, np.int64])
        fr = Frame(store)
        assert_is_instance(fr, Frame)

        assert repr(fr)
        eq(["float", "int"], fr.columns)
        eq((100,2), fr.shape)
        eq((100,2), fr.chunks)
        eq(100, fr.nrows)
        eq('', fr.path)
        assert_is_none(fr.name)
        assert_is(store, fr.store)

    def create_frame(self, read_only=False, **kwargs):
        store = dict()
        kwargs.setdefault('compressor', Zlib(level=1))
        init_frame(store, **kwargs)
        return Frame(store, read_only=read_only)

    def test_frame(self):

        df = pd.DataFrame({'A': [1, 2, 3], 'B': [1., 2., 3.], 'C': pd.date_range('20130101', periods=3),
                           'D': ['foo', 'bar', 'baz']},
                          columnslist('ABCD'))

        import pdb; pdb.set_trace()
        fr = self.create_frame(nrows=len(df), columns=df.columns, dtypes=df.dtypes.values)

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
        # ...across chunk boundaries...
        assert_array_equal(a[:110], z[:110])
        assert_array_equal(a[190:310], z[190:310])
        assert_array_equal(a[-110:], z[-110:])
        # single item
        eq(a[0], z[0])
        eq(a[-1], z[-1])

        # check partial assignment
        b = np.arange(1e5, 2e5)
        z[190:310] = b[190:310]
        assert_array_equal(a[:190], z[:190])
        assert_array_equal(b[190:310], z[190:310])
        assert_array_equal(a[310:], z[310:])
