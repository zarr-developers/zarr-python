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
from pandas.util import testing as tm
from pandas.util.testing import assert_frame_equal, assert_series_equal
from nose.tools import eq_ as eq, assert_is_instance, \
    assert_raises, assert_true, assert_false, assert_is, assert_is_none

from zarr.storage import (DirectoryStore, ZipStore,
                          init_array, init_frame, init_group)
from zarr.hierarchy import group
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
        a = Frame(store)
        assert_is_instance(a, Frame)

        assert repr(a)
        assert_true(pd.Index(["float", "int"]).equals(a.columns))
        eq((100,2), a.shape)
        eq((100,2), a.chunks)
        eq(100, a.nrows)
        eq('', a.path)
        assert_is_none(a.name)
        assert_is(store, a.store)

    def create_frame(self, read_only=False, **kwargs):
        store = dict()
        kwargs.setdefault('compressor', Zlib(level=1))
        init_frame(store, **kwargs)
        return Frame(store, read_only=read_only)

    def test_frame(self):

        df = pd.DataFrame({'A': [1, 2, 3], 'B': [1., 2., 3.], 'C': pd.date_range('20130101', periods=3),
                           'D': ['foo', 'bar', 'baz']},
                          columns=list('ABCD'))

        a = self.create_frame(nrows=len(df), columns=df.columns, dtypes=df.dtypes.values)

        # check properties
        eq(len(a), len(df))
        eq(a.ndim, df.ndim)
        eq(a.shape, df.shape)

        # check empty
        b = a[:]
        assert_is_instance(b, pd.DataFrame)
        eq(a.shape, b.shape)
        assert_series_equal(b.dtypes, df.dtypes)

        # check attributes
        a.attrs['foo'] = 'bar'
        eq('bar', a.attrs['foo'])

        # set data
        a[:] = df

        # get data
        result = a[:]
        assert_frame_equal(result, df)

    def test_mixed_frame(self):

        np.random.seed(1234)
        N = 10000
        ngroups = 10
        strings = tm.rands_array(10, 1000)

        df = pd.DataFrame({'A': np.arange(N),
                           'B': np.random.randint(0, ngroups, size=N),
                           'C': np.random.randn(N),
                           'D': pd.date_range('20130101', periods=ngroups).take(np.random.randint(0, ngroups, size=N)),
                           #'E': pd.Series(strings.take(np.random.randint(0, ngroups, size=N))).astype('category')})
                           #'f': strings.take(np.random.randint(0, ngroups, size=N)),
                           })

        g = group('foo')
        g['df'] = df
        result = g['df'][:]
        assert_frame_equal(result, df)
