# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division

import pickle

import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import eq_ as eq, assert_is_instance

from zarr.store.dictstore import DictStore
from zarr.array import Array


def test_dictstore():
    data = {}
    attrs = {}
    meta = {}
    store = DictStore(data, attrs, meta,
                      shape=(100, 100), chunks=(10, 10), dtype='f4')
    x = Array(store)
    eq(x.shape, (100, 100))
    eq(x.chunks, (10, 10))
    eq(x.dtype, np.float32)
    assert_is_instance(x.cbytes, int)

    x[:] = 1
    assert_array_equal(x[:5, :5], np.ones((5, 5), dtype='f4'))

    x.attrs['hello'] = 'world'

    y = pickle.loads(pickle.dumps(x))
    assert x.store.meta == y.store.meta
    assert x.attrs == y.attrs
    assert_array_equal(x[:], y[:])

    x.resize((50, 200))
    assert x.store.data is data
    assert set(data) == {'%d.%d' % (i, j) for i in range(5) for j in range(10)}
