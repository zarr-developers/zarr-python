# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np
from nose.tools import eq_ as eq, assert_is_instance


from zarr.store import ArrayStore
from zarr.core import Array


def test_arraystore():

    data = dict()
    attrs = dict()
    meta = dict(shape=(100, 100), chunks=(10, 10), dtype='f4')
    store = ArrayStore(meta, data, attrs)
    z = Array(store)
    eq(z.shape, (100, 100))
    eq(z.chunks, (10, 10))
    eq(z.dtype, np.float32)
    assert_is_instance(z.cbytes, int)

    assert z.store is store
    assert store.meta is meta
    assert store.data is data
    assert store.attrs is attrs
    eq(set(data), set())
    z[:] = 1
    eq(set(data), {'%d.%d' % (i, j) for i in range(10) for j in range(10)})
