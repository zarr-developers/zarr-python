# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np
from nose.tools import eq_ as eq


from zarr.store.memory import MemoryStore
from zarr.array import Array
from zarr import defaults


def test_1d():

    a = np.arange(1050)
    store = MemoryStore(a.shape, chunks=100, dtype=a.dtype)
    z = Array(store)

    # check properties
    eq(a.shape, z.shape)
    eq(a.dtype, z.dtype)
    eq((100,), z.chunks)
    eq(defaults.cname, z.cname)
    eq(defaults.clevel, z.clevel)
    eq(defaults.shuffle, z.shuffle)
    eq(a.nbytes, z.nbytes)
    eq(0, z.cbytes)
    eq(0, z.initialized)

    # TODO
