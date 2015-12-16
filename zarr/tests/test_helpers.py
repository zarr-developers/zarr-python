# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


from nose.tools import eq_ as eq, assert_raises
import zarr
from zarr.ext import get_cparams


def test_get_cparams():

    # defaults
    cname, clevel, shuffle = get_cparams()
    eq(zarr.defaults.cname, cname)
    eq(zarr.defaults.clevel, clevel)
    eq(zarr.defaults.shuffle, shuffle)

    # valid
    cname, clevel, shuffle = get_cparams('zlib', 1, 2)
    eq(b'zlib', cname)
    eq(1, clevel)
    eq(2, shuffle)

    # bad cname
    with assert_raises(ValueError):
        get_cparams('foo', 1, True)

    # bad clevel
    with assert_raises(ValueError):
        get_cparams('zlib', 11, True)

    # bad shuffle
    with assert_raises(ValueError):
        get_cparams('zlib', 1, 3)
