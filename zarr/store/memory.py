# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np


from zarr.store.base import ArrayStore
from zarr.compat import itervalues
from zarr.util import normalize_cparams, normalize_shape, normalize_chunks, \
    normalize_resize_args
from zarr.mappings import frozendict


class MemoryStore(ArrayStore):

    def __init__(self, shape, chunks, dtype=None, cname=None, clevel=None,
                 shuffle=None, fill_value=None):
        meta = dict(
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            cname=cname,
            clevel=clevel,
            shuffle=shuffle,
            fill_value=fill_value
        )
        data = dict()
        attrs = dict()
        super(MemoryStore, self).__init__(meta, data, attrs)
