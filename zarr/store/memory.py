# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np


from zarr.store.base import ArrayStore
from zarr.compat import itervalues
from zarr.util import normalize_cparams, normalize_shape, normalize_chunks, \
    frozendict


class MemoryStore(ArrayStore):

    def __init__(self, shape, chunks, dtype=None, cname=None, clevel=None,
                 shuffle=None, fill_value=None):

        # normalize arguments
        shape = normalize_shape(shape)
        chunks = normalize_chunks(chunks, shape)
        dtype = np.dtype(dtype)
        cname, clevel, shuffle = normalize_cparams(cname, clevel, shuffle)
        if fill_value is not None:
            fill_value = np.array(fill_value, dtype=dtype)[()]

        # setup internal data structures
        self._meta = frozendict(
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            cname=cname,
            clevel=clevel,
            shuffle=shuffle,
            fill_value=fill_value
        )
        self._data = dict()
        self._attrs = dict()

    @property
    def meta(self):
        return self._meta

    @property
    def data(self):
        return self._data

    @property
    def attrs(self):
        return self._attrs

    @property
    def cbytes(self):
        return sum(len(v) for v in itervalues(self.data))

    @property
    def initialized(self):
        return len(self._data)
