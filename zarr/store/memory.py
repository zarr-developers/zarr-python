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

        # normalize arguments
        shape = normalize_shape(shape)
        chunks = normalize_chunks(chunks, shape)
        dtype = np.dtype(dtype)
        cname, clevel, shuffle = normalize_cparams(cname, clevel, shuffle)

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

    def resize(self, *args):

        # normalize new shape argument
        old_shape = self.meta['shape']
        new_shape = normalize_resize_args(old_shape, *args)

        # determine the new number and arrangement of chunks
        chunks = self.meta['chunks']
        new_cdata_shape = tuple(int(np.ceil(s / c))
                                for s, c in zip(new_shape, chunks))

        # setup new chunks container
        new_data = dict()

        # delete any chunks not within range
        for ckey in self.data:
            cidx = map(int, ckey.split('.'))
            if all(i < c for i, c in zip(cidx, new_cdata_shape)):
                # keep the chunk
                new_data[ckey] = self._data[ckey]

        # update data and metadata
        self._meta = frozendict(self._meta, shape=new_shape)
        self._data = new_data
