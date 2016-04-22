# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np


from zarr.util import normalize_shape, normalize_chunks, normalize_cparams, \
    normalize_resize_args
from zarr.compat import itervalues


class ArrayStore(object):
    """Storage for a single array.

    Parameters
    ----------
    meta : MutableMapping
        Array configuration metadata. Must contain at least 'shape' and
        'chunks'.
    data : MutableMapping
        Holds a mapping from chunk indices to compressed chunk data.
    attrs : MutableMapping
        Holds user-defined attributes.

    Examples
    --------
    >>> import zarr
    >>> meta = dict(shape=(100,), chunks=(10,))
    >>> data = dict()
    >>> attrs = dict()
    >>> store = zarr.ArrayStore(meta, data, attrs)
    >>> z = zarr.Array(store)
    >>> meta
    >>> z[:] = 42
    >>> sorted(data.keys())

    """

    def __init__(self, meta, data, attrs):

        # normalize configuration metadata
        shape = normalize_shape(meta['shape'])
        chunks = normalize_chunks(meta['chunks'], shape)
        dtype = np.dtype(meta.get('dtype', None))  # float64 default
        cname, clevel, shuffle = \
            normalize_cparams(meta.get('cname', None),
                              meta.get('clevel', None),
                              meta.get('shuffle', None))
        fill_value = meta.get('fill_value', None)
        # assume meta implements 'update'
        meta.update(shape=shape, chunks=chunks, dtype=dtype, cname=cname,
                    clevel=clevel, shuffle=shuffle, fill_value=fill_value)

        # setup members
        self.meta = meta
        self.data = data
        self.attrs = attrs

    @property
    def cbytes(self):
        """The total size in number of bytes of compressed data held for the
        array."""
        if hasattr(self.data, 'cbytes'):
            # pass through
            return self.data.cbytes
        elif isinstance(self.data, dict):
            # cheap to compute by summing length of values
            return sum(len(v) for v in itervalues(self.data))
        else:
            return -1

    @property
    def initialized(self):
        """The number of chunks that have been initialized."""
        return len(self.data)

    def resize(self, *args):
        """Resize the array."""

        # normalize new shape argument
        old_shape = self.meta['shape']
        new_shape = normalize_resize_args(old_shape, *args)

        # determine the new number and arrangement of chunks
        chunks = self.meta['chunks']
        new_cdata_shape = tuple(int(np.ceil(s / c))
                                for s, c in zip(new_shape, chunks))

        # remove any chunks not within range
        for ckey in list(self.data):
            cidx = map(int, ckey.split('.'))
            if all(i < c for i, c in zip(cidx, new_cdata_shape)):
                pass  # keep the chunk
            else:
                del self.data[ckey]

        # update metadata
        self.meta['shape'] = new_shape
