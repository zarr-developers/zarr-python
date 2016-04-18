# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


from functools import reduce  # TODO PY2 compatibility
import operator
import itertools
import numpy as np


from zarr.blosc import compress, decompress


def _is_total_slice(item, shape):
    """Determine whether `item` specifies a complete slice of array with the
    given `shape`. Used to optimise __setitem__ operations on the Chunk
    class."""

    if item == Ellipsis:
        return True
    if item == slice(None):
        return True
    if isinstance(item, tuple):
        return all(
            (isinstance(s, slice) and
            ((s == slice(None)) or (s.stop - s.start == l)))
            for s, l in zip(item, shape)
        )
    return False


def _normalize_axis_selection(item, l):
    """Convenience function to normalize a selection within a single axis
    of size `l`."""

    if isinstance(item, int):
        if item < 0:
            # handle wraparound
            item = l + item
        if item > (l - 1) or item < 0:
            raise IndexError('index out of bounds: %s' % item)
        return item, item + 1

    elif isinstance(item, slice):
        if item.step is not None and item.step != 1:
            raise NotImplementedError('slice with step not supported')
        start = 0 if item.start is None else item.start
        stop = l if item.stop is None else item.stop
        if start < 0:
            start = l + start
        if stop < 0:
            stop = l + stop
        if start < 0 or stop < 0:
            raise IndexError('index out of bounds: %s, %s' % (start, stop))
        if stop > l:
            stop = l
        return start, stop

    else:
        raise ValueError('expected integer or slice, found: %r' % item)


def _normalize_array_selection(item, shape):
    """Convenience function to normalize a selection within an array with
    the given `shape`."""

    # normalize item
    if isinstance(item, int):
        item = (item,)
    elif isinstance(item, slice):
        item = (item,)
    elif item == Ellipsis:
        item = (slice(None),)

    # handle tuple of indices/slices
    if isinstance(item, tuple):

        # determine start and stop indices for all axes
        selection = tuple(_normalize_axis_selection(i, l)
                          for i, l in zip(item, shape))

        # fill out selection if not completely specified
        if len(selection) < len(shape):
            selection += tuple((0, l) for l in shape[len(selection):])

        return selection

    else:
        raise ValueError('expected indices or slice, found: %r' % item)


def _get_chunk_range(selection, chunks):
    """Convenience function to get a range over all chunk indices,
    for iterating over chunks."""
    chunk_range = [range(start//l, int(np.ceil(stop/l)))
                   for (start, stop), l in zip(selection, chunks)]
    return chunk_range


class Array(object):

    def __init__(self, store):
        self._store = store

        # store configuration metadata
        self._shape = store.meta['shape']
        self._chunks = store.meta['chunks']
        self._dtype = store.meta['dtype']
        self._cname = store.meta['cname']
        # TODO check valid cname here?
        self._clevel = store.meta['clevel']
        # TODO check valid clevel here?
        self._shuffle = store.meta['shuffle']
        self._fill_value = store.meta['fill_value']

        # store user-defined attributes
        self._attrs = store.attrs

    @property
    def shape(self):
        return self._shape

    @property
    def chunks(self):
        return self._chunks

    @property
    def dtype(self):
        return self._dtype

    @property
    def cname(self):
        return self._cname

    @property
    def clevel(self):
        return self._clevel

    @property
    def shuffle(self):
        return self._shuffle

    @property
    def fill_value(self):
        return self._fill_value

    @property
    def attrs(self):
        return self._attrs

    @property
    def cbytes(self):
        # pass through
        return self._store.cbytes

    # derived properties

    @property
    def size(self):
        return reduce(operator.mul, self._shape)

    @property
    def itemsize(self):
        return self._dtype.itemsize

    @property
    def nbytes(self):
        return self.size * self.itemsize

    # methods

    def __getitem__(self, item):

        # normalize selection
        selection = _normalize_array_selection(item, self._shape)

        # determine output array shape
        out_shape = tuple(stop - start for start, stop in selection)

        # setup output array
        out = np.empty(out_shape, dtype=self._dtype)

        # determine indices of chunks overlapping the selection
        chunk_range = _get_chunk_range(selection, self._chunks)

        # iterate over chunks in range
        for cidx in itertools.product(*chunk_range):

            # determine chunk offset
            offset = [i * c for i, c in zip(cidx, self._chunks)]

            # determine region within output array
            out_selection = tuple(
                slice(max(0, o - start), min(o + c - start, stop - start))
                for (start, stop), o, c, in zip(selection, offset, self._chunks)
            )

            # determine region within chunk
            chunk_selection = tuple(
                slice(max(0, start - o), min(c, stop - o))
                for (start, stop), o, c in zip(selection, offset, self._chunks)
            )

            # obtain the destination array as a view of the output array
            dest = out[out_selection]

            # load chunk selection into output array
            self._chunk_getitem(cidx, chunk_selection, dest)

        return out

    def __array__(self):
        return self[:]

    def __setitem__(self, key, value):

        # normalize selection
        selection = _normalize_array_selection(key, self._shape)

        # determine indices of chunks overlapping the selection
        chunk_range = _get_chunk_range(selection, self._chunks)

        # iterate over chunks in range
        for cidx in itertools.product(*chunk_range):

            # determine chunk offset
            offset = [i * c for i, c in zip(cidx, self._chunks)]

            # determine required index range within chunk
            chunk_selection = tuple(
                slice(max(0, start - o), min(c, stop - o))
                for (start, stop), o, c in zip(selection, offset, self._chunks)
            )

            if np.isscalar(value):

                # put data
                self._chunk_setitem(cidx, chunk_selection, value)

            else:
                # assume value is array-like

                # determine index within value
                value_selection = tuple(
                    slice(max(0, o - start), min(o + c - start, stop - start))
                    for (start, stop), o, c, in zip(selection, offset, self._chunks)
                )

                # put data
                self._chunk_setitem(cidx, chunk_selection, value[value_selection])

    def _chunk_getitem(self, cidx, item, dest):

        # override this in sub-classes, e.g., if need to use a lock

        # obtain compressed data for chunk
        cdata = self._store.data[cidx]

        if _is_total_slice(item, self._chunks) and dest.flags.c_contiguous:

            # optimisation: we want the whole chunk, and the destination is
            # C contiguous, so we can decompress directly from the chunk
            # into the destination array
            decompress(cdata, dest, self._cname, self._clevel, self._shuffle)

        else:

            # decompress chunk
            chunk = np.empty(self._chunks, dtype=self._dtype)
            decompress(cdata, chunk, self._cname, self._clevel, self._shuffle)

            # set data in output array
            # (split into two lines for profiling)
            tmp = chunk[item]
            dest[:] = tmp

    def _chunk_setitem(self, cidx, key, value):

        # override this in sub-classes, e.g., if need to use a lock

        if _is_total_slice(key, self._chunks):

            # optimisation: we are completely replacing the chunk, so no need
            # to access the existing chunk data

            if np.isscalar(value):

                # setup array filled with value
                chunk = np.empty(self._chunks, dtype=self._dtype)
                chunk.fill(value)

            else:

                # ensure array is C contiguous
                chunk = np.ascontiguousarray(value, dtype=self._dtype)

        else:
            # partially replace the contents of this chunk

            # obtain compressed data for chunk
            cdata = self._store.data[cidx]

            # decompress
            chunk = np.empty(self._chunks, dtype=self._dtype)
            decompress(cdata, chunk, self._cname, self._clevel, self._shuffle)

            # modify
            chunk[key] = value

        # compress
        cdata = compress(chunk, self._cname, self._clevel, self._shuffle)

        # store
        self._store.data[cidx] = cdata

    def __repr__(self):
        # TODO
        pass

    def __str__(self):
        # TODO
        pass

    def resize(self, *args):
        # TODO
        pass

    def append(self, data, axis=0):
        # TODO
        pass

    # TODO


class SynchronizedArray(Array):

    def __init__(self, store, synchronizer):
        super(SynchronizedArray, self).__init__(store)
        self._synchronizer = synchronizer

    # TODO
