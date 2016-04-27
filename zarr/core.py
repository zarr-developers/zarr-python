# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from functools import reduce  # TODO PY2 compatibility
import operator
import itertools
import multiprocessing
import json


import numpy as np


from zarr import blosc
from zarr.sync import SynchronizedAttributes
from zarr.util import is_total_slice, normalize_array_selection, \
    get_chunk_range, human_readable_size, normalize_shape, normalize_chunks, \
    normalize_resize_args
from zarr.meta import decode_metadata, encode_metadata
from zarr.compat import itervalues


_blosc_use_context = False


def set_blosc_options(use_context=False, nthreads=None):
    """Set options for how the blosc compressor is used.

    Parameters
    ----------
    use_context : bool, optional
        If False, blosc will be used in non-contextual mode, which is best
        when using zarr in a single-threaded environment because it allows
        blosc to use multiple threads internally. If True, blosc will be used
        in contextual mode, which is better when using zarr in a
        multi-threaded environment like dask.array because it avoids the blosc
        global lock and so multiple blosc operations can be running
        concurrently.
    nthreads : int, optional
        Number of internal threads to use when running blosc in non-contextual
        mode.

    """
    global _blosc_use_context
    _blosc_use_context = use_context
    if not use_context:
        if nthreads is None:
            # diminishing returns beyond 4 threads?
            nthreads = min(4, multiprocessing.cpu_count())
        blosc.set_nthreads(nthreads)


_repr_shuffle = [
    '0 (NOSHUFFLE)',
    '1 (BYTESHUFFLE)',
    '2 (BITSHUFFLE)',
]


def init_store(store, shape, chunks, dtype=None, compression='blosc',
               compression_opts=None, fill_value=None, overwrite=False):
    """Initialise an array store with the given configuration."""

    # guard conditions
    empty = len(store) == 0
    if not empty and not overwrite:
        raise ValueError('store is not empty')

    # normalise metadata
    shape = normalize_shape(shape)
    chunks = normalize_chunks(chunks, shape)
    dtype = np.dtype(dtype)
    if compression != 'blosc':
        raise NotImplementedError('only blosc compression is '
                                  'currently implemented')
    compression_opts = normalize_blosc_opts(compression_opts)

    # handle any pre-existing data in store
    for key in list(store.keys()):
        del store[key]

    # initialise metadata
    meta = dict(shape=shape, chunks=chunks, dtype=dtype,
                compression=compression, compression_opts=compression_opts,
                fill_value=fill_value)
    store['meta'] = encode_metadata(meta)

    # initialise attributes
    store['attrs'] = json.dumps(dict())


class Array(object):

    def __init__(self, store, readonly=False):
        """Instantiate an array from an existing store.

        Parameters
        ----------
        store : MutableMapping
            Array store.
        readonly : bool, optional
            True if array should be protected against modification.

        """

        self.store = store
        self.readonly = readonly

        # initialise metadata
        try:
            meta_bytes = store['meta']
        except KeyError:
            raise ValueError('store has no metadata')
        else:
            meta = decode_metadata(meta_bytes)
            self.meta = meta
            self.shape = meta['shape']
            self.chunks = meta['chunks']
            self.dtype = meta['dtype']
            self.compression = meta['compression']
            if self.compression != 'blosc':
                raise NotImplementedError('only blosc compression is '
                                          'currently implemented')
            self.compression_opts = meta['compression_opts']
            self.fill_value = meta['fill_value']

        # initialise attributes
        self.attrs = Attributes(store, readonly=readonly)

    def flush_metadata(self):
        meta = dict(shape=self.shape, chunks=self.chunks, dtype=self.dtype,
                    compression=self.compression,
                    compression_opts=self.compression_opts,
                    fill_value=self.fill_value)
        self.store['meta'] = encode_metadata(meta)

    @property
    def size(self):
        return reduce(operator.mul, self.shape)

    @property
    def itemsize(self):
        return self.dtype.itemsize

    @property
    def nbytes(self):
        return self.size * self.itemsize

    @property
    def cbytes(self):
        """The total number of stored bytes of data for the array."""

        if hasattr(self.store, 'size'):
            # pass through
            return self.store.size
        elif isinstance(self.store, dict):
            # cheap to compute by summing length of values
            return sum(len(v) for v in itervalues(self.store))
        else:
            return -1

    @property
    def initialized(self):
        """The number of chunks that have been initialized."""
        # N.B., expect 'meta' and 'attrs' keys in store also, so subtract 2
        return len(self.store) - 2

    @property
    def cdata_shape(self):
        return tuple(
            int(np.ceil(s / c)) for s, c in zip(self.shape, self.chunks)
        )

    def __getitem__(self, item):

        # normalize selection
        selection = normalize_array_selection(item, self.shape)

        # determine output array shape
        out_shape = tuple(stop - start for start, stop in selection)

        # setup output array
        out = np.empty(out_shape, dtype=self.dtype)

        # determine indices of chunks overlapping the selection
        chunk_range = get_chunk_range(selection, self.chunks)

        # iterate over chunks in range
        for cidx in itertools.product(*chunk_range):

            # determine chunk offset
            offset = [i * c for i, c in zip(cidx, self.chunks)]

            # determine region within output array
            out_selection = tuple(
                slice(max(0, o - start), min(o + c - start, stop - start))
                for (start, stop), o, c, in zip(selection, offset, self.chunks)
            )

            # determine region within chunk
            chunk_selection = tuple(
                slice(max(0, start - o), min(c, stop - o))
                for (start, stop), o, c in zip(selection, offset, self.chunks)
            )

            # obtain the destination array as a view of the output array
            dest = out[out_selection]

            # load chunk selection into output array
            self._chunk_getitem(cidx, chunk_selection, dest)

        return out

    def __array__(self):
        return self[:]

    def __setitem__(self, key, value):

        # guard conditions
        if self.readonly:
            raise PermissionError('array is read-only')

        # normalize selection
        selection = normalize_array_selection(key, self.shape)

        # determine indices of chunks overlapping the selection
        chunk_range = get_chunk_range(selection, self.chunks)

        # iterate over chunks in range
        for cidx in itertools.product(*chunk_range):

            # determine chunk offset
            offset = [i * c for i, c in zip(cidx, self.chunks)]

            # determine required index range within chunk
            chunk_selection = tuple(
                slice(max(0, start - o), min(c, stop - o))
                for (start, stop), o, c in zip(selection, offset, self.chunks)
            )

            if np.isscalar(value):

                # put data
                self._chunk_setitem(cidx, chunk_selection, value)

            else:
                # assume value is array-like

                # determine index within value
                value_selection = tuple(
                    slice(max(0, o - start), min(o + c - start, stop - start))
                    for (start, stop), o, c, in zip(selection, offset,
                                                    self.chunks)
                )

                # put data
                self._chunk_setitem(cidx, chunk_selection,
                                    value[value_selection])

    def _chunk_getitem(self, cidx, item, dest):
        """Obtain part or whole of a chunk.

        Parameters
        ----------
        cidx : tuple of ints
            Indices of the chunk.
        item : tuple of slices
            Location of region within the chunk.
        dest : ndarray
            Numpy array to store result in.

        """

        # override this in sub-classes, e.g., if need to use a lock

        try:

            # obtain compressed data for chunk
            ckey = '.'.join(map(str, cidx))
            cdata = self.store.data[ckey]

        except KeyError:

            # chunk not initialized
            if self.fill_value is not None:
                dest.fill(self.fill_value)

        else:

            if is_total_slice(item, self.chunks) and dest.flags.c_contiguous:

                # optimisation: we want the whole chunk, and the destination is
                # C contiguous, so we can decompress directly from the chunk
                # into the destination array
                blosc.decompress(cdata, dest, _blosc_use_context)

            else:

                # decompress chunk
                chunk = np.empty(self.chunks, dtype=self.dtype)
                blosc.decompress(cdata, chunk, _blosc_use_context)

                # set data in output array
                # (split into two lines for profiling)
                tmp = chunk[item]
                dest[:] = tmp

    def _chunk_setitem(self, cidx, key, value):
        """Replace part or whole of a chunk.

        Parameters
        ----------
        cidx : tuple of ints
            Indices of the chunk.
        key : tuple of slices
            Location of region within the chunk.
        value : scalar or ndarray
            Value to set.

        """

        # override this in sub-classes, e.g., if need to use a lock

        if is_total_slice(key, self.chunks):

            # optimisation: we are completely replacing the chunk, so no need
            # to access the existing chunk data

            if np.isscalar(value):

                # setup array filled with value
                chunk = np.empty(self.chunks, dtype=self.dtype)
                chunk.fill(value)

            else:

                # ensure array is C contiguous
                chunk = np.ascontiguousarray(value, dtype=self.dtype)

        else:
            # partially replace the contents of this chunk

            try:

                # obtain compressed data for chunk
                ckey = '.'.join(map(str, cidx))
                cdata = self.store.data[ckey]

            except KeyError:

                # chunk not initialized
                chunk = np.empty(self.chunks, dtype=self.dtype)
                if self.fill_value is not None:
                    chunk.fill(self.fill_value)

            else:

                # decompress chunk
                chunk = np.empty(self.chunks, dtype=self.dtype)
                blosc.decompress(cdata, chunk, _blosc_use_context)

            # modify
            chunk[key] = value

        # compress
        # TODO translate compression options
        cdata = blosc.compress(chunk, self.cname, self.clevel,
                               self.shuffle, _blosc_use_context)

        # store
        ckey = '.'.join(map(str, cidx))
        self.store.data[ckey] = cdata

    def __repr__(self):
        # TODO handle compression options
        r = '%s.%s(' % (type(self).__module__, type(self).__name__)
        r += '%s' % str(self.shape)
        r += ', %s' % str(self.dtype)
        r += ', chunks=%s' % str(self.chunks)
        r += ')'
        r += '\n  cname: %s' % str(self.cname, 'ascii')
        r += '; clevel: %s' % self.clevel
        r += '; shuffle: %s' % _repr_shuffle[self.shuffle]
        r += '\n  nbytes: %s' % human_readable_size(self.nbytes)
        r += '; cbytes: %s' % human_readable_size(self.cbytes)
        if self.cbytes > 0:
            r += '; ratio: %.1f' % (self.nbytes / self.cbytes)
        n_chunks = reduce(operator.mul, self.cdata_shape)
        r += '; initialized: %s/%s' % (self.initialized, n_chunks)
        r += '\n  store: %s.%s' % (type(self.store).__module__,
                                   type(self.store).__name__)
        return r

    def __str__(self):
        return repr(self)

    def resize(self, *args):
        """Resize the array."""

        # guard conditions
        if self.readonly:
            raise PermissionError('array is read-only')

        # normalize new shape argument
        old_shape = self.shape
        new_shape = normalize_resize_args(old_shape, *args)

        # determine the new number and arrangement of chunks
        chunks = self.chunks
        new_cdata_shape = tuple(int(np.ceil(s / c))
                                for s, c in zip(new_shape, chunks))

        # remove any chunks not within range
        for key in list(self.store):
            if key not in ['meta', 'attrs']:
                cidx = map(int, key.split('.'))
                if all(i < c for i, c in zip(cidx, new_cdata_shape)):
                    pass  # keep the chunk
                else:
                    del self.store[key]

        # update metadata
        self.shape = new_shape
        self.flush_metadata()

    def append(self, data, axis=0):
        """Append `data` to `axis`.

        Parameters
        ----------
        data : array_like
            Data to be appended.
        axis : int
            Axis along which to append.

        Notes
        -----
        The size of all dimensions other than `axis` must match between this
        array and `data`.

        """

        # guard conditions
        if self.readonly:
            raise PermissionError('array is read-only')

        # ensure data is array-like
        if not hasattr(data, 'shape') or not hasattr(data, 'dtype'):
            data = np.asanyarray(data)

        # ensure shapes are compatible for non-append dimensions
        self_shape_preserved = tuple(s for i, s in enumerate(self.shape)
                                     if i != axis)
        data_shape_preserved = tuple(s for i, s in enumerate(data.shape)
                                     if i != axis)
        if self_shape_preserved != data_shape_preserved:
            raise ValueError('shapes not compatible')

        # remember old shape
        old_shape = self.shape

        # determine new shape
        new_shape = tuple(
            self.shape[i] if i != axis else self.shape[i] + data.shape[i]
            for i in range(len(self.shape))
        )

        # resize
        self.resize(new_shape)

        # store data
        # noinspection PyTypeChecker
        append_selection = tuple(
            slice(None) if i != axis else slice(old_shape[i], new_shape[i])
            for i in range(len(self.shape))
        )
        self[append_selection] = data


class SynchronizedArray(Array):

    def __init__(self, store, synchronizer, readonly=False):
        super(SynchronizedArray, self).__init__(store, readonly=readonly)
        self.synchronizer = synchronizer
        self.attrs = SynchronizedAttributes(store, synchronizer,
                                            readonly=readonly)

    def _chunk_setitem(self, cidx, key, value):
        with self.synchronizer.lock_chunk(cidx):
            super(SynchronizedArray, self)._chunk_setitem(cidx, key, value)

    def resize(self, *args):
        with self.synchronizer.lock_array():
            super(SynchronizedArray, self).resize(*args)

    def append(self, data, axis=0):
        with self.synchronizer.lock_array():
            super(SynchronizedArray, self).append(data, axis=axis)

    def __repr__(self):
        r = super(SynchronizedArray, self).__repr__()
        r += ('\n  synchronizer: %s.%s' %
              (type(self.synchronizer).__module__,
               type(self.synchronizer).__name__))
        return r
