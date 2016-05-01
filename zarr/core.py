# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from functools import reduce  # TODO PY2 compatibility
import operator
import itertools
import json


import numpy as np


from zarr.compression import get_compressor_cls
from zarr.util import is_total_slice, normalize_array_selection, \
    get_chunk_range, human_readable_size, normalize_shape, normalize_chunks, \
    normalize_resize_args, normalize_order
from zarr.meta import decode_metadata, encode_metadata
from zarr.attrs import Attributes, SynchronizedAttributes
from zarr.compat import itervalues
from zarr.errors import ReadOnlyError


def init_store(store, shape, chunks, dtype=None, compression='blosc',
               compression_opts=None, fill_value=None,
               order='C', overwrite=False):
    """Initialise an array store with the given configuration."""

    # guard conditions
    empty = len(store) == 0
    if not empty and not overwrite:
        raise ValueError('store is not empty')

    # normalise metadata
    shape = normalize_shape(shape)
    chunks = normalize_chunks(chunks, shape)
    dtype = np.dtype(dtype)
    compressor_cls = get_compressor_cls(compression)
    compression_opts = compressor_cls.normalize_opts(
        compression_opts
    )
    order = normalize_order(order)

    # delete any pre-existing items in store
    store.clear()

    # initialise metadata
    meta = dict(shape=shape, chunks=chunks, dtype=dtype,
                compression=compression, compression_opts=compression_opts,
                fill_value=fill_value, order=order)
    store['meta'] = encode_metadata(meta)

    # initialise attributes
    store['attrs'] = json.dumps(dict()).encode('ascii')


class Array(object):

    def __init__(self, store, readonly=False):
        """Instantiate an array from an existing store.

        Parameters
        ----------
        store : MutableMapping
            Array store, already initialised.
        readonly : bool, optional
            True if array should be protected against modification.

        Examples
        --------
        >>> import zarr
        >>> store = dict()
        >>> zarr.init_store(store, shape=1000, chunks=100)
        >>> z = zarr.Array(store)
        >>> z
        zarr.core.Array((1000,), float64, chunks=(100,), order=C)
          compression: blosc; compression_opts: {'clevel': 5, 'cname': 'blosclz', 'shuffle': 1}
          nbytes: 7.8K; nbytes_stored: 289; ratio: 27.7; initialized: 0/10
          store: builtins.dict

        """  # flake8: noqa

        # N.B., expect at this point store is fully initialised with all
        # configuration metadata fully specified and normalised

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
            self.compression_opts = meta['compression_opts']
            self.fill_value = meta['fill_value']
            self.order = meta['order']
            compressor_cls = get_compressor_cls(self.compression)
            self.compressor = compressor_cls(self.compression_opts)

        # initialise attributes
        self.attrs = Attributes(store, readonly=readonly)

    def flush_metadata(self):
        meta = dict(shape=self.shape, chunks=self.chunks, dtype=self.dtype,
                    compression=self.compression,
                    compression_opts=self.compression_opts,
                    fill_value=self.fill_value, order=self.order)
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
    def nbytes_stored(self):
        """The total number of stored bytes of data for the array. N.B.,
        this will include configuration metadata and user attributes encoded
        as JSON."""
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
        out = np.empty(out_shape, dtype=self.dtype, order=self.order)

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
            raise ReadOnlyError('array is read-only')

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
            cdata = self.store[ckey]

        except KeyError:

            # chunk not initialized
            if self.fill_value is not None:
                dest.fill(self.fill_value)

        else:

            if is_total_slice(item, self.chunks) and \
                    ((self.order == 'C' and dest.flags.c_contiguous) or
                     (self.order == 'F' and dest.flags.f_contiguous)):

                # optimisation: we want the whole chunk, and the destination is
                # contiguous, so we can decompress directly from the chunk
                # into the destination array
                self.compressor.decompress(cdata, dest)

            else:

                # decompress chunk
                chunk = np.empty(self.chunks, dtype=self.dtype,
                                 order=self.order)
                self.compressor.decompress(cdata, chunk)

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
                chunk = np.empty(self.chunks, dtype=self.dtype,
                                 order=self.order)
                chunk.fill(value)

            else:

                # ensure array is contiguous
                if self.order == 'F':
                    chunk = np.asfortranarray(value, dtype=self.dtype)
                else:
                    chunk = np.ascontiguousarray(value, dtype=self.dtype)

        else:
            # partially replace the contents of this chunk

            try:

                # obtain compressed data for chunk
                ckey = '.'.join(map(str, cidx))
                cdata = self.store[ckey]

            except KeyError:

                # chunk not initialized
                chunk = np.empty(self.chunks, dtype=self.dtype,
                                 order=self.order)
                if self.fill_value is not None:
                    chunk.fill(self.fill_value)

            else:

                # decompress chunk
                chunk = np.empty(self.chunks, dtype=self.dtype,
                                 order=self.order)
                self.compressor.decompress(cdata, chunk)

            # modify
            chunk[key] = value

        # compress
        cdata = self.compressor.compress(chunk)

        # store
        ckey = '.'.join(map(str, cidx))
        self.store[ckey] = cdata

    def __repr__(self):
        r = '%s.%s(' % (type(self).__module__, type(self).__name__)
        r += '%s' % str(self.shape)
        r += ', %s' % str(self.dtype)
        r += ', chunks=%s' % str(self.chunks)
        r += ', order=%s' % self.order
        r += ')'
        r += '\n  compression: %s' % self.compression
        r += '; compression_opts: %s' % str(self.compression_opts)
        r += '\n  nbytes: %s' % human_readable_size(self.nbytes)
        if self.nbytes_stored > 0:
            r += '; nbytes_stored: %s' % human_readable_size(
                self.nbytes_stored)
            r += '; ratio: %.1f' % (self.nbytes / self.nbytes_stored)
        n_chunks = reduce(operator.mul, self.cdata_shape)
        r += '; initialized: %s/%s' % (self.initialized, n_chunks)
        r += '\n  store: %s.%s' % (type(self.store).__module__,
                                   type(self.store).__name__)
        return r

    def resize(self, *args):
        """Resize the array."""

        # guard conditions
        if self.readonly:
            raise ReadOnlyError('array is read-only')

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
            raise ReadOnlyError('array is read-only')

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
        """Instantiate a synchronized array.

        Parameters
        ----------
        store : MutableMapping
            Array store, already initialised.
        synchronizer : object
            Array synchronizer.
        readonly : bool, optional
            True if array should be protected against modification.

        Examples
        --------
        >>> import zarr
        >>> store = dict()
        >>> zarr.init_store(store, shape=1000, chunks=100)
        >>> synchronizer = zarr.ThreadSynchronizer()
        >>> z = zarr.SynchronizedArray(store, synchronizer)
        >>> z
        zarr.core.SynchronizedArray((1000,), float64, chunks=(100,), order=C)
          compression: blosc; compression_opts: {'clevel': 5, 'cname': 'blosclz', 'shuffle': 1}
          nbytes: 7.8K; nbytes_stored: 289; ratio: 27.7; initialized: 0/10
          store: builtins.dict
          synchronizer: zarr.sync.ThreadSynchronizer

        Notes
        -----
        Only writing data to the array via the __setitem__() method and
        modification of user attributes are synchronized. Neither append() nor
        resize() are synchronized.

        Writing to the array is synchronized at the chunk level. I.e.,
        the array supports concurrent write operations via the __setitem__()
        method, but these will only exclude each other if they both require
        modification of the same chunk.

        """  # flake8: noqa
        super(SynchronizedArray, self).__init__(store, readonly=readonly)
        self.synchronizer = synchronizer
        self.attrs = SynchronizedAttributes(store, synchronizer,
                                            readonly=readonly)

    def _chunk_setitem(self, cidx, key, value):
        ckey = '.'.join(map(str, cidx))
        with self.synchronizer.chunk_lock(ckey):
            super(SynchronizedArray, self)._chunk_setitem(cidx, key, value)

    def __repr__(self):
        r = super(SynchronizedArray, self).__repr__()
        r += ('\n  synchronizer: %s.%s' %
              (type(self.synchronizer).__module__,
               type(self.synchronizer).__name__))
        return r
