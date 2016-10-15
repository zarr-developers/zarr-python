# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import operator
import itertools

import numpy as np
from zarr.core import Base, Array
from zarr.util import is_total_slice, normalize_array_selection, \
    get_chunk_range, human_readable_size, normalize_resize_args, \
    normalize_storage_path, normalize_shape, normalize_chunks
from zarr.storage import (frame_meta_key, attrs_key, listdir, getsize,
                          init_array, init_group)
from zarr.meta import decode_frame_metadata, encode_frame_metadata
from zarr.attrs import Attributes
from zarr.errors import PermissionError, err_read_only, err_frame_not_found
from zarr.codecs import get_codec


class Frame(Base):
    """Instantiate a frame from an initialized store.

    Parameters
    ----------
    store : MutableMapping
        Array store, already initialized.
    path : string, optional
        Storage path.
    read_only : bool, optional
        True if array should be protected against modification.
    chunk_store : MutableMapping, optional
        Separate storage for chunks. If not provided, `store` will be used
        for storage of both chunks and metadata.
    synchronizer : object, optional
        Array synchronizer.
    cache_metadata : bool, optional
        If True, array configuration metadata will be cached for the
        lifetime of the object. If False, array metadata will be reloaded
        prior to all data access and modification operations (may incur
        overhead depending on storage and data access pattern).

    Attributes
    ----------
    store
    path
    name
    read_only
    chunk_store
    shape
    chunks
    dtype
    compression
    compression_opts
    order
    synchronizer
    filters
    attrs
    size
    cdata_shape
    nchunks
    nchunks_initialized
    is_view

    Methods
    -------
    __getitem__
    __setitem__
    resize
    append
    view

    """  # flake8: noqa
    _key_meta = frame_meta_key

    def __init__(self, store, path=None, read_only=False, chunk_store=None,
                 synchronizer=None, cache_metadata=True):
        # N.B., expect at this point store is fully initialized with all
        # configuration metadata fully specified and normalized

        self._store = store
        self._path = normalize_storage_path(path)
        if self._path:
            self._key_prefix = self._path + '/'
        else:
            self._key_prefix = ''
        self._read_only = read_only
        if chunk_store is None:
            self._chunk_store = store
        else:
            self._chunk_store = chunk_store
        self._synchronizer = synchronizer
        self._cache_metadata = cache_metadata

        # initialize metadata
        self._load_metadata()

        # initialize attributes
        akey = self._key_prefix + attrs_key
        self._attrs = Attributes(store, key=akey, read_only=read_only,
                                 synchronizer=synchronizer)
        # create our arrays
        self._arrays = {}
        for c, dtype in zip(self._columns, self._dtypes):
            path = self._key_prefix + '/data/' + c
            init_array(store,
                       (self._nrows, 1),
                       chunks=self._chunks,
                       dtype=dtype,
                       compressor=self._compressor,
                       path=path,
                       chunk_store=self._chunk_store,
                       filters=self._filters)
            self._arrays[c] = Array(store, path=path, read_only=True)

    def _load_metadata_nosync(self):
        try:
            mkey = self._key_prefix + frame_meta_key
            meta_bytes = self._store[mkey]
        except KeyError:
            err_frame_not_found(self._path)
        else:

            # decode and store metadata
            meta = decode_frame_metadata(meta_bytes)
            self._meta = meta
            self._nrows = meta['nrows']
            self._columns = meta['columns']
            self._dtypes = meta['dtypes']
            self._chunks = meta['chunks']

            # setup compressor
            config = meta['compressor']
            if config is None:
                self._compressor = None
            else:
                self._compressor = get_codec(config)

            # setup filters
            filters = meta['filters']
            if filters:
                filters = [get_codec(config) for config in filters]
            self._filters = filters

    def _flush_metadata_nosync(self):

        if self._compressor:
            compressor_config = self._compressor.get_config()
        else:
            compressor_config = None
        if self._filters:
            filters_config = [f.get_config() for f in self._filters]
        else:
            filters_config = None
        meta = dict(nrows=self._nrows, chunks=self._chunks, dtypes=self._dtypes,
                    compressor=compressor_config, filters=filters_config)
        mkey = self._key_prefix + frame_meta_key
        self._store[mkey] = encode_frame_metadata(meta)

    @property
    def nrows(self):
        """ our number of rows """
        return self._nrows

    @property
    def columns(self):
        """ return a list of our columns """
        return self._columns

    @property
    def dtypes(self):
        """ a list of our dtypes """
        return self._dtypes

    @property
    def _ncols(self):
        return len(self.dtypes)

    @property
    def ncols(self):
        return self._ncols

    @property
    def itemsize(self):
        """The size in bytes of each item in the array."""
        return sum(dtype.itemsize for dtype in self.dtypes)

    @property
    def _nbytes(self):
        return sum(arr.nbytes for arr in self._arrays)

    def __eq__(self, other):
        return (
            isinstance(other, Frame) and
            self.store == other.store and
            self.read_only == other.read_only and
            self.path == other.path
        )

    def __getitem__(self, item):
        """Retrieve a column or columns. Always returns a DataFrame of the requires column or columns.

        Returns
        -------
        out : DataFrame

        Examples
        --------

        """  # flake8: noqa

        # refresh metadata
        if not self._cache_metadata:
            self._load_metadata()


        import pdb; pdb.set_trace()


        # normalize selection
        selection = normalize_array_selection(item, self._shape)

        # determine output array shape
        out_shape = tuple(s.stop - s.start for s in selection
                          if isinstance(s, slice))

        # setup output array
        out = np.empty(out_shape, dtype=self._dtype, order=self._order)

        # determine indices of chunks overlapping the selection
        chunk_range = get_chunk_range(selection, self._chunks)

        # iterate over chunks in range
        for cidx in itertools.product(*chunk_range):

            # determine chunk offset
            offset = [i * c for i, c in zip(cidx, self._chunks)]

            # determine region within output array
            out_selection = tuple(
                slice(max(0, o - s.start),
                      min(o + c - s.start, s.stop - s.start))
                for s, o, c, in zip(selection, offset, self._chunks)
                if isinstance(s, slice)
            )

            # determine region within chunk
            chunk_selection = tuple(
                slice(max(0, s.start - o), min(c, s.stop - o))
                if isinstance(s, slice)
                else s - o
                for s, o, c in zip(selection, offset, self._chunks)
            )

            # obtain the destination array as a view of the output array
            if out_selection:
                dest = out[out_selection]
            else:
                dest = out

            # load chunk selection into output array
            self._chunk_getitem(cidx, chunk_selection, dest)

        if out.shape:
            return out
        else:
            return out[()]

    def __setitem__(self, item, value):
        raise NotImplementedError("__setitem__ is not implemented")

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

        try:

            # obtain compressed data for chunk
            ckey = self._chunk_key(cidx)
            cdata = self._chunk_store[ckey]

        except KeyError:

            # chunk not initialized
            if self._fill_value is not None:
                dest.fill(self._fill_value)

        else:

            if is_total_slice(item, self._chunks) and \
                    not self._filters and \
                    ((self._order == 'C' and dest.flags.c_contiguous) or
                     (self._order == 'F' and dest.flags.f_contiguous)):

                # optimization: we want the whole chunk, and the destination is
                # contiguous, so we can decompress directly from the chunk
                # into the destination array
                if self._compressor:
                    self._compressor.decode(cdata, dest)
                else:
                    arr = np.frombuffer(cdata, dtype=self._dtype)
                    arr = arr.reshape(self._chunks, order=self._order)
                    np.copyto(dest, arr)

            else:

                # decode chunk
                chunk = self._decode_chunk(cdata)

                # set data in output array
                # (split into two lines for profiling)
                tmp = chunk[item]
                if dest.shape:
                    dest[:] = tmp
                else:
                    dest[()] = tmp

    def _chunk_setitem(self, cidx, item, value):
        """Replace part or whole of a chunk.

        Parameters
        ----------
        cidx : tuple of ints
            Indices of the chunk.
        item : tuple of slices
            Location of region within the chunk.
        value : scalar or ndarray
            Value to set.

        """

        # synchronization
        if self._synchronizer is None:
            self._chunk_setitem_nosync(cidx, item, value)
        else:
            # synchronize on the chunk
            ckey = self._chunk_key(cidx)
            with self._synchronizer[ckey]:
                self._chunk_setitem_nosync(cidx, item, value)

    def _chunk_setitem_nosync(self, cidx, item, value):

        # obtain key for chunk storage
        ckey = self._chunk_key(cidx)

        if is_total_slice(item, self._chunks):
            # totally replace chunk

            # optimization: we are completely replacing the chunk, so no need
            # to access the existing chunk data

            if np.isscalar(value):

                # setup array filled with value
                chunk = np.empty(self._chunks, dtype=self._dtype,
                                 order=self._order)
                chunk.fill(value)

            else:

                if not self._compressor and not self._filters:

                    # https://github.com/alimanfoo/zarr/issues/79
                    # Ensure a copy is taken so we don't end up storing
                    # a view into someone else's array.
                    # N.B., this assumes that filters or compressor always
                    # take a copy and never attempt to apply encoding in-place.
                    chunk = np.array(value, dtype=self._dtype,
                                     order=self._order)

                else:
                    # ensure array is contiguous
                    if self._order == 'F':
                        chunk = np.asfortranarray(value, dtype=self._dtype)
                    else:
                        chunk = np.ascontiguousarray(value, dtype=self._dtype)

        else:
            # partially replace the contents of this chunk

            try:

                # obtain compressed data for chunk
                cdata = self._chunk_store[ckey]

            except KeyError:

                # chunk not initialized
                chunk = np.empty(self._chunks, dtype=self._dtype,
                                 order=self._order)
                if self._fill_value is not None:
                    chunk.fill(self._fill_value)

            else:

                # decode chunk
                chunk = self._decode_chunk(cdata)
                if not chunk.flags.writeable:
                    chunk = chunk.copy(order='K')

            # modify
            chunk[item] = value

        # encode chunk
        cdata = self._encode_chunk(chunk)

        # store
        self._chunk_store[ckey] = cdata

    def _chunk_key(self, cidx):
        return self._key_prefix + '.'.join(map(str, cidx))

    def _decode_chunk(self, cdata):

        # decompress
        if self._compressor:
            chunk = self._compressor.decode(cdata)
        else:
            chunk = cdata

        # apply filters
        if self._filters:
            for f in self._filters[::-1]:
                chunk = f.decode(chunk)

        # view as correct dtype
        if isinstance(chunk, np.ndarray):
            chunk = chunk.view(self._dtype)
        else:
            chunk = np.frombuffer(chunk, self._dtype)

        # reshape
        chunk = chunk.reshape(self._chunks, order=self._order)

        return chunk

    def _encode_chunk(self, chunk):

        # apply filters
        if self._filters:
            for f in self._filters:
                chunk = f.encode(chunk)

        # compress
        if self._compressor:
            cdata = self._compressor.encode(chunk)
        else:
            cdata = chunk

        return cdata

    def __repr__(self):
        # N.B., __repr__ needs to be synchronized to ensure consistent view
        # of metadata AND when retrieving nbytes_stored from filesystem storage
        return self._synchronized_op(self._repr_nosync)

    def _repr_nosync(self):

        # main line
        r = '%s(' % type(self).__name__
        if self.name:
            r += '%s, ' % self.name
        r += '%s, ' % str(self._shape)
        r += '%s, ' % str(self._dtypes)
        r += 'chunks=%s, ' % str(self._chunks)
        r += ')'

        # storage size info
        r += '\n  nbytes: %s' % human_readable_size(self._nbytes)
        if self.nbytes_stored > 0:
            r += '; nbytes_stored: %s' % human_readable_size(
                self.nbytes_stored)
            r += '; ratio: %.1f' % (self._nbytes / self.nbytes_stored)
        r += '; initialized: %s/%s' % (self.nchunks_initialized,
                                       self._nchunks)

        # filters
        if self._filters:
            # first line
            r += '\n  filters: %r' % self._filters[0]
            # subsequent lines
            for f in self._filters[1:]:
                r += '\n           %r' % f

        # compressor
        if self._compressor:
            r += '\n  compressor: %r' % self._compressor

        # storage and synchronizer classes
        r += '\n  store: %s' % type(self._store).__name__
        if self._store != self._chunk_store:
            r += '; chunk_store: %s' % type(self._chunk_store).__name__
        if self._synchronizer is not None:
            r += '; synchronizer: %s' % type(self._synchronizer).__name__

        return r

    def __getstate__(self):
        return self._store, self._path, self._read_only, self._chunk_store, \
               self._synchronizer, self._cache_metadata

    def __setstate__(self, state):
        self.__init__(*state)

    def _synchronized_op(self, f, *args, **kwargs):

        # no synchronization
        if self._synchronizer is None:
            self._refresh_metadata_nosync()
            return f(*args, **kwargs)

        else:
            # synchronize on the array
            mkey = self._key_prefix + frame_meta_key
            with self._synchronizer[mkey]:
                self._refresh_metadata_nosync()
                result = f(*args, **kwargs)
            return result

    def _write_op(self, f, *args, **kwargs):

        # guard condition
        if self._read_only:
            err_read_only()

        return self._synchronized_op(f, *args, **kwargs)

    def resize(self, *args):
        raise NotImplementedError("resize is not implemented")

    def append(self, data, axis=0):
        raise NotImplementedError("append is not implemented")

    def view(self, *args, **kwargs):
        raise NotImplementedError("view is not implemented")
