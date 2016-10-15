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
from zarr.codecs import get_codec, PickleCodec

from pandas import DataFrame, concat
from pandas.api.types import is_object_dtype, is_categorical_dtype


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
        filters = self._filters
        self._arrays = {}
        for c, dtype in zip(self._columns, self._dtypes):
            path = self._key_prefix + '/data/' + c

            if is_object_dtype(dtype):
                filters = self._filters
                if filters is None:
                    filters = []
                filters += [PickleCodec()]
            else:
                filters = self._filters
            init_array(store,
                       self._nrows,
                       chunks=self._chunks[0],
                       dtype=dtype,
                       compressor=self._compressor,
                       path=path,
                       chunk_store=self._chunk_store,
                       filters=filters)
            self._arrays[c] = Array(store, path=path, read_only=False)

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

            from pandas import Index
            self._columns = Index(meta['columns'])
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
    def _shape(self):
        return (self._nrows, self._ncols)

    @property
    def columns(self):
        """ return a list of our columns """
        return self._columns

    @property
    def _ncols(self):
        return len(self.dtypes)

    @property
    def ncols(self):
        return self._ncols

    @property
    def dtypes(self):
        """ a list of our dtypes """
        return self._dtypes

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

    def _array_to_series(self, c, indexer):
        """
        Return a pandas Series for this array with name c
        Raise KeyError if not found
        """
        from pandas import Series
        arr = self._arrays[c]
        arr = arr[indexer]
        return Series(arr, name=c)

    def _series_to_array(self, c, indexer, value):
        """
        Set the array with name c for this value (a Series)
        and the indexer
        """
        arr = self._arrays[c]
        arr[indexer] = value.values

    def __getitem__(self, item):
        """
        Retrieve a column or columns. Always returns a DataFrame of the requires column or columns.

        Returns
        -------
        out : DataFrame

        Examples
        --------

        """  # flake8: noqa

        # refresh metadata
        if not self._cache_metadata:
            self._load_metadata()

        columns = self._columns[item]
        return concat([self._array_to_series(c, slice(None))
                       for c in columns],
                      axis=1)

    def __setitem__(self, item, value):
        """
        Set particular data. item item refers to column or columns.
        The shape and dtypes must match to the existing store.

        Examples
        --------
        """

        # refresh metadata
        if not self._cache_metadata:
            self._load_metadata_nosync()

        # normalize selection
        columns = self._columns[item]
        if not isinstance(value, DataFrame):
            raise ValueError("setting must be with a DataFrame")

        for c in columns:
            self._series_to_array(c, slice(None), value[c])

    def _repr_nosync(self):

        # main line
        r = '%s(' % type(self).__name__
        if self.name:
            r += '%s, ' % self.name
        r += '%s, ' % str(self._shape)
        r += 'chunks=%s, ' % str(self._chunks)
        r += ')'

        # storage and synchronizer classes
        r += '\n store: %s' % type(self._store).__name__
        if self._store != self._chunk_store:
            r += '; chunk_store: %s' % type(self._chunk_store).__name__
        if self._synchronizer is not None:
            r += '; synchronizer: %s' % type(self._synchronizer).__name__

        # arrays
        r += '\n'
        for c in self._columns:
            arr = self._arrays[c]
            r += '\n %s' % arr._repr_abbv_nosync()

        return r

    def resize(self, *args):
        raise NotImplementedError("resize is not implemented")

    def append(self, data, axis=0):
        raise NotImplementedError("append is not implemented")

    def view(self, *args, **kwargs):
        raise NotImplementedError("view is not implemented")
