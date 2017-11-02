# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import operator
import itertools
import re


import numpy as np


from zarr.util import is_total_slice, normalize_array_selection, get_chunks_for_selection, \
    human_readable_size, normalize_resize_args, normalize_storage_path, normalize_shape, \
    normalize_chunks, InfoReporter, get_chunk_selections
from zarr.storage import array_meta_key, attrs_key, listdir, getsize
from zarr.meta import decode_array_metadata, encode_array_metadata
from zarr.attrs import Attributes
from zarr.errors import PermissionError, err_read_only, err_array_not_found
from zarr.compat import reduce
from zarr.codecs import AsType, get_codec
from zarr.indexing import OIndex, OrthogonalIndexer, BasicIndexer


class Array(object):
    """Instantiate an array from an initialized store.

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
    fill_value
    order
    synchronizer
    filters
    attrs
    size
    itemsize
    nbytes
    nbytes_stored
    cdata_shape
    nchunks
    nchunks_initialized
    is_view
    info

    Methods
    -------
    __getitem__
    __setitem__
    resize
    append
    view
    astype

    """

    def __init__(self, store, path=None, read_only=False, chunk_store=None,
                 synchronizer=None, cache_metadata=True):
        # N.B., expect at this point store is fully initialized with all
        # configuration metadata fully specified and normalized

        self._store = store
        self._chunk_store = chunk_store
        self._path = normalize_storage_path(path)
        if self._path:
            self._key_prefix = self._path + '/'
        else:
            self._key_prefix = ''
        self._read_only = read_only
        self._synchronizer = synchronizer
        self._cache_metadata = cache_metadata
        self._is_view = False

        # initialize metadata
        self._load_metadata()

        # initialize attributes
        akey = self._key_prefix + attrs_key
        self._attrs = Attributes(store, key=akey, read_only=read_only,
                                 synchronizer=synchronizer)

        # initialize info reporter
        self._info_reporter = InfoReporter(self)

        # initialize indexing helpers
        self._oindex = OIndex(self)

    def _load_metadata(self):
        """(Re)load metadata from store."""
        if self._synchronizer is None:
            self._load_metadata_nosync()
        else:
            mkey = self._key_prefix + array_meta_key
            with self._synchronizer[mkey]:
                self._load_metadata_nosync()

    def _load_metadata_nosync(self):
        try:
            mkey = self._key_prefix + array_meta_key
            meta_bytes = self._store[mkey]
        except KeyError:
            err_array_not_found(self._path)
        else:

            # decode and store metadata
            meta = decode_array_metadata(meta_bytes)
            self._meta = meta
            self._shape = meta['shape']
            self._chunks = meta['chunks']
            self._dtype = meta['dtype']
            self._fill_value = meta['fill_value']
            self._order = meta['order']

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

    def _refresh_metadata(self):
        if not self._cache_metadata:
            self._load_metadata()

    def _refresh_metadata_nosync(self):
        if not self._cache_metadata and not self._is_view:
            self._load_metadata_nosync()

    def _flush_metadata_nosync(self):
        if self._is_view:
            raise PermissionError('not permitted for views')

        if self._compressor:
            compressor_config = self._compressor.get_config()
        else:
            compressor_config = None
        if self._filters:
            filters_config = [f.get_config() for f in self._filters]
        else:
            filters_config = None
        meta = dict(shape=self._shape, chunks=self._chunks, dtype=self._dtype,
                    compressor=compressor_config, fill_value=self._fill_value,
                    order=self._order, filters=filters_config)
        mkey = self._key_prefix + array_meta_key
        self._store[mkey] = encode_array_metadata(meta)

    @property
    def store(self):
        """A MutableMapping providing the underlying storage for the array."""
        return self._store

    @property
    def path(self):
        """Storage path."""
        return self._path

    @property
    def name(self):
        """Array name following h5py convention."""
        if self.path:
            # follow h5py convention: add leading slash
            name = self.path
            if name[0] != '/':
                name = '/' + name
            return name
        return None

    @property
    def read_only(self):
        """A boolean, True if modification operations are not permitted."""
        return self._read_only

    @property
    def chunk_store(self):
        """A MutableMapping providing the underlying storage for array chunks."""
        if self._chunk_store is None:
            return self._store
        else:
            return self._chunk_store

    @property
    def shape(self):
        """A tuple of integers describing the length of each dimension of
        the array."""
        # N.B., shape may change if array is resized, hence need to refresh
        # metadata
        self._refresh_metadata()
        return self._shape

    @shape.setter
    def shape(self, value):
        self.resize(value)

    @property
    def chunks(self):
        """A tuple of integers describing the length of each dimension of a
        chunk of the array."""
        return self._chunks

    @property
    def dtype(self):
        """The NumPy data type."""
        return self._dtype

    @property
    def compressor(self):
        """Primary compression codec."""
        return self._compressor

    @property
    def fill_value(self):
        """A value used for uninitialized portions of the array."""
        return self._fill_value

    @property
    def order(self):
        """A string indicating the order in which bytes are arranged within
        chunks of the array."""
        return self._order

    @property
    def filters(self):
        """One or more codecs used to transform data prior to compression."""
        return self._filters

    @property
    def synchronizer(self):
        """Object used to synchronize write access to the array."""
        return self._synchronizer

    @property
    def attrs(self):
        """A MutableMapping containing user-defined attributes. Note that
        attribute values must be JSON serializable."""
        return self._attrs

    @property
    def ndim(self):
        """Number of dimensions."""
        return len(self.shape)

    @property
    def _size(self):
        return reduce(operator.mul, self._shape, 1)

    @property
    def size(self):
        """The total number of elements in the array."""
        # N.B., this property depends on shape, and shape may change if array
        # is resized, hence need to refresh metadata
        self._refresh_metadata()
        return self._size

    @property
    def itemsize(self):
        """The size in bytes of each item in the array."""
        return self.dtype.itemsize

    @property
    def _nbytes(self):
        return self._size * self.itemsize

    @property
    def nbytes(self):
        """The total number of bytes that would be required to store the
        array without compression."""
        # N.B., this property depends on shape, and shape may change if array
        # is resized, hence need to refresh metadata
        self._refresh_metadata()
        return self._nbytes

    @property
    def nbytes_stored(self):
        """The total number of stored bytes of data for the array. This
        includes storage required for configuration metadata and user
        attributes."""
        m = getsize(self._store, self._path)
        if self._chunk_store is None:
            return m
        else:
            n = getsize(self._chunk_store, self._path)
            if m < 0 or n < 0:
                return -1
            else:
                return m + n

    @property
    def _cdata_shape(self):
        if self._shape == ():
            return (1,)
        else:
            return tuple(int(np.ceil(s / c))
                         for s, c in zip(self._shape, self._chunks))

    @property
    def cdata_shape(self):
        """A tuple of integers describing the number of chunks along each
        dimension of the array."""
        self._refresh_metadata()
        return self._cdata_shape

    @property
    def _nchunks(self):
        return reduce(operator.mul, self._cdata_shape, 1)

    @property
    def nchunks(self):
        """Total number of chunks."""
        self._refresh_metadata()
        return self._nchunks

    @property
    def nchunks_initialized(self):
        """The number of chunks that have been initialized with some data."""
        # TODO fix bug here, need to only count chunks

        # key pattern for chunk keys
        prog = re.compile(r'\.'.join([r'\d+'] * min(1, self.ndim)))

        # count chunk keys
        return sum(1 for k in listdir(self.chunk_store, self._path) if prog.match(k))

    # backwards compability
    initialized = nchunks_initialized

    @property
    def is_view(self):
        """A boolean, True if this array is a view on another array."""
        return self._is_view

    @property
    def oindex(self):
        """TODO"""
        return self._oindex

    def __eq__(self, other):
        return (
            isinstance(other, Array) and
            self.store == other.store and
            self.read_only == other.read_only and
            self.path == other.path and
            not self._is_view
            # N.B., no need to compare other properties, should be covered by
            # store comparison
        )

    def __array__(self, *args):
        a = self[...]
        if args:
            a = a.astype(args[0])
        return a

    def __len__(self):
        if self.shape:
            return self.shape[0]
        else:
            raise TypeError('len() of unsized object')

    def __getitem__(self, selection):
        """Retrieve data for some portion of the array. Most NumPy-style
        slicing operations are supported.

        Returns
        -------
        out : ndarray
            A NumPy array containing the data for the requested region.

        Examples
        --------

        Setup a 1-dimensional array::

            >>> import zarr
            >>> import numpy as np
            >>> z = zarr.array(np.arange(100000000), chunks=1000000, dtype='i4')
            >>> z
            <zarr.core.Array (100000000,) int32>

        Take some slices::

            >>> z[5]
            5
            >>> z[:5]
            array([0, 1, 2, 3, 4], dtype=int32)
            >>> z[-5:]
            array([99999995, 99999996, 99999997, 99999998, 99999999], dtype=int32)
            >>> z[5:10]
            array([5, 6, 7, 8, 9], dtype=int32)
            >>> z[:]
            array([       0,        1,        2, ..., 99999997, 99999998, 99999999], dtype=int32)

        Setup a 2-dimensional array::

            >>> import zarr
            >>> import numpy as np
            >>> z = zarr.array(np.arange(100000000).reshape(10000, 10000),
            ...                chunks=(1000, 1000), dtype='i4')
            >>> z
            <zarr.core.Array (10000, 10000) int32>

        Take some slices::

            >>> z[2, 2]
            20002
            >>> z[:2, :2]
            array([[    0,     1],
                   [10000, 10001]], dtype=int32)
            >>> z[:2]
            array([[    0,     1,     2, ...,  9997,  9998,  9999],
                   [10000, 10001, 10002, ..., 19997, 19998, 19999]], dtype=int32)
            >>> z[:, :2]
            array([[       0,        1],
                   [   10000,    10001],
                   [   20000,    20001],
                   ...,
                   [99970000, 99970001],
                   [99980000, 99980001],
                   [99990000, 99990001]], dtype=int32)
            >>> z[:]
            array([[       0,        1,        2, ...,     9997,     9998,     9999],
                   [   10000,    10001,    10002, ...,    19997,    19998,    19999],
                   [   20000,    20001,    20002, ...,    29997,    29998,    29999],
                   ...,
                   [99970000, 99970001, 99970002, ..., 99979997, 99979998, 99979999],
                   [99980000, 99980001, 99980002, ..., 99989997, 99989998, 99989999],
                   [99990000, 99990001, 99990002, ..., 99999997, 99999998, 99999999]], dtype=int32)

        """

        # delegate to method
        return self.get_basic_selection(selection)

    def get_basic_selection(self, selection, out=None):
        """TODO"""

        # refresh metadata
        if not self._cache_metadata:
            self._load_metadata()

        # handle zero-dimensional arrays
        if self._shape == ():
            return self._get_basic_selection_zd(selection, out=out)
        else:
            return self._get_basic_selection_nd(selection, out=out)

    def _get_basic_selection_zd(self, selection, out=None):
        # special case basic selection for zero-dimensional array

        # check selection is valid
        if selection not in ((), Ellipsis):
            raise IndexError('too many indices for array')

        try:
            # obtain encoded data for chunk
            ckey = self._chunk_key((0,))
            cdata = self.chunk_store[ckey]

        except KeyError:
            # chunk not initialized
            if self._fill_value is not None:
                chunk = np.empty((), dtype=self._dtype)
                chunk.fill(self._fill_value)
            else:
                chunk = np.zeros((), dtype=self._dtype)

        else:
            chunk = self._decode_chunk(cdata)

        # handle selection of the scalar value via empty tuple
        if out is None:
            out = chunk[selection]
        else:
            out[selection] = chunk[selection]

        return out

    def _get_basic_selection_nd(self, selection, out=None):
        # implementation of basic selection for array with at least one dimension

        # setup indexer
        indexer = BasicIndexer(selection, self)

        return self._get_selection(indexer, out=out)

    def get_orthogonal_selection(self, selection, out=None):

        # refresh metadata
        if not self._cache_metadata:
            self._load_metadata()

        # setup indexer
        indexer = OrthogonalIndexer(selection, self)

        return self._get_selection(indexer, out=out)

    def _get_selection(self, indexer, out=None):

        # We iterate over all chunks which overlap the selection and thus contain data that needs
        # to be extracted. Each chunk is processed in turn, extracting the necessary data and
        # storing into the correct location in the output array.

        # N.B., it is an important optimisation that we only visit chunks which overlap the
        # selection. This minimises the nuimber of iterations in the main for loop.

        # determine indices of chunks overlapping the selection
        chunk_ranges, sel_shape = indexer.get_overlapping_chunks()

        # setup output array
        if out is None:
            out = np.empty(sel_shape, dtype=self._dtype, order=self._order)
        else:
            # validate 'out' parameter
            if not hasattr(out, 'shape'):
                raise TypeError('out must be an array-like object')
            if out.shape != sel_shape:
                raise ValueError('out has wrong shape for selection')

        # iterate over chunks in range, i.e., chunks overlapping the selection
        for chunk_coords in itertools.product(*chunk_ranges):

            # obtain selections for chunk and output arrays
            chunk_selection, out_selection = indexer.get_chunk_projection(chunk_coords)

            # load chunk selection into output array
            self._chunk_getitem(chunk_coords, chunk_selection, out, out_selection,
                                squeeze_axes=indexer.squeeze_axes)

        if out.shape:
            return out
        else:
            return out[()]

    def __setitem__(self, item, value):
        """Modify data for some portion of the array.

        Examples
        --------

        Setup a 1-dimensional array::

            >>> import zarr
            >>> z = zarr.zeros(100000000, chunks=1000000, dtype='i4')
            >>> z
            <zarr.core.Array (100000000,) int32>

        Set all array elements to the same scalar value::

            >>> z[:] = 42
            >>> z[:]
            array([42, 42, 42, ..., 42, 42, 42], dtype=int32)

        Set a portion of the array::

            >>> z[:100] = np.arange(100)
            >>> z[-100:] = np.arange(100)[::-1]
            >>> z[:]
            array([0, 1, 2, ..., 2, 1, 0], dtype=int32)

        Setup a 2-dimensional array::

            >>> z = zarr.zeros((10000, 10000), chunks=(1000, 1000), dtype='i4')
            >>> z
            <zarr.core.Array (10000, 10000) int32>

        Set all array elements to the same scalar value::

            >>> z[:] = 42
            >>> z[:]
            array([[42, 42, 42, ..., 42, 42, 42],
                   [42, 42, 42, ..., 42, 42, 42],
                   [42, 42, 42, ..., 42, 42, 42],
                   ...,
                   [42, 42, 42, ..., 42, 42, 42],
                   [42, 42, 42, ..., 42, 42, 42],
                   [42, 42, 42, ..., 42, 42, 42]], dtype=int32)

        Set a portion of the array::

            >>> z[0, :] = np.arange(z.shape[1])
            >>> z[:, 0] = np.arange(z.shape[0])
            >>> z[:]
            array([[   0,    1,    2, ..., 9997, 9998, 9999],
                   [   1,   42,   42, ...,   42,   42,   42],
                   [   2,   42,   42, ...,   42,   42,   42],
                   ...,
                   [9997,   42,   42, ...,   42,   42,   42],
                   [9998,   42,   42, ...,   42,   42,   42],
                   [9999,   42,   42, ...,   42,   42,   42]], dtype=int32)

        """

        # guard conditions
        if self._read_only:
            err_read_only()

        # refresh metadata
        if not self._cache_metadata:
            self._load_metadata_nosync()

        # handle zero-dimensional arrays
        if self._shape == ():
            return self._setitem_zd(item, value)
        else:
            return self._setitem_nd(item, value)

    def _setitem_zd(self, item, value):
        # special case __setitem__ for zero-dimensional array

        # check item is valid
        if item not in ((), Ellipsis):
            raise IndexError('too many indices for array')

        # setup data to store
        arr = np.asarray(value, dtype=self._dtype)

        # check value
        if arr.shape != ():
            raise ValueError('bad value; expected scalar, found %r' % value)

        # obtain key for chunk storage
        ckey = self._chunk_key((0,))

        # encode and store
        cdata = self._encode_chunk(arr)
        self.chunk_store[ckey] = cdata

    def _setitem_nd(self, item, value):
        # implementation of __setitem__ for array with at least one dimension

        # normalize selection
        selection = normalize_array_selection(item, self._shape, self._chunks)

        # figure out if we're doing advanced indexing, count number of advanced selections - if
        # more than one need special handling, because we are doing orthogonal indexing here,
        # which is different from fancy indexing if there is more than one array selection
        n_advanced_selection = sum(1 for dim_sel in selection
                                   if not isinstance(dim_sel, (int, slice)))

        # axes that need to get squeezed out if doing advanced selection
        if n_advanced_selection > 0:
            squeeze_axes = tuple([i for i, dim_sel in enumerate(selection)
                                  if isinstance(dim_sel, int)])
        else:
            squeeze_axes = None

        # determine indices of chunks overlapping the selection
        chunk_ranges, sel_shape = get_chunks_for_selection(selection, self._chunks)

        # check value shape
        if np.isscalar(value):
            pass
        elif sel_shape != value.shape:
            raise ValueError('value shape does not match selection shape; expected %s, found %s'
                             % (str(sel_shape), str(value.shape)))

        # iterate over chunks in range
        for chunk_coords in itertools.product(*chunk_ranges):

            # obtain selections for chunk and destination arrays
            chunk_selection, out_selection = \
                get_chunk_selections(selection, chunk_coords, self._chunks, n_advanced_selection)

            if np.isscalar(value):

                # put data
                self._chunk_setitem(chunk_coords, chunk_selection, value)

            else:
                # assume value is array-like

                # put data
                dest = value[out_selection]
                self._chunk_setitem(chunk_coords, chunk_selection, dest, squeeze_axes)

    def _chunk_getitem(self, chunk_coords, chunk_selection, out, out_selection, squeeze_axes=None):
        """Obtain part or whole of a chunk.

        Parameters
        ----------
        chunk_coords : tuple of ints
            Indices of the chunk.
        chunk_selection : selection
            Location of region within the chunk to extract.
        out : ndarray
            Array to store result in.
        out_selection : selection
            Location of region within output array to store results in.
        squeeze_axes : tuple of ints
            Axes to squeeze out of the chunk.

        """

        try:

            # obtain compressed data for chunk
            ckey = self._chunk_key(chunk_coords)
            cdata = self.chunk_store[ckey]

        except KeyError:

            # chunk not initialized
            if self._fill_value is not None:
                out[out_selection] = self._fill_value

        else:

            if isinstance(out, np.ndarray) and \
                    is_total_slice(chunk_selection, self._chunks) and \
                    not self._filters:

                dest = out[out_selection]
                contiguous = ((self._order == 'C' and dest.flags.c_contiguous) or
                              (self._order == 'F' and dest.flags.f_contiguous))

                if contiguous:

                    # optimization: we want the whole chunk, and the destination is
                    # contiguous, so we can decompress directly from the chunk
                    # into the destination array

                    if self._compressor:
                        self._compressor.decode(cdata, dest)
                    else:
                        chunk = np.frombuffer(cdata, dtype=self._dtype)
                        chunk = chunk.reshape(self._chunks, order=self._order)
                        np.copyto(dest, chunk)
                    return

            # decode chunk
            chunk = self._decode_chunk(cdata)

            # set data in output array
            tmp = chunk[chunk_selection]
            if squeeze_axes:
                tmp = np.squeeze(tmp, axis=squeeze_axes)
            out[out_selection] = tmp

    def _chunk_setitem(self, chunk_coords, chunk_selection, value, squeeze_axes=None):
        """Replace part or whole of a chunk.

        Parameters
        ----------
        chunk_coords : tuple of ints
            Indices of the chunk.
        chunk_selection : tuple of slices
            Location of region within the chunk.
        value : scalar or ndarray
            Value to set.

        """

        # synchronization
        if self._synchronizer is None:
            self._chunk_setitem_nosync(chunk_coords, chunk_selection, value, squeeze_axes)
        else:
            # synchronize on the chunk
            ckey = self._chunk_key(chunk_coords)
            with self._synchronizer[ckey]:
                self._chunk_setitem_nosync(chunk_coords, chunk_selection, value, squeeze_axes)

    def _chunk_setitem_nosync(self, chunk_coords, chunk_selection, value, squeeze_axes=None):

        # obtain key for chunk storage
        ckey = self._chunk_key(chunk_coords)

        if is_total_slice(chunk_selection, self._chunks):
            # totally replace chunk

            # optimization: we are completely replacing the chunk, so no need
            # to access the existing chunk data

            if np.isscalar(value):

                # setup array filled with value
                chunk = np.empty(self._chunks, dtype=self._dtype, order=self._order)
                chunk.fill(value)

            else:

                if not self._compressor and not self._filters:

                    # https://github.com/alimanfoo/zarr/issues/79
                    # Ensure a copy is taken so we don't end up storing
                    # a view into someone else's array.
                    # N.B., this assumes that filters or compressor always
                    # take a copy and never attempt to apply encoding in-place.
                    chunk = np.array(value, dtype=self._dtype, order=self._order)

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
                cdata = self.chunk_store[ckey]

            except KeyError:

                # chunk not initialized
                if self._fill_value is not None:
                    chunk = np.empty(self._chunks, dtype=self._dtype, order=self._order)
                    chunk.fill(self._fill_value)
                else:
                    # N.B., use zeros here so any region beyond the array has consistent and
                    # compressible data
                    chunk = np.zeros(self._chunks, dtype=self._dtype, order=self._order)

            else:

                # decode chunk
                chunk = self._decode_chunk(cdata)
                if not chunk.flags.writeable:
                    chunk = chunk.copy(order='K')

            # handle missing singleton dimensions
            if squeeze_axes:
                item = [slice(None)] * self.ndim
                for a in squeeze_axes:
                    item[a] = np.newaxis
                value = value[item]

            # modify
            chunk[chunk_selection] = value

        # encode chunk
        cdata = self._encode_chunk(chunk)

        # store
        self.chunk_store[ckey] = cdata

    def _chunk_key(self, chunk_coords):
        return self._key_prefix + '.'.join(map(str, chunk_coords))

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
        t = type(self)
        r = '<%s.%s' % (t.__module__, t.__name__)
        if self.name:
            r += ' %r' % self.name
        r += ' %s' % str(self.shape)
        r += ' %s' % self.dtype
        r += '>'
        return r

    @property
    def info(self):
        """Report some diagnostic information about the array.

        Examples
        --------
        >>> import zarr
        >>> z = zarr.zeros(1000000, chunks=100000, dtype='i4')
        >>> z.info
        Type               : zarr.core.Array
        Data type          : int32
        Shape              : (1000000,)
        Chunk shape        : (100000,)
        Order              : C
        Read-only          : False
        Compressor         : Blosc(cname='lz4', clevel=5, shuffle=SHUFFLE, blocksize=0)
        Store type         : builtins.dict
        No. bytes          : 4000000 (3.8M)
        No. bytes stored   : ...
        Storage ratio      : ...
        Chunks initialized : 0/10

        """
        return self._info_reporter

    def info_items(self):
        return self._synchronized_op(self._info_items_nosync)

    def _info_items_nosync(self):

        def typestr(o):
            return '%s.%s' % (type(o).__module__, type(o).__name__)

        def bytestr(n):
            if n > 2**10:
                return '%s (%s)' % (n, human_readable_size(n))
            else:
                return str(n)

        items = []

        # basic info
        if self.name is not None:
            items += [('Name', self.name)]
        items += [
            ('Type', typestr(self)),
            ('Data type', '%s' % self.dtype),
            ('Shape', str(self.shape)),
            ('Chunk shape', str(self.chunks)),
            ('Order', self.order),
            ('Read-only', str(self.read_only)),
        ]

        # filters
        if self.filters:
            for i, f in enumerate(self.filters):
                items += [('Filter [%s]' % i, repr(f))]

        # compressor
        items += [('Compressor', repr(self.compressor))]

        # synchronizer
        if self._synchronizer is not None:
            items += [('Synchronizer type', typestr(self._synchronizer))]

        # storage info
        items += [('Store type', typestr(self._store))]
        if self._chunk_store is not None:
            items += [('Chunk store type', typestr(self._chunk_store))]
        items += [('No. bytes', bytestr(self.nbytes))]
        if self.nbytes_stored > 0:
            items += [
                ('No. bytes stored', bytestr(self.nbytes_stored)),
                ('Storage ratio', '%.1f' % (self.nbytes / self.nbytes_stored)),
            ]
        items += [
            ('Chunks initialized', '%s/%s' % (self.nchunks_initialized, self.nchunks))
        ]

        return items

    def __getstate__(self):
        return self._store, self._path, self._read_only, self._chunk_store, self._synchronizer, \
               self._cache_metadata

    def __setstate__(self, state):
        self.__init__(*state)

    def _synchronized_op(self, f, *args, **kwargs):

        # no synchronization
        if self._synchronizer is None:
            self._refresh_metadata_nosync()
            return f(*args, **kwargs)

        else:
            # synchronize on the array
            mkey = self._key_prefix + array_meta_key
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
        """Change the shape of the array by growing or shrinking one or more
        dimensions.

        Examples
        --------
        >>> import zarr
        >>> z = zarr.zeros(shape=(10000, 10000), chunks=(1000, 1000))
        >>> z.shape
        (10000, 10000)
        >>> z.resize(20000, 10000)
        >>> z.shape
        (20000, 10000)
        >>> z.resize(30000, 1000)
        >>> z.shape
        (30000, 1000)

        Notes
        -----
        When resizing an array, the data are not rearranged in any way.

        If one or more dimensions are shrunk, any chunks falling outside the
        new array shape will be deleted from the underlying store.

        """

        return self._write_op(self._resize_nosync, *args)

    def _resize_nosync(self, *args):

        # normalize new shape argument
        old_shape = self._shape
        new_shape = normalize_resize_args(old_shape, *args)
        old_cdata_shape = self._cdata_shape

        # update metadata
        self._shape = new_shape
        self._flush_metadata_nosync()

        # determine the new number and arrangement of chunks
        chunks = self._chunks
        new_cdata_shape = tuple(int(np.ceil(s / c))
                                for s, c in zip(new_shape, chunks))

        # remove any chunks not within range
        chunk_store = self.chunk_store
        for cidx in itertools.product(*[range(n) for n in old_cdata_shape]):
            if all(i < c for i, c in zip(cidx, new_cdata_shape)):
                pass  # keep the chunk
            else:
                key = self._chunk_key(cidx)
                try:
                    del chunk_store[key]
                except KeyError:
                    # chunk not initialized
                    pass

    def append(self, data, axis=0):
        """Append `data` to `axis`.

        Parameters
        ----------
        data : array_like
            Data to be appended.
        axis : int
            Axis along which to append.

        Returns
        -------
        new_shape : tuple

        Notes
        -----
        The size of all dimensions other than `axis` must match between this
        array and `data`.

        Examples
        --------
        >>> import numpy as np
        >>> import zarr
        >>> a = np.arange(10000000, dtype='i4').reshape(10000, 1000)
        >>> z = zarr.array(a, chunks=(1000, 100))
        >>> z.shape
        (10000, 1000)
        >>> z.append(a)
        (20000, 1000)
        >>> z.append(np.vstack([a, a]), axis=1)
        (20000, 2000)
        >>> z
        <zarr.core.Array (20000, 2000) int32>

        """
        return self._write_op(self._append_nosync, data, axis=axis)

    def _append_nosync(self, data, axis=0):

        # ensure data is array-like
        if not hasattr(data, 'shape') or not hasattr(data, 'dtype'):
            data = np.asanyarray(data)

        # ensure shapes are compatible for non-append dimensions
        self_shape_preserved = tuple(s for i, s in enumerate(self._shape)
                                     if i != axis)
        data_shape_preserved = tuple(s for i, s in enumerate(data.shape)
                                     if i != axis)
        if self_shape_preserved != data_shape_preserved:
            raise ValueError('shapes not compatible')

        # remember old shape
        old_shape = self._shape

        # determine new shape
        new_shape = tuple(
            self._shape[i] if i != axis else self._shape[i] + data.shape[i]
            for i in range(len(self._shape))
        )

        # resize
        self._resize_nosync(new_shape)

        # store data
        # noinspection PyTypeChecker
        append_selection = tuple(
            slice(None) if i != axis else slice(old_shape[i], new_shape[i])
            for i in range(len(self._shape))
        )
        self[append_selection] = data

        return new_shape

    def view(self, shape=None, chunks=None, dtype=None,
             fill_value=None, filters=None, read_only=None,
             synchronizer=None):
        """Return an array sharing the same data.

        Parameters
        ----------
        shape : int or tuple of ints
            Array shape.
        chunks : int or tuple of ints, optional
            Chunk shape.
        dtype : string or dtype, optional
            NumPy dtype.
        fill_value : object
            Default value to use for uninitialized portions of the array.
        filters : sequence, optional
            Sequence of filters to use to encode chunk data prior to
            compression.
        read_only : bool, optional
            True if array should be protected against modification.
        synchronizer : object, optional
            Array synchronizer.

        Notes
        -----
        WARNING: This is an experimental feature and should be used with care.
        There are plenty of ways to generate errors and/or cause data
        corruption.

        Examples
        --------

        Bypass filters:

            >>> import zarr
            >>> import numpy as np
            >>> np.random.seed(42)
            >>> labels = [b'female', b'male']
            >>> data = np.random.choice(labels, size=10000)
            >>> filters = [zarr.Categorize(labels=labels,
            ...                                  dtype=data.dtype,
            ...                                  astype='u1')]
            >>> a = zarr.array(data, chunks=1000, filters=filters)
            >>> a[:]
            array([b'female', b'male', b'female', ..., b'male', b'male', b'female'],
                  dtype='|S6')
            >>> v = a.view(dtype='u1', filters=[])
            >>> v.is_view
            True
            >>> v[:]
            array([1, 2, 1, ..., 2, 2, 1], dtype=uint8)

        Views can be used to modify data:

            >>> x = v[:]
            >>> x.sort()
            >>> v[:] = x
            >>> v[:]
            array([1, 1, 1, ..., 2, 2, 2], dtype=uint8)
            >>> a[:]
            array([b'female', b'female', b'female', ..., b'male', b'male', b'male'],
                  dtype='|S6')

        View as a different dtype with the same itemsize:

            >>> data = np.random.randint(0, 2, size=10000, dtype='u1')
            >>> a = zarr.array(data, chunks=1000)
            >>> a[:]
            array([0, 0, 1, ..., 1, 0, 0], dtype=uint8)
            >>> v = a.view(dtype=bool)
            >>> v[:]
            array([False, False,  True, ...,  True, False, False], dtype=bool)
            >>> np.all(a[:].view(dtype=bool) == v[:])
            True

        An array can be viewed with a dtype with a different itemsize, however
        some care is needed to adjust the shape and chunk shape so that chunk
        data is interpreted correctly:

            >>> data = np.arange(10000, dtype='u2')
            >>> a = zarr.array(data, chunks=1000)
            >>> a[:10]
            array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint16)
            >>> v = a.view(dtype='u1', shape=20000, chunks=2000)
            >>> v[:10]
            array([0, 0, 1, 0, 2, 0, 3, 0, 4, 0], dtype=uint8)
            >>> np.all(a[:].view('u1') == v[:])
            True

        Change fill value for uninitialized chunks:

            >>> a = zarr.full(10000, chunks=1000, fill_value=-1, dtype='i1')
            >>> a[:]
            array([-1, -1, -1, ..., -1, -1, -1], dtype=int8)
            >>> v = a.view(fill_value=42)
            >>> v[:]
            array([42, 42, 42, ..., 42, 42, 42], dtype=int8)

        Note that resizing or appending to views is not permitted:

            >>> a = zarr.empty(10000)
            >>> v = a.view()
            >>> try:
            ...     v.resize(20000)
            ... except PermissionError as e:
            ...     print(e)
            not permitted for views

        """

        store = self._store
        chunk_store = self._chunk_store
        path = self._path
        if read_only is None:
            read_only = self._read_only
        if synchronizer is None:
            synchronizer = self._synchronizer
        a = Array(store=store, path=path, chunk_store=chunk_store, read_only=read_only,
                  synchronizer=synchronizer, cache_metadata=True)
        a._is_view = True

        # allow override of some properties
        if dtype is None:
            dtype = self._dtype
        else:
            dtype = np.dtype(dtype)
            a._dtype = dtype
        if shape is None:
            shape = self._shape
        else:
            shape = normalize_shape(shape)
            a._shape = shape
        if chunks is not None:
            chunks = normalize_chunks(chunks, shape, dtype.itemsize)
            a._chunks = chunks
        if fill_value is not None:
            a._fill_value = fill_value
        if filters is not None:
            a._filters = filters

        return a

    def astype(self, dtype):
        """Does on the fly type conversion of the underlying data.

        Parameters
        ----------
        dtype : string or dtype
            NumPy dtype.

        Notes
        -----
        This method returns a new Array object which is a view on the same
        underlying chunk data. Modifying any data via the view is currently
        not permitted and will result in an error. This is an experimental
        feature and its behavior is subject to change in the future.

        See Also
        --------
        Array.view

        Examples
        --------

        >>> import zarr
        >>> import numpy as np
        >>> data = np.arange(100, dtype=np.uint8)
        >>> a = zarr.array(data, chunks=10)
        >>> a[:]
        array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
               16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
               32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
               48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
               64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
               80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
               96, 97, 98, 99], dtype=uint8)
        >>> v = a.astype(np.float32)
        >>> v.is_view
        True
        >>> v[:]
        array([  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,
                10.,  11.,  12.,  13.,  14.,  15.,  16.,  17.,  18.,  19.,
                20.,  21.,  22.,  23.,  24.,  25.,  26.,  27.,  28.,  29.,
                30.,  31.,  32.,  33.,  34.,  35.,  36.,  37.,  38.,  39.,
                40.,  41.,  42.,  43.,  44.,  45.,  46.,  47.,  48.,  49.,
                50.,  51.,  52.,  53.,  54.,  55.,  56.,  57.,  58.,  59.,
                60.,  61.,  62.,  63.,  64.,  65.,  66.,  67.,  68.,  69.,
                70.,  71.,  72.,  73.,  74.,  75.,  76.,  77.,  78.,  79.,
                80.,  81.,  82.,  83.,  84.,  85.,  86.,  87.,  88.,  89.,
                90.,  91.,  92.,  93.,  94.,  95.,  96.,  97.,  98.,  99.],
              dtype=float32)
        """

        dtype = np.dtype(dtype)

        filters = []
        if self._filters:
            filters.extend(self._filters)
        filters.insert(0, AsType(encode_dtype=self._dtype, decode_dtype=dtype))

        return self.view(filters=filters, dtype=dtype, read_only=True)
