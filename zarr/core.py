# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import operator
import itertools


import numpy as np


from zarr.util import is_total_slice, normalize_array_selection, \
    get_chunk_range, human_readable_size, normalize_resize_args, \
    normalize_storage_path, normalize_shape, normalize_chunks
from zarr.storage import array_meta_key, attrs_key, listdir, getsize
from zarr.meta import decode_array_metadata, encode_array_metadata
from zarr.attrs import Attributes
from zarr.errors import PermissionError, err_read_only, err_array_not_found
from zarr.compat import reduce
from zarr.codecs import get_codec


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

    Methods
    -------
    __getitem__
    __setitem__
    resize
    append
    view

    """  # flake8: noqa

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
        self._is_view = False

        # initialize metadata
        self._load_metadata()

        # initialize attributes
        akey = self._key_prefix + attrs_key
        self._attrs = Attributes(store, key=akey, read_only=read_only,
                                 synchronizer=synchronizer)

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
        """A MutableMapping providing the underlying storage for array
        chunks."""
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
        return reduce(operator.mul, self._shape)

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
        if self._store == self._chunk_store:
            return m
        else:
            n = getsize(self._chunk_store, self._path)
            if m < 0 or n < 0:
                return -1
            else:
                return m + n

    @property
    def _cdata_shape(self):
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
        return reduce(operator.mul, self._cdata_shape)

    @property
    def nchunks(self):
        """Total number of chunks."""
        self._refresh_metadata()
        return self._nchunks

    @property
    def nchunks_initialized(self):
        """The number of chunks that have been initialized with some data."""
        return sum(1 for k in listdir(self._chunk_store, self._path)
                   if k not in [array_meta_key, attrs_key])

    # backwards compability
    initialized = nchunks_initialized

    @property
    def is_view(self):
        """A boolean, True if this array is a view on another array."""
        return self._is_view

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
        a = self[:]
        if args:
            a = a.astype(args[0])
        return a

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, item):
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
            Array((100000000,), int32, chunks=(1000000,), order=C)
              nbytes: 381.5M; nbytes_stored: 6.4M; ratio: 59.9; initialized: 100/100
              compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
              store: dict

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
            Array((10000, 10000), int32, chunks=(1000, 1000), order=C)
              nbytes: 381.5M; nbytes_stored: 9.2M; ratio: 41.6; initialized: 100/100
              compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
              store: dict

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

        """  # flake8: noqa

        # refresh metadata
        if not self._cache_metadata:
            self._load_metadata()

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
        """Modify data for some portion of the array.

        Examples
        --------

        Setup a 1-dimensional array::

            >>> import zarr
            >>> z = zarr.zeros(100000000, chunks=1000000, dtype='i4')
            >>> z
            Array((100000000,), int32, chunks=(1000000,), order=C)
              nbytes: 381.5M; nbytes_stored: 301; ratio: 1328903.7; initialized: 0/100
              compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
              store: dict

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
            Array((10000, 10000), int32, chunks=(1000, 1000), order=C)
              nbytes: 381.5M; nbytes_stored: 323; ratio: 1238390.1; initialized: 0/100
              compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
              store: dict

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

        # normalize selection
        selection = normalize_array_selection(item, self._shape)

        # check value shape
        expected_shape = tuple(
            s.stop - s.start for s in selection
            if isinstance(s, slice)
        )
        if np.isscalar(value):
            pass
        elif expected_shape != value.shape:
            raise ValueError('value has wrong shape, expecting %s, found %s'
                             % (str(expected_shape),
                                str(value.shape)))

        # determine indices of chunks overlapping the selection
        chunk_range = get_chunk_range(selection, self._chunks)

        # iterate over chunks in range
        for cidx in itertools.product(*chunk_range):

            # determine chunk offset
            offset = [i * c for i, c in zip(cidx, self._chunks)]

            # determine required index range within chunk
            chunk_selection = tuple(
                slice(max(0, s.start - o), min(c, s.stop - o))
                if isinstance(s, slice)
                else s - o
                for s, o, c in zip(selection, offset, self._chunks)
            )

            if np.isscalar(value):

                # put data
                self._chunk_setitem(cidx, chunk_selection, value)

            else:
                # assume value is array-like

                # determine index within value
                value_selection = tuple(
                    slice(max(0, o - s.start),
                          min(o + c - s.start, s.stop - s.start))
                    for s, o, c in zip(selection, offset, self._chunks)
                    if isinstance(s, slice)
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
        r += '%s, ' % str(self._dtype)
        r += 'chunks=%s, ' % str(self._chunks)
        r += 'order=%s' % self._order
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
        >>> z
        Array((10000, 10000), float64, chunks=(1000, 1000), order=C)
          nbytes: 762.9M; nbytes_stored: 323; ratio: 2476780.2; initialized: 0/100
          compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
          store: dict
        >>> z.resize(20000, 10000)
        >>> z
        Array((20000, 10000), float64, chunks=(1000, 1000), order=C)
          nbytes: 1.5G; nbytes_stored: 323; ratio: 4953560.4; initialized: 0/200
          compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
          store: dict
        >>> z.resize(30000, 1000)
        >>> z
        Array((30000, 1000), float64, chunks=(1000, 1000), order=C)
          nbytes: 228.9M; nbytes_stored: 322; ratio: 745341.6; initialized: 0/30
          compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
          store: dict

        Notes
        -----
        When resizing an array, the data are not rearranged in any way.

        If one or more dimensions are shrunk, any chunks falling outside the
        new array shape will be deleted from the underlying store.

        """  # flake8: noqa

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
        for cidx in itertools.product(*[range(n) for n in old_cdata_shape]):
            if all(i < c for i, c in zip(cidx, new_cdata_shape)):
                pass  # keep the chunk
            else:
                key = self._chunk_key(cidx)
                try:
                    del self._chunk_store[key]
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
        >>> z
        Array((10000, 1000), int32, chunks=(1000, 100), order=C)
          nbytes: 38.1M; nbytes_stored: 1.9M; ratio: 20.3; initialized: 100/100
          compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
          store: dict
        >>> z.append(a)
        (20000, 1000)
        >>> z
        Array((20000, 1000), int32, chunks=(1000, 100), order=C)
          nbytes: 76.3M; nbytes_stored: 3.8M; ratio: 20.3; initialized: 200/200
          compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
          store: dict
        >>> z.append(np.vstack([a, a]), axis=1)
        (20000, 2000)
        >>> z
        Array((20000, 2000), int32, chunks=(1000, 100), order=C)
          nbytes: 152.6M; nbytes_stored: 7.5M; ratio: 20.3; initialized: 400/400
          compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
          store: dict

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

        """  # flake8: noqa

        store = self._store
        chunk_store = self._chunk_store
        path = self._path
        if read_only is None:
            read_only = self._read_only
        if synchronizer is None:
            synchronizer = self._synchronizer
        a = Array(store=store, path=path, chunk_store=chunk_store,
                  read_only=read_only, synchronizer=synchronizer,
                  cache_metadata=True)
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
