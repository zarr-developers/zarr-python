# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import operator
import itertools


import numpy as np


from zarr.compressors import get_compressor_cls
from zarr.util import is_total_slice, normalize_array_selection, \
    get_chunk_range, human_readable_size, normalize_resize_args, \
    normalize_storage_path
from zarr.storage import array_meta_key, attrs_key, listdir, getsize
from zarr.meta import decode_array_metadata, encode_array_metadata
from zarr.attrs import Attributes
from zarr.errors import ReadOnlyError
from zarr.compat import reduce
from zarr.filters import get_filters


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
    initialized
    cdata_shape

    Methods
    -------
    __getitem__
    __setitem__
    resize
    append

    """  # flake8: noqa

    def __init__(self, store, path=None, read_only=False, chunk_store=None,
                 synchronizer=None):
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

        # initialize metadata
        try:
            mkey = self._key_prefix + array_meta_key
            meta_bytes = store[mkey]
        except KeyError:
            raise ValueError('store has no metadata')
        else:
            meta = decode_array_metadata(meta_bytes)
            self._meta = meta
            self._shape = meta['shape']
            self._chunks = meta['chunks']
            self._dtype = meta['dtype']
            self._compression = meta['compression']
            self._compression_opts = meta['compression_opts']
            self._fill_value = meta['fill_value']
            self._order = meta['order']
            compressor_cls = get_compressor_cls(self._compression)
            self._compressor = compressor_cls(self._compression_opts)
            self._filters = get_filters(meta['filters'])
            # TODO validate filter dtypes

        # initialize attributes
        akey = self._key_prefix + attrs_key
        self._attrs = Attributes(store, key=akey, read_only=read_only,
                                 synchronizer=synchronizer)

    def _flush_metadata(self):
        meta = dict(shape=self._shape, chunks=self._chunks, dtype=self._dtype,
                    compression=self._compression,
                    compression_opts=self._compression_opts,
                    fill_value=self._fill_value, order=self._order,
                    filters=self._filters)
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
        return self._shape

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
    def compression(self):
        """A string naming the primary compression algorithm used to
        compress chunks of the array."""
        return self._compression

    @property
    def compression_opts(self):
        """Parameters controlling the behaviour of the primary compression
        algorithm."""
        return self._compression_opts

    @property
    def compressor(self):
        """TODO doc me"""
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
    def synchronizer(self):
        """TODO doc me"""
        return self._synchronizer

    @property
    def filters(self):
        """TODO doc me"""
        return self._filters

    @property
    def attrs(self):
        """A MutableMapping containing user-defined attributes. Note that
        attribute values must be JSON serializable."""
        return self._attrs

    @property
    def size(self):
        """The total number of elements in the array."""
        return reduce(operator.mul, self._shape)

    @property
    def itemsize(self):
        """The size in bytes of each item in the array."""
        return self._dtype.itemsize

    @property
    def nbytes(self):
        """The total number of bytes that would be required to store the
        array without compression."""
        return self.size * self.itemsize

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
    def initialized(self):
        """The number of chunks that have been initialized with some data."""
        return sum(1 for k in listdir(self._chunk_store, self._path)
                   if k not in [array_meta_key, attrs_key])

    @property
    def cdata_shape(self):
        """A tuple of integers describing the number of chunks along each
        dimension of the array."""
        return tuple(
            int(np.ceil(s / c)) for s, c in zip(self._shape, self._chunks)
        )

    def __eq__(self, other):
        return (
            isinstance(other, Array) and
            self.store == other.store and
            self.read_only == other.read_only and
            self.path == other.path
            # N.B., no need to compare other properties, should be covered by
            # store comparison
        )

    def __array__(self):
        return self[:]

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
            zarr.core.Array((100000000,), int32, chunks=(1000000,), order=C)
              compression: blosc; compression_opts: {'clevel': 5, 'cname': 'lz4', 'shuffle': 1}
              nbytes: 381.5M; nbytes_stored: 6.7M; ratio: 56.8; initialized: 100/100
              store: builtins.dict

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
            zarr.core.Array((10000, 10000), int32, chunks=(1000, 1000), order=C)
              compression: blosc; compression_opts: {'clevel': 5, 'cname': 'lz4', 'shuffle': 1}
              nbytes: 381.5M; nbytes_stored: 9.5M; ratio: 40.1; initialized: 100/100
              store: builtins.dict

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

    def __setitem__(self, key, value):
        """Modify data for some portion of the array.

        Examples
        --------

        Setup a 1-dimensional array::

            >>> import zarr
            >>> z = zarr.zeros(100000000, chunks=1000000, dtype='i4')
            >>> z
            zarr.core.Array((100000000,), int32, chunks=(1000000,), order=C)
              compression: blosc; compression_opts: {'clevel': 5, 'cname': 'lz4', 'shuffle': 1}
              nbytes: 381.5M; nbytes_stored: 312; ratio: 1282051.3; initialized: 0/100
              store: builtins.dict

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
            zarr.core.Array((10000, 10000), int32, chunks=(1000, 1000), order=C)
              compression: blosc; compression_opts: {'clevel': 5, 'cname': 'lz4', 'shuffle': 1}
              nbytes: 381.5M; nbytes_stored: 334; ratio: 1197604.8; initialized: 0/100
              store: builtins.dict

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
            raise ReadOnlyError('array is read-only')

        # normalize selection
        selection = normalize_array_selection(key, self._shape)

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
                self._compressor.decompress(cdata, dest)

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

        # synchronization
        if self._synchronizer is None:
            self._chunk_setitem_nosync(cidx, key, value)
        else:
            # synchronize on the chunk
            ckey = self._chunk_key(cidx)
            with self._synchronizer[ckey]:
                self._chunk_setitem_nosync(cidx, key, value)

    def _chunk_setitem_nosync(self, cidx, key, value):

        # obtain key for chunk storage
        ckey = self._chunk_key(cidx)

        if is_total_slice(key, self._chunks):
            # totally replace chunk

            # optimization: we are completely replacing the chunk, so no need
            # to access the existing chunk data

            if np.isscalar(value):

                # setup array filled with value
                chunk = np.empty(self._chunks, dtype=self._dtype,
                                 order=self._order)
                chunk.fill(value)

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
            chunk[key] = value

        # encode chunk
        cdata = self._encode_chunk(chunk)

        # store
        self._chunk_store[ckey] = cdata

    def _chunk_key(self, cidx):
        return self._key_prefix + '.'.join(map(str, cidx))

    def _decode_chunk(self, cdata):

        # decompress
        chunk = self._compressor.decompress(cdata)

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
        cdata = self._compressor.compress(chunk)

        return cdata

    def __repr__(self):
        r = '%s.%s(' % (type(self).__module__, type(self).__name__)
        if self.name:
            r += '%s, ' % self.name
        r += '%s, ' % str(self.shape)
        r += '%s, ' % str(self.dtype)
        r += 'chunks=%s, ' % str(self.chunks)
        r += 'order=%s' % self.order
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
        if self._filters:
            r += '\n  filters: %s' % ', '.join([f.filter_name for f in
                                                self._filters])
        r += '\n  store: %s.%s' % (type(self.store).__module__,
                                   type(self.store).__name__)
        if self._store != self._chunk_store:
            r += '\n  chunk_store: %s.%s' % \
                 (type(self._chunk_store).__module__,
                  type(self._chunk_store).__name__)
        if self._synchronizer is not None:
            r += ('\n  synchronizer: %s.%s' %
                  (type(self._synchronizer).__module__,
                   type(self._synchronizer).__name__))
        return r

    def __getstate__(self):
        return self._store, self._path, self._read_only, self._chunk_store, \
               self._synchronizer

    def __setstate__(self, state):
        self.__init__(*state)

    def _write_op(self, f, *args, **kwargs):

        # guard condition
        if self._read_only:
            raise ReadOnlyError('array is read-only')

        # synchronization
        if self._synchronizer is None:
            return f(*args, **kwargs)
        else:
            # synchronize on the array
            mkey = self._key_prefix + array_meta_key
            with self._synchronizer[mkey]:
                return f(*args, **kwargs)

    def resize(self, *args):
        """Change the shape of the array by growing or shrinking one or more
        dimensions.

        Examples
        --------
        >>> import zarr
        >>> z = zarr.zeros(shape=(10000, 10000), chunks=(1000, 1000))
        >>> z
        zarr.core.Array((10000, 10000), float64, chunks=(1000, 1000), order=C)
          compression: blosc; compression_opts: {'clevel': 5, 'cname': 'lz4', 'shuffle': 1}
          nbytes: 762.9M; nbytes_stored: 334; ratio: 2395209.6; initialized: 0/100
          store: builtins.dict
        >>> z.resize(20000, 10000)
        >>> z
        zarr.core.Array((20000, 10000), float64, chunks=(1000, 1000), order=C)
          compression: blosc; compression_opts: {'clevel': 5, 'cname': 'lz4', 'shuffle': 1}
          nbytes: 1.5G; nbytes_stored: 334; ratio: 4790419.2; initialized: 0/200
          store: builtins.dict
        >>> z.resize(30000, 1000)
        >>> z
        zarr.core.Array((30000, 1000), float64, chunks=(1000, 1000), order=C)
          compression: blosc; compression_opts: {'clevel': 5, 'cname': 'lz4', 'shuffle': 1}
          nbytes: 228.9M; nbytes_stored: 333; ratio: 720720.7; initialized: 0/30
          store: builtins.dict

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

        # determine the new number and arrangement of chunks
        chunks = self._chunks
        new_cdata_shape = tuple(int(np.ceil(s / c))
                                for s, c in zip(new_shape, chunks))

        # remove any chunks not within range
        for key in listdir(self._chunk_store, self._path):
            if key not in [array_meta_key, attrs_key]:
                cidx = map(int, key.split('.'))
                if all(i < c for i, c in zip(cidx, new_cdata_shape)):
                    pass  # keep the chunk
                else:
                    del self._chunk_store[self._key_prefix + key]

        # update metadata
        self._shape = new_shape
        self._flush_metadata()

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

        Examples
        --------
        >>> import numpy as np
        >>> import zarr
        >>> a = np.arange(10000000, dtype='i4').reshape(10000, 1000)
        >>> z = zarr.array(a, chunks=(1000, 100))
        >>> z
        zarr.core.Array((10000, 1000), int32, chunks=(1000, 100), order=C)
          compression: blosc; compression_opts: {'clevel': 5, 'cname': 'lz4', 'shuffle': 1}
          nbytes: 38.1M; nbytes_stored: 1.9M; ratio: 20.0; initialized: 100/100
          store: builtins.dict
        >>> z.append(a)
        >>> z
        zarr.core.Array((20000, 1000), int32, chunks=(1000, 100), order=C)
          compression: blosc; compression_opts: {'clevel': 5, 'cname': 'lz4', 'shuffle': 1}
          nbytes: 76.3M; nbytes_stored: 3.8M; ratio: 20.0; initialized: 200/200
          store: builtins.dict
        >>> z.append(np.vstack([a, a]), axis=1)
        >>> z
        zarr.core.Array((20000, 2000), int32, chunks=(1000, 100), order=C)
          compression: blosc; compression_opts: {'clevel': 5, 'cname': 'lz4', 'shuffle': 1}
          nbytes: 152.6M; nbytes_stored: 7.6M; ratio: 20.0; initialized: 400/400
          store: builtins.dict

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
