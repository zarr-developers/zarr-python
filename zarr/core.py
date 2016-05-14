# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from functools import reduce  # TODO PY2 compatibility
import operator
import itertools


import numpy as np


from zarr.compressors import get_compressor_cls
from zarr.util import is_total_slice, normalize_array_selection, \
    get_chunk_range, human_readable_size, normalize_resize_args
from zarr.meta import decode_metadata, encode_metadata
from zarr.attrs import Attributes
from zarr.compat import itervalues
from zarr.errors import ReadOnlyError


class Array(object):
    """Instantiate an array from an initialised store.

    Parameters
    ----------
    store : MutableMapping
        Array store, already initialised.
    readonly : bool, optional
        True if array should be protected against modification.

    Attributes
    ----------
    store
    readonly
    shape
    chunks
    dtype
    compression
    compression_opts
    fill_value
    order
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

    Examples
    --------
    >>> import zarr
    >>> store = dict()
    >>> zarr.init_store(store, shape=(10000, 10000), chunks=(1000, 1000))
    >>> z = zarr.Array(store)
    >>> z
    zarr.core.Array((10000, 10000), float64, chunks=(1000, 1000), order=C)
      compression: blosc; compression_opts: {'clevel': 5, 'cname': 'blosclz', 'shuffle': 1}
      nbytes: 762.9M; nbytes_stored: 320; ratio: 2500000.0; initialized: 0/100
      store: builtins.dict

    """  # flake8: noqa

    def __init__(self, store, readonly=False):
        # N.B., expect at this point store is fully initialised with all
        # configuration metadata fully specified and normalised

        #: store docstring
        self._store = store  #: inline docstring
        self._readonly = readonly

        # initialise metadata
        try:
            meta_bytes = store['meta']
        except KeyError:
            raise ValueError('store has no metadata')
        else:
            meta = decode_metadata(meta_bytes)
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

        # initialise attributes
        self._attrs = Attributes(store, readonly=readonly)

    def flush_metadata(self):
        meta = dict(shape=self._shape, chunks=self._chunks, dtype=self._dtype,
                    compression=self._compression,
                    compression_opts=self._compression_opts,
                    fill_value=self._fill_value, order=self._order)
        self._store['meta'] = encode_metadata(meta)

    @property
    def store(self):
        """A MutableMapping providing the underlying storage for the array."""
        return self._store

    @property
    def readonly(self):
        """A boolean, True if write operations are not permitted."""
        return self._readonly

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
    def fill_value(self):
        """A value used for uninitialized portions of the array."""
        return self._fill_value

    @property
    def order(self):
        """A string indicating the order in which bytes are arranged within
        chunks of the array."""
        return self._order

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
        attributes encoded as JSON."""
        if hasattr(self._store, 'size'):
            # pass through
            return self._store.size
        elif isinstance(self._store, dict):
            # cheap to compute by summing length of values
            return sum(len(v) for v in itervalues(self._store))
        else:
            return -1

    @property
    def initialized(self):
        """The number of chunks that have been initialized with some data."""
        # N.B., expect 'meta' and 'attrs' keys in store also, so subtract 2
        return len(self._store) - 2

    @property
    def cdata_shape(self):
        """A tuple of integers describing the number of chunks along each
        dimension of the array."""
        return tuple(
            int(np.ceil(s / c)) for s, c in zip(self._shape, self._chunks)
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
              compression: blosc; compression_opts: {'clevel': 5, 'cname': 'blosclz', 'shuffle': 1}
              nbytes: 381.5M; nbytes_stored: 6.8M; ratio: 56.0; initialized: 100/100
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
              compression: blosc; compression_opts: {'clevel': 5, 'cname': 'blosclz', 'shuffle': 1}
              nbytes: 381.5M; nbytes_stored: 10.0M; ratio: 38.0; initialized: 100/100
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
              compression: blosc; compression_opts: {'clevel': 5, 'cname': 'blosclz', 'shuffle': 1}
              nbytes: 381.5M; nbytes_stored: 295; ratio: 1355932.2; initialized: 0/100
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
              compression: blosc; compression_opts: {'clevel': 5, 'cname': 'blosclz', 'shuffle': 1}
              nbytes: 381.5M; nbytes_stored: 317; ratio: 1261829.7; initialized: 0/100
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
        if self._readonly:
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
            ckey = '.'.join(map(str, cidx))
            cdata = self._store[ckey]

        except KeyError:

            # chunk not initialized
            if self._fill_value is not None:
                dest.fill(self._fill_value)

        else:

            if is_total_slice(item, self._chunks) and \
                    ((self._order == 'C' and dest.flags.c_contiguous) or
                     (self._order == 'F' and dest.flags.f_contiguous)):

                # optimisation: we want the whole chunk, and the destination is
                # contiguous, so we can decompress directly from the chunk
                # into the destination array
                self._compressor.decompress(cdata, dest)

            else:

                # decompress chunk
                chunk = np.empty(self._chunks, dtype=self._dtype,
                                 order=self._order)
                self._compressor.decompress(cdata, chunk)

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

        if is_total_slice(key, self._chunks):

            # optimisation: we are completely replacing the chunk, so no need
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
                ckey = '.'.join(map(str, cidx))
                cdata = self._store[ckey]

            except KeyError:

                # chunk not initialized
                chunk = np.empty(self._chunks, dtype=self._dtype,
                                 order=self._order)
                if self._fill_value is not None:
                    chunk.fill(self._fill_value)

            else:

                # decompress chunk
                chunk = np.empty(self._chunks, dtype=self._dtype,
                                 order=self._order)
                self._compressor.decompress(cdata, chunk)

            # modify
            chunk[key] = value

        # compress
        cdata = self._compressor.compress(chunk)

        # store
        ckey = '.'.join(map(str, cidx))
        self._store[ckey] = cdata

    def __repr__(self):
        r = '%s.%s(' % (type(self).__module__, type(self).__name__)
        r += '%s' % str(self._shape)
        r += ', %s' % str(self._dtype)
        r += ', chunks=%s' % str(self._chunks)
        r += ', order=%s' % self._order
        r += ')'
        r += '\n  compression: %s' % self._compression
        r += '; compression_opts: %s' % str(self._compression_opts)
        r += '\n  nbytes: %s' % human_readable_size(self.nbytes)
        if self.nbytes_stored > 0:
            r += '; nbytes_stored: %s' % human_readable_size(
                self.nbytes_stored)
            r += '; ratio: %.1f' % (self.nbytes / self.nbytes_stored)
        n_chunks = reduce(operator.mul, self.cdata_shape)
        r += '; initialized: %s/%s' % (self.initialized, n_chunks)
        r += '\n  store: %s.%s' % (type(self._store).__module__,
                                   type(self._store).__name__)
        return r

    def __getstate__(self):
        return self._store, self._readonly

    def __setstate__(self, state):
        self.__init__(*state)

    def resize(self, *args):
        """Change the shape of the array by growing or shrinking one or more
        dimensions.

        Examples
        --------
        >>> import zarr
        >>> z = zarr.zeros(shape=(10000, 10000), chunks=(1000, 1000))
        >>> z
        zarr.core.Array((10000, 10000), float64, chunks=(1000, 1000), order=C)
          compression: blosc; compression_opts: {'clevel': 5, 'cname': 'blosclz', 'shuffle': 1}
          nbytes: 762.9M; nbytes_stored: 317; ratio: 2523659.3; initialized: 0/100
          store: builtins.dict
        >>> z.resize(20000, 10000)
        >>> z
        zarr.core.Array((20000, 10000), float64, chunks=(1000, 1000), order=C)
          compression: blosc; compression_opts: {'clevel': 5, 'cname': 'blosclz', 'shuffle': 1}
          nbytes: 1.5G; nbytes_stored: 317; ratio: 5047318.6; initialized: 0/200
          store: builtins.dict
        >>> z.resize(30000, 1000)
        >>> z
        zarr.core.Array((30000, 1000), float64, chunks=(1000, 1000), order=C)
          compression: blosc; compression_opts: {'clevel': 5, 'cname': 'blosclz', 'shuffle': 1}
          nbytes: 228.9M; nbytes_stored: 316; ratio: 759493.7; initialized: 0/30
          store: builtins.dict

        Notes
        -----
        When resizing an array, the data are not rearranged in any way.

        If one or more dimensions are shrunk, any chunks falling outside the
        new array shape will be deleted from the underlying store.

        """  # flake8: noqa

        # guard conditions
        if self._readonly:
            raise ReadOnlyError('array is read-only')

        # normalize new shape argument
        old_shape = self._shape
        new_shape = normalize_resize_args(old_shape, *args)

        # determine the new number and arrangement of chunks
        chunks = self._chunks
        new_cdata_shape = tuple(int(np.ceil(s / c))
                                for s, c in zip(new_shape, chunks))

        # remove any chunks not within range
        for key in list(self._store):
            if key not in ['meta', 'attrs']:
                cidx = map(int, key.split('.'))
                if all(i < c for i, c in zip(cidx, new_cdata_shape)):
                    pass  # keep the chunk
                else:
                    del self._store[key]

        # update metadata
        self._shape = new_shape
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

        Examples
        --------
        >>> import numpy as np
        >>> import zarr
        >>> a = np.arange(10000000, dtype='i4').reshape(10000, 1000)
        >>> z = zarr.array(a, chunks=(1000, 100))
        >>> z
        zarr.core.Array((10000, 1000), int32, chunks=(1000, 100), order=C)
          compression: blosc; compression_opts: {'clevel': 5, 'cname': 'blosclz', 'shuffle': 1}
          nbytes: 38.1M; nbytes_stored: 2.0M; ratio: 19.3; initialized: 100/100
          store: builtins.dict
        >>> z.append(a)
        >>> z
        zarr.core.Array((20000, 1000), int32, chunks=(1000, 100), order=C)
          compression: blosc; compression_opts: {'clevel': 5, 'cname': 'blosclz', 'shuffle': 1}
          nbytes: 76.3M; nbytes_stored: 4.0M; ratio: 19.3; initialized: 200/200
          store: builtins.dict
        >>> z.append(np.vstack([a, a]), axis=1)
        >>> z
        zarr.core.Array((20000, 2000), int32, chunks=(1000, 100), order=C)
          compression: blosc; compression_opts: {'clevel': 5, 'cname': 'blosclz', 'shuffle': 1}
          nbytes: 152.6M; nbytes_stored: 7.9M; ratio: 19.3; initialized: 400/400
          store: builtins.dict

        """

        # guard conditions
        if self._readonly:
            raise ReadOnlyError('array is read-only')

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
        self.resize(new_shape)

        # store data
        # noinspection PyTypeChecker
        append_selection = tuple(
            slice(None) if i != axis else slice(old_shape[i], new_shape[i])
            for i in range(len(self._shape))
        )
        self[append_selection] = data
