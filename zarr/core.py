# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import binascii
import operator
import itertools
import hashlib
import re


import numpy as np
from numcodecs.compat import ensure_bytes, ensure_ndarray


from zarr.util import (is_total_slice, human_readable_size, normalize_resize_args,
                       normalize_storage_path, normalize_shape, normalize_chunks,
                       InfoReporter, check_array_shape, nolock)
from zarr.storage import array_meta_key, attrs_key, listdir, getsize
from zarr.meta import decode_array_metadata, encode_array_metadata
from zarr.attrs import Attributes
from zarr.errors import PermissionError, err_read_only, err_array_not_found
from zarr.compat import reduce
from zarr.codecs import AsType, get_codec
from zarr.indexing import (OIndex, OrthogonalIndexer, BasicIndexer, VIndex,
                           CoordinateIndexer, MaskIndexer, check_fields, pop_fields,
                           ensure_tuple, is_scalar, is_contiguous_selection,
                           err_too_many_indices, check_no_multi_fields)


# noinspection PyUnresolvedReferences
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
        If True (default), array configuration metadata will be cached for the
        lifetime of the object. If False, array metadata will be reloaded
        prior to all data access and modification operations (may incur
        overhead depending on storage and data access pattern).
    cache_attrs : bool, optional
        If True (default), user attributes will be cached for attribute read
        operations. If False, user attributes are reloaded from the store prior
        to all attribute read operations.

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
    vindex
    oindex

    Methods
    -------
    __getitem__
    __setitem__
    get_basic_selection
    set_basic_selection
    get_orthogonal_selection
    set_orthogonal_selection
    get_mask_selection
    set_mask_selection
    get_coordinate_selection
    set_coordinate_selection
    digest
    hexdigest
    resize
    append
    view
    astype

    """

    def __init__(self, store, path=None, read_only=False, chunk_store=None,
                 synchronizer=None, cache_metadata=True, cache_attrs=True):
        # N.B., expect at this point store is fully initialized with all
        # configuration metadata fully specified and normalized

        self._store = store
        self._chunk_store = chunk_store
        self._path = normalize_storage_path(path)
        if self._path:
            self._key_prefix = self._path + '/'
        else:
            self._key_prefix = ''
        self._read_only = bool(read_only)
        self._synchronizer = synchronizer
        self._cache_metadata = cache_metadata
        self._is_view = False

        # initialize metadata
        self._load_metadata()

        # initialize attributes
        akey = self._key_prefix + attrs_key
        self._attrs = Attributes(store, key=akey, read_only=read_only,
                                 synchronizer=synchronizer, cache=cache_attrs)

        # initialize info reporter
        self._info_reporter = InfoReporter(self)

        # initialize indexing helpers
        self._oindex = OIndex(self)
        self._vindex = VIndex(self)

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

            # decode and store metadata as instance members
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
            raise PermissionError('operation not permitted for views')

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
    def basename(self):
        """Final component of name."""
        if self.name is not None:
            return self.name.split('/')[-1]
        return None

    @property
    def read_only(self):
        """A boolean, True if modification operations are not permitted."""
        return self._read_only

    @read_only.setter
    def read_only(self, value):
        self._read_only = bool(value)

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
            return 1,
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
        """Shortcut for orthogonal (outer) indexing, see :func:`get_orthogonal_selection` and
        :func:`set_orthogonal_selection` for documentation and examples."""
        return self._oindex

    @property
    def vindex(self):
        """Shortcut for vectorized (inner) indexing, see :func:`get_coordinate_selection`,
        :func:`set_coordinate_selection`, :func:`get_mask_selection` and
        :func:`set_mask_selection` for documentation and examples."""
        return self._vindex

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

    def __iter__(self):
        if len(self.shape) == 0:
            # Same error as numpy
            raise TypeError("iteration over a 0-d array")
        # Avoid repeatedly decompressing chunks by iterating over the chunks
        # in the first dimension.
        chunk_size = self.chunks[0]
        for j in range(self.shape[0]):
            if j % chunk_size == 0:
                chunk = self[j: j + chunk_size]
            yield chunk[j % chunk_size]

    def __len__(self):
        if self.shape:
            return self.shape[0]
        else:
            # 0-dimensional array, same error message as numpy
            raise TypeError('len() of unsized object')

    def __getitem__(self, selection):
        """Retrieve data for an item or region of the array.

        Parameters
        ----------
        selection : tuple
            An integer index or slice or tuple of int/slice objects specifying the
            requested item or region for each dimension of the array.

        Returns
        -------
        out : ndarray
            A NumPy array containing the data for the requested region.

        Examples
        --------
        Setup a 1-dimensional array::

            >>> import zarr
            >>> import numpy as np
            >>> z = zarr.array(np.arange(100))

        Retrieve a single item::

            >>> z[5]
            5

        Retrieve a region via slicing::

            >>> z[:5]
            array([0, 1, 2, 3, 4])
            >>> z[-5:]
            array([95, 96, 97, 98, 99])
            >>> z[5:10]
            array([5, 6, 7, 8, 9])
            >>> z[5:10:2]
            array([5, 7, 9])
            >>> z[::2]
            array([ 0,  2,  4, ..., 94, 96, 98])

        Load the entire array into memory::

            >>> z[...]
            array([ 0,  1,  2, ..., 97, 98, 99])

        Setup a 2-dimensional array::

            >>> z = zarr.array(np.arange(100).reshape(10, 10))

        Retrieve an item::

            >>> z[2, 2]
            22

        Retrieve a region via slicing::

            >>> z[1:3, 1:3]
            array([[11, 12],
                   [21, 22]])
            >>> z[1:3, :]
            array([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                   [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]])
            >>> z[:, 1:3]
            array([[ 1,  2],
                   [11, 12],
                   [21, 22],
                   [31, 32],
                   [41, 42],
                   [51, 52],
                   [61, 62],
                   [71, 72],
                   [81, 82],
                   [91, 92]])
            >>> z[0:5:2, 0:5:2]
            array([[ 0,  2,  4],
                   [20, 22, 24],
                   [40, 42, 44]])
            >>> z[::2, ::2]
            array([[ 0,  2,  4,  6,  8],
                   [20, 22, 24, 26, 28],
                   [40, 42, 44, 46, 48],
                   [60, 62, 64, 66, 68],
                   [80, 82, 84, 86, 88]])

        Load the entire array into memory::

            >>> z[...]
            array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
                   [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                   [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                   [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
                   [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
                   [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
                   [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
                   [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
                   [80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
                   [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]])

        For arrays with a structured dtype, specific fields can be retrieved, e.g.::

            >>> a = np.array([(b'aaa', 1, 4.2),
            ...               (b'bbb', 2, 8.4),
            ...               (b'ccc', 3, 12.6)],
            ...              dtype=[('foo', 'S3'), ('bar', 'i4'), ('baz', 'f8')])
            >>> z = zarr.array(a)
            >>> z['foo']
            array([b'aaa', b'bbb', b'ccc'],
                  dtype='|S3')

        Notes
        -----
        Slices with step > 1 are supported, but slices with negative step are not.

        Currently the implementation for __getitem__ is provided by
        :func:`get_basic_selection`. For advanced ("fancy") indexing, see the methods
        listed under See Also.

        See Also
        --------
        get_basic_selection, set_basic_selection, get_mask_selection, set_mask_selection,
        get_coordinate_selection, set_coordinate_selection, get_orthogonal_selection,
        set_orthogonal_selection, vindex, oindex, __setitem__

        """

        fields, selection = pop_fields(selection)
        return self.get_basic_selection(selection, fields=fields)

    def get_basic_selection(self, selection=Ellipsis, out=None, fields=None):
        """Retrieve data for an item or region of the array.

        Parameters
        ----------
        selection : tuple
            A tuple specifying the requested item or region for each dimension of the
            array. May be any combination of int and/or slice for multidimensional arrays.
        out : ndarray, optional
            If given, load the selected data directly into this array.
        fields : str or sequence of str, optional
            For arrays with a structured dtype, one or more fields can be specified to
            extract data for.

        Returns
        -------
        out : ndarray
            A NumPy array containing the data for the requested region.

        Examples
        --------
        Setup a 1-dimensional array::

            >>> import zarr
            >>> import numpy as np
            >>> z = zarr.array(np.arange(100))

        Retrieve a single item::

            >>> z.get_basic_selection(5)
            5

        Retrieve a region via slicing::

            >>> z.get_basic_selection(slice(5))
            array([0, 1, 2, 3, 4])
            >>> z.get_basic_selection(slice(-5, None))
            array([95, 96, 97, 98, 99])
            >>> z.get_basic_selection(slice(5, 10))
            array([5, 6, 7, 8, 9])
            >>> z.get_basic_selection(slice(5, 10, 2))
            array([5, 7, 9])
            >>> z.get_basic_selection(slice(None, None, 2))
            array([  0,  2,  4, ..., 94, 96, 98])

        Setup a 2-dimensional array::

            >>> z = zarr.array(np.arange(100).reshape(10, 10))

        Retrieve an item::

            >>> z.get_basic_selection((2, 2))
            22

        Retrieve a region via slicing::

            >>> z.get_basic_selection((slice(1, 3), slice(1, 3)))
            array([[11, 12],
                   [21, 22]])
            >>> z.get_basic_selection((slice(1, 3), slice(None)))
            array([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                   [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]])
            >>> z.get_basic_selection((slice(None), slice(1, 3)))
            array([[ 1,  2],
                   [11, 12],
                   [21, 22],
                   [31, 32],
                   [41, 42],
                   [51, 52],
                   [61, 62],
                   [71, 72],
                   [81, 82],
                   [91, 92]])
            >>> z.get_basic_selection((slice(0, 5, 2), slice(0, 5, 2)))
            array([[ 0,  2,  4],
                   [20, 22, 24],
                   [40, 42, 44]])
            >>> z.get_basic_selection((slice(None, None, 2), slice(None, None, 2)))
            array([[ 0,  2,  4,  6,  8],
                   [20, 22, 24, 26, 28],
                   [40, 42, 44, 46, 48],
                   [60, 62, 64, 66, 68],
                   [80, 82, 84, 86, 88]])

        For arrays with a structured dtype, specific fields can be retrieved, e.g.::

            >>> a = np.array([(b'aaa', 1, 4.2),
            ...               (b'bbb', 2, 8.4),
            ...               (b'ccc', 3, 12.6)],
            ...              dtype=[('foo', 'S3'), ('bar', 'i4'), ('baz', 'f8')])
            >>> z = zarr.array(a)
            >>> z.get_basic_selection(slice(2), fields='foo')
            array([b'aaa', b'bbb'],
                  dtype='|S3')

        Notes
        -----
        Slices with step > 1 are supported, but slices with negative step are not.

        Currently this method provides the implementation for accessing data via the
        square bracket notation (__getitem__). See :func:`__getitem__` for examples
        using the alternative notation.

        See Also
        --------
        set_basic_selection, get_mask_selection, set_mask_selection,
        get_coordinate_selection, set_coordinate_selection, get_orthogonal_selection,
        set_orthogonal_selection, vindex, oindex, __getitem__, __setitem__

        """

        # refresh metadata
        if not self._cache_metadata:
            self._load_metadata()

        # check args
        check_fields(fields, self._dtype)

        # handle zero-dimensional arrays
        if self._shape == ():
            return self._get_basic_selection_zd(selection=selection, out=out,
                                                fields=fields)
        else:
            return self._get_basic_selection_nd(selection=selection, out=out,
                                                fields=fields)

    def _get_basic_selection_zd(self, selection, out=None, fields=None):
        # special case basic selection for zero-dimensional array

        # check selection is valid
        selection = ensure_tuple(selection)
        if selection not in ((), (Ellipsis,)):
            err_too_many_indices(selection, ())

        try:
            # obtain encoded data for chunk
            ckey = self._chunk_key((0,))
            cdata = self.chunk_store[ckey]

        except KeyError:
            # chunk not initialized
            chunk = np.zeros((), dtype=self._dtype)
            if self._fill_value is not None:
                chunk.fill(self._fill_value)

        else:
            chunk = self._decode_chunk(cdata)

        # handle fields
        if fields:
            chunk = chunk[fields]

        # handle selection of the scalar value via empty tuple
        if out is None:
            out = chunk[selection]
        else:
            out[selection] = chunk[selection]

        return out

    def _get_basic_selection_nd(self, selection, out=None, fields=None):
        # implementation of basic selection for array with at least one dimension

        # setup indexer
        indexer = BasicIndexer(selection, self)

        return self._get_selection(indexer=indexer, out=out, fields=fields)

    def get_orthogonal_selection(self, selection, out=None, fields=None):
        """Retrieve data by making a selection for each dimension of the array. For
        example, if an array has 2 dimensions, allows selecting specific rows and/or
        columns. The selection for each dimension can be either an integer (indexing a
        single item), a slice, an array of integers, or a Boolean array where True
        values indicate a selection.

        Parameters
        ----------
        selection : tuple
            A selection for each dimension of the array. May be any combination of int,
            slice, integer array or Boolean array.
        out : ndarray, optional
            If given, load the selected data directly into this array.
        fields : str or sequence of str, optional
            For arrays with a structured dtype, one or more fields can be specified to
            extract data for.

        Returns
        -------
        out : ndarray
            A NumPy array containing the data for the requested selection.

        Examples
        --------
        Setup a 2-dimensional array::

            >>> import zarr
            >>> import numpy as np
            >>> z = zarr.array(np.arange(100).reshape(10, 10))

        Retrieve rows and columns via any combination of int, slice, integer array and/or
        Boolean array::

            >>> z.get_orthogonal_selection(([1, 4], slice(None)))
            array([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                   [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]])
            >>> z.get_orthogonal_selection((slice(None), [1, 4]))
            array([[ 1,  4],
                   [11, 14],
                   [21, 24],
                   [31, 34],
                   [41, 44],
                   [51, 54],
                   [61, 64],
                   [71, 74],
                   [81, 84],
                   [91, 94]])
            >>> z.get_orthogonal_selection(([1, 4], [1, 4]))
            array([[11, 14],
                   [41, 44]])
            >>> sel = np.zeros(z.shape[0], dtype=bool)
            >>> sel[1] = True
            >>> sel[4] = True
            >>> z.get_orthogonal_selection((sel, sel))
            array([[11, 14],
                   [41, 44]])

        For convenience, the orthogonal selection functionality is also available via the
        `oindex` property, e.g.::

            >>> z.oindex[[1, 4], :]
            array([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                   [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]])
            >>> z.oindex[:, [1, 4]]
            array([[ 1,  4],
                   [11, 14],
                   [21, 24],
                   [31, 34],
                   [41, 44],
                   [51, 54],
                   [61, 64],
                   [71, 74],
                   [81, 84],
                   [91, 94]])
            >>> z.oindex[[1, 4], [1, 4]]
            array([[11, 14],
                   [41, 44]])
            >>> sel = np.zeros(z.shape[0], dtype=bool)
            >>> sel[1] = True
            >>> sel[4] = True
            >>> z.oindex[sel, sel]
            array([[11, 14],
                   [41, 44]])

        Notes
        -----
        Orthogonal indexing is also known as outer indexing.

        Slices with step > 1 are supported, but slices with negative step are not.

        See Also
        --------
        get_basic_selection, set_basic_selection, get_mask_selection, set_mask_selection,
        get_coordinate_selection, set_coordinate_selection, set_orthogonal_selection,
        vindex, oindex, __getitem__, __setitem__

        """

        # refresh metadata
        if not self._cache_metadata:
            self._load_metadata()

        # check args
        check_fields(fields, self._dtype)

        # setup indexer
        indexer = OrthogonalIndexer(selection, self)

        return self._get_selection(indexer=indexer, out=out, fields=fields)

    def get_coordinate_selection(self, selection, out=None, fields=None):
        """Retrieve a selection of individual items, by providing the indices
        (coordinates) for each selected item.

        Parameters
        ----------
        selection : tuple
            An integer (coordinate) array for each dimension of the array.
        out : ndarray, optional
            If given, load the selected data directly into this array.
        fields : str or sequence of str, optional
            For arrays with a structured dtype, one or more fields can be specified to
            extract data for.

        Returns
        -------
        out : ndarray
            A NumPy array containing the data for the requested selection.

        Examples
        --------
        Setup a 2-dimensional array::

            >>> import zarr
            >>> import numpy as np
            >>> z = zarr.array(np.arange(100).reshape(10, 10))

        Retrieve items by specifying their coordinates::

            >>> z.get_coordinate_selection(([1, 4], [1, 4]))
            array([11, 44])

        For convenience, the coordinate selection functionality is also available via the
        `vindex` property, e.g.::

            >>> z.vindex[[1, 4], [1, 4]]
            array([11, 44])

        Notes
        -----
        Coordinate indexing is also known as point selection, and is a form of vectorized
        or inner indexing.

        Slices are not supported. Coordinate arrays must be provided for all dimensions
        of the array.

        Coordinate arrays may be multidimensional, in which case the output array will
        also be multidimensional. Coordinate arrays are broadcast against each other
        before being applied. The shape of the output will be the same as the shape of
        each coordinate array after broadcasting.

        See Also
        --------
        get_basic_selection, set_basic_selection, get_mask_selection, set_mask_selection,
        get_orthogonal_selection, set_orthogonal_selection, set_coordinate_selection,
        vindex, oindex, __getitem__, __setitem__

        """

        # refresh metadata
        if not self._cache_metadata:
            self._load_metadata()

        # check args
        check_fields(fields, self._dtype)

        # setup indexer
        indexer = CoordinateIndexer(selection, self)

        # handle output - need to flatten
        if out is not None:
            out = out.reshape(-1)

        out = self._get_selection(indexer=indexer, out=out, fields=fields)

        # restore shape
        out = out.reshape(indexer.sel_shape)

        return out

    def get_mask_selection(self, selection, out=None, fields=None):
        """Retrieve a selection of individual items, by providing a Boolean array of the
        same shape as the array against which the selection is being made, where True
        values indicate a selected item.

        Parameters
        ----------
        selection : ndarray, bool
            A Boolean array of the same shape as the array against which the selection is
            being made.
        out : ndarray, optional
            If given, load the selected data directly into this array.
        fields : str or sequence of str, optional
            For arrays with a structured dtype, one or more fields can be specified to
            extract data for.

        Returns
        -------
        out : ndarray
            A NumPy array containing the data for the requested selection.

        Examples
        --------
        Setup a 2-dimensional array::

            >>> import zarr
            >>> import numpy as np
            >>> z = zarr.array(np.arange(100).reshape(10, 10))

        Retrieve items by specifying a maks::

            >>> sel = np.zeros_like(z, dtype=bool)
            >>> sel[1, 1] = True
            >>> sel[4, 4] = True
            >>> z.get_mask_selection(sel)
            array([11, 44])

        For convenience, the mask selection functionality is also available via the
        `vindex` property, e.g.::

            >>> z.vindex[sel]
            array([11, 44])

        Notes
        -----
        Mask indexing is a form of vectorized or inner indexing, and is equivalent to
        coordinate indexing. Internally the mask array is converted to coordinate
        arrays by calling `np.nonzero`.

        See Also
        --------
        get_basic_selection, set_basic_selection, set_mask_selection,
        get_orthogonal_selection, set_orthogonal_selection, get_coordinate_selection,
        set_coordinate_selection, vindex, oindex, __getitem__, __setitem__

        """

        # refresh metadata
        if not self._cache_metadata:
            self._load_metadata()

        # check args
        check_fields(fields, self._dtype)

        # setup indexer
        indexer = MaskIndexer(selection, self)

        return self._get_selection(indexer=indexer, out=out, fields=fields)

    def _get_selection(self, indexer, out=None, fields=None):

        # We iterate over all chunks which overlap the selection and thus contain data
        # that needs to be extracted. Each chunk is processed in turn, extracting the
        # necessary data and storing into the correct location in the output array.

        # N.B., it is an important optimisation that we only visit chunks which overlap
        # the selection. This minimises the number of iterations in the main for loop.

        # check fields are sensible
        out_dtype = check_fields(fields, self._dtype)

        # determine output shape
        out_shape = indexer.shape

        # setup output array
        if out is None:
            out = np.empty(out_shape, dtype=out_dtype, order=self._order)
        else:
            check_array_shape('out', out, out_shape)

        # iterate over chunks
        for chunk_coords, chunk_selection, out_selection in indexer:

            # load chunk selection into output array
            self._chunk_getitem(chunk_coords, chunk_selection, out, out_selection,
                                drop_axes=indexer.drop_axes, fields=fields)

        if out.shape:
            return out
        else:
            return out[()]

    def __setitem__(self, selection, value):
        """Modify data for an item or region of the array.

        Parameters
        ----------
        selection : tuple
            An integer index or slice or tuple of int/slice specifying the requested
            region for each dimension of the array.
        value : scalar or array-like
            Value to be stored into the array.

        Examples
        --------
        Setup a 1-dimensional array::

            >>> import zarr
            >>> z = zarr.zeros(100, dtype=int)

        Set all array elements to the same scalar value::

            >>> z[...] = 42
            >>> z[...]
            array([42, 42, 42, ..., 42, 42, 42])

        Set a portion of the array::

            >>> z[:10] = np.arange(10)
            >>> z[-10:] = np.arange(10)[::-1]
            >>> z[...]
            array([ 0, 1, 2, ..., 2, 1, 0])

        Setup a 2-dimensional array::

            >>> z = zarr.zeros((5, 5), dtype=int)

        Set all array elements to the same scalar value::

            >>> z[...] = 42

        Set a portion of the array::

            >>> z[0, :] = np.arange(z.shape[1])
            >>> z[:, 0] = np.arange(z.shape[0])
            >>> z[...]
            array([[ 0,  1,  2,  3,  4],
                   [ 1, 42, 42, 42, 42],
                   [ 2, 42, 42, 42, 42],
                   [ 3, 42, 42, 42, 42],
                   [ 4, 42, 42, 42, 42]])

        For arrays with a structured dtype, specific fields can be modified, e.g.::

            >>> a = np.array([(b'aaa', 1, 4.2),
            ...               (b'bbb', 2, 8.4),
            ...               (b'ccc', 3, 12.6)],
            ...              dtype=[('foo', 'S3'), ('bar', 'i4'), ('baz', 'f8')])
            >>> z = zarr.array(a)
            >>> z['foo'] = b'zzz'
            >>> z[...]
            array([(b'zzz', 1,   4.2), (b'zzz', 2,   8.4), (b'zzz', 3,  12.6)],
                  dtype=[('foo', 'S3'), ('bar', '<i4'), ('baz', '<f8')])

        Notes
        -----
        Slices with step > 1 are supported, but slices with negative step are not.

        Currently the implementation for __setitem__ is provided by
        :func:`set_basic_selection`, which means that only integers and slices are
        supported within the selection. For advanced ("fancy") indexing, see the
        methods listed under See Also.

        See Also
        --------
        get_basic_selection, set_basic_selection, get_mask_selection, set_mask_selection,
        get_coordinate_selection, set_coordinate_selection, get_orthogonal_selection,
        set_orthogonal_selection, vindex, oindex, __getitem__

        """

        fields, selection = pop_fields(selection)
        self.set_basic_selection(selection, value, fields=fields)

    def set_basic_selection(self, selection, value, fields=None):
        """Modify data for an item or region of the array.

        Parameters
        ----------
        selection : tuple
            An integer index or slice or tuple of int/slice specifying the requested
            region for each dimension of the array.
        value : scalar or array-like
            Value to be stored into the array.
        fields : str or sequence of str, optional
            For arrays with a structured dtype, one or more fields can be specified to set
            data for.

        Examples
        --------
        Setup a 1-dimensional array::

            >>> import zarr
            >>> import numpy as np
            >>> z = zarr.zeros(100, dtype=int)

        Set all array elements to the same scalar value::

            >>> z.set_basic_selection(..., 42)
            >>> z[...]
            array([42, 42, 42, ..., 42, 42, 42])

        Set a portion of the array::

            >>> z.set_basic_selection(slice(10), np.arange(10))
            >>> z.set_basic_selection(slice(-10, None), np.arange(10)[::-1])
            >>> z[...]
            array([ 0, 1, 2, ..., 2, 1, 0])

        Setup a 2-dimensional array::

            >>> z = zarr.zeros((5, 5), dtype=int)

        Set all array elements to the same scalar value::

            >>> z.set_basic_selection(..., 42)

        Set a portion of the array::

            >>> z.set_basic_selection((0, slice(None)), np.arange(z.shape[1]))
            >>> z.set_basic_selection((slice(None), 0), np.arange(z.shape[0]))
            >>> z[...]
            array([[ 0,  1,  2,  3,  4],
                   [ 1, 42, 42, 42, 42],
                   [ 2, 42, 42, 42, 42],
                   [ 3, 42, 42, 42, 42],
                   [ 4, 42, 42, 42, 42]])

        For arrays with a structured dtype, the `fields` parameter can be used to set
        data for a specific field, e.g.::

            >>> a = np.array([(b'aaa', 1, 4.2),
            ...               (b'bbb', 2, 8.4),
            ...               (b'ccc', 3, 12.6)],
            ...              dtype=[('foo', 'S3'), ('bar', 'i4'), ('baz', 'f8')])
            >>> z = zarr.array(a)
            >>> z.set_basic_selection(slice(0, 2), b'zzz', fields='foo')
            >>> z[:]
            array([(b'zzz', 1,   4.2), (b'zzz', 2,   8.4), (b'ccc', 3,  12.6)],
                  dtype=[('foo', 'S3'), ('bar', '<i4'), ('baz', '<f8')])

        Notes
        -----
        This method provides the underlying implementation for modifying data via square
        bracket notation, see :func:`__setitem__` for equivalent examples using the
        alternative notation.

        See Also
        --------
        get_basic_selection, get_mask_selection, set_mask_selection,
        get_coordinate_selection, set_coordinate_selection, get_orthogonal_selection,
        set_orthogonal_selection, vindex, oindex, __getitem__, __setitem__

        """

        # guard conditions
        if self._read_only:
            err_read_only()

        # refresh metadata
        if not self._cache_metadata:
            self._load_metadata_nosync()

        # handle zero-dimensional arrays
        if self._shape == ():
            return self._set_basic_selection_zd(selection, value, fields=fields)
        else:
            return self._set_basic_selection_nd(selection, value, fields=fields)

    def set_orthogonal_selection(self, selection, value, fields=None):
        """Modify data via a selection for each dimension of the array.

        Parameters
        ----------
        selection : tuple
            A selection for each dimension of the array. May be any combination of int,
            slice, integer array or Boolean array.
        value : scalar or array-like
            Value to be stored into the array.
        fields : str or sequence of str, optional
            For arrays with a structured dtype, one or more fields can be specified to set
            data for.

        Examples
        --------
        Setup a 2-dimensional array::

            >>> import zarr
            >>> import numpy as np
            >>> z = zarr.zeros((5, 5), dtype=int)

        Set data for a selection of rows::

            >>> z.set_orthogonal_selection(([1, 4], slice(None)), 1)
            >>> z[...]
            array([[0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1]])

        Set data for a selection of columns::

            >>> z.set_orthogonal_selection((slice(None), [1, 4]), 2)
            >>> z[...]
            array([[0, 2, 0, 0, 2],
                   [1, 2, 1, 1, 2],
                   [0, 2, 0, 0, 2],
                   [0, 2, 0, 0, 2],
                   [1, 2, 1, 1, 2]])

        Set data for a selection of rows and columns::

            >>> z.set_orthogonal_selection(([1, 4], [1, 4]), 3)
            >>> z[...]
            array([[0, 2, 0, 0, 2],
                   [1, 3, 1, 1, 3],
                   [0, 2, 0, 0, 2],
                   [0, 2, 0, 0, 2],
                   [1, 3, 1, 1, 3]])

        For convenience, this functionality is also available via the `oindex` property.
        E.g.::

            >>> z.oindex[[1, 4], [1, 4]] = 4
            >>> z[...]
            array([[0, 2, 0, 0, 2],
                   [1, 4, 1, 1, 4],
                   [0, 2, 0, 0, 2],
                   [0, 2, 0, 0, 2],
                   [1, 4, 1, 1, 4]])

        Notes
        -----
        Orthogonal indexing is also known as outer indexing.

        Slices with step > 1 are supported, but slices with negative step are not.

        See Also
        --------
        get_basic_selection, set_basic_selection, get_mask_selection, set_mask_selection,
        get_coordinate_selection, set_coordinate_selection, get_orthogonal_selection,
        vindex, oindex, __getitem__, __setitem__

        """

        # guard conditions
        if self._read_only:
            err_read_only()

        # refresh metadata
        if not self._cache_metadata:
            self._load_metadata_nosync()

        # setup indexer
        indexer = OrthogonalIndexer(selection, self)

        self._set_selection(indexer, value, fields=fields)

    def set_coordinate_selection(self, selection, value, fields=None):
        """Modify a selection of individual items, by providing the indices (coordinates)
        for each item to be modified.

        Parameters
        ----------
        selection : tuple
            An integer (coordinate) array for each dimension of the array.
        value : scalar or array-like
            Value to be stored into the array.
        fields : str or sequence of str, optional
            For arrays with a structured dtype, one or more fields can be specified to set
            data for.

        Examples
        --------
        Setup a 2-dimensional array::

            >>> import zarr
            >>> import numpy as np
            >>> z = zarr.zeros((5, 5), dtype=int)

        Set data for a selection of items::

            >>> z.set_coordinate_selection(([1, 4], [1, 4]), 1)
            >>> z[...]
            array([[0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1]])

        For convenience, this functionality is also available via the `vindex` property.
        E.g.::

            >>> z.vindex[[1, 4], [1, 4]] = 2
            >>> z[...]
            array([[0, 0, 0, 0, 0],
                   [0, 2, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 2]])

        Notes
        -----
        Coordinate indexing is also known as point selection, and is a form of vectorized
        or inner indexing.

        Slices are not supported. Coordinate arrays must be provided for all dimensions
        of the array.

        See Also
        --------
        get_basic_selection, set_basic_selection, get_mask_selection, set_mask_selection,
        get_orthogonal_selection, set_orthogonal_selection, get_coordinate_selection,
        vindex, oindex, __getitem__, __setitem__

        """

        # guard conditions
        if self._read_only:
            err_read_only()

        # refresh metadata
        if not self._cache_metadata:
            self._load_metadata_nosync()

        # setup indexer
        indexer = CoordinateIndexer(selection, self)

        # handle value - need to flatten
        if not is_scalar(value, self._dtype):
            value = np.asanyarray(value)
        if hasattr(value, 'shape') and len(value.shape) > 1:
            value = value.reshape(-1)

        self._set_selection(indexer, value, fields=fields)

    def set_mask_selection(self, selection, value, fields=None):
        """Modify a selection of individual items, by providing a Boolean array of the
        same shape as the array against which the selection is being made, where True
        values indicate a selected item.

        Parameters
        ----------
        selection : ndarray, bool
            A Boolean array of the same shape as the array against which the selection is
            being made.
        value : scalar or array-like
            Value to be stored into the array.
        fields : str or sequence of str, optional
            For arrays with a structured dtype, one or more fields can be specified to set
            data for.

        Examples
        --------
        Setup a 2-dimensional array::

            >>> import zarr
            >>> import numpy as np
            >>> z = zarr.zeros((5, 5), dtype=int)

        Set data for a selection of items::

            >>> sel = np.zeros_like(z, dtype=bool)
            >>> sel[1, 1] = True
            >>> sel[4, 4] = True
            >>> z.set_mask_selection(sel, 1)
            >>> z[...]
            array([[0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1]])

        For convenience, this functionality is also available via the `vindex` property.
        E.g.::

            >>> z.vindex[sel] = 2
            >>> z[...]
            array([[0, 0, 0, 0, 0],
                   [0, 2, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 2]])

        Notes
        -----
        Mask indexing is a form of vectorized or inner indexing, and is equivalent to
        coordinate indexing. Internally the mask array is converted to coordinate
        arrays by calling `np.nonzero`.

        See Also
        --------
        get_basic_selection, set_basic_selection, get_mask_selection,
        get_orthogonal_selection, set_orthogonal_selection, get_coordinate_selection,
        set_coordinate_selection, vindex, oindex, __getitem__, __setitem__

        """

        # guard conditions
        if self._read_only:
            err_read_only()

        # refresh metadata
        if not self._cache_metadata:
            self._load_metadata_nosync()

        # setup indexer
        indexer = MaskIndexer(selection, self)

        self._set_selection(indexer, value, fields=fields)

    def _set_basic_selection_zd(self, selection, value, fields=None):
        # special case __setitem__ for zero-dimensional array

        # check selection is valid
        selection = ensure_tuple(selection)
        if selection not in ((), (Ellipsis,)):
            err_too_many_indices(selection, self._shape)

        # check fields
        check_fields(fields, self._dtype)
        fields = check_no_multi_fields(fields)

        # obtain key for chunk
        ckey = self._chunk_key((0,))

        # setup chunk
        try:
            # obtain compressed data for chunk
            cdata = self.chunk_store[ckey]

        except KeyError:
            # chunk not initialized
            chunk = np.zeros((), dtype=self._dtype)
            if self._fill_value is not None:
                chunk.fill(self._fill_value)

        else:
            # decode chunk
            chunk = self._decode_chunk(cdata).copy()

        # set value
        if fields:
            chunk[fields][selection] = value
        else:
            chunk[selection] = value

        # encode and store
        cdata = self._encode_chunk(chunk)
        self.chunk_store[ckey] = cdata

    def _set_basic_selection_nd(self, selection, value, fields=None):
        # implementation of __setitem__ for array with at least one dimension

        # setup indexer
        indexer = BasicIndexer(selection, self)

        self._set_selection(indexer, value, fields=fields)

    def _set_selection(self, indexer, value, fields=None):

        # We iterate over all chunks which overlap the selection and thus contain data
        # that needs to be replaced. Each chunk is processed in turn, extracting the
        # necessary data from the value array and storing into the chunk array.

        # N.B., it is an important optimisation that we only visit chunks which overlap
        # the selection. This minimises the nuimber of iterations in the main for loop.

        # check fields are sensible
        check_fields(fields, self._dtype)
        fields = check_no_multi_fields(fields)

        # determine indices of chunks overlapping the selection
        sel_shape = indexer.shape

        # check value shape
        if sel_shape == ():
            # setting a single item
            pass
        elif is_scalar(value, self._dtype):
            # setting a scalar value
            pass
        else:
            if not hasattr(value, 'shape'):
                value = np.asanyarray(value)
            check_array_shape('value', value, sel_shape)

        # iterate over chunks in range
        for chunk_coords, chunk_selection, out_selection in indexer:

            # extract data to store
            if sel_shape == ():
                chunk_value = value
            elif is_scalar(value, self._dtype):
                chunk_value = value
            else:
                chunk_value = value[out_selection]
                # handle missing singleton dimensions
                if indexer.drop_axes:
                    item = [slice(None)] * self.ndim
                    for a in indexer.drop_axes:
                        item[a] = np.newaxis
                    item = tuple(item)
                    chunk_value = chunk_value[item]

            # put data
            self._chunk_setitem(chunk_coords, chunk_selection, chunk_value, fields=fields)

    def _chunk_getitem(self, chunk_coords, chunk_selection, out, out_selection,
                       drop_axes=None, fields=None):
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
        drop_axes : tuple of ints
            Axes to squeeze out of the chunk.
        fields
            TODO

        """

        assert len(chunk_coords) == len(self._cdata_shape)

        # obtain key for chunk
        ckey = self._chunk_key(chunk_coords)

        try:
            # obtain compressed data for chunk
            cdata = self.chunk_store[ckey]

        except KeyError:
            # chunk not initialized
            if self._fill_value is not None:
                if fields:
                    fill_value = self._fill_value[fields]
                else:
                    fill_value = self._fill_value
                out[out_selection] = fill_value

        else:

            if (isinstance(out, np.ndarray) and
                    not fields and
                    is_contiguous_selection(out_selection) and
                    is_total_slice(chunk_selection, self._chunks) and
                    not self._filters and
                    self._dtype != object):

                dest = out[out_selection]
                write_direct = (
                    dest.flags.writeable and (
                        (self._order == 'C' and dest.flags.c_contiguous) or
                        (self._order == 'F' and dest.flags.f_contiguous)
                    )
                )

                if write_direct:

                    # optimization: we want the whole chunk, and the destination is
                    # contiguous, so we can decompress directly from the chunk
                    # into the destination array

                    if self._compressor:
                        self._compressor.decode(cdata, dest)
                    else:
                        chunk = ensure_ndarray(cdata).view(self._dtype)
                        chunk = chunk.reshape(self._chunks, order=self._order)
                        np.copyto(dest, chunk)
                    return

            # decode chunk
            chunk = self._decode_chunk(cdata)

            # select data from chunk
            if fields:
                chunk = chunk[fields]
            tmp = chunk[chunk_selection]
            if drop_axes:
                tmp = np.squeeze(tmp, axis=drop_axes)

            # store selected data in output
            out[out_selection] = tmp

    def _chunk_setitem(self, chunk_coords, chunk_selection, value, fields=None):
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

        if self._synchronizer is None:
            # no synchronization
            lock = nolock
        else:
            # synchronize on the chunk
            ckey = self._chunk_key(chunk_coords)
            lock = self._synchronizer[ckey]

        with lock:
            self._chunk_setitem_nosync(chunk_coords, chunk_selection, value,
                                       fields=fields)

    def _chunk_setitem_nosync(self, chunk_coords, chunk_selection, value, fields=None):

        # obtain key for chunk storage
        ckey = self._chunk_key(chunk_coords)

        if is_total_slice(chunk_selection, self._chunks) and not fields:
            # totally replace chunk

            # optimization: we are completely replacing the chunk, so no need
            # to access the existing chunk data

            if is_scalar(value, self._dtype):

                # setup array filled with value
                chunk = np.empty(self._chunks, dtype=self._dtype, order=self._order)
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
                cdata = self.chunk_store[ckey]

            except KeyError:

                # chunk not initialized
                if self._fill_value is not None:
                    chunk = np.empty(self._chunks, dtype=self._dtype, order=self._order)
                    chunk.fill(self._fill_value)
                elif self._dtype == object:
                    chunk = np.empty(self._chunks, dtype=self._dtype, order=self._order)
                else:
                    # N.B., use zeros here so any region beyond the array has consistent
                    # and compressible data
                    chunk = np.zeros(self._chunks, dtype=self._dtype, order=self._order)

            else:

                # decode chunk
                chunk = self._decode_chunk(cdata)
                if not chunk.flags.writeable:
                    chunk = chunk.copy(order='K')

            # modify
            if fields:
                # N.B., currently multi-field assignment is not supported in numpy, so
                # this only works for a single field
                chunk[fields][chunk_selection] = value
            else:
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
            for f in reversed(self._filters):
                chunk = f.decode(chunk)

        # view as numpy array with correct dtype
        chunk = ensure_ndarray(chunk)
        # special case object dtype, because incorrect handling can lead to
        # segfaults and other bad things happening
        if self._dtype != object:
            chunk = chunk.view(self._dtype)
        elif chunk.dtype != object:
            # If we end up here, someone must have hacked around with the filters.
            # We cannot deal with object arrays unless there is an object
            # codec in the filter chain, i.e., a filter that converts from object
            # array to something else during encoding, and converts back to object
            # array during decoding.
            raise RuntimeError('cannot read object array without object codec')

        # ensure correct chunk shape
        chunk = chunk.reshape(-1, order='A')
        chunk = chunk.reshape(self._chunks, order=self._order)

        return chunk

    def _encode_chunk(self, chunk):

        # apply filters
        if self._filters:
            for f in self._filters:
                chunk = f.encode(chunk)

        # check object encoding
        if isinstance(chunk, np.ndarray) and chunk.dtype == object:
            raise RuntimeError('cannot write object array without object codec')

        # compress
        if self._compressor:
            cdata = self._compressor.encode(chunk)
        else:
            cdata = chunk

        # ensure in-memory data is immutable and easy to compare
        if isinstance(self.chunk_store, dict):
            cdata = ensure_bytes(cdata)

        return cdata

    def __repr__(self):
        t = type(self)
        r = '<%s.%s' % (t.__module__, t.__name__)
        if self.name:
            r += ' %r' % self.name
        r += ' %s' % str(self.shape)
        r += ' %s' % self.dtype
        if self._read_only:
            r += ' read-only'
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

    def digest(self, hashname="sha1"):
        """
        Compute a checksum for the data. Default uses sha1 for speed.

        Examples
        --------
        >>> import binascii
        >>> import zarr
        >>> z = zarr.empty(shape=(10000, 10000), chunks=(1000, 1000))
        >>> binascii.hexlify(z.digest())
        b'041f90bc7a571452af4f850a8ca2c6cddfa8a1ac'
        >>> z = zarr.zeros(shape=(10000, 10000), chunks=(1000, 1000))
        >>> binascii.hexlify(z.digest())
        b'7162d416d26a68063b66ed1f30e0a866e4abed60'
        >>> z = zarr.zeros(shape=(10000, 10000), dtype="u1", chunks=(1000, 1000))
        >>> binascii.hexlify(z.digest())
        b'cb387af37410ae5a3222e893cf3373e4e4f22816'
        """

        h = hashlib.new(hashname)

        for i in itertools.product(*[range(s) for s in self.cdata_shape]):
            h.update(self.chunk_store.get(self._chunk_key(i), b""))

        h.update(self.store.get(self._key_prefix + array_meta_key, b""))

        h.update(self.store.get(self.attrs.key, b""))

        checksum = h.digest()

        return checksum

    def hexdigest(self, hashname="sha1"):
        """
        Compute a checksum for the data. Default uses sha1 for speed.

        Examples
        --------
        >>> import zarr
        >>> z = zarr.empty(shape=(10000, 10000), chunks=(1000, 1000))
        >>> z.hexdigest()
        '041f90bc7a571452af4f850a8ca2c6cddfa8a1ac'
        >>> z = zarr.zeros(shape=(10000, 10000), chunks=(1000, 1000))
        >>> z.hexdigest()
        '7162d416d26a68063b66ed1f30e0a866e4abed60'
        >>> z = zarr.zeros(shape=(10000, 10000), dtype="u1", chunks=(1000, 1000))
        >>> z.hexdigest()
        'cb387af37410ae5a3222e893cf3373e4e4f22816'
        """

        checksum = binascii.hexlify(self.digest(hashname=hashname))

        # This is a bytes object on Python 3 and we want a str.
        if type(checksum) is not str:  # pragma: py2 no cover
            checksum = checksum.decode('utf8')

        return checksum

    def __getstate__(self):
        return (self._store, self._path, self._read_only, self._chunk_store,
                self._synchronizer, self._cache_metadata, self._attrs.cache)

    def __setstate__(self, state):
        self.__init__(*state)

    def _synchronized_op(self, f, *args, **kwargs):

        if self._synchronizer is None:
            # no synchronization
            lock = nolock

        else:
            # synchronize on the array
            mkey = self._key_prefix + array_meta_key
            lock = self._synchronizer[mkey]

        with lock:
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
        >>> z.shape
        (20000, 2000)

        """
        return self._write_op(self._append_nosync, data, axis=axis)

    def _append_nosync(self, data, axis=0):

        # ensure data is array-like
        if not hasattr(data, 'shape'):
            data = np.asanyarray(data)

        # ensure shapes are compatible for non-append dimensions
        self_shape_preserved = tuple(s for i, s in enumerate(self._shape)
                                     if i != axis)
        data_shape_preserved = tuple(s for i, s in enumerate(data.shape)
                                     if i != axis)
        if self_shape_preserved != data_shape_preserved:
            raise ValueError('shape of data to append is not compatible with the array; '
                             'all dimensions must match except for the dimension being '
                             'appended')

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
            >>> labels = ['female', 'male']
            >>> data = np.random.choice(labels, size=10000)
            >>> filters = [zarr.Categorize(labels=labels,
            ...                            dtype=data.dtype,
            ...                            astype='u1')]
            >>> a = zarr.array(data, chunks=1000, filters=filters)
            >>> a[:]
            array(['female', 'male', 'female', ..., 'male', 'male', 'female'],
                  dtype='<U6')
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
            array(['female', 'female', 'female', ..., 'male', 'male', 'male'],
                  dtype='<U6')

        View as a different dtype with the same item size:

            >>> data = np.random.randint(0, 2, size=10000, dtype='u1')
            >>> a = zarr.array(data, chunks=1000)
            >>> a[:]
            array([0, 0, 1, ..., 1, 0, 0], dtype=uint8)
            >>> v = a.view(dtype=bool)
            >>> v[:]
            array([False, False,  True, ...,  True, False, False])
            >>> np.all(a[:].view(dtype=bool) == v[:])
            True

        An array can be viewed with a dtype with a different item size, however
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
            operation not permitted for views

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
        """Returns a view that does on the fly type conversion of the underlying data.

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
