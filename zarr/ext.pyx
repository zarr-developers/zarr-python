# -*- coding: utf-8 -*-
# cython: embedsignature=True
# cython: profile=True
# cython: linetrace=True
# cython: binding=True
from __future__ import absolute_import, print_function, division
from threading import RLock
import itertools
# TODO PY2 compatibility
from functools import reduce
import operator
import sys
import os
import struct
import ctypes
import pickle
import shutil
import logging
import tempfile
from collections import namedtuple


from zarr import util as _util
from zarr import defaults


###############################################################################
# CYTHON IMPORTS                                                              #
###############################################################################


import numpy as np
cimport numpy as np
from numpy cimport ndarray, dtype
from libc.stdint cimport uintptr_t
from .definitions cimport (malloc, realloc, free, PyBytes_AsString,
    PyBytes_FromStringAndSize)


###############################################################################
# BLOSC IMPORTS                                                               #
###############################################################################


cdef extern from "blosc.h":
    cdef enum:
        BLOSC_MAX_OVERHEAD,
        BLOSC_VERSION_STRING,
        BLOSC_VERSION_DATE

    int blosc_compname_to_compcode(const char *compname)
    int blosc_compress_ctx(int clevel, int doshuffle, size_t typesize,
                           size_t nbytes, const void* src, void* dest,
                           size_t destsize, const char* compressor,
				           size_t blocksize, int numinternalthreads) nogil
    int blosc_decompress_ctx(const void *src, void *dest, size_t destsize,
                             int numinternalthreads) nogil
    void blosc_cbuffer_sizes(void *cbuffer, size_t *nbytes,
                             size_t *cbytes, size_t *blocksize)


def blosc_version():
    """Return the version of c-blosc that zarr was compiled with."""

    # all the 'decode' contorsions are for Python 3 returning actual strings
    ver_str = <char *> BLOSC_VERSION_STRING
    if hasattr(ver_str, "decode"):
        ver_str = ver_str.decode()
    ver_date = <char *> BLOSC_VERSION_DATE
    if hasattr(ver_date, "decode"):
        ver_date = ver_date.decode()
    return ver_str, ver_date


###############################################################################
# SETUP LOGGING                                                               #
###############################################################################


logger = logging.getLogger(__name__)


def debug(*args):
    msg = str(args[0]) + ': ' + ', '.join(map(repr, args[1:]))
    logger.debug(msg)


###############################################################################
# MISC HELPERS                                                                #
###############################################################################


def normalise_cparams(cname=None, clevel=None, shuffle=None):
    """Convenience function to normalise compression parameters.

    If any values are None, they will be substituted with values from the
    `zarr.defaults` module.

    Parameters
    ----------
    cname : string, optional
        Name of compression library to use, e.g., 'blosclz', 'lz4', 'zlib',
        'snappy'.
    clevel : int, optional
        Compression level, 0 means no compression.
    shuffle : int, optional
        Shuffle filter, 0 means no shuffle, 1 means byte shuffle, 2 means
        bit shuffle.

    Returns
    -------
    cname
    clevel
    shuffle

    """

    # determine compressor
    cname = cname if cname is not None else defaults.cname
    if type(cname) != bytes:
        cname = cname.encode()
    # check compressor is available
    if blosc_compname_to_compcode(cname) < 0:
        raise ValueError("compressor not available: %s" % cname)

    # determine compression level
    clevel = clevel if clevel is not None else defaults.clevel
    clevel = int(clevel)
    if clevel < 0 or clevel > 9:
        raise ValueError('invalid compression level: %s' % clevel)

    # determine shuffle filter
    shuffle = shuffle if shuffle is not None else defaults.shuffle
    shuffle = int(shuffle)
    if shuffle not in [0, 1, 2]:
        raise ValueError('invalid shuffle: %s' % shuffle)

    return cname, clevel, shuffle


def normalise_shape(shape):
    """Convenience function to normalise the `shape` argument."""
    try:
        shape = tuple(int(s) for s in shape)
    except TypeError:
        shape = (int(shape),)
    return shape


cdef is_total_slice(item, tuple shape):
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


# for the persistence layer
BLOSC_HEADER_LENGTH = 16


if sys.version_info >= (3, 0):
    def decode_byte(byte):
        return byte
else:
    def decode_byte(byte):
        return int(byte.encode('hex'), 16)


def decode_uint32(bytes fourbyte):
    return struct.unpack('<I', fourbyte)[0]


BloscHeader = namedtuple('BloscHeader', ['version', 'versionlz', 'flags',
                                         'typesize', 'nbytes', 'blocksize',
                                         'cbytes'])


def decode_blosc_header(bytes b):
    """Read and decode header from compressed Blosc buffer.

    Parameters
    ----------
    buffer_ : string of bytes
        the compressed buffer

    Returns
    -------
    settings : dict
        a dict containing the settings from Blosc

    Notes
    -----

    The Blosc 1.1.3 header is 16 bytes as follows:

    |-0-|-1-|-2-|-3-|-4-|-5-|-6-|-7-|-8-|-9-|-A-|-B-|-C-|-D-|-E-|-F-|
      ^   ^   ^   ^ |     nbytes    |   blocksize   |    cbytes     |
      |   |   |   |
      |   |   |   +--typesize
      |   |   +------flags
      |   +----------versionlz
      +--------------version

    The first four are simply bytes, the last three are are each unsigned ints
    (uint32) each occupying 4 bytes. The header is always little-endian.
    'cbytes' is the length of the buffer including header and nbytes is the
    length of the data when uncompressed.

    """
    debug('decode_blosc_header', len(b))
    header = BloscHeader(version=decode_byte(b[0]),
                         versionlz=decode_byte(b[1]),
                         flags=decode_byte(b[2]),
                         typesize=decode_byte(b[3]),
                         nbytes=decode_uint32(b[4:8]),
                         blocksize=decode_uint32(b[8:12]),
                         cbytes=decode_uint32(b[12:16]))
    return header


###############################################################################
# CHUNK CLASSES                                                               #
###############################################################################


# noinspection PyAttributeOutsideInit
cdef class BaseChunk:

    def __cinit__(self, shape=None, dtype=None, cname=None,
                  clevel=None, shuffle=None, fill_value=0, **kwargs):

        # set shape and dtype
        self._shape = normalise_shape(shape)
        self._dtype = np.dtype(dtype)

        # set derived attributes
        self._size = reduce(operator.mul, self._shape)
        self._itemsize = self._dtype.itemsize
        self._nbytes = self._size * self._itemsize

        # set compression options
        self._cname, self._clevel, self._shuffle = \
            normalise_cparams(cname, clevel, shuffle)

        # set fill_value
        self.fill_value = fill_value

    property shape:
        def __get__(self):
            return self._shape
    
    property dtype:
        def __get__(self):
            return self._dtype

    property size:
        def __get__(self):
            return self._size

    property itemsize:
        def __get__(self):
            return self._itemsize

    property nbytes:
        def __get__(self):
            return self._nbytes

    property cname:
        def __get__(self):
            return self._cname

    property clevel:
        def __get__(self):
            return self._clevel

    property shuffle:
        def __get__(self):
            return self._shuffle

    def __getitem__(self, item):
        cdef ndarray array

        # setup output array
        array = np.empty(self._shape, dtype=self._dtype)

        if self.is_initialised:

            # data initialised, decompress into array
            self.get(array.data)

        else:

            # data not initialised, use fill_value
            if self.fill_value is not None:
                array.fill(self.fill_value)

        return array[item]

    def __array__(self):
        return self[:]

    def __setitem__(self, key, value):
        cdef ndarray array
        
        if is_total_slice(key, self._shape):
            # completely replace the contents of this chunk
    
            if np.isscalar(value):

                # setup array filled with value
                array = np.empty(self._shape, dtype=self._dtype)
                array.fill(value)
    
            else:

                # ensure array is C contiguous
                array = np.ascontiguousarray(value, dtype=self._dtype)
                if array.shape != self._shape:
                    raise ValueError('bad value shape')
    
        else:
            # partially replace the contents of this chunk
    
            # decompress existing data
            array = self[:]
    
            # modify
            array[key] = value
    
        # compress the data and store
        self.put(array.data)

    # abstract properties and methods follow
    ########################################

    property is_initialised:
        def __get__(self):
            # override in sub-class
            raise NotImplementedError()

    property cbytes:
        def __get__(self):
            # override in sub-class
            raise NotImplementedError()

    cdef get(self, char *dest):
        # override in sub-class
        raise NotImplementedError()

    cdef put(self, char *source):
        # override in sub-class
        raise NotImplementedError()


# noinspection PyAttributeOutsideInit
cdef class Chunk(BaseChunk):

    def __cinit__(self, shape=None, dtype=None, cname=None, clevel=None,
                  shuffle=None, fill_value=None, **kwargs):

        # initialise attributes
        self._data = NULL
        self._nbytes = 0
        self._cbytes = 0

    property is_initialised:
        def __get__(self):
            return self._data != NULL

    property cbytes:
        def __get__(self):
            return self._cbytes

    cdef get(self, char *dest):
        cdef int ret

        # do decompression
        with nogil:
            ret = blosc_decompress_ctx(self._data, dest, self._nbytes, 1)

        # handle errors
        if ret <= 0:
            raise RuntimeError('error during blosc compression: %d' % ret)

    cdef put(self, char *source):
        cdef:
            size_t nbytes_check, cbytes, blocksize
            char *dest

        # ensure any existing data is cleared and memory freed
        self.clear()

        # allocate memory for compressed data
        dest = <char *> malloc(self._nbytes + BLOSC_MAX_OVERHEAD)

        # perform compression
        with nogil:
            cbytes = blosc_compress_ctx(self._clevel, self._shuffle,
                                        self._itemsize, self._nbytes, 
                                        source, dest, 
                                        self._nbytes + BLOSC_MAX_OVERHEAD,
                                        self._cname, 0, 1)

        # check compression was successful
        if cbytes <= 0:
            raise RuntimeError('error during blosc compression: %d' % cbytes)

        # free the unused memory
        self._data = <char *> realloc(dest, cbytes)

        # get information about the compressed data
        blosc_cbuffer_sizes(self._data, &nbytes_check, &cbytes, &blocksize)
        assert nbytes_check == self._nbytes
        self._cbytes = cbytes
        self._blocksize = blocksize

    cdef free(self):
        if self._data != NULL:
            free(self._data)

    cdef clear(self):
        self.free()
        self._data = NULL
        self._nbytes = 0
        self._cbytes = 0

    def __dealloc__(self):
        self.free()


cdef class SynchronizedChunk(Chunk):

    def __cinit__(self, **kargs):
        self._lock = RLock()

    def __setitem__(self, key, value):
        with self._lock:
            super(SynchronizedChunk, self).__setitem__(key, value)


cdef class PersistentChunk(BaseChunk):

    def __cinit__(self, path=None, **kwargs):

        # set file path
        self._path = path
        self._basename = os.path.basename(path)
        self._dirname = os.path.dirname(path)

    property is_initialised:
        def __get__(self):
            return os.path.exists(self._path)
        
    cdef dict read_header(self):
        with open(self._path, 'rb') as f:
            header_raw = f.read(BLOSC_HEADER_LENGTH)
            header = decode_blosc_header(header_raw)
            return header

    property cbytes:
        def __get__(self):
            if not self.is_initialised:
                return 0
            else:
                header = self.read_header()
                return header.cbytes

    cdef bytes read(self):
        with open(self._path, 'rb') as f:
            header_raw = f.read(BLOSC_HEADER_LENGTH)
            header = decode_blosc_header(header_raw)
            # check nbytes consistency here
            if self._nbytes != header.nbytes:
                raise RuntimeError('expected nbytes %s, found %s' %
                                   (self._nbytes, header.nbytes))
            # seek back BLOSC_HEADER_LENGTH bytes relative to current position
            f.seek(-BLOSC_HEADER_LENGTH, 1)
            data = f.read(header.cbytes)
            return data

    cdef get(self, char *dest):
        cdef:
            int ret
            bytes data
            char *source

        # read compressed data from file
        data = self.read()
        source = PyBytes_AsString(data)

        # do decompression
        with nogil:
            ret = blosc_decompress_ctx(source, dest, self._nbytes, 1)

        # handle errors
        if ret <= 0:
            raise RuntimeError('error during blosc compression: %d' % ret)
    
    cdef write(self, bytes data):
        # N.B., write to a temporary file then move into place to avoid data
        # corruption due to errors during write leaving partially written files
        tmp = tempfile.mktemp(suffix='.partial', prefix=self._basename + '.',
                              dir=self._dirname)
        with open(tmp, 'wb') as f:
            f.write(data)
        os.replace(tmp, self._path)

    cdef put(self, char *source):
        cdef:
            size_t nbytes_check, cbytes, blocksize
            char *dest
            char *data

        # allocate memory for compressed data
        dest = <char *> malloc(self._nbytes + BLOSC_MAX_OVERHEAD)

        # perform compression
        with nogil:
            cbytes = blosc_compress_ctx(self._clevel, self._shuffle, 
                                        self._itemsize, self._nbytes, 
                                        source, dest,
                                        self._nbytes + BLOSC_MAX_OVERHEAD,
                                        self._cname, 0, 1)

        # check compression was successful
        if cbytes <= 0:
            raise RuntimeError('error during blosc compression: %d' % cbytes)

        # free the unused memory
        data = <char *> realloc(dest, cbytes)

        # get information about the compressed data
        blosc_cbuffer_sizes(data, &nbytes_check, &cbytes, &blocksize)
        assert nbytes_check == self._nbytes

        # wrap as python bytes
        data_bytes = ctypes.string_at(<uintptr_t> data, cbytes)
        # # original implementation from bcolz, implies a copy?
        # data_bytes = PyBytes_FromStringAndSize(data, <Py_ssize_t> cbytes)

        # write bytes to file
        self.write(data_bytes)

        # free memory
        free(data)    


cdef class SynchronizedPersistentChunk(PersistentChunk):

    def __cinit__(self, **kargs):
        # TODO
        pass

    def __setitem__(self, key, value):
        # TODO
        pass


###############################################################################
# ARRAY HELPERS                                                               #
###############################################################################


def normalise_array_selection(item, shape):
    """Convenience function to normalise a selection within an array with
    the given `shape`."""

    # normalise item
    if isinstance(item, int):
        item = (item,)
    elif isinstance(item, slice):
        item = (item,)
    elif item == Ellipsis:
        item = (slice(None),)

    # handle tuple of indices/slices
    if isinstance(item, tuple):

        # determine start and stop indices for all axes
        selection = tuple(normalise_axis_selection(i, l)
                          for i, l in zip(item, shape))

        # fill out selection if not completely specified
        if len(selection) < len(shape):
            selection += tuple((0, l) for l in shape[len(selection):])

        return selection

    else:
        raise ValueError('expected indices or slice, found: %r' % item)


def normalise_axis_selection(item, l):
    """Convenience function to normalise a selection within a single axis
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


def get_chunk_range(tuple selection, tuple chunks):
    """Convenience function to get a range over all chunk indices,
    for iterating over chunks."""
    chunk_range = [range(start//l, int(np.ceil(stop/l)))
                   for (start, stop), l in zip(selection, chunks)]
    return chunk_range


def normalise_chunks(chunks, tuple shape):
    """Convenience function to normalise the `chunks` argument for an array
    with the given `shape`."""
    try:
        chunks = tuple(int(c) for c in chunks)
    except TypeError:
        chunks = (int(chunks),)
    if len(chunks) < len(shape):
        # assume chunks across remaining dimensions
        chunks += shape[len(chunks):]
    if len(chunks) != len(shape):
        raise ValueError('chunks and shape not compatible: %r, %r' %
                         (chunks, shape))
    # handle None in chunks
    chunks = tuple(s if c is None else c for s, c in zip(shape, chunks))
    return chunks


###############################################################################
# ARRAY CLASSES                                                               #
###############################################################################


cdef class BaseArray:

    def __cinit__(self, shape=None, chunks=None, dtype=None, cname=None,
                  clevel=None, shuffle=None, fill_value=None, **kwargs):

        # N.B., the convention in h5py and dask is to use the "chunks"
        # argument as a tuple representing the shape of each chunk,
        # so we follow that convention here.

        # set shape
        self._shape = normalise_shape(shape)

        # set chunks
        self._chunks = normalise_chunks(chunks, self._shape)

        # set dtype
        self._dtype = np.dtype(dtype)

        # set derived attributes
        self._size = reduce(operator.mul, self._shape)
        self._itemsize = self._dtype.itemsize
        self._nbytes = self._size * self._itemsize

        # set compression options
        self._cname, self._clevel, self._shuffle = \
            normalise_cparams(cname, clevel, shuffle)

        # set fill_value
        self._fill_value = fill_value

    property shape:
        def __get__(self):
            return self._shape

    property chunks:
        def __get__(self):
            return self._chunks

    property dtype:
        def __get__(self):
            return self._dtype

    property size:
        def __get__(self):
            return self._size

    property itemsize:
        def __get__(self):
            return self._itemsize

    property nbytes:
        def __get__(self):
            return self._nbytes

    property cbytes:
        def __get__(self):
            # override in sub-class
            pass

    property is_initialised:
        def __get__(self):
            # override in sub-class
            pass

    cdef void init_chunks(self):
        # override in sub-class
        pass

    cdef BaseChunk create_chunk(self, tuple cidx):
        # override in sub-class
        pass

    cdef BaseChunk get_chunk(self, tuple cidx):
        # override in sub-class
        pass

    def __getitem__(self, item):
        cdef ndarray dest
        cdef BaseChunk chunk

        # normalise selection
        selection = normalise_array_selection(item, self._shape)

        # determine output array shape
        out_shape = tuple(stop - start for start, stop in selection)

        # setup output array
        out = np.empty(out_shape, dtype=self._dtype)

        # determine indices of overlapping chunks
        chunk_range = get_chunk_range(selection, self._chunks)

        # iterate over chunks in range
        for cidx in itertools.product(*chunk_range):

            # get current chunk
            chunk = self.get_chunk(cidx)

            # determine chunk offset
            offset = [i * c for i, c in zip(cidx, self._chunks)]

            # determine index range within output array
            out_selection = tuple(
                slice(max(0, o - start), min(o + c - start, stop - start))
                for (start, stop), o, c, in zip(selection, offset, self._chunks)
            )

            # determine required index range within chunk
            chunk_selection = tuple(
                slice(max(0, start - o), min(c, stop - o))
                for (start, stop), o, c in zip(selection, offset, self._chunks)
            )

            # obtain the destination array as a view of the output array
            dest = out[out_selection]

            if chunk.is_initialised and \
                    is_total_slice(chunk_selection, chunk.shape) and \
                    dest.flags.c_contiguous:

                # optimisation, destination is C contiguous so we can decompress
                # directly from the chunk into the output array
                chunk.get(dest.data)

            else:

                # set data in output array
                tmp = chunk[chunk_selection]
                dest[:] = tmp

        return out

    def __array__(self):
        return self[:]

    def __setitem__(self, key, value):

        # normalise selection
        selection = normalise_array_selection(key, self._shape)

        # determine indices of overlapping chunks
        chunk_range = get_chunk_range(selection, self._chunks)

        # iterate over chunks in range
        for cidx in itertools.product(*chunk_range):

            # current chunk
            chunk = self.get_chunk(cidx)

            # determine chunk offset
            offset = [i * c for i, c in zip(cidx, self._chunks)]

            # determine required index range within chunk
            chunk_selection = tuple(
                slice(max(0, start - o), min(c, stop - o))
                for (start, stop), o, c in zip(selection, offset, self._chunks)
            )

            if np.isscalar(value):

                # fill chunk with scalar value
                chunk[chunk_selection] = value

            else:
                # assume value is array-like

                # determine index within value
                value_selection = tuple(
                    slice(max(0, o - start), min(o + c - start, stop - start))
                    for (start, stop), o, c, in zip(selection, offset, self._chunks)
                )

                # set data in chunk
                chunk[chunk_selection] = value[value_selection]

    def __repr__(self):
        r = '%s.%s(' % (type(self).__module__, type(self).__name__)
        r += '%s' % str(self._shape)
        r += ', %s' % str(self._dtype)
        r += ', chunks=%s' % str(self._chunks)
        r += ', cname=%r' % str(self._cname, 'ascii')
        r += ', clevel=%s' % self._clevel
        r += ', shuffle=%s' % self._shuffle
        r += ')'
        r += '\n  nbytes: %s' % _util.human_readable_size(self.nbytes)
        r += '; cbytes: %s' % _util.human_readable_size(self.cbytes)
        if self.cbytes > 0:
            r += '; ratio: %.1f' % (self.nbytes / self.cbytes)
        return r

    def resize(self, *args):
        # override in sub-class
        pass


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
        append_selection = tuple(
            slice(None) if i != axis else slice(old_shape[i], new_shape[i])
            for i in range(len(self._shape))
        )
        self[append_selection] = data


# noinspection PyAttributeOutsideInit
cdef class Array(BaseArray):
    # TODO review me

    def __cinit__(self, **kwargs):

        # TODO initialise chunks

        # determine the number and arrangement of chunks
        cdata_shape = tuple(int(np.ceil(s / c))
                            for s, c in zip(self._shape, self._chunks))

        # initialise an object array to hold pointers to chunk objects
        self.cdata = np.empty(cdata_shape, dtype=object)

        # determine function for instantiating chunks
        if self.synchronized:
            create_chunk = SynchronizedChunk
        else:
            create_chunk = Chunk

        # instantiate chunks
        self.cdata.flat = [create_chunk(self._chunks, dtype=self._dtype,
                                        cname=self._cname,
                                        clevel=self._clevel,
                                        shuffle=self._shuffle,
                                        fill_value=self.fill_value)
                           for _ in self.cdata.flat]

        # N.B., in the current implementation, some chunks may overhang
        # the edge of the array. This is handled during the __getitem__ and
        # __setitem__ methods by setting appropriate slices on the chunks,
        # however it may be wasteful depending on chunk shape and the
        # relationship to array shape.

    property cbytes:
        def __get__(self):
            return sum(c.cbytes for c in self.cdata.flat)

    property is_initialised:
        def __get__(self):
            a = np.empty_like(self.cdata, dtype='b1')
            a.flat = [c.is_initialised for c in self.cdata.flat]
            return a

    def resize(self, *args):
        """Resize the array by growing or shrinking one or more dimensions.

        Parameters
        ----------
        args : int or sequence of ints
            New shape to resize to.

        Notes
        -----
        This function can only be used to change the size of existing
        dimensions, it cannot add or drop dimensions.

        N.B., this function does *not* behave in the same way as the numpy
        resize method on the ndarray class. Existing data are *not*
        reorganised in any way. Axes are simply grown or shrunk. When
        growing an axis, uninitialised portions of the array will appear to
        contain the value of the `fill_value` attribute on this array,
        and when shrinking an array any data beyond the new shape will be
        lost (although see note below).

        N.B., because of the way the underlying chunks are organised,
        and in particular the fact that chunks may overhang the edge of the
        array, the value of uninitialised portions of this array is not
        guaranteed to respect the setting of the `fill_value` attribute when
        shrinking then regrowing an array.

        """

        # normalise new shape argument
        if len(args) == 1:
            new_shape = args[0]
        else:
            new_shape = args
        if isinstance(new_shape, int):
            new_shape = (new_shape,)
        else:
            new_shape = tuple(new_shape)
        if len(new_shape) != len(self._shape):
            raise ValueError('new shape must have same number of dimensions')

        # handle None in new_shape
        new_shape = tuple(s if n is None else n
                          for s, n in zip(self._shape, new_shape))

        # work-around Cython problems with accessing .shape attribute as tuple
        old_cdata = np.asarray(self.cdata)

        # remember old cdata shape
        old_cdata_shape = old_cdata.shape

        # determine the new number and arrangement of chunks
        new_cdata_shape = tuple(int(np.ceil(s / c))
                                for s, c in zip(new_shape, self._chunks))

        # setup new chunks array
        new_cdata = np.empty(new_cdata_shape, dtype=object)

        # copy across any chunks to be kept
        cdata_overlap = tuple(
            slice(min(o, n)) for o, n in zip(old_cdata_shape, new_cdata_shape)
        )
        new_cdata[cdata_overlap] = old_cdata[cdata_overlap]

        # determine function for instantiating chunks
        if self.synchronized:
            create_chunk = SynchronizedChunk
        else:
            create_chunk = Chunk

        # instantiate any new chunks as needed
        new_cdata.flat = [create_chunk(self._chunks, dtype=self._dtype,
                                       cname=self._cname,
                                       clevel=self._clevel,
                                       shuffle=self._shuffle,
                                       fill_value=self.fill_value)
                          if c is None else c
                          for c in new_cdata.flat]

        # set new shape
        self._shape = new_shape

        # set new chunks
        self.cdata = new_cdata


def read_array_metadata(path):

    # check path exists
    if not os.path.exists(path):
        raise ValueError('path not found: %s' % path)

    # check metadata file
    meta_path = os.path.join(path, 'meta')
    if not os.path.exists(meta_path):
        raise ValueError('path does not contain an array: %s' % path)

    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
        return meta


def write_array_metadata(path, meta):
    meta_path = os.path.join(path, 'meta')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f, protocol=0)


# TODO remove code duplication with Array class


def persistent_array_getitem(PersistentArray self, item):
    """TODO refactor"""

    cdef ndarray dest
    cdef PersistentChunk chunk

    # normalise selection
    selection = normalise_array_selection(item, self._shape)

    # determine output array shape
    out_shape = tuple(stop - start for start, stop in selection)

    # setup output array
    out = np.empty(out_shape, dtype=self._dtype)

    # determine indices of overlapping chunks
    chunk_range = get_chunk_range(selection, self._chunks)

    # iterate over chunks in range
    for cidx in itertools.product(*chunk_range):

        # access current chunk
        chunk = self.cdata[cidx]

        # determine chunk offset
        offset = [i * c for i, c in zip(cidx, self._chunks)]

        # determine index range within output array
        out_selection = tuple(
            slice(max(0, o - start), min(o + c - start, stop - start))
            for (start, stop), o, c, in zip(selection, offset, self._chunks)
        )

        # determine required index range within chunk
        chunk_selection = tuple(
            slice(max(0, start - o), min(c, stop - o))
            for (start, stop), o, c in zip(selection, offset, self._chunks)
        )

        # obtain the destination array as a view of the output array
        dest = out[out_selection]

        if chunk.is_initialised and \
                is_total_slice(chunk_selection, chunk.shape) and \
                dest.flags.c_contiguous:

            # optimisation, destination is C contiguous so we can decompress
            # directly from the chunk into the output array
            chunk.get(dest.data)

        else:

            # set data in output array
            tmp = chunk[chunk_selection]
            dest[:] = tmp

    return out


cdef class PersistentArray:

    def __cinit__(self, path, mode='r', shape=None, chunks=None, dtype=None,
                  cname=None, clevel=None, shuffle=None, fill_value=None):

        # TODO synchronization

        if mode in 'ra':
            metadata = read_array_metadata(path)
            self._shape = metadata['shape']
            self._dtype = metadata['dtype']
            self._chunks = metadata['chunks']
            self._cname = metadata['cname']
            self._clevel = metadata['clevel']
            self._shuffle = metadata['shuffle']
            self.fill_value = metadata['fill_value']

        elif mode == 'w':
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(os.path.join(path, 'data'))
            self._shape = normalise_shape(shape)
            self._chunks = normalise_chunks(chunks, self._shape)
            self._dtype = np.dtype(dtype)
            self._cname, self._clevel, self._shuffle = \
                normalise_cparams(cname, clevel, shuffle)
            self.fill_value = fill_value
            metadata = {'shape': self._shape,
                        'chunks': self._chunks,
                        'dtype': self._dtype,
                        'cname': self._cname,
                        'clevel': self._clevel,
                        'shuffle': self._shuffle,
                        'fill_value': self.fill_value}
            write_array_metadata(path, metadata)

        else:
            raise ValueError('unexpected mode: %r' % mode)

        # set mode
        self.mode = mode

        # determine the number and arrangement of chunks
        cdata_shape = tuple(int(np.ceil(s / c))
                            for s, c in zip(self._shape, self._chunks))

        # initialise an object array to hold pointers to chunk objects
        self.cdata = np.empty(cdata_shape, dtype=object)

        # instantiate chunks
        for cidx in itertools.product(*(range(n) for n in cdata_shape)):
            chunk_fn = '.'.join(map(str, cidx)) + '.blosc'
            chunk_path = os.path.join(path, 'data', chunk_fn)
            self.cdata[cidx] = PersistentChunk(
                chunk_path, self._chunks, dtype=self._dtype, cname=self._cname,
                clevel=self._clevel, shuffle=self._shuffle,
                fill_value=self.fill_value
            )

    property nbytes:
        def __get__(self):
            return self._dtype.itemsize * reduce(operator.mul, self._shape)

    property cbytes:
        def __get__(self):
            return sum(c.cbytes for c in self.cdata.flat)

    property is_initialised:
        def __get__(self):
            a = np.empty_like(self.cdata, dtype='b1')
            a.flat = [c.is_initialised for c in self.cdata.flat]
            return a

    def __getitem__(self, item):
        return persistent_array_getitem(self, item)

    def __array__(self):
        return self[:]

    def __setitem__(self, key, value):
        if self.mode == 'r':
            raise ValueError('array is read-only')
        array_setitem(self, key, value)

    def __repr__(self):
        # TODO more line breaks?
        r = '%s.%s(' % (type(self).__module__, type(self).__name__)
        r += '%s' % str(self._shape)
        r += ', %s' % str(self._dtype)
        r += ', path=%s' % self.path
        r += ', mode=%r' % self.mode
        r += ', chunks=%s' % str(self._chunks)
        r += ', cname=%r' % str(self._cname, 'ascii')
        r += ', clevel=%s' % self._clevel
        r += ', shuffle=%s' % self._shuffle
        r += ')'
        r += '\n  nbytes: %s' % _util.human_readable_size(self.nbytes)
        r += '; cbytes: %s' % _util.human_readable_size(self.cbytes)
        if self.cbytes > 0:
            r += '; ratio: %.1f' % (self.nbytes / self.cbytes)
        return r

    # TODO resize
    # TODO append
