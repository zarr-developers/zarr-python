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


import numpy as np
cimport numpy as np
from numpy cimport ndarray, dtype


from zarr import util as _util


# import logging
# logger = logging.getLogger(__name__)
#
#
# def debug(*args):
#     msg = str(args[0]) + ': ' + ', '.join(map(repr, args[1:]))
#     logger.debug(msg)


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


from .definitions cimport malloc, realloc, free


from zarr import defaults


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


def get_cparams(cname=None, clevel=None, shuffle=None):
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


def chunk_setitem(Chunk self, key, value):
    """Chunk.__setitem__ broken out as separate function to enable line
    profiling."""

    if is_total_slice(key, self.shape):
        # completely replace the contents of this chunk

        if np.isscalar(value):
            array = np.empty(self.shape, dtype=self.dtype)
            array.fill(value)

        else:
            # ensure array is C contiguous
            array = np.ascontiguousarray(value, dtype=self.dtype)
            if array.shape != self.shape:
                raise ValueError('bad value shape')

    else:
        # partially replace the contents of this chunk

        # decompress existing data
        array = self[:]

        # modify
        array[key] = value

    # compress the data and store
    self.compress(array)


def chunk_getitem(Chunk self, item):
    """Chunk.__getitem__ broken out as separate function to enable line
    profiling."""

    cdef:
        ndarray array

    # setup output array
    array = np.empty(self.shape, dtype=self.dtype)

    if self.data == NULL:
        # data not initialised, use fill_value
        if self.fill_value is not None:
            array.fill(self.fill_value)

    else:
        # data initialised, decompress into array
        self.decompress(array.data)

    return array[item]


# noinspection PyAttributeOutsideInit
cdef class Chunk:

    def __cinit__(self, shape, dtype=None, cname=None, clevel=None,
                  shuffle=None, fill_value=None, **kwargs):

        # set shape and dtype
        self.shape = normalise_shape(shape)
        self.dtype = np.dtype(dtype)

        # set compression options
        self.cname, self.clevel, self.shuffle = \
            get_cparams(cname, clevel, shuffle)

        # set fill_value
        self.fill_value = fill_value

        # initialise other attributes
        self.data = NULL
        self.nbytes = 0
        self.cbytes = 0
        self.blocksize = 0

    property is_initialised:
        def __get__(self):
            return self.data != NULL

    def __setitem__(self, key, value):
        chunk_setitem(self, key, value)

    cdef compress(self, ndarray array):
        cdef:
            size_t nbytes, nbytes_check, cbytes, blocksize, itemsize
            char *dest

        # ensure any existing data is cleared and memory freed
        self.clear()

        # compute the total number of bytes in the array
        nbytes = array.itemsize * array.size

        # determine itemsize
        itemsize = array.dtype.base.itemsize

        # allocate memory for compressed data
        dest = <char *> malloc(nbytes + BLOSC_MAX_OVERHEAD)

        # perform compression
        with nogil:
            cbytes = blosc_compress_ctx(self.clevel, self.shuffle, itemsize,
                                        nbytes, array.data, dest,
                                        nbytes + BLOSC_MAX_OVERHEAD,
                                        self.cname, 0, 1)

        # check compression was successful
        if cbytes <= 0:
            raise RuntimeError("error during blosc compression: %d" % cbytes)

        # free the unused memory
        self.data = <char *> realloc(dest, cbytes)

        # get information about the compressed data
        blosc_cbuffer_sizes(dest, &nbytes_check, &cbytes, &blocksize)
        assert nbytes_check == nbytes

        # store compression information
        self.nbytes = nbytes
        self.cbytes = cbytes
        self.blocksize = blocksize

    def __getitem__(self, item):
        return chunk_getitem(self, item)

    def __array__(self):
        return self[:]

    cdef decompress(self, char *dest):
        cdef:
            int ret

        # do decompression
        with nogil:
            ret = blosc_decompress_ctx(self.data, dest, self.nbytes, 1)

        # handle errors
        if ret <= 0:
            raise RuntimeError("error during blosc compression: %d" % ret)

    cdef free(self):
        if self.data != NULL:
            free(self.data)

    cdef clear(self):
        self.free()
        self.data = NULL
        self.nbytes = 0
        self.cbytes = 0
        self.blocksize = 0

    def __dealloc__(self):
        self.free()


cdef class SynchronizedChunk(Chunk):

    def __cinit__(self, *args, **kargs):
        self.lock = RLock()

    def __setitem__(self, key, value):
        with self.lock:
            super(SynchronizedChunk, self).__setitem__(key, value)


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


def normalise_shape(shape):
    """Convenience function to normalise the `shape` argument."""
    try:
        shape = tuple(int(s) for s in shape)
    except TypeError:
        shape = (int(shape),)
    return shape


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


def array_getitem(Array self, item):
    """Array.__getitem__ broken out as separate function to enable line
    profiling."""

    cdef ndarray dest
    cdef Chunk chunk

    # normalise selection
    selection = normalise_array_selection(item, self.shape)

    # determine output array shape
    out_shape = tuple(stop - start for start, stop in selection)

    # setup output array
    out = np.empty(out_shape, dtype=self.dtype)

    # determine indices of overlapping chunks
    chunk_range = get_chunk_range(selection, self.chunks)

    # iterate over chunks in range
    for cidx in itertools.product(*chunk_range):

        # access current chunk
        chunk = self.cdata[cidx]

        # determine chunk offset
        offset = [i * c for i, c in zip(cidx, self.chunks)]

        # determine index range within output array
        out_selection = tuple(
            slice(max(0, o - start), min(o + c - start, stop - start))
            for (start, stop), o, c, in zip(selection, offset, self.chunks)
        )

        # determine required index range within chunk
        chunk_selection = tuple(
            slice(max(0, start - o), min(c, stop - o))
            for (start, stop), o, c in zip(selection, offset, self.chunks)
        )

        # obtain the destination array as a view of the output array
        dest = out[out_selection]

        if chunk.is_initialised and \
                is_total_slice(chunk_selection, chunk.shape) and \
                dest.flags.c_contiguous:

            # optimisation, destination is C contiguous so we can decompress
            # directly from the chunk into the output array
            chunk.decompress(dest.data)

        else:

            # set data in output array
            tmp = chunk[chunk_selection]
            dest[:] = tmp

    return out


def array_setitem(Array self, key, value):
    """Array.__setitem__ broken out as separate function to enable line
    profiling."""

    # normalise selection
    selection = normalise_array_selection(key, self.shape)

    # determine indices of overlapping chunks
    chunk_range = get_chunk_range(selection, self.chunks)

    # iterate over chunks in range
    for cidx in itertools.product(*chunk_range):

        # current chunk
        chunk = self.cdata[cidx]

        # determine chunk offset
        offset = [i * c for i, c in zip(cidx, self.chunks)]

        # determine required index range within chunk
        chunk_selection = tuple(
            slice(max(0, start - o), min(c, stop - o))
            for (start, stop), o, c in zip(selection, offset, self.chunks)
        )

        if np.isscalar(value):

            # fill chunk with scalar value
            chunk[chunk_selection] = value

        else:
            # assume value is array-like

            # determine index within value
            value_selection = tuple(
                slice(max(0, o - start), min(o + c - start, stop - start))
                for (start, stop), o, c, in zip(selection, offset, self.chunks)
            )

            # set data in chunk
            chunk[chunk_selection] = value[value_selection]


# noinspection PyAttributeOutsideInit
cdef class Array:

    def __cinit__(self, shape, chunks, dtype=None, cname=None, clevel=None,
                  shuffle=None, fill_value=None, synchronized=True):

        # N.B., the convention in h5py and dask is to use the "chunks"
        # argument as a tuple representing the shape of each chunk,
        # so we follow that convention here. The actual array of chunk
        # objects will be stored as the "cdata" attribute

        # set shape
        self.shape = normalise_shape(shape)

        # set chunks
        self.chunks = normalise_chunks(chunks, self.shape)

        # set dtype
        self.dtype = np.dtype(dtype)

        # set compression options
        self.cname, self.clevel, self.shuffle = \
            get_cparams(cname, clevel, shuffle)

        # set fill_value
        self.fill_value = fill_value

        # set synchronization option
        self.synchronized = synchronized

        # determine the number and arrangement of chunks
        cdata_shape = tuple(int(np.ceil(s / c))
                            for s, c in zip(self.shape, self.chunks))

        # initialise an object array to hold pointers to chunk objects
        self.cdata = np.empty(cdata_shape, dtype=object)

        # determine function for instantiating chunks
        if self.synchronized:
            create_chunk = SynchronizedChunk
        else:
            create_chunk = Chunk

        # instantiate chunks
        self.cdata.flat = [create_chunk(self.chunks, dtype=self.dtype,
                                        cname=self.cname,
                                        clevel=self.clevel,
                                        shuffle=self.shuffle,
                                        fill_value=self.fill_value)
                           for _ in self.cdata.flat]

        # N.B., in the current implementation, some chunks may overhang
        # the edge of the array. This is handled during the __getitem__ and
        # __setitem__ methods by setting appropriate slices on the chunks,
        # however it may be wasteful depending on chunk shape and the
        # relationship to array shape.

    property nbytes:
        def __get__(self):
            return self.dtype.itemsize * reduce(operator.mul, self.shape)

    property cbytes:
        def __get__(self):
            return sum(c.cbytes for c in self.cdata.flat)

    property is_initialised:
        def __get__(self):
            a = np.empty_like(self.cdata, dtype='b1')
            a.flat = [c.is_initialised for c in self.cdata.flat]
            return a

    def __getitem__(self, item):
        return array_getitem(self, item)

    def __array__(self):
        return self[:]

    def __setitem__(self, key, value):
        array_setitem(self, key, value)

    def __repr__(self):
        r = '%s.%s(' % (type(self).__module__, type(self).__name__)
        r += '%s' % str(self.shape)
        r += ', %s' % str(self.dtype)
        r += ', chunks=%s' % str(self.chunks)
        r += ', cname=%r' % str(self.cname, 'ascii')
        r += ', clevel=%s' % self.clevel
        r += ', shuffle=%s' % self.shuffle
        r += ')'
        r += '\n  nbytes: %s' % _util.human_readable_size(self.nbytes)
        r += '; cbytes: %s' % _util.human_readable_size(self.cbytes)
        if self.cbytes > 0:
            r += '; ratio: %.1f' % (self.nbytes / self.cbytes)
        return r

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
        if len(new_shape) != len(self.shape):
            raise ValueError('new shape must have same number of dimensions')

        # handle None in new_shape
        new_shape = tuple(s if n is None else n
                          for s, n in zip(self.shape, new_shape))

        # work-around Cython problems with accessing .shape attribute as tuple
        old_cdata = np.asarray(self.cdata)

        # remember old cdata shape
        old_cdata_shape = old_cdata.shape

        # determine the new number and arrangement of chunks
        new_cdata_shape = tuple(int(np.ceil(s / c))
                                for s, c in zip(new_shape, self.chunks))

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
        new_cdata.flat = [create_chunk(self.chunks, dtype=self.dtype,
                                       cname=self.cname,
                                       clevel=self.clevel,
                                       shuffle=self.shuffle,
                                       fill_value=self.fill_value)
                          if c is None else c
                          for c in new_cdata.flat]

        # set new shape
        self.shape = new_shape

        # set new chunks
        self.cdata = new_cdata

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
        append_selection = tuple(
            slice(None) if i != axis else slice(old_shape[i], new_shape[i])
            for i in range(len(self.shape))
        )
        self[append_selection] = data
