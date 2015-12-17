# -*- coding: utf-8 -*-
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
    # all the 'decode' contorsions are for Python 3 returning actual strings
    ver_str = <char *> BLOSC_VERSION_STRING
    if hasattr(ver_str, "decode"):
        ver_str = ver_str.decode()
    ver_date = <char *> BLOSC_VERSION_DATE
    if hasattr(ver_date, "decode"):
        ver_date = ver_date.decode()
    return ver_str, ver_date


def get_cparams(cname=None, clevel=None, shuffle=None):

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


def is_total_slice(item, shape):
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


# noinspection PyAttributeOutsideInit
cdef class Chunk:

    def __cinit__(self, shape, dtype=None, cname=None, clevel=None,
                  shuffle=None, fill_value=None):

        # set shape and dtype
        if isinstance(shape, int):
            shape = (shape,)
        else:
            shape = tuple(shape)
        self.shape = shape
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

        if is_total_slice(key, self.shape):
            # completely replace the contents of this chunk

            if np.isscalar(value):
                array = np.empty(self.shape, dtype=self.dtype)
                array.fill(value)

            else:
                # ensure array is C contiguous
                # TODO adapt to either C or F layout
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

    cdef compress(self, ndarray array):
        cdef:
            size_t nbytes, nbytes_check, cbytes, blocksize, itemsize
            char *dest

        # ensure any existing data is cleared
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
        cdef:
            ndarray array

        # setup output array
        array = np.empty(self.shape, dtype=self.dtype)

        if self.data == NULL:
            # data not initialised
            if self.fill_value is not None:
                array.fill(self.fill_value)

        else:
            # data initialised, decompress into array
            self.decompress(array.data)

        return array[item]

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


class Synchronized(object):

    def __init__(self, inner):
        self.inner = inner
        self.lock = RLock()

    def __getitem__(self, item):
        with self.lock:
            return self.inner.__getitem__(item)
            
    def __setitem__(self, key, value):
        with self.lock:
            self.inner.__setitem__(key, value)

    def __getattr__(self, item):
        return getattr(self.inner, item)


def normalise_array_selection(item, shape):

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
    if isinstance(item, int):
        if item < 0:
            # handle wraparound
            item = l + item
        if item > (l - 1) or item < 0:
            raise IndexError('index out of bounds: %s' % item)
        return item, item + 1

    elif isinstance(item, slice):
        if item.step is not None and item.step != 1:
            raise NotImplementedError('TODO')
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


def get_chunk_range(selection, chunks):
    chunk_range = [range(start//l, int(np.ceil(stop/l)))
                   for (start, stop), l in zip(selection, chunks)]
    return chunk_range


def normalise_shape(shape):
    if isinstance(shape, int):
        shape = (shape,)
    else:
        shape = tuple(shape)
    return shape


def normalise_chunks(chunks, shape):
    if isinstance(chunks, int):
        chunks = (chunks,)
    else:
        chunks = tuple(chunks)
    if len(chunks) != len(shape):
        raise ValueError('chunks and shape not compatible: %r, %r' %
                         (chunks, shape))
    # handle None in chunks
    chunks = tuple(s if c is None else c for s, c in zip(shape, chunks))
    return chunks


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
        if synchronized:
            def create_chunk(*args, **kwargs):
                return Synchronized(Chunk(*args, **kwargs))
        else:
            create_chunk = Chunk

        # instantiate chunks
        self.cdata.flat = [create_chunk(self.chunks, dtype=dtype, cname=cname,
                                        clevel=clevel, shuffle=shuffle,
                                        fill_value=fill_value)
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

            # determine chunk offset
            offset = [i * c for i, c in zip(cidx, self.chunks)]

            # determine required index range within chunk
            chunk_selection = tuple(
                slice(max(0, start - o), min(c, stop - o))
                for (start, stop), o, c in zip(selection, offset, self.chunks)
            )

            # determine index range within output array
            out_selection = tuple(
                slice(max(0, o - start), min(o + c - start, stop - start))
                for (start, stop), o, c, in zip(selection, offset, self.chunks)
            )

            # read data into output array
            chunk = self.cdata[cidx]
            out[out_selection] = chunk[chunk_selection]

        return out

    def __setitem__(self, key, value):

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

    def __repr__(self):
        r = '%s.%s(' % (type(self).__module__, type(self).__name__)
        r += '%s' % str(self.shape)
        r += ', %s' % str(self.dtype)
        r += ', chunks=%s' % str(self.chunks)
        r += ', nbytes=%s' % _util.human_readable_size(self.nbytes)
        r += ', cbytes=%s' % _util.human_readable_size(self.cbytes)
        if self.cbytes > 0:
            r += ', cratio=%.1f' % (self.nbytes / self.cbytes)
        r += ', cname=%s' % str(self.cname, 'ascii')
        r += ', clevel=%s' % self.clevel
        r += ', shuffle=%s' % self.shuffle
        r += ')'
        return r
