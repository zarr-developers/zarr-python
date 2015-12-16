# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from threading import RLock


import numpy as np
cimport numpy as np
from numpy cimport ndarray, dtype


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


# noinspection PyAttributeOutsideInit
cdef class Chunk:

    def __init__(self, shape, dtype=None, cname=None, clevel=None,
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

    def __setitem__(self, key, value):

        if key == Ellipsis or key == slice(None, None, None):
            # completely replace the contents of this chunk

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

        # compress the data
        self.compress(array)

    cdef compress(self, ndarray array):
        cdef:
            size_t nbytes, nbytes_check, cbytes, blocksize, itemsize
            char *dest

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

    def __dealloc__(self):
        if self.data != NULL:
            free(self.data)


class Synchronized(object):

    def __init__(self, chunk):
        self.chunk = chunk
        self.lock = RLock()

    def __getitem__(self, item):
        with self.lock:
            return self.chunk.__getitem__(item)
            
    def __setitem__(self, key, value):
        with self.lock:
            self.chunk.__setitem__(key, value)


def blosc_version():
    # all the 'decode' contorsions are for Python 3 returning actual strings
    ver_str = <char *> BLOSC_VERSION_STRING
    if hasattr(ver_str, "decode"):
        ver_str = ver_str.decode()
    ver_date = <char *> BLOSC_VERSION_DATE
    if hasattr(ver_date, "decode"):
        ver_date = ver_date.decode()
    return ver_str, ver_date
