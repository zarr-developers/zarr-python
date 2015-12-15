# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


# TODO use blosc contextual functions
# TODO free memory from chunk if garbage collected


import numpy as np
cimport numpy as np
from numpy cimport ndarray, dtype, import_array


from definitions cimport (malloc, realloc, PyBytes_FromStringAndSize)


cdef extern from "blosc.h":
    cdef enum:
        BLOSC_MAX_OVERHEAD,
        BLOSC_VERSION_STRING,
        BLOSC_VERSION_DATE,
        BLOSC_MAX_TYPESIZE

    void blosc_init()
    void blosc_destroy()
    void blosc_get_versions(char *version_str, char *version_date)
    int blosc_set_nthreads(int nthreads)
    int blosc_set_compressor(const char *compname)
    int blosc_compress(int clevel, int doshuffle, size_t typesize,
                       size_t nbytes, void *src, void *dest,
                       size_t destsize) nogil
    int blosc_decompress(void *src, void *dest, size_t destsize) nogil
    int blosc_getitem(void *src, int start, int nitems, void *dest) nogil
    void blosc_free_resources()
    void blosc_cbuffer_sizes(void *cbuffer, size_t *nbytes,
                             size_t *cbytes, size_t *blocksize)
    void blosc_cbuffer_metainfo(void *cbuffer, size_t *typesize, int *flags)
    void blosc_cbuffer_versions(void *cbuffer, int *version, int *versionlz)
    void blosc_set_blocksize(size_t blocksize)
    char *blosc_list_compressors()


import_array()


from zarr import defaults


def get_cparams(cname, clevel, shuffle):

    # determine compressor
    cname = cname if cname is not None else defaults.cname
    if type(cname) != bytes:
        cname = cname.encode()

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


cdef class zchunk:

    def __cinit__(self, array, cname=None, clevel=None, shuffle=None):

        # ensure array is C contiguous
        # TODO adapt to either C or F layout
        array = np.ascontiguousarray(array)

        # determine compression options
        self.cname, self.clevel, self.shuffle = \
            get_cparams(cname, clevel, shuffle)

        # determine size, shape and dtype
        self.size = array.size
        self.shape = array.shape
        self.dtype = np.dtype(array.dtype.base)

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

        # set compressor
        if blosc_set_compressor(self.cname) < 0:
            raise ValueError("compressor not available: %s" % self.cname)

        # allocate memory for compressed data
        dest = <char *> malloc(nbytes + BLOSC_MAX_OVERHEAD)

        # perform compression
        with nogil:
            cbytes = blosc_compress(self.clevel, self.shuffle, itemsize,
                                    nbytes, array.data, dest,
                                    nbytes + BLOSC_MAX_OVERHEAD)

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

    def __array__(self):
        cdef:
            ndarray array

        # setup output array
        array = np.empty(self.shape, dtype=self.dtype)

        # decompress data
        self.decompress(array.data)

        return array

    cdef decompress(self, char *dest):
        cdef:
            int ret

        # do decompression
        with nogil:
            ret = blosc_decompress(self.data, dest, self.nbytes)

        # handle errors
        if ret <= 0:
            raise RuntimeError("error during blosc compression: %d" % ret)


def _blosc_set_nthreads(nthreads):
    return blosc_set_nthreads(nthreads)

def _blosc_init():
    blosc_init()

def _blosc_destroy():
    blosc_destroy()

def blosc_version():
    # all the 'decode' contorsions are for Python 3 returning actual strings
    ver_str = <char *> BLOSC_VERSION_STRING
    if hasattr(ver_str, "decode"):
        ver_str = ver_str.decode()
    ver_date = <char *> BLOSC_VERSION_DATE
    if hasattr(ver_date, "decode"):
        ver_date = ver_date.decode()
    return ver_str, ver_date
