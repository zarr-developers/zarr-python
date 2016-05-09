# -*- coding: utf-8 -*-
# cython: embedsignature=True
# cython: profile=False
# cython: linetrace=False
# cython: binding=False
from __future__ import absolute_import, print_function, division
import threading


from numpy cimport ndarray
from cpython.bytes cimport PyBytes_AsString, PyBytes_FromStringAndSize
from cpython.mem cimport PyMem_Malloc, PyMem_Free


from zarr.compat import PY2, text_type


cdef extern from "blosc.h":
    cdef enum:
        BLOSC_MAX_OVERHEAD,
        BLOSC_VERSION_STRING,
        BLOSC_VERSION_DATE

    void blosc_init()
    void blosc_destroy()
    int blosc_set_nthreads(int nthreads)
    int blosc_set_compressor(const char *compname)
    int blosc_compress(int clevel, int doshuffle, size_t typesize,
                       size_t nbytes, void *src, void *dest,
                       size_t destsize) nogil
    int blosc_decompress(void *src, void *dest, size_t destsize) nogil
    int blosc_compname_to_compcode(const char *compname)
    int blosc_compress_ctx(int clevel, int doshuffle, size_t typesize,
                           size_t nbytes, const void* src, void* dest,
                           size_t destsize, const char* compressor,
				           size_t blocksize, int numinternalthreads) nogil
    int blosc_decompress_ctx(const void *src, void *dest, size_t destsize,
                             int numinternalthreads) nogil


def version():
    """Return the version of blosc that zarr was compiled with."""

    ver_str = <char *> BLOSC_VERSION_STRING
    ver_date = <char *> BLOSC_VERSION_DATE
    if not PY2:
        ver_str = ver_str.decode()
        ver_date = ver_date.decode()
    return ver_str, ver_date


def init():
    blosc_init()


def destroy():
    blosc_destroy()


def compname_to_compcode(cname):
    if isinstance(cname, text_type):
        cname = cname.encode('ascii')
    return blosc_compname_to_compcode(cname)


def set_nthreads(int nthreads):
    """Set the number of threads that Blosc uses internally for compression
    and decompression.
    ."""
    return blosc_set_nthreads(nthreads)


def decompress(bytes cdata, ndarray array):
    """Decompress data into a numpy array.

    Parameters
    ----------
    cdata : bytes
        Compressed data, including blosc header.
    array : ndarray
        Numpy array to decompress into.

    Notes
    -----
    Assumes that the size of the destination array is correct for the size of
    the uncompressed data.

    """
    cdef:
        int ret
        char *source
        char *dest
        size_t nbytes

    # setup
    source = PyBytes_AsString(cdata)
    dest = array.data
    nbytes = array.nbytes

    # perform decompression
    if _get_use_threads():
        # allow blosc to use threads internally
        with nogil:
            ret = blosc_decompress(source, dest, nbytes)
    else:
        with nogil:
            ret = blosc_decompress_ctx(source, dest, nbytes, 1)

    # handle errors
    if ret <= 0:
        raise RuntimeError('error during blosc decompression: %d' % ret)


def compress(ndarray array, char* cname, int clevel, int shuffle):
    """Compress data in a numpy array.

    Parameters
    ----------
    array : ndarray
        Numpy array containing data to be compressed.
    cname : bytes
        Name of compression library to use.
    clevel : int
        Compression level.
    shuffle : int
        Shuffle filter.

    Returns
    -------
    cdata : bytes
        Compressed data.

    """

    cdef:
        char *source
        char *dest
        size_t nbytes, cbytes, itemsize
        bytes cdata

    # obtain reference to underlying buffer
    source = array.data

    # allocate memory for compressed data
    nbytes = array.nbytes
    itemsize = array.dtype.itemsize
    dest = <char *> PyMem_Malloc(nbytes + BLOSC_MAX_OVERHEAD)

    # perform compression
    if _get_use_threads():
        # allow blosc to use threads internally
        compressor_set = blosc_set_compressor(cname)
        if compressor_set < 0:
            raise ValueError('compressor not supported: %r' % cname)
        with nogil:
            cbytes = blosc_compress(clevel, shuffle, itemsize, nbytes, source,
                                    dest, nbytes + BLOSC_MAX_OVERHEAD)

    else:
        with nogil:
            cbytes = blosc_compress_ctx(clevel, shuffle, itemsize, nbytes,
                                        source, dest,
                                        nbytes + BLOSC_MAX_OVERHEAD, cname,
                                        0, 1)

    # check compression was successful
    if cbytes <= 0:
        raise RuntimeError('error during blosc compression: %d' % cbytes)

    # store as bytes
    cdata = PyBytes_FromStringAndSize(dest, cbytes)

    # release memory
    PyMem_Free(dest)

    return cdata


# set the value of this variable to True or False to override the
# default adaptive behaviour
use_threads = None


def _get_use_threads():
    global use_threads

    if use_threads in [True, False]:
        # user has manually overridden the default behaviour
        _use_threads = use_threads

    else:
        # adaptive behaviour: allow blosc to use threads if it is being
        # called from the main Python thread, inferring that it is being run
        # from within a single-threaded program; otherwise do not allow
        # blosc to use threads, inferring it is being run from within a
        # multi-threaded program
        if hasattr(threading, 'main_thread'):
            _use_threads = (threading.main_thread() ==
                            threading.current_thread())
        else:
            _use_threads = threading.current_thread().name == 'MainThread'

    return _use_threads
