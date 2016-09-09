# -*- coding: utf-8 -*-
# cython: embedsignature=True
# cython: profile=False
# cython: linetrace=False
# cython: binding=False
from __future__ import absolute_import, print_function, division
import threading


# noinspection PyUnresolvedReferences
from cpython cimport array, PyObject
import array
from cpython.buffer cimport PyObject_GetBuffer, PyBuffer_Release, \
    PyBUF_ANY_CONTIGUOUS, PyBUF_WRITEABLE
from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AS_STRING, \
    _PyBytes_Resize


from zarr.compat import PY2, text_type


cdef extern from "blosc.h":
    cdef enum:
        BLOSC_MAX_OVERHEAD,
        BLOSC_VERSION_STRING,
        BLOSC_VERSION_DATE,
        BLOSC_NOSHUFFLE,
        BLOSC_SHUFFLE,
        BLOSC_BITSHUFFLE,
        BLOSC_MAX_BUFFERSIZE,
        BLOSC_MAX_THREADS,
        BLOSC_MAX_TYPESIZE

    void blosc_init()
    void blosc_destroy()
    int blosc_get_nthreads()
    int blosc_set_nthreads(int nthreads)
    int blosc_set_compressor(const char *compname)
    char* blosc_list_compressors()
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
    void blosc_cbuffer_sizes(const void *cbuffer, size_t *nbytes,
                             size_t *cbytes, size_t *blocksize)


MAX_OVERHEAD = BLOSC_MAX_OVERHEAD
MAX_BUFFERSIZE = BLOSC_MAX_BUFFERSIZE
MAX_THREADS = BLOSC_MAX_THREADS
MAX_TYPESIZE = BLOSC_MAX_TYPESIZE
VERSION_STRING = <char *> BLOSC_VERSION_STRING
VERSION_DATE = <char *> BLOSC_VERSION_DATE
if not PY2:
    VERSION_STRING = VERSION_STRING.decode()
    VERSION_DATE = VERSION_DATE.decode()
__version__ = VERSION_STRING
NOSHUFFLE = BLOSC_NOSHUFFLE
SHUFFLE = BLOSC_SHUFFLE
BITSHUFFLE = BLOSC_BITSHUFFLE


def version():
    return VERSION_STRING, VERSION_DATE


def init():
    blosc_init()


def destroy():
    blosc_destroy()


def compname_to_compcode(cname):
    if isinstance(cname, text_type):
        cname = cname.encode('ascii')
    return blosc_compname_to_compcode(cname)


def list_compressors():
    return text_type(blosc_list_compressors(), 'ascii').split(',')


def get_nthreads():
    """Get the number of threads that Blosc uses internally for compression
    and decompression.
    ."""
    return blosc_get_nthreads()


def set_nthreads(int nthreads):
    """Set the number of threads that Blosc uses internally for compression
    and decompression.
    ."""
    return blosc_set_nthreads(nthreads)


def cbuffer_sizes(source):
    """Return information from the blosc header of some compressed data."""
    cdef:
        char *source_ptr
        Py_buffer source_buffer
        array.array source_array
        size_t nbytes, cbytes, blocksize

    # setup source buffer
    if PY2 and isinstance(source, array.array):
        # workaround fact that array.array does not support new-style buffer
        # interface in PY2
        release_source_buffer = False
        source_array = source
        source_ptr = <char *> source_array.data.as_voidptr
    else:
        release_source_buffer = True
        PyObject_GetBuffer(source, &source_buffer, PyBUF_ANY_CONTIGUOUS)
        source_ptr = <char *> source_buffer.buf

    # determine buffer size
    blosc_cbuffer_sizes(source_ptr, &nbytes, &cbytes, &blocksize)

    # release buffers
    if release_source_buffer:
        PyBuffer_Release(&source_buffer)

    return nbytes, cbytes, blocksize


def decompress(source, dest=None):
    """Decompress data.

    Parameters
    ----------
    source : bytes-like
        Compressed data, including blosc header. Can be any object
        supporting the buffer protocol.
    dest : array-like, optional
        Object to decompress into.

    Returns
    -------
    dest : array-like
        Object containing decompressed data.

    """
    cdef:
        int ret
        char *source_ptr
        char *dest_ptr
        Py_buffer source_buffer
        array.array source_array
        Py_buffer dest_buffer
        size_t nbytes, cbytes, blocksize

    # setup source buffer
    if PY2 and isinstance(source, array.array):
        # workaround fact that array.array does not support new-style buffer
        # interface in PY2
        release_source_buffer = False
        source_array = source
        source_ptr = <char *> source_array.data.as_voidptr
    else:
        release_source_buffer = True
        PyObject_GetBuffer(source, &source_buffer, PyBUF_ANY_CONTIGUOUS)
        source_ptr = <char *> source_buffer.buf

    # determine buffer size
    blosc_cbuffer_sizes(source_ptr, &nbytes, &cbytes, &blocksize)

    # setup destination buffer
    release_dest_buffer = False
    if dest is None:
        # allocate memory
        dest = PyBytes_FromStringAndSize(NULL, nbytes)
        dest_ptr = PyBytes_AS_STRING(dest)
        dest_nbytes = nbytes
    elif PY2 and isinstance(dest, array.array):
        # workaround fact that array.array does not support new-style buffer
        # interface in PY2
        dest_array = dest
        dest_ptr = <char *> dest_array.data.as_voidptr
        dest_nbytes = dest_array.buffer_info()[1] * dest_array.itemsize
    else:
        release_dest_buffer = True
        PyObject_GetBuffer(dest, &dest_buffer,
                           PyBUF_ANY_CONTIGUOUS | PyBUF_WRITEABLE)
        dest_ptr = <char *> dest_buffer.buf
        dest_nbytes = dest_buffer.len

    try:

        # guard condition
        if dest_nbytes != nbytes:
            raise ValueError('destination buffer has wrong size; expected %s, '
                             'got %s' % (nbytes, dest_nbytes))

        # perform decompression
        if _get_use_threads():
            # allow blosc to use threads internally
            with nogil:
                ret = blosc_decompress(source_ptr, dest_ptr, nbytes)
        else:
            with nogil:
                ret = blosc_decompress_ctx(source_ptr, dest_ptr, nbytes, 1)

    finally:

        # release buffers
        if release_source_buffer:
            PyBuffer_Release(&source_buffer)
        if release_dest_buffer:
            PyBuffer_Release(&dest_buffer)

    # handle errors
    if ret <= 0:
        raise RuntimeError('error during blosc decompression: %d' % ret)

    return dest


def compress(source, char* cname, int clevel, int shuffle):
    """Compression.

    Parameters
    ----------
    source : bytes-like
        Data to be compressed. Can be any object supporting the buffer
        protocol.
    cname : bytes
        Name of compression library to use.
    clevel : int
        Compression level.
    shuffle : int
        Shuffle filter.

    Returns
    -------
    dest : array
        Compressed data.

    """

    cdef:
        char *source_ptr
        char *dest_ptr
        Py_buffer source_buffer
        size_t nbytes, cbytes, itemsize
        array.array source_array
        bytes dest

    # setup source buffer
    if PY2 and isinstance(source, array.array):
        # workaround fact that array.array does not support new-style buffer
        # interface in PY2
        release_source_buffer = False
        source_array = source
        source_ptr = <char *> source_array.data.as_voidptr
        itemsize = source_array.itemsize
        nbytes = source_array.buffer_info()[1] * itemsize
    else:
        release_source_buffer = True
        PyObject_GetBuffer(source, &source_buffer, PyBUF_ANY_CONTIGUOUS)
        source_ptr = <char *> source_buffer.buf
        itemsize = source_buffer.itemsize
        nbytes = source_buffer.len

    try:

        # setup destination
        dest = PyBytes_FromStringAndSize(NULL, nbytes + BLOSC_MAX_OVERHEAD)
        dest_ptr = PyBytes_AS_STRING(dest)

        # perform compression
        if _get_use_threads():
            # allow blosc to use threads internally
            compressor_set = blosc_set_compressor(cname)
            if compressor_set < 0:
                raise ValueError('compressor not supported: %r' % cname)
            with nogil:
                cbytes = blosc_compress(clevel, shuffle, itemsize, nbytes,
                                        source_ptr, dest_ptr,
                                        nbytes + BLOSC_MAX_OVERHEAD)

        else:
            with nogil:
                cbytes = blosc_compress_ctx(clevel, shuffle, itemsize, nbytes,
                                            source_ptr, dest_ptr,
                                            nbytes + BLOSC_MAX_OVERHEAD, cname,
                                            0, 1)

    finally:

        # release buffers
        if release_source_buffer:
            PyBuffer_Release(&source_buffer)

    # check compression was successful
    if cbytes <= 0:
        raise RuntimeError('error during blosc compression: %d' % cbytes)

    # resize after compression
    dest = dest[:cbytes]

    return dest


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
