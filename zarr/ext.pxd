from numpy cimport ndarray, dtype


cdef class Chunk:
    cdef char *data
    cdef public object fill_value
    cdef public size_t nbytes, cbytes, blocksize
    cdef public char *cname
    cdef public int clevel
    cdef public int shuffle
    cdef public tuple shape
    cdef public dtype dtype
    cdef free(self)
    cdef clear(self)
    cdef compress(self, ndarray array)
    cdef decompress(self, char *dest)


cdef class SynchronizedChunk(Chunk):
    cdef object lock


cdef class Array:
    cdef public bytes cname
    cdef public int clevel
    cdef public int shuffle
    cdef public tuple shape
    cdef public tuple chunks
    cdef public dtype dtype
    cdef public ndarray cdata
    cdef public object fill_value
    cdef public bint synchronized
