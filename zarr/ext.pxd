from numpy cimport ndarray, dtype


cdef class Chunk:
    cdef char *data
    cdef public size_t size, nbytes, cbytes, blocksize
    cdef public char *cname
    cdef public int clevel
    cdef public int shuffle
    cdef public tuple shape
    cdef public dtype dtype
    cdef compress(self, ndarray array)
    cdef decompress(self, char *dest)
