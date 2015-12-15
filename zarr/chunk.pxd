from numpy cimport ndarray, dtype


cdef class zchunk:
    cdef char *data
    cdef public size_t size, nbytes, cbytes, blocksize
    cdef public tuple shape
    cdef public dtype dtype
    cdef compress(self, ndarray array, bytes cname, int clevel, int shuffle)
    cdef decompress(self, char *dest)
