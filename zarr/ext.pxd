from numpy cimport ndarray, dtype


cdef class AbstractChunk:
    cdef object _fill_value
    cdef char *_cname
    cdef int _clevel
    cdef int _shuffle
    cdef tuple _shape
    cdef dtype _dtype
    cdef void get(self, ndarray array)
    cdef void put(self, ndarray array)
    # these methods to be overridden in sub-classes
    cdef tuple retrieve(self)
    cdef void store(self, char *data, size_t nbytes, size_t cbytes)
    cdef void clear(self)


cdef class Chunk(AbstractChunk):
    cdef size_t _nbytes, _cbytes
    cdef char *_data
    cdef free(self)


cdef class SynchronizedChunk(Chunk):
    cdef object _lock


cdef class PersistentChunk(AbstractChunk):
    cdef object _path
    cdef read_header(self)
    cdef read(self)
    cdef write(self, bytes data)


cdef class AbstractArray:
    cdef bytes _cname
    cdef int _clevel
    cdef int _shuffle
    cdef tuple _shape
    cdef tuple _chunks
    cdef dtype _dtype
    cdef ndarray _cdata
    cdef object _fill_value


cdef class Array(AbstractArray):
    pass


cdef class PersistentArray(AbstractArray):
    cdef object _mode
    cdef object _path
