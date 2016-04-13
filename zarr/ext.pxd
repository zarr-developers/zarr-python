from numpy cimport ndarray, dtype


cdef class BaseChunk:
    cdef object _fill_value
    cdef char *_cname
    cdef int _clevel
    cdef int _shuffle
    cdef tuple _shape
    cdef dtype _dtype
    cdef size_t _size
    cdef size_t _itemsize
    cdef size_t _nbytes
    # abstract methods
    cdef void get(self, char *dest)
    cdef void put(self, char *source)


cdef class Chunk(BaseChunk):
    cdef char *_data
    cdef size_t _cbytes
    cdef void clear(self)
    cdef void free(self)


cdef class SynchronizedChunk(Chunk):
    cdef object _lock


cdef class PersistentChunk(BaseChunk):
    cdef object _path
    cdef object _basename
    cdef object _dirname
    cdef object read_header(self)
    cdef bytes read(self)
    cdef void write(self, bytes data)


cdef class SynchronizedPersistentChunk(PersistentChunk):
    cdef object _thread_lock
    cdef object _file_lock


cdef class BaseArray:
    cdef tuple _shape
    cdef tuple _cdata_shape
    cdef tuple _chunks
    cdef dtype _dtype
    cdef size_t _size
    cdef size_t _itemsize
    cdef size_t _nbytes
    cdef bytes _cname
    cdef int _clevel
    cdef int _shuffle
    cdef object _fill_value
    cdef object _attrs
    cdef object _cdata
    # abstract methods
    cdef BaseChunk create_chunk(self, tuple cidx)
    cdef BaseChunk get_chunk(self, tuple cidx)


cdef class Array(BaseArray):
    pass


cdef class SynchronizedArray(Array):
    pass


cdef class LazyArray(BaseArray):
    pass


cdef class SynchronizedLazyArray(LazyArray):
    cdef object _lock


cdef class PersistentArray(BaseArray):
    cdef object _mode
    cdef object _path
    cdef object get_chunk_path(self, tuple cidx)


cdef class SynchronizedPersistentArray(PersistentArray):
    pass


cdef class LazyPersistentArray(PersistentArray):
    pass


cdef class SynchronizedLazyPersistentArray(LazyPersistentArray):
    cdef object _lock
