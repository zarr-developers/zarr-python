########################################################################
#
#       License: BSD
#       Created: August 05, 2010
#       Author:  Francesc Alted - francesc@blosc.org
#
########################################################################

"""Here are some definitions for some C headers dependencies."""

import sys

# Standard C functions.
cdef extern from "stdlib.h":
    ctypedef long size_t
    ctypedef long uintptr_t
    void *malloc(size_t size)
    void *realloc(void *ptr, size_t size)
    void free(void *ptr)

cdef extern from "string.h":
    char *strchr(char *s, int c)
    char *strcpy(char *dest, char *src)
    char *strncpy(char *dest, char *src, size_t n)
    int strcmp(char *s1, char *s2)
    char *strdup(char *s)
    void *memcpy(void *dest, void *src, size_t n)
    void *memset(void *s, int c, size_t n)

cdef extern from "time.h":
    ctypedef int time_t


#-----------------------------------------------------------------------------

# Some helper routines from the Python API
# PythonHelper.h is used to help make
# python 2 and 3 both work.
cdef extern from "PythonHelper.h":

    # special types
    ctypedef int Py_ssize_t

    # references
    void Py_INCREF(object)
    void Py_DECREF(object)

    # To release global interpreter lock (GIL) for threading
    void Py_BEGIN_ALLOW_THREADS()
    void Py_END_ALLOW_THREADS()

    # Functions for integers
    object PyInt_FromLong(long)
    long PyInt_AsLong(object)
    object PyLong_FromLongLong(long long)
    long long PyLong_AsLongLong(object)

    # Functions for floating points
    object PyFloat_FromDouble(double)

    # Functions for strings
    object PyBytes_FromString(char *)
    object PyBytes_FromStringAndSize(char *s, int len)
    char *PyBytes_AsString(object string)
    char *PyBytes_AS_STRING(object string)
    size_t PyBytes_GET_SIZE(object string)

    # Functions for lists
    int PyList_Append(object list, object item)

    # Functions for tuples
    object PyTuple_New(int)
    int PyTuple_SetItem(object, int, object)
    object PyTuple_GetItem(object, int)
    int PyTuple_Size(object tuple)

    # Functions for dicts
    int PyDict_Contains(object p, object key)
    object PyDict_GetItem(object p, object key)

    # Functions for objects
    object PyObject_GetItem(object o, object key)
    int PyObject_SetItem(object o, object key, object v)
    int PyObject_DelItem(object o, object key)
    long PyObject_Length(object o)
    int PyObject_Compare(object o1, object o2)
    int PyObject_AsReadBuffer(object obj, void **buffer, Py_ssize_t *buffer_len)

    # Functions for buffers
    object PyBuffer_FromMemory(void *ptr, Py_ssize_t size)

    ctypedef unsigned int Py_uintptr_t


#-----------------------------------------------------------------------------


## Local Variables:
## mode: python
## tab-width: 4
## fill-column: 78
## End:
