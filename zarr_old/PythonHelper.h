/*
 * This header is to help compile on both Python
 * 2 and Python 3 with the same Cython source code.
 */
#ifndef PYTHON_HELPER_H
#define PYTHON_HELPER_H

#include "Python.h"

#if PY_VERSION_HEX >= 0x03000000
#define PyBuffer_FromMemory(ptr, size) PyMemoryView_FromMemory(ptr, size, PyBUF_READ|PyBUF_WRITE)
#endif

#endif /* PYTHON_HELPER_H */
