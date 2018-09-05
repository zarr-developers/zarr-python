# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from tempfile import mkdtemp
import atexit
import shutil


import numpy as np
import pytest


from zarr.creation import array, open_array
from zarr.storage import DirectoryStore


@pytest.mark.parametrize(
    "dtype",
    map(np.dtype, [
        # NOTE(onalant): unstructured dtypes
        "f8", "10f8", "(10, 10)f8",
        # NOTE(onalant): structured dtypes
        "i8, f8", "i8, 10f8", "i8, (10, 10)f8",  "i8, (10, 10)f8, (5, 10, 15)u1",
        # NOTE(onalant): nested dtypes
        [("f0", "i8"), ("f1", [("f0", "f8"), ("f1", "10f8"), ("f2", "(10, 10)f8")])]
    ])
)
def test_write_read(dtype):

    path = mkdtemp()
    atexit.register(shutil.rmtree, path)
    shape = (100, 100)
    sbytes = np.random.bytes(np.product(shape) * dtype.itemsize)
    s = np.frombuffer(sbytes, dtype=dtype)
    # NOTE(onalant): np.frombuffer only creates 1D arrays; expand to shape
    s.reshape(shape + s.shape[1:])

    store = DirectoryStore(path)
    z = array(s, store=store)

    assert(s.dtype == z.dtype)
    assert(s.shape == z.shape)
    assert(s.tobytes() == z[:].tobytes())

    del store
    del z

    store = DirectoryStore(path)
    z = open_array(store)

    assert(s.dtype == z.dtype)
    assert(s.shape == z.shape)
    assert(s.tobytes() == z[:].tobytes())

