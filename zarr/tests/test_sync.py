# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from tempfile import mkdtemp
import atexit
import json
import shutil
from multiprocessing.pool import ThreadPool, Pool as ProcessPool
from multiprocessing import cpu_count
import os
import tempfile


import numpy as np
from nose.tools import eq_ as eq


from zarr.tests.test_attrs import TestAttributes
from zarr.tests.test_core import TestArray
from zarr.sync import ThreadSynchronizer, ProcessSynchronizer
from zarr.core import Array
from zarr.attrs import Attributes
from zarr.storage import init_array, TempStore
from zarr.compat import PY2
from zarr.codecs import Zlib


class TestThreadSynchronizedAttributes(TestAttributes):

    def init_attributes(self, store, read_only=False):
        key = 'attrs'
        store[key] = json.dumps(dict()).encode('ascii')
        synchronizer = ThreadSynchronizer()
        return Attributes(store, synchronizer=synchronizer, key=key,
                          read_only=read_only)


class TestProcessSynchronizedAttributes(TestAttributes):

    def init_attributes(self, store, read_only=False):
        key = 'attrs'
        store[key] = json.dumps(dict()).encode('ascii')
        sync_path = mkdtemp()
        atexit.register(shutil.rmtree, sync_path)
        synchronizer = ProcessSynchronizer(sync_path)
        return Attributes(store, synchronizer=synchronizer, key=key,
                          read_only=read_only)


def _append_data(arg):
    z, i = arg
    import numpy as np
    x = np.empty(1000, dtype='i4')
    x[:] = i
    z.append(x)
    return z.shape


class MixinArraySyncTests(object):

    def test_parallel_append(self):

        # setup
        arr = self.create_array(shape=1000, chunks=100, dtype='i4')
        arr[:] = 0
        pool = self.create_pool(cpu_count())

        results = pool.map_async(_append_data, zip([arr] * 39, range(1, 40, 1)))
        print(results.get())

        eq((40000,), arr.shape)


class TestThreadSynchronizedArray(TestArray, MixinArraySyncTests):

    def create_array(self, read_only=False, **kwargs):
        store = dict()
        init_array(store, **kwargs)
        return Array(store, synchronizer=ThreadSynchronizer(),
                     read_only=read_only)

    def test_repr(self):
        if not PY2:

            z = self.create_array(shape=100, chunks=10, dtype='f4',
                                  compressor=Zlib(1))
            # flake8: noqa
            expect = """Array((100,), float32, chunks=(10,), order=C)
  nbytes: 400; nbytes_stored: 245; ratio: 1.6; initialized: 0/10
  compressor: Zlib(level=1)
  store: dict; synchronizer: ThreadSynchronizer
"""
            actual = repr(z)
            for l1, l2 in zip(expect.split('\n'), actual.split('\n')):
                eq(l1, l2)

    def create_pool(self, size):
        pool = ThreadPool(size)
        return pool


class TestProcessSynchronizedArray(TestArray, MixinArraySyncTests):

    def create_array(self, read_only=False, **kwargs):
        store = TempStore()
        init_array(store, **kwargs)
        synchronizer = ProcessSynchronizer(tempfile.TemporaryDirectory().name)
        return Array(store, synchronizer=synchronizer,
                     read_only=read_only, cache_metadata=False)

    def test_repr(self):
        if not PY2:

            z = self.create_array(shape=100, chunks=10, dtype='f4',
                                  compressor=Zlib(1))
            # flake8: noqa
            expect = """Array((100,), float32, chunks=(10,), order=C)
  nbytes: 400; nbytes_stored: 245; ratio: 1.6; initialized: 0/10
  compressor: Zlib(level=1)
  store: TempStore; synchronizer: ProcessSynchronizer
"""
            actual = repr(z)
            for l1, l2 in zip(expect.split('\n'), actual.split('\n')):
                eq(l1, l2)

    def create_pool(self, size):
        pool = ProcessPool(size)
        return pool
