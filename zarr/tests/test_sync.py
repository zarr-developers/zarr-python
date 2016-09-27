# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from tempfile import mkdtemp
import atexit
import json
import shutil
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool as ProcessPool
from multiprocessing import cpu_count
import tempfile
import traceback
import sys


import numpy as np
from nose.tools import eq_ as eq, assert_is_instance
from numpy.testing import assert_array_equal


from zarr.tests.test_attrs import TestAttributes
from zarr.tests.test_core import TestArray
from zarr.tests.test_hierarchy import TestGroup
from zarr.sync import ThreadSynchronizer, ProcessSynchronizer
from zarr.core import Array
from zarr.attrs import Attributes
from zarr.storage import init_array, DirectoryStore, init_group, atexit_rmtree
from zarr.compat import PY2
from zarr.codecs import Zlib
from zarr.hierarchy import Group


class TestAttributesWithThreadSynchronizer(TestAttributes):

    def init_attributes(self, store, read_only=False):
        key = 'attrs'
        store[key] = json.dumps(dict()).encode('ascii')
        synchronizer = ThreadSynchronizer()
        return Attributes(store, synchronizer=synchronizer, key=key,
                          read_only=read_only)


class TestAttributesProcessSynchronizer(TestAttributes):

    def init_attributes(self, store, read_only=False):
        key = 'attrs'
        store[key] = json.dumps(dict()).encode('ascii')
        sync_path = mkdtemp()
        atexit.register(shutil.rmtree, sync_path)
        synchronizer = ProcessSynchronizer(sync_path)
        return Attributes(store, synchronizer=synchronizer, key=key,
                          read_only=read_only)


def _append(arg):
    z, i = arg
    import numpy as np
    x = np.empty(1000, dtype='i4')
    x[:] = i
    shape = z.append(x)
    return shape


def _set_arange(arg):
    z, i = arg
    import numpy as np
    x = np.arange(i*1000, (i*1000)+1000, 1)
    z[i*1000:(i*1000)+1000] = x
    return i


class MixinArraySyncTests(object):

    def test_parallel_setitem(self):
        n = 100

        # setup
        arr = self.create_array(shape=n * 1000, chunks=999, dtype='i4')
        arr[:] = 0
        pool = self.create_pool()

        # parallel setitem
        results = pool.map(_set_arange, zip([arr] * n, range(n)), chunksize=1)
        results = sorted(results)

        print(results)
        eq(list(range(n)), results)
        assert_array_equal(np.arange(n * 1000), arr[:])

        pool.terminate()

    def test_parallel_append(self):
        n = 100

        # setup
        arr = self.create_array(shape=1000, chunks=999, dtype='i4')
        arr[:] = 0
        pool = self.create_pool()

        # parallel append
        results = pool.map(_append, zip([arr] * n, range(n)), chunksize=1)
        results = sorted(results)

        print(results)
        eq([((i+2)*1000,) for i in range(n)], results)
        eq(((n+1)*1000,), arr.shape)

        pool.terminate()


class TestArrayWithThreadSynchronizer(TestArray, MixinArraySyncTests):

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

    def create_pool(self):
        pool = ThreadPool(cpu_count())
        return pool


class TestArrayWithProcessSynchronizer(TestArray, MixinArraySyncTests):

    def create_array(self, read_only=False, **kwargs):
        path = tempfile.mkdtemp()
        atexit.register(atexit_rmtree, path)
        store = DirectoryStore(path)
        init_array(store, **kwargs)
        sync_path = tempfile.mkdtemp()
        atexit.register(atexit_rmtree, sync_path)
        synchronizer = ProcessSynchronizer(sync_path)
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
  store: DirectoryStore; synchronizer: ProcessSynchronizer
"""
            actual = repr(z)
            for l1, l2 in zip(expect.split('\n'), actual.split('\n')):
                eq(l1, l2)

    def create_pool(self):
        pool = ProcessPool(processes=cpu_count())
        return pool


def _create_group(arg):
    g, name = arg
    h = g.create_group(name)
    return h.name


def _require_group(arg):
    g, name = arg
    h = g.require_group(name)
    return h.name


class MixinGroupSyncTests(object):

    def test_parallel_create_group(self):

        # setup
        g = self.create_group()
        pool = self.create_pool()

        # parallel create group
        n = 100
        results = pool.map(
            _create_group,
            zip([g] * n, [str(i) for i in range(n)]),
            chunksize=1
        )
        results = sorted(results)
        pool.close()
        pool.terminate()

        print(results)
        eq(n, len(g))

        pool.terminate()

    def test_parallel_require_group(self):

        # setup
        g = self.create_group()
        pool = self.create_pool()

        # parallel require group
        n = 100
        results = pool.map(
            _require_group,
            zip([g] * n, [str(i//10) for i in range(n)]),
            chunksize=1
        )
        results = sorted(results)
        pool.close()
        pool.terminate()

        print(results)
        eq(n//10, len(g))

        pool.terminate()


class TestGroupWithThreadSynchronizer(TestGroup, MixinGroupSyncTests):

    def create_group(self, store=None, path=None, read_only=False,
                     chunk_store=None, synchronizer=None):
        if store is None:
            store, chunk_store = self.create_store()
        init_group(store, path=path, chunk_store=chunk_store)
        synchronizer = ThreadSynchronizer()
        g = Group(store, path=path, read_only=read_only,
                  chunk_store=chunk_store, synchronizer=synchronizer)
        return g

    def create_pool(self):
        pool = ThreadPool(cpu_count())
        return pool

    def test_group_repr(self):
        if not PY2:
            g = self.create_group()
            expect = 'Group(/, 0)\n' \
                     '  store: dict; synchronizer: ThreadSynchronizer'
            actual = repr(g)
            for l1, l2 in zip(expect.split('\n'), actual.split('\n')):
                eq(l1, l2)

    def test_synchronizer_property(self):
        g = self.create_group()
        assert_is_instance(g.synchronizer, ThreadSynchronizer)


class TestGroupWithProcessSynchronizer(TestGroup, MixinGroupSyncTests):

    def create_store(self):
        path = tempfile.mkdtemp()
        atexit.register(atexit_rmtree, path)
        store = DirectoryStore(path)
        return store, None

    def create_group(self, store=None, path=None, read_only=False,
                     chunk_store=None, synchronizer=None):
        if store is None:
            store, chunk_store = self.create_store()
        init_group(store, path=path, chunk_store=chunk_store)
        sync_path = tempfile.mkdtemp()
        atexit.register(atexit_rmtree, sync_path)
        synchronizer = ProcessSynchronizer(sync_path)
        g = Group(store, path=path, read_only=read_only,
                  synchronizer=synchronizer, chunk_store=chunk_store)
        return g

    def create_pool(self):
        pool = ProcessPool(processes=cpu_count())
        return pool

    def test_group_repr(self):
        if not PY2:
            g = self.create_group()
            expect = 'Group(/, 0)\n' \
                     '  store: DirectoryStore; synchronizer: ProcessSynchronizer'
            actual = repr(g)
            for l1, l2 in zip(expect.split('\n'), actual.split('\n')):
                eq(l1, l2)

    def test_synchronizer_property(self):
        g = self.create_group()
        assert_is_instance(g.synchronizer, ProcessSynchronizer)
