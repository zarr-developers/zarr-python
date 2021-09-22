import atexit
import shutil
import tempfile
from multiprocessing import Pool as ProcessPool
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from tempfile import mkdtemp

import numpy as np
from numpy.testing import assert_array_equal

from zarr.attrs import Attributes
from zarr.core import Array
from zarr.hierarchy import Group
from zarr.storage import (DirectoryStore, KVStore, atexit_rmtree, init_array,
                          init_group)
from zarr.sync import ProcessSynchronizer, ThreadSynchronizer
from zarr.tests.test_attrs import TestAttributes
from zarr.tests.test_core import TestArray
from zarr.tests.test_hierarchy import TestGroup


class TestAttributesWithThreadSynchronizer(TestAttributes):

    def init_attributes(self, store, read_only=False, cache=True):
        key = 'attrs'
        synchronizer = ThreadSynchronizer()
        return Attributes(store, synchronizer=synchronizer, key=key,
                          read_only=read_only, cache=cache)


class TestAttributesProcessSynchronizer(TestAttributes):

    def init_attributes(self, store, read_only=False, cache=True):
        key = 'attrs'
        sync_path = mkdtemp()
        atexit.register(shutil.rmtree, sync_path)
        synchronizer = ProcessSynchronizer(sync_path)
        return Attributes(store, synchronizer=synchronizer, key=key,
                          read_only=read_only, cache=cache)


def _append(arg):
    z, i = arg
    import numpy
    x = numpy.empty(1000, dtype='i4')
    x[:] = i
    shape = z.append(x)
    return shape


def _set_arange(arg):
    z, i = arg
    import numpy
    x = numpy.arange(i*1000, (i*1000)+1000, 1)
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

        assert list(range(n)) == results
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

        assert [((i+2)*1000,) for i in range(n)] == results
        assert ((n+1)*1000,) == arr.shape

        pool.terminate()


class TestArrayWithThreadSynchronizer(TestArray, MixinArraySyncTests):

    def create_array(self, read_only=False, **kwargs):
        store = KVStore(dict())
        cache_metadata = kwargs.pop('cache_metadata', True)
        cache_attrs = kwargs.pop('cache_attrs', True)
        init_array(store, **kwargs)
        return Array(store, synchronizer=ThreadSynchronizer(),
                     read_only=read_only, cache_metadata=cache_metadata,
                     cache_attrs=cache_attrs)

    # noinspection PyMethodMayBeStatic
    def create_pool(self):
        pool = ThreadPool(cpu_count())
        return pool

    def test_hexdigest(self):
        # Check basic 1-D array
        z = self.create_array(shape=(1050,), chunks=100, dtype='<i4')
        assert 'f710da18d45d38d4aaf2afd7fb822fdd73d02957' == z.hexdigest()

        # Check basic 1-D array with different type
        z = self.create_array(shape=(1050,), chunks=100, dtype='<f4')
        assert '1437428e69754b1e1a38bd7fc9e43669577620db' == z.hexdigest()

        # Check basic 2-D array
        z = self.create_array(shape=(20, 35,), chunks=10, dtype='<i4')
        assert '6c530b6b9d73e108cc5ee7b6be3d552cc994bdbe' == z.hexdigest()

        # Check basic 1-D array with some data
        z = self.create_array(shape=(1050,), chunks=100, dtype='<i4')
        z[200:400] = np.arange(200, 400, dtype='i4')
        assert '4c0a76fb1222498e09dcd92f7f9221d6cea8b40e' == z.hexdigest()

        # Check basic 1-D array with attributes
        z = self.create_array(shape=(1050,), chunks=100, dtype='<i4')
        z.attrs['foo'] = 'bar'
        assert '05b0663ffe1785f38d3a459dec17e57a18f254af' == z.hexdigest()


class TestArrayWithProcessSynchronizer(TestArray, MixinArraySyncTests):

    def create_array(self, read_only=False, **kwargs):
        path = tempfile.mkdtemp()
        atexit.register(atexit_rmtree, path)
        store = DirectoryStore(path)
        cache_metadata = kwargs.pop('cache_metadata', False)
        cache_attrs = kwargs.pop('cache_attrs', False)
        init_array(store, **kwargs)
        sync_path = tempfile.mkdtemp()
        atexit.register(atexit_rmtree, sync_path)
        synchronizer = ProcessSynchronizer(sync_path)
        return Array(store, synchronizer=synchronizer, read_only=read_only,
                     cache_metadata=cache_metadata, cache_attrs=cache_attrs)

    # noinspection PyMethodMayBeStatic
    def create_pool(self):
        pool = ProcessPool(processes=cpu_count())
        return pool

    def test_hexdigest(self):
        # Check basic 1-D array
        z = self.create_array(shape=(1050,), chunks=100, dtype='<i4')
        assert 'f710da18d45d38d4aaf2afd7fb822fdd73d02957' == z.hexdigest()

        # Check basic 1-D array with different type
        z = self.create_array(shape=(1050,), chunks=100, dtype='<f4')
        assert '1437428e69754b1e1a38bd7fc9e43669577620db' == z.hexdigest()

        # Check basic 2-D array
        z = self.create_array(shape=(20, 35,), chunks=10, dtype='<i4')
        assert '6c530b6b9d73e108cc5ee7b6be3d552cc994bdbe' == z.hexdigest()

        # Check basic 1-D array with some data
        z = self.create_array(shape=(1050,), chunks=100, dtype='<i4')
        z[200:400] = np.arange(200, 400, dtype='i4')
        assert '4c0a76fb1222498e09dcd92f7f9221d6cea8b40e' == z.hexdigest()

        # Check basic 1-D array with attributes
        z = self.create_array(shape=(1050,), chunks=100, dtype='<i4')
        z.attrs['foo'] = 'bar'
        assert '05b0663ffe1785f38d3a459dec17e57a18f254af' == z.hexdigest()

    def test_object_arrays_danger(self):
        # skip this one, metadata get reloaded in each process
        pass


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
        results = list(pool.map(
            _create_group,
            zip([g] * n, [str(i) for i in range(n)]),
            chunksize=1
        ))
        assert n == len(results)
        pool.close()
        pool.terminate()

        assert n == len(g)

        pool.terminate()

    def test_parallel_require_group(self):

        # setup
        g = self.create_group()
        pool = self.create_pool()

        # parallel require group
        n = 100
        results = list(pool.map(
            _require_group,
            zip([g] * n, [str(i//10) for i in range(n)]),
            chunksize=1
        ))
        assert n == len(results)
        pool.close()
        pool.terminate()

        assert n//10 == len(g)

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

    # noinspection PyMethodMayBeStatic
    def create_pool(self):
        pool = ThreadPool(cpu_count())
        return pool

    def test_synchronizer_property(self):
        g = self.create_group()
        assert isinstance(g.synchronizer, ThreadSynchronizer)


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

    # noinspection PyMethodMayBeStatic
    def create_pool(self):
        pool = ProcessPool(processes=cpu_count())
        return pool

    def test_synchronizer_property(self):
        g = self.create_group()
        assert isinstance(g.synchronizer, ProcessSynchronizer)
