# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from contextlib import contextmanager
import unittest
import tempfile
import atexit
import pickle
import json
import array
import shutil
import os
from pickle import PicklingError


import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest

try:
    import azure.storage.blob as asb
except ImportError:  # pragma: no cover
    asb = None


from zarr.storage import (init_array, array_meta_key, attrs_key, DictStore,
                          DirectoryStore, ZipStore, init_group, group_meta_key,
                          getsize, migrate_1to2, TempStore, atexit_rmtree,
                          NestedDirectoryStore, default_compressor, DBMStore,
                          LMDBStore, SQLiteStore, ABSStore, atexit_rmglob, LRUStoreCache,
                          ConsolidatedMetadataStore, MongoDBStore, RedisStore)
from zarr.meta import (decode_array_metadata, encode_array_metadata, ZARR_FORMAT,
                       decode_group_metadata, encode_group_metadata)
from zarr.compat import PY2
from zarr.codecs import AsType, Zlib, Blosc, BZ2
from zarr.errors import PermissionError, MetadataError
from zarr.hierarchy import group
from zarr.n5 import N5Store
from zarr.tests.util import CountingDict

try:
    from zarr.codecs import LZMA
except ImportError:  # pragma: no cover
    LZMA = None


@contextmanager
def does_not_raise():
    yield


class StoreTests(object):
    """Abstract store tests."""

    def create_store(self, **kwargs):  # pragma: no cover
        # implement in sub-class
        raise NotImplementedError

    def test_get_set_del_contains(self):
        store = self.create_store()

        # test __contains__, __getitem__, __setitem__
        assert 'foo' not in store
        with pytest.raises(KeyError):
            # noinspection PyStatementEffect
            store['foo']
        store['foo'] = b'bar'
        assert 'foo' in store
        assert b'bar' == store['foo']

        # test __delitem__ (optional)
        try:
            del store['foo']
        except NotImplementedError:
            pass
        else:
            assert 'foo' not in store
            with pytest.raises(KeyError):
                # noinspection PyStatementEffect
                store['foo']
            with pytest.raises(KeyError):
                # noinspection PyStatementEffect
                del store['foo']

    def test_set_invalid_content(self):
        store = self.create_store()

        with pytest.raises(TypeError):
            store['baz'] = list(range(5))

    def test_clear(self):
        store = self.create_store()
        store['foo'] = b'bar'
        store['baz'] = b'qux'
        assert len(store) == 2
        store.clear()
        assert len(store) == 0
        assert 'foo' not in store
        assert 'baz' not in store

    def test_pop(self):
        store = self.create_store()
        store['foo'] = b'bar'
        store['baz'] = b'qux'
        assert len(store) == 2
        v = store.pop('foo')
        assert v == b'bar'
        assert len(store) == 1
        v = store.pop('baz')
        assert v == b'qux'
        assert len(store) == 0
        with pytest.raises(KeyError):
            store.pop('xxx')
        v = store.pop('xxx', b'default')
        assert v == b'default'
        v = store.pop('xxx', b'')
        assert v == b''
        v = store.pop('xxx', None)
        assert v is None

    def test_popitem(self):
        store = self.create_store()
        store['foo'] = b'bar'
        k, v = store.popitem()
        assert k == 'foo'
        assert v == b'bar'
        assert len(store) == 0
        with pytest.raises(KeyError):
            store.popitem()

    def test_writeable_values(self):
        store = self.create_store()

        # __setitem__ should accept any value that implements buffer interface
        store['foo1'] = b'bar'
        store['foo2'] = bytearray(b'bar')
        store['foo3'] = array.array('B', b'bar')
        store['foo4'] = np.frombuffer(b'bar', dtype='u1')

    def test_update(self):
        store = self.create_store()
        assert 'foo' not in store
        assert 'baz' not in store
        store.update(foo=b'bar', baz=b'quux')
        assert b'bar' == store['foo']
        assert b'quux' == store['baz']

    def test_iterators(self):
        store = self.create_store()

        # test iterator methods on empty store
        assert 0 == len(store)
        assert set() == set(store)
        assert set() == set(store.keys())
        assert set() == set(store.values())
        assert set() == set(store.items())

        # setup some values
        store['a'] = b'aaa'
        store['b'] = b'bbb'
        store['c/d'] = b'ddd'
        store['c/e/f'] = b'fff'

        # test iterators on store with data
        assert 4 == len(store)
        assert {'a', 'b', 'c/d', 'c/e/f'} == set(store)
        assert {'a', 'b', 'c/d', 'c/e/f'} == set(store.keys())
        assert {b'aaa', b'bbb', b'ddd', b'fff'} == set(store.values())
        assert ({('a', b'aaa'), ('b', b'bbb'), ('c/d', b'ddd'), ('c/e/f', b'fff')} ==
                set(store.items()))

    def test_pickle(self):

        # setup store
        store = self.create_store()
        store['foo'] = b'bar'
        store['baz'] = b'quux'
        n = len(store)
        keys = sorted(store.keys())

        # round-trip through pickle
        dump = pickle.dumps(store)
        # some stores cannot be opened twice at the same time, need to close
        # store before can round-trip through pickle
        if hasattr(store, 'close'):
            store.close()
            # check can still pickle after close
            assert dump == pickle.dumps(store)
        store2 = pickle.loads(dump)

        # verify
        assert n == len(store2)
        assert keys == sorted(store2.keys())
        assert b'bar' == store2['foo']
        assert b'quux' == store2['baz']

    def test_getsize(self):
        store = self.create_store()
        if isinstance(store, dict) or hasattr(store, 'getsize'):
            assert 0 == getsize(store)
            store['foo'] = b'x'
            assert 1 == getsize(store)
            assert 1 == getsize(store, 'foo')
            store['bar'] = b'yy'
            assert 3 == getsize(store)
            assert 2 == getsize(store, 'bar')
            store['baz'] = bytearray(b'zzz')
            assert 6 == getsize(store)
            assert 3 == getsize(store, 'baz')
            store['quux'] = array.array('B', b'zzzz')
            assert 10 == getsize(store)
            assert 4 == getsize(store, 'quux')
            store['spong'] = np.frombuffer(b'zzzzz', dtype='u1')
            assert 15 == getsize(store)
            assert 5 == getsize(store, 'spong')

    # noinspection PyStatementEffect
    def test_hierarchy(self):
        # setup
        store = self.create_store()
        store['a'] = b'aaa'
        store['b'] = b'bbb'
        store['c/d'] = b'ddd'
        store['c/e/f'] = b'fff'
        store['c/e/g'] = b'ggg'

        # check keys
        assert 'a' in store
        assert 'b' in store
        assert 'c/d' in store
        assert 'c/e/f' in store
        assert 'c/e/g' in store
        assert 'c' not in store
        assert 'c/' not in store
        assert 'c/e' not in store
        assert 'c/e/' not in store
        assert 'c/d/x' not in store

        # check __getitem__
        with pytest.raises(KeyError):
            store['c']
        with pytest.raises(KeyError):
            store['c/e']
        with pytest.raises(KeyError):
            store['c/d/x']

        # test getsize (optional)
        if hasattr(store, 'getsize'):
            assert 6 == store.getsize()
            assert 3 == store.getsize('a')
            assert 3 == store.getsize('b')
            assert 3 == store.getsize('c')
            assert 3 == store.getsize('c/d')
            assert 6 == store.getsize('c/e')
            assert 3 == store.getsize('c/e/f')
            assert 3 == store.getsize('c/e/g')
            # non-existent paths
            assert 0 == store.getsize('x')
            assert 0 == store.getsize('a/x')
            assert 0 == store.getsize('c/x')
            assert 0 == store.getsize('c/x/y')
            assert 0 == store.getsize('c/d/y')
            assert 0 == store.getsize('c/d/y/z')

        # test listdir (optional)
        if hasattr(store, 'listdir'):
            assert {'a', 'b', 'c'} == set(store.listdir())
            assert {'d', 'e'} == set(store.listdir('c'))
            assert {'f', 'g'} == set(store.listdir('c/e'))
            # no exception raised if path does not exist or is leaf
            assert [] == store.listdir('x')
            assert [] == store.listdir('a/x')
            assert [] == store.listdir('c/x')
            assert [] == store.listdir('c/x/y')
            assert [] == store.listdir('c/d/y')
            assert [] == store.listdir('c/d/y/z')
            assert [] == store.listdir('c/e/f')

        # test rename (optional)
        if hasattr(store, 'rename'):
            store.rename('c/e', 'c/e2')
            assert 'c/d' in store
            assert 'c/e' not in store
            assert 'c/e/f' not in store
            assert 'c/e/g' not in store
            assert 'c/e2' not in store
            assert 'c/e2/f' in store
            assert 'c/e2/g' in store
            store.rename('c/e2', 'c/e')
            assert 'c/d' in store
            assert 'c/e2' not in store
            assert 'c/e2/f' not in store
            assert 'c/e2/g' not in store
            assert 'c/e' not in store
            assert 'c/e/f' in store
            assert 'c/e/g' in store
            store.rename('c', 'c1/c2/c3')
            assert 'a' in store
            assert 'c' not in store
            assert 'c/d' not in store
            assert 'c/e' not in store
            assert 'c/e/f' not in store
            assert 'c/e/g' not in store
            assert 'c1' not in store
            assert 'c1/c2' not in store
            assert 'c1/c2/c3' not in store
            assert 'c1/c2/c3/d' in store
            assert 'c1/c2/c3/e' not in store
            assert 'c1/c2/c3/e/f' in store
            assert 'c1/c2/c3/e/g' in store
            store.rename('c1/c2/c3', 'c')
            assert 'c' not in store
            assert 'c/d' in store
            assert 'c/e' not in store
            assert 'c/e/f' in store
            assert 'c/e/g' in store
            assert 'c1' not in store
            assert 'c1/c2' not in store
            assert 'c1/c2/c3' not in store
            assert 'c1/c2/c3/d' not in store
            assert 'c1/c2/c3/e' not in store
            assert 'c1/c2/c3/e/f' not in store
            assert 'c1/c2/c3/e/g' not in store

        # test rmdir (optional)
        if hasattr(store, 'rmdir'):
            store.rmdir('c/e')
            assert 'c/d' in store
            assert 'c/e/f' not in store
            assert 'c/e/g' not in store
            store.rmdir('c')
            assert 'c/d' not in store
            store.rmdir()
            assert 'a' not in store
            assert 'b' not in store
            store['a'] = b'aaa'
            store['c/d'] = b'ddd'
            store['c/e/f'] = b'fff'
            # no exceptions raised if path does not exist or is leaf
            store.rmdir('x')
            store.rmdir('a/x')
            store.rmdir('c/x')
            store.rmdir('c/x/y')
            store.rmdir('c/d/y')
            store.rmdir('c/d/y/z')
            store.rmdir('c/e/f')
            assert 'a' in store
            assert 'c/d' in store
            assert 'c/e/f' in store

    def test_init_array(self):
        store = self.create_store()
        init_array(store, shape=1000, chunks=100)

        # check metadata
        assert array_meta_key in store
        meta = decode_array_metadata(store[array_meta_key])
        assert ZARR_FORMAT == meta['zarr_format']
        assert (1000,) == meta['shape']
        assert (100,) == meta['chunks']
        assert np.dtype(None) == meta['dtype']
        assert default_compressor.get_config() == meta['compressor']
        assert meta['fill_value'] is None

    def test_init_array_overwrite(self):
        self._test_init_array_overwrite('F')

    def test_init_array_overwrite_path(self):
        self._test_init_array_overwrite_path('F')

    def test_init_array_overwrite_chunk_store(self):
        self._test_init_array_overwrite_chunk_store('F')

    def test_init_group_overwrite(self):
        self._test_init_group_overwrite('F')

    def test_init_group_overwrite_path(self):
        self._test_init_group_overwrite_path('F')

    def test_init_group_overwrite_chunk_store(self):
        self._test_init_group_overwrite_chunk_store('F')

    def _test_init_array_overwrite(self, order):
        # setup
        store = self.create_store()
        store[array_meta_key] = encode_array_metadata(
            dict(shape=(2000,),
                 chunks=(200,),
                 dtype=np.dtype('u1'),
                 compressor=Zlib(1).get_config(),
                 fill_value=0,
                 order=order,
                 filters=None)
        )

        # don't overwrite (default)
        with pytest.raises(ValueError):
            init_array(store, shape=1000, chunks=100)

        # do overwrite
        try:
            init_array(store, shape=1000, chunks=100, dtype='i4',
                       overwrite=True)
        except NotImplementedError:
            pass
        else:
            assert array_meta_key in store
            meta = decode_array_metadata(store[array_meta_key])
            assert ZARR_FORMAT == meta['zarr_format']
            assert (1000,) == meta['shape']
            assert (100,) == meta['chunks']
            assert np.dtype('i4') == meta['dtype']

    def test_init_array_path(self):
        path = 'foo/bar'
        store = self.create_store()
        init_array(store, shape=1000, chunks=100, path=path)

        # check metadata
        key = path + '/' + array_meta_key
        assert key in store
        meta = decode_array_metadata(store[key])
        assert ZARR_FORMAT == meta['zarr_format']
        assert (1000,) == meta['shape']
        assert (100,) == meta['chunks']
        assert np.dtype(None) == meta['dtype']
        assert default_compressor.get_config() == meta['compressor']
        assert meta['fill_value'] is None

    def _test_init_array_overwrite_path(self, order):
        # setup
        path = 'foo/bar'
        store = self.create_store()
        meta = dict(shape=(2000,),
                    chunks=(200,),
                    dtype=np.dtype('u1'),
                    compressor=Zlib(1).get_config(),
                    fill_value=0,
                    order=order,
                    filters=None)
        store[array_meta_key] = encode_array_metadata(meta)
        store[path + '/' + array_meta_key] = encode_array_metadata(meta)

        # don't overwrite
        with pytest.raises(ValueError):
            init_array(store, shape=1000, chunks=100, path=path)

        # do overwrite
        try:
            init_array(store, shape=1000, chunks=100, dtype='i4', path=path,
                       overwrite=True)
        except NotImplementedError:
            pass
        else:
            assert group_meta_key in store
            assert array_meta_key not in store
            assert (path + '/' + array_meta_key) in store
            # should have been overwritten
            meta = decode_array_metadata(store[path + '/' + array_meta_key])
            assert ZARR_FORMAT == meta['zarr_format']
            assert (1000,) == meta['shape']
            assert (100,) == meta['chunks']
            assert np.dtype('i4') == meta['dtype']

    def test_init_array_overwrite_group(self):
        # setup
        path = 'foo/bar'
        store = self.create_store()
        store[path + '/' + group_meta_key] = encode_group_metadata()

        # don't overwrite
        with pytest.raises(ValueError):
            init_array(store, shape=1000, chunks=100, path=path)

        # do overwrite
        try:
            init_array(store, shape=1000, chunks=100, dtype='i4', path=path,
                       overwrite=True)
        except NotImplementedError:
            pass
        else:
            assert (path + '/' + group_meta_key) not in store
            assert (path + '/' + array_meta_key) in store
            meta = decode_array_metadata(store[path + '/' + array_meta_key])
            assert ZARR_FORMAT == meta['zarr_format']
            assert (1000,) == meta['shape']
            assert (100,) == meta['chunks']
            assert np.dtype('i4') == meta['dtype']

    def _test_init_array_overwrite_chunk_store(self, order):
        # setup
        store = self.create_store()
        chunk_store = self.create_store()
        store[array_meta_key] = encode_array_metadata(
            dict(shape=(2000,),
                 chunks=(200,),
                 dtype=np.dtype('u1'),
                 compressor=None,
                 fill_value=0,
                 filters=None,
                 order=order)
        )
        chunk_store['0'] = b'aaa'
        chunk_store['1'] = b'bbb'

        # don't overwrite (default)
        with pytest.raises(ValueError):
            init_array(store, shape=1000, chunks=100, chunk_store=chunk_store)

        # do overwrite
        try:
            init_array(store, shape=1000, chunks=100, dtype='i4',
                       overwrite=True, chunk_store=chunk_store)
        except NotImplementedError:
            pass
        else:
            assert array_meta_key in store
            meta = decode_array_metadata(store[array_meta_key])
            assert ZARR_FORMAT == meta['zarr_format']
            assert (1000,) == meta['shape']
            assert (100,) == meta['chunks']
            assert np.dtype('i4') == meta['dtype']
            assert '0' not in chunk_store
            assert '1' not in chunk_store

    def test_init_array_compat(self):
        store = self.create_store()
        init_array(store, shape=1000, chunks=100, compressor='none')
        meta = decode_array_metadata(store[array_meta_key])
        assert meta['compressor'] is None

    def test_init_group(self):
        store = self.create_store()
        init_group(store)

        # check metadata
        assert group_meta_key in store
        meta = decode_group_metadata(store[group_meta_key])
        assert ZARR_FORMAT == meta['zarr_format']

    def _test_init_group_overwrite(self, order):
        # setup
        store = self.create_store()
        store[array_meta_key] = encode_array_metadata(
            dict(shape=(2000,),
                 chunks=(200,),
                 dtype=np.dtype('u1'),
                 compressor=None,
                 fill_value=0,
                 order=order,
                 filters=None)
        )

        # don't overwrite array (default)
        with pytest.raises(ValueError):
            init_group(store)

        # do overwrite
        try:
            init_group(store, overwrite=True)
        except NotImplementedError:
            pass
        else:
            assert array_meta_key not in store
            assert group_meta_key in store
            meta = decode_group_metadata(store[group_meta_key])
            assert ZARR_FORMAT == meta['zarr_format']

        # don't overwrite group
        with pytest.raises(ValueError):
            init_group(store)

    def _test_init_group_overwrite_path(self, order):
        # setup
        path = 'foo/bar'
        store = self.create_store()
        meta = dict(shape=(2000,),
                    chunks=(200,),
                    dtype=np.dtype('u1'),
                    compressor=None,
                    fill_value=0,
                    order=order,
                    filters=None)
        store[array_meta_key] = encode_array_metadata(meta)
        store[path + '/' + array_meta_key] = encode_array_metadata(meta)

        # don't overwrite
        with pytest.raises(ValueError):
            init_group(store, path=path)

        # do overwrite
        try:
            init_group(store, overwrite=True, path=path)
        except NotImplementedError:
            pass
        else:
            assert array_meta_key not in store
            assert group_meta_key in store
            assert (path + '/' + array_meta_key) not in store
            assert (path + '/' + group_meta_key) in store
            # should have been overwritten
            meta = decode_group_metadata(store[path + '/' + group_meta_key])
            assert ZARR_FORMAT == meta['zarr_format']

    def _test_init_group_overwrite_chunk_store(self, order):
        # setup
        store = self.create_store()
        chunk_store = self.create_store()
        store[array_meta_key] = encode_array_metadata(
            dict(shape=(2000,),
                 chunks=(200,),
                 dtype=np.dtype('u1'),
                 compressor=None,
                 fill_value=0,
                 filters=None,
                 order=order)
        )
        chunk_store['foo'] = b'bar'
        chunk_store['baz'] = b'quux'

        # don't overwrite array (default)
        with pytest.raises(ValueError):
            init_group(store, chunk_store=chunk_store)

        # do overwrite
        try:
            init_group(store, overwrite=True, chunk_store=chunk_store)
        except NotImplementedError:
            pass
        else:
            assert array_meta_key not in store
            assert group_meta_key in store
            meta = decode_group_metadata(store[group_meta_key])
            assert ZARR_FORMAT == meta['zarr_format']
            assert 'foo' not in chunk_store
            assert 'baz' not in chunk_store

        # don't overwrite group
        with pytest.raises(ValueError):
            init_group(store)


class TestMappingStore(StoreTests, unittest.TestCase):

    def create_store(self):
        return dict()

    def test_set_invalid_content(self):
        # Generic mappings support non-buffer types
        pass


def setdel_hierarchy_checks(store):
    # these tests are for stores that are aware of hierarchy levels; this
    # behaviour is not stricly required by Zarr but these tests are included
    # to define behaviour of DictStore and DirectoryStore classes

    # check __setitem__ and __delitem__ blocked by leaf

    store['a/b'] = b'aaa'
    with pytest.raises(KeyError):
        store['a/b/c'] = b'xxx'
    with pytest.raises(KeyError):
        del store['a/b/c']

    store['d'] = b'ddd'
    with pytest.raises(KeyError):
        store['d/e/f'] = b'xxx'
    with pytest.raises(KeyError):
        del store['d/e/f']

    # test __setitem__ overwrite level
    store['x/y/z'] = b'xxx'
    store['x/y'] = b'yyy'
    assert b'yyy' == store['x/y']
    assert 'x/y/z' not in store
    store['x'] = b'zzz'
    assert b'zzz' == store['x']
    assert 'x/y' not in store

    # test __delitem__ overwrite level
    store['r/s/t'] = b'xxx'
    del store['r/s']
    assert 'r/s/t' not in store
    store['r/s'] = b'xxx'
    del store['r']
    assert 'r/s' not in store


class TestDictStore(StoreTests, unittest.TestCase):

    def create_store(self):
        return DictStore()

    def test_store_contains_bytes(self):
        store = self.create_store()
        store['foo'] = np.array([97, 98, 99, 100, 101], dtype=np.uint8)
        assert store['foo'] == b'abcde'

    def test_setdel(self):
        store = self.create_store()
        setdel_hierarchy_checks(store)


class TestDirectoryStore(StoreTests, unittest.TestCase):

    def create_store(self):
        path = tempfile.mkdtemp()
        atexit.register(atexit_rmtree, path)
        store = DirectoryStore(path)
        return store

    def test_filesystem_path(self):

        # test behaviour with path that does not exist
        path = 'data/store'
        if os.path.exists(path):
            shutil.rmtree(path)
        store = DirectoryStore(path)
        # should only be created on demand
        assert not os.path.exists(path)
        store['foo'] = b'bar'
        assert os.path.isdir(path)

        # test behaviour with file path
        with tempfile.NamedTemporaryFile() as f:
            with pytest.raises(ValueError):
                DirectoryStore(f.name)

    def test_pickle_ext(self):
        store = self.create_store()
        store2 = pickle.loads(pickle.dumps(store))

        # check path is preserved
        assert store.path == store2.path

        # check point to same underlying directory
        assert 'xxx' not in store
        store2['xxx'] = b'yyy'
        assert b'yyy' == store['xxx']

    def test_setdel(self):
        store = self.create_store()
        setdel_hierarchy_checks(store)


class TestNestedDirectoryStore(TestDirectoryStore, unittest.TestCase):

    def create_store(self):
        path = tempfile.mkdtemp()
        atexit.register(atexit_rmtree, path)
        store = NestedDirectoryStore(path)
        return store

    def test_chunk_nesting(self):
        store = self.create_store()
        # any path where last segment looks like a chunk key gets special handling
        store['0.0'] = b'xxx'
        assert b'xxx' == store['0.0']
        assert b'xxx' == store['0/0']
        store['foo/10.20.30'] = b'yyy'
        assert b'yyy' == store['foo/10.20.30']
        assert b'yyy' == store['foo/10/20/30']
        store['42'] = b'zzz'
        assert b'zzz' == store['42']


class TestN5Store(TestNestedDirectoryStore, unittest.TestCase):

    def create_store(self):
        path = tempfile.mkdtemp(suffix='.n5')
        atexit.register(atexit_rmtree, path)
        store = N5Store(path)
        return store

    def test_equal(self):
        store_a = self.create_store()
        store_b = N5Store(store_a.path)
        assert store_a == store_b

    def test_chunk_nesting(self):
        store = self.create_store()
        store['0.0'] = b'xxx'
        assert '0.0' in store
        assert b'xxx' == store['0.0']
        assert b'xxx' == store['0/0']
        store['foo/10.20.30'] = b'yyy'
        assert 'foo/10.20.30' in store
        assert b'yyy' == store['foo/10.20.30']
        # N5 reverses axis order
        assert b'yyy' == store['foo/30/20/10']
        store['42'] = b'zzz'
        assert '42' in store
        assert b'zzz' == store['42']

    def test_init_array(self):
        store = self.create_store()
        init_array(store, shape=1000, chunks=100)

        # check metadata
        assert array_meta_key in store
        meta = decode_array_metadata(store[array_meta_key])
        assert ZARR_FORMAT == meta['zarr_format']
        assert (1000,) == meta['shape']
        assert (100,) == meta['chunks']
        assert np.dtype(None) == meta['dtype']
        # N5Store wraps the actual compressor
        compressor_config = meta['compressor']['compressor_config']
        assert default_compressor.get_config() == compressor_config
        # N5Store always has a fill value of 0
        assert meta['fill_value'] == 0

    def test_init_array_path(self):
        path = 'foo/bar'
        store = self.create_store()
        init_array(store, shape=1000, chunks=100, path=path)

        # check metadata
        key = path + '/' + array_meta_key
        assert key in store
        meta = decode_array_metadata(store[key])
        assert ZARR_FORMAT == meta['zarr_format']
        assert (1000,) == meta['shape']
        assert (100,) == meta['chunks']
        assert np.dtype(None) == meta['dtype']
        # N5Store wraps the actual compressor
        compressor_config = meta['compressor']['compressor_config']
        assert default_compressor.get_config() == compressor_config
        # N5Store always has a fill value of 0
        assert meta['fill_value'] == 0

    def test_init_array_compat(self):
        store = self.create_store()
        init_array(store, shape=1000, chunks=100, compressor='none')
        meta = decode_array_metadata(store[array_meta_key])
        # N5Store wraps the actual compressor
        compressor_config = meta['compressor']['compressor_config']
        assert compressor_config is None

    def test_init_array_overwrite(self):
        self._test_init_array_overwrite('C')

    def test_init_array_overwrite_path(self):
        self._test_init_array_overwrite_path('C')

    def test_init_array_overwrite_chunk_store(self):
        self._test_init_array_overwrite_chunk_store('C')

    def test_init_group_overwrite(self):
        self._test_init_group_overwrite('C')

    def test_init_group_overwrite_path(self):
        self._test_init_group_overwrite_path('C')

    def test_init_group_overwrite_chunk_store(self):
        self._test_init_group_overwrite_chunk_store('C')

    def test_init_group(self):
        store = self.create_store()
        init_group(store)

        # check metadata
        assert group_meta_key in store
        assert group_meta_key in store.listdir()
        assert group_meta_key in store.listdir('')
        meta = decode_group_metadata(store[group_meta_key])
        assert ZARR_FORMAT == meta['zarr_format']

    def test_filters(self):
        all_filters, all_errors = zip(*[
            (None, does_not_raise()),
            ([], does_not_raise()),
            ([AsType('f4', 'f8')], pytest.raises(ValueError)),
        ])
        for filters, error in zip(all_filters, all_errors):
            store = self.create_store()
            with error:
                init_array(store, shape=1000, chunks=100, filters=filters)


class TestTempStore(StoreTests, unittest.TestCase):

    def create_store(self):
        return TempStore()

    def test_setdel(self):
        store = self.create_store()
        setdel_hierarchy_checks(store)


class TestZipStore(StoreTests, unittest.TestCase):

    def create_store(self):
        path = tempfile.mktemp(suffix='.zip')
        atexit.register(os.remove, path)
        store = ZipStore(path, mode='w')
        return store

    def test_mode(self):
        with ZipStore('data/store.zip', mode='w') as store:
            store['foo'] = b'bar'
        store = ZipStore('data/store.zip', mode='r')
        with pytest.raises(PermissionError):
            store['foo'] = b'bar'
        with pytest.raises(PermissionError):
            store.clear()

    def test_flush(self):
        store = ZipStore('data/store.zip', mode='w')
        store['foo'] = b'bar'
        store.flush()
        assert store['foo'] == b'bar'
        store.close()

        store = ZipStore('data/store.zip', mode='r')
        store.flush()  # no-op

    def test_context_manager(self):
        with self.create_store() as store:
            store['foo'] = b'bar'
            store['baz'] = b'qux'
            assert 2 == len(store)

    def test_pop(self):
        # override because not implemented
        store = self.create_store()
        store['foo'] = b'bar'
        with pytest.raises(NotImplementedError):
            store.pop('foo')

    def test_popitem(self):
        # override because not implemented
        store = self.create_store()
        store['foo'] = b'bar'
        with pytest.raises(NotImplementedError):
            store.popitem()


class TestDBMStore(StoreTests, unittest.TestCase):

    def create_store(self):
        path = tempfile.mktemp(suffix='.anydbm')
        atexit.register(atexit_rmglob, path + '*')
        # create store using default dbm implementation
        store = DBMStore(path, flag='n')
        return store

    def test_context_manager(self):
        with self.create_store() as store:
            store['foo'] = b'bar'
            store['baz'] = b'qux'
            assert 2 == len(store)


class TestDBMStoreDumb(TestDBMStore):

    def create_store(self):
        path = tempfile.mktemp(suffix='.dumbdbm')
        atexit.register(atexit_rmglob, path + '*')
        if PY2:  # pragma: py3 no cover
            import dumbdbm
        else:  # pragma: py2 no cover
            import dbm.dumb as dumbdbm
        store = DBMStore(path, flag='n', open=dumbdbm.open)
        return store


try:
    if PY2:  # pragma: py3 no cover
        import gdbm
    else:  # pragma: py2 no cover
        import dbm.gnu as gdbm
except ImportError:  # pragma: no cover
    gdbm = None


@unittest.skipIf(gdbm is None, 'gdbm is not installed')
class TestDBMStoreGnu(TestDBMStore):

    def create_store(self):
        path = tempfile.mktemp(suffix='.gdbm')
        atexit.register(os.remove, path)
        store = DBMStore(path, flag='n', open=gdbm.open, write_lock=False)
        return store


if not PY2:  # pragma: py2 no cover
    try:
        import dbm.ndbm as ndbm
    except ImportError:  # pragma: no cover
        ndbm = None

    @unittest.skipIf(ndbm is None, 'ndbm is not installed')
    class TestDBMStoreNDBM(TestDBMStore):

        def create_store(self):
            path = tempfile.mktemp(suffix='.ndbm')
            atexit.register(atexit_rmglob, path + '*')
            store = DBMStore(path, flag='n', open=ndbm.open)
            return store


try:
    import bsddb3
except ImportError:  # pragma: no cover
    bsddb3 = None


@unittest.skipIf(bsddb3 is None, 'bsddb3 is not installed')
class TestDBMStoreBerkeleyDB(TestDBMStore):

    def create_store(self):
        path = tempfile.mktemp(suffix='.dbm')
        atexit.register(os.remove, path)
        store = DBMStore(path, flag='n', open=bsddb3.btopen, write_lock=False)
        return store


try:
    import lmdb
except ImportError:  # pragma: no cover
    lmdb = None


@unittest.skipIf(lmdb is None, 'lmdb is not installed')
class TestLMDBStore(StoreTests, unittest.TestCase):

    def create_store(self):
        path = tempfile.mktemp(suffix='.lmdb')
        atexit.register(atexit_rmtree, path)
        if PY2:  # pragma: py3 no cover
            # don't use buffers, otherwise would have to rewrite tests as bytes and
            # buffer don't compare equal in PY2
            buffers = False
        else:  # pragma: py2 no cover
            buffers = True
        store = LMDBStore(path, buffers=buffers)
        return store

    def test_context_manager(self):
        with self.create_store() as store:
            store['foo'] = b'bar'
            store['baz'] = b'qux'
            assert 2 == len(store)


try:
    import sqlite3
except ImportError:  # pragma: no cover
    sqlite3 = None

try:
    import pymongo
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    try:
        client = pymongo.MongoClient(host='127.0.0.1',
                                     serverSelectionTimeoutMS=1e3)
        client.server_info()
    except (ConnectionFailure, ServerSelectionTimeoutError):  # pragma: no cover
        pymongo = None
except ImportError:  # pragma: no cover
    pymongo = None

try:
    import redis
    from redis import ConnectionError
    try:
        rs = redis.Redis("localhost", port=6379)
        rs.ping()
    except ConnectionError:  # pragma: no cover
        redis = None
except ImportError:  # pragma: no cover
    redis = None


@unittest.skipIf(sqlite3 is None, 'python built without sqlite')
class TestSQLiteStore(StoreTests, unittest.TestCase):

    def create_store(self):
        path = tempfile.mktemp(suffix='.db')
        atexit.register(atexit_rmtree, path)
        store = SQLiteStore(path)
        return store


@unittest.skipIf(sqlite3 is None, 'python built without sqlite')
class TestSQLiteStoreInMemory(TestSQLiteStore, unittest.TestCase):

    def create_store(self):
        store = SQLiteStore(':memory:')
        return store

    def test_pickle(self):

        # setup store
        store = self.create_store()
        store['foo'] = b'bar'
        store['baz'] = b'quux'

        # round-trip through pickle
        with pytest.raises(PicklingError):
            pickle.dumps(store)


@unittest.skipIf(pymongo is None, 'test requires pymongo')
class TestMongoDBStore(StoreTests, unittest.TestCase):

    def create_store(self):
        store = MongoDBStore(host='127.0.0.1', database='zarr_tests',
                             collection='zarr_tests')
        # start with an empty store
        store.clear()
        return store


@unittest.skipIf(redis is None, 'test requires redis')
class TestRedisStore(StoreTests, unittest.TestCase):

    def create_store(self):
        # TODO: this is the default host for Redis on Travis,
        # we probably want to generalize this though
        store = RedisStore(host='localhost', port=6379)
        # start with an empty store
        store.clear()
        return store


class TestLRUStoreCache(StoreTests, unittest.TestCase):

    def create_store(self):
        return LRUStoreCache(dict(), max_size=2**27)

    def test_cache_values_no_max_size(self):

        # setup store
        store = CountingDict()
        store['foo'] = b'xxx'
        store['bar'] = b'yyy'
        assert 0 == store.counter['__getitem__', 'foo']
        assert 1 == store.counter['__setitem__', 'foo']
        assert 0 == store.counter['__getitem__', 'bar']
        assert 1 == store.counter['__setitem__', 'bar']

        # setup cache
        cache = LRUStoreCache(store, max_size=None)
        assert 0 == cache.hits
        assert 0 == cache.misses

        # test first __getitem__, cache miss
        assert b'xxx' == cache['foo']
        assert 1 == store.counter['__getitem__', 'foo']
        assert 1 == store.counter['__setitem__', 'foo']
        assert 0 == cache.hits
        assert 1 == cache.misses

        # test second __getitem__, cache hit
        assert b'xxx' == cache['foo']
        assert 1 == store.counter['__getitem__', 'foo']
        assert 1 == store.counter['__setitem__', 'foo']
        assert 1 == cache.hits
        assert 1 == cache.misses

        # test __setitem__, __getitem__
        cache['foo'] = b'zzz'
        assert 1 == store.counter['__getitem__', 'foo']
        assert 2 == store.counter['__setitem__', 'foo']
        # should be a cache hit
        assert b'zzz' == cache['foo']
        assert 1 == store.counter['__getitem__', 'foo']
        assert 2 == store.counter['__setitem__', 'foo']
        assert 2 == cache.hits
        assert 1 == cache.misses

        # manually invalidate all cached values
        cache.invalidate_values()
        assert b'zzz' == cache['foo']
        assert 2 == store.counter['__getitem__', 'foo']
        assert 2 == store.counter['__setitem__', 'foo']
        cache.invalidate()
        assert b'zzz' == cache['foo']
        assert 3 == store.counter['__getitem__', 'foo']
        assert 2 == store.counter['__setitem__', 'foo']

        # test __delitem__
        del cache['foo']
        with pytest.raises(KeyError):
            # noinspection PyStatementEffect
            cache['foo']
        with pytest.raises(KeyError):
            # noinspection PyStatementEffect
            store['foo']

        # verify other keys untouched
        assert 0 == store.counter['__getitem__', 'bar']
        assert 1 == store.counter['__setitem__', 'bar']

    def test_cache_values_with_max_size(self):

        # setup store
        store = CountingDict()
        store['foo'] = b'xxx'
        store['bar'] = b'yyy'
        assert 0 == store.counter['__getitem__', 'foo']
        assert 0 == store.counter['__getitem__', 'bar']
        # setup cache - can only hold one item
        cache = LRUStoreCache(store, max_size=5)
        assert 0 == cache.hits
        assert 0 == cache.misses

        # test first 'foo' __getitem__, cache miss
        assert b'xxx' == cache['foo']
        assert 1 == store.counter['__getitem__', 'foo']
        assert 0 == cache.hits
        assert 1 == cache.misses

        # test second 'foo' __getitem__, cache hit
        assert b'xxx' == cache['foo']
        assert 1 == store.counter['__getitem__', 'foo']
        assert 1 == cache.hits
        assert 1 == cache.misses

        # test first 'bar' __getitem__, cache miss
        assert b'yyy' == cache['bar']
        assert 1 == store.counter['__getitem__', 'bar']
        assert 1 == cache.hits
        assert 2 == cache.misses

        # test second 'bar' __getitem__, cache hit
        assert b'yyy' == cache['bar']
        assert 1 == store.counter['__getitem__', 'bar']
        assert 2 == cache.hits
        assert 2 == cache.misses

        # test 'foo' __getitem__, should have been evicted, cache miss
        assert b'xxx' == cache['foo']
        assert 2 == store.counter['__getitem__', 'foo']
        assert 2 == cache.hits
        assert 3 == cache.misses

        # test 'bar' __getitem__, should have been evicted, cache miss
        assert b'yyy' == cache['bar']
        assert 2 == store.counter['__getitem__', 'bar']
        assert 2 == cache.hits
        assert 4 == cache.misses

        # setup store
        store = CountingDict()
        store['foo'] = b'xxx'
        store['bar'] = b'yyy'
        assert 0 == store.counter['__getitem__', 'foo']
        assert 0 == store.counter['__getitem__', 'bar']
        # setup cache - can hold two items
        cache = LRUStoreCache(store, max_size=6)
        assert 0 == cache.hits
        assert 0 == cache.misses

        # test first 'foo' __getitem__, cache miss
        assert b'xxx' == cache['foo']
        assert 1 == store.counter['__getitem__', 'foo']
        assert 0 == cache.hits
        assert 1 == cache.misses

        # test second 'foo' __getitem__, cache hit
        assert b'xxx' == cache['foo']
        assert 1 == store.counter['__getitem__', 'foo']
        assert 1 == cache.hits
        assert 1 == cache.misses

        # test first 'bar' __getitem__, cache miss
        assert b'yyy' == cache['bar']
        assert 1 == store.counter['__getitem__', 'bar']
        assert 1 == cache.hits
        assert 2 == cache.misses

        # test second 'bar' __getitem__, cache hit
        assert b'yyy' == cache['bar']
        assert 1 == store.counter['__getitem__', 'bar']
        assert 2 == cache.hits
        assert 2 == cache.misses

        # test 'foo' __getitem__, should still be cached
        assert b'xxx' == cache['foo']
        assert 1 == store.counter['__getitem__', 'foo']
        assert 3 == cache.hits
        assert 2 == cache.misses

        # test 'bar' __getitem__, should still be cached
        assert b'yyy' == cache['bar']
        assert 1 == store.counter['__getitem__', 'bar']
        assert 4 == cache.hits
        assert 2 == cache.misses

    def test_cache_keys(self):

        # setup
        store = CountingDict()
        store['foo'] = b'xxx'
        store['bar'] = b'yyy'
        assert 0 == store.counter['__contains__', 'foo']
        assert 0 == store.counter['__iter__']
        assert 0 == store.counter['keys']
        cache = LRUStoreCache(store, max_size=None)

        # keys should be cached on first call
        keys = sorted(cache.keys())
        assert keys == ['bar', 'foo']
        assert 1 == store.counter['keys']
        # keys should now be cached
        assert keys == sorted(cache.keys())
        assert 1 == store.counter['keys']
        assert 'foo' in cache
        assert 0 == store.counter['__contains__', 'foo']
        assert keys == sorted(cache)
        assert 0 == store.counter['__iter__']
        assert 1 == store.counter['keys']

        # cache should be cleared if store is modified - crude but simple for now
        cache['baz'] = b'zzz'
        keys = sorted(cache.keys())
        assert keys == ['bar', 'baz', 'foo']
        assert 2 == store.counter['keys']
        # keys should now be cached
        assert keys == sorted(cache.keys())
        assert 2 == store.counter['keys']

        # manually invalidate keys
        cache.invalidate_keys()
        keys = sorted(cache.keys())
        assert keys == ['bar', 'baz', 'foo']
        assert 3 == store.counter['keys']
        assert 0 == store.counter['__contains__', 'foo']
        assert 0 == store.counter['__iter__']
        cache.invalidate_keys()
        keys = sorted(cache)
        assert keys == ['bar', 'baz', 'foo']
        assert 4 == store.counter['keys']
        assert 0 == store.counter['__contains__', 'foo']
        assert 0 == store.counter['__iter__']
        cache.invalidate_keys()
        assert 'foo' in cache
        assert 5 == store.counter['keys']
        assert 0 == store.counter['__contains__', 'foo']
        assert 0 == store.counter['__iter__']

        # check these would get counted if called directly
        assert 'foo' in store
        assert 1 == store.counter['__contains__', 'foo']
        assert keys == sorted(store)
        assert 1 == store.counter['__iter__']


def test_getsize():
    store = dict()
    store['foo'] = b'aaa'
    store['bar'] = b'bbbb'
    store['baz/quux'] = b'ccccc'
    assert 7 == getsize(store)
    assert 5 == getsize(store, 'baz')

    store = dict()
    store['boo'] = None
    assert -1 == getsize(store)


def test_migrate_1to2():
    from zarr import meta_v1

    # N.B., version 1 did not support hierarchies, so we only have to be
    # concerned about migrating a single array at the root of the store

    # setup
    store = dict()
    meta = dict(
        shape=(100,),
        chunks=(10,),
        dtype=np.dtype('f4'),
        compression='zlib',
        compression_opts=1,
        fill_value=None,
        order='C'
    )
    meta_json = meta_v1.encode_metadata(meta)
    store['meta'] = meta_json
    store['attrs'] = json.dumps(dict()).encode('ascii')

    # run migration
    migrate_1to2(store)

    # check results
    assert 'meta' not in store
    assert array_meta_key in store
    assert 'attrs' not in store
    assert attrs_key in store
    meta_migrated = decode_array_metadata(store[array_meta_key])
    assert 2 == meta_migrated['zarr_format']

    # preserved fields
    for f in 'shape', 'chunks', 'dtype', 'fill_value', 'order':
        assert meta[f] == meta_migrated[f]

    # migrate should have added empty filters field
    assert meta_migrated['filters'] is None

    # check compression and compression_opts migrated to compressor
    assert 'compression' not in meta_migrated
    assert 'compression_opts' not in meta_migrated
    assert meta_migrated['compressor'] == Zlib(1).get_config()

    # check dict compression_opts
    store = dict()
    meta['compression'] = 'blosc'
    meta['compression_opts'] = dict(cname='lz4', clevel=5, shuffle=1)
    meta_json = meta_v1.encode_metadata(meta)
    store['meta'] = meta_json
    store['attrs'] = json.dumps(dict()).encode('ascii')
    migrate_1to2(store)
    meta_migrated = decode_array_metadata(store[array_meta_key])
    assert 'compression' not in meta_migrated
    assert 'compression_opts' not in meta_migrated
    assert (meta_migrated['compressor'] ==
            Blosc(cname='lz4', clevel=5, shuffle=1).get_config())

    # check 'none' compression is migrated to None (null in JSON)
    store = dict()
    meta['compression'] = 'none'
    meta_json = meta_v1.encode_metadata(meta)
    store['meta'] = meta_json
    store['attrs'] = json.dumps(dict()).encode('ascii')
    migrate_1to2(store)
    meta_migrated = decode_array_metadata(store[array_meta_key])
    assert 'compression' not in meta_migrated
    assert 'compression_opts' not in meta_migrated
    assert meta_migrated['compressor'] is None


def test_format_compatibility():

    # This test is intended to catch any unintended changes that break the ability to
    # read data stored with a previous minor version (which should be format-compatible).

    # fixture data
    fixture = group(store=DirectoryStore('fixture'))

    # set seed to get consistent random data
    np.random.seed(42)

    arrays_chunks = [
        (np.arange(1111, dtype='i1'), 100),
        (np.arange(1111, dtype='i2'), 100),
        (np.arange(1111, dtype='i4'), 100),
        (np.arange(1111, dtype='i8'), 1000),
        (np.random.randint(0, 200, size=2222, dtype='u1'), 100),
        (np.random.randint(0, 2000, size=2222, dtype='u2'), 100),
        (np.random.randint(0, 2000, size=2222, dtype='u4'), 100),
        (np.random.randint(0, 2000, size=2222, dtype='u8'), 100),
        (np.linspace(0, 1, 3333, dtype='f2'), 100),
        (np.linspace(0, 1, 3333, dtype='f4'), 100),
        (np.linspace(0, 1, 3333, dtype='f8'), 100),
        (np.random.normal(loc=0, scale=1, size=4444).astype('f2'), 100),
        (np.random.normal(loc=0, scale=1, size=4444).astype('f4'), 100),
        (np.random.normal(loc=0, scale=1, size=4444).astype('f8'), 100),
        (np.random.choice([b'A', b'C', b'G', b'T'],
                          size=5555, replace=True).astype('S'), 100),
        (np.random.choice(['foo', 'bar', 'baz', 'quux'],
                          size=5555, replace=True).astype('U'), 100),
        (np.random.choice([0, 1/3, 1/7, 1/9, np.nan],
                          size=5555, replace=True).astype('f8'), 100),
        (np.random.randint(0, 2, size=5555, dtype=bool), 100),
        (np.arange(20000, dtype='i4').reshape(2000, 10, order='C'), (100, 3)),
        (np.arange(20000, dtype='i4').reshape(200, 100, order='F'), (100, 30)),
        (np.arange(20000, dtype='i4').reshape(200, 10, 10, order='C'), (100, 3, 3)),
        (np.arange(20000, dtype='i4').reshape(20, 100, 10, order='F'), (10, 30, 3)),
        (np.arange(20000, dtype='i4').reshape(20, 10, 10, 10, order='C'), (10, 3, 3, 3)),
        (np.arange(20000, dtype='i4').reshape(20, 10, 10, 10, order='F'), (10, 3, 3, 3)),
    ]

    compressors = [
        None,
        Zlib(level=1),
        BZ2(level=1),
        Blosc(cname='zstd', clevel=1, shuffle=0),
        Blosc(cname='zstd', clevel=1, shuffle=1),
        Blosc(cname='zstd', clevel=1, shuffle=2),
        Blosc(cname='lz4', clevel=1, shuffle=0),
    ]

    for i, (arr, chunks) in enumerate(arrays_chunks):

        if arr.flags.f_contiguous:
            order = 'F'
        else:
            order = 'C'

        for j, compressor in enumerate(compressors):
            path = '{}/{}'.format(i, j)

            if path not in fixture:  # pragma: no cover
                # store the data - should be one-time operation
                fixture.array(path, data=arr, chunks=chunks, order=order,
                              compressor=compressor)

            # setup array
            z = fixture[path]

            # check contents
            if arr.dtype.kind == 'f':
                assert_array_almost_equal(arr, z[:])
            else:
                assert_array_equal(arr, z[:])

            # check dtype
            assert arr.dtype == z.dtype

            # check compressor
            if compressor is None:
                assert z.compressor is None
            else:
                assert compressor.codec_id == z.compressor.codec_id
                assert compressor.get_config() == z.compressor.get_config()


@pytest.mark.skipif(asb is None,
                    reason="azure-blob-storage could not be imported")
class TestABSStore(StoreTests, unittest.TestCase):

    def create_store(self):
        blob_client = asb.BlockBlobService(is_emulated=True)
        blob_client.delete_container('test')
        blob_client.create_container('test')
        store = ABSStore(container='test', prefix='zarrtesting/', account_name='foo',
                         account_key='bar', blob_service_kwargs={'is_emulated': True})
        store.rmdir()
        return store


class TestConsolidatedMetadataStore(unittest.TestCase):

    def test_bad_format(self):

        # setup store with consolidated metdata
        store = dict()
        consolidated = {
            # bad format version
            'zarr_consolidated_format': 0,
        }
        store['.zmetadata'] = json.dumps(consolidated).encode()

        # check appropriate error is raised
        with pytest.raises(MetadataError):
            ConsolidatedMetadataStore(store)

    def test_read_write(self):

        # setup store with consolidated metdata
        store = dict()
        consolidated = {
            'zarr_consolidated_format': 1,
            'metadata': {
                'foo': 'bar',
                'baz': 42,
            }
        }
        store['.zmetadata'] = json.dumps(consolidated).encode()

        # create consolidated store
        cs = ConsolidatedMetadataStore(store)

        # test __contains__, __getitem__
        for key, value in consolidated['metadata'].items():
            assert key in cs
            assert value == cs[key]

        # test __delitem__, __setitem__
        with pytest.raises(PermissionError):
            del cs['foo']
        with pytest.raises(PermissionError):
            cs['bar'] = 0
        with pytest.raises(PermissionError):
            cs['spam'] = 'eggs'
