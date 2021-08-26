import array
import atexit
import json
import os
import pathlib
import sys
import pickle
import shutil
import tempfile
from contextlib import contextmanager
from pickle import PicklingError
from zipfile import ZipFile

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from numcodecs.compat import ensure_bytes

from zarr.codecs import BZ2, AsType, Blosc, Zlib
from zarr.errors import MetadataError
from zarr.hierarchy import group
from zarr.meta import (ZARR_FORMAT, decode_array_metadata,
                       decode_group_metadata, encode_array_metadata,
                       encode_group_metadata)
from zarr.n5 import N5Store
from zarr.storage import (ABSStore, ConsolidatedMetadataStore, DBMStore,
                          DictStore, DirectoryStore, LMDBStore, LRUStoreCache,
                          MemoryStore, MongoDBStore, NestedDirectoryStore,
                          RedisStore, SQLiteStore, TempStore, ZipStore,
                          array_meta_key, atexit_rmglob, atexit_rmtree,
                          attrs_key, default_compressor, getsize,
                          group_meta_key, init_array, init_group, migrate_1to2)
from zarr.storage import FSStore
from zarr.tests.util import CountingDict, have_fsspec, skip_test_env_var, abs_container


@contextmanager
def does_not_raise():
    yield


@pytest.fixture(params=[
    (None, "."),
    (".", "."),
    ("/", "/"),
])
def dimension_separator_fixture(request):
    return request.param


def skip_if_nested_chunks(**kwargs):
    if kwargs.get("dimension_separator") == "/":
        pytest.skip("nested chunks are unsupported")


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
        assert b'bar' == ensure_bytes(store['foo'])

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

        if hasattr(store, 'close'):
            store.close()

    def test_set_invalid_content(self):
        store = self.create_store()

        with pytest.raises(TypeError):
            store['baz'] = list(range(5))

        if hasattr(store, 'close'):
            store.close()

    def test_clear(self):
        store = self.create_store()
        store['foo'] = b'bar'
        store['baz'] = b'qux'
        assert len(store) == 2
        store.clear()
        assert len(store) == 0
        assert 'foo' not in store
        assert 'baz' not in store

        if hasattr(store, 'close'):
            store.close()

    def test_pop(self):
        store = self.create_store()
        store['foo'] = b'bar'
        store['baz'] = b'qux'
        assert len(store) == 2
        v = store.pop('foo')
        assert ensure_bytes(v) == b'bar'
        assert len(store) == 1
        v = store.pop('baz')
        assert ensure_bytes(v) == b'qux'
        assert len(store) == 0
        with pytest.raises(KeyError):
            store.pop('xxx')
        v = store.pop('xxx', b'default')
        assert v == b'default'
        v = store.pop('xxx', b'')
        assert v == b''
        v = store.pop('xxx', None)
        assert v is None

        if hasattr(store, 'close'):
            store.close()

    def test_popitem(self):
        store = self.create_store()
        store['foo'] = b'bar'
        k, v = store.popitem()
        assert k == 'foo'
        assert ensure_bytes(v) == b'bar'
        assert len(store) == 0
        with pytest.raises(KeyError):
            store.popitem()

        if hasattr(store, 'close'):
            store.close()

    def test_writeable_values(self):
        store = self.create_store()

        # __setitem__ should accept any value that implements buffer interface
        store['foo1'] = b'bar'
        store['foo2'] = bytearray(b'bar')
        store['foo3'] = array.array('B', b'bar')
        store['foo4'] = np.frombuffer(b'bar', dtype='u1')

        if hasattr(store, 'close'):
            store.close()

    def test_update(self):
        store = self.create_store()
        assert 'foo' not in store
        assert 'baz' not in store
        store.update(foo=b'bar', baz=b'quux')
        assert b'bar' == ensure_bytes(store['foo'])
        assert b'quux' == ensure_bytes(store['baz'])

        if hasattr(store, 'close'):
            store.close()

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
        assert {b'aaa', b'bbb', b'ddd', b'fff'} == set(map(ensure_bytes, store.values()))
        assert ({('a', b'aaa'), ('b', b'bbb'), ('c/d', b'ddd'), ('c/e/f', b'fff')} ==
                set(map(lambda kv: (kv[0], ensure_bytes(kv[1])), store.items())))

        if hasattr(store, 'close'):
            store.close()

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
        assert b'bar' == ensure_bytes(store2['foo'])
        assert b'quux' == ensure_bytes(store2['baz'])

        if hasattr(store2, 'close'):
            store2.close()

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

        if hasattr(store, 'close'):
            store.close()

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

        if hasattr(store, 'close'):
            store.close()

    def test_init_array(self, dimension_separator_fixture):

        pass_dim_sep, want_dim_sep = dimension_separator_fixture

        store = self.create_store(dimension_separator=pass_dim_sep)
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
        # Missing MUST be assumed to be "."
        assert meta.get('dimension_separator', ".") is want_dim_sep

        if hasattr(store, 'close'):
            store.close()

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

        if hasattr(store, 'close'):
            store.close()

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

        if hasattr(store, 'close'):
            store.close()

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

        if hasattr(store, 'close'):
            store.close()

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

        if hasattr(store, 'close'):
            store.close()

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

        if hasattr(store, 'close'):
            store.close()
        if hasattr(chunk_store, 'close'):
            chunk_store.close()

    def test_init_array_compat(self):
        store = self.create_store()
        init_array(store, shape=1000, chunks=100, compressor='none')
        meta = decode_array_metadata(store[array_meta_key])
        assert meta['compressor'] is None

        if hasattr(store, 'close'):
            store.close()

    def test_init_group(self):
        store = self.create_store()
        init_group(store)

        # check metadata
        assert group_meta_key in store
        meta = decode_group_metadata(store[group_meta_key])
        assert ZARR_FORMAT == meta['zarr_format']

        if hasattr(store, 'close'):
            store.close()

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

        if hasattr(store, 'close'):
            store.close()

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

        if hasattr(store, 'close'):
            store.close()

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

        if hasattr(store, 'close'):
            store.close()
        if hasattr(chunk_store, 'close'):
            chunk_store.close()


class TestMappingStore(StoreTests):

    def create_store(self, **kwargs):
        skip_if_nested_chunks(**kwargs)
        return dict()

    def test_set_invalid_content(self):
        # Generic mappings support non-buffer types
        pass


def setdel_hierarchy_checks(store):
    # these tests are for stores that are aware of hierarchy levels; this
    # behaviour is not stricly required by Zarr but these tests are included
    # to define behaviour of MemoryStore and DirectoryStore classes

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
    assert b'yyy' == ensure_bytes(store['x/y'])
    assert 'x/y/z' not in store
    store['x'] = b'zzz'
    assert b'zzz' == ensure_bytes(store['x'])
    assert 'x/y' not in store

    # test __delitem__ overwrite level
    store['r/s/t'] = b'xxx'
    del store['r/s']
    assert 'r/s/t' not in store
    store['r/s'] = b'xxx'
    del store['r']
    assert 'r/s' not in store


class TestMemoryStore(StoreTests):

    def create_store(self, **kwargs):
        skip_if_nested_chunks(**kwargs)
        return MemoryStore(**kwargs)

    def test_store_contains_bytes(self):
        store = self.create_store()
        store['foo'] = np.array([97, 98, 99, 100, 101], dtype=np.uint8)
        assert store['foo'] == b'abcde'

    def test_setdel(self):
        store = self.create_store()
        setdel_hierarchy_checks(store)


class TestDictStore(StoreTests):

    def create_store(self, **kwargs):
        skip_if_nested_chunks(**kwargs)

        with pytest.warns(DeprecationWarning):
            return DictStore(**kwargs)

    def test_deprecated(self):
        store = self.create_store()
        assert isinstance(store, MemoryStore)

    def test_pickle(self):
        with pytest.warns(DeprecationWarning):
            # pickle.load() will also trigger deprecation warning
            super().test_pickle()


class TestDirectoryStore(StoreTests):

    def create_store(self,
                     normalize_keys=False,
                     dimension_separator=".",
                     **kwargs):
        path = tempfile.mkdtemp()
        atexit.register(atexit_rmtree, path)
        store = DirectoryStore(path,
                               normalize_keys=normalize_keys,
                               dimension_separator=dimension_separator,
                               **kwargs)
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

        # check correct permissions
        # regression test for https://github.com/zarr-developers/zarr-python/issues/325
        stat = os.stat(path)
        mode = stat.st_mode & 0o666
        umask = os.umask(0)
        os.umask(umask)
        assert mode == (0o666 & ~umask)

        # test behaviour with file path
        with tempfile.NamedTemporaryFile() as f:
            with pytest.raises(ValueError):
                DirectoryStore(f.name)

    def test_init_pathlib(self):
        path = tempfile.mkdtemp()
        atexit.register(atexit_rmtree, path)
        DirectoryStore(pathlib.Path(path))

    def test_pickle_ext(self):
        store = self.create_store()
        store2 = pickle.loads(pickle.dumps(store))

        # check path is preserved
        assert store.path == store2.path

        # check point to same underlying directory
        assert 'xxx' not in store
        store2['xxx'] = b'yyy'
        assert b'yyy' == ensure_bytes(store['xxx'])

    def test_setdel(self):
        store = self.create_store()
        setdel_hierarchy_checks(store)

    def test_normalize_keys(self):
        store = self.create_store(normalize_keys=True)
        store['FOO'] = b'bar'
        assert 'FOO' in store
        assert 'foo' in store

    def test_listing_keys_slash(self):

        def mock_walker_slash(_path):
            yield from [
                # trailing slash in first key
                ('root_with_slash/', ['d1', 'g1'], ['.zgroup']),
                ('root_with_slash/d1', [], ['.zarray']),
                ('root_with_slash/g1', [], ['.zgroup'])
            ]

        res = set(DirectoryStore._keys_fast('root_with_slash/', walker=mock_walker_slash))
        assert res == {'.zgroup', 'g1/.zgroup', 'd1/.zarray'}

    def test_listing_keys_no_slash(self):

        def mock_walker_no_slash(_path):
            yield from [
                # no trainling slash in first key
                ('root_with_no_slash', ['d1', 'g1'], ['.zgroup']),
                ('root_with_no_slash/d1', [], ['.zarray']),
                ('root_with_no_slash/g1', [], ['.zgroup'])
            ]

        res = set(
            DirectoryStore._keys_fast('root_with_no_slash', mock_walker_no_slash)
                )
        assert res == {'.zgroup', 'g1/.zgroup', 'd1/.zarray'}


@pytest.mark.skipif(have_fsspec is False, reason="needs fsspec")
class TestFSStore(StoreTests):

    def create_store(self, normalize_keys=False, dimension_separator="."):
        path = tempfile.mkdtemp()
        atexit.register(atexit_rmtree, path)
        store = FSStore(
            path,
            normalize_keys=normalize_keys,
            dimension_separator=dimension_separator)
        return store

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
        assert meta['dimension_separator'] == "."

    def test_dimension_separator(self):
        for x in (".", "/"):
            store = self.create_store(dimension_separator=x)
            norm = store._normalize_key
            assert ".zarray" == norm(".zarray")
            assert ".zarray" == norm("/.zarray")
            assert ".zgroup" == norm("/.zgroup")
            assert "group/.zarray" == norm("group/.zarray")
            assert "group/.zgroup" == norm("group/.zgroup")
            assert "group/.zarray" == norm("/group/.zarray")
            assert "group/.zgroup" == norm("/group/.zgroup")

    def test_complex(self):
        path1 = tempfile.mkdtemp()
        path2 = tempfile.mkdtemp()
        store = FSStore("simplecache::file://" + path1,
                        simplecache={"same_names": True, "cache_storage": path2})
        assert not store
        assert not os.listdir(path1)
        assert not os.listdir(path2)
        store['foo'] = b"hello"
        assert 'foo' in os.listdir(path1)
        assert 'foo' in store
        assert not os.listdir(path2)
        assert store["foo"] == b"hello"
        assert 'foo' in os.listdir(path2)

    def test_not_fsspec(self):
        import zarr
        path = tempfile.mkdtemp()
        with pytest.raises(ValueError, match="storage_options"):
            zarr.open_array(path, mode='w', storage_options={"some": "kwargs"})
        with pytest.raises(ValueError, match="storage_options"):
            zarr.open_group(path, mode='w', storage_options={"some": "kwargs"})
        zarr.open_array("file://" + path, mode='w', shape=(1,), dtype="f8")

    def test_create(self):
        import zarr
        path1 = tempfile.mkdtemp()
        path2 = tempfile.mkdtemp()
        g = zarr.open_group("file://" + path1, mode='w',
                            storage_options={"auto_mkdir": True})
        a = g.create_dataset("data", shape=(8,))
        a[:4] = [0, 1, 2, 3]
        assert "data" in os.listdir(path1)
        assert ".zgroup" in os.listdir(path1)

        g = zarr.open_group("simplecache::file://" + path1, mode='r',
                            storage_options={"cache_storage": path2,
                                             "same_names": True})
        assert g.data[:].tolist() == [0, 1, 2, 3, 0, 0, 0, 0]
        with pytest.raises(PermissionError):
            g.data[:] = 1

    def test_read_only(self):
        path = tempfile.mkdtemp()
        atexit.register(atexit_rmtree, path)
        store = FSStore(path)
        store['foo'] = b"bar"

        store = FSStore(path, mode='r')

        with pytest.raises(PermissionError):
            store['foo'] = b"hex"

        with pytest.raises(PermissionError):
            del store['foo']

        with pytest.raises(PermissionError):
            store.clear()

        with pytest.raises(PermissionError):
            store.rmdir("anydir")

        assert store['foo'] == b"bar"

        filepath = os.path.join(path, "foo")
        with pytest.raises(ValueError):
            FSStore(filepath, mode='r')

    def test_eq(self):
        store1 = FSStore("anypath")
        store2 = FSStore("anypath")
        assert store1 == store2

    @pytest.mark.usefixtures("s3")
    def test_s3(self):
        import zarr
        g = zarr.open_group("s3://test/out.zarr", mode='w',
                            storage_options=self.s3so)
        a = g.create_dataset("data", shape=(8,))
        a[:4] = [0, 1, 2, 3]

        g = zarr.open_group("s3://test/out.zarr", mode='r',
                            storage_options=self.s3so)

        assert g.data[:].tolist() == [0, 1, 2, 3, 0, 0, 0, 0]

        # test via convenience
        g = zarr.open("s3://test/out.zarr", mode='r',
                      storage_options=self.s3so)
        assert g.data[:].tolist() == [0, 1, 2, 3, 0, 0, 0, 0]

    @pytest.mark.usefixtures("s3")
    def test_s3_complex(self):
        import zarr
        g = zarr.open_group("s3://test/out.zarr", mode='w',
                            storage_options=self.s3so)
        expected = np.empty((8, 8, 8), dtype='int64')
        expected[:] = -1
        a = g.create_dataset(
            "data", shape=(8, 8, 8), fill_value=-1, chunks=(1, 1, 1), overwrite=True
        )
        expected[0] = 0
        expected[3] = 3
        expected[6, 6, 6] = 6
        a[6, 6, 6] = 6
        a[:4] = expected[:4]

        b = g.create_dataset("data_f", shape=(8, ), chunks=(1,),
                             dtype=[('foo', 'S3'), ('bar', 'i4')],
                             fill_value=(b"b", 1))
        b[:4] = (b"aaa", 2)
        g2 = zarr.open_group("s3://test/out.zarr", mode='r',
                             storage_options=self.s3so)

        assert (g2.data[:] == expected).all()
        a.chunk_store.fs.invalidate_cache("test/out.zarr/data")
        a[:] = 5
        assert (a[:] == 5).all()

        assert g2.data_f['foo'].tolist() == [b"aaa"] * 4 + [b"b"] * 4
        with pytest.raises(PermissionError):
            g2.data[:] = 5

        with pytest.raises(PermissionError):
            g2.store.setitems({})

        with pytest.raises(PermissionError):
            # even though overwrite=True, store is read-only, so fails
            g2.create_dataset("data", shape=(8, 8, 8), mode='w',
                              fill_value=-1, chunks=(1, 1, 1), overwrite=True)

        a = g.create_dataset("data", shape=(8, 8, 8), mode='w',
                             fill_value=-1, chunks=(1, 1, 1), overwrite=True)
        assert (a[:] == -np.ones((8, 8, 8))).all()


@pytest.mark.skipif(have_fsspec is False, reason="needs fsspec")
class TestFSStoreWithKeySeparator(StoreTests):

    def create_store(self, normalize_keys=False, key_separator=".", **kwargs):

        # Since the user is passing key_separator, that will take priority.
        skip_if_nested_chunks(**kwargs)

        path = tempfile.mkdtemp()
        atexit.register(atexit_rmtree, path)
        return FSStore(
            path,
            normalize_keys=normalize_keys,
            key_separator=key_separator)


@pytest.fixture()
def s3(request):
    # writable local S3 system
    import shlex
    import subprocess
    import time
    if "BOTO_CONFIG" not in os.environ:  # pragma: no cover
        os.environ["BOTO_CONFIG"] = "/dev/null"
    if "AWS_ACCESS_KEY_ID" not in os.environ:  # pragma: no cover
        os.environ["AWS_ACCESS_KEY_ID"] = "foo"
    if "AWS_SECRET_ACCESS_KEY" not in os.environ:  # pragma: no cover
        os.environ["AWS_SECRET_ACCESS_KEY"] = "bar"
    requests = pytest.importorskip("requests")
    s3fs = pytest.importorskip("s3fs")
    pytest.importorskip("moto")

    port = 5555
    endpoint_uri = 'http://127.0.0.1:%s/' % port
    proc = subprocess.Popen(shlex.split("moto_server s3 -p %s" % port),
                            stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    timeout = 5
    while timeout > 0:
        try:
            r = requests.get(endpoint_uri)
            if r.ok:
                break
        except Exception:  # pragma: no cover
            pass
        timeout -= 0.1  # pragma: no cover
        time.sleep(0.1)  # pragma: no cover
    s3so = dict(client_kwargs={'endpoint_url': endpoint_uri},
                use_listings_cache=False)
    s3 = s3fs.S3FileSystem(anon=False, **s3so)
    s3.mkdir("test")
    request.cls.s3so = s3so
    yield
    proc.terminate()
    proc.wait()


class TestNestedDirectoryStore(TestDirectoryStore):

    def create_store(self, normalize_keys=False, **kwargs):
        path = tempfile.mkdtemp()
        atexit.register(atexit_rmtree, path)
        store = NestedDirectoryStore(path, normalize_keys=normalize_keys, **kwargs)
        return store

    def test_init_array(self):
        store = self.create_store()
        assert store._dimension_separator == "/"
        init_array(store, shape=1000, chunks=100)

        # check metadata
        assert array_meta_key in store
        meta = decode_array_metadata(store[array_meta_key])
        assert ZARR_FORMAT == meta['zarr_format']
        assert (1000,) == meta['shape']
        assert (100,) == meta['chunks']
        assert np.dtype(None) == meta['dtype']
        assert meta['dimension_separator'] == "/"

    def test_chunk_nesting(self):
        store = self.create_store()
        # any path where last segment looks like a chunk key gets special handling
        store['0.0'] = b'xxx'
        assert b'xxx' == store['0.0']
        # assert b'xxx' == store['0/0']
        store['foo/10.20.30'] = b'yyy'
        assert b'yyy' == store['foo/10.20.30']
        # assert b'yyy' == store['foo/10/20/30']
        store['42'] = b'zzz'
        assert b'zzz' == store['42']


class TestNestedDirectoryStoreNone:

    def test_value_error(self):
        path = tempfile.mkdtemp()
        atexit.register(atexit_rmtree, path)
        store = NestedDirectoryStore(
            path, normalize_keys=True,
            dimension_separator=None)
        assert store._dimension_separator == "/"


class TestNestedDirectoryStoreWithWrongValue:

    def test_value_error(self):
        path = tempfile.mkdtemp()
        atexit.register(atexit_rmtree, path)
        with pytest.raises(ValueError):
            NestedDirectoryStore(
                path, normalize_keys=True,
                dimension_separator=".")


class TestN5Store(TestNestedDirectoryStore):

    def create_store(self, normalize_keys=False):
        path = tempfile.mkdtemp(suffix='.n5')
        atexit.register(atexit_rmtree, path)
        store = N5Store(path, normalize_keys=normalize_keys)
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
        # assert b'xxx' == store['0/0']
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


@pytest.mark.skipif(have_fsspec is False, reason="needs fsspec")
class TestNestedFSStore(TestNestedDirectoryStore):

    def create_store(self, normalize_keys=False, path=None, **kwargs):
        if path is None:
            path = tempfile.mkdtemp()
        atexit.register(atexit_rmtree, path)
        store = FSStore(path, normalize_keys=normalize_keys,
                        dimension_separator='/', auto_mkdir=True, **kwargs)
        return store

    def test_numbered_groups(self):
        import zarr

        # Create an array
        store = self.create_store()
        group = zarr.group(store=store)
        arr = group.create_dataset('0', shape=(10, 10))
        arr[1] = 1

        # Read it back
        store = self.create_store(path=store.path)
        zarr.open_group(store.path)["0"]


class TestTempStore(StoreTests):

    def create_store(self, **kwargs):
        skip_if_nested_chunks(**kwargs)
        return TempStore(**kwargs)

    def test_setdel(self):
        store = self.create_store()
        setdel_hierarchy_checks(store)


class TestZipStore(StoreTests):

    def create_store(self, **kwargs):
        path = tempfile.mktemp(suffix='.zip')
        atexit.register(os.remove, path)
        store = ZipStore(path, mode='w', **kwargs)
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

    def test_permissions(self):
        store = ZipStore('data/store.zip', mode='w')
        store['foo'] = b'bar'
        store['baz/'] = b''
        store.flush()
        store.close()
        z = ZipFile('data/store.zip', 'r')
        info = z.getinfo('foo')
        perm = oct(info.external_attr >> 16)
        assert perm == '0o644'
        info = z.getinfo('baz/')
        perm = oct(info.external_attr >> 16)
        # only for posix platforms
        if os.name == 'posix':
            assert perm == '0o40775'
        z.close()


class TestDBMStore(StoreTests):

    def create_store(self, dimension_separator=None):
        path = tempfile.mktemp(suffix='.anydbm')
        atexit.register(atexit_rmglob, path + '*')
        # create store using default dbm implementation
        store = DBMStore(path, flag='n', dimension_separator=dimension_separator)
        return store

    def test_context_manager(self):
        with self.create_store() as store:
            store['foo'] = b'bar'
            store['baz'] = b'qux'
            assert 2 == len(store)


class TestDBMStoreDumb(TestDBMStore):

    def create_store(self, **kwargs):
        path = tempfile.mktemp(suffix='.dumbdbm')
        atexit.register(atexit_rmglob, path + '*')

        import dbm.dumb as dumbdbm
        store = DBMStore(path, flag='n', open=dumbdbm.open, **kwargs)
        return store


class TestDBMStoreGnu(TestDBMStore):

    def create_store(self, **kwargs):
        gdbm = pytest.importorskip("dbm.gnu")
        path = tempfile.mktemp(suffix=".gdbm")  # pragma: no cover
        atexit.register(os.remove, path)  # pragma: no cover
        store = DBMStore(
            path, flag="n", open=gdbm.open, write_lock=False, **kwargs
        )  # pragma: no cover
        return store  # pragma: no cover


class TestDBMStoreNDBM(TestDBMStore):

    def create_store(self, **kwargs):
        ndbm = pytest.importorskip("dbm.ndbm")
        path = tempfile.mktemp(suffix=".ndbm")  # pragma: no cover
        atexit.register(atexit_rmglob, path + "*")  # pragma: no cover
        store = DBMStore(path, flag="n", open=ndbm.open, **kwargs)  # pragma: no cover
        return store  # pragma: no cover


class TestDBMStoreBerkeleyDB(TestDBMStore):

    def create_store(self, **kwargs):
        bsddb3 = pytest.importorskip("bsddb3")
        path = tempfile.mktemp(suffix='.dbm')
        atexit.register(os.remove, path)
        store = DBMStore(path, flag='n', open=bsddb3.btopen, write_lock=False, **kwargs)
        return store


class TestLMDBStore(StoreTests):

    def create_store(self, **kwargs):
        pytest.importorskip("lmdb")
        path = tempfile.mktemp(suffix='.lmdb')
        atexit.register(atexit_rmtree, path)
        buffers = True
        store = LMDBStore(path, buffers=buffers, **kwargs)
        return store

    def test_context_manager(self):
        with self.create_store() as store:
            store['foo'] = b'bar'
            store['baz'] = b'qux'
            assert 2 == len(store)


class TestSQLiteStore(StoreTests):

    def create_store(self, **kwargs):
        pytest.importorskip("sqlite3")
        path = tempfile.mktemp(suffix='.db')
        atexit.register(atexit_rmtree, path)
        store = SQLiteStore(path, **kwargs)
        return store

    def test_underscore_in_name(self):
        path = tempfile.mktemp(suffix='.db')
        atexit.register(atexit_rmtree, path)
        store = SQLiteStore(path)
        store['a'] = b'aaa'
        store['a_b'] = b'aa_bb'
        store.rmdir('a')
        assert 'a_b' in store


class TestSQLiteStoreInMemory(TestSQLiteStore):

    def create_store(self, **kwargs):
        pytest.importorskip("sqlite3")
        store = SQLiteStore(':memory:', **kwargs)
        return store

    def test_pickle(self):

        # setup store
        store = self.create_store()
        store['foo'] = b'bar'
        store['baz'] = b'quux'

        # round-trip through pickle
        with pytest.raises(PicklingError):
            pickle.dumps(store)


@skip_test_env_var("ZARR_TEST_MONGO")
class TestMongoDBStore(StoreTests):

    def create_store(self, **kwargs):
        pytest.importorskip("pymongo")
        store = MongoDBStore(host='127.0.0.1', database='zarr_tests',
                             collection='zarr_tests', **kwargs)
        # start with an empty store
        store.clear()
        return store


@skip_test_env_var("ZARR_TEST_REDIS")
class TestRedisStore(StoreTests):

    def create_store(self, **kwargs):
        # TODO: this is the default host for Redis on Travis,
        # we probably want to generalize this though
        pytest.importorskip("redis")
        store = RedisStore(host='localhost', port=6379, **kwargs)
        # start with an empty store
        store.clear()
        return store


class TestLRUStoreCache(StoreTests):

    def create_store(self, **kwargs):
        # wrapper therefore no dimension_separator argument
        skip_if_nested_chunks(**kwargs)
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
        (np.arange(1111, dtype='<i1'), 100),
        (np.arange(1111, dtype='<i2'), 100),
        (np.arange(1111, dtype='<i4'), 100),
        (np.arange(1111, dtype='<i8'), 1000),
        (np.random.randint(0, 200, size=2222, dtype='u1').astype('<u1'), 100),
        (np.random.randint(0, 2000, size=2222, dtype='u2').astype('<u2'), 100),
        (np.random.randint(0, 2000, size=2222, dtype='u4').astype('<u4'), 100),
        (np.random.randint(0, 2000, size=2222, dtype='u8').astype('<u8'), 100),
        (np.linspace(0, 1, 3333, dtype='<f2'), 100),
        (np.linspace(0, 1, 3333, dtype='<f4'), 100),
        (np.linspace(0, 1, 3333, dtype='<f8'), 100),
        (np.random.normal(loc=0, scale=1, size=4444).astype('<f2'), 100),
        (np.random.normal(loc=0, scale=1, size=4444).astype('<f4'), 100),
        (np.random.normal(loc=0, scale=1, size=4444).astype('<f8'), 100),
        (np.random.choice([b'A', b'C', b'G', b'T'],
                          size=5555, replace=True).astype('S'), 100),
        (np.random.choice(['foo', 'bar', 'baz', 'quux'],
                          size=5555, replace=True).astype('<U'), 100),
        (np.random.choice([0, 1/3, 1/7, 1/9, np.nan],
                          size=5555, replace=True).astype('<f8'), 100),
        (np.random.randint(0, 2, size=5555, dtype=bool), 100),
        (np.arange(20000, dtype='<i4').reshape(2000, 10, order='C'), (100, 3)),
        (np.arange(20000, dtype='<i4').reshape(200, 100, order='F'), (100, 30)),
        (np.arange(20000, dtype='<i4').reshape(200, 10, 10, order='C'), (100, 3, 3)),
        (np.arange(20000, dtype='<i4').reshape(20, 100, 10, order='F'), (10, 30, 3)),
        (np.arange(20000, dtype='<i4').reshape(20, 10, 10, 10, order='C'), (10, 3, 3, 3)),
        (np.arange(20000, dtype='<i4').reshape(20, 10, 10, 10, order='F'), (10, 3, 3, 3)),
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


@skip_test_env_var("ZARR_TEST_ABS")
class TestABSStore(StoreTests):

    def create_store(self, prefix=None, **kwargs):
        container_client = abs_container()
        store = ABSStore(
            prefix=prefix,
            client=container_client,
            **kwargs,
        )
        store.rmdir()
        return store

    def test_non_client_deprecated(self):
        with pytest.warns(FutureWarning, match='Providing'):
            store = ABSStore("container", account_name="account_name", account_key="account_key")

        for attr in ["container", "account_name", "account_key"]:
            with pytest.warns(FutureWarning, match=attr):
                result = getattr(store, attr)
            assert result == attr

    def test_iterators_with_prefix(self):
        for prefix in ['test_prefix', '/test_prefix', 'test_prefix/', 'test/prefix', '', None]:
            store = self.create_store(prefix=prefix)

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

    def test_getsize(self):
        return super().test_getsize()

    def test_hierarchy(self):
        return super().test_hierarchy()

    @pytest.mark.skipif(sys.version_info < (3, 7), reason="attr not serializable in py36")
    def test_pickle(self):
        # internal attribute on ContainerClient isn't serializable for py36 and earlier
        super().test_pickle()


class TestConsolidatedMetadataStore:

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
