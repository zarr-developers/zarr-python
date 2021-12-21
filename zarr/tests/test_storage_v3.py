import array
import atexit
import os
import tempfile

import numpy as np
import pytest

from zarr._storage.store import _valid_key_characters
from zarr.storage import (atexit_rmglob, atexit_rmtree, default_compressor,
                          getsize, init_array)
from zarr.storage import (KVStoreV3, MemoryStoreV3, ZipStoreV3, FSStoreV3,
                          DirectoryStoreV3, NestedDirectoryStoreV3,
                          RedisStoreV3, MongoDBStoreV3, DBMStoreV3,
                          LMDBStoreV3, SQLiteStoreV3, LRUStoreCacheV3,
                          StoreV3, normalize_store_arg, KVStore)
from zarr.tests.util import CountingDictV3, have_fsspec, skip_test_env_var

from .test_storage import (
    StoreTests as _StoreTests,
    TestMemoryStore as _TestMemoryStore,
    TestDirectoryStore as _TestDirectoryStore,
    TestFSStore as _TestFSStore,
    TestNestedDirectoryStore as _TestNestedDirectoryStore,
    TestZipStore as _TestZipStore,
    TestDBMStore as _TestDBMStore,
    TestDBMStoreDumb as _TestDBMStoreDumb,
    TestDBMStoreGnu as _TestDBMStoreGnu,
    TestDBMStoreNDBM as _TestDBMStoreNDBM,
    TestDBMStoreBerkeleyDB as _TestDBMStoreBerkeleyDB,
    TestLMDBStore as _TestLMDBStore,
    TestSQLiteStore as _TestSQLiteStore,
    TestSQLiteStoreInMemory as _TestSQLiteStoreInMemory,
    TestLRUStoreCache as _TestLRUStoreCache,
    skip_if_nested_chunks)

# pytest will fail to run if the following fixtures aren't imported here
from .test_storage import dimension_separator_fixture, s3  # noqa


@pytest.fixture(params=[
    (None, "/"),
    (".", "."),
    ("/", "/"),
])
def dimension_separator_fixture_v3(request):
    return request.param


class DummyStore():
    # contains all methods expected of Mutable Mapping

    def keys(self):
        pass

    def values(self):
        pass

    def get(self, value, default=None):
        pass

    def __setitem__(self, key, value):
        pass

    def __getitem__(self, key):
        pass

    def __delitem__(self, key):
        pass

    def __contains__(self, key):
        pass


class InvalidDummyStore():
    # does not contain expected methods of a MutableMapping

    def keys(self):
        pass


def test_ensure_store_v3():
    class InvalidStore:
        pass

    with pytest.raises(ValueError):
        StoreV3._ensure_store(InvalidStore())

    assert StoreV3._ensure_store(None) is None

    # class with all methods of a MutableMapping will become a KVStoreV3
    assert isinstance(StoreV3._ensure_store(DummyStore), KVStoreV3)

    with pytest.raises(ValueError):
        # does not have the methods expected of a MutableMapping
        StoreV3._ensure_store(InvalidDummyStore)


def test_valid_key():
    store = KVStoreV3(dict)

    # only ascii keys are valid
    assert not store._valid_key(5)
    assert not store._valid_key(2.8)

    for key in _valid_key_characters:
        assert store._valid_key(key)

    # other characters not in _valid_key_characters are not allowed
    assert not store._valid_key('*')
    assert not store._valid_key('~')
    assert not store._valid_key('^')


def test_validate_key():
    store = KVStoreV3(dict)

    # zarr.json is a valid key
    store._validate_key('zarr.json')
    # but other keys not starting with meta/ or data/ are not
    with pytest.raises(ValueError):
        store._validate_key('zar.json')

    # valid ascii keys
    for valid in ['meta/root/arr1.array.json',
                  'data/root/arr1.array.json',
                  'meta/root/subfolder/item_1-0.group.json']:
        store._validate_key(valid)
        # but otherwise valid keys cannot end in /
        with pytest.raises(ValueError):
            assert store._validate_key(valid + '/')

    for invalid in [0, '*', '~', '^', '&']:
        with pytest.raises(ValueError):
            store._validate_key(invalid)


class StoreV3Tests(_StoreTests):

    version = 3
    root = 'meta/root/'

    def test_getsize(self):
        # TODO: determine proper getsize() behavior for v3
        #       Currently returns the combined size of entries under
        #       meta/root/path and data/root/path.
        #       Any path not under meta/root/ or data/root/ (including zarr.json)
        #       returns size 0.

        store = self.create_store()
        if isinstance(store, dict) or hasattr(store, 'getsize'):
            assert 0 == getsize(store, 'zarr.json')
            store['meta/root/foo/a'] = b'x'
            assert 1 == getsize(store)
            assert 1 == getsize(store, 'foo')
            store['meta/root/foo/b'] = b'x'
            assert 2 == getsize(store, 'foo')
            assert 1 == getsize(store, 'foo/b')
            store['meta/root/bar/a'] = b'yy'
            assert 2 == getsize(store, 'bar')
            store['data/root/bar/a'] = b'zzz'
            assert 5 == getsize(store, 'bar')
            store['data/root/baz/a'] = b'zzz'
            assert 3 == getsize(store, 'baz')
            assert 10 == getsize(store)
            store['data/root/quux'] = array.array('B', b'zzzz')
            assert 14 == getsize(store)
            assert 4 == getsize(store, 'quux')
            store['data/root/spong'] = np.frombuffer(b'zzzzz', dtype='u1')
            assert 19 == getsize(store)
            assert 5 == getsize(store, 'spong')
        store.close()

    def test_init_array(self, dimension_separator_fixture_v3):

        pass_dim_sep, want_dim_sep = dimension_separator_fixture_v3

        store = self.create_store()
        path = 'arr1'
        init_array(store, path=path, shape=1000, chunks=100,
                   dimension_separator=pass_dim_sep)

        # check metadata
        mkey = 'meta/root/' + path + '.array.json'
        assert mkey in store
        meta = store._metadata_class.decode_array_metadata(store[mkey])
        assert (1000,) == meta['shape']
        assert (100,) == meta['chunk_grid']['chunk_shape']
        assert np.dtype(None) == meta['data_type']
        assert default_compressor == meta['compressor']
        assert meta['fill_value'] is None
        # Missing MUST be assumed to be "/"
        assert meta['chunk_grid']['separator'] is want_dim_sep
        store.close()

    def test_list_prefix(self):

        store = self.create_store()
        path = 'arr1'
        init_array(store, path=path, shape=1000, chunks=100)

        expected = ['meta/root/arr1.array.json', 'zarr.json']
        assert sorted(store.list_prefix('')) == expected

        expected = ['meta/root/arr1.array.json']
        assert sorted(store.list_prefix('meta/root')) == expected

        # cannot start prefix with '/'
        with pytest.raises(ValueError):
            store.list_prefix(prefix='/meta/root')


class TestMappingStoreV3(StoreV3Tests):

    def create_store(self, **kwargs):
        return KVStoreV3(dict())

    def test_set_invalid_content(self):
        # Generic mappings support non-buffer types
        pass


class TestMemoryStoreV3(_TestMemoryStore, StoreV3Tests):

    def create_store(self, **kwargs):
        skip_if_nested_chunks(**kwargs)
        return MemoryStoreV3(**kwargs)


class TestDirectoryStoreV3(_TestDirectoryStore, StoreV3Tests):

    def create_store(self, normalize_keys=False, **kwargs):
        # For v3, don't have to skip if nested.
        # skip_if_nested_chunks(**kwargs)

        path = tempfile.mkdtemp()
        atexit.register(atexit_rmtree, path)
        store = DirectoryStoreV3(path, normalize_keys=normalize_keys, **kwargs)
        return store


@pytest.mark.skipif(have_fsspec is False, reason="needs fsspec")
class TestFSStoreV3(_TestFSStore, StoreV3Tests):

    def create_store(self, normalize_keys=False,
                     dimension_separator=".",
                     path=None,
                     **kwargs):

        if path is None:
            path = tempfile.mkdtemp()
            atexit.register(atexit_rmtree, path)

        store = FSStoreV3(
            path,
            normalize_keys=normalize_keys,
            dimension_separator=dimension_separator,
            **kwargs)
        return store

    def test_init_array(self):
        store = self.create_store()
        path = 'arr1'
        init_array(store, path=path, shape=1000, chunks=100)

        # check metadata
        mkey = 'meta/root/' + path + '.array.json'
        assert mkey in store
        meta = store._metadata_class.decode_array_metadata(store[mkey])
        assert (1000,) == meta['shape']
        assert (100,) == meta['chunk_grid']['chunk_shape']
        assert np.dtype(None) == meta['data_type']
        assert meta['chunk_grid']['separator'] == "/"


@pytest.mark.skipif(have_fsspec is False, reason="needs fsspec")
class TestFSStoreV3WithKeySeparator(StoreV3Tests):

    def create_store(self, normalize_keys=False, key_separator=".", **kwargs):

        # Since the user is passing key_separator, that will take priority.
        skip_if_nested_chunks(**kwargs)

        path = tempfile.mkdtemp()
        atexit.register(atexit_rmtree, path)
        return FSStoreV3(
            path,
            normalize_keys=normalize_keys,
            key_separator=key_separator)


# TODO: remove NestedDirectoryStoreV3?
class TestNestedDirectoryStoreV3(_TestNestedDirectoryStore,
                                 TestDirectoryStoreV3):

    def create_store(self, normalize_keys=False, **kwargs):
        path = tempfile.mkdtemp()
        atexit.register(atexit_rmtree, path)
        store = NestedDirectoryStoreV3(path, normalize_keys=normalize_keys, **kwargs)
        return store

    def test_init_array(self):
        store = self.create_store()
        # assert store._dimension_separator == "/"
        path = 'arr1'
        init_array(store, path=path, shape=1000, chunks=100)

        # check metadata
        mkey = self.root + path + '.array.json'
        assert mkey in store
        meta = store._metadata_class.decode_array_metadata(store[mkey])
        assert (1000,) == meta['shape']
        assert (100,) == meta['chunk_grid']['chunk_shape']
        assert np.dtype(None) == meta['data_type']
        # assert meta['dimension_separator'] == "/"
        assert meta['chunk_grid']['separator'] == "/"

# TODO: enable once N5StoreV3 has been implemented
# @pytest.mark.skipif(True, reason="N5StoreV3 not yet fully implemented")
# class TestN5StoreV3(_TestN5Store, TestNestedDirectoryStoreV3, StoreV3Tests):


class TestZipStoreV3(_TestZipStore, StoreV3Tests):

    ZipStoreClass = ZipStoreV3

    def create_store(self, **kwargs):
        path = tempfile.mktemp(suffix='.zip')
        atexit.register(os.remove, path)
        store = ZipStoreV3(path, mode='w', **kwargs)
        return store


class TestDBMStoreV3(_TestDBMStore, StoreV3Tests):

    def create_store(self, dimension_separator=None):
        path = tempfile.mktemp(suffix='.anydbm')
        atexit.register(atexit_rmglob, path + '*')
        # create store using default dbm implementation
        store = DBMStoreV3(path, flag='n', dimension_separator=dimension_separator)
        return store


class TestDBMStoreV3Dumb(_TestDBMStoreDumb, StoreV3Tests):

    def create_store(self, **kwargs):
        path = tempfile.mktemp(suffix='.dumbdbm')
        atexit.register(atexit_rmglob, path + '*')

        import dbm.dumb as dumbdbm
        store = DBMStoreV3(path, flag='n', open=dumbdbm.open, **kwargs)
        return store


class TestDBMStoreV3Gnu(_TestDBMStoreGnu, StoreV3Tests):

    def create_store(self, **kwargs):
        gdbm = pytest.importorskip("dbm.gnu")
        path = tempfile.mktemp(suffix=".gdbm")  # pragma: no cover
        atexit.register(os.remove, path)  # pragma: no cover
        store = DBMStoreV3(
            path, flag="n", open=gdbm.open, write_lock=False, **kwargs
        )  # pragma: no cover
        return store  # pragma: no cover


class TestDBMStoreV3NDBM(_TestDBMStoreNDBM, StoreV3Tests):

    def create_store(self, **kwargs):
        ndbm = pytest.importorskip("dbm.ndbm")
        path = tempfile.mktemp(suffix=".ndbm")  # pragma: no cover
        atexit.register(atexit_rmglob, path + "*")  # pragma: no cover
        store = DBMStoreV3(path, flag="n", open=ndbm.open, **kwargs)  # pragma: no cover
        return store  # pragma: no cover


class TestDBMStoreV3BerkeleyDB(_TestDBMStoreBerkeleyDB, StoreV3Tests):

    def create_store(self, **kwargs):
        bsddb3 = pytest.importorskip("bsddb3")
        path = tempfile.mktemp(suffix='.dbm')
        atexit.register(os.remove, path)
        store = DBMStoreV3(path, flag='n', open=bsddb3.btopen, write_lock=False, **kwargs)
        return store


class TestLMDBStoreV3(_TestLMDBStore, StoreV3Tests):

    def create_store(self, **kwargs):
        pytest.importorskip("lmdb")
        path = tempfile.mktemp(suffix='.lmdb')
        atexit.register(atexit_rmtree, path)
        buffers = True
        store = LMDBStoreV3(path, buffers=buffers, **kwargs)
        return store


class TestSQLiteStoreV3(_TestSQLiteStore, StoreV3Tests):

    def create_store(self, **kwargs):
        pytest.importorskip("sqlite3")
        path = tempfile.mktemp(suffix='.db')
        atexit.register(atexit_rmtree, path)
        store = SQLiteStoreV3(path, **kwargs)
        return store


class TestSQLiteStoreV3InMemory(_TestSQLiteStoreInMemory, StoreV3Tests):

    def create_store(self, **kwargs):
        pytest.importorskip("sqlite3")
        store = SQLiteStoreV3(':memory:', **kwargs)
        return store


@skip_test_env_var("ZARR_TEST_MONGO")
class TestMongoDBStoreV3(StoreV3Tests):

    def create_store(self, **kwargs):
        pytest.importorskip("pymongo")
        store = MongoDBStoreV3(host='127.0.0.1', database='zarr_tests',
                               collection='zarr_tests', **kwargs)
        # start with an empty store
        store.clear()
        return store


@skip_test_env_var("ZARR_TEST_REDIS")
class TestRedisStoreV3(StoreV3Tests):

    def create_store(self, **kwargs):
        # TODO: this is the default host for Redis on Travis,
        # we probably want to generalize this though
        pytest.importorskip("redis")
        store = RedisStoreV3(host='localhost', port=6379, **kwargs)
        # start with an empty store
        store.clear()
        return store


class TestLRUStoreCacheV3(_TestLRUStoreCache, StoreV3Tests):

    CountingClass = CountingDictV3
    LRUStoreClass = LRUStoreCacheV3


# TODO: implement ABSStoreV3
# @skip_test_env_var("ZARR_TEST_ABS")
# class TestABSStoreV3(_TestABSStore, StoreV3Tests):

def test_normalize_store_arg_v3(tmpdir):

    fn = tmpdir.join('store.zip')
    store = normalize_store_arg(str(fn), zarr_version=3, mode='w', clobber=True)
    assert isinstance(store, ZipStoreV3)
    assert 'zarr.json' in store

    if have_fsspec:
        path = tempfile.mkdtemp()
        store = normalize_store_arg("file://" + path, zarr_version=3, mode='w', clobber=True)
        assert isinstance(store, FSStoreV3)
        assert 'zarr.json' in store

    fn = tmpdir.join('store.n5')
    with pytest.raises(NotImplementedError):
        normalize_store_arg(str(fn), zarr_version=3, mode='w', clobber=True)

    # error on zarr_version=3 with a v2 store
    with pytest.raises(ValueError):
        normalize_store_arg(KVStore(dict()), zarr_version=3, mode='w', clobber=True)

    # error on zarr_version=2 with a v3 store
    with pytest.raises(ValueError):
        normalize_store_arg(KVStoreV3(dict()), zarr_version=2, mode='w', clobber=True)
