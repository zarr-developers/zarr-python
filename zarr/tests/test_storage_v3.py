import array
import atexit
import copy
import os
import tempfile

import numpy as np
import pytest

import zarr
from zarr._storage.store import _get_hierarchy_metadata, v3_api_available
from zarr.meta import _default_entry_point_metadata_v3
from zarr.storage import (atexit_rmglob, atexit_rmtree, data_root,
                          default_compressor, getsize, init_array, meta_root,
                          normalize_store_arg)
from zarr._storage.v3 import (ABSStoreV3, ConsolidatedMetadataStoreV3, DBMStoreV3,
                              DirectoryStoreV3, FSStoreV3, KVStore, KVStoreV3,
                              LMDBStoreV3, LRUStoreCacheV3, MemoryStoreV3,
                              MongoDBStoreV3, RedisStoreV3, SQLiteStoreV3, StoreV3,
                              ZipStoreV3)
from zarr.tests.util import CountingDictV3, have_fsspec, skip_test_env_var, mktemp

# pytest will fail to run if the following fixtures aren't imported here
from .test_storage import StoreTests as _StoreTests
from .test_storage import TestABSStore as _TestABSStore
from .test_storage import TestConsolidatedMetadataStore as _TestConsolidatedMetadataStore
from .test_storage import TestDBMStore as _TestDBMStore
from .test_storage import TestDBMStoreBerkeleyDB as _TestDBMStoreBerkeleyDB
from .test_storage import TestDBMStoreDumb as _TestDBMStoreDumb
from .test_storage import TestDBMStoreGnu as _TestDBMStoreGnu
from .test_storage import TestDBMStoreNDBM as _TestDBMStoreNDBM
from .test_storage import TestDirectoryStore as _TestDirectoryStore
from .test_storage import TestFSStore as _TestFSStore
from .test_storage import TestLMDBStore as _TestLMDBStore
from .test_storage import TestLRUStoreCache as _TestLRUStoreCache
from .test_storage import TestMemoryStore as _TestMemoryStore
from .test_storage import TestSQLiteStore as _TestSQLiteStore
from .test_storage import TestSQLiteStoreInMemory as _TestSQLiteStoreInMemory
from .test_storage import TestZipStore as _TestZipStore
from .test_storage import (dimension_separator_fixture, s3,  # noqa
                           skip_if_nested_chunks)


pytestmark = pytest.mark.skipif(
    not v3_api_available,
    reason="v3 api is not available"
)


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
        """keys"""

    def values(self):
        """values"""

    def get(self, value, default=None):
        """get"""

    def __setitem__(self, key, value):
        """__setitem__"""

    def __getitem__(self, key):
        """__getitem__"""

    def __delitem__(self, key):
        """__delitem__"""

    def __contains__(self, key):
        """__contains__"""


class InvalidDummyStore():
    # does not contain expected methods of a MutableMapping

    def keys(self):
        """keys"""


def test_ensure_store_v3():
    class InvalidStore:
        pass

    with pytest.raises(ValueError):
        StoreV3._ensure_store(InvalidStore())

    # cannot initialize with a store from a different Zarr version
    with pytest.raises(ValueError):
        StoreV3._ensure_store(KVStore(dict()))

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

    for key in store._valid_key_characters:
        assert store._valid_key(key)

    # other characters not in store._valid_key_characters are not allowed
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
    for valid in [meta_root + 'arr1.array.json',
                  data_root + 'arr1.array.json',
                  meta_root + 'subfolder/item_1-0.group.json']:
        store._validate_key(valid)
        # but otherwise valid keys cannot end in /
        with pytest.raises(ValueError):
            assert store._validate_key(valid + '/')

    for invalid in [0, '*', '~', '^', '&']:
        with pytest.raises(ValueError):
            store._validate_key(invalid)


class StoreV3Tests(_StoreTests):

    version = 3
    root = meta_root

    def test_getsize(self):
        # TODO: determine proper getsize() behavior for v3
        #       Currently returns the combined size of entries under
        #       meta/root/path and data/root/path.
        #       Any path not under meta/root/ or data/root/ (including zarr.json)
        #       returns size 0.

        store = self.create_store()
        if isinstance(store, dict) or hasattr(store, 'getsize'):
            assert 0 == getsize(store, 'zarr.json')
            store[meta_root + 'foo/a'] = b'x'
            assert 1 == getsize(store)
            assert 1 == getsize(store, 'foo')
            store[meta_root + 'foo/b'] = b'x'
            assert 2 == getsize(store, 'foo')
            assert 1 == getsize(store, 'foo/b')
            store[meta_root + 'bar/a'] = b'yy'
            assert 2 == getsize(store, 'bar')
            store[data_root + 'bar/a'] = b'zzz'
            assert 5 == getsize(store, 'bar')
            store[data_root + 'baz/a'] = b'zzz'
            assert 3 == getsize(store, 'baz')
            assert 10 == getsize(store)
            store[data_root + 'quux'] = array.array('B', b'zzzz')
            assert 14 == getsize(store)
            assert 4 == getsize(store, 'quux')
            store[data_root + 'spong'] = np.frombuffer(b'zzzzz', dtype='u1')
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
        mkey = meta_root + path + '.array.json'
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

        expected = [meta_root + 'arr1.array.json', 'zarr.json']
        assert sorted(store.list_prefix('')) == expected

        expected = [meta_root + 'arr1.array.json']
        assert sorted(store.list_prefix(meta_root.rstrip('/'))) == expected

        # cannot start prefix with '/'
        with pytest.raises(ValueError):
            store.list_prefix(prefix='/' + meta_root.rstrip('/'))

    def test_equal(self):
        store = self.create_store()
        assert store == store

    def test_rename_nonexisting(self):
        store = self.create_store()
        if store.is_erasable():
            with pytest.raises(ValueError):
                store.rename('a', 'b')
        else:
            with pytest.raises(NotImplementedError):
                store.rename('a', 'b')


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

    def test_rename_nonexisting(self):
        store = self.create_store()
        with pytest.raises(FileNotFoundError):
            store.rename(meta_root + 'a', meta_root + 'b')


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
        mkey = meta_root + path + '.array.json'
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


# TODO: enable once N5StoreV3 has been implemented
# @pytest.mark.skipif(True, reason="N5StoreV3 not yet fully implemented")
# class TestN5StoreV3(_TestN5Store, TestDirectoryStoreV3, StoreV3Tests):


class TestZipStoreV3(_TestZipStore, StoreV3Tests):

    ZipStoreClass = ZipStoreV3

    def create_store(self, **kwargs):
        path = mktemp(suffix='.zip')
        atexit.register(os.remove, path)
        store = ZipStoreV3(path, mode='w', **kwargs)
        return store


class TestDBMStoreV3(_TestDBMStore, StoreV3Tests):

    def create_store(self, dimension_separator=None):
        path = mktemp(suffix='.anydbm')
        atexit.register(atexit_rmglob, path + '*')
        # create store using default dbm implementation
        store = DBMStoreV3(path, flag='n', dimension_separator=dimension_separator)
        return store


class TestDBMStoreV3Dumb(_TestDBMStoreDumb, StoreV3Tests):

    def create_store(self, **kwargs):
        path = mktemp(suffix='.dumbdbm')
        atexit.register(atexit_rmglob, path + '*')

        import dbm.dumb as dumbdbm
        store = DBMStoreV3(path, flag='n', open=dumbdbm.open, **kwargs)
        return store


class TestDBMStoreV3Gnu(_TestDBMStoreGnu, StoreV3Tests):

    def create_store(self, **kwargs):
        gdbm = pytest.importorskip("dbm.gnu")
        path = mktemp(suffix=".gdbm")  # pragma: no cover
        atexit.register(os.remove, path)  # pragma: no cover
        store = DBMStoreV3(
            path, flag="n", open=gdbm.open, write_lock=False, **kwargs
        )  # pragma: no cover
        return store  # pragma: no cover


class TestDBMStoreV3NDBM(_TestDBMStoreNDBM, StoreV3Tests):

    def create_store(self, **kwargs):
        ndbm = pytest.importorskip("dbm.ndbm")
        path = mktemp(suffix=".ndbm")  # pragma: no cover
        atexit.register(atexit_rmglob, path + "*")  # pragma: no cover
        store = DBMStoreV3(path, flag="n", open=ndbm.open, **kwargs)  # pragma: no cover
        return store  # pragma: no cover


class TestDBMStoreV3BerkeleyDB(_TestDBMStoreBerkeleyDB, StoreV3Tests):

    def create_store(self, **kwargs):
        bsddb3 = pytest.importorskip("bsddb3")
        path = mktemp(suffix='.dbm')
        atexit.register(os.remove, path)
        store = DBMStoreV3(path, flag='n', open=bsddb3.btopen, write_lock=False, **kwargs)
        return store


class TestLMDBStoreV3(_TestLMDBStore, StoreV3Tests):

    def create_store(self, **kwargs):
        pytest.importorskip("lmdb")
        path = mktemp(suffix='.lmdb')
        atexit.register(atexit_rmtree, path)
        buffers = True
        store = LMDBStoreV3(path, buffers=buffers, **kwargs)
        return store


class TestSQLiteStoreV3(_TestSQLiteStore, StoreV3Tests):

    def create_store(self, **kwargs):
        pytest.importorskip("sqlite3")
        path = mktemp(suffix='.db')
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


@skip_test_env_var("ZARR_TEST_ABS")
class TestABSStoreV3(_TestABSStore, StoreV3Tests):

    ABSStoreClass = ABSStoreV3


def test_normalize_store_arg_v3(tmpdir):

    fn = tmpdir.join('store.zip')
    store = normalize_store_arg(str(fn), zarr_version=3, mode='w')
    assert isinstance(store, ZipStoreV3)
    assert 'zarr.json' in store

    # can't pass storage_options to non-fsspec store
    with pytest.raises(ValueError):
        normalize_store_arg(str(fn), zarr_version=3, mode='w', storage_options={"some": "kwargs"})

    if have_fsspec:
        path = tempfile.mkdtemp()
        store = normalize_store_arg("file://" + path, zarr_version=3, mode='w')
        assert isinstance(store, FSStoreV3)
        assert 'zarr.json' in store

    fn = tmpdir.join('store.n5')
    with pytest.raises(NotImplementedError):
        normalize_store_arg(str(fn), zarr_version=3, mode='w')

    # error on zarr_version=3 with a v2 store
    with pytest.raises(ValueError):
        normalize_store_arg(KVStore(dict()), zarr_version=3, mode='w')

    # error on zarr_version=2 with a v3 store
    with pytest.raises(ValueError):
        normalize_store_arg(KVStoreV3(dict()), zarr_version=2, mode='w')


class TestConsolidatedMetadataStoreV3(_TestConsolidatedMetadataStore):

    version = 3
    ConsolidatedMetadataClass = ConsolidatedMetadataStoreV3

    @property
    def metadata_key(self):
        return meta_root + 'consolidated/.zmetadata'

    def test_bad_store_version(self):
        with pytest.raises(ValueError):
            self.ConsolidatedMetadataClass(KVStore(dict()))


def test_get_hierarchy_metadata():
    store = KVStoreV3({})

    # error raised if 'jarr.json' is not in the store
    with pytest.raises(ValueError):
        _get_hierarchy_metadata(store)

    store['zarr.json'] = _default_entry_point_metadata_v3
    assert _get_hierarchy_metadata(store) == _default_entry_point_metadata_v3

    # ValueError if only a subset of keys are present
    store['zarr.json'] = {'zarr_format': 'https://purl.org/zarr/spec/protocol/core/3.0'}
    with pytest.raises(ValueError):
        _get_hierarchy_metadata(store)

    # ValueError if any unexpected keys are present
    extra_metadata = copy.copy(_default_entry_point_metadata_v3)
    extra_metadata['extra_key'] = 'value'
    store['zarr.json'] = extra_metadata
    with pytest.raises(ValueError):
        _get_hierarchy_metadata(store)


def test_top_level_imports():
    for store_name in ['ABSStoreV3', 'DBMStoreV3', 'KVStoreV3', 'DirectoryStoreV3',
                       'LMDBStoreV3', 'LRUStoreCacheV3', 'MemoryStoreV3', 'MongoDBStoreV3',
                       'RedisStoreV3', 'SQLiteStoreV3', 'ZipStoreV3']:
        if v3_api_available:
            assert hasattr(zarr, store_name)  # pragma: no cover
        else:
            assert not hasattr(zarr, store_name)  # pragma: no cover
