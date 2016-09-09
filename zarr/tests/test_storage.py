# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import unittest
import tempfile
import atexit
import pickle
import json
import array
import shutil
import os


import numpy as np
from nose.tools import assert_raises, eq_ as eq, assert_is_none


from zarr.storage import init_array, array_meta_key, attrs_key, DictStore, \
    DirectoryStore, ZipStore, init_group, group_meta_key, getsize, \
    migrate_1to2, TempStore
from zarr.meta import decode_array_metadata, encode_array_metadata, \
    ZARR_FORMAT, decode_group_metadata, encode_group_metadata
from zarr.compat import text_type
from zarr.storage import default_compressor
from zarr.codecs import Zlib, Blosc
from zarr.errors import PermissionError


class StoreTests(object):
    """Abstract store tests."""

    def create_store(self, **kwargs):  # pragma: no cover
        # implement in sub-class
        raise NotImplementedError

    def test_get_set_del_contains(self):
        store = self.create_store()

        # test __contains__, __getitem__, __setitem__
        assert 'foo' not in store
        with assert_raises(KeyError):
            store['foo']
        store['foo'] = b'bar'
        assert 'foo' in store
        eq(b'bar', store['foo'])

        # test __delitem__ (optional)
        try:
            del store['foo']
        except NotImplementedError:
            pass
        else:
            assert 'foo' not in store
            with assert_raises(KeyError):
                store['foo']
            with assert_raises(KeyError):
                del store['foo']

    def test_writeable_values(self):
        store = self.create_store()

        # __setitem__ should accept any value that implements buffer interface
        store['foo'] = b'bar'
        store['foo'] = bytearray(b'bar')
        store['foo'] = array.array('B', b'bar')
        store['foo'] = np.frombuffer(b'bar', dtype='u1')

    def test_update(self):
        store = self.create_store()
        assert 'foo' not in store
        assert 'baz' not in store
        store.update(foo=b'bar', baz=b'quux')
        eq(b'bar', store['foo'])
        eq(b'quux', store['baz'])

    def test_iterators(self):
        store = self.create_store()

        # test iterator methods on empty store
        eq(0, len(store))
        eq(set(), set(store))
        eq(set(), set(store.keys()))
        eq(set(), set(store.values()))
        eq(set(), set(store.items()))

        # setup some values
        store['a'] = b'aaa'
        store['b'] = b'bbb'
        store['c/d'] = b'ddd'
        store['c/e/f'] = b'fff'

        # test iterators on store with data
        eq(4, len(store))
        eq({'a', 'b', 'c/d', 'c/e/f'}, set(store))
        eq({'a', 'b', 'c/d', 'c/e/f'}, set(store.keys()))
        eq({b'aaa', b'bbb', b'ddd', b'fff'}, set(store.values()))
        eq({('a', b'aaa'), ('b', b'bbb'), ('c/d', b'ddd'), ('c/e/f', b'fff')},
           set(store.items()))

    def test_pickle(self):
        store = self.create_store()
        store['foo'] = b'bar'
        store['baz'] = b'quux'
        if hasattr(store, 'flush'):
            store.flush()
        store2 = pickle.loads(pickle.dumps(store))
        eq(len(store), len(store2))
        eq(b'bar', store2['foo'])
        eq(b'quux', store2['baz'])
        eq(sorted(store.keys()), sorted(store2.keys()))
        for k in dir(store):
            v = getattr(store, k)
            if isinstance(v, (str, bool, int)):
                eq(v, getattr(store2, k))

    def test_getsize(self):
        store = self.create_store()
        if hasattr(store, 'getsize'):
            eq(0, store.getsize())
            store['foo'] = b'x'
            eq(1, store.getsize())
            eq(1, store.getsize('foo'))
            store['bar'] = b'yy'
            eq(3, store.getsize())
            eq(2, store.getsize('bar'))
            store['baz'] = bytearray(b'zzz')
            eq(6, store.getsize())
            eq(3, store.getsize('baz'))
            store['quux'] = array.array('B', b'zzzz')
            eq(10, store.getsize())
            eq(4, store.getsize('quux'))
            store['spong'] = np.frombuffer(b'zzzzz', dtype='u1')
            eq(15, store.getsize())
            eq(5, store.getsize('spong'))

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
        with assert_raises(KeyError):
            store['c']
        with assert_raises(KeyError):
            store['c/e']
        with assert_raises(KeyError):
            store['c/d/x']

        # test getsize (optional)
        if hasattr(store, 'getsize'):
            eq(6, store.getsize())
            eq(3, store.getsize('a'))
            eq(3, store.getsize('b'))
            eq(3, store.getsize('c'))
            eq(3, store.getsize('c/d'))
            eq(6, store.getsize('c/e'))
            eq(3, store.getsize('c/e/f'))
            eq(3, store.getsize('c/e/g'))
            with assert_raises(KeyError):
                store.getsize('x')
            with assert_raises(KeyError):
                store.getsize('a/x')
            with assert_raises(KeyError):
                store.getsize('c/x')
            with assert_raises(KeyError):
                store.getsize('c/x/y')
            with assert_raises(KeyError):
                store.getsize('c/d/y')
            with assert_raises(KeyError):
                store.getsize('c/d/y/z')

        # test listdir (optional)
        if hasattr(store, 'listdir'):
            eq({'a', 'b', 'c'}, set(store.listdir()))
            eq({'d', 'e'}, set(store.listdir('c')))
            eq({'f', 'g'}, set(store.listdir('c/e')))
            # no exception raised if path does not exist or is leaf
            eq([], store.listdir('x'))
            eq([], store.listdir('a/x'))
            eq([], store.listdir('c/x'))
            eq([], store.listdir('c/x/y'))
            eq([], store.listdir('c/d/y'))
            eq([], store.listdir('c/d/y/z'))
            eq([], store.listdir('c/e/f'))

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
        eq(ZARR_FORMAT, meta['zarr_format'])
        eq((1000,), meta['shape'])
        eq((100,), meta['chunks'])
        eq(np.dtype(None), meta['dtype'])
        eq(default_compressor.get_config(), meta['compressor'])
        assert_is_none(meta['fill_value'])

        # check attributes
        assert attrs_key in store
        eq(dict(), json.loads(text_type(store[attrs_key], 'ascii')))

    def test_init_array_overwrite(self):
        # setup
        store = self.create_store()
        store[array_meta_key] = encode_array_metadata(
            dict(shape=(2000,),
                 chunks=(200,),
                 dtype=np.dtype('u1'),
                 compressor=Zlib(1).get_config(),
                 fill_value=0,
                 order='F',
                 filters=None)
        )

        # don't overwrite (default)
        with assert_raises(KeyError):
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
            eq(ZARR_FORMAT, meta['zarr_format'])
            eq((1000,), meta['shape'])
            eq((100,), meta['chunks'])
            eq(np.dtype('i4'), meta['dtype'])

    def test_init_array_path(self):
        path = 'foo/bar'
        store = self.create_store()
        init_array(store, shape=1000, chunks=100, path=path)

        # check metadata
        key = path + '/' + array_meta_key
        assert key in store
        meta = decode_array_metadata(store[key])
        eq(ZARR_FORMAT, meta['zarr_format'])
        eq((1000,), meta['shape'])
        eq((100,), meta['chunks'])
        eq(np.dtype(None), meta['dtype'])
        eq(default_compressor.get_config(), meta['compressor'])
        assert_is_none(meta['fill_value'])

        # check attributes
        key = path + '/' + attrs_key
        assert key in store
        eq(dict(), json.loads(text_type(store[key], 'ascii')))

    def test_init_array_overwrite_path(self):
        # setup
        path = 'foo/bar'
        store = self.create_store()
        meta = dict(shape=(2000,),
                    chunks=(200,),
                    dtype=np.dtype('u1'),
                    compressor=Zlib(1).get_config(),
                    fill_value=0,
                    order='F',
                    filters=None)
        store[array_meta_key] = encode_array_metadata(meta)
        store[path + '/' + array_meta_key] = encode_array_metadata(meta)

        # don't overwrite
        with assert_raises(KeyError):
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
            eq(ZARR_FORMAT, meta['zarr_format'])
            eq((1000,), meta['shape'])
            eq((100,), meta['chunks'])
            eq(np.dtype('i4'), meta['dtype'])

    def test_init_array_overwrite_group(self):
        # setup
        path = 'foo/bar'
        store = self.create_store()
        store[path + '/' + group_meta_key] = encode_group_metadata()

        # don't overwrite
        with assert_raises(KeyError):
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
            eq(ZARR_FORMAT, meta['zarr_format'])
            eq((1000,), meta['shape'])
            eq((100,), meta['chunks'])
            eq(np.dtype('i4'), meta['dtype'])

    def test_init_array_overwrite_chunk_store(self):
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
                 order='F')
        )
        chunk_store['0'] = b'aaa'
        chunk_store['1'] = b'bbb'

        # don't overwrite (default)
        with assert_raises(KeyError):
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
            eq(ZARR_FORMAT, meta['zarr_format'])
            eq((1000,), meta['shape'])
            eq((100,), meta['chunks'])
            eq(np.dtype('i4'), meta['dtype'])
            assert '0' not in chunk_store
            assert '1' not in chunk_store

    def test_init_array_compat(self):
        store = self.create_store()
        init_array(store, shape=1000, chunks=100, compressor='none')
        meta = decode_array_metadata(store[array_meta_key])
        assert_is_none(meta['compressor'])

    def test_init_group(self):
        store = self.create_store()
        init_group(store)

        # check metadata
        assert group_meta_key in store
        meta = decode_group_metadata(store[group_meta_key])
        eq(ZARR_FORMAT, meta['zarr_format'])

        # check attributes
        assert attrs_key in store
        eq(dict(), json.loads(text_type(store[attrs_key], 'ascii')))

    def test_init_group_overwrite(self):
        # setup
        store = self.create_store()
        store[array_meta_key] = encode_array_metadata(
            dict(shape=(2000,),
                 chunks=(200,),
                 dtype=np.dtype('u1'),
                 compressor=None,
                 fill_value=0,
                 order='F',
                 filters=None)
        )

        # don't overwrite array (default)
        with assert_raises(KeyError):
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
            eq(ZARR_FORMAT, meta['zarr_format'])

        # don't overwrite group
        with assert_raises(KeyError):
            init_group(store)

    def test_init_group_overwrite_path(self):
        # setup
        path = 'foo/bar'
        store = self.create_store()
        meta = dict(shape=(2000,),
                    chunks=(200,),
                    dtype=np.dtype('u1'),
                    compressor=None,
                    fill_value=0,
                    order='F',
                    filters=None)
        store[array_meta_key] = encode_array_metadata(meta)
        store[path + '/' + array_meta_key] = encode_array_metadata(meta)

        # don't overwrite
        with assert_raises(KeyError):
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
            eq(ZARR_FORMAT, meta['zarr_format'])

    def test_init_group_overwrite_chunk_store(self):
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
                 order='F')
        )
        chunk_store['foo'] = b'bar'
        chunk_store['baz'] = b'quux'

        # don't overwrite array (default)
        with assert_raises(KeyError):
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
            eq(ZARR_FORMAT, meta['zarr_format'])
            assert 'foo' not in chunk_store
            assert 'baz' not in chunk_store

        # don't overwrite group
        with assert_raises(KeyError):
            init_group(store)


class TestMappingStore(StoreTests, unittest.TestCase):

    def create_store(self):
        return dict()


def setdel_hierarchy_checks(store):
    # these tests are for stores that are aware of hierarchy levels; this
    # behaviour is not stricly required by Zarr but these tests are included
    # to define behaviour of DictStore and DirectoryStore classes

    # check __setitem__ and __delitem__ blocked by leaf

    store['a/b'] = b'aaa'
    with assert_raises(KeyError):
        store['a/b/c'] = b'xxx'
    with assert_raises(KeyError):
        del store['a/b/c']

    store['d'] = b'ddd'
    with assert_raises(KeyError):
        store['d/e/f'] = b'xxx'
    with assert_raises(KeyError):
        del store['d/e/f']

    # test __setitem__ overwrite level
    store['x/y/z'] = b'xxx'
    store['x/y'] = b'yyy'
    eq(b'yyy', store['x/y'])
    assert 'x/y/z' not in store
    store['x'] = b'zzz'
    eq(b'zzz', store['x'])
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

    def test_setdel(self):
        store = self.create_store()
        setdel_hierarchy_checks(store)

    def test_getsize_ext(self):
        store = self.create_store()
        store['a'] = list(range(10))
        store['b/c'] = list(range(10))
        eq(-1, store.getsize())
        eq(-1, store.getsize('a'))
        eq(-1, store.getsize('b'))


def rmtree(p, f=shutil.rmtree, g=os.path.isdir):  # pragma: no cover
    """Version of rmtree that will work atexit and only remove if directory."""
    if g(p):
        f(p)


class TestDirectoryStore(StoreTests, unittest.TestCase):

    def create_store(self):
        path = tempfile.mkdtemp()
        atexit.register(rmtree, path)
        store = DirectoryStore(path)
        return store

    def test_filesystem_path(self):

        # test behaviour with path that does not exist
        path = 'example'
        if os.path.exists(path):
            shutil.rmtree(path)
        store = DirectoryStore(path)
        # should only be created on demand
        assert not os.path.exists(path)
        store['foo'] = b'bar'
        assert os.path.isdir(path)

        # test behaviour with file path
        with tempfile.NamedTemporaryFile() as f:
            with assert_raises(ValueError):
                DirectoryStore(f.name)

    def test_pickle_ext(self):
        store = self.create_store()
        store2 = pickle.loads(pickle.dumps(store))

        # check path is preserved
        eq(store.path, store2.path)

        # check point to same underlying directory
        assert 'xxx' not in store
        store2['xxx'] = b'yyy'
        eq(b'yyy', store['xxx'])

    def test_setdel(self):
        store = self.create_store()
        setdel_hierarchy_checks(store)


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
        store = ZipStore(path)
        return store

    def test_mode(self):
        with ZipStore('example.zip', mode='w') as store:
            store['foo'] = b'bar'
        store = ZipStore('example.zip', mode='r')
        with assert_raises(PermissionError):
            store['foo'] = b'bar'


def test_getsize():
    store = dict()
    store['foo'] = b'aaa'
    store['bar'] = b'bbbb'
    store['baz/quux'] = b'ccccc'
    eq(7, getsize(store))
    eq(5, getsize(store, 'baz'))


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
    eq(2, meta_migrated['zarr_format'])

    # preserved fields
    for f in 'shape', 'chunks', 'dtype', 'fill_value', 'order':
        eq(meta[f], meta_migrated[f])

    # migrate should have added empty filters field
    assert_is_none(meta_migrated['filters'])

    # check compression and compression_opts migrated to compressor
    assert 'compression' not in meta_migrated
    assert 'compression_opts' not in meta_migrated
    eq(meta_migrated['compressor'], Zlib(1).get_config())

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
    eq(meta_migrated['compressor'],
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
    assert_is_none(meta_migrated['compressor'])
