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


from zarr.storage import DirectoryStore, DictStore, ZipStore, init_array, \
    listdir, rmdir
from zarr.meta import decode_metadata, encode_metadata
from zarr.compat import text_type


def test_init_array():

    store = dict()
    init_array(store, shape=1000, chunks=100)

    # check metadata
    assert 'meta' in store
    meta = decode_metadata(store['meta'])
    eq((1000,), meta['shape'])
    eq((100,), meta['chunks'])
    eq(np.dtype(None), meta['dtype'])
    eq('blosc', meta['compression'])
    assert 'compression_opts' in meta
    assert_is_none(meta['fill_value'])

    # check attributes
    assert 'attrs' in store
    eq(dict(), json.loads(text_type(store['attrs'], 'ascii')))


def test_init_array_overwrite():

    store = dict()
    store['meta'] = encode_metadata(dict(shape=(2000,),
                                         chunks=(200,),
                                         dtype=np.dtype('u1'),
                                         compression='zlib',
                                         compression_opts=1,
                                         fill_value=0,
                                         order='F'))

    # overwrite
    init_array(store, shape=1000, chunks=100, dtype='i4', overwrite=True)
    assert 'meta' in store
    meta = decode_metadata(store['meta'])
    eq((1000,), meta['shape'])
    eq((100,), meta['chunks'])
    eq(np.dtype('i4'), meta['dtype'])

    # don't overwrite
    with assert_raises(ValueError):
        init_array(store, shape=1000, chunks=100, overwrite=False)


def test_init_array_prefix():

    store = dict()
    init_array(store, shape=1000, chunks=100, prefix='foo/bar')

    # check metadata
    assert 'foo/bar/meta' in store
    meta = decode_metadata(store['foo/bar/meta'])
    eq((1000,), meta['shape'])
    eq((100,), meta['chunks'])
    eq(np.dtype(None), meta['dtype'])
    eq('blosc', meta['compression'])
    assert 'compression_opts' in meta
    assert_is_none(meta['fill_value'])

    # check attributes
    assert 'foo/bar/attrs' in store
    eq(dict(), json.loads(text_type(store['foo/bar/attrs'], 'ascii')))


def test_init_array_overwrite_prefix():

    store = dict()
    meta = dict(shape=(2000,),
                chunks=(200,),
                dtype=np.dtype('u1'),
                compression='zlib',
                compression_opts=1,
                fill_value=0,
                order='F')
    store['meta'] = encode_metadata(meta)
    store['foo/bar/meta'] = encode_metadata(meta)

    # overwrite
    init_array(store, shape=1000, chunks=100, dtype='i4', overwrite=True,
               prefix='foo/bar')
    assert 'meta' in store
    assert 'foo/bar/meta' in store
    # should have been overwritten
    meta = decode_metadata(store['foo/bar/meta'])
    eq((1000,), meta['shape'])
    eq((100,), meta['chunks'])
    eq(np.dtype('i4'), meta['dtype'])
    # should have been left untouched
    meta = decode_metadata(store['meta'])
    eq((2000,), meta['shape'])
    eq((200,), meta['chunks'])
    eq(np.dtype('u1'), meta['dtype'])

    # don't overwrite
    with assert_raises(ValueError):
        init_array(store, shape=1000, chunks=100, overwrite=False,
                   prefix='foo/bar')


class StoreTests(object):

    def create_store(self, **kwargs):
        pass

    def test_get_set_del_contains(self):
        store = self.create_store()

        assert 'foo' not in store
        store['foo'] = b'bar'
        assert 'foo' in store
        eq(b'bar', store['foo'])

        try:
            del store['foo']
            assert 'foo' not in store
            with assert_raises(KeyError):
                store['foo']
            with assert_raises(KeyError):
                del store['foo']
        except NotImplementedError:
            pass

    def test_writeable_values(self):
        store = self.create_store()
        # store should accept anything that implements buffer interface
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
        eq(0, len(store))
        eq(set(), set(store))
        eq(set(), set(store.keys()))
        eq(set(), set(store.values()))
        eq(set(), set(store.items()))

        store['a'] = b'xxx'
        store['b'] = b'yyy'
        store['c/d'] = b'zzz'

        eq(3, len(store))
        eq({'a', 'b', 'c/d'}, set(store))
        eq({'a', 'b', 'c/d'}, set(store.keys()))
        eq({b'xxx', b'yyy', b'zzz'}, set(store.values()))
        eq({('a', b'xxx'), ('b', b'yyy'), ('c/d', b'zzz')},
           set(store.items()))

    def test_nbytes_stored(self):
        store = self.create_store()
        if hasattr(store, 'nbytes_stored'):
            eq(0, store.nbytes_stored)
            store['foo'] = b'bar'
            eq(3, store.nbytes_stored)
            store['baz'] = b'quux'
            eq(7, store.nbytes_stored)

    def test_pickle(self):
        store = self.create_store()
        store['foo'] = b'bar'
        store['baz'] = b'quux'
        store2 = pickle.loads(pickle.dumps(store))
        eq(len(store), len(store2))
        eq(b'bar', store2['foo'])
        eq(b'quux', store2['baz'])
        eq(list(store.keys()), list(store2.keys()))
        for k in dir(store):
            v = getattr(store, k)
            if not callable(v):
                eq(v, getattr(store2, k))

    def test_listdir_rmdir(self):
        # setup
        store = self.create_store()
        store['a'] = b'aaa'
        store['b'] = b'bbb'
        store['c/d'] = b'ddd'
        store['c/e/f'] = b'fff'

        # test listdir
        eq({'a', 'b', 'c'}, set(listdir(store)))
        eq({'d', 'e'}, set(listdir(store, 'c')))
        eq({'f'}, set(listdir(store, 'c/e')))
        assert 'a' in store
        assert 'b' in store
        assert 'c' not in store
        assert 'c/d' in store
        assert 'c/e' not in store
        assert 'c/e/f' in store

        # test rmdir
        try:
            rmdir(store, 'c/e')
            assert 'c/d' in store
            assert 'c/e/f' not in store
            eq({'d'}, set(listdir(store, 'c')))
            rmdir(store, 'c')
            assert 'c/d' not in store
            eq({'a', 'b'}, set(listdir(store)))
            rmdir(store)
            assert 'a' not in store
            assert 'b' not in store
            eq([], listdir(store))
        except NotImplementedError:
            pass


class TestGenericStore(StoreTests, unittest.TestCase):

    def create_store(self):
        return dict()


class TestDictStore(StoreTests, unittest.TestCase):

    def create_store(self):
        return DictStore()


def rmtree_if_exists(path, rmtree=shutil.rmtree, isdir=os.path.isdir):
    if isdir(path):
        rmtree(path)


class TestDirectoryStore(StoreTests, unittest.TestCase):

    def create_store(self):
        path = tempfile.mkdtemp()
        atexit.register(rmtree_if_exists, path)
        store = DirectoryStore(path)
        return store

    def test_path(self):

        # test behaviour with path that does not exist
        path = 'doesnotexist'
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


class TestZipStore(StoreTests, unittest.TestCase):

    def create_store(self):
        path = tempfile.mktemp(suffix='.zip')
        atexit.register(os.remove, path)
        store = ZipStore(path)
        return store
