# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import unittest
import tempfile
import atexit
import shutil
import pickle
import json
import array


import numpy as np
from nose.tools import assert_raises, eq_ as eq, assert_is_none


from zarr.storage import DirectoryStore, MemoryStore, init_array
from zarr.meta import decode_metadata
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

    store = dict(shape=(2000,), chunks=(200,))

    # overwrite
    init_array(store, shape=1000, chunks=100, overwrite=True)
    assert 'meta' in store
    meta = decode_metadata(store['meta'])
    eq((1000,), meta['shape'])
    eq((100,), meta['chunks'])

    # don't overwrite
    with assert_raises(ValueError):
        init_array(store, shape=1000, chunks=100, overwrite=False)


class StoreTests(object):

    def create_store(self, **kwargs):
        pass

    def test_get_set_del_contains(self):
        store = self.create_store()
        assert 'foo' not in store
        store['foo'] = b'bar'
        assert 'foo' in store
        eq(b'bar', store['foo'])
        del store['foo']
        assert 'foo' not in store
        with assert_raises(KeyError):
            store['foo']
        with assert_raises(KeyError):
            del store['foo']
        with assert_raises(TypeError):
            # non-writeable value
            store['foo'] = 42
        # alternative values
        store['foo'] = bytearray(b'bar')
        eq(b'bar', store['foo'])
        store['foo'] = array.array('B', b'bar')
        eq(b'bar', store['foo'])

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

        store['foo'] = b'bar'
        store['baz'] = b'quux'

        eq(2, len(store))
        eq(set(['foo', 'baz']), set(store))
        eq(set(['foo', 'baz']), set(store.keys()))
        eq(set([b'bar', b'quux']), set(store.values()))
        eq(set([('foo', b'bar'), ('baz', b'quux')]), set(store.items()))

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


class TestMemoryStore(StoreTests, unittest.TestCase):

    def create_store(self, **kwargs):
        return MemoryStore(**kwargs)


class TestDirectoryStore(StoreTests, unittest.TestCase):

    def create_store(self, **kwargs):
        path = tempfile.mkdtemp()
        atexit.register(shutil.rmtree, path)
        store = DirectoryStore(path, **kwargs)
        return store

    def test_path(self):

        # test behaviour with path that does not exist
        with assert_raises(ValueError):
            DirectoryStore('doesnotexist', readonly=True)

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
