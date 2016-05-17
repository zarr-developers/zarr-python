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


from zarr.storage import DirectoryStore, init_store
from zarr.meta import decode_metadata
from zarr.compat import text_type


def test_init_store():

    store = dict()
    init_store(store, shape=1000, chunks=100)

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


def test_init_store_overwrite():

    store = dict(shape=(2000,), chunks=(200,))

    # overwrite
    init_store(store, shape=1000, chunks=100, overwrite=True)
    assert 'meta' in store
    meta = decode_metadata(store['meta'])
    eq((1000,), meta['shape'])
    eq((100,), meta['chunks'])

    # don't overwrite
    with assert_raises(ValueError):
        init_store(store, shape=1000, chunks=100, overwrite=False)


class MappingTests(object):

    def create_mapping(self, **kwargs):
        pass

    def test_get_set_del_contains(self):
        m = self.create_mapping()
        assert 'foo' not in m
        m['foo'] = b'bar'
        assert 'foo' in m
        eq(b'bar', m['foo'])
        del m['foo']
        assert 'foo' not in m
        with assert_raises(KeyError):
            m['foo']
        with assert_raises(KeyError):
            del m['foo']
        with assert_raises(TypeError):
            # non-writeable value
            m['foo'] = 42
        # alternative values
        m['foo'] = bytearray(b'bar')
        eq(b'bar', m['foo'])
        m['foo'] = array.array('B', b'bar')
        eq(b'bar', m['foo'])

    def test_update(self):
        m = self.create_mapping()
        assert 'foo' not in m
        assert 'baz' not in m
        m.update(foo=b'bar', baz=b'quux')
        eq(b'bar', m['foo'])
        eq(b'quux', m['baz'])

    def test_iterators(self):
        m = self.create_mapping()
        eq(0, len(m))
        eq(set(), set(m))
        eq(set(), set(m.keys()))
        eq(set(), set(m.values()))
        eq(set(), set(m.items()))

        m['foo'] = b'bar'
        m['baz'] = b'quux'

        eq(2, len(m))
        eq(set(['foo', 'baz']), set(m))
        eq(set(['foo', 'baz']), set(m.keys()))
        eq(set([b'bar', b'quux']), set(m.values()))
        eq(set([('foo', b'bar'), ('baz', b'quux')]), set(m.items()))


class TestDirectoryMap(MappingTests, unittest.TestCase):

    def create_mapping(self, **kwargs):
        path = tempfile.mkdtemp()
        atexit.register(shutil.rmtree, path)
        m = DirectoryStore(path, **kwargs)
        return m

    def test_size(self):
        m = self.create_mapping()
        eq(0, m.size)
        m['foo'] = b'bar'
        eq(3, m.size)
        m['baz'] = b'quux'
        eq(7, m.size)

    def test_path(self):
        with assert_raises(ValueError):
            DirectoryStore('doesnotexist')
        with tempfile.NamedTemporaryFile() as f:
            with assert_raises(ValueError):
                DirectoryStore(f.name)

    def test_pickle(self):
        m = self.create_mapping()
        m['foo'] = b'bar'
        m['baz'] = b'quux'
        m2 = pickle.loads(pickle.dumps(m))
        eq(len(m), len(m2))
        eq(m.path, m2.path)
        eq(b'bar', m2['foo'])
        eq(b'quux', m2['baz'])
        assert 'xxx' not in m
        m2['xxx'] = b'yyy'
        eq(b'yyy', m['xxx'])
