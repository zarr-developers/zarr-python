# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import json
import unittest


from nose.tools import eq_ as eq, assert_raises


from zarr.attrs import Attributes
from zarr.compat import binary_type, text_type
from zarr.errors import ReadOnlyError


class TestAttributes(unittest.TestCase):

    def init_attributes(self, store, readonly=False):
        return Attributes(store, readonly=readonly)

    def test_storage(self):

        store = dict()
        a = self.init_attributes(store)
        assert 'attrs' in store
        assert isinstance(store['attrs'], binary_type)
        d = json.loads(text_type(store['attrs'], 'ascii'))
        eq(dict(), d)

        a['foo'] = 'bar'
        a['baz'] = 42

        d = json.loads(text_type(store['attrs'], 'ascii'))
        eq(dict(foo='bar', baz=42), d)

    def test_get_set_del_contains(self):

        a = self.init_attributes(dict())
        assert 'foo' not in a
        a['foo'] = 'bar'
        a['baz'] = 42
        assert 'foo' in a
        assert 'baz' in a
        eq('bar', a['foo'])
        eq(42, a['baz'])
        del a['foo']
        assert 'foo' not in a
        with assert_raises(KeyError):
            a['foo']

    def test_update(self):

        a = self.init_attributes(dict())
        assert 'foo' not in a
        assert 'baz' not in a
        a.update(foo='bar', baz=42)
        eq(a['foo'], 'bar')
        eq(a['baz'], 42)

    def test_iterators(self):

        a = self.init_attributes(dict())
        eq(0, len(a))
        eq(set(), set(a))
        eq(set(), set(a.keys()))
        eq(set(), set(a.values()))
        eq(set(), set(a.items()))

        a['foo'] = 'bar'
        a['baz'] = 42

        eq(2, len(a))
        eq({'foo', 'baz'}, set(a))
        eq({'foo', 'baz'}, set(a.keys()))
        eq({'bar', 42}, set(a.values()))
        eq({('foo', 'bar'), ('baz', 42)}, set(a.items()))

    def test_readonly(self):

        store = dict()
        store['attrs'] = json.dumps(dict(foo='bar', baz=42)).encode('ascii')
        a = self.init_attributes(store, readonly=True)
        eq(a['foo'], 'bar')
        eq(a['baz'], 42)
        with assert_raises(ReadOnlyError):
            a['foo'] = 'quux'
        with assert_raises(ReadOnlyError):
            del a['foo']
        with assert_raises(ReadOnlyError):
            a.update(foo='quux')
        with assert_raises(ReadOnlyError):
            a.put(dict())
