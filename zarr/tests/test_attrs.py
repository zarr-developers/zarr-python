# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import json
import unittest


import pytest


from zarr.attrs import Attributes
from zarr.compat import binary_type, text_type
from zarr.errors import PermissionError
from zarr.tests.util import CountingDict


class TestAttributes(unittest.TestCase):

    def init_attributes(self, store, read_only=False, cache=True):
        return Attributes(store, key='attrs', read_only=read_only, cache=cache)

    def test_storage(self):

        store = dict()
        a = Attributes(store=store, key='attrs')
        assert 'foo' not in a
        assert 'bar' not in a
        assert dict() == a.asdict()

        a['foo'] = 'bar'
        a['baz'] = 42
        assert 'attrs' in store
        assert isinstance(store['attrs'], binary_type)
        d = json.loads(text_type(store['attrs'], 'ascii'))
        assert dict(foo='bar', baz=42) == d

    def test_get_set_del_contains(self):

        a = self.init_attributes(dict())
        assert 'foo' not in a
        a['foo'] = 'bar'
        a['baz'] = 42
        assert 'foo' in a
        assert 'baz' in a
        assert 'bar' == a['foo']
        assert 42 == a['baz']
        del a['foo']
        assert 'foo' not in a
        with pytest.raises(KeyError):
            # noinspection PyStatementEffect
            a['foo']

    def test_update_put(self):

        a = self.init_attributes(dict())
        assert 'foo' not in a
        assert 'bar' not in a
        assert 'baz' not in a

        a.update(foo='spam', bar=42, baz=4.2)
        assert a['foo'] == 'spam'
        assert a['bar'] == 42
        assert a['baz'] == 4.2

        a.put(dict(foo='eggs', bar=84))
        assert a['foo'] == 'eggs'
        assert a['bar'] == 84
        assert 'baz' not in a

    def test_iterators(self):

        a = self.init_attributes(dict())
        assert 0 == len(a)
        assert set() == set(a)
        assert set() == set(a.keys())
        assert set() == set(a.values())
        assert set() == set(a.items())

        a['foo'] = 'bar'
        a['baz'] = 42

        assert 2 == len(a)
        assert {'foo', 'baz'} == set(a)
        assert {'foo', 'baz'} == set(a.keys())
        assert {'bar', 42} == set(a.values())
        assert {('foo', 'bar'), ('baz', 42)} == set(a.items())

    def test_read_only(self):
        store = dict()
        a = self.init_attributes(store, read_only=True)
        store['attrs'] = json.dumps(dict(foo='bar', baz=42)).encode('ascii')
        assert a['foo'] == 'bar'
        assert a['baz'] == 42
        with pytest.raises(PermissionError):
            a['foo'] = 'quux'
        with pytest.raises(PermissionError):
            del a['foo']
        with pytest.raises(PermissionError):
            a.update(foo='quux')

    def test_key_completions(self):
        a = self.init_attributes(dict())
        d = a._ipython_key_completions_()
        assert 'foo' not in d
        assert '123' not in d
        assert 'baz' not in d
        assert 'asdf;' not in d
        a['foo'] = 42
        a['123'] = 4.2
        a['asdf;'] = 'ghjkl;'
        d = a._ipython_key_completions_()
        assert 'foo' in d
        assert '123' in d
        assert 'asdf;' in d
        assert 'baz' not in d

    def test_caching_on(self):
        # caching is turned on by default

        # setup store
        store = CountingDict()
        assert 0 == store.counter['__getitem__', 'attrs']
        assert 0 == store.counter['__setitem__', 'attrs']
        store['attrs'] = json.dumps(dict(foo='xxx', bar=42)).encode('ascii')
        assert 0 == store.counter['__getitem__', 'attrs']
        assert 1 == store.counter['__setitem__', 'attrs']

        # setup attributes
        a = self.init_attributes(store)

        # test __getitem__ causes all attributes to be cached
        assert a['foo'] == 'xxx'
        assert 1 == store.counter['__getitem__', 'attrs']
        assert a['bar'] == 42
        assert 1 == store.counter['__getitem__', 'attrs']
        assert a['foo'] == 'xxx'
        assert 1 == store.counter['__getitem__', 'attrs']

        # test __setitem__ updates the cache
        a['foo'] = 'yyy'
        assert 2 == store.counter['__getitem__', 'attrs']
        assert 2 == store.counter['__setitem__', 'attrs']
        assert a['foo'] == 'yyy'
        assert 2 == store.counter['__getitem__', 'attrs']
        assert 2 == store.counter['__setitem__', 'attrs']

        # test update() updates the cache
        a.update(foo='zzz', bar=84)
        assert 3 == store.counter['__getitem__', 'attrs']
        assert 3 == store.counter['__setitem__', 'attrs']
        assert a['foo'] == 'zzz'
        assert a['bar'] == 84
        assert 3 == store.counter['__getitem__', 'attrs']
        assert 3 == store.counter['__setitem__', 'attrs']

        # test __contains__ uses the cache
        assert 'foo' in a
        assert 3 == store.counter['__getitem__', 'attrs']
        assert 3 == store.counter['__setitem__', 'attrs']
        assert 'spam' not in a
        assert 3 == store.counter['__getitem__', 'attrs']
        assert 3 == store.counter['__setitem__', 'attrs']

        # test __delitem__ updates the cache
        del a['bar']
        assert 4 == store.counter['__getitem__', 'attrs']
        assert 4 == store.counter['__setitem__', 'attrs']
        assert 'bar' not in a
        assert 4 == store.counter['__getitem__', 'attrs']
        assert 4 == store.counter['__setitem__', 'attrs']

        # test refresh()
        store['attrs'] = json.dumps(dict(foo='xxx', bar=42)).encode('ascii')
        assert 4 == store.counter['__getitem__', 'attrs']
        a.refresh()
        assert 5 == store.counter['__getitem__', 'attrs']
        assert a['foo'] == 'xxx'
        assert 5 == store.counter['__getitem__', 'attrs']
        assert a['bar'] == 42
        assert 5 == store.counter['__getitem__', 'attrs']

    def test_caching_off(self):

        # setup store
        store = CountingDict()
        assert 0 == store.counter['__getitem__', 'attrs']
        assert 0 == store.counter['__setitem__', 'attrs']
        store['attrs'] = json.dumps(dict(foo='xxx', bar=42)).encode('ascii')
        assert 0 == store.counter['__getitem__', 'attrs']
        assert 1 == store.counter['__setitem__', 'attrs']

        # setup attributes
        a = self.init_attributes(store, cache=False)

        # test __getitem__
        assert a['foo'] == 'xxx'
        assert 1 == store.counter['__getitem__', 'attrs']
        assert a['bar'] == 42
        assert 2 == store.counter['__getitem__', 'attrs']
        assert a['foo'] == 'xxx'
        assert 3 == store.counter['__getitem__', 'attrs']

        # test __setitem__
        a['foo'] = 'yyy'
        assert 4 == store.counter['__getitem__', 'attrs']
        assert 2 == store.counter['__setitem__', 'attrs']
        assert a['foo'] == 'yyy'
        assert 5 == store.counter['__getitem__', 'attrs']
        assert 2 == store.counter['__setitem__', 'attrs']

        # test update()
        a.update(foo='zzz', bar=84)
        assert 6 == store.counter['__getitem__', 'attrs']
        assert 3 == store.counter['__setitem__', 'attrs']
        assert a['foo'] == 'zzz'
        assert a['bar'] == 84
        assert 8 == store.counter['__getitem__', 'attrs']
        assert 3 == store.counter['__setitem__', 'attrs']

        # test __contains__
        assert 'foo' in a
        assert 9 == store.counter['__getitem__', 'attrs']
        assert 3 == store.counter['__setitem__', 'attrs']
        assert 'spam' not in a
        assert 10 == store.counter['__getitem__', 'attrs']
        assert 3 == store.counter['__setitem__', 'attrs']
