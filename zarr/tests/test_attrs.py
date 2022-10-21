import json

import pytest

from zarr._storage.store import meta_root
from zarr.attrs import Attributes
from zarr.storage import KVStore
from zarr._storage.v3 import KVStoreV3
from zarr.tests.util import CountingDict, CountingDictV3


@pytest.fixture(params=[2, 3])
def zarr_version(request):
    return request.param


def _init_store(version):
    """Use a plain dict() for v2, but KVStoreV3 otherwise."""
    if version == 2:
        return dict()
    return KVStoreV3(dict())


class TestAttributes():

    def init_attributes(self, store, read_only=False, cache=True, zarr_version=2):
        root = '.z' if zarr_version == 2 else meta_root
        return Attributes(store, key=root + 'attrs', read_only=read_only, cache=cache)

    def test_storage(self, zarr_version):

        store = _init_store(zarr_version)
        root = '.z' if zarr_version == 2 else meta_root
        attrs_key = root + 'attrs'
        a = Attributes(store=store, key=attrs_key)
        assert isinstance(a.store, KVStore)
        assert 'foo' not in a
        assert 'bar' not in a
        assert dict() == a.asdict()

        a['foo'] = 'bar'
        a['baz'] = 42
        assert attrs_key in store
        assert isinstance(store[attrs_key], bytes)
        d = json.loads(str(store[attrs_key], 'ascii'))
        if zarr_version == 3:
            d = d['attributes']
        assert dict(foo='bar', baz=42) == d

    def test_get_set_del_contains(self, zarr_version):

        store = _init_store(zarr_version)
        a = self.init_attributes(store, zarr_version=zarr_version)
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

    def test_update_put(self, zarr_version):

        store = _init_store(zarr_version)
        a = self.init_attributes(store, zarr_version=zarr_version)
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

    def test_iterators(self, zarr_version):

        store = _init_store(zarr_version)
        a = self.init_attributes(store, zarr_version=zarr_version)
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

    def test_read_only(self, zarr_version):
        store = _init_store(zarr_version)
        a = self.init_attributes(store, read_only=True, zarr_version=zarr_version)
        if zarr_version == 2:
            store['.zattrs'] = json.dumps(dict(foo='bar', baz=42)).encode('ascii')
        else:
            store['meta/root/attrs'] = json.dumps(
                    dict(attributes=dict(foo='bar', baz=42))
            ).encode('ascii')
        assert a['foo'] == 'bar'
        assert a['baz'] == 42
        with pytest.raises(PermissionError):
            a['foo'] = 'quux'
        with pytest.raises(PermissionError):
            del a['foo']
        with pytest.raises(PermissionError):
            a.update(foo='quux')

    def test_key_completions(self, zarr_version):
        store = _init_store(zarr_version)
        a = self.init_attributes(store, zarr_version=zarr_version)
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

    def test_caching_on(self, zarr_version):
        # caching is turned on by default

        # setup store
        store = CountingDict() if zarr_version == 2 else CountingDictV3()
        attrs_key = '.zattrs' if zarr_version == 2 else 'meta/root/attrs'
        assert 0 == store.counter['__getitem__', attrs_key]
        assert 0 == store.counter['__setitem__', attrs_key]
        if zarr_version == 2:
            store[attrs_key] = json.dumps(dict(foo='xxx', bar=42)).encode('ascii')
        else:
            store[attrs_key] = json.dumps(dict(attributes=dict(foo='xxx', bar=42))).encode('ascii')
        assert 0 == store.counter['__getitem__', attrs_key]
        assert 1 == store.counter['__setitem__', attrs_key]

        # setup attributes
        a = self.init_attributes(store, zarr_version=zarr_version)

        # test __getitem__ causes all attributes to be cached
        assert a['foo'] == 'xxx'
        assert 1 == store.counter['__getitem__', attrs_key]
        assert a['bar'] == 42
        assert 1 == store.counter['__getitem__', attrs_key]
        assert a['foo'] == 'xxx'
        assert 1 == store.counter['__getitem__', attrs_key]

        # test __setitem__ updates the cache
        a['foo'] = 'yyy'
        get_cnt = 2 if zarr_version == 2 else 3
        assert get_cnt == store.counter['__getitem__', attrs_key]
        assert 2 == store.counter['__setitem__', attrs_key]
        assert a['foo'] == 'yyy'
        assert get_cnt == store.counter['__getitem__', attrs_key]
        assert 2 == store.counter['__setitem__', attrs_key]

        # test update() updates the cache
        a.update(foo='zzz', bar=84)
        get_cnt = 3 if zarr_version == 2 else 5
        assert get_cnt == store.counter['__getitem__', attrs_key]
        assert 3 == store.counter['__setitem__', attrs_key]
        assert a['foo'] == 'zzz'
        assert a['bar'] == 84
        assert get_cnt == store.counter['__getitem__', attrs_key]
        assert 3 == store.counter['__setitem__', attrs_key]

        # test __contains__ uses the cache
        assert 'foo' in a
        assert get_cnt == store.counter['__getitem__', attrs_key]
        assert 3 == store.counter['__setitem__', attrs_key]
        assert 'spam' not in a
        assert get_cnt == store.counter['__getitem__', attrs_key]
        assert 3 == store.counter['__setitem__', attrs_key]

        # test __delitem__ updates the cache
        del a['bar']
        get_cnt = 4 if zarr_version == 2 else 7
        assert get_cnt == store.counter['__getitem__', attrs_key]
        assert 4 == store.counter['__setitem__', attrs_key]
        assert 'bar' not in a
        assert get_cnt == store.counter['__getitem__', attrs_key]
        assert 4 == store.counter['__setitem__', attrs_key]

        # test refresh()
        if zarr_version == 2:
            store[attrs_key] = json.dumps(dict(foo='xxx', bar=42)).encode('ascii')
        else:
            store[attrs_key] = json.dumps(dict(attributes=dict(foo='xxx', bar=42))).encode('ascii')
        assert get_cnt == store.counter['__getitem__', attrs_key]
        a.refresh()
        get_cnt = 5 if zarr_version == 2 else 8
        assert get_cnt == store.counter['__getitem__', attrs_key]
        assert a['foo'] == 'xxx'
        assert get_cnt == store.counter['__getitem__', attrs_key]
        assert a['bar'] == 42
        assert get_cnt == store.counter['__getitem__', attrs_key]

    def test_caching_off(self, zarr_version):

        # setup store
        store = CountingDict() if zarr_version == 2 else CountingDictV3()
        attrs_key = '.zattrs' if zarr_version == 2 else 'meta/root/attrs'
        assert 0 == store.counter['__getitem__', attrs_key]
        assert 0 == store.counter['__setitem__', attrs_key]

        if zarr_version == 2:
            store[attrs_key] = json.dumps(dict(foo='xxx', bar=42)).encode('ascii')
        else:
            store[attrs_key] = json.dumps(dict(attributes=dict(foo='xxx', bar=42))).encode('ascii')
        assert 0 == store.counter['__getitem__', attrs_key]
        assert 1 == store.counter['__setitem__', attrs_key]

        # setup attributes
        a = self.init_attributes(store, cache=False, zarr_version=zarr_version)

        # test __getitem__
        assert a['foo'] == 'xxx'
        assert 1 == store.counter['__getitem__', attrs_key]
        assert a['bar'] == 42
        assert 2 == store.counter['__getitem__', attrs_key]
        assert a['foo'] == 'xxx'
        assert 3 == store.counter['__getitem__', attrs_key]

        # test __setitem__
        a['foo'] = 'yyy'
        get_cnt = 4 if zarr_version == 2 else 5
        assert get_cnt == store.counter['__getitem__', attrs_key]
        assert 2 == store.counter['__setitem__', attrs_key]
        assert a['foo'] == 'yyy'
        get_cnt = 5 if zarr_version == 2 else 6
        assert get_cnt == store.counter['__getitem__', attrs_key]
        assert 2 == store.counter['__setitem__', attrs_key]

        # test update()
        a.update(foo='zzz', bar=84)
        get_cnt = 6 if zarr_version == 2 else 8
        assert get_cnt == store.counter['__getitem__', attrs_key]
        assert 3 == store.counter['__setitem__', attrs_key]
        assert a['foo'] == 'zzz'
        assert a['bar'] == 84
        get_cnt = 8 if zarr_version == 2 else 10
        assert get_cnt == store.counter['__getitem__', attrs_key]
        assert 3 == store.counter['__setitem__', attrs_key]

        # test __contains__
        assert 'foo' in a
        get_cnt = 9 if zarr_version == 2 else 11
        assert get_cnt == store.counter['__getitem__', attrs_key]
        assert 3 == store.counter['__setitem__', attrs_key]
        assert 'spam' not in a
        get_cnt = 10 if zarr_version == 2 else 12
        assert get_cnt == store.counter['__getitem__', attrs_key]
        assert 3 == store.counter['__setitem__', attrs_key]

    def test_wrong_keys(self, zarr_version):
        store = _init_store(zarr_version)
        a = self.init_attributes(store, zarr_version=zarr_version)

        warning_msg = "only attribute keys of type 'string' will be allowed in the future"

        with pytest.warns(DeprecationWarning, match=warning_msg):
            a[1] = "foo"

        with pytest.warns(DeprecationWarning, match=warning_msg):
            a.put({1: "foo"})

        with pytest.warns(DeprecationWarning, match=warning_msg):
            a.update({1: "foo"})
