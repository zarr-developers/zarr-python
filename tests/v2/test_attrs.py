import json

import pytest

from zarr.v2.attrs import Attributes
from zarr.v2.storage import KVStore, DirectoryStore
from .util import CountingDict
from zarr.v2.hierarchy import group


def _init_store():
    return dict()


class TestAttributes:
    def init_attributes(self, store, read_only=False, cache=True):
        root = ".z"
        return Attributes(store, key=root + "attrs", read_only=read_only, cache=cache)

    def test_storage(self):
        store = _init_store()
        root = ".z"
        attrs_key = root + "attrs"
        a = Attributes(store=store, key=attrs_key)
        assert isinstance(a.store, KVStore)
        assert "foo" not in a
        assert "bar" not in a
        assert dict() == a.asdict()

        a["foo"] = "bar"
        a["baz"] = 42
        assert attrs_key in store
        assert isinstance(store[attrs_key], bytes)
        d = json.loads(str(store[attrs_key], "utf-8"))
        assert dict(foo="bar", baz=42) == d

    def test_utf8_encoding(self, project_root):
        fixdir = project_root / "fixture"
        testdir = fixdir / "utf8attrs"
        if not testdir.exists():  # pragma: no cover
            # store the data - should be one-time operation
            testdir.mkdir(parents=True, exist_ok=True)
            with (testdir / ".zattrs").open("w", encoding="utf-8") as f:
                f.write('{"foo": "た"}')
            with (testdir / ".zgroup").open("w", encoding="utf-8") as f:
                f.write("""{\n    "zarr_format": 2\n}""")

        # fixture data
        fixture = group(store=DirectoryStore(str(fixdir)))
        assert fixture["utf8attrs"].attrs.asdict() == dict(foo="た")

    def test_get_set_del_contains(self):
        store = _init_store()
        a = self.init_attributes(store)
        assert "foo" not in a
        a["foo"] = "bar"
        a["baz"] = 42
        assert "foo" in a
        assert "baz" in a
        assert "bar" == a["foo"]
        assert 42 == a["baz"]
        del a["foo"]
        assert "foo" not in a
        with pytest.raises(KeyError):
            # noinspection PyStatementEffect
            a["foo"]

    def test_update_put(self):
        store = _init_store()
        a = self.init_attributes(store)
        assert "foo" not in a
        assert "bar" not in a
        assert "baz" not in a

        a.update(foo="spam", bar=42, baz=4.2)
        assert a["foo"] == "spam"
        assert a["bar"] == 42
        assert a["baz"] == 4.2

        a.put(dict(foo="eggs", bar=84))
        assert a["foo"] == "eggs"
        assert a["bar"] == 84
        assert "baz" not in a

    def test_iterators(self):
        store = _init_store()
        a = self.init_attributes(store)
        assert 0 == len(a)
        assert set() == set(a)
        assert set() == set(a.keys())
        assert set() == set(a.values())
        assert set() == set(a.items())

        a["foo"] = "bar"
        a["baz"] = 42

        assert 2 == len(a)
        assert {"foo", "baz"} == set(a)
        assert {"foo", "baz"} == set(a.keys())
        assert {"bar", 42} == set(a.values())
        assert {("foo", "bar"), ("baz", 42)} == set(a.items())

    def test_read_only(self):
        store = _init_store()
        a = self.init_attributes(store, read_only=True)
        store[".zattrs"] = json.dumps(dict(foo="bar", baz=42)).encode("ascii")
        assert a["foo"] == "bar"
        assert a["baz"] == 42
        with pytest.raises(PermissionError):
            a["foo"] = "quux"
        with pytest.raises(PermissionError):
            del a["foo"]
        with pytest.raises(PermissionError):
            a.update(foo="quux")

    def test_key_completions(self):
        store = _init_store()
        a = self.init_attributes(store)
        d = a._ipython_key_completions_()
        assert "foo" not in d
        assert "123" not in d
        assert "baz" not in d
        assert "asdf;" not in d
        a["foo"] = 42
        a["123"] = 4.2
        a["asdf;"] = "ghjkl;"
        d = a._ipython_key_completions_()
        assert "foo" in d
        assert "123" in d
        assert "asdf;" in d
        assert "baz" not in d

    def test_caching_on(self):
        # caching is turned on by default

        # setup store
        store = CountingDict()
        attrs_key = ".zattrs"
        assert 0 == store.counter["__getitem__", attrs_key]
        assert 0 == store.counter["__setitem__", attrs_key]
        store[attrs_key] = json.dumps(dict(foo="xxx", bar=42)).encode("ascii")
        assert 0 == store.counter["__getitem__", attrs_key]
        assert 1 == store.counter["__setitem__", attrs_key]

        # setup attributes
        a = self.init_attributes(store)

        # test __getitem__ causes all attributes to be cached
        assert a["foo"] == "xxx"
        assert 1 == store.counter["__getitem__", attrs_key]
        assert a["bar"] == 42
        assert 1 == store.counter["__getitem__", attrs_key]
        assert a["foo"] == "xxx"
        assert 1 == store.counter["__getitem__", attrs_key]

        # test __setitem__ updates the cache
        a["foo"] = "yyy"
        get_cnt = 2
        assert get_cnt == store.counter["__getitem__", attrs_key]
        assert 2 == store.counter["__setitem__", attrs_key]
        assert a["foo"] == "yyy"
        assert get_cnt == store.counter["__getitem__", attrs_key]
        assert 2 == store.counter["__setitem__", attrs_key]

        # test update() updates the cache
        a.update(foo="zzz", bar=84)
        get_cnt = 3
        assert get_cnt == store.counter["__getitem__", attrs_key]
        assert 3 == store.counter["__setitem__", attrs_key]
        assert a["foo"] == "zzz"
        assert a["bar"] == 84
        assert get_cnt == store.counter["__getitem__", attrs_key]
        assert 3 == store.counter["__setitem__", attrs_key]

        # test __contains__ uses the cache
        assert "foo" in a
        assert get_cnt == store.counter["__getitem__", attrs_key]
        assert 3 == store.counter["__setitem__", attrs_key]
        assert "spam" not in a
        assert get_cnt == store.counter["__getitem__", attrs_key]
        assert 3 == store.counter["__setitem__", attrs_key]

        # test __delitem__ updates the cache
        del a["bar"]
        get_cnt = 4
        assert get_cnt == store.counter["__getitem__", attrs_key]
        assert 4 == store.counter["__setitem__", attrs_key]
        assert "bar" not in a
        assert get_cnt == store.counter["__getitem__", attrs_key]
        assert 4 == store.counter["__setitem__", attrs_key]

        # test refresh()
        store[attrs_key] = json.dumps(dict(foo="xxx", bar=42)).encode("ascii")
        assert get_cnt == store.counter["__getitem__", attrs_key]
        a.refresh()
        get_cnt = 5
        assert get_cnt == store.counter["__getitem__", attrs_key]
        assert a["foo"] == "xxx"
        assert get_cnt == store.counter["__getitem__", attrs_key]
        assert a["bar"] == 42
        assert get_cnt == store.counter["__getitem__", attrs_key]

    def test_caching_off(self):
        # setup store
        store = CountingDict()
        attrs_key = ".zattrs"
        assert 0 == store.counter["__getitem__", attrs_key]
        assert 0 == store.counter["__setitem__", attrs_key]
        store[attrs_key] = json.dumps(dict(foo="xxx", bar=42)).encode("ascii")
        assert 0 == store.counter["__getitem__", attrs_key]
        assert 1 == store.counter["__setitem__", attrs_key]

        # setup attributes
        a = self.init_attributes(store, cache=False)

        # test __getitem__
        assert a["foo"] == "xxx"
        assert 1 == store.counter["__getitem__", attrs_key]
        assert a["bar"] == 42
        assert 2 == store.counter["__getitem__", attrs_key]
        assert a["foo"] == "xxx"
        assert 3 == store.counter["__getitem__", attrs_key]

        # test __setitem__
        a["foo"] = "yyy"
        get_cnt = 4
        assert get_cnt == store.counter["__getitem__", attrs_key]
        assert 2 == store.counter["__setitem__", attrs_key]
        assert a["foo"] == "yyy"
        get_cnt = 5
        assert get_cnt == store.counter["__getitem__", attrs_key]
        assert 2 == store.counter["__setitem__", attrs_key]

        # test update()
        a.update(foo="zzz", bar=84)
        get_cnt = 6
        assert get_cnt == store.counter["__getitem__", attrs_key]
        assert 3 == store.counter["__setitem__", attrs_key]
        assert a["foo"] == "zzz"
        assert a["bar"] == 84
        get_cnt = 8
        assert get_cnt == store.counter["__getitem__", attrs_key]
        assert 3 == store.counter["__setitem__", attrs_key]

        # test __contains__
        assert "foo" in a
        get_cnt = 9
        assert get_cnt == store.counter["__getitem__", attrs_key]
        assert 3 == store.counter["__setitem__", attrs_key]
        assert "spam" not in a
        get_cnt = 10
        assert get_cnt == store.counter["__getitem__", attrs_key]
        assert 3 == store.counter["__setitem__", attrs_key]

    def test_wrong_keys(self):
        store = _init_store()
        a = self.init_attributes(store)

        warning_msg = "only attribute keys of type 'string' will be allowed in the future"

        with pytest.warns(DeprecationWarning, match=warning_msg):
            a[1] = "foo"

        with pytest.warns(DeprecationWarning, match=warning_msg):
            a.put({1: "foo"})

        with pytest.warns(DeprecationWarning, match=warning_msg):
            a.update({1: "foo"})
