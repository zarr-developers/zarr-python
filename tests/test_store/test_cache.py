from __future__ import annotations

import pytest
from collections import Counter
from typing import Any

from zarr.core.buffer import cpu
from zarr.storage import LRUStoreCache, MemoryStore
from zarr.testing.store import StoreTests


class CountingDict(dict):
    """A dictionary that counts operations for testing purposes."""
    
    def __init__(self):
        super().__init__()
        self.counter = Counter()
    
    def __getitem__(self, key):
        self.counter["__getitem__", key] += 1
        return super().__getitem__(key)
    
    def __setitem__(self, key, value):
        self.counter["__setitem__", key] += 1
        return super().__setitem__(key, value)
    
    def __contains__(self, key):
        self.counter["__contains__", key] += 1
        return super().__contains__(key)
    
    def __iter__(self):
        self.counter["__iter__"] += 1
        return super().__iter__()
    
    def keys(self):
        self.counter["keys"] += 1
        return super().keys()


def skip_if_nested_chunks(**kwargs):
    if kwargs.get("dimension_separator") == "/":
        pytest.skip("nested chunks are unsupported")

class TestLRUStoreCache(StoreTests[LRUStoreCache, cpu.Buffer]):
    store_cls = LRUStoreCache
    buffer_cls = cpu.buffer_prototype.buffer
    CountingClass = CountingDict
    LRUStoreClass = LRUStoreCache
    root = ""

    async def get(self, store: LRUStoreCache, key: str) -> cpu.Buffer:
        """Get method required by StoreTests."""
        return await store.get(key, prototype=cpu.buffer_prototype)

    async def set(self, store: LRUStoreCache, key: str, value: cpu.Buffer) -> None:
        """Set method required by StoreTests."""
        await store.set(key, value)

    @pytest.fixture
    def store_kwargs(self):
        """Provide default kwargs for store creation."""
        return {"store": MemoryStore(), "max_size": 2**27}

    @pytest.fixture
    async def store(self, store_kwargs: dict[str, Any]) -> LRUStoreCache:
        """Override store fixture to use constructor instead of open."""
        return self.store_cls(**store_kwargs)

    @pytest.fixture
    def open_kwargs(self):
        """Provide default kwargs for store.open()."""
        return {"store": MemoryStore(), "max_size": 2**27}

    def create_store(self, **kwargs):
        # wrapper therefore no dimension_separator argument
        skip_if_nested_chunks(**kwargs)
        return self.LRUStoreClass(MemoryStore(), max_size=2**27)

    def create_store_from_mapping(self, mapping, **kwargs):
        # Handle creation from existing mapping
        skip_if_nested_chunks(**kwargs)
        # Create a MemoryStore from the mapping
        underlying_store = MemoryStore()
        if mapping:
            # Convert mapping to store data
            for k, v in mapping.items():
                underlying_store._store_dict[k] = v
        return self.LRUStoreClass(underlying_store, max_size=2**27)

    def test_cache_values_no_max_size(self):
        # setup store
        store = self.CountingClass()
        foo_key = self.root + "foo"
        bar_key = self.root + "bar"
        store[foo_key] = b"xxx"
        store[bar_key] = b"yyy"
        assert 0 == store.counter["__getitem__", foo_key]
        assert 1 == store.counter["__setitem__", foo_key]
        assert 0 == store.counter["__getitem__", bar_key]
        assert 1 == store.counter["__setitem__", bar_key]

        # setup cache
        cache = self.LRUStoreClass(store, max_size=None)
        assert 0 == cache.hits
        assert 0 == cache.misses

        # test first __getitem__, cache miss
        assert b"xxx" == cache[foo_key]
        assert 1 == store.counter["__getitem__", foo_key]
        assert 1 == store.counter["__setitem__", foo_key]
        assert 0 == cache.hits
        assert 1 == cache.misses

        # test second __getitem__, cache hit
        assert b"xxx" == cache[foo_key]
        assert 1 == store.counter["__getitem__", foo_key]
        assert 1 == store.counter["__setitem__", foo_key]
        assert 1 == cache.hits
        assert 1 == cache.misses

        # test __setitem__, __getitem__
        cache[foo_key] = b"zzz"
        assert 1 == store.counter["__getitem__", foo_key]
        assert 2 == store.counter["__setitem__", foo_key]
        # should be a cache hit
        assert b"zzz" == cache[foo_key]
        assert 1 == store.counter["__getitem__", foo_key]
        assert 2 == store.counter["__setitem__", foo_key]
        assert 2 == cache.hits
        assert 1 == cache.misses

        # manually invalidate all cached values
        cache.invalidate_values()
        assert b"zzz" == cache[foo_key]
        assert 2 == store.counter["__getitem__", foo_key]
        assert 2 == store.counter["__setitem__", foo_key]
        cache.invalidate()
        assert b"zzz" == cache[foo_key]
        assert 3 == store.counter["__getitem__", foo_key]
        assert 2 == store.counter["__setitem__", foo_key]

        # test __delitem__
        del cache[foo_key]
        with pytest.raises(KeyError):
            # noinspection PyStatementEffect
            cache[foo_key]
        with pytest.raises(KeyError):
            # noinspection PyStatementEffect
            store[foo_key]

        # verify other keys untouched
        assert 0 == store.counter["__getitem__", bar_key]
        assert 1 == store.counter["__setitem__", bar_key]

    def test_cache_values_with_max_size(self):
        # setup store
        store = self.CountingClass()
        foo_key = self.root + "foo"
        bar_key = self.root + "bar"
        store[foo_key] = b"xxx"
        store[bar_key] = b"yyy"
        assert 0 == store.counter["__getitem__", foo_key]
        assert 0 == store.counter["__getitem__", bar_key]
        # setup cache - can only hold one item
        cache = self.LRUStoreClass(store, max_size=5)
        assert 0 == cache.hits
        assert 0 == cache.misses

        # test first 'foo' __getitem__, cache miss
        assert b"xxx" == cache[foo_key]
        assert 1 == store.counter["__getitem__", foo_key]
        assert 0 == cache.hits
        assert 1 == cache.misses

        # test second 'foo' __getitem__, cache hit
        assert b"xxx" == cache[foo_key]
        assert 1 == store.counter["__getitem__", foo_key]
        assert 1 == cache.hits
        assert 1 == cache.misses

        # test first 'bar' __getitem__, cache miss
        assert b"yyy" == cache[bar_key]
        assert 1 == store.counter["__getitem__", bar_key]
        assert 1 == cache.hits
        assert 2 == cache.misses

        # test second 'bar' __getitem__, cache hit
        assert b"yyy" == cache[bar_key]
        assert 1 == store.counter["__getitem__", bar_key]
        assert 2 == cache.hits
        assert 2 == cache.misses

        # test 'foo' __getitem__, should have been evicted, cache miss
        assert b"xxx" == cache[foo_key]
        assert 2 == store.counter["__getitem__", foo_key]
        assert 2 == cache.hits
        assert 3 == cache.misses

        # test 'bar' __getitem__, should have been evicted, cache miss
        assert b"yyy" == cache[bar_key]
        assert 2 == store.counter["__getitem__", bar_key]
        assert 2 == cache.hits
        assert 4 == cache.misses

        # setup store
        store = self.CountingClass()
        store[foo_key] = b"xxx"
        store[bar_key] = b"yyy"
        assert 0 == store.counter["__getitem__", foo_key]
        assert 0 == store.counter["__getitem__", bar_key]
        # setup cache - can hold two items
        cache = self.LRUStoreClass(store, max_size=6)
        assert 0 == cache.hits
        assert 0 == cache.misses

        # test first 'foo' __getitem__, cache miss
        assert b"xxx" == cache[foo_key]
        assert 1 == store.counter["__getitem__", foo_key]
        assert 0 == cache.hits
        assert 1 == cache.misses

        # test second 'foo' __getitem__, cache hit
        assert b"xxx" == cache[foo_key]
        assert 1 == store.counter["__getitem__", foo_key]
        assert 1 == cache.hits
        assert 1 == cache.misses

        # test first 'bar' __getitem__, cache miss
        assert b"yyy" == cache[bar_key]
        assert 1 == store.counter["__getitem__", bar_key]
        assert 1 == cache.hits
        assert 2 == cache.misses

        # test second 'bar' __getitem__, cache hit
        assert b"yyy" == cache[bar_key]
        assert 1 == store.counter["__getitem__", bar_key]
        assert 2 == cache.hits
        assert 2 == cache.misses

        # test 'foo' __getitem__, should still be cached
        assert b"xxx" == cache[foo_key]
        assert 1 == store.counter["__getitem__", foo_key]
        assert 3 == cache.hits
        assert 2 == cache.misses

        # test 'bar' __getitem__, should still be cached
        assert b"yyy" == cache[bar_key]
        assert 1 == store.counter["__getitem__", bar_key]
        assert 4 == cache.hits
        assert 2 == cache.misses

    def test_cache_keys(self):
        # setup
        store = self.CountingClass()
        foo_key = self.root + "foo"
        bar_key = self.root + "bar"
        baz_key = self.root + "baz"
        store[foo_key] = b"xxx"
        store[bar_key] = b"yyy"
        assert 0 == store.counter["__contains__", foo_key]
        assert 0 == store.counter["__iter__"]
        assert 0 == store.counter["keys"]
        cache = self.LRUStoreClass(store, max_size=None)

        # keys should be cached on first call
        keys = sorted(cache.keys())
        assert keys == [bar_key, foo_key]
        assert 1 == store.counter["keys"]
        # keys should now be cached
        assert keys == sorted(cache.keys())
        assert 1 == store.counter["keys"]
        assert foo_key in cache
        assert 1 == store.counter["__contains__", foo_key]
        # the next check for `foo_key` is cached
        assert foo_key in cache
        assert 1 == store.counter["__contains__", foo_key]
        assert keys == sorted(cache)
        assert 0 == store.counter["__iter__"]
        assert 1 == store.counter["keys"]

        # cache should be cleared if store is modified - crude but simple for now
        cache[baz_key] = b"zzz"
        keys = sorted(cache.keys())
        assert keys == [bar_key, baz_key, foo_key]
        assert 2 == store.counter["keys"]
        # keys should now be cached
        assert keys == sorted(cache.keys())
        assert 2 == store.counter["keys"]

        # manually invalidate keys
        cache.invalidate_keys()
        keys = sorted(cache.keys())
        assert keys == [bar_key, baz_key, foo_key]
        assert 3 == store.counter["keys"]
        assert 1 == store.counter["__contains__", foo_key]
        assert 0 == store.counter["__iter__"]
        cache.invalidate_keys()
        keys = sorted(cache)
        assert keys == [bar_key, baz_key, foo_key]
        assert 4 == store.counter["keys"]
        assert 1 == store.counter["__contains__", foo_key]
        assert 0 == store.counter["__iter__"]
        cache.invalidate_keys()
        assert foo_key in cache
        assert 4 == store.counter["keys"]
        assert 2 == store.counter["__contains__", foo_key]
        assert 0 == store.counter["__iter__"]

        # check these would get counted if called directly
        assert foo_key in store
        assert 3 == store.counter["__contains__", foo_key]
        assert keys == sorted(store)
        assert 1 == store.counter["__iter__"]


