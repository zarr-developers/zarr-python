# ruff: noqa: E402
from typing import Any

import pytest

obstore = pytest.importorskip("obstore")
from obstore.store import LocalStore

from zarr.core.buffer import Buffer, cpu
from zarr.storage import ObjectStore
from zarr.testing.store import StoreTests


class TestObjectStore(StoreTests[ObjectStore, cpu.Buffer]):
    store_cls = ObjectStore
    buffer_cls = cpu.Buffer

    @pytest.fixture
    def store_kwargs(self, tmpdir) -> dict[str, Any]:
        store = LocalStore(prefix=tmpdir)
        return {"store": store, "read_only": False}

    @pytest.fixture
    def store(self, store_kwargs: dict[str, str | bool]) -> ObjectStore:
        return self.store_cls(**store_kwargs)

    async def get(self, store: ObjectStore, key: str) -> Buffer:
        assert isinstance(store.store, LocalStore)
        new_local_store = LocalStore(prefix=store.store.prefix)
        return self.buffer_cls.from_bytes(obstore.get(new_local_store, key).bytes())

    async def set(self, store: ObjectStore, key: str, value: Buffer) -> None:
        assert isinstance(store.store, LocalStore)
        new_local_store = LocalStore(prefix=store.store.prefix)
        obstore.put(new_local_store, key, value.to_bytes())

    def test_store_repr(self, store: ObjectStore) -> None:
        from fnmatch import fnmatch

        pattern = "ObjectStore(object://LocalStore(*))"
        assert fnmatch(f"{store!r}", pattern)

    def test_store_supports_writes(self, store: ObjectStore) -> None:
        assert store.supports_writes

    def test_store_supports_partial_writes(self, store: ObjectStore) -> None:
        assert not store.supports_partial_writes

    def test_store_supports_listing(self, store: ObjectStore) -> None:
        assert store.supports_listing
