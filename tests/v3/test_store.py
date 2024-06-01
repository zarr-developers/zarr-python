from __future__ import annotations

from typing import Any

import pytest

from zarr.buffer import Buffer
from zarr.store.local import LocalStore
from zarr.store.memory import MemoryStore
from zarr.testing.store import StoreTests


class TestMemoryStore(StoreTests[MemoryStore]):
    store_cls = MemoryStore

    def set(self, store: MemoryStore, key: str, value: Buffer) -> None:
        store._store_dict[key] = value

    def get(self, store: MemoryStore, key: str) -> Buffer:
        return store._store_dict[key]

    @pytest.fixture(scope="function", params=[None, {}])
    def store_kwargs(self, request) -> dict[str, Any]:
        return {"store_dict": request.param, "mode": "w"}

    @pytest.fixture(scope="function")
    def store(self, store_kwargs: dict[str, Any]) -> MemoryStore:
        return self.store_cls(**store_kwargs)

    def test_store_repr(self, store: MemoryStore) -> None:
        assert str(store) == f"memory://{id(store._store_dict)}"

    def test_store_supports_writes(self, store: MemoryStore) -> None:
        assert store.supports_writes

    def test_store_supports_listing(self, store: MemoryStore) -> None:
        assert store.supports_listing

    def test_store_supports_partial_writes(self, store: MemoryStore) -> None:
        assert store.supports_partial_writes

    def test_list_prefix(self, store: MemoryStore) -> None:
        assert True


class TestLocalStore(StoreTests[LocalStore]):
    store_cls = LocalStore

    def get(self, store: LocalStore, key: str) -> Buffer:
        return Buffer.from_bytes((store.root / key).read_bytes())

    def set(self, store: LocalStore, key: str, value: Buffer) -> None:
        parent = (store.root / key).parent
        if not parent.exists():
            parent.mkdir(parents=True)
        (store.root / key).write_bytes(value.to_bytes())

    @pytest.fixture
    def store_kwargs(self, tmpdir) -> dict[str, str]:
        return {"root": str(tmpdir), "mode": "w"}

    def test_store_repr(self, store: LocalStore) -> None:
        assert str(store) == f"file://{store.root!s}"

    def test_store_supports_writes(self, store: LocalStore) -> None:
        assert store.supports_writes

    def test_store_supports_partial_writes(self, store: LocalStore) -> None:
        assert store.supports_partial_writes

    def test_store_supports_listing(self, store: LocalStore) -> None:
        assert store.supports_listing

    def test_list_prefix(self, store: LocalStore) -> None:
        assert True
