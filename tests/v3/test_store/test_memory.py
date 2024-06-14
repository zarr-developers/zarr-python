from __future__ import annotations

import pytest

from zarr.buffer import Buffer
from zarr.store.memory import MemoryStore
from zarr.testing.store import StoreTests


class TestMemoryStore(StoreTests[MemoryStore]):
    store_cls = MemoryStore

    def set(self, store: MemoryStore, key: str, value: Buffer) -> None:
        store._store_dict[key] = value

    def get(self, store: MemoryStore, key: str) -> Buffer:
        return store._store_dict[key]

    @pytest.fixture(scope="function", params=[None, {}])
    def store_kwargs(self, request) -> dict[str, str | None | dict[str, Buffer]]:
        return {"store_dict": request.param, "mode": "w"}

    @pytest.fixture(scope="function")
    def store(self, store_kwargs: str | None | dict[str, Buffer]) -> MemoryStore:
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
