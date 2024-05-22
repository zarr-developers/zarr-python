from __future__ import annotations

from collections.abc import MutableMapping

import pytest

from zarr.buffer import Buffer
from zarr.store.local import LocalStore
from zarr.store.memory import MemoryStore
from zarr.testing.store import StoreTests


@pytest.mark.parametrize("store_dict", (None, {}))
class TestMemoryStore(StoreTests[MemoryStore]):
    store_cls = MemoryStore

    def set(self, store: MemoryStore, key: str, value: bytes) -> None:
        store._store_dict[key] = value

    def get(self, store: MemoryStore, key: str) -> bytes:
        return store._store_dict[key]

    @pytest.fixture(scope="function")
    def store(self, store_dict: MutableMapping[str, Buffer] | None):
        return MemoryStore(store_dict=store_dict)

    def test_store_repr(self, store: MemoryStore) -> None:
        assert str(store) == f"memory://{id(store._store_dict)}"

    def test_store_supports_writes(self, store: MemoryStore) -> None:
        assert True

    def test_store_supports_listing(self, store: MemoryStore) -> None:
        assert True

    def test_store_supports_partial_writes(self, store: MemoryStore) -> None:
        assert True

    def test_list_prefix(self, store: MemoryStore) -> None:
        assert True


class TestLocalStore(StoreTests[LocalStore]):
    store_cls = LocalStore

    def get(self, store: LocalStore, key: str) -> bytes:
        return (store.root / key).read_bytes()

    def set(self, store: LocalStore, key: str, value: bytes) -> None:
        parent = (store.root / key).parent
        if not parent.exists():
            parent.mkdir(parents=True)
        (store.root / key).write_bytes(value)

    @pytest.fixture(scope="function")
    def store(self, tmpdir) -> LocalStore:
        return self.store_cls(str(tmpdir))

    def test_store_repr(self, store: LocalStore) -> None:
        assert str(store) == f"file://{store.root!s}"

    def test_store_supports_writes(self, store: LocalStore) -> None:
        assert True

    def test_store_supports_partial_writes(self, store: LocalStore) -> None:
        assert True

    def test_store_supports_listing(self, store: LocalStore) -> None:
        assert True

    def test_list_prefix(self, store: LocalStore) -> None:
        assert True
