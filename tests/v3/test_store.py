from __future__ import annotations

from collections.abc import MutableMapping

import pytest

from zarr.buffer import Buffer
from zarr.store.local import LocalStore
from zarr.store.memory import MemoryStore
from zarr.testing.store import StoreTests


class TestMemoryStore(StoreTests[MemoryStore]):
    @pytest.fixture(scope="function")
    @pytest.mark.parametrize("store_dict", (None, {}))
    def store(self, store_dict: MutableMapping[str, Buffer] | None):
        return MemoryStore(store_dict=store_dict)

    def test_store_repr(self, store: MemoryStore) -> None:
        assert str(store) == f"memory://{id(store._store_dict)}"

    def test_store_supports_writes(self, store: MemoryStore) -> None:
        return True

    def test_store_supports_listing(self, store: MemoryStore) -> None:
        return True

    def test_store_supports_partial_writes(self, store: MemoryStore) -> None:
        return True


class TestLocalStore(StoreTests[LocalStore]):
    @pytest.fixture(scope="function")
    @pytest.mark.parametrize("auto_mkdir", (True, False))
    def store(self, tmpdir) -> LocalStore:
        return self.store_cls(str(tmpdir))

    def test_store_repr(self, store: LocalStore) -> None:
        assert str(store) == f"file://{store.root!s}"

    def test_store_supports_writes(self, store: LocalStore) -> None:
        return True

    def test_store_supports_partial_writes(self, store: LocalStore) -> None:
        return True

    def test_store_supports_listing(self, store: LocalStore) -> None:
        return True
