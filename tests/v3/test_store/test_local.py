from __future__ import annotations

import pytest

from zarr.buffer import Buffer, cpu
from zarr.store.local import LocalStore
from zarr.testing.store import StoreTests


class TestLocalStore(StoreTests[LocalStore, cpu.Buffer]):
    store_cls = LocalStore
    buffer_cls = cpu.Buffer

    def get(self, store: LocalStore, key: str) -> Buffer:
        return self.buffer_cls.from_bytes((store.root / key).read_bytes())

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
