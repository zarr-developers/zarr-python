from __future__ import annotations

from pathlib import Path

import pytest

from zarr.core.buffer import Buffer, cpu
from zarr.store.local import LocalStore
from zarr.testing.store import StoreTests


class TestLocalStore(StoreTests[LocalStore, cpu.Buffer]):
    store_cls = LocalStore
    buffer_cls = cpu.Buffer

    def get(self, store: LocalStore, key: str) -> Buffer:
        return self.buffer_cls.from_bytes((Path(store.path) / key).read_bytes())

    def set(self, store: LocalStore, key: str, value: Buffer) -> None:
        target = Path(store.path) / key
        parent = target.parent
        if not parent.exists():
            parent.mkdir(parents=True)
        target.write_bytes(value.to_bytes())

    @pytest.fixture
    def store_kwargs(self, tmpdir) -> dict[str, str]:
        return {"path": str(tmpdir), "mode": "r+"}

    def test_store_repr(self, store: LocalStore) -> None:
        assert str(store) == f"file://{store.path!s}"

    def test_store_supports_writes(self, store: LocalStore) -> None:
        assert store.supports_writes

    def test_store_supports_partial_writes(self, store: LocalStore) -> None:
        assert store.supports_partial_writes

    def test_store_supports_listing(self, store: LocalStore) -> None:
        assert store.supports_listing

    async def test_empty_with_empty_subdir(self, store: LocalStore) -> None:
        assert await store.empty()
        (Path(store.path) / "foo/bar").mkdir(parents=True)
        assert await store.empty()
