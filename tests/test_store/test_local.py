from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import zarr
from zarr.core.buffer import Buffer, cpu
from zarr.storage.local import LocalStore
from zarr.testing.store import StoreTests

if TYPE_CHECKING:
    import pathlib


class TestLocalStore(StoreTests[LocalStore, cpu.Buffer]):
    store_cls = LocalStore
    buffer_cls = cpu.Buffer

    async def get(self, store: LocalStore, key: str) -> Buffer:
        return self.buffer_cls.from_bytes((Path(store.path) / key).read_bytes())

    async def set(self, store: LocalStore, key: str, value: Buffer) -> None:
        target = Path(store.path) / key
        parent = target.parent
        if not parent.exists():
            parent.mkdir(parents=True)
        target.write_bytes(value.to_bytes())

    @pytest.fixture
    def store_kwargs(self, tmpdir) -> dict[str, str]:
        return {"path": str(tmpdir), "mode": "r+"}

    def test_store_repr(self, store: LocalStore) -> None:
        assert str(store) == f"file:///{store.path}"

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

    def test_creates_new_directory(self, tmp_path: pathlib.Path):
        target = tmp_path.joinpath("a", "b", "c")
        assert not target.exists()

        store = self.store_cls(path=target, mode="w")
        zarr.group(store=store)
