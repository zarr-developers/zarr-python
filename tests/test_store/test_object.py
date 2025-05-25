# ruff: noqa: E402
from typing import Any

import pytest

obstore = pytest.importorskip("obstore")
from pathlib import Path

import pytest
from hypothesis.stateful import (
    run_state_machine_as_test,
)
from obstore.store import LocalStore, MemoryStore

from zarr.core.buffer import Buffer, cpu
from zarr.storage import ObjectStore
from zarr.testing.stateful import ZarrHierarchyStateMachine
from zarr.testing.store import StoreTests


class TestObjectStore(StoreTests[ObjectStore[LocalStore], cpu.Buffer]):
    store_cls = ObjectStore
    buffer_cls = cpu.Buffer

    @pytest.fixture
    def store_kwargs(self, tmp_path: Path) -> dict[str, Any]:
        store = LocalStore(prefix=str(tmp_path))
        return {"store": store, "read_only": False}

    @pytest.fixture
    async def store(self, store_kwargs: dict[str, Any]) -> ObjectStore[LocalStore]:
        return self.store_cls(**store_kwargs)

    async def get(self, store: ObjectStore[LocalStore], key: str) -> Buffer:
        new_local_store = LocalStore(prefix=store.store.prefix)
        return self.buffer_cls.from_bytes(obstore.get(new_local_store, key).bytes())

    async def set(self, store: ObjectStore[LocalStore], key: str, value: Buffer) -> None:
        new_local_store = LocalStore(prefix=store.store.prefix)
        obstore.put(new_local_store, key, value.to_bytes())

    def test_store_repr(self, store: ObjectStore[LocalStore]) -> None:
        from fnmatch import fnmatch

        pattern = "ObjectStore(object_store://LocalStore(*))"
        assert fnmatch(f"{store!r}", pattern)

    def test_store_supports_writes(self, store: ObjectStore[LocalStore]) -> None:
        assert store.supports_writes

    def test_store_supports_partial_writes(self, store: ObjectStore[LocalStore]) -> None:
        assert not store.supports_partial_writes

    def test_store_supports_listing(self, store: ObjectStore[LocalStore]) -> None:
        assert store.supports_listing

    def test_store_equal(self, store: ObjectStore[LocalStore]) -> None:
        """Test store equality"""
        # Test equality against a different instance type
        assert store != 0
        # Test equality against a different store type
        new_memory_store = ObjectStore(MemoryStore())
        assert store != new_memory_store
        # Test equality against a read only store
        new_local_store = ObjectStore(LocalStore(prefix=store.store.prefix), read_only=True)
        assert store != new_local_store
        # Test two memory stores cannot be equal
        second_memory_store = ObjectStore(MemoryStore())
        assert new_memory_store != second_memory_store

    def test_store_init_raises(self) -> None:
        """Test __init__ raises appropriate error for improper store type"""
        with pytest.raises(TypeError):
            ObjectStore("path/to/store")  # type: ignore[type-var]


@pytest.mark.slow_hypothesis
def test_zarr_hierarchy() -> None:
    sync_store = ObjectStore(MemoryStore())

    def mk_test_instance_sync() -> ZarrHierarchyStateMachine:
        return ZarrHierarchyStateMachine(sync_store)

    run_state_machine_as_test(mk_test_instance_sync)  # type ignore[no-untyped-call]
