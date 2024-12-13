# ruff: noqa: E402
import pytest

obstore = pytest.importorskip("obstore")

from zarr.core.buffer import cpu
from zarr.storage.object_store import ObjectStore
from zarr.testing.store import StoreTests


class TestObjectStore(StoreTests[ObjectStore, cpu.Buffer]):
    store_cls = ObjectStore
    buffer_cls = cpu.Buffer

    @pytest.fixture
    def store_kwargs(self, tmpdir) -> dict[str, str | bool]:
        store = obstore.store.LocalStore(prefix=tmpdir)
        return {"store": store, "read_only": False}

    @pytest.fixture
    def store(self, store_kwargs: dict[str, str | bool]) -> ObjectStore:
        return self.store_cls(**store_kwargs)

    def test_store_repr(self, store: ObjectStore) -> None:
        from fnmatch import fnmatch

        pattern = "ObjectStore(object://LocalStore(file:///*))"
        assert fnmatch(f"{store!r}", pattern)

    def test_store_supports_writes(self, store: ObjectStore) -> None:
        assert store.supports_writes

    def test_store_supports_partial_writes(self, store: ObjectStore) -> None:
        assert not store.supports_partial_writes

    def test_store_supports_listing(self, store: ObjectStore) -> None:
        assert store.supports_listing
