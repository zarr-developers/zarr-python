import pytest

from zarr.testing.store import StoreTests
from zarr.store.local import LocalStore
from zarr.store.memory import MemoryStore


class TestMemoryStore(StoreTests):
    store_cls = MemoryStore


class TestLocalStore(StoreTests):
    store_cls = LocalStore

    @pytest.fixture(scope="function")
    @pytest.mark.parametrize("auto_mkdir", (True, False))
    def store(self, tmpdir) -> LocalStore:
        return self.store_cls(str(tmpdir))
