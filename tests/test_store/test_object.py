# ruff: noqa: E402
import pytest

pytest.importorskip("obstore")

from zarr.core.buffer import cpu
from zarr.storage.object_store import ObjectStore
from zarr.testing.store import StoreTests


class TestObjectStore(StoreTests[ObjectStore, cpu.Buffer]):
    store_cls = ObjectStore
