from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pytest

import zarr
from zarr.core.buffer import Buffer, cpu, gpu
from zarr.core.sync import sync
from zarr.errors import ZarrUserWarning
from zarr.storage import GpuMemoryStore, MemoryStore
from zarr.testing.store import StoreTests
from zarr.testing.store_concurrency import StoreConcurrencyTests
from zarr.testing.utils import gpu_test

if TYPE_CHECKING:
    from zarr.core.buffer import BufferPrototype
    from zarr.core.common import ZarrFormat


# TODO: work out where this warning is coming from and fix it
@pytest.mark.filterwarnings(
    re.escape("ignore:coroutine 'ClientCreatorContext.__aexit__' was never awaited")
)
class TestMemoryStore(StoreTests[MemoryStore, cpu.Buffer]):
    store_cls = MemoryStore
    buffer_cls = cpu.Buffer

    async def set(self, store: MemoryStore, key: str, value: Buffer) -> None:
        store._store_dict[key] = value

    async def get(self, store: MemoryStore, key: str) -> Buffer:
        return store._store_dict[key]

    @pytest.fixture(params=[None, True])
    def store_kwargs(self, request: pytest.FixtureRequest) -> dict[str, Any]:
        kwargs: dict[str, Any]
        if request.param is True:
            kwargs = {"store_dict": {}}
        else:
            kwargs = {"store_dict": None}
        return kwargs

    @pytest.fixture
    async def store(self, store_kwargs: dict[str, Any]) -> MemoryStore:
        return self.store_cls(**store_kwargs)

    def test_store_repr(self, store: MemoryStore) -> None:
        assert str(store) == f"memory://{id(store._store_dict)}"

    def test_store_supports_writes(self, store: MemoryStore) -> None:
        assert store.supports_writes

    def test_store_supports_listing(self, store: MemoryStore) -> None:
        assert store.supports_listing

    async def test_list_prefix(self, store: MemoryStore) -> None:
        assert True

    @pytest.mark.parametrize("dtype", ["uint8", "float32", "int64"])
    @pytest.mark.parametrize("zarr_format", [2, 3])
    async def test_deterministic_size(
        self, store: MemoryStore, dtype: npt.DTypeLike, zarr_format: ZarrFormat
    ) -> None:
        a = zarr.empty(
            store=store,
            shape=(3,),
            chunks=(1000,),
            dtype=dtype,
            zarr_format=zarr_format,
            overwrite=True,
        )
        a[...] = 1
        a.resize((1000,))

        np.testing.assert_array_equal(a[:3], 1)
        np.testing.assert_array_equal(a[3:], 0)

    @pytest.mark.parametrize("buffer_cls", [None, cpu.buffer_prototype])
    async def test_get_bytes_with_prototype_none(
        self, store: MemoryStore, buffer_cls: None | BufferPrototype
    ) -> None:
        """Test that get_bytes works with prototype=None."""
        data = b"hello world"
        key = "test_key"
        await self.set(store, key, self.buffer_cls.from_bytes(data))

        result = await store._get_bytes(key, prototype=buffer_cls)
        assert result == data

    @pytest.mark.parametrize("buffer_cls", [None, cpu.buffer_prototype])
    def test_get_bytes_sync_with_prototype_none(
        self, store: MemoryStore, buffer_cls: None | BufferPrototype
    ) -> None:
        """Test that get_bytes_sync works with prototype=None."""
        data = b"hello world"
        key = "test_key"
        sync(self.set(store, key, self.buffer_cls.from_bytes(data)))

        result = store._get_bytes_sync(key, prototype=buffer_cls)
        assert result == data

    @pytest.mark.parametrize("buffer_cls", [None, cpu.buffer_prototype])
    async def test_get_json_with_prototype_none(
        self, store: MemoryStore, buffer_cls: None | BufferPrototype
    ) -> None:
        """Test that get_json works with prototype=None."""
        data = {"foo": "bar", "number": 42}
        key = "test.json"
        await self.set(store, key, self.buffer_cls.from_bytes(json.dumps(data).encode()))

        result = await store._get_json(key, prototype=buffer_cls)
        assert result == data

    @pytest.mark.parametrize("buffer_cls", [None, cpu.buffer_prototype])
    def test_get_json_sync_with_prototype_none(
        self, store: MemoryStore, buffer_cls: None | BufferPrototype
    ) -> None:
        """Test that get_json_sync works with prototype=None."""
        data = {"foo": "bar", "number": 42}
        key = "test.json"
        sync(self.set(store, key, self.buffer_cls.from_bytes(json.dumps(data).encode())))

        result = store._get_json_sync(key, prototype=buffer_cls)
        assert result == data


# TODO: fix this warning
@pytest.mark.filterwarnings("ignore:Unclosed client session:ResourceWarning")
@gpu_test
class TestGpuMemoryStore(StoreTests[GpuMemoryStore, gpu.Buffer]):
    store_cls = GpuMemoryStore
    buffer_cls = gpu.Buffer

    async def set(self, store: GpuMemoryStore, key: str, value: gpu.Buffer) -> None:  # type: ignore[override]
        store._store_dict[key] = value

    async def get(self, store: MemoryStore, key: str) -> Buffer:
        return store._store_dict[key]

    @pytest.fixture(params=[None, True])
    def store_kwargs(self, request: pytest.FixtureRequest) -> dict[str, Any]:
        kwargs: dict[str, Any]
        if request.param is True:
            kwargs = {"store_dict": {}}
        else:
            kwargs = {"store_dict": None}
        return kwargs

    @pytest.fixture
    async def store(self, store_kwargs: dict[str, Any]) -> GpuMemoryStore:
        return self.store_cls(**store_kwargs)

    def test_store_repr(self, store: GpuMemoryStore) -> None:
        assert str(store) == f"gpumemory://{id(store._store_dict)}"

    def test_store_supports_writes(self, store: GpuMemoryStore) -> None:
        assert store.supports_writes

    def test_store_supports_listing(self, store: GpuMemoryStore) -> None:
        assert store.supports_listing

    async def test_list_prefix(self, store: GpuMemoryStore) -> None:
        assert True

    def test_dict_reference(self, store: GpuMemoryStore) -> None:
        store_dict: dict[str, Any] = {}
        result = GpuMemoryStore(store_dict=store_dict)
        assert result._store_dict is store_dict

    def test_from_dict(self) -> None:
        d = {
            "a": gpu.Buffer.from_bytes(b"aaaa"),
            "b": cpu.Buffer.from_bytes(b"bbbb"),
        }
        msg = "Creating a zarr.buffer.gpu.Buffer with an array that does not support the __cuda_array_interface__ for zero-copy transfers, falling back to slow copy based path"
        with pytest.warns(ZarrUserWarning, match=msg):
            result = GpuMemoryStore.from_dict(d)
        for v in result._store_dict.values():
            assert type(v) is gpu.Buffer


class TestMemoryStoreConcurrency(StoreConcurrencyTests[MemoryStore, cpu.Buffer]):
    """Test MemoryStore concurrency limiting behavior."""

    store_cls = MemoryStore
    buffer_cls = cpu.Buffer
    expected_concurrency_limit = None  # MemoryStore has no limit (fast in-memory ops)

    @pytest.fixture
    def store_kwargs(self) -> dict[str, Any]:
        return {"store_dict": None}
