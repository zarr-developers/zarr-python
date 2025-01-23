from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import zarr
from zarr.core.buffer import Buffer, cpu, gpu
from zarr.storage import GpuMemoryStore, MemoryStore
from zarr.testing.store import StoreTests
from zarr.testing.utils import gpu_test

if TYPE_CHECKING:
    from zarr.core.common import ZarrFormat


class TestMemoryStore(StoreTests[MemoryStore, cpu.Buffer]):
    store_cls = MemoryStore
    buffer_cls = cpu.Buffer

    async def set(self, store: MemoryStore, key: str, value: Buffer) -> None:
        store._store_dict[key] = value

    async def get(self, store: MemoryStore, key: str) -> Buffer:
        return store._store_dict[key]

    @pytest.fixture(params=[None, True])
    def store_kwargs(
        self, request: pytest.FixtureRequest
    ) -> dict[str, str | dict[str, Buffer] | None]:
        kwargs = {"store_dict": None}
        if request.param is True:
            kwargs["store_dict"] = {}
        return kwargs

    @pytest.fixture
    def store(self, store_kwargs: str | dict[str, Buffer] | None) -> MemoryStore:
        return self.store_cls(**store_kwargs)

    def test_store_repr(self, store: MemoryStore) -> None:
        assert str(store) == f"memory://{id(store._store_dict)}"

    def test_store_supports_writes(self, store: MemoryStore) -> None:
        assert store.supports_writes

    def test_store_supports_listing(self, store: MemoryStore) -> None:
        assert store.supports_listing

    def test_store_supports_partial_writes(self, store: MemoryStore) -> None:
        assert store.supports_partial_writes

    def test_list_prefix(self, store: MemoryStore) -> None:
        assert True

    @pytest.mark.parametrize("dtype", ["uint8", "float32", "int64"])
    @pytest.mark.parametrize("zarr_format", [2, 3])
    async def test_deterministic_size(
        self, store: MemoryStore, dtype, zarr_format: ZarrFormat
    ) -> None:
        def padding_size() -> int:
            a = zarr.empty(
                store=store,
                shape=(3,),
                chunks=(1000,),
                dtype=dtype,
                zarr_format=zarr_format,
                overwrite=True,
            )
            a[...] = 1
            key = "0" if zarr_format == 2 else "c/0"
            return len(store._store_dict[key])

        l1 = padding_size()
        l2 = padding_size()
        assert l1 == l2


@gpu_test
class TestGpuMemoryStore(StoreTests[GpuMemoryStore, gpu.Buffer]):
    store_cls = GpuMemoryStore
    buffer_cls = gpu.Buffer

    async def set(self, store: GpuMemoryStore, key: str, value: Buffer) -> None:
        store._store_dict[key] = value

    async def get(self, store: MemoryStore, key: str) -> Buffer:
        return store._store_dict[key]

    @pytest.fixture(params=[None, True])
    def store_kwargs(
        self, request: pytest.FixtureRequest
    ) -> dict[str, str | dict[str, Buffer] | None]:
        kwargs = {"store_dict": None}
        if request.param is True:
            kwargs["store_dict"] = {}
        return kwargs

    @pytest.fixture
    def store(self, store_kwargs: str | dict[str, gpu.Buffer] | None) -> GpuMemoryStore:
        return self.store_cls(**store_kwargs)

    def test_store_repr(self, store: GpuMemoryStore) -> None:
        assert str(store) == f"gpumemory://{id(store._store_dict)}"

    def test_store_supports_writes(self, store: GpuMemoryStore) -> None:
        assert store.supports_writes

    def test_store_supports_listing(self, store: GpuMemoryStore) -> None:
        assert store.supports_listing

    def test_store_supports_partial_writes(self, store: GpuMemoryStore) -> None:
        assert store.supports_partial_writes

    def test_list_prefix(self, store: GpuMemoryStore) -> None:
        assert True

    def test_dict_reference(self, store: GpuMemoryStore) -> None:
        store_dict = {}
        result = GpuMemoryStore(store_dict=store_dict)
        assert result._store_dict is store_dict

    def test_from_dict(self):
        d = {
            "a": gpu.Buffer.from_bytes(b"aaaa"),
            "b": cpu.Buffer.from_bytes(b"bbbb"),
        }
        result = GpuMemoryStore.from_dict(d)
        for v in result._store_dict.values():
            assert type(v) is gpu.Buffer
