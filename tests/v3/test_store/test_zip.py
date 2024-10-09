from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING

import numpy as np
import pytest

import zarr
from zarr.abc.store import AccessMode
from zarr.core.buffer import Buffer, cpu, default_buffer_prototype
from zarr.storage.zip import ZipStore
from zarr.testing.store import StoreTests

if TYPE_CHECKING:
    from collections.abc import Coroutine
    from typing import Any


class TestZipStore(StoreTests[ZipStore, cpu.Buffer]):
    store_cls = ZipStore
    buffer_cls = cpu.Buffer

    @pytest.fixture
    def store_kwargs(self, request) -> dict[str, str | bool]:
        fd, temp_path = tempfile.mkstemp()
        os.close(fd)

        return {"path": temp_path, "mode": "w"}

    async def get(self, store: ZipStore, key: str) -> Buffer:
        return store._get(key, prototype=default_buffer_prototype())

    async def set(self, store: ZipStore, key: str, value: Buffer) -> None:
        return store._set(key, value)

    def test_store_mode(self, store: ZipStore, store_kwargs: dict[str, Any]) -> None:
        assert store.mode == AccessMode.from_literal(store_kwargs["mode"])
        assert not store.mode.readonly

    async def test_not_writable_store_raises(self, store_kwargs: dict[str, Any]) -> None:
        # we need to create the zipfile in write mode before switching to read mode
        store = await self.store_cls.open(**store_kwargs)
        store.close()

        kwargs = {**store_kwargs, "mode": "r"}
        store = await self.store_cls.open(**kwargs)
        assert store.mode == AccessMode.from_literal("r")
        assert store.mode.readonly

        # set
        with pytest.raises(ValueError):
            await store.set("foo", cpu.Buffer.from_bytes(b"bar"))

    def test_store_repr(self, store: ZipStore) -> None:
        assert str(store) == f"zip://{store.path!s}"

    def test_store_supports_writes(self, store: ZipStore) -> None:
        assert store.supports_writes

    def test_store_supports_partial_writes(self, store: ZipStore) -> None:
        assert store.supports_partial_writes is False

    def test_store_supports_listing(self, store: ZipStore) -> None:
        assert store.supports_listing

    def test_delete(self, store: ZipStore) -> Coroutine[Any, Any, None]:
        pass

    def test_api_integration(self, store: ZipStore) -> None:
        root = zarr.open_group(store=store)

        data = np.arange(10000, dtype=np.uint16).reshape(100, 100)
        z = root.create_array(
            shape=data.shape, chunks=(10, 10), name="foo", dtype=np.uint16, fill_value=99
        )
        z[:] = data

        assert np.array_equal(data, z[:])

        # you can overwrite existing chunks but zipfile will issue a warning
        with pytest.warns(UserWarning, match="Duplicate name: 'foo/c/0/0'"):
            z[0, 0] = 100

        # TODO: assigning an entire chunk to fill value ends up deleting the chunk which is not supported
        # a work around will be needed here.
        with pytest.raises(NotImplementedError):
            z[0:10, 0:10] = 99

        bar = root.create_group("bar", attributes={"hello": "world"})
        assert "hello" in dict(bar.attrs)

        # keys cannot be deleted
        with pytest.raises(NotImplementedError):
            del root["bar"]

        store.close()

    async def test_with_mode(self, store: ZipStore) -> None:
        with pytest.raises(NotImplementedError, match="new mode"):
            await super().test_with_mode(store)

    @pytest.mark.parametrize("mode", ["a", "w"])
    async def test_store_open_mode(self, store_kwargs: dict[str, Any], mode: str) -> None:
        super().test_store_open_mode(store_kwargs, mode)
