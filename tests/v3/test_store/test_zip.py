from __future__ import annotations

import os
import tempfile
from collections.abc import Coroutine
from typing import Any

import pytest

from zarr.abc.store import AccessMode
from zarr.core.buffer import Buffer, default_buffer_prototype
from zarr.store.zip import ZipStore
from zarr.testing.store import StoreTests


class TestZipStore(StoreTests[ZipStore]):
    store_cls = ZipStore

    @pytest.fixture(scope="function")
    def store_kwargs(self, request) -> dict[str, str | bool]:
        fd, temp_path = tempfile.mkstemp()
        os.close(fd)

        return {"path": temp_path, "mode": "w"}

    def get(self, store: ZipStore, key: str) -> Buffer:
        return store._get(key, prototype=default_buffer_prototype())

    def set(self, store: ZipStore, key: str, value: Buffer) -> None:
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
            await store.set("foo", Buffer.from_bytes(b"bar"))

        # # delete
        # TODO: uncomment once deletes are implemented
        # with pytest.raises(ValueError):
        #     await store.delete("foo")

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
