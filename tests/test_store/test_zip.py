from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from typing import TYPE_CHECKING

import numpy as np
import pytest

import zarr
from zarr import create_array
from zarr.core.buffer import Buffer, cpu, default_buffer_prototype
from zarr.core.group import Group
from zarr.storage import ZipStore
from zarr.testing.store import StoreTests

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any


# TODO: work out where this is coming from and fix
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:coroutine method 'aclose' of 'ZipStore.list' was never awaited:RuntimeWarning"
    )
]


class TestZipStore(StoreTests[ZipStore, cpu.Buffer]):
    store_cls = ZipStore
    buffer_cls = cpu.Buffer

    @pytest.fixture
    def store_kwargs(self) -> dict[str, str | bool]:
        fd, temp_path = tempfile.mkstemp()
        os.close(fd)
        os.unlink(temp_path)

        return {"path": temp_path, "mode": "w", "read_only": False}

    async def get(self, store: ZipStore, key: str) -> Buffer:
        buf = store._get(key, prototype=default_buffer_prototype())
        assert buf is not None
        return buf

    async def set(self, store: ZipStore, key: str, value: Buffer) -> None:
        return store._set(key, value)

    def test_store_read_only(self, store: ZipStore) -> None:
        assert not store.read_only

    async def test_read_only_store_raises(self, store_kwargs: dict[str, Any]) -> None:
        # we need to create the zipfile in write mode before switching to read mode
        store = await self.store_cls.open(**store_kwargs)
        store.close()

        kwargs = {**store_kwargs, "mode": "a", "read_only": True}
        store = await self.store_cls.open(**kwargs)
        assert store._zmode == "a"
        assert store.read_only

        # set
        with pytest.raises(ValueError):
            await store.set("foo", cpu.Buffer.from_bytes(b"bar"))

    def test_store_repr(self, store: ZipStore) -> None:
        assert str(store) == f"zip://{store.path}"

    def test_store_supports_writes(self, store: ZipStore) -> None:
        assert store.supports_writes

    def test_store_supports_partial_writes(self, store: ZipStore) -> None:
        assert store.supports_partial_writes is False

    def test_store_supports_listing(self, store: ZipStore) -> None:
        assert store.supports_listing

    # TODO: fix this warning
    @pytest.mark.filterwarnings("ignore:Unclosed client session:ResourceWarning")
    def test_api_integration(self, store: ZipStore) -> None:
        root = zarr.open_group(store=store, mode="a")

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

    @pytest.mark.parametrize("read_only", [True, False])
    async def test_store_open_read_only(
        self, store_kwargs: dict[str, Any], read_only: bool
    ) -> None:
        if read_only:
            # create an empty zipfile
            with zipfile.ZipFile(store_kwargs["path"], mode="w"):
                pass

        await super().test_store_open_read_only(store_kwargs, read_only)

    @pytest.mark.parametrize(("zip_mode", "read_only"), [("w", False), ("a", False), ("x", False)])
    async def test_zip_open_mode_translation(
        self, store_kwargs: dict[str, Any], zip_mode: str, read_only: bool
    ) -> None:
        kws = {**store_kwargs, "mode": zip_mode}
        store = await self.store_cls.open(**kws)
        assert store.read_only == read_only

    def test_externally_zipped_store(self, tmp_path: Path) -> None:
        # See: https://github.com/zarr-developers/zarr-python/issues/2757
        zarr_path = tmp_path / "foo.zarr"
        root = zarr.open_group(store=zarr_path, mode="w")
        root.require_group("foo")
        assert isinstance(foo := root["foo"], Group)  # noqa: RUF018
        foo["bar"] = np.array([1])
        shutil.make_archive(str(zarr_path), "zip", zarr_path)
        zip_path = tmp_path / "foo.zarr.zip"
        zipped = zarr.open_group(ZipStore(zip_path, mode="r"), mode="r")
        assert list(zipped.keys()) == list(root.keys())
        assert isinstance(group := zipped["foo"], Group)
        assert list(group.keys()) == list(group.keys())

    async def test_move(self, tmp_path: Path) -> None:
        origin = tmp_path / "origin.zip"
        destination = tmp_path / "some_folder" / "destination.zip"

        store = await ZipStore.open(path=origin, mode="a")
        array = create_array(store, data=np.arange(10))

        await store.move(str(destination))

        assert store.path == destination
        assert destination.exists()
        assert not origin.exists()
        assert np.array_equal(array[...], np.arange(10))
