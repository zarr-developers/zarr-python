from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from typing import TYPE_CHECKING, Any

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

    async def test_file_handle_support(self, tmp_path: Path) -> None:
        """Test that ZipStore works with open file handles."""
        zip_path = tmp_path / "test_file_handle.zip"

        # First create a ZIP file with some data using path-based approach
        store = await ZipStore.open(path=zip_path, mode="w")
        await store.set("test_key", cpu.Buffer.from_bytes(b"test_data"))
        store.close()

        # Now test reading via open file handle
        with open(zip_path, "rb") as file_handle:
            store = ZipStore(path=file_handle, mode="r")
            await store._open()

            # Test representation shows file-like object
            assert "file-like-object" in str(store)
            assert "file-like-object" in repr(store)

            # Test that we can read the data
            buffer = await store.get("test_key", default_buffer_prototype())
            assert buffer is not None
            assert buffer.to_bytes() == b"test_data"

            # Test that the store reports no path
            assert store.path is None
            assert store._file_opener is file_handle

            store.close()

    async def test_file_opener_support(self, tmp_path: Path) -> None:
        """Test that ZipStore works with file opener objects (like fsspec)."""
        zip_path = tmp_path / "test_file_opener.zip"

        # Create a ZIP file with some zarr data
        store = await ZipStore.open(path=zip_path, mode="w")
        # Create a simple zarr group structure
        await store.set(
            "zarr.json", cpu.Buffer.from_bytes(b'{"zarr_format":3,"node_type":"group"}')
        )
        await store.set(
            "data/zarr.json",
            cpu.Buffer.from_bytes(
                b'{"zarr_format":3,"node_type":"array","shape":[2],"data_type":"int32","chunk_grid":{"name":"regular","chunk_shape":[2]},"chunk_key_encoding":{"name":"default"},"codecs":[{"name":"bytes"}]}'
            ),
        )
        await store.set("data/c/0", cpu.Buffer.from_bytes(b"\x01\x00\x00\x00\x02\x00\x00\x00"))
        store.close()

        # Mock file opener (similar to what fsspec provides)
        class MockFileOpener:
            def __init__(self, path: Path) -> None:
                self.path = path

            def open(self) -> Any:
                return open(self.path, "rb")

        # Test with file opener
        opener = MockFileOpener(zip_path)
        store = ZipStore(path=opener, mode="r")
        await store._open()

        # Test representation shows file-like object
        assert "file-like-object" in str(store)
        assert "file-like-object" in repr(store)

        # Test that we can read zarr data
        group_buffer = await store.get("zarr.json", default_buffer_prototype())
        assert group_buffer is not None
        assert b"zarr_format" in group_buffer.to_bytes()

        # Test array data
        data_buffer = await store.get("data/c/0", default_buffer_prototype())
        assert data_buffer is not None
        assert len(data_buffer.to_bytes()) == 8  # 2 int32 values

        # Test that store attributes are correct
        assert store.path is None
        assert store._file_opener is opener
        assert hasattr(store, "_file_handle")

        store.close()

    def test_file_handle_backward_compatibility(self, tmp_path: Path) -> None:
        """Test that path-based ZipStore continues to work unchanged."""
        zip_path = tmp_path / "test_backward_compat.zip"

        # Test that traditional path-based usage still works exactly as before
        store = ZipStore(path=str(zip_path), mode="w")
        # These should be the traditional attributes
        assert store.path == zip_path
        assert store._file_opener is None

        # String representation should show the path
        assert str(zip_path) in str(store)
        assert str(zip_path) in repr(store)
