from __future__ import annotations

import json
import pathlib
import re

import numpy as np
import pytest

import zarr
from zarr import create_array
from zarr.core.buffer import Buffer, cpu
from zarr.core.buffer.core import BufferPrototype, default_buffer_prototype
from zarr.core.sync import sync
from zarr.storage import LocalStore
from zarr.storage._local import _atomic_write
from zarr.testing.store import StoreTests
from zarr.testing.utils import assert_bytes_equal


class TestLocalStore(StoreTests[LocalStore, cpu.Buffer]):
    store_cls = LocalStore
    buffer_cls = cpu.Buffer

    async def get(self, store: LocalStore, key: str) -> Buffer:
        return self.buffer_cls.from_bytes((store.root / key).read_bytes())

    async def set(self, store: LocalStore, key: str, value: Buffer) -> None:
        parent = (store.root / key).parent
        if not parent.exists():
            parent.mkdir(parents=True)
        (store.root / key).write_bytes(value.to_bytes())

    @pytest.fixture
    def store_kwargs(self, tmpdir: str) -> dict[str, str]:
        return {"root": str(tmpdir)}

    def test_store_repr(self, store: LocalStore) -> None:
        assert str(store) == f"file://{store.root.as_posix()}"

    def test_store_supports_writes(self, store: LocalStore) -> None:
        assert store.supports_writes

    def test_store_supports_listing(self, store: LocalStore) -> None:
        assert store.supports_listing

    async def test_empty_with_empty_subdir(self, store: LocalStore) -> None:
        assert await store.is_empty("")
        (store.root / "foo/bar").mkdir(parents=True)
        assert await store.is_empty("")

    def test_creates_new_directory(self, tmp_path: pathlib.Path) -> None:
        target = tmp_path.joinpath("a", "b", "c")
        assert not target.exists()

        store = self.store_cls(root=target)
        zarr.group(store=store)

    def test_invalid_root_raises(self) -> None:
        """
        Test that a TypeError is raised when a non-str/Path type is used for the `root` argument
        """
        with pytest.raises(
            TypeError,
            match=r"'root' must be a string or Path instance. Got an instance of <class 'int'> instead.",
        ):
            LocalStore(root=0)  # type: ignore[arg-type]

    async def test_get_with_prototype_default(self, store: LocalStore) -> None:
        """
        Ensure that data can be read via ``store.get`` if the prototype keyword argument is unspecified, i.e. set to ``None``.
        """
        data_buf = self.buffer_cls.from_bytes(b"\x01\x02\x03\x04")
        key = "c/0"
        await self.set(store, key, data_buf)
        observed = await store.get(key, prototype=None)
        assert_bytes_equal(observed, data_buf)

    @pytest.mark.parametrize("ndim", [0, 1, 3])
    @pytest.mark.parametrize(
        "destination", ["destination", "foo/bar/destintion", pathlib.Path("foo/bar/destintion")]
    )
    async def test_move(
        self, tmp_path: pathlib.Path, ndim: int, destination: pathlib.Path | str
    ) -> None:
        origin = tmp_path / "origin"
        if isinstance(destination, str):
            destination = str(tmp_path / destination)
        else:
            destination = tmp_path / destination

        print(type(destination))
        store = await LocalStore.open(root=origin)
        shape = (4,) * ndim
        chunks = (2,) * ndim
        data = np.arange(4**ndim)
        if ndim > 0:
            data = data.reshape(*shape)
        array = create_array(store, data=data, chunks=chunks or "auto")

        await store.move(destination)

        assert store.root == pathlib.Path(destination)
        assert pathlib.Path(destination).exists()
        assert not origin.exists()
        assert np.array_equal(array[...], data)

        store2 = await LocalStore.open(root=origin)
        with pytest.raises(
            FileExistsError, match=re.escape(f"Destination root {destination} already exists")
        ):
            await store2.move(destination)


@pytest.mark.parametrize("exclusive", [True, False])
def test_atomic_write_successful(tmp_path: pathlib.Path, exclusive: bool) -> None:
    path = tmp_path / "data"
    with _atomic_write(path, "wb", exclusive=exclusive) as f:
        f.write(b"abc")
    assert path.read_bytes() == b"abc"
    assert list(path.parent.iterdir()) == [path]  # no temp files


@pytest.mark.parametrize("exclusive", [True, False])
def test_atomic_write_incomplete(tmp_path: pathlib.Path, exclusive: bool) -> None:
    path = tmp_path / "data"
    with pytest.raises(RuntimeError):  # noqa: PT012
        with _atomic_write(path, "wb", exclusive=exclusive) as f:
            f.write(b"a")
            raise RuntimeError
    assert not path.exists()
    assert list(path.parent.iterdir()) == []  # no temp files


def test_atomic_write_non_exclusive_preexisting(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "data"
    with path.open("wb") as f:
        f.write(b"xyz")
    assert path.read_bytes() == b"xyz"
    with _atomic_write(path, "wb", exclusive=False) as f:
        f.write(b"abc")
    assert path.read_bytes() == b"abc"
    assert list(path.parent.iterdir()) == [path]  # no temp files


def test_atomic_write_exclusive_preexisting(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "data"
    with path.open("wb") as f:
        f.write(b"xyz")
    assert path.read_bytes() == b"xyz"
    with pytest.raises(FileExistsError):
        with _atomic_write(path, "wb", exclusive=True) as f:
            f.write(b"abc")
    assert path.read_bytes() == b"xyz"
    assert list(path.parent.iterdir()) == [path]  # no temp files


async def test_get_bytes_with_prototype_none(tmp_path: pathlib.Path) -> None:
    """Test that get_bytes works with prototype=None."""
    from zarr.core.buffer import cpu

    store = await LocalStore.open(root=tmp_path)
    data = b"hello world"
    key = "test_key"
    await store.set(key, cpu.Buffer.from_bytes(data))

    # Test with None (default)
    result_none = await store.get_bytes(key)
    assert result_none == data

    # Test with explicit prototype
    result_proto = await store.get_bytes(key, prototype=default_buffer_prototype())
    assert result_proto == data


def test_get_bytes_sync_with_prototype_none(tmp_path: pathlib.Path) -> None:
    """Test that get_bytes_sync works with prototype=None."""
    from zarr.core.buffer import cpu
    from zarr.core.sync import sync

    store = sync(LocalStore.open(root=tmp_path))
    data = b"hello world"
    key = "test_key"
    sync(store.set(key, cpu.Buffer.from_bytes(data)))

    # Test with None (default)
    result_none = store.get_bytes_sync(key)
    assert result_none == data

    # Test with explicit prototype
    result_proto = store.get_bytes_sync(key, prototype=default_buffer_prototype())
    assert result_proto == data


@pytest.mark.parametrize("buffer_cls", [None, cpu.buffer_prototype])
async def test_get_json_with_prototype_none(
    tmp_path: pathlib.Path, buffer_cls: None | BufferPrototype
) -> None:
    """Test that get_json works with prototype=None."""

    store = await LocalStore.open(root=tmp_path)
    data = {"foo": "bar", "number": 42}
    key = "test.json"
    await store.set(key, cpu.Buffer.from_bytes(json.dumps(data).encode()))

    # Test with None (default)
    result = await store.get_json(key, prototype=buffer_cls)
    assert result == data


@pytest.mark.parametrize("buffer_cls", [None, cpu.buffer_prototype])
def test_get_json_sync_with_prototype(
    tmp_path: pathlib.Path, buffer_cls: None | BufferPrototype
) -> None:
    """Test that get_json_sync works with prototype=None."""

    store = sync(LocalStore.open(root=tmp_path))
    data = {"foo": "bar", "number": 42}
    key = "test.json"
    sync(store.set(key, cpu.Buffer.from_bytes(json.dumps(data).encode())))

    # Test with None (default)
    result = store.get_json_sync(key, prototype=buffer_cls)
    assert result == data
