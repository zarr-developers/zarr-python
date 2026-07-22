from __future__ import annotations

import errno
import os
import pathlib
import re
from tempfile import gettempdir

import numpy as np
import pytest

import zarr
from zarr import create_array
from zarr.core.buffer import Buffer, cpu
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
    def store_kwargs(self, tmp_path: pathlib.Path) -> dict[str, str]:
        return {"root": str(tmp_path)}

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

    # --- byte-range-write tests: disabled ---
    # Byte-range-write support (set_range / set_range_sync / SupportsSetRange)
    # was removed from this PR pending a decision on the store interface. These
    # tests are known-good and kept commented out to restore once that lands.
    # def test_supports_set_range(self, store: LocalStore) -> None:
    #     """LocalStore should implement SupportsSetRange."""
    #     assert isinstance(store, SupportsSetRange)
    #
    # @pytest.mark.parametrize(
    #     ("start", "patch", "expected"),
    #     [
    #         (0, b"XX", b"XXAAAAAAAA"),
    #         (3, b"XX", b"AAAXXAAAAA"),
    #         (8, b"XX", b"AAAAAAAAXX"),
    #         (0, b"ZZZZZZZZZZ", b"ZZZZZZZZZZ"),
    #         (5, b"B", b"AAAAABAAAA"),
    #         (0, b"BCDE", b"BCDEAAAAAA"),
    #     ],
    #     ids=["start", "middle", "end", "full-overwrite", "single-byte", "multi-byte-start"],
    # )
    # async def test_set_range(
    #     self, store: LocalStore, start: int, patch: bytes, expected: bytes
    # ) -> None:
    #     """set_range should overwrite bytes at the given offset."""
    #     await store.set("test/key", cpu.Buffer.from_bytes(b"AAAAAAAAAA"))
    #     await store.set_range("test/key", cpu.Buffer.from_bytes(patch), start=start)
    #     result = await store.get("test/key", prototype=cpu.buffer_prototype)
    #     assert result is not None
    #     assert result.to_bytes() == expected
    #
    # @pytest.mark.parametrize(
    #     ("start", "patch", "expected"),
    #     [
    #         (0, b"XX", b"XXAAAAAAAA"),
    #         (3, b"XX", b"AAAXXAAAAA"),
    #         (8, b"XX", b"AAAAAAAAXX"),
    #         (0, b"ZZZZZZZZZZ", b"ZZZZZZZZZZ"),
    #         (5, b"B", b"AAAAABAAAA"),
    #         (0, b"BCDE", b"BCDEAAAAAA"),
    #     ],
    #     ids=["start", "middle", "end", "full-overwrite", "single-byte", "multi-byte-start"],
    # )
    # def test_set_range_sync(
    #     self, store: LocalStore, start: int, patch: bytes, expected: bytes
    # ) -> None:
    #     """set_range_sync should overwrite bytes at the given offset."""
    #     sync(store.set("test/key", cpu.Buffer.from_bytes(b"AAAAAAAAAA")))
    #     store.set_range_sync("test/key", cpu.Buffer.from_bytes(patch), start=start)
    #     result = store.get_sync(key="test/key", prototype=cpu.buffer_prototype)
    #     assert result is not None
    #     assert result.to_bytes() == expected


@pytest.mark.parametrize("exclusive", [True, False])
def test_atomic_write_successful(tmp_path: pathlib.Path, exclusive: bool) -> None:
    path = tmp_path / "data"
    with _atomic_write(path, "wb", tmp_path, exclusive=exclusive) as f:
        f.write(b"abc")
    assert path.read_bytes() == b"abc"
    assert list(path.parent.iterdir()) == [path]  # no temp files


@pytest.mark.parametrize("exclusive", [True, False])
def test_atomic_write_incomplete(tmp_path: pathlib.Path, exclusive: bool) -> None:
    path = tmp_path / "data"
    with pytest.raises(RuntimeError):  # noqa: PT012
        with _atomic_write(path, "wb", tmp_path, exclusive=exclusive) as f:
            f.write(b"a")
            raise RuntimeError
    assert not path.exists()
    assert list(path.parent.iterdir()) == []  # no temp files


def test_atomic_write_non_exclusive_preexisting(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "data"
    with path.open("wb") as f:
        f.write(b"xyz")
    assert path.read_bytes() == b"xyz"
    with _atomic_write(path, "wb", tmp_path, exclusive=False) as f:
        f.write(b"abc")
    assert path.read_bytes() == b"abc"
    assert list(path.parent.iterdir()) == [path]  # no temp files


def test_atomic_write_exclusive_preexisting(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "data"
    with path.open("wb") as f:
        f.write(b"xyz")
    assert path.read_bytes() == b"xyz"
    with pytest.raises(FileExistsError):
        with _atomic_write(path, "wb", tmp_path, exclusive=True) as f:
            f.write(b"abc")
    assert path.read_bytes() == b"xyz"
    assert list(path.parent.iterdir()) == [path]  # no temp files


def test_tmp_dir_arg(tmp_path: pathlib.Path) -> None:
    store = LocalStore(tmp_path, tmp_dir=tmp_path / "scratch")
    assert store._resolve_tmp_dir() == tmp_path / "scratch"


def test_tmp_dir_from_config(tmp_path: pathlib.Path) -> None:
    with zarr.config.set({"store.local.tmp_dir": str(tmp_path / "cfg")}):
        store = LocalStore(tmp_path)
        assert store._resolve_tmp_dir() == tmp_path / "cfg"


def test_tmp_dir_default(tmp_path: pathlib.Path) -> None:
    store = LocalStore(tmp_path)
    assert store._resolve_tmp_dir() == pathlib.Path(gettempdir())


@pytest.mark.parametrize("exclusive", [True, False])
def test_atomic_write_cross_device_raises(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, exclusive: bool
) -> None:
    def _raise_exdev(*args: object, **kwargs: object) -> None:
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    if exclusive:
        monkeypatch.setattr("zarr.storage._local._safe_move", _raise_exdev)
    else:
        monkeypatch.setattr(os, "replace", _raise_exdev)

    path = tmp_path / "data"
    with pytest.raises(OSError, match="same filesystem") as excinfo:
        with _atomic_write(path, "wb", tmp_path, exclusive=exclusive) as f:
            f.write(b"abc")

    assert excinfo.value.errno == errno.EXDEV  # errno preserved
    assert not path.exists()  # target never got created
    assert list(tmp_path.iterdir()) == []  # tmp cleans up
