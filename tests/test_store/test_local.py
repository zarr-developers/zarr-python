from __future__ import annotations

import pathlib
import re
import time

import numpy as np
import pytest

import zarr
from zarr import create_array
from zarr.core.buffer import Buffer, cpu
from zarr.storage import LocalStore
from zarr.storage._local import _RETRY_DELAYS, _atomic_write, _move_with_retry
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

    def test_delete_sync_directory(self, store: LocalStore) -> None:
        """`delete_sync` on a key that is a directory must remove the whole tree.

        Mirrors the async `delete_dir` behavior: deleting `"foo"` where
        `"foo"` is a directory containing further nested paths should remove
        everything under it, not just fail or delete a single file.
        """
        (store.root / "foo" / "bar").mkdir(parents=True)
        (store.root / "foo" / "bar" / "baz").write_bytes(b"data")

        store.delete_sync("foo")

        assert not (store.root / "foo").exists()

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


def _oserror(winerror: int) -> OSError:
    """An OSError shaped like the one a failed MoveFileEx produces."""
    error = OSError(13, "Access is denied")
    error.winerror = winerror  # type: ignore[attr-defined]
    return error


class _FlakyMove:
    """A move that raises `error` the first `failures` times it is called."""

    def __init__(self, failures: int, error: OSError) -> None:
        self.failures = failures
        self.error = error
        self.attempts = 0

    def __call__(self, src: pathlib.Path, dst: pathlib.Path) -> None:
        self.attempts += 1
        if self.attempts <= self.failures:
            raise self.error
        src.replace(dst)


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record the delays `_move_with_retry` would sleep for instead of sleeping."""
    recorded: list[float] = []
    monkeypatch.setattr(time, "sleep", recorded.append)
    return recorded


@pytest.mark.parametrize("winerror", [5, 32])
@pytest.mark.parametrize("failures", [1, 2, 3])
def test_move_with_retry_recovers(
    tmp_path: pathlib.Path, sleeps: list[float], winerror: int, failures: int
) -> None:
    """A destination that is briefly busy is retried, not reported."""
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.write_bytes(b"abc")
    move = _FlakyMove(failures, _oserror(winerror))

    _move_with_retry(src, dst, move)

    assert dst.read_bytes() == b"abc"
    assert move.attempts == failures + 1
    assert sleeps == list(_RETRY_DELAYS[:failures])


def test_move_with_retry_gives_up(tmp_path: pathlib.Path, sleeps: list[float]) -> None:
    """A destination that never frees still raises, after a bounded wait."""
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.write_bytes(b"abc")
    move = _FlakyMove(len(_RETRY_DELAYS) + 1, _oserror(5))

    with pytest.raises(OSError, match="Access is denied"):
        _move_with_retry(src, dst, move)

    assert move.attempts == len(_RETRY_DELAYS) + 1
    assert sleeps == list(_RETRY_DELAYS)
    assert sum(sleeps) < 1.0


@pytest.mark.parametrize("winerror", [2, 183, None])
def test_move_with_retry_does_not_retry_other_errors(
    tmp_path: pathlib.Path, sleeps: list[float], winerror: int | None
) -> None:
    """Only a busy destination is transient; everything else fails at once.

    183 is the case that matters: `ERROR_ALREADY_EXISTS` is how the
    `exclusive` path reports that a node is already there, and retrying it
    would overwrite what `_safe_move` refused to touch.
    """
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.write_bytes(b"abc")
    error = OSError(17, "boom")
    if winerror is not None:
        error.winerror = winerror  # type: ignore[attr-defined]
    move = _FlakyMove(1, error)

    with pytest.raises(OSError, match="boom"):
        _move_with_retry(src, dst, move)

    assert move.attempts == 1
    assert sleeps == []


@pytest.mark.parametrize("exclusive", [True, False])
def test_atomic_write_onto_directory(
    tmp_path: pathlib.Path, sleeps: list[float], exclusive: bool
) -> None:
    """Writing a key whose destination is a directory fails and leaves no temp file."""
    path = tmp_path / "node"
    path.mkdir()
    with pytest.raises(OSError), _atomic_write(path, "wb", exclusive=exclusive) as f:
        f.write(b"abc")
    assert path.is_dir()
    assert list(tmp_path.iterdir()) == [path]  # no temp files
