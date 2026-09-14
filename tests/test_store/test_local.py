from __future__ import annotations

import asyncio
import inspect
import io
import os
import pathlib
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import zarr
import zarr.storage._local
from zarr import create_array
from zarr.core.buffer import Buffer, cpu
from zarr.storage import LocalStore
from zarr.storage._local import _atomic_write
from zarr.testing.store import StoreTests
from zarr.testing.utils import assert_bytes_equal

if TYPE_CHECKING:
    from collections.abc import Iterator

_LOCAL_STORE_FILE = zarr.storage._local.__file__
_ASYNC_CODE_FLAGS = inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR

try:
    import coverage as _coverage
except ImportError:  # pragma: no cover
    _COVERAGE_DIR = None
else:
    # coverage.py canonicalizes a source file's path (os.path.realpath, hence os.lstat) the
    # first time its tracer sees code from that file, and it does so on whatever thread is
    # running that code. Such calls carry coverage's own frames and are not LocalStore's.
    _COVERAGE_DIR = os.path.dirname(_coverage.__file__) + os.sep

# The syscall-level entry points that pathlib / os.path / shutil helpers bottom out in.
# Patching these, rather than each Path method, catches a blocking call no matter which
# helper made it. The list is deliberately wider than what LocalStore uses today.
_FILESYSTEM_CALLS: tuple[tuple[Any, str], ...] = (
    (os, "stat"),
    (os, "lstat"),
    (os, "access"),
    (os, "scandir"),
    (os, "listdir"),
    (os, "mkdir"),
    (os, "rmdir"),
    (os, "unlink"),
    (os, "remove"),
    (os, "link"),
    (os, "rename"),
    (os, "replace"),
    (os, "utime"),
    (os, "open"),
    (os, "fsync"),
    (io, "open"),
)


@dataclass
class _FilesystemCalls:
    """What the patched filesystem entry points saw from LocalStore code during one test."""

    off_loop: set[tuple[str, str]] = field(default_factory=set)
    """``(outermost LocalStore function, op)`` pairs made from a thread with no running loop."""
    on_loop: list[str] = field(default_factory=list)
    """Calls made on an event loop's thread from inside a LocalStore coroutine: violations."""

    def record(self, op: str) -> None:
        frame = inspect.currentframe()
        innermost = outermost = None
        while frame is not None:
            if _COVERAGE_DIR is not None and frame.f_code.co_filename.startswith(_COVERAGE_DIR):
                return  # the coverage tracer resolving a filename, not LocalStore doing I/O
            if frame.f_code.co_filename == _LOCAL_STORE_FILE:
                if innermost is None:
                    innermost = frame
                outermost = frame
            frame = frame.f_back
        if outermost is None or innermost is None:
            return  # not LocalStore's doing (pytest, tmp_path, the test body, ...)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # A worker thread, such as the one asyncio.to_thread uses: blocking is fine here.
            self.off_loop.add((outermost.f_code.co_name, op))
            return
        if not outermost.f_code.co_flags & _ASYNC_CODE_FLAGS:
            return  # a synchronous LocalStore method: its caller chose to block the loop
        site = (
            f"LocalStore.{outermost.f_code.co_name} called {op} at _local.py:{innermost.f_lineno}"
        )
        if site not in self.on_loop:
            self.on_loop.append(site)


class TestLocalStore(StoreTests[LocalStore, cpu.Buffer]):
    store_cls = LocalStore
    buffer_cls = cpu.Buffer

    async def test_delete_dir_on_a_file_raises(self, tmp_path: pathlib.Path) -> None:
        """`delete_dir` refuses a prefix that names a file rather than a directory."""
        store = await LocalStore.open(tmp_path)
        await store.set("file", self.buffer_cls.from_bytes(b"x"))
        with pytest.raises(ValueError, match="that is a file"):
            await store.delete_dir("file")
        assert await store.exists("file")

    @pytest.fixture(autouse=True)
    def filesystem_calls(self, monkeypatch: pytest.MonkeyPatch) -> Iterator[_FilesystemCalls]:
        """Fail if any LocalStore coroutine does filesystem I/O on the event loop thread.

        Every async method must hand its filesystem work to ``asyncio.to_thread``;
        doing it inline stalls every other task sharing the loop.
        """
        calls = _FilesystemCalls()

        def patch(module: Any, name: str) -> None:
            original = getattr(module, name)

            def wrapper(*args: Any, **kwargs: Any) -> Any:
                calls.record(f"{module.__name__}.{name}")
                return original(*args, **kwargs)

            monkeypatch.setattr(module, name, wrapper)

        for module, name in _FILESYSTEM_CALLS:
            patch(module, name)
        yield calls
        assert not calls.on_loop, "filesystem calls on the event loop thread:\n" + "\n".join(
            calls.on_loop
        )

    async def test_filesystem_calls_are_observed(
        self, store: LocalStore, filesystem_calls: _FilesystemCalls
    ) -> None:
        """The detector must actually see LocalStore's I/O, or its silence means nothing."""
        await store.set("foo", self.buffer_cls.from_bytes(b"x"))
        await store.get("foo")
        assert ("_put", "io.open") in filesystem_calls.off_loop
        assert ("_get", "io.open") in filesystem_calls.off_loop

    async def test_concurrent_lazy_open(self, store_not_open: LocalStore) -> None:
        """Concurrent first calls on an unopened store all succeed.

        Opening now suspends (the root check runs in a thread), so every caller that
        finds the store closed races to open it; none of them may hit
        ``Store._open``'s "already open" error.
        """
        data = self.buffer_cls.from_bytes(b"x")
        keys = [f"k{i}" for i in range(8)]
        await asyncio.gather(
            store_not_open.get("missing"), *(store_not_open.set(k, data) for k in keys)
        )
        assert store_not_open._is_open
        for key in keys:
            assert_bytes_equal(await store_not_open.get(key), data)

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
