from __future__ import annotations

import io
import os
import pickle
import shutil
import tempfile
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, wait
from typing import TYPE_CHECKING

import numpy as np
import pytest
from hypothesis import settings
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    precondition,
    rule,
    run_state_machine_as_test,
)

import zarr
from zarr import create_array
from zarr.abc.store import Store
from zarr.core.buffer import Buffer, cpu, default_buffer_prototype
from zarr.core.sync import sync
from zarr.storage import ZipStore
from zarr.testing.store import StoreTests

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path
    from typing import Any

    from zarr.storage._zip import ZipStoreAccessModeLiteral


# TODO: work out where this is coming from and fix
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:coroutine method 'aclose' of 'ZipStore.list' was never awaited:RuntimeWarning"
    )
]


async def _drain(keys: AsyncIterator[str]) -> list[str]:
    return [k async for k in keys]


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
        foo = root.get_group("foo")
        foo["bar"] = np.array([1])
        shutil.make_archive(str(zarr_path), "zip", zarr_path)
        zip_path = tmp_path / "foo.zarr.zip"
        zipped = zarr.open_group(ZipStore(zip_path, mode="r"), mode="r")
        assert list(zipped.keys()) == list(root.keys())
        group = zipped.get_group("foo")
        assert list(group.keys()) == list(group.keys())

    async def test_list_without_explicit_open(self, tmp_path: Path) -> None:
        # ZipStore.list(), list_dir(), and exists() should auto-open
        # the zip file just like _get() and _set() do.
        zip_path = tmp_path / "data.zip"
        zarr_path = tmp_path / "foo.zarr"
        root = zarr.open_group(store=zarr_path, mode="w")
        root["x"] = np.array([1, 2, 3])
        shutil.make_archive(str(zarr_path), "zip", zarr_path)
        shutil.move(f"{zarr_path}.zip", zip_path)

        store = ZipStore(zip_path, mode="r")
        assert not store._is_open

        keys = [k async for k in store.list()]
        assert len(keys) > 0

        store2 = ZipStore(zip_path, mode="r")
        assert not store2._is_open
        assert await store2.exists(keys[0])

        store3 = ZipStore(zip_path, mode="r")
        assert not store3._is_open
        dir_keys = [k async for k in store3.list_dir("")]
        assert len(dir_keys) > 0

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

    @pytest.mark.parametrize(
        ("call", "result", "keys"),
        [
            pytest.param(
                lambda s: s.get("foo", default_buffer_prototype()), b"bar", ["foo"], id="get"
            ),
            pytest.param(
                lambda s: s.get_partial_values(default_buffer_prototype(), [("foo", None)]),
                [b"bar"],
                ["foo"],
                id="get_partial_values",
            ),
            pytest.param(lambda s: s.exists("foo"), True, ["foo"], id="exists"),
            pytest.param(
                lambda s: s.set("bar", cpu.Buffer.from_bytes(b"x")),
                None,
                ["foo", "bar"],
                id="set",
            ),
            pytest.param(
                lambda s: s.set_if_not_exists("bar", cpu.Buffer.from_bytes(b"x")),
                None,
                ["foo", "bar"],
                id="set_if_not_exists",
            ),
            pytest.param(lambda s: s.clear(), None, [], id="clear"),
            pytest.param(lambda s: s.delete("bar"), None, ["foo"], id="delete"),
            pytest.param(lambda s: s.delete_dir("bar"), None, ["foo"], id="delete_dir"),
            pytest.param(lambda s: s.is_empty(""), False, ["foo"], id="is_empty"),
        ],
    )
    async def test_methods_open_store_on_first_use(
        self, store_kwargs: dict[str, Any], call: Any, result: Any, keys: list[str]
    ) -> None:
        # every method works on a store that was constructed but never opened,
        # and sees the entries already in the archive
        seed = await self.store_cls.open(**store_kwargs)
        await seed.set("foo", cpu.Buffer.from_bytes(b"bar"))
        seed.close()

        store = self.store_cls(**{**store_kwargs, "mode": "a"})
        assert not store._is_open
        out = await call(store)
        if isinstance(out, list):
            out = [b.to_bytes() for b in out]
        elif isinstance(out, Buffer):
            out = out.to_bytes()
        assert out == result
        store.close()
        with zipfile.ZipFile(store_kwargs["path"]) as zf:
            assert zf.namelist() == keys

    @pytest.mark.parametrize("method", ["list", "list_dir"])
    async def test_listing_opens_store_on_first_use(
        self, store_kwargs: dict[str, Any], method: str
    ) -> None:
        seed = await self.store_cls.open(**store_kwargs)
        await seed.set("foo", cpu.Buffer.from_bytes(b"bar"))
        seed.close()

        store = self.store_cls(**{**store_kwargs, "mode": "r"})
        args = () if method == "list" else ("",)
        assert [k async for k in getattr(store, method)(*args)] == ["foo"]
        store.close()

    @pytest.mark.parametrize("mode", ["w", "x"])
    @pytest.mark.parametrize("reopen", ["use_after_close", "move", "pickle"])
    async def test_reopen_keeps_entries(
        self, tmp_path: Path, mode: ZipStoreAccessModeLiteral, reopen: str
    ) -> None:
        # "w" truncates and "x" refuses an existing file; neither may apply
        # when a store that already wrote entries is opened again
        store = ZipStore(tmp_path / "data.zip", mode=mode)
        await store.set("foo", cpu.Buffer.from_bytes(b"bar"))
        if reopen == "use_after_close":
            store.close()
            value = await store.get("foo", default_buffer_prototype())
            assert value is not None
            assert value.to_bytes() == b"bar"
            store.close()
        elif reopen == "move":
            await store.move(tmp_path / "moved" / "data.zip")
            store.close()
        else:
            store.close()
            pickle.loads(pickle.dumps(store)).close()
        with zipfile.ZipFile(store.path) as zf:  # type: ignore[arg-type]
            assert zf.namelist() == ["foo"]

    async def test_close_blocks_concurrent_reopen(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # a thread that uses the store while close() runs must wait until the
        # archive is closed, or it reopens a file with no central directory
        store = ZipStore(tmp_path / "data.zip", mode="w")
        await store.set("foo", cpu.Buffer.from_bytes(b"1"))

        writer = threading.Thread(
            target=lambda: sync(store.set("bar", cpu.Buffer.from_bytes(b"2")))
        )
        mark_closed = Store.close

        def mark_closed_then_write(self: Store) -> None:
            # the store now reports closed; start a write before close() returns
            mark_closed(self)
            writer.start()
            writer.join(timeout=0.2)

        monkeypatch.setattr(Store, "close", mark_closed_then_write)
        store.close()
        monkeypatch.undo()
        writer.join(timeout=5)
        assert not writer.is_alive()
        store.close()

        with zipfile.ZipFile(store.path) as zf:  # type: ignore[arg-type]
            assert zf.namelist() == ["foo", "bar"]

    async def test_move_blocks_concurrent_reopen(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # a thread that uses the store while move() runs must wait for the
        # move, or it reopens the old path between close and reopen
        origin = tmp_path / "data.zip"
        destination = tmp_path / "moved" / "data.zip"
        store = ZipStore(origin, mode="w")
        await store.set("foo", cpu.Buffer.from_bytes(b"1"))

        writer = threading.Thread(
            target=lambda: sync(store.set("bar", cpu.Buffer.from_bytes(b"2")))
        )
        move_file = shutil.move

        def write_then_move(src: Any, dst: Any) -> Any:
            # the store is closed here; start a write before move() reopens it
            writer.start()
            writer.join(timeout=0.2)
            return move_file(src, dst)

        monkeypatch.setattr(shutil, "move", write_then_move)
        await store.move(destination)
        monkeypatch.undo()
        writer.join(timeout=5)
        assert not writer.is_alive()
        store.close()

        assert not origin.exists()
        with zipfile.ZipFile(destination) as zf:
            assert zf.namelist() == ["foo", "bar"]

    async def test_clear_exclusive_mode_keeps_existing_file(self, tmp_path: Path) -> None:
        # a never-opened "x" store has not claimed the file, so clear() must
        # refuse it the way the first open would instead of deleting it
        path = tmp_path / "data.zip"
        with zipfile.ZipFile(path, mode="w") as zf:
            zf.writestr("foo", b"1")

        store = ZipStore(path, mode="x")
        with pytest.raises(FileExistsError):
            await store.clear()
        with zipfile.ZipFile(path) as zf:
            assert zf.namelist() == ["foo"]

    @pytest.mark.parametrize(
        "use",
        [
            pytest.param(lambda s: s._ensure_open(), id="ensure_open"),
            pytest.param(lambda s: _drain(s.list_dir("")), id="list_dir"),
        ],
    )
    async def test_first_use_waits_for_lock(self, tmp_path: Path, use: Any) -> None:
        # a thread's first use must wait while another thread holds the lock,
        # then find the archive that thread opened instead of opening it again
        store = ZipStore(tmp_path / "data.zip", mode="w")
        with ThreadPoolExecutor(max_workers=1) as pool:
            with store._lock:
                first_use = pool.submit(sync, use(store))
                wait([first_use], timeout=0.2)
                assert not store._is_open
                store._zipfile()
            first_use.result(timeout=5)
        store.close()

    async def test_failed_close_leaves_store_reusable(self, tmp_path: Path) -> None:
        # if closing the archive raises, the store is still marked closed so
        # the next use reopens it instead of hitting a dead handle
        store = ZipStore(tmp_path / "data.zip", mode="w")
        await store.set("foo", cpu.Buffer.from_bytes(b"1"))
        close_archive = store._zf.close

        def close_then_fail() -> None:
            close_archive()
            raise OSError("disk full")

        store._zf.close = close_then_fail  # type: ignore[method-assign]
        with pytest.raises(OSError, match="disk full"):
            store.close()
        assert not store._is_open

        await store.set("bar", cpu.Buffer.from_bytes(b"2"))
        store.close()
        with zipfile.ZipFile(store.path) as zf:  # type: ignore[arg-type]
            assert zf.namelist() == ["foo", "bar"]


class TestZipStoreFileObj:
    """ZipStore backed by an open binary file-like object instead of a path."""

    @pytest.fixture
    def zip_bytes(self, tmp_path: Path) -> bytes:
        path = tmp_path / "data.zip"
        store = ZipStore(path, mode="w")
        zarr.create_array(store, data=np.arange(10), chunks=(5,))
        store.close()
        return path.read_bytes()

    def test_read_from_fileobj(self, zip_bytes: bytes) -> None:
        # an existing archive can be read through any seekable binary reader
        store = ZipStore(io.BytesIO(zip_bytes), mode="r")
        array = zarr.open_array(store, mode="r")
        assert np.array_equal(array[...], np.arange(10))
        assert store.path is None

    def test_write_to_fileobj(self) -> None:
        # a writable file object receives the archive; the bytes it holds
        # after close() are a complete, reopenable zip
        buffer = io.BytesIO()
        store = ZipStore(buffer, mode="w", read_only=False)
        zarr.create_array(store, data=np.arange(4))
        store.close()

        roundtrip = ZipStore(io.BytesIO(buffer.getvalue()), mode="r")
        array = zarr.open_array(roundtrip, mode="r")
        assert np.array_equal(array[...], np.arange(4))

    async def test_write_after_close_keeps_entries(self) -> None:
        # reopening a file object after close() appends to the archive it
        # already holds instead of starting a new one after it
        buffer = io.BytesIO()
        store = ZipStore(buffer, mode="w", read_only=False)
        await store.set("foo", cpu.Buffer.from_bytes(b"1"))
        store.close()
        await store.set("bar", cpu.Buffer.from_bytes(b"2"))
        store.close()

        with zipfile.ZipFile(io.BytesIO(buffer.getvalue())) as zf:
            assert zf.namelist() == ["foo", "bar"]

    async def test_write_only_reuse_after_close_raises(self, tmp_path: Path) -> None:
        # a write-only file object cannot be read back, so reopening it would
        # start a new archive and drop the entries already written
        path = tmp_path / "data.zip"
        with path.open("wb") as f:
            store = ZipStore(f, mode="w", read_only=False)
            await store.set("foo", cpu.Buffer.from_bytes(b"1"))
            store.close()
            with pytest.raises(io.UnsupportedOperation, match="readable and seekable"):
                await store.set("bar", cpu.Buffer.from_bytes(b"2"))

        with zipfile.ZipFile(path) as zf:
            assert zf.namelist() == ["foo"]

    async def test_unseekable_reuse_after_close_raises(self) -> None:
        # a file object that cannot seek cannot be read back either, so reuse
        # after close() raises the same error as a write-only one
        buffer = io.BytesIO()
        pipe = io.BufferedRWPair(buffer, buffer)  # readable, not seekable
        store = ZipStore(pipe, mode="w", read_only=False)  # type: ignore[arg-type]
        await store.set("foo", cpu.Buffer.from_bytes(b"1"))
        store.close()
        with pytest.raises(io.UnsupportedOperation, match="readable and seekable"):
            await store.set("bar", cpu.Buffer.from_bytes(b"2"))

    async def test_clear_unsupported(self, zip_bytes: bytes) -> None:
        # clear() requires a filesystem location, so it raises a clear error
        # for file-object-backed stores
        store = ZipStore(io.BytesIO(zip_bytes), mode="a", read_only=False)
        store._sync_open()
        with pytest.raises(NotImplementedError, match="clear.*file-like"):
            await store.clear()

    async def test_move_unsupported(self, zip_bytes: bytes) -> None:
        # move() requires a filesystem location, so it raises a clear error
        # for file-object-backed stores
        store = ZipStore(io.BytesIO(zip_bytes), mode="a", read_only=False)
        store._sync_open()
        with pytest.raises(NotImplementedError, match="move.*file-like"):
            await store.move("elsewhere.zip")

    def test_invalid_file_object_rejected(self) -> None:
        # objects without read/seek/tell are rejected at construction, not
        # deep inside zipfile
        with pytest.raises(TypeError, match="read/seek/tell"):
            ZipStore(42, mode="r")  # type: ignore[arg-type]

    @pytest.mark.parametrize("mode", ["w", "a", "x"])
    def test_non_iobase_reader_write_modes_rejected(self, zip_bytes: bytes, mode: str) -> None:
        # readers that are not io.IOBase instances are adapted for reading
        # only; write modes are rejected at construction with a clear error
        class MinimalReader:
            def __init__(self, data: bytes) -> None:
                self._buffer = io.BytesIO(data)

            def read(self, size: int, /) -> bytes:
                return self._buffer.read(size)

            def seek(self, pos: int, whence: int = 0, /) -> int:
                return self._buffer.seek(pos, whence)

            def tell(self) -> int:
                return self._buffer.tell()

        with pytest.raises(TypeError, match="opened for reading"):
            ZipStore(MinimalReader(zip_bytes), mode=mode, read_only=False)  # type: ignore[arg-type]

    def test_fsspec_file(self, tmp_path: Path, zip_bytes: bytes) -> None:
        # a file opened through fsspec (already an io.IOBase) is used directly;
        # fsspec's local filesystem stands in for a remote one
        fsspec = pytest.importorskip("fsspec")

        path = tmp_path / "fsspec.zip"
        path.write_bytes(zip_bytes)
        with fsspec.open(f"local://{path}", "rb") as fileobj:
            store = ZipStore(fileobj, mode="r")
            array = zarr.open_array(store, mode="r")
            assert np.array_equal(array[...], np.arange(10))
            assert store.path is None

    def test_obstore_reader(self, tmp_path: Path, zip_bytes: bytes) -> None:
        # obstore's ReadableFile is not an io.IOBase and its read() returns a
        # buffer-protocol object; ZipStore adapts it via _RawReaderAdapter
        obstore = pytest.importorskip("obstore")
        from obstore.store import LocalStore as ObstoreLocalStore

        (tmp_path / "obstore.zip").write_bytes(zip_bytes)
        reader = obstore.open_reader(ObstoreLocalStore(str(tmp_path)), "obstore.zip")
        store = ZipStore(reader, mode="r")
        array = zarr.open_array(store, mode="r")
        assert np.array_equal(array[...], np.arange(10))

    def test_raw_reader_adapter_eof(self) -> None:
        from zarr.storage._zip import _RawReaderAdapter

        class MinimalReader:
            """Non-io.IOBase reader exposing only read/seek/tell, like obstore."""

            def __init__(self, data: bytes) -> None:
                self._buffer = io.BytesIO(data)

            def read(self, size: int, /) -> bytes:
                return self._buffer.read(size)

            def seek(self, pos: int, whence: int = 0, /) -> int:
                return self._buffer.seek(pos, whence)

            def tell(self) -> int:
                return self._buffer.tell()

        # the adapter must clamp reads to EOF: some readers (obstore < 0.6)
        # raise on short reads instead of returning fewer bytes
        data = b"0123456789"
        adapter = _RawReaderAdapter(MinimalReader(data))  # type: ignore[arg-type]

        # A read straddling EOF returns only the remaining bytes.
        adapter.seek(len(data) - 3)
        buf = bytearray(8)
        assert adapter.readinto(buf) == 3
        assert bytes(buf[:3]) == data[-3:]

        # A read at EOF returns 0.
        assert adapter.tell() == len(data)
        assert adapter.readinto(bytearray(8)) == 0

    def test_pickle_fileobj_raises(self, zip_bytes: bytes) -> None:
        # an open file object cannot be reliably serialized, so pickling a
        # file-object-backed store raises with a pointer at the alternative
        store = ZipStore(io.BytesIO(zip_bytes), mode="r")
        with pytest.raises(TypeError, match="cannot pickle a ZipStore backed by a file-like"):
            pickle.dumps(store)

    def test_pickle_path_backed_roundtrip(self, tmp_path: Path, zip_bytes: bytes) -> None:
        # path-backed stores remain picklable: the path is serialized and the
        # archive is reopened on unpickling
        path = tmp_path / "pickled.zip"
        path.write_bytes(zip_bytes)
        store = ZipStore(path, mode="r")
        unpickled = pickle.loads(pickle.dumps(store))
        array = zarr.open_array(unpickled, mode="r")
        assert np.array_equal(array[...], np.arange(10))

    def test_str_and_eq(self, zip_bytes: bytes) -> None:
        # file-object-backed stores stringify with the object repr and
        # compare equal only when backed by the very same file object
        fileobj = io.BytesIO(zip_bytes)
        store = ZipStore(fileobj, mode="r")
        assert str(store).startswith("zip://<")
        assert store == ZipStore(fileobj, mode="r")
        assert store != ZipStore(io.BytesIO(zip_bytes), mode="r")


class ZipStoreLifecycleMachine(RuleBasedStateMachine):
    """Drive a ZipStore through construct / open / write / close transitions.

    Invariant under test: a constructed ZipStore can always be closed without
    raising, regardless of whether it was ever opened or did any I/O. This is a
    property-based generalization of the former example-based regression tests
    for ZipStore.close() being called on a never-opened store.
    """

    def __init__(self, tmp_path: Path) -> None:
        super().__init__()
        self._tmp_path = tmp_path
        self._counter = 0
        self.store: ZipStore | None = None
        self._opened = False

    @initialize()
    def start(self) -> None:
        self.store = None
        self._opened = False

    @precondition(lambda self: self.store is None)
    @rule()
    def construct(self) -> None:
        # Fresh path each time so mode="w" never clobbers a closed archive.
        self._counter += 1
        self.store = ZipStore(self._tmp_path / f"s{self._counter}.zip", mode="w")
        self._opened = False

    @precondition(lambda self: self.store is not None and not self._opened)
    @rule()
    def open(self) -> None:
        assert self.store is not None
        self.store._sync_open()
        self._opened = True

    @precondition(lambda self: self.store is not None and not self._opened)
    @rule()
    def write(self) -> None:
        assert self.store is not None
        # store.set auto-opens the store.
        sync(self.store.set("a", cpu.Buffer.from_bytes(b"hi")))
        self._opened = True

    @precondition(lambda self: self.store is not None)
    @rule()
    def close(self) -> None:
        assert self.store is not None
        # The property under test: close() must never raise, even with no
        # prior open or I/O.
        self.store.close()
        self.store = None
        self._opened = False


def test_zipstore_close_lifecycle(tmp_path: Path) -> None:
    run_state_machine_as_test(  # type: ignore[no-untyped-call]
        lambda: ZipStoreLifecycleMachine(tmp_path),
        settings=settings(max_examples=50, deadline=None),
    )
