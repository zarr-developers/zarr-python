from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
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
from zarr.core.buffer import Buffer, cpu, default_buffer_prototype
from zarr.core.group import Group
from zarr.core.sync import sync
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

    def test_store_supports_listing(self, store: ZipStore) -> None:
        assert store.supports_listing

    def test_store_supports_deletes(self, store: ZipStore) -> None:
        assert store.supports_deletes

    async def test_delete_compacts_duplicates(self, store: ZipStore) -> None:
        # Overwriting a key leaves a duplicate member in the archive; deleting
        # another key rewrites the archive and should compact the duplicates so
        # the surviving key has a single, most-recent entry (issue #828).
        await store.set("foo", cpu.Buffer.from_bytes(b"v1"))
        with pytest.warns(UserWarning, match="Duplicate name: 'foo'"):
            await store.set("foo", cpu.Buffer.from_bytes(b"v2"))
        await store.set("bar", cpu.Buffer.from_bytes(b"bar"))

        await store.delete("bar")

        assert not await store.exists("bar")
        assert store._zf.namelist().count("foo") == 1
        buf = await self.get(store, "foo")
        assert buf.to_bytes() == b"v2"

    async def test_delete_then_set(self, store: ZipStore) -> None:
        # after a delete (which reopens the archive) writes must still work
        await store.set("foo", cpu.Buffer.from_bytes(b"foo"))
        await store.delete("foo")
        assert not await store.exists("foo")
        await store.set("baz", cpu.Buffer.from_bytes(b"baz"))
        buf = await self.get(store, "baz")
        assert buf.to_bytes() == b"baz"

    async def test_delete_and_delete_dir_auto_open(self, tmp_path: Path) -> None:
        # delete() and delete_dir() should auto-open the archive like _get/_set,
        # rather than assuming the caller opened it first.
        store = ZipStore(tmp_path / "del.zip", mode="w", read_only=False)
        assert not store._is_open
        await store.delete("missing")  # exercises the auto-open branch in delete()
        assert store._is_open

        store2 = ZipStore(tmp_path / "deldir.zip", mode="w", read_only=False)
        assert not store2._is_open
        await store2.delete_dir("missing")  # auto-open branch in delete_dir()
        assert store2._is_open

    async def test_delete_dir_prefix_already_normalized(self, store: ZipStore) -> None:
        # a prefix that already ends in "/" must skip the slash-appending branch
        await store.set("foo/zarr.json", cpu.Buffer.from_bytes(b"a"))
        await store.set("foo/c/0", cpu.Buffer.from_bytes(b"b"))
        await store.set("bar/zarr.json", cpu.Buffer.from_bytes(b"c"))

        await store.delete_dir("foo/")

        assert not await store.exists("foo/zarr.json")
        assert not await store.exists("foo/c/0")
        assert await store.exists("bar/zarr.json")

    async def test_delete_dir_empty_prefix_removes_all(self, store: ZipStore) -> None:
        # an empty prefix also skips normalization and should remove everything
        await store.set("a", cpu.Buffer.from_bytes(b"a"))
        await store.set("b/c", cpu.Buffer.from_bytes(b"b"))

        await store.delete_dir("")

        assert not await store.exists("a")
        assert not await store.exists("b/c")
        assert store._zf.namelist() == []

    async def test_delete_cleans_up_temp_on_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # if the rewrite fails (e.g. os.replace raises), the temporary archive
        # must be removed and the original left untouched.
        import zarr.storage._zip as zip_module

        store = ZipStore(tmp_path / "fail.zip", mode="w", read_only=False)
        await store.set("foo", cpu.Buffer.from_bytes(b"v"))

        def boom(*args: Any, **kwargs: Any) -> None:
            raise OSError("replace failed")

        monkeypatch.setattr(zip_module.os, "replace", boom)

        with pytest.raises(OSError, match="replace failed"):
            await store.delete("foo")

        # no leftover temp file: only the original archive remains
        assert set(os.listdir(tmp_path)) == {"fail.zip"}

    async def test_delete_failure_when_temp_already_removed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # defensive cleanup branch: if the temp archive is already gone when the
        # rewrite fails, cleanup is skipped and the original error still propagates.
        import zarr.storage._zip as zip_module

        store = ZipStore(tmp_path / "fail2.zip", mode="w", read_only=False)
        await store.set("foo", cpu.Buffer.from_bytes(b"v"))

        real_remove = zip_module.os.remove

        def replace_then_vanish(src: str, dst: str) -> None:
            real_remove(src)  # temp disappears before the except block runs
            raise OSError("replace failed")

        monkeypatch.setattr(zip_module.os, "replace", replace_then_vanish)

        with pytest.raises(OSError, match="replace failed"):
            await store.delete("foo")

        assert set(os.listdir(tmp_path)) == {"fail2.zip"}

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

        # assigning an entire chunk to the fill value deletes the chunk;
        # ZipStore now supports deletes by rewriting the archive (issue #828)
        z[0:10, 0:10] = 99
        expected = data.copy()
        expected[0:10, 0:10] = 99
        assert np.array_equal(expected, z[:])

        bar = root.create_group("bar", attributes={"hello": "world"})
        assert "hello" in dict(bar.attrs)

        # keys can now be deleted
        del root["bar"]
        assert "bar" not in root

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


class ZipStoreLifecycleMachine(RuleBasedStateMachine):
    """Drive a ZipStore through construct / open / write / close transitions.

    Invariant under test: a constructed ZipStore can always be closed without
    raising, regardless of whether it was ever opened or did any I/O. This is a
    property-based generalization of the former example-based regression tests
    for ZipStore.close() being called on a never-opened store (which raised
    AttributeError because ``_lock`` is created lazily in ``_sync_open``).
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
