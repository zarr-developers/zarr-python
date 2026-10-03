"""Tests for the builtin, read-only `zip:` URL pipeline adapter."""

from __future__ import annotations

import dataclasses
import io
import zipfile
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pytest

import zarr
from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
    SuffixByteRequest,
)
from zarr.abc.url_pipeline import (
    AdapterResolution,
    PipelineContext,
    PipelineSegment,
    URLPipelineAdapter,
)
from zarr.core.buffer import default_buffer_prototype
from zarr.errors import URLPipelineError
from zarr.registry import get_url_adapter, register_url_adapter
from zarr.storage import LocalStore, WrapperStore, ZipStore
from zarr.storage._common import _has_fsspec
from zarr.storage._url_adapters._zip import ZipAdapter, ZipReaderStore
from zarr.storage._url_pipeline import is_url_pipeline, resolve_pipeline

if TYPE_CHECKING:
    from pathlib import Path

    from zarr.core.common import AccessModeLiteral

pytestmark = pytest.mark.usefixtures("clean_url_adapter_registry")


def _write_archive(path: Path, *, compression: int = zipfile.ZIP_STORED) -> Path:
    """An archive with a v3 hierarchy at `inner-dir/subgroup`, written by ZipStore."""
    with ZipStore(path, mode="w", compression=compression) as store:
        root = zarr.open_group(store, mode="w", path="inner-dir")
        sub = root.create_group("subgroup", attributes={"title": "headline"})
        arr = sub.create_array("x", shape=(6,), chunks=(2,), dtype="i4")
        arr[:] = np.arange(6)
    return path


def _url_path(path: Path) -> str:
    """A path spelled for a `file:` URL (a Windows drive gains a leading slash)."""
    posix = path.as_posix()
    return posix if posix.startswith("/") else f"/{posix}"


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    return _write_archive(tmp_path / "my-archive.zip")


class _CloseSpyStore(WrapperStore[Store]):
    closed: ClassVar[list[str]] = []

    def close(self) -> None:
        type(self).closed.append(str(self))
        super().close()


class _CloseSpyAdapter(URLPipelineAdapter):
    """Wraps the preceding store so tests can observe it being closed."""

    @classmethod
    async def open_pipeline_segment(
        cls, segment: PipelineSegment, context: PipelineContext
    ) -> AdapterResolution:
        preceding = await context.resolve_preceding()
        return dataclasses.replace(preceding, store=_CloseSpyStore(preceding.store))


def test_registered_as_builtin() -> None:
    assert get_url_adapter("zip") is ZipAdapter


@pytest.mark.parametrize("url", ["zip:/tmp/a.zip", "zip:///tmp/a.zip", "zip::file:///tmp/a.zip"])
def test_zip_is_not_a_root_scheme(url: str) -> None:
    # strings without '|' keep their pre-pipeline meaning (fsspec's zip://)
    assert not is_url_pipeline(url)


class TestRead:
    def test_headline_example(self, archive: Path) -> None:
        # zarr-developers/zarr-python#2943's headline example, with a local root
        group = zarr.open_group(f"{archive}|zip:inner-dir|zarr3:subgroup", mode="r")
        assert group.attrs["title"] == "headline"
        x = group["x"]
        assert isinstance(x, zarr.Array)
        np.testing.assert_array_equal(x[:], np.arange(6))

    @pytest.mark.parametrize(
        "template",
        [
            "file:{url}|zip:",
            "file://{url}|zip:",
            "file://localhost{url}|zip",
            "{path}|zip:",
        ],
    )
    def test_root_spellings(self, archive: Path, template: str) -> None:
        url = template.format(url=_url_path(archive), path=archive)
        group = zarr.open_group(f"{url}|zarr3:inner-dir/subgroup", mode="r")
        assert group.attrs["title"] == "headline"

    @pytest.mark.parametrize("body", ["inner-dir/subgroup", "/inner-dir/subgroup", "inner-dir/"])
    def test_leading_slash_is_ignored(self, archive: Path, body: str) -> None:
        result = zarr.open_group(f"file:{_url_path(archive)}|zip:{body}", mode="r")
        assert isinstance(result, zarr.Group)

    def test_double_leading_slash_is_invalid(self, archive: Path) -> None:
        with pytest.raises(URLPipelineError, match="more than one leading '/'"):
            zarr.open_group(f"{archive}|zip://inner-dir", mode="r")

    @pytest.mark.parametrize("body", ["..", "a/../b"])
    def test_invalid_path(self, archive: Path, body: str) -> None:
        with pytest.raises(URLPipelineError, match="invalid path"):
            zarr.open_group(f"{archive}|zip:{body}", mode="r")

    def test_query_is_rejected(self, archive: Path) -> None:
        with pytest.raises(URLPipelineError, match="do not accept a query"):
            zarr.open_group(f"{archive}|zip:a?x=1", mode="r")

    def test_open_array_through_paths(self, archive: Path) -> None:
        arr = zarr.open_array(f"{archive}|zip:inner-dir/subgroup/x", mode="r")
        np.testing.assert_array_equal(arr[1:4], [1, 2, 3])
        via_path = zarr.open_array(f"{archive}|zip:inner-dir", path="subgroup/x", mode="r")
        np.testing.assert_array_equal(via_path[:], np.arange(6))

    @pytest.mark.parametrize("mode", [None, "a", "r"])
    def test_default_and_open_modes_are_read_only(
        self, archive: Path, mode: AccessModeLiteral | None
    ) -> None:
        # "a" (the zarr.open default) is open-or-create: zip: serves the open half
        url = f"{archive}|zip:inner-dir/subgroup"
        node = zarr.open(url) if mode is None else zarr.open(url, mode=mode)
        assert isinstance(node, zarr.Group)
        assert node.store.read_only
        with pytest.raises(ValueError, match="read-only"):
            node.create_group("new")

    @pytest.mark.parametrize("mode", ["w", "w-", "r+"])
    def test_write_modes_raise(self, archive: Path, mode: AccessModeLiteral) -> None:
        with pytest.raises(URLPipelineError, match="zip: is read-only until"):
            zarr.open_group(f"{archive}|zip:", mode=mode)

    @pytest.mark.parametrize(
        "compression", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED, zipfile.ZIP_BZIP2]
    )
    def test_compression_methods(self, tmp_path: Path, compression: int) -> None:
        path = _write_archive(tmp_path / "c.zip", compression=compression)
        arr = zarr.open_array(f"{path}|zip:inner-dir/subgroup/x", mode="r")
        np.testing.assert_array_equal(arr[:], np.arange(6))

    def test_unsupported_compression(self, tmp_path: Path) -> None:
        path = tmp_path / "lzma.zip"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_LZMA) as zf:
            zf.writestr("zarr.json", b"{}")
        group_url = f"{path}|zip:"
        with pytest.raises(NotImplementedError, match="compression method"):
            zarr.open_group(group_url, mode="r")

    def test_archive_with_directory_entries(self, tmp_path: Path, archive: Path) -> None:
        # archives made by `zip -r` carry directory entries; they are not keys
        rewritten = tmp_path / "with-dirs.zip"
        with zipfile.ZipFile(archive) as src, zipfile.ZipFile(rewritten, "w") as dst:
            dst.writestr("inner-dir/", b"")
            dst.writestr("inner-dir/subgroup/", b"")
            for info in src.infolist():
                dst.writestr(info, src.read(info))
        group = zarr.open_group(f"{rewritten}|zip:inner-dir", mode="r")
        assert sorted(group.group_keys()) == ["subgroup"]

    def test_large_central_directory(self, tmp_path: Path) -> None:
        # a central directory larger than the initial suffix read is fetched
        # in further range requests
        path = tmp_path / "many.zip"
        with zipfile.ZipFile(path, "w") as zf:
            for i in range(3000):
                zf.writestr(f"entries/padding-name-to-grow-the-directory-{i:05d}", b"x")
            zf.writestr("zarr.json", b'{"zarr_format": 3, "node_type": "group"}')
        group = zarr.open_group(f"{path}|zip:", mode="r")
        assert isinstance(group, zarr.Group)

    def test_nested_zip(self, tmp_path: Path, archive: Path) -> None:
        outer = tmp_path / "outer.zip"
        with zipfile.ZipFile(outer, "w") as zf:
            zf.write(archive, "nested/inner.zip")
        group = zarr.open_group(
            f"file:{_url_path(outer)}|zip:nested/inner.zip|zip:inner-dir|zarr3:subgroup",
            mode="r",
        )
        assert group.attrs["title"] == "headline"

    def test_nested_zip_compressed_outer(self, tmp_path: Path, archive: Path) -> None:
        outer = tmp_path / "outer.zip"
        with zipfile.ZipFile(outer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(archive, "inner.zip")
        arr = zarr.open_array(f"{outer}|zip:inner.zip|zip:inner-dir/subgroup/x", mode="r")
        np.testing.assert_array_equal(arr[:], np.arange(6))

    @pytest.mark.skipif(not _has_fsspec, reason="requires fsspec")
    def test_fsspec_root(self, archive: Path) -> None:
        # fsspec's `local://` protocol: an FsspecStore root, read by range requests
        from zarr.storage import FsspecStore

        result = zarr.open_group(f"local://{archive.as_posix()}|zip:inner-dir|zarr3:subgroup")
        assert isinstance(result.store, ZipReaderStore)
        assert isinstance(result.store._source, FsspecStore)
        assert result.attrs["title"] == "headline"

    @pytest.mark.skipif(not _has_fsspec, reason="requires fsspec")
    def test_fsspec_memory_filesystem_root(self, archive: Path) -> None:
        # an fsspec in-memory filesystem stands in for a remote object store
        fsspec = pytest.importorskip("fsspec")
        from fsspec.implementations.memory import MemoryFileSystem

        class RemoteLikeFS(MemoryFileSystem):  # type: ignore[misc]
            protocol = ("zarr-test-remote",)
            store: ClassVar[dict[str, object]] = {}
            pseudo_dirs: ClassVar[list[str]] = [""]

            @classmethod
            def _strip_protocol(cls, path: str) -> str:
                return "/" + path.removeprefix("zarr-test-remote://").lstrip("/")

        fsspec.register_implementation("zarr-test-remote", RemoteLikeFS, clobber=True)
        RemoteLikeFS().pipe("/bucket/my-archive.zip", archive.read_bytes())
        group = zarr.open_group("zarr-test-remote://bucket/my-archive.zip|zip:inner-dir/subgroup")
        assert group.attrs["title"] == "headline"

    def test_memory_root(self, archive: Path) -> None:
        from zarr.core.buffer import cpu
        from zarr.storage import ManagedMemoryStore

        holder = ManagedMemoryStore(name="zip-memory-root")
        zarr.core.sync.sync(holder.set("dir/a.zip", cpu.Buffer.from_bytes(archive.read_bytes())))
        group = zarr.open_group("memory:zip-memory-root/dir/a.zip|zip:inner-dir/subgroup")
        assert group.attrs["title"] == "headline"


class TestErrors:
    def test_not_a_zip_file(self, tmp_path: Path) -> None:
        path = tmp_path / "not.zip"
        path.write_bytes(b"definitely not a zip archive")
        with pytest.raises(URLPipelineError, match="could not open the ZIP archive"):
            zarr.open_group(f"{path}|zip:", mode="r")

    def test_missing_archive(self, tmp_path: Path) -> None:
        with pytest.raises((URLPipelineError, FileNotFoundError)):
            zarr.open_group(f"{tmp_path / 'missing.zip'}|zip:", mode="r")

    def test_directory_root(self, tmp_path: Path) -> None:
        with pytest.raises(URLPipelineError, match="could not open the ZIP archive"):
            zarr.open_group(f"{tmp_path}|zip:", mode="r")

    def test_preceding_store_closed_on_error(self, tmp_path: Path) -> None:
        register_url_adapter("spy", _CloseSpyAdapter)
        _CloseSpyStore.closed.clear()
        path = tmp_path / "not.zip"
        path.write_bytes(b"definitely not a zip archive")
        with pytest.raises(URLPipelineError):
            zarr.open_group(f"{path}|spy:|zip:", mode="r")
        assert len(_CloseSpyStore.closed) == 1

    async def test_close_closes_source(self, archive: Path) -> None:
        register_url_adapter("spy", _CloseSpyAdapter)
        _CloseSpyStore.closed.clear()
        result = await resolve_pipeline(f"{archive}|spy:|zip:", mode="r")
        assert _CloseSpyStore.closed == []
        result.store.close()
        assert len(_CloseSpyStore.closed) == 1


class TestZipReaderStore:
    @pytest.fixture
    async def store(self, tmp_path: Path) -> ZipReaderStore:
        path = tmp_path / "s.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("a/b", b"0123456789")
            zf.writestr("a/c/d", b"x")
            zf.writestr("e", b"y")
        with zipfile.ZipFile(path, "a", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("z", b"0123456789" * 10)
        source = await LocalStore.open(path, read_only=True)
        return await ZipReaderStore.open_archive(source)

    async def test_get_and_ranges(self, store: ZipReaderStore) -> None:
        proto = default_buffer_prototype()
        for key in ("a/b", "z"):
            full = await store.get(key, proto)
            assert full is not None
            data = full.to_bytes()
            cases: list[tuple[ByteRequest, bytes]] = [
                (RangeByteRequest(2, 5), data[2:5]),
                (RangeByteRequest(8, 50), data[8:50]),
                (OffsetByteRequest(7), data[7:]),
                (SuffixByteRequest(3), data[-3:]),
                (RangeByteRequest(200, 300), b""),
            ]
            for request, expected in cases:
                got = await store.get(key, proto, request)
                assert got is not None
                assert got.to_bytes() == expected
        assert await store.get("missing", proto) is None
        values = await store.get_partial_values(
            proto, [("e", None), ("a/b", RangeByteRequest(0, 1))]
        )
        assert [v.to_bytes() if v else None for v in values] == [b"y", b"0"]

    async def test_listing(self, store: ZipReaderStore) -> None:
        assert sorted([k async for k in store.list()]) == ["a/b", "a/c/d", "e", "z"]
        assert sorted([k async for k in store.list_prefix("a/")]) == ["a/b", "a/c/d"]
        assert sorted([k async for k in store.list_dir("")]) == ["a", "e", "z"]
        assert sorted([k async for k in store.list_dir("a")]) == ["b", "c"]
        assert await store.exists("e")
        assert not await store.exists("a")
        assert await store.getsize("z") == 100
        with pytest.raises(FileNotFoundError):
            await store.getsize("missing")

    async def test_read_only(self, store: ZipReaderStore) -> None:
        from zarr.core.buffer import cpu

        assert store.read_only
        with pytest.raises(ValueError, match="read-only"):
            await store.set("k", cpu.Buffer.from_bytes(b""))
        with pytest.raises(ValueError, match="read-only"):
            await store.delete("e")

    async def test_eq_and_str(self, store: ZipReaderStore) -> None:
        same = store
        assert store == same
        assert store != object()
        assert str(store).endswith("|zip:")
        assert "ZipReaderStore" in repr(store)

    async def test_bad_crc(self, tmp_path: Path) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("k", b"payload")
        raw = bytearray(buf.getvalue())
        raw[raw.index(b"payload")] ^= 0xFF
        path = tmp_path / "bad.zip"
        path.write_bytes(bytes(raw))
        store = await ZipReaderStore.open_archive(await LocalStore.open(path, read_only=True))
        with pytest.raises(OSError, match="bad CRC-32"):
            await store.get("k", default_buffer_prototype())
