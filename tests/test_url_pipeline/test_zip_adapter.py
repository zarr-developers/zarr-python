"""Tests for the builtin, read-only `zip:` URL pipeline adapter."""

from __future__ import annotations

import io
import zipfile
from importlib.metadata import entry_points, version
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pytest
from packaging.version import parse as parse_version

import zarr
import zarr.api.asynchronous
from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
    SuffixByteRequest,
)
from zarr.abc.url_pipeline import URLPipelineAdapter
from zarr.core.buffer import cpu, default_buffer_prototype
from zarr.errors import URLPipelineError
from zarr.registry import register_url_adapter
from zarr.storage import (
    FsspecStore,
    ManagedMemoryStore,
    PipelineContext,
    PipelineSegment,
    StorePath,
    URLPipeline,
    WrapperStore,
    ZipStore,
)
from zarr.storage._common import _has_fsspec

if TYPE_CHECKING:
    from pathlib import Path

    from zarr.core.common import AccessModeLiteral

pytestmark = pytest.mark.usefixtures("clean_url_adapter_registry")

# fsspec < 2024.12.0 has no AsyncFileSystemWrapper, so zarr cannot open a
# synchronous fsspec filesystem (local://, memory) there
_has_async_fsspec_wrapper = _has_fsspec and parse_version(version("fsspec")) >= parse_version(
    "2024.12.0"
)
requires_async_fsspec_wrapper = pytest.mark.skipif(
    not _has_async_fsspec_wrapper, reason="requires fsspec>=2024.12.0"
)


def _write_archive(
    path: Path, *, compression: int = zipfile.ZIP_STORED, zarr_format: int = 3
) -> Path:
    """An archive with a hierarchy at `inner-dir/subgroup`, written by ZipStore."""
    with ZipStore(path, mode="w", compression=compression) as store:
        root = zarr.open_group(store, mode="w", path="inner-dir", zarr_format=zarr_format)  # type: ignore[arg-type]
        sub = root.create_group("subgroup", attributes={"title": "headline"})
        arr = sub.create_array("x", shape=(6,), chunks=(2,), dtype="i4")
        arr[:] = np.arange(6)
    return path


def _file_url(path: Path) -> str:
    """A `file:` URL for a local path (a Windows drive gains a leading slash)."""
    posix = path.as_posix()
    return f"file:{posix}" if posix.startswith("/") else f"file:/{posix}"


def _pipeline(url: str) -> URLPipeline:
    return URLPipeline.from_url(url)


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
    ) -> StorePath:
        preceding = await context.resolve_preceding()
        return StorePath(_CloseSpyStore(preceding.store), preceding.path)


def test_registered_as_builtin_entry_point() -> None:
    (entry_point,) = entry_points(group="zarr.url_adapters", name="zip")
    adapter = entry_point.load()
    assert issubclass(adapter, URLPipelineAdapter)


class TestRead:
    def test_group_and_array(self, archive: Path) -> None:
        group = zarr.open_group(
            _pipeline(f"{_file_url(archive)}|zip:inner-dir|zarr3:subgroup"), mode="r"
        )
        assert group.attrs["title"] == "headline"
        assert group.store.read_only
        x = group["x"]
        assert isinstance(x, zarr.Array)
        np.testing.assert_array_equal(x[:], np.arange(6))

    def test_open_array(self, archive: Path) -> None:
        arr = zarr.open_array(
            _pipeline(f"{_file_url(archive)}|zip:inner-dir/subgroup/x|zarr3:"), mode="r"
        )
        np.testing.assert_array_equal(arr[1:4], [1, 2, 3])

    @pytest.mark.parametrize(
        ("zip_body", "format_body"),
        [
            ("", "inner-dir/subgroup"),
            ("inner-dir", "subgroup"),
            ("inner-dir/subgroup", ""),
            ("/inner-dir/subgroup", ""),
            ("inner-dir/subgroup/", ""),
        ],
    )
    def test_nested_paths(self, archive: Path, zip_body: str, format_body: str) -> None:
        group = zarr.open_group(
            _pipeline(f"{_file_url(archive)}|zip:{zip_body}|zarr3:{format_body}"), mode="r"
        )
        assert group.attrs["title"] == "headline"

    @pytest.mark.parametrize("zarr_format", [2, 3])
    def test_format_segments(self, tmp_path: Path, zarr_format: int) -> None:
        path = _write_archive(tmp_path / "f.zip", zarr_format=zarr_format)
        group = zarr.open_group(
            _pipeline(f"{_file_url(path)}|zip:inner-dir|zarr{zarr_format}:subgroup"), mode="r"
        )
        assert group.metadata.zarr_format == zarr_format
        assert group.attrs["title"] == "headline"
        # the other format finds no node
        other = 5 - zarr_format
        with pytest.raises(FileNotFoundError):
            zarr.open_group(
                _pipeline(f"{_file_url(path)}|zip:inner-dir|zarr{other}:subgroup"), mode="r"
            )

    def test_without_format_segment(self, archive: Path) -> None:
        group = zarr.open_group(_pipeline(f"{_file_url(archive)}|zip:inner-dir/subgroup"), mode="r")
        assert group.attrs["title"] == "headline"

    @pytest.mark.parametrize(
        "compression", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED, zipfile.ZIP_BZIP2]
    )
    def test_compression_methods(self, tmp_path: Path, compression: int) -> None:
        path = _write_archive(tmp_path / "c.zip", compression=compression)
        arr = zarr.open_array(_pipeline(f"{_file_url(path)}|zip:inner-dir/subgroup/x"), mode="r")
        np.testing.assert_array_equal(arr[:], np.arange(6))

    def test_archive_with_directory_entries(self, tmp_path: Path, archive: Path) -> None:
        # archives made by `zip -r` carry directory entries; they are not keys
        rewritten = tmp_path / "with-dirs.zip"
        with zipfile.ZipFile(archive) as src, zipfile.ZipFile(rewritten, "w") as dst:
            dst.writestr("inner-dir/", b"")
            dst.writestr("inner-dir/subgroup/", b"")
            for info in src.infolist():
                dst.writestr(info, src.read(info))
        group = zarr.open_group(_pipeline(f"{_file_url(rewritten)}|zip:inner-dir"), mode="r")
        assert sorted(group.group_keys()) == ["subgroup"]

    def test_large_central_directory(self, tmp_path: Path) -> None:
        # a central directory larger than the initial suffix read is fetched
        # in further range requests
        path = tmp_path / "many.zip"
        with zipfile.ZipFile(path, "w") as zf:
            for i in range(3000):
                zf.writestr(f"entries/padding-name-to-grow-the-directory-{i:05d}", b"x")
            zf.writestr("zarr.json", b'{"zarr_format": 3, "node_type": "group"}')
        group = zarr.open_group(_pipeline(f"{_file_url(path)}|zip:"), mode="r")
        assert isinstance(group, zarr.Group)

    def test_nested_zip(self, tmp_path: Path, archive: Path) -> None:
        outer = tmp_path / "outer.zip"
        with zipfile.ZipFile(outer, "w") as zf:
            zf.write(archive, "nested/inner.zip")
        group = zarr.open_group(
            _pipeline(f"{_file_url(outer)}|zip:nested/inner.zip|zip:inner-dir|zarr3:subgroup"),
            mode="r",
        )
        assert group.attrs["title"] == "headline"

    def test_nested_zip_compressed_outer(self, tmp_path: Path, archive: Path) -> None:
        outer = tmp_path / "outer.zip"
        with zipfile.ZipFile(outer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(archive, "inner.zip")
        arr = zarr.open_array(
            _pipeline(f"{_file_url(outer)}|zip:inner.zip|zip:inner-dir/subgroup/x"), mode="r"
        )
        np.testing.assert_array_equal(arr[:], np.arange(6))

    async def test_memory_root(self, archive: Path, tmp_path: Path) -> None:
        # `memory:` roots address one process-wide in-memory store
        key = f"zip-adapter-tests/{tmp_path.name}/a.zip"
        holder = ManagedMemoryStore(name="")
        await holder.set(key, cpu.Buffer.from_bytes(archive.read_bytes()))
        try:
            group = await zarr.api.asynchronous.open_group(
                _pipeline(f"memory:{key}|zip:inner-dir|zarr3:subgroup"), mode="r"
            )
            assert group.attrs["title"] == "headline"
            arr = await group.getitem("x")
            assert isinstance(arr, zarr.AsyncArray)
            np.testing.assert_array_equal(await arr.getitem(slice(None)), np.arange(6))
        finally:
            await holder.delete(key)

    @requires_async_fsspec_wrapper
    def test_fsspec_local_root(self, archive: Path) -> None:
        # fsspec's `local://` protocol: an FsspecStore root, read by range requests
        group = zarr.open_group(
            _pipeline(f"local://{archive.as_posix()}|zip:inner-dir|zarr3:subgroup"), mode="r"
        )
        assert group.attrs["title"] == "headline"

    @requires_async_fsspec_wrapper
    def test_fsspec_remote_like_root(self, archive: Path) -> None:
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
        group = zarr.open_group(
            _pipeline("zarr-test-remote://bucket/my-archive.zip|zip:inner-dir/subgroup|zarr3:"),
            mode="r",
        )
        assert group.attrs["title"] == "headline"
        assert not isinstance(group.store, FsspecStore)  # the reader wraps the fsspec store


class TestModes:
    @pytest.mark.parametrize("mode", ["a", "w", "w-", "r+"])
    def test_non_read_modes_rejected(self, archive: Path, mode: AccessModeLiteral) -> None:
        with pytest.raises(URLPipelineError, match=r"'zip:' URL pipelines are read-only"):
            zarr.open_group(_pipeline(f"{_file_url(archive)}|zip:inner-dir"), mode=mode)
        # the archive is untouched
        with zipfile.ZipFile(archive) as zf:
            assert "inner-dir/subgroup/zarr.json" in zf.namelist()

    def test_default_mode_rejected(self, archive: Path) -> None:
        # zarr.open_group defaults to mode "a"
        with pytest.raises(URLPipelineError, match="mode='a'"):
            zarr.open_group(_pipeline(f"{_file_url(archive)}|zip:inner-dir"))

    def test_store_is_read_only(self, archive: Path) -> None:
        group = zarr.open_group(_pipeline(f"{_file_url(archive)}|zip:inner-dir"), mode="r")
        assert group.store.read_only
        with pytest.raises(ValueError, match="read-only"):
            group.create_group("new")


class TestErrors:
    @pytest.mark.parametrize("url", ["zip:/tmp/a.zip", "zip:", "zip:inner|zarr3:"])
    def test_root_position_rejected(self, url: str) -> None:
        with pytest.raises(URLPipelineError, match="cannot be the root of a URL pipeline"):
            zarr.open_group(_pipeline(url), mode="r")

    def test_missing_archive(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.zip"
        with pytest.raises(URLPipelineError, match="was not found"):
            zarr.open_group(_pipeline(f"{_file_url(missing)}|zip:"), mode="r")

    def test_missing_nested_archive(self, archive: Path) -> None:
        with pytest.raises(URLPipelineError, match="was not found"):
            zarr.open_group(_pipeline(f"{_file_url(archive)}|zip:nope.zip|zip:"), mode="r")

    def test_not_a_zip_file(self, tmp_path: Path) -> None:
        path = tmp_path / "not.zip"
        path.write_bytes(b"definitely not a zip archive")
        with pytest.raises(URLPipelineError, match="could not open the ZIP archive"):
            zarr.open_group(_pipeline(f"{_file_url(path)}|zip:"), mode="r")

    def test_directory_root(self, tmp_path: Path) -> None:
        with pytest.raises(URLPipelineError, match="not found or is not a file"):
            zarr.open_group(_pipeline(f"{_file_url(tmp_path)}|zip:"), mode="r")

    def test_double_leading_slash_is_invalid(self, archive: Path) -> None:
        with pytest.raises(URLPipelineError, match="more than one leading '/'"):
            zarr.open_group(_pipeline(f"{_file_url(archive)}|zip://inner-dir"), mode="r")

    @pytest.mark.parametrize("body", ["..", "a/../b", "./a"])
    def test_invalid_path(self, archive: Path, body: str) -> None:
        with pytest.raises(URLPipelineError, match="invalid path"):
            zarr.open_group(_pipeline(f"{_file_url(archive)}|zip:{body}"), mode="r")

    def test_query_is_rejected(self, archive: Path) -> None:
        with pytest.raises(URLPipelineError, match="do not accept a query"):
            zarr.open_group(_pipeline(f"{_file_url(archive)}|zip:a?x=1"), mode="r")

    def test_query_is_redacted(self, tmp_path: Path) -> None:
        register_url_adapter("vendor.spy", _CloseSpyAdapter)
        path = tmp_path / "not.zip"
        path.write_bytes(b"not a zip")
        with pytest.raises(URLPipelineError) as excinfo:
            zarr.open_group(
                _pipeline(f"{_file_url(path)}|vendor.spy:x?token=SECRET|zip:"), mode="r"
            )
        assert "SECRET" not in str(excinfo.value)

    def test_unsupported_compression(self, tmp_path: Path) -> None:
        path = tmp_path / "lzma.zip"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_LZMA) as zf:
            zf.writestr("zarr.json", b"{}")
        with pytest.raises(NotImplementedError, match="compression method"):
            zarr.open_group(_pipeline(f"{_file_url(path)}|zip:"), mode="r")

    def test_preceding_store_closed_on_error(self, tmp_path: Path) -> None:
        register_url_adapter("vendor.spy", _CloseSpyAdapter)
        _CloseSpyStore.closed.clear()
        path = tmp_path / "not.zip"
        path.write_bytes(b"definitely not a zip archive")
        with pytest.raises(URLPipelineError):
            zarr.open_group(_pipeline(f"{_file_url(path)}|vendor.spy:|zip:"), mode="r")
        assert len(_CloseSpyStore.closed) == 1

    def test_close_closes_preceding_store(self, archive: Path) -> None:
        register_url_adapter("vendor.spy", _CloseSpyAdapter)
        _CloseSpyStore.closed.clear()
        group = zarr.open_group(
            _pipeline(f"{_file_url(archive)}|vendor.spy:|zip:inner-dir"), mode="r"
        )
        assert _CloseSpyStore.closed == []
        group.store.close()
        assert len(_CloseSpyStore.closed) == 1


class TestReaderStore:
    """The store the adapter returns, exercised through the public `Store` API."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> Store:
        path = tmp_path / "s.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("zarr.json", b'{"zarr_format": 3, "node_type": "group"}')
            zf.writestr("a/b", b"0123456789")
            zf.writestr("a/c/d", b"x")
            zf.writestr("e", b"y")
        with zipfile.ZipFile(path, "a", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("z", b"0123456789" * 10)
        return zarr.open_group(_pipeline(f"{_file_url(path)}|zip:"), mode="r").store

    async def test_get_and_ranges(self, store: Store) -> None:
        proto = default_buffer_prototype()
        for key in ("a/b", "z"):  # a stored and a deflated member
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

    async def test_listing(self, store: Store) -> None:
        keys = ["a/b", "a/c/d", "e", "z", "zarr.json"]
        assert sorted([k async for k in store.list()]) == keys
        assert sorted([k async for k in store.list_prefix("a/")]) == ["a/b", "a/c/d"]
        assert sorted([k async for k in store.list_dir("")]) == ["a", "e", "z", "zarr.json"]
        assert sorted([k async for k in store.list_dir("a")]) == ["b", "c"]
        assert await store.exists("e")
        assert not await store.exists("a")
        assert await store.getsize("z") == 100
        with pytest.raises(FileNotFoundError):
            await store.getsize("missing")

    async def test_writes_rejected(self, store: Store) -> None:
        assert store.read_only
        assert not store.supports_writes
        with pytest.raises(ValueError, match="read-only"):
            await store.set("k", cpu.Buffer.from_bytes(b""))
        with pytest.raises(ValueError, match="read-only"):
            await store.delete("e")

    def test_eq_and_str(self, store: Store) -> None:
        assert store == store  # noqa: PLR0124
        assert store != object()
        assert str(store).endswith("|zip:")
        assert "|zip:" in repr(store)

    async def test_bad_crc(self, tmp_path: Path) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("k", b"payload")
            zf.writestr("zarr.json", b'{"zarr_format": 3, "node_type": "group"}')
        raw = bytearray(buf.getvalue())
        raw[raw.index(b"payload")] ^= 0xFF
        path = tmp_path / "bad.zip"
        path.write_bytes(bytes(raw))
        store = zarr.open_group(_pipeline(f"{_file_url(path)}|zip:"), mode="r").store
        with pytest.raises(OSError, match="bad CRC-32"):
            await store.get("k", default_buffer_prototype())
