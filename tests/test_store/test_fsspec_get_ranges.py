# tests/test_store/test_fsspec_get_ranges.py
"""Lightweight integration tests for FsspecStore.get_ranges using MemoryFileSystem.

These don't need moto/s3 — they exercise the new method against an in-process
fsspec MemoryFileSystem wrapped in the async wrapper.
"""

from __future__ import annotations

import pytest
from packaging.version import parse as parse_version

from zarr.abc.store import RangeByteRequest
from zarr.core._coalesce import DEFAULT_COALESCE_OPTIONS, CoalesceOptions
from zarr.core.buffer import Buffer, default_buffer_prototype
from zarr.storage import FsspecStore
from zarr.storage._fsspec import _make_async

fsspec = pytest.importorskip("fsspec")

# AsyncFileSystemWrapper (needed to wrap a sync MemoryFileSystem) landed in fsspec 2024.12.0.
# Older versions are pinned by the min-deps CI job, so skip the whole file there.
pytestmark = pytest.mark.skipif(
    parse_version(fsspec.__version__) < parse_version("2024.12.0"),
    reason="No AsyncFileSystemWrapper",
)


@pytest.fixture
def memory_store() -> FsspecStore:
    """An FsspecStore backed by fsspec MemoryFileSystem (wrapped async)."""
    from fsspec.implementations.memory import MemoryFileSystem

    # Each test gets a clean filesystem; MemoryFileSystem is a singleton per target_options,
    # so clear state explicitly.
    fs: MemoryFileSystem = MemoryFileSystem()
    fs.store.clear()
    fs.pseudo_dirs.clear()
    async_fs = _make_async(fs)
    return FsspecStore(fs=async_fs, path="/root")


async def _write(store: FsspecStore, key: str, data: bytes) -> None:
    buf = default_buffer_prototype().buffer.from_bytes(data)
    await store.set(key, buf)


async def test_get_ranges_happy_path(memory_store: FsspecStore) -> None:
    blob = bytes(i % 256 for i in range(1024))
    await _write(memory_store, "blob", blob)
    proto = default_buffer_prototype()

    ranges = [
        RangeByteRequest(0, 10),
        RangeByteRequest(100, 110),
        RangeByteRequest(500, 520),
    ]
    groups: list[list[tuple[int, Buffer | None]]] = [
        list(group) async for group in memory_store.get_ranges("blob", ranges, prototype=proto)
    ]

    flat: dict[int, bytes] = {}
    for group in groups:
        for idx, buf in group:
            assert buf is not None
            flat[idx] = buf.to_bytes()

    assert flat[0] == blob[0:10]
    assert flat[1] == blob[100:110]
    assert flat[2] == blob[500:520]


async def test_get_ranges_missing_key_yields_nothing(memory_store: FsspecStore) -> None:
    proto = default_buffer_prototype()
    groups: list[list[tuple[int, Buffer | None]]] = [
        list(group)
        async for group in memory_store.get_ranges(
            "does-not-exist", [RangeByteRequest(0, 10)], prototype=proto
        )
    ]
    assert groups == []


async def test_default_coalesce_options_on_store_without_arg() -> None:
    from fsspec.implementations.memory import MemoryFileSystem

    fs = MemoryFileSystem()
    fs.store.clear()
    store = FsspecStore(fs=_make_async(fs), path="/x")
    assert store.coalesce_options == DEFAULT_COALESCE_OPTIONS


async def test_coalesce_options_wired_through() -> None:
    from fsspec.implementations.memory import MemoryFileSystem

    fs = MemoryFileSystem()
    fs.store.clear()
    custom: CoalesceOptions = {
        "max_gap_bytes": 0,
        "max_coalesced_bytes": 1 << 20,
        "max_concurrency": 2,
    }
    store = FsspecStore(fs=_make_async(fs), path="/x", coalesce_options=custom)
    assert store.coalesce_options == custom


async def test_get_ranges_mixed_range_types(memory_store: FsspecStore) -> None:
    """Covers RangeByteRequest, OffsetByteRequest, SuffixByteRequest, and None in one call."""
    from zarr.abc.store import ByteRequest, OffsetByteRequest, SuffixByteRequest

    blob = bytes(i % 256 for i in range(512))
    await _write(memory_store, "mixed", blob)
    proto = default_buffer_prototype()

    ranges: list[ByteRequest | None] = [
        RangeByteRequest(0, 10),
        OffsetByteRequest(500),
        SuffixByteRequest(12),
        None,
    ]
    flat: dict[int, bytes] = {}
    async for group in memory_store.get_ranges("mixed", ranges, prototype=proto):
        for idx, buf in group:
            assert buf is not None
            flat[idx] = buf.to_bytes()

    assert flat[0] == blob[0:10]
    assert flat[1] == blob[500:]
    assert flat[2] == blob[-12:]
    assert flat[3] == blob
