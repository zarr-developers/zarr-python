# tests/test_store/test_fsspec_get_ranges.py
"""Lightweight integration tests for FsspecStore.get_ranges using MemoryFileSystem.

These don't need moto/s3 — they exercise the new method against an in-process
fsspec MemoryFileSystem wrapped in the async wrapper.
"""

from __future__ import annotations

import pytest
from packaging.version import parse as parse_version

from zarr.abc.store import RangeByteRequest
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


async def test_get_ranges_missing_key_raises(memory_store: FsspecStore) -> None:
    """A request against a missing key raises BaseExceptionGroup containing FileNotFoundError."""
    proto = default_buffer_prototype()
    agen = memory_store.get_ranges("does-not-exist", [RangeByteRequest(0, 10)], prototype=proto)
    with pytest.RaisesGroup(pytest.RaisesExc(FileNotFoundError)):
        await anext(agen)


async def test_get_ranges_forwards_coalescing_kwargs(memory_store: FsspecStore) -> None:
    """`max_gap_bytes=-1` forces no merging; we should see three groups for three ranges."""
    blob = bytes(i % 256 for i in range(1024))
    await _write(memory_store, "blob", blob)
    proto = default_buffer_prototype()

    ranges = [
        RangeByteRequest(0, 10),
        RangeByteRequest(11, 20),  # adjacent: would merge under defaults
        RangeByteRequest(21, 30),
    ]
    groups: list[list[tuple[int, Buffer | None]]] = [
        list(group)
        async for group in memory_store.get_ranges(
            "blob", ranges, prototype=proto, max_gap_bytes=-1
        )
    ]
    # With merging disabled, every range becomes its own one-tuple group.
    assert sorted(len(g) for g in groups) == [1, 1, 1]


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
