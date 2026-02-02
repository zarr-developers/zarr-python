from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from zarr.codecs.sharding import (
    MAX_UINT_64,
    ShardingCodec,
    _ChunkCoordsByteSlice,
    _ShardIndex,
    _ShardReader,
)
from zarr.core.buffer import default_buffer_prototype
from zarr.core.buffer.cpu import Buffer
from zarr.core.common import concurrent_map

if TYPE_CHECKING:
    from zarr.abc.store import ByteRequest
    from zarr.core.buffer import BufferPrototype


# ============================================================================
# _ShardIndex tests
# ============================================================================


def test_shard_index_create_empty() -> None:
    """Test that create_empty creates an index filled with MAX_UINT_64."""
    chunks_per_shard = (2, 3)
    index = _ShardIndex.create_empty(chunks_per_shard)

    assert index.chunks_per_shard == chunks_per_shard
    assert index.offsets_and_lengths.shape == (2, 3, 2)
    assert index.offsets_and_lengths.dtype == np.dtype("<u8")
    assert np.all(index.offsets_and_lengths == MAX_UINT_64)


def test_shard_index_create_empty_1d() -> None:
    """Test create_empty with 1D chunks_per_shard."""
    chunks_per_shard = (4,)
    index = _ShardIndex.create_empty(chunks_per_shard)

    assert index.chunks_per_shard == chunks_per_shard
    assert index.offsets_and_lengths.shape == (4, 2)


def test_shard_index_is_all_empty_true() -> None:
    """Test is_all_empty returns True for a freshly created empty index."""
    index = _ShardIndex.create_empty((2, 2))
    assert index.is_all_empty() is True


def test_shard_index_is_all_empty_false() -> None:
    """Test is_all_empty returns False when at least one chunk is set."""
    index = _ShardIndex.create_empty((2, 2))
    index.set_chunk_slice((0, 0), slice(0, 100))
    assert index.is_all_empty() is False


def test_shard_index_get_chunk_slice_empty() -> None:
    """Test get_chunk_slice returns None for empty chunks."""
    index = _ShardIndex.create_empty((2, 2))
    assert index.get_chunk_slice((0, 0)) is None
    assert index.get_chunk_slice((1, 1)) is None


def test_shard_index_get_chunk_slice_set() -> None:
    """Test get_chunk_slice returns correct (start, end) tuple after setting."""
    index = _ShardIndex.create_empty((2, 2))
    index.set_chunk_slice((0, 1), slice(100, 200))

    result = index.get_chunk_slice((0, 1))
    assert result == (100, 200)


def test_shard_index_set_chunk_slice() -> None:
    """Test set_chunk_slice correctly sets offset and length."""
    index = _ShardIndex.create_empty((3, 3))

    # Set a chunk slice
    index.set_chunk_slice((1, 2), slice(50, 150))

    # Verify the underlying array
    assert index.offsets_and_lengths[1, 2, 0] == 50  # offset
    assert index.offsets_and_lengths[1, 2, 1] == 100  # length (150 - 50)


def test_shard_index_set_chunk_slice_none() -> None:
    """Test set_chunk_slice with None marks chunk as empty."""
    index = _ShardIndex.create_empty((2, 2))

    # First set a value
    index.set_chunk_slice((0, 0), slice(0, 100))
    assert index.get_chunk_slice((0, 0)) == (0, 100)

    # Then clear it
    index.set_chunk_slice((0, 0), None)
    assert index.get_chunk_slice((0, 0)) is None
    assert index.offsets_and_lengths[0, 0, 0] == MAX_UINT_64
    assert index.offsets_and_lengths[0, 0, 1] == MAX_UINT_64


def test_shard_index_get_full_chunk_map() -> None:
    """Test get_full_chunk_map returns correct boolean array."""
    index = _ShardIndex.create_empty((2, 3))

    # Set some chunks
    index.set_chunk_slice((0, 0), slice(0, 10))
    index.set_chunk_slice((1, 2), slice(10, 20))

    chunk_map = index.get_full_chunk_map()

    assert chunk_map.shape == (2, 3)
    assert chunk_map.dtype == np.bool_
    assert chunk_map[0, 0] is np.True_
    assert chunk_map[0, 1] is np.False_
    assert chunk_map[0, 2] is np.False_
    assert chunk_map[1, 0] is np.False_
    assert chunk_map[1, 1] is np.False_
    assert chunk_map[1, 2] is np.True_


def test_shard_index_localize_chunk() -> None:
    """Test _localize_chunk maps global coords to local shard coords via modulo."""
    index = _ShardIndex.create_empty((2, 3))

    # Within bounds - should return same coords
    assert index._localize_chunk((0, 0)) == (0, 0)
    assert index._localize_chunk((1, 2)) == (1, 2)

    # Out of bounds - should wrap via modulo
    assert index._localize_chunk((2, 0)) == (0, 0)  # 2 % 2 = 0
    assert index._localize_chunk((3, 5)) == (1, 2)  # 3 % 2 = 1, 5 % 3 = 2
    assert index._localize_chunk((4, 6)) == (0, 0)  # 4 % 2 = 0, 6 % 3 = 0


def test_shard_index_is_dense_true() -> None:
    """Test is_dense returns True when chunks are contiguously packed."""
    index = _ShardIndex.create_empty((2,))
    chunk_byte_length = 100

    # Set chunks contiguously: [0-100), [100-200)
    index.set_chunk_slice((0,), slice(0, 100))
    index.set_chunk_slice((1,), slice(100, 200))

    assert index.is_dense(chunk_byte_length) is True


def test_shard_index_is_dense_false_duplicate_offsets() -> None:
    """Test is_dense returns False when chunks have duplicate offsets."""
    index = _ShardIndex.create_empty((2,))
    chunk_byte_length = 100

    # Set both chunks to same offset (duplicate)
    index.set_chunk_slice((0,), slice(0, 100))
    index.set_chunk_slice((1,), slice(0, 100))

    assert index.is_dense(chunk_byte_length) is False


def test_shard_index_is_dense_false_wrong_alignment() -> None:
    """Test is_dense returns False when chunks are not aligned to chunk_byte_length."""
    index = _ShardIndex.create_empty((2,))
    chunk_byte_length = 100

    # Set chunks not aligned: [0-100), [150-250)
    index.set_chunk_slice((0,), slice(0, 100))
    index.set_chunk_slice((1,), slice(150, 250))

    assert index.is_dense(chunk_byte_length) is False


def test_shard_index_is_dense_with_empty_chunks() -> None:
    """Test is_dense handles empty chunks correctly."""
    index = _ShardIndex.create_empty((3,))
    chunk_byte_length = 100

    # Only set first and third chunk, skip middle
    index.set_chunk_slice((0,), slice(0, 100))
    # (1,) is empty
    index.set_chunk_slice((2,), slice(100, 200))

    # Should still be dense since only non-empty chunks are considered
    assert index.is_dense(chunk_byte_length) is True


# ============================================================================
# _coalesce_chunks tests
# ============================================================================


def test_coalesce_chunks_empty_list() -> None:
    """Test _coalesce_chunks returns empty list for empty input."""
    codec = ShardingCodec(chunk_shape=(8,))
    result = codec._coalesce_chunks([], max_gap_bytes=100, coalesce_max_bytes=1000)
    assert result == []


def test_coalesce_chunks_single_chunk() -> None:
    """Test _coalesce_chunks returns single group for single chunk."""
    codec = ShardingCodec(chunk_shape=(8,))
    chunk = _ChunkCoordsByteSlice(chunk_coords=(0,), byte_slice=slice(0, 100))

    result = codec._coalesce_chunks([chunk], max_gap_bytes=100, coalesce_max_bytes=1000)

    assert len(result) == 1
    assert len(result[0]) == 1
    assert result[0][0] == chunk


def test_coalesce_chunks_adjacent_small_gap() -> None:
    """Test adjacent chunks with small gap are coalesced."""
    codec = ShardingCodec(chunk_shape=(8,))
    chunk0 = _ChunkCoordsByteSlice(chunk_coords=(0,), byte_slice=slice(0, 100))
    chunk1 = _ChunkCoordsByteSlice(chunk_coords=(1,), byte_slice=slice(110, 210))  # 10 byte gap

    result = codec._coalesce_chunks([chunk0, chunk1], max_gap_bytes=20, coalesce_max_bytes=1000)

    assert len(result) == 1
    assert len(result[0]) == 2
    assert result[0][0] == chunk0
    assert result[0][1] == chunk1


def test_coalesce_chunks_distant_large_gap() -> None:
    """Test chunks with large gap are not coalesced."""
    codec = ShardingCodec(chunk_shape=(8,))
    chunk0 = _ChunkCoordsByteSlice(chunk_coords=(0,), byte_slice=slice(0, 100))
    chunk1 = _ChunkCoordsByteSlice(chunk_coords=(1,), byte_slice=slice(500, 600))  # 400 byte gap

    result = codec._coalesce_chunks([chunk0, chunk1], max_gap_bytes=100, coalesce_max_bytes=1000)

    assert len(result) == 2
    assert result[0] == [chunk0]
    assert result[1] == [chunk1]


def test_coalesce_chunks_disabled_negative_gap() -> None:
    """Test coalescing is disabled when max_gap_bytes is negative (like -1)."""
    codec = ShardingCodec(chunk_shape=(8,))
    chunk0 = _ChunkCoordsByteSlice(chunk_coords=(0,), byte_slice=slice(0, 100))
    chunk1 = _ChunkCoordsByteSlice(chunk_coords=(1,), byte_slice=slice(100, 200))  # Adjacent!

    result = codec._coalesce_chunks([chunk0, chunk1], max_gap_bytes=-1, coalesce_max_bytes=1000)

    # Even adjacent chunks should not be coalesced
    assert len(result) == 2


def test_coalesce_chunks_exceeds_max_bytes() -> None:
    """Test chunks are split when total size exceeds coalesce_max_bytes."""
    codec = ShardingCodec(chunk_shape=(8,))
    chunk0 = _ChunkCoordsByteSlice(chunk_coords=(0,), byte_slice=slice(0, 100))
    chunk1 = _ChunkCoordsByteSlice(chunk_coords=(1,), byte_slice=slice(100, 200))
    chunk2 = _ChunkCoordsByteSlice(chunk_coords=(2,), byte_slice=slice(200, 300))

    # Total would be 300 bytes, but max is 250
    result = codec._coalesce_chunks(
        [chunk0, chunk1, chunk2], max_gap_bytes=100, coalesce_max_bytes=250
    )

    # First two chunks (200 bytes) should be coalesced, third separate
    assert len(result) == 2
    assert len(result[0]) == 2
    assert result[0][0] == chunk0
    assert result[0][1] == chunk1
    assert result[1] == [chunk2]


def test_coalesce_chunks_unsorted_input() -> None:
    """Test chunks are sorted by byte_slice.start before coalescing."""
    codec = ShardingCodec(chunk_shape=(8,))
    chunk0 = _ChunkCoordsByteSlice(chunk_coords=(0,), byte_slice=slice(200, 300))
    chunk1 = _ChunkCoordsByteSlice(chunk_coords=(1,), byte_slice=slice(0, 100))
    chunk2 = _ChunkCoordsByteSlice(chunk_coords=(2,), byte_slice=slice(100, 200))

    # Input is out of order
    result = codec._coalesce_chunks(
        [chunk0, chunk1, chunk2], max_gap_bytes=100, coalesce_max_bytes=1000
    )

    # All should be coalesced and in sorted order
    assert len(result) == 1
    assert len(result[0]) == 3
    assert result[0][0] == chunk1  # slice(0, 100)
    assert result[0][1] == chunk2  # slice(100, 200)
    assert result[0][2] == chunk0  # slice(200, 300)


def test_coalesce_chunks_mixed_coalescing() -> None:
    """Test mixed scenario with some chunks coalesced and some separate."""
    codec = ShardingCodec(chunk_shape=(8,))
    # Group 1: chunks at 0-100, 100-200 (adjacent)
    chunk0 = _ChunkCoordsByteSlice(chunk_coords=(0,), byte_slice=slice(0, 100))
    chunk1 = _ChunkCoordsByteSlice(chunk_coords=(1,), byte_slice=slice(100, 200))
    # Gap of 300 bytes
    # Group 2: chunks at 500-600, 600-700 (adjacent)
    chunk2 = _ChunkCoordsByteSlice(chunk_coords=(2,), byte_slice=slice(500, 600))
    chunk3 = _ChunkCoordsByteSlice(chunk_coords=(3,), byte_slice=slice(600, 700))

    result = codec._coalesce_chunks(
        [chunk0, chunk1, chunk2, chunk3], max_gap_bytes=100, coalesce_max_bytes=1000
    )

    assert len(result) == 2
    assert len(result[0]) == 2
    assert result[0][0] == chunk0
    assert result[0][1] == chunk1
    assert len(result[1]) == 2
    assert result[1][0] == chunk2
    assert result[1][1] == chunk3


def test_coalesce_chunks_boundary_gap_equals_max() -> None:
    """Test boundary condition where gap equals max_gap_bytes (should NOT coalesce)."""
    codec = ShardingCodec(chunk_shape=(8,))
    chunk0 = _ChunkCoordsByteSlice(chunk_coords=(0,), byte_slice=slice(0, 100))
    chunk1 = _ChunkCoordsByteSlice(chunk_coords=(1,), byte_slice=slice(150, 250))  # 50 byte gap

    # Gap is exactly max_gap_bytes, condition is `gap < max_gap_bytes` so should NOT coalesce
    result = codec._coalesce_chunks([chunk0, chunk1], max_gap_bytes=50, coalesce_max_bytes=1000)

    assert len(result) == 2


def test_coalesce_chunks_boundary_gap_less_than_max() -> None:
    """Test boundary condition where gap is just under max_gap_bytes (should coalesce)."""
    codec = ShardingCodec(chunk_shape=(8,))
    chunk0 = _ChunkCoordsByteSlice(chunk_coords=(0,), byte_slice=slice(0, 100))
    chunk1 = _ChunkCoordsByteSlice(chunk_coords=(1,), byte_slice=slice(149, 249))  # 49 byte gap

    result = codec._coalesce_chunks([chunk0, chunk1], max_gap_bytes=50, coalesce_max_bytes=1000)

    assert len(result) == 1


# ============================================================================
# _get_group_bytes tests
# ============================================================================


@dataclass
class MockByteGetter:
    """Mock ByteGetter for testing _get_group_bytes."""

    data: bytes
    return_none: bool = False
    call_count: int = 0

    async def get(
        self, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        self.call_count += 1
        if self.return_none:
            return None
        if byte_range is None:
            return Buffer.from_bytes(self.data)
        # For RangeByteRequest, extract start and end
        start = getattr(byte_range, "start", 0)
        end = getattr(byte_range, "end", len(self.data))
        return Buffer.from_bytes(self.data[start:end])


async def test_get_group_bytes_single_chunk() -> None:
    """Test _get_group_bytes extracts single chunk correctly."""
    codec = ShardingCodec(chunk_shape=(8,))
    data = b"0123456789" * 10  # 100 bytes
    byte_getter = MockByteGetter(data=data)

    chunk = _ChunkCoordsByteSlice(chunk_coords=(0,), byte_slice=slice(10, 30))
    group = [chunk]

    result = await codec._get_group_bytes(group, byte_getter, default_buffer_prototype())

    assert result is not None
    assert (0,) in result
    chunk_buf = result[(0,)]
    assert chunk_buf is not None
    assert chunk_buf.as_numpy_array().tobytes() == data[10:30]


async def test_get_group_bytes_multiple_chunks() -> None:
    """Test _get_group_bytes extracts multiple chunks with correct offsets."""
    codec = ShardingCodec(chunk_shape=(8,))
    data = b"0123456789" * 10  # 100 bytes
    byte_getter = MockByteGetter(data=data)

    # Two chunks: [10, 30) and [30, 50)
    chunk0 = _ChunkCoordsByteSlice(chunk_coords=(0,), byte_slice=slice(10, 30))
    chunk1 = _ChunkCoordsByteSlice(chunk_coords=(1,), byte_slice=slice(30, 50))
    group = [chunk0, chunk1]

    result = await codec._get_group_bytes(group, byte_getter, default_buffer_prototype())

    assert result is not None
    assert len(result) == 2
    chunk0_buf = result[(0,)]
    chunk1_buf = result[(1,)]
    assert chunk0_buf is not None
    assert chunk1_buf is not None
    assert chunk0_buf.as_numpy_array().tobytes() == data[10:30]
    assert chunk1_buf.as_numpy_array().tobytes() == data[30:50]


async def test_get_group_bytes_with_gap() -> None:
    """Test _get_group_bytes handles chunks with gaps correctly."""
    codec = ShardingCodec(chunk_shape=(8,))
    data = b"0123456789" * 10  # 100 bytes
    byte_getter = MockByteGetter(data=data)

    # Two chunks with a gap: [10, 20) and [40, 60)
    chunk0 = _ChunkCoordsByteSlice(chunk_coords=(0,), byte_slice=slice(10, 20))
    chunk1 = _ChunkCoordsByteSlice(chunk_coords=(1,), byte_slice=slice(40, 60))
    group = [chunk0, chunk1]

    result = await codec._get_group_bytes(group, byte_getter, default_buffer_prototype())

    assert result is not None
    assert len(result) == 2
    # The byte_getter.get is called with range [10, 60), then sliced
    chunk0_buf = result[(0,)]
    chunk1_buf = result[(1,)]
    assert chunk0_buf is not None
    assert chunk1_buf is not None
    assert chunk0_buf.as_numpy_array().tobytes() == data[10:20]
    assert chunk1_buf.as_numpy_array().tobytes() == data[40:60]


async def test_get_group_bytes_returns_none_on_failed_read() -> None:
    """Test _get_group_bytes returns None when ByteGetter.get returns None."""
    codec = ShardingCodec(chunk_shape=(8,))
    byte_getter = MockByteGetter(data=b"", return_none=True)

    chunk = _ChunkCoordsByteSlice(chunk_coords=(0,), byte_slice=slice(0, 100))
    group = [chunk]

    result = await codec._get_group_bytes(group, byte_getter, default_buffer_prototype())

    assert result is None


# ============================================================================
# Coalescing and concurrent_map optimization tests
# ============================================================================


async def test_coalescing_reduces_byte_getter_calls_adjacent_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that adjacent chunks result in a single byte getter call due to coalescing."""
    codec = ShardingCodec(chunk_shape=(8,))
    chunks_per_shard = (4,)

    index = _ShardIndex.create_empty(chunks_per_shard)
    index.set_chunk_slice((0,), slice(0, 100))
    index.set_chunk_slice((1,), slice(100, 200))  # Adjacent, will coalesce

    async def mock_load_index(
        self: ShardingCodec, byte_getter: MockByteGetter, cps: tuple[int, ...]
    ) -> _ShardIndex:
        return index

    monkeypatch.setattr(ShardingCodec, "_load_shard_index_maybe", mock_load_index)

    chunk_data = b"A" * 100 + b"B" * 100
    byte_getter = MockByteGetter(data=chunk_data)

    all_chunk_coords: set[tuple[int, ...]] = {(0,), (1,)}

    await codec._load_partial_shard_maybe(
        byte_getter=byte_getter,
        prototype=default_buffer_prototype(),
        chunks_per_shard=chunks_per_shard,
        all_chunk_coords=all_chunk_coords,
        max_gap_bytes=1000,
        coalesce_max_bytes=10000,
        async_concurrency=10,
    )

    # Adjacent chunks should result in only 1 call (coalesced)
    assert byte_getter.call_count == 1


async def test_coalescing_no_reduction_distant_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that distant chunks result in multiple byte getter calls (not coalesced)."""
    codec = ShardingCodec(chunk_shape=(8,))
    chunks_per_shard = (4,)

    index = _ShardIndex.create_empty(chunks_per_shard)
    index.set_chunk_slice((0,), slice(0, 100))
    index.set_chunk_slice((2,), slice(1000, 1100))  # Large gap, separate group

    async def mock_load_index(
        self: ShardingCodec, byte_getter: MockByteGetter, cps: tuple[int, ...]
    ) -> _ShardIndex:
        return index

    monkeypatch.setattr(ShardingCodec, "_load_shard_index_maybe", mock_load_index)

    chunk_data = b"C" * 100 + b"\x00" * 900 + b"D" * 100
    byte_getter = MockByteGetter(data=chunk_data)

    all_chunk_coords: set[tuple[int, ...]] = {(0,), (2,)}

    await codec._load_partial_shard_maybe(
        byte_getter=byte_getter,
        prototype=default_buffer_prototype(),
        chunks_per_shard=chunks_per_shard,
        all_chunk_coords=all_chunk_coords,
        max_gap_bytes=100,  # Too small to bridge the 900-byte gap
        coalesce_max_bytes=10000,
        async_concurrency=10,
    )

    # Distant chunks should result in 2 calls (not coalesced)
    assert byte_getter.call_count == 2


async def test_concurrent_map_not_called_single_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that concurrent_map is NOT called when there's only one group."""
    codec = ShardingCodec(chunk_shape=(8,))
    chunks_per_shard = (4,)

    index = _ShardIndex.create_empty(chunks_per_shard)
    index.set_chunk_slice((0,), slice(0, 100))
    index.set_chunk_slice((1,), slice(100, 200))  # Adjacent, will coalesce

    async def mock_load_index(
        self: ShardingCodec, byte_getter: MockByteGetter, cps: tuple[int, ...]
    ) -> _ShardIndex:
        return index

    monkeypatch.setattr(ShardingCodec, "_load_shard_index_maybe", mock_load_index)

    # Track if concurrent_map was called
    concurrent_map_called = False
    original_concurrent_map = concurrent_map

    async def tracked_concurrent_map(*args: Any, **kwargs: Any) -> Any:
        nonlocal concurrent_map_called
        concurrent_map_called = True
        return await original_concurrent_map(*args, **kwargs)

    monkeypatch.setattr("zarr.codecs.sharding.concurrent_map", tracked_concurrent_map)

    chunk_data = b"A" * 100 + b"B" * 100
    byte_getter = MockByteGetter(data=chunk_data)

    all_chunk_coords: set[tuple[int, ...]] = {(0,), (1,)}

    result = await codec._load_partial_shard_maybe(
        byte_getter=byte_getter,
        prototype=default_buffer_prototype(),
        chunks_per_shard=chunks_per_shard,
        all_chunk_coords=all_chunk_coords,
        max_gap_bytes=1000,
        coalesce_max_bytes=10000,
        async_concurrency=10,
    )

    assert result is not None
    assert len(result) == 2
    # Single group after coalescing should not call concurrent_map
    assert not concurrent_map_called


async def test_concurrent_map_called_multiple_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that concurrent_map IS called when there are multiple groups."""
    codec = ShardingCodec(chunk_shape=(8,))
    chunks_per_shard = (4,)

    index = _ShardIndex.create_empty(chunks_per_shard)
    index.set_chunk_slice((0,), slice(0, 100))
    index.set_chunk_slice((2,), slice(1000, 1100))  # Large gap, separate group

    async def mock_load_index(
        self: ShardingCodec, byte_getter: MockByteGetter, cps: tuple[int, ...]
    ) -> _ShardIndex:
        return index

    monkeypatch.setattr(ShardingCodec, "_load_shard_index_maybe", mock_load_index)

    # Track if concurrent_map was called
    concurrent_map_called = False
    original_concurrent_map = concurrent_map

    async def tracked_concurrent_map(*args: Any, **kwargs: Any) -> Any:
        nonlocal concurrent_map_called
        concurrent_map_called = True
        return await original_concurrent_map(*args, **kwargs)

    monkeypatch.setattr("zarr.codecs.sharding.concurrent_map", tracked_concurrent_map)

    chunk_data = b"C" * 100 + b"\x00" * 900 + b"D" * 100
    byte_getter = MockByteGetter(data=chunk_data)

    all_chunk_coords: set[tuple[int, ...]] = {(0,), (2,)}

    result = await codec._load_partial_shard_maybe(
        byte_getter=byte_getter,
        prototype=default_buffer_prototype(),
        chunks_per_shard=chunks_per_shard,
        all_chunk_coords=all_chunk_coords,
        max_gap_bytes=100,  # Too small to bridge the 900-byte gap
        coalesce_max_bytes=10000,
        async_concurrency=10,
    )

    # Verify result is correct
    assert result is not None
    assert len(result) == 2
    # Multiple groups after coalescing should call concurrent_map
    assert concurrent_map_called


# ============================================================================
# _load_partial_shard_maybe tests
# ============================================================================


@dataclass
class MockByteGetterWithIndex:
    """Mock ByteGetter that can return a shard index and chunk data."""

    index_data: bytes | None
    chunk_data: bytes | None
    call_count: int = 0

    async def get(
        self, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        self.call_count += 1
        # First call is typically for the index
        if self.call_count == 1:
            if self.index_data is None:
                return None
            return Buffer.from_bytes(self.index_data)
        # Subsequent calls are for chunk data
        if self.chunk_data is None:
            return None
        if byte_range is None:
            return Buffer.from_bytes(self.chunk_data)
        # For RangeByteRequest, extract start and end
        start = getattr(byte_range, "start", 0)
        end = getattr(byte_range, "end", len(self.chunk_data))
        return Buffer.from_bytes(self.chunk_data[start:end])


async def test_load_partial_shard_maybe_index_load_fails() -> None:
    """Test _load_partial_shard_maybe returns None when index load fails."""
    codec = ShardingCodec(chunk_shape=(8,))
    byte_getter = MockByteGetterWithIndex(index_data=None, chunk_data=None)

    chunks_per_shard = (2,)
    all_chunk_coords: set[tuple[int, ...]] = {(0,)}

    result = await codec._load_partial_shard_maybe(
        byte_getter=byte_getter,
        prototype=default_buffer_prototype(),
        chunks_per_shard=chunks_per_shard,
        all_chunk_coords=all_chunk_coords,
        max_gap_bytes=100,
        coalesce_max_bytes=1000,
        async_concurrency=1,
    )

    assert result is None


async def test_load_partial_shard_maybe_with_empty_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test _load_partial_shard_maybe skips chunks where get_chunk_slice returns None."""
    codec = ShardingCodec(chunk_shape=(8,))
    chunks_per_shard = (4,)

    # Create an index where chunk (1,) is empty (returns None from get_chunk_slice)
    index = _ShardIndex.create_empty(chunks_per_shard)
    index.set_chunk_slice((0,), slice(0, 100))
    # (1,) is intentionally left empty
    index.set_chunk_slice((2,), slice(100, 200))
    index.set_chunk_slice((3,), slice(200, 300))

    # Mock _load_shard_index_maybe on the class to return our custom index
    async def mock_load_index(
        self: ShardingCodec, byte_getter: MockByteGetter, cps: tuple[int, ...]
    ) -> _ShardIndex:
        return index

    monkeypatch.setattr(ShardingCodec, "_load_shard_index_maybe", mock_load_index)

    # Create byte getter with chunk data
    chunk_data = b"x" * 300
    byte_getter = MockByteGetter(data=chunk_data)

    # Request chunks including the empty one
    all_chunk_coords: set[tuple[int, ...]] = {(0,), (1,), (2,)}

    result = await codec._load_partial_shard_maybe(
        byte_getter=byte_getter,
        prototype=default_buffer_prototype(),
        chunks_per_shard=chunks_per_shard,
        all_chunk_coords=all_chunk_coords,
        max_gap_bytes=1000,
        coalesce_max_bytes=10000,
        async_concurrency=1,
    )

    assert result is not None
    # Only chunks (0,) and (2,) should be in result, (1,) is empty and skipped
    assert (0,) in result
    assert (1,) not in result  # Empty chunk should be skipped
    assert (2,) in result


async def test_load_partial_shard_maybe_all_chunks_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test _load_partial_shard_maybe returns empty dict when all requested chunks are empty."""
    codec = ShardingCodec(chunk_shape=(8,))
    chunks_per_shard = (4,)

    # Create an empty index (all chunks empty)
    index = _ShardIndex.create_empty(chunks_per_shard)

    # Mock _load_shard_index_maybe on the class to return our empty index
    async def mock_load_index(
        self: ShardingCodec, byte_getter: MockByteGetter, cps: tuple[int, ...]
    ) -> _ShardIndex:
        return index

    monkeypatch.setattr(ShardingCodec, "_load_shard_index_maybe", mock_load_index)

    byte_getter = MockByteGetter(data=b"")

    # Request some chunks - all will be empty
    all_chunk_coords: set[tuple[int, ...]] = {(0,), (1,), (2,)}

    result = await codec._load_partial_shard_maybe(
        byte_getter=byte_getter,
        prototype=default_buffer_prototype(),
        chunks_per_shard=chunks_per_shard,
        all_chunk_coords=all_chunk_coords,
        max_gap_bytes=1000,
        coalesce_max_bytes=10000,
        async_concurrency=1,
    )

    assert result is not None
    assert result == {}  # All chunks were empty, so result is empty dict


async def test_load_partial_shard_single_group_coalesced_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that single group with coalesced adjacent chunks works correctly."""
    codec = ShardingCodec(chunk_shape=(8,))
    chunks_per_shard = (4,)

    index = _ShardIndex.create_empty(chunks_per_shard)
    index.set_chunk_slice((0,), slice(0, 100))
    index.set_chunk_slice((1,), slice(100, 200))  # Adjacent, will coalesce

    async def mock_load_index(
        self: ShardingCodec, byte_getter: MockByteGetter, cps: tuple[int, ...]
    ) -> _ShardIndex:
        return index

    monkeypatch.setattr(ShardingCodec, "_load_shard_index_maybe", mock_load_index)

    chunk_data = b"A" * 100 + b"B" * 100
    byte_getter = MockByteGetter(data=chunk_data)

    # Request chunks that will form a single group
    all_chunk_coords: set[tuple[int, ...]] = {(0,), (1,)}

    result = await codec._load_partial_shard_maybe(
        byte_getter=byte_getter,
        prototype=default_buffer_prototype(),
        chunks_per_shard=chunks_per_shard,
        all_chunk_coords=all_chunk_coords,
        max_gap_bytes=1000,
        coalesce_max_bytes=10000,
        async_concurrency=10,
    )

    assert result is not None
    assert (0,) in result
    assert (1,) in result
    assert len(result) == 2

    chunk0 = result[(0,)]
    assert chunk0 is not None
    assert chunk0.as_numpy_array().tobytes() == b"A" * 100

    chunk1 = result[(1,)]
    assert chunk1 is not None
    assert chunk1.as_numpy_array().tobytes() == b"B" * 100


async def test_load_partial_shard_multiple_groups_large_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that multiple groups are handled correctly when chunks have large gaps."""
    codec = ShardingCodec(chunk_shape=(8,))
    chunks_per_shard = (4,)

    index = _ShardIndex.create_empty(chunks_per_shard)
    index.set_chunk_slice((0,), slice(0, 100))
    index.set_chunk_slice((2,), slice(1000, 1100))  # Large gap, separate group

    async def mock_load_index(
        self: ShardingCodec, byte_getter: MockByteGetter, cps: tuple[int, ...]
    ) -> _ShardIndex:
        return index

    monkeypatch.setattr(ShardingCodec, "_load_shard_index_maybe", mock_load_index)

    chunk_data = b"C" * 100 + b"\x00" * 900 + b"D" * 100
    byte_getter = MockByteGetter(data=chunk_data)

    # Request chunks that will form multiple groups due to large gap
    all_chunk_coords: set[tuple[int, ...]] = {(0,), (2,)}

    result = await codec._load_partial_shard_maybe(
        byte_getter=byte_getter,
        prototype=default_buffer_prototype(),
        chunks_per_shard=chunks_per_shard,
        all_chunk_coords=all_chunk_coords,
        max_gap_bytes=100,  # Too small to bridge the 900-byte gap
        coalesce_max_bytes=10000,
        async_concurrency=10,
    )

    assert result is not None
    assert (0,) in result
    assert (2,) in result
    assert len(result) == 2

    chunk0 = result[(0,)]
    assert chunk0 is not None
    assert chunk0.as_numpy_array().tobytes() == b"C" * 100

    chunk2 = result[(2,)]
    assert chunk2 is not None
    assert chunk2.as_numpy_array().tobytes() == b"D" * 100


async def test_load_partial_shard_single_chunk_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test single chunk read (most common case for single element access)."""
    codec = ShardingCodec(chunk_shape=(8,))
    chunks_per_shard = (4,)

    # Create an index with just one chunk
    index = _ShardIndex.create_empty(chunks_per_shard)
    index.set_chunk_slice((1,), slice(100, 200))

    async def mock_load_index(
        self: ShardingCodec, byte_getter: MockByteGetter, cps: tuple[int, ...]
    ) -> _ShardIndex:
        return index

    monkeypatch.setattr(ShardingCodec, "_load_shard_index_maybe", mock_load_index)

    chunk_data = b"\x00" * 100 + b"E" * 100
    byte_getter = MockByteGetter(data=chunk_data)

    all_chunk_coords: set[tuple[int, ...]] = {(1,)}

    result = await codec._load_partial_shard_maybe(
        byte_getter=byte_getter,
        prototype=default_buffer_prototype(),
        chunks_per_shard=chunks_per_shard,
        all_chunk_coords=all_chunk_coords,
        max_gap_bytes=100,
        coalesce_max_bytes=1000,
        async_concurrency=10,
    )

    assert result is not None
    assert (1,) in result
    assert len(result) == 1

    chunk1 = result[(1,)]
    assert chunk1 is not None
    assert chunk1.as_numpy_array().tobytes() == b"E" * 100


async def test_load_partial_shard_three_groups_mixed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test handling of three separate groups with mixed coalescing."""
    codec = ShardingCodec(chunk_shape=(8,))
    chunks_per_shard = (8,)

    index = _ShardIndex.create_empty(chunks_per_shard)
    index.set_chunk_slice((0,), slice(0, 100))
    index.set_chunk_slice((1,), slice(100, 200))
    index.set_chunk_slice((4,), slice(1000, 1100))
    index.set_chunk_slice((6,), slice(2000, 2100))
    index.set_chunk_slice((7,), slice(2100, 2200))

    async def mock_load_index(
        self: ShardingCodec, byte_getter: MockByteGetter, cps: tuple[int, ...]
    ) -> _ShardIndex:
        return index

    monkeypatch.setattr(ShardingCodec, "_load_shard_index_maybe", mock_load_index)

    chunk_data = (
        b"F" * 100  # chunk 0: [0:100]
        + b"G" * 100  # chunk 1: [100:200]
        + b"\x00" * 800  # gap
        + b"H" * 100  # chunk 4: [1000:1100]
        + b"\x00" * 900  # gap
        + b"I" * 100  # chunk 6: [2000:2100]
        + b"J" * 100  # chunk 7: [2100:2200]
    )
    byte_getter = MockByteGetter(data=chunk_data)

    all_chunk_coords: set[tuple[int, ...]] = {(0,), (1,), (4,), (6,), (7,)}

    result = await codec._load_partial_shard_maybe(
        byte_getter=byte_getter,
        prototype=default_buffer_prototype(),
        chunks_per_shard=chunks_per_shard,
        all_chunk_coords=all_chunk_coords,
        max_gap_bytes=100,  # Small enough to prevent cross-group coalescing
        coalesce_max_bytes=10000,
        async_concurrency=10,
    )

    assert result is not None
    assert len(result) == 5
    assert (0,) in result
    assert (1,) in result
    assert (4,) in result
    assert (6,) in result
    assert (7,) in result

    chunk0 = result[(0,)]
    assert chunk0 is not None
    assert chunk0.as_numpy_array().tobytes() == b"F" * 100

    chunk1 = result[(1,)]
    assert chunk1 is not None
    assert chunk1.as_numpy_array().tobytes() == b"G" * 100

    chunk4 = result[(4,)]
    assert chunk4 is not None
    assert chunk4.as_numpy_array().tobytes() == b"H" * 100

    chunk6 = result[(6,)]
    assert chunk6 is not None
    assert chunk6.as_numpy_array().tobytes() == b"I" * 100

    chunk7 = result[(7,)]
    assert chunk7 is not None
    assert chunk7.as_numpy_array().tobytes() == b"J" * 100


async def test_load_partial_shard_single_group_getter_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test single group path handles None return from byte_getter.get."""
    codec = ShardingCodec(chunk_shape=(8,))
    chunks_per_shard = (4,)

    index = _ShardIndex.create_empty(chunks_per_shard)
    index.set_chunk_slice((0,), slice(0, 100))

    async def mock_load_index(
        self: ShardingCodec, byte_getter: MockByteGetter, cps: tuple[int, ...]
    ) -> _ShardIndex:
        return index

    monkeypatch.setattr(ShardingCodec, "_load_shard_index_maybe", mock_load_index)

    byte_getter = MockByteGetter(data=b"", return_none=True)

    all_chunk_coords: set[tuple[int, ...]] = {(0,)}

    result = await codec._load_partial_shard_maybe(
        byte_getter=byte_getter,
        prototype=default_buffer_prototype(),
        chunks_per_shard=chunks_per_shard,
        all_chunk_coords=all_chunk_coords,
        max_gap_bytes=100,
        coalesce_max_bytes=1000,
        async_concurrency=10,
    )

    assert result is not None
    assert len(result) == 0


# ============================================================================
# Supporting class tests (_ShardReader, _is_total_shard, _ChunkCoordsByteSlice)
# ============================================================================


def test_chunk_coords_byte_slice() -> None:
    """Test _ChunkCoordsByteSlice dataclass."""
    chunk = _ChunkCoordsByteSlice(chunk_coords=(1, 2, 3), byte_slice=slice(100, 200))

    assert chunk.chunk_coords == (1, 2, 3)
    assert chunk.byte_slice == slice(100, 200)
    assert chunk.byte_slice.start == 100
    assert chunk.byte_slice.stop == 200


def test_shard_reader_create_empty() -> None:
    """Test _ShardReader.create_empty creates reader with empty index."""
    chunks_per_shard = (2, 3)
    reader = _ShardReader.create_empty(chunks_per_shard)

    assert reader.index.is_all_empty()
    assert len(reader.buf) == 0
    assert len(reader) == 6  # 2 * 3


def test_shard_reader_iteration() -> None:
    """Test _ShardReader iteration yields all chunk coordinates."""
    chunks_per_shard = (2, 2)
    reader = _ShardReader.create_empty(chunks_per_shard)

    coords = list(reader)

    assert len(coords) == 4
    assert (0, 0) in coords
    assert (0, 1) in coords
    assert (1, 0) in coords
    assert (1, 1) in coords


def test_shard_reader_getitem_raises_for_empty() -> None:
    """Test _ShardReader.__getitem__ raises KeyError for empty chunks."""
    chunks_per_shard = (2,)
    reader = _ShardReader.create_empty(chunks_per_shard)

    with pytest.raises(KeyError):
        _ = reader[(0,)]


def test_is_total_shard_full() -> None:
    """Test _is_total_shard returns True when all chunk coords are present."""
    codec = ShardingCodec(chunk_shape=(8,))
    chunks_per_shard = (2, 2)
    all_chunk_coords: set[tuple[int, ...]] = {(0, 0), (0, 1), (1, 0), (1, 1)}

    assert codec._is_total_shard(all_chunk_coords, chunks_per_shard) is True


def test_is_total_shard_partial() -> None:
    """Test _is_total_shard returns False for partial chunk coords."""
    codec = ShardingCodec(chunk_shape=(8,))
    chunks_per_shard = (2, 2)
    all_chunk_coords: set[tuple[int, ...]] = {(0, 0), (1, 1)}  # Missing (0, 1) and (1, 0)

    assert codec._is_total_shard(all_chunk_coords, chunks_per_shard) is False


def test_is_total_shard_empty() -> None:
    """Test _is_total_shard returns False for empty chunk coords."""
    codec = ShardingCodec(chunk_shape=(8,))
    chunks_per_shard = (2, 2)
    all_chunk_coords: set[tuple[int, ...]] = set()

    assert codec._is_total_shard(all_chunk_coords, chunks_per_shard) is False


def test_is_total_shard_1d() -> None:
    """Test _is_total_shard works with 1D shards."""
    codec = ShardingCodec(chunk_shape=(8,))
    chunks_per_shard = (4,)
    all_chunk_coords: set[tuple[int, ...]] = {(0,), (1,), (2,), (3,)}

    assert codec._is_total_shard(all_chunk_coords, chunks_per_shard) is True

    # Partial
    partial_coords: set[tuple[int, ...]] = {(0,), (2,)}
    assert codec._is_total_shard(partial_coords, chunks_per_shard) is False
