from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pytest

from zarr.abc.store import ByteRequest
from zarr.codecs.sharding import (
    MAX_UINT_64,
    ShardingCodec,
    _ShardIndex,
    _ShardReader,
)
from zarr.core.buffer import BufferPrototype, default_buffer_prototype
from zarr.core.buffer.cpu import Buffer

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
# Mock ByteGetter for _load_partial_shard_maybe tests
# ============================================================================


@dataclass
class MockByteGetter:
    """Mock ByteGetter for testing."""

    data: bytes
    return_none: bool = False
    get_call_count: int = 0
    get_partial_values_call_count: int = 0

    async def get(
        self, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        self.get_call_count += 1
        if self.return_none:
            return None
        if byte_range is None:
            return Buffer.from_bytes(self.data)
        # For RangeByteRequest, extract start and end
        start = getattr(byte_range, "start", 0)
        end = getattr(byte_range, "end", len(self.data))
        return Buffer.from_bytes(self.data[start:end])

    async def get_partial_values(
        self, prototype: BufferPrototype, byte_ranges: Iterable[ByteRequest | None]
    ) -> list[Buffer | None]:
        self.get_partial_values_call_count += 1
        return [await self.get(prototype, br) for br in byte_ranges]


@dataclass
class MockByteGetterWithIndex:
    """Mock ByteGetter that returns index on first get() and chunk data on get_partial_values()."""

    index_data: bytes | None
    chunk_data: bytes | None
    get_call_count: int = 0
    get_partial_values_call_count: int = 0
    return_none_for_chunks: bool = False

    async def get(
        self, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        self.get_call_count += 1
        if self.index_data is None:
            return None
        return Buffer.from_bytes(self.index_data)

    async def get_partial_values(
        self, prototype: BufferPrototype, byte_ranges: Iterable[ByteRequest | None]
    ) -> list[Buffer | None]:
        self.get_partial_values_call_count += 1
        if self.return_none_for_chunks or self.chunk_data is None:
            return [None for _ in byte_ranges]
        results: list[Buffer | None] = []
        for br in byte_ranges:
            if br is None:
                results.append(Buffer.from_bytes(self.chunk_data))
            else:
                start = getattr(br, "start", 0)
                end = getattr(br, "end", len(self.chunk_data))
                results.append(Buffer.from_bytes(self.chunk_data[start:end]))
        return results


# ============================================================================
# _load_partial_shard_maybe tests
# ============================================================================


async def test_load_partial_shard_maybe_index_load_fails() -> None:
    """Test _load_partial_shard_maybe returns None when index load fails."""
    codec = ShardingCodec(chunk_shape=(8,))
    byte_getter = MockByteGetterWithIndex(index_data=None, chunk_data=None)

    chunks_per_shard = (2,)
    all_chunk_coords: set[tuple[int, ...]] = {(0,)}

    result = await codec._load_partial_shard_maybe(
        byte_getter=byte_getter,  # type: ignore[arg-type]  # mypy false positive: identical signatures
        prototype=default_buffer_prototype(),
        chunks_per_shard=chunks_per_shard,
        all_chunk_coords=all_chunk_coords,
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

    async def mock_load_index(
        self: ShardingCodec, byte_getter: MockByteGetter, cps: tuple[int, ...]
    ) -> _ShardIndex:
        return index

    monkeypatch.setattr(ShardingCodec, "_load_shard_index_maybe", mock_load_index)

    chunk_data = b"x" * 300
    byte_getter = MockByteGetter(data=chunk_data)

    # Request chunks including the empty one
    all_chunk_coords: set[tuple[int, ...]] = {(0,), (1,), (2,)}

    result = await codec._load_partial_shard_maybe(
        byte_getter=byte_getter,  # type: ignore[arg-type]  # mypy false positive: identical signatures
        prototype=default_buffer_prototype(),
        chunks_per_shard=chunks_per_shard,
        all_chunk_coords=all_chunk_coords,
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

    async def mock_load_index(
        self: ShardingCodec, byte_getter: MockByteGetter, cps: tuple[int, ...]
    ) -> _ShardIndex:
        return index

    monkeypatch.setattr(ShardingCodec, "_load_shard_index_maybe", mock_load_index)

    byte_getter = MockByteGetter(data=b"")

    # Request some chunks - all will be empty
    all_chunk_coords: set[tuple[int, ...]] = {(0,), (1,), (2,)}

    result = await codec._load_partial_shard_maybe(
        byte_getter=byte_getter,  # type: ignore[arg-type]  # mypy false positive: identical signatures
        prototype=default_buffer_prototype(),
        chunks_per_shard=chunks_per_shard,
        all_chunk_coords=all_chunk_coords,
    )

    assert result is not None
    assert result == {}  # All chunks were empty, so result is empty dict


async def test_load_partial_shard_uses_get_partial_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that _load_partial_shard_maybe uses get_partial_values for chunk reads."""
    codec = ShardingCodec(chunk_shape=(8,))
    chunks_per_shard = (4,)

    index = _ShardIndex.create_empty(chunks_per_shard)
    index.set_chunk_slice((0,), slice(0, 100))
    index.set_chunk_slice((1,), slice(100, 200))

    async def mock_load_index(
        self: ShardingCodec, byte_getter: MockByteGetter, cps: tuple[int, ...]
    ) -> _ShardIndex:
        return index

    monkeypatch.setattr(ShardingCodec, "_load_shard_index_maybe", mock_load_index)

    chunk_data = b"A" * 100 + b"B" * 100
    byte_getter = MockByteGetter(data=chunk_data)

    all_chunk_coords: set[tuple[int, ...]] = {(0,), (1,)}

    result = await codec._load_partial_shard_maybe(
        byte_getter=byte_getter,  # type: ignore[arg-type]  # mypy false positive: identical signatures
        prototype=default_buffer_prototype(),
        chunks_per_shard=chunks_per_shard,
        all_chunk_coords=all_chunk_coords,
    )

    assert result is not None
    assert len(result) == 2
    assert (0,) in result
    assert (1,) in result

    # get_partial_values should have been called exactly once
    assert byte_getter.get_partial_values_call_count == 1


async def test_load_partial_shard_single_chunk_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test single chunk read (most common case for single element access)."""
    codec = ShardingCodec(chunk_shape=(8,))
    chunks_per_shard = (4,)

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
        byte_getter=byte_getter,  # type: ignore[arg-type]  # mypy false positive: identical signatures
        prototype=default_buffer_prototype(),
        chunks_per_shard=chunks_per_shard,
        all_chunk_coords=all_chunk_coords,
    )

    assert result is not None
    assert (1,) in result
    assert len(result) == 1

    chunk1 = result[(1,)]
    assert chunk1 is not None
    assert chunk1.as_numpy_array().tobytes() == b"E" * 100


async def test_load_partial_shard_chunk_load_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that chunks are omitted from result when get_partial_values returns None."""
    codec = ShardingCodec(chunk_shape=(8,))
    chunks_per_shard = (4,)

    index = _ShardIndex.create_empty(chunks_per_shard)
    index.set_chunk_slice((0,), slice(0, 100))

    async def mock_load_index(
        self: ShardingCodec, byte_getter: MockByteGetterWithIndex, cps: tuple[int, ...]
    ) -> _ShardIndex:
        return index

    monkeypatch.setattr(ShardingCodec, "_load_shard_index_maybe", mock_load_index)

    byte_getter = MockByteGetterWithIndex(
        index_data=b"", chunk_data=None, return_none_for_chunks=True
    )

    all_chunk_coords: set[tuple[int, ...]] = {(0,)}

    result = await codec._load_partial_shard_maybe(
        byte_getter=byte_getter,  # type: ignore[arg-type]  # mypy false positive: identical signatures
        prototype=default_buffer_prototype(),
        chunks_per_shard=chunks_per_shard,
        all_chunk_coords=all_chunk_coords,
    )

    assert result is not None
    assert len(result) == 0


# ============================================================================
# Supporting class tests (_ShardReader, _is_total_shard)
# ============================================================================


def test_shard_reader_create_empty() -> None:
    """Test _ShardReader.create_empty creates reader with empty index."""
    chunks_per_shard = (2, 3)
    reader = _ShardReader.create_empty(chunks_per_shard)

    assert reader.index.is_all_empty()
    assert len(reader.buf) == 0
    assert len(reader) == 2 * 3


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
