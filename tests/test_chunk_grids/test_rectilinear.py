"""Tests for RectilinearChunkGrid implementation."""

import json
from typing import Literal

import numpy as np
import pytest

import zarr
from zarr.core.chunk_grids import (
    RectilinearChunkGrid,
    _compress_run_length_encoding,
    _expand_run_length_encoding,
    _parse_chunk_shapes,
)
from zarr.storage import MemoryStore


def test_expand_run_length_encoding_simple_integers() -> None:
    """Test with simple integer values"""
    assert _expand_run_length_encoding([2, 3, 1]) == (2, 3, 1)


def test_expand_run_length_encoding_single_run_length() -> None:
    """Test with single run-length encoded value"""
    assert _expand_run_length_encoding([[2, 3]]) == (2, 2, 2)


def test_expand_run_length_encoding_mixed() -> None:
    """Test with mix of integers and run-length encoded values"""
    assert _expand_run_length_encoding([1, [2, 1], 3]) == (1, 2, 3)
    assert _expand_run_length_encoding([[1, 3], 3]) == (1, 1, 1, 3)


def test_expand_run_length_encoding_zero_count() -> None:
    """Test with zero count in run-length encoding"""
    assert _expand_run_length_encoding([[2, 0], 3]) == (3,)


def test_expand_run_length_encoding_empty() -> None:
    """Test with empty input"""
    assert _expand_run_length_encoding([]) == ()


def test_expand_run_length_encoding_invalid_run_length_type() -> None:
    """Test error handling for invalid run-length encoding types"""
    with pytest.raises(TypeError, match="must be \\[int, int\\]"):
        _expand_run_length_encoding([["a", 2]])  # type: ignore[list-item]


def test_expand_run_length_encoding_invalid_item_type() -> None:
    """Test error handling for invalid item types"""
    with pytest.raises(TypeError, match="must be int or \\[int, int\\]"):
        _expand_run_length_encoding(["string"])  # type: ignore[list-item]


def test_expand_run_length_encoding_negative_count() -> None:
    """Test error handling for negative count"""
    with pytest.raises(ValueError, match="must be non-negative"):
        _expand_run_length_encoding([[2, -1]])


def test_parse_chunk_shapes_simple_2d() -> None:
    """Test parsing simple 2D chunk shapes"""
    result = _parse_chunk_shapes([[2, 2, 2], [3, 3]])
    assert result == ((2, 2, 2), (3, 3))


def test_parse_chunk_shapes_with_run_length_encoding() -> None:
    """Test parsing with run-length encoding"""
    result = _parse_chunk_shapes([[[2, 3]], [[1, 6]]])
    assert result == ((2, 2, 2), (1, 1, 1, 1, 1, 1))


def test_parse_chunk_shapes_mixed_encoding() -> None:
    """Test parsing with mixed encoding styles"""
    result = _parse_chunk_shapes(
        [
            [1, [2, 1], 3],
            [[1, 3], 3],
        ]
    )
    assert result == ((1, 2, 3), (1, 1, 1, 3))


def test_parse_chunk_shapes_invalid_type() -> None:
    """Test error handling for invalid types"""
    with pytest.raises(TypeError, match="must be a sequence"):
        _parse_chunk_shapes("not a sequence")  # type: ignore[arg-type]


def test_parse_chunk_shapes_invalid_axis_type() -> None:
    """Test error handling for invalid axis type"""
    with pytest.raises(TypeError, match="chunk_shapes\\[0\\] must be a sequence"):
        _parse_chunk_shapes([123])  # type: ignore[list-item]


def test_rectilinear_init_simple() -> None:
    """Test simple initialization"""
    grid = RectilinearChunkGrid(chunk_shapes=[[2, 2, 2], [3, 3]])
    assert grid.chunk_shapes == ((2, 2, 2), (3, 3))


def test_rectilinear_init_validation_non_positive() -> None:
    """Test validation rejects non-positive chunk sizes"""
    with pytest.raises(ValueError, match="must be positive"):
        RectilinearChunkGrid(chunk_shapes=[[2, 0, 2], [3, 3]])


def test_rectilinear_init_validation_non_integer() -> None:
    """Test validation rejects non-integer chunk sizes"""
    with pytest.raises(TypeError, match="must be an int"):
        RectilinearChunkGrid(chunk_shapes=[[2, 2.5, 2], [3, 3]])  # type: ignore[list-item]


def test_rectilinear_from_dict_spec_example() -> None:
    """Test parsing the example from the spec"""
    metadata = {
        "name": "rectilinear",
        "configuration": {
            "kind": "inline",
            "chunk_shapes": [
                [[2, 3]],  # expands to [2, 2, 2]
                [[1, 6]],  # expands to [1, 1, 1, 1, 1, 1]
                [1, [2, 1], 3],  # expands to [1, 2, 3]
                [[1, 3], 3],  # expands to [1, 1, 1, 3]
                [6],  # expands to [6]
            ],
        },
    }

    grid = RectilinearChunkGrid._from_dict(metadata)  # type: ignore[arg-type]

    assert grid.chunk_shapes == (
        (2, 2, 2),
        (1, 1, 1, 1, 1, 1),
        (1, 2, 3),
        (1, 1, 1, 3),
        (6,),
    )


def test_rectilinear_from_dict_invalid_kind() -> None:
    """Test error handling for invalid kind"""
    metadata = {
        "name": "rectilinear",
        "configuration": {
            "kind": "invalid",
            "chunk_shapes": [[2, 2]],
        },
    }
    with pytest.raises(ValueError, match="Only 'inline' kind is supported"):
        RectilinearChunkGrid._from_dict(metadata)  # type: ignore[arg-type]


def test_rectilinear_from_dict_missing_chunk_shapes() -> None:
    """Test error handling for missing chunk_shapes"""
    metadata = {
        "name": "rectilinear",
        "configuration": {
            "kind": "inline",
        },
    }
    with pytest.raises(ValueError, match="must contain 'chunk_shapes'"):
        RectilinearChunkGrid._from_dict(metadata)  # type: ignore[arg-type]


def test_rectilinear_to_dict() -> None:
    """Test serialization to dict with automatic RLE compression"""
    grid = RectilinearChunkGrid(chunk_shapes=[[2, 2, 2], [3, 3]])
    result = grid.to_dict()

    # Chunks are automatically compressed using RLE
    assert result == {
        "name": "rectilinear",
        "configuration": {
            "kind": "inline",
            "chunk_shapes": [[[2, 3]], [[3, 2]]],  # Compressed with RLE
        },
    }


def test_rectilinear_all_chunk_coords_2d() -> None:
    """Test generating all chunk coordinates for 2D array"""
    grid = RectilinearChunkGrid(chunk_shapes=[[2, 2, 2], [3, 3]])
    array_shape = (6, 6)

    coords = list(grid.all_chunk_coords(array_shape))

    # Should have 3 chunks along first axis, 2 along second
    assert len(coords) == 6
    assert coords == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]


def test_rectilinear_all_chunk_coords_validation_mismatch() -> None:
    """Test validation when array shape doesn't match chunk shapes"""
    grid = RectilinearChunkGrid(chunk_shapes=[[2, 2, 2], [3, 3]])

    # Wrong sum
    with pytest.raises(ValueError, match="Sum of chunk sizes"):
        list(grid.all_chunk_coords((7, 6)))

    # Wrong dimensions
    with pytest.raises(ValueError, match="dimensions"):
        list(grid.all_chunk_coords((6, 6, 6)))


def test_rectilinear_get_nchunks() -> None:
    """Test getting total number of chunks"""
    grid = RectilinearChunkGrid(chunk_shapes=[[2, 2, 2], [3, 3], [1, 1, 1, 1, 1, 1]])
    array_shape = (6, 6, 6)

    nchunks = grid.get_nchunks(array_shape)

    # 3 chunks x 2 chunks x 6 chunks = 36 chunks
    assert nchunks == 36


def test_rectilinear_get_nchunks_validation() -> None:
    """Test validation in get_nchunks"""
    grid = RectilinearChunkGrid(chunk_shapes=[[2, 2, 2], [3, 3]])

    # Wrong sum
    with pytest.raises(ValueError, match="Sum of chunk sizes"):
        grid.get_nchunks((7, 6))

    # Wrong dimensions
    with pytest.raises(ValueError, match="dimensions"):
        grid.get_nchunks((6, 6, 6))


def test_rectilinear_roundtrip() -> None:
    """Test that to_dict and from_dict are inverses"""
    original = RectilinearChunkGrid(chunk_shapes=[[1, 2, 3], [4, 5]])
    metadata = original.to_dict()
    reconstructed = RectilinearChunkGrid._from_dict(metadata)

    assert reconstructed.chunk_shapes == original.chunk_shapes


@pytest.mark.parametrize(
    ("input_chunks", "expected_output"),
    [
        # All uniform values
        ((10, 10, 10, 10, 10, 10), [[10, 6]]),
        # All different values - no compression
        ((10, 20, 30), [10, 20, 30]),
        # Mixed runs and single values
        ((10, 10, 10, 20, 20, 30), [[10, 3], [20, 2], 30]),
        # Run at the end
        ((5, 10, 10, 10, 10), [5, [10, 4]]),
        # Run at the beginning
        ((10, 10, 10, 10, 20), [[10, 4], 20]),
        # Alternating runs
        ((5, 5, 10, 10, 10, 10, 15), [[5, 2], [10, 4], 15]),
        # Pairs are compressed
        ((10, 10, 20, 20), [[10, 2], [20, 2]]),
        # Single value stays explicit
        ((10,), [10]),
        # Empty sequence
        ((), []),
    ],
)
def test_compress_run_length_encoding(
    input_chunks: tuple[int, ...], expected_output: list[int | list[int]]
) -> None:
    """Test _compress_run_length_encoding with various input patterns."""
    result = _compress_run_length_encoding(input_chunks)
    assert result == expected_output


def test_compress_rle_large_run() -> None:
    """Test very large run for efficiency."""
    result = _compress_run_length_encoding(tuple([10] * 1000))
    assert result == [[10, 1000]]
    # Verify this is much more compact than expanded
    assert len(str(result)) < len(str([10] * 1000))


@pytest.mark.parametrize(
    ("chunk_shapes", "expected_compressed"),
    [
        # Uniform chunks - fully compressed
        (
            [[10, 10, 10, 10, 10, 10], [5, 5, 5, 5, 5]],
            [[[10, 6]], [[5, 5]]],
        ),
        # Irregular chunks - no compression
        (
            [[10, 20, 30], [5, 10, 15]],
            [[10, 20, 30], [5, 10, 15]],
        ),
        # Mixed compression - some dims compress, others don't
        (
            [[10, 10, 10, 10], [5, 10, 15, 20]],
            [[[10, 4]], [5, 10, 15, 20]],
        ),
        # Partial runs within dimensions
        (
            [[10, 10, 10, 20, 20, 30], [5, 5, 5, 5]],
            [[[10, 3], [20, 2], 30], [[5, 4]]],
        ),
    ],
)
def test_to_dict_compression(
    chunk_shapes: list[list[int]], expected_compressed: list[list[int | list[int]]]
) -> None:
    """Test that RectilinearChunkGrid.to_dict() compresses metadata correctly."""
    grid = RectilinearChunkGrid(chunk_shapes=chunk_shapes)
    result = grid.to_dict()

    assert result == {
        "name": "rectilinear",
        "configuration": {
            "kind": "inline",
            "chunk_shapes": expected_compressed,
        },
    }


def test_roundtrip_with_compression() -> None:
    """Test that compressed metadata can be read back correctly."""
    # Create grid with uniform chunks
    grid1 = RectilinearChunkGrid(chunk_shapes=[[10, 10, 10, 10, 10, 10], [5, 5, 5, 5, 5]])

    # Serialize to dict (should compress)
    metadata = grid1.to_dict()

    # Verify it's compressed
    assert metadata["configuration"]["chunk_shapes"] == [[[10, 6]], [[5, 5]]]  # type: ignore[call-overload,index]

    # Deserialize from dict
    grid2 = RectilinearChunkGrid._from_dict(metadata)

    # Verify the expanded chunk_shapes match
    assert grid2.chunk_shapes == grid1.chunk_shapes
    assert grid2.chunk_shapes == ((10, 10, 10, 10, 10, 10), (5, 5, 5, 5, 5))


def test_json_serialization_with_compression() -> None:
    """Test that compressed metadata is valid JSON."""
    grid = RectilinearChunkGrid(chunk_shapes=[[10] * 100, [5] * 50])
    metadata = grid.to_dict()

    # Should be valid JSON
    json_str = json.dumps(metadata)
    parsed = json.loads(json_str)

    assert parsed == metadata
    # Verify compression happened
    assert parsed["configuration"]["chunk_shapes"] == [[[10, 100]], [[5, 50]]]


def test_compression_saves_space() -> None:
    """Verify that compression actually reduces metadata size."""
    # Large array with uniform chunks
    grid = RectilinearChunkGrid(chunk_shapes=[[10] * 1000, [20] * 500])

    # Serialize with compression
    compressed = grid.to_dict()
    compressed_str = json.dumps(compressed)

    # Manually create uncompressed version
    uncompressed = {
        "name": "rectilinear",
        "configuration": {
            "kind": "inline",
            "chunk_shapes": [[10] * 1000, [20] * 500],
        },
    }
    uncompressed_str = json.dumps(uncompressed)

    # Compressed should be much smaller
    assert len(compressed_str) < len(uncompressed_str) / 10


async def test_api_create_array_with_rle_simple() -> None:
    """Test creating an array using simple RLE format."""
    store = MemoryStore()

    # [[10, 6]] means 6 chunks of size 10 each (RLE format)
    arr = await zarr.api.asynchronous.create_array(
        store=store,
        shape=(60, 60),
        chunks=[[[10, 6]], [[10, 6]]],
        dtype="i4",
        zarr_format=3,
    )

    # Verify the chunk grid was created correctly
    assert isinstance(arr.metadata.chunk_grid, RectilinearChunkGrid)
    # The RLE should be expanded to explicit chunk sizes
    assert arr.metadata.chunk_grid.chunk_shapes == (
        (10, 10, 10, 10, 10, 10),
        (10, 10, 10, 10, 10, 10),
    )

    # Verify functionality
    data = np.arange(60 * 60, dtype="i4").reshape(60, 60)
    await arr.setitem(slice(None), data)
    result = await arr.getitem(slice(None))
    np.testing.assert_array_equal(result, data)


async def test_api_create_array_with_mixed_rle_and_explicit() -> None:
    """Test creating an array with mixed RLE and explicit chunk sizes."""
    store = MemoryStore()

    arr = await zarr.api.asynchronous.create_array(
        store=store,
        shape=(6, 6),
        chunks=[[[2, 3]], [1, [2, 1], 3]],
        dtype="f8",
        zarr_format=3,
    )

    assert isinstance(arr.metadata.chunk_grid, RectilinearChunkGrid)
    assert arr.metadata.chunk_grid.chunk_shapes == ((2, 2, 2), (1, 2, 3))

    # Test data operations
    data = np.random.random((6, 6))
    await arr.setitem(slice(None), data)
    result = await arr.getitem(slice(None))
    np.testing.assert_array_almost_equal(result, data)  # type: ignore[arg-type]


async def test_api_rle_chunk_grid_roundtrip_persistence() -> None:
    """Test that arrays created with RLE persist correctly."""
    store = MemoryStore()

    # Create array with RLE chunks
    arr1 = await zarr.api.asynchronous.create_array(
        store=store,
        name="rle_array",
        shape=(100, 50),
        chunks=[[[10, 10]], [[10, 5]]],
        dtype="u2",
        zarr_format=3,
    )

    # Write data
    data = np.arange(100 * 50, dtype="u2").reshape(100, 50)
    await arr1.setitem(slice(None), data)

    # Re-open the array
    arr2 = await zarr.api.asynchronous.open_array(store=store, path="rle_array")

    # Verify chunk_grid is preserved with expanded RLE
    assert isinstance(arr2.metadata.chunk_grid, RectilinearChunkGrid)
    assert arr2.metadata.chunk_grid.chunk_shapes == (
        (10, 10, 10, 10, 10, 10, 10, 10, 10, 10),
        (10, 10, 10, 10, 10),
    )

    # Verify data is preserved
    result = await arr2.getitem(slice(None))
    np.testing.assert_array_equal(result, data)


async def test_api_rle_spec_example() -> None:
    """Test the exact RLE example from the Zarr v3 spec."""
    store = MemoryStore()

    arr = await zarr.api.asynchronous.create_array(
        store=store,
        shape=(6, 6, 6, 4, 6),
        chunks=[
            [[2, 3]],
            [[1, 6]],
            [1, [2, 1], 3],
            [[1, 3], 1],
            [6],
        ],
        dtype="i1",
        zarr_format=3,
    )

    assert isinstance(arr.metadata.chunk_grid, RectilinearChunkGrid)
    assert arr.metadata.chunk_grid.chunk_shapes == (
        (2, 2, 2),
        (1, 1, 1, 1, 1, 1),
        (1, 2, 3),
        (1, 1, 1, 1),
        (6,),
    )

    # Verify we can read/write with this complex chunking
    data = np.arange(6 * 6 * 6 * 4 * 6, dtype="i1").reshape(6, 6, 6, 4, 6)
    await arr.setitem(slice(None), data)
    result = await arr.getitem(slice(None))
    np.testing.assert_array_equal(result, data)


def test_api_synchronous_api_with_rle_chunks() -> None:
    """Test that RLE chunks work with the synchronous API."""
    store = MemoryStore()

    arr = zarr.create_array(
        store=store,
        shape=(30, 40),
        chunks=[[[10, 3]], [[10, 4]]],
        dtype="f4",
        zarr_format=3,
    )

    assert isinstance(arr.metadata.chunk_grid, RectilinearChunkGrid)
    assert arr.metadata.chunk_grid.chunk_shapes == ((10, 10, 10), (10, 10, 10, 10))

    # Test write/read
    data = np.random.random((30, 40)).astype("f4")
    arr[:] = data
    np.testing.assert_array_almost_equal(arr[:], data)  # type: ignore[arg-type]


async def test_api_rle_with_zero_count() -> None:
    """Test RLE with zero count (should result in no chunks from that entry)."""
    store = MemoryStore()

    arr = await zarr.api.asynchronous.create_array(
        store=store,
        shape=(10, 10),
        chunks=[[[5, 0], 5, 5], [[5, 2]]],
        dtype="u1",
        zarr_format=3,
    )

    assert isinstance(arr.metadata.chunk_grid, RectilinearChunkGrid)
    assert arr.metadata.chunk_grid.chunk_shapes == ((5, 5), (5, 5))

    # Test functionality
    data = np.arange(10 * 10, dtype="u1").reshape(10, 10)
    await arr.setitem(slice(None), data)
    result = await arr.getitem(slice(None))
    np.testing.assert_array_equal(result, data)


def test_api_group_create_array_with_rle() -> None:
    """Test creating arrays with RLE chunks via Group.create_array()."""
    store = MemoryStore()
    root = zarr.open_group(store, mode="w", zarr_format=3)

    arr = root.create_array(
        "rle_test",
        shape=(50, 50),
        chunks=[[[10, 5]], [[10, 5]]],
        dtype="i8",
    )

    assert isinstance(arr.metadata.chunk_grid, RectilinearChunkGrid)
    assert arr.metadata.chunk_grid.chunk_shapes == (
        (10, 10, 10, 10, 10),
        (10, 10, 10, 10, 10),
    )

    # Verify the array is accessible from the group
    arr2 = root["rle_test"]
    assert isinstance(arr2.metadata.chunk_grid, RectilinearChunkGrid)  # type: ignore[union-attr]


async def test_api_rle_with_large_repeat_count() -> None:
    """Test RLE with large repeat counts for efficiency."""
    store = MemoryStore()

    arr = await zarr.api.asynchronous.create_array(
        store=store,
        shape=(1000, 1000),
        chunks=[[[10, 100]], [[10, 100]]],
        dtype="i2",
        zarr_format=3,
    )

    assert isinstance(arr.metadata.chunk_grid, RectilinearChunkGrid)
    # Verify that RLE was expanded to 100 chunks per dimension
    assert len(arr.metadata.chunk_grid.chunk_shapes[0]) == 100
    assert len(arr.metadata.chunk_grid.chunk_shapes[1]) == 100
    assert all(c == 10 for c in arr.metadata.chunk_grid.chunk_shapes[0])
    assert all(c == 10 for c in arr.metadata.chunk_grid.chunk_shapes[1])

    # Verify basic functionality (don't write full array for speed)
    await arr.setitem((slice(0, 10), slice(0, 10)), np.ones((10, 10), dtype="i2"))
    result = await arr.getitem((slice(0, 10), slice(0, 10)))
    np.testing.assert_array_equal(result, np.ones((10, 10), dtype="i2"))


async def test_api_rle_mixed_with_irregular_chunks() -> None:
    """Test RLE combined with irregular explicit chunk sizes."""
    store = MemoryStore()

    arr = await zarr.api.asynchronous.create_array(
        store=store,
        shape=(100, 100),
        chunks=[[[10, 5], 50], [25, 30, 20, 25]],
        dtype="u4",
        zarr_format=3,
    )

    assert isinstance(arr.metadata.chunk_grid, RectilinearChunkGrid)
    assert arr.metadata.chunk_grid.chunk_shapes == (
        (10, 10, 10, 10, 10, 50),
        (25, 30, 20, 25),
    )

    # Test read/write
    data = np.arange(100 * 100, dtype="u4").reshape(100, 100)
    await arr.setitem(slice(None), data)
    result = await arr.getitem(slice(None))
    np.testing.assert_array_equal(result, data)


@pytest.mark.parametrize("zarr_format", [2])
async def test_api_v2_rejects_rle_chunks(zarr_format: Literal[2, 3]) -> None:
    """Test that Zarr v2 rejects RLE chunk specifications."""
    store = MemoryStore()

    with pytest.raises(ValueError, match="Variable chunks.*only supported in Zarr format 3"):
        await zarr.api.asynchronous.create_array(
            store=store,
            shape=(60, 60),
            chunks=[[[10, 6]], [[10, 6]]],
            dtype="i4",
            zarr_format=zarr_format,
        )


async def test_api_from_array_rejects_rle_chunks() -> None:
    """Test that from_array rejects RLE chunks."""
    store = MemoryStore()
    data = np.arange(30 * 30, dtype="i4").reshape(30, 30)

    with pytest.raises(
        ValueError,
        match="Cannot use RectilinearChunkGrid.*when creating array from data",
    ):
        await zarr.api.asynchronous.from_array(
            store=store,
            data=data,
            chunks=[[[10, 3]], [[10, 3]]],  # type: ignore[arg-type]
            zarr_format=3,
        )


# =============================================================================
# Tests for Partial Edge Chunks
# =============================================================================


def test_partial_edge_chunk_validation() -> None:
    """Test that chunk sums >= array shape is valid."""
    # Chunk sum (15) > array shape (12) - should be valid
    grid = RectilinearChunkGrid(chunk_shapes=[[5, 5, 5], [10, 10]])
    # Should not raise - partial edge chunks are allowed
    grid._validate_array_shape((12, 18))


def test_partial_edge_chunk_exact_match() -> None:
    """Test that chunk sums == array shape still works."""
    grid = RectilinearChunkGrid(chunk_shapes=[[5, 5, 5], [10, 10]])
    # Exact match should work
    grid._validate_array_shape((15, 20))


def test_partial_edge_chunk_all_chunk_coords() -> None:
    """Test that all_chunk_coords works with partial edge chunks."""
    grid = RectilinearChunkGrid(chunk_shapes=[[10, 10, 10], [20, 20]])
    # Array shape (25, 35) is less than chunk sum (30, 40)
    coords = list(grid.all_chunk_coords((25, 35)))
    # Should still have 3 x 2 = 6 chunks
    assert len(coords) == 6
    assert coords == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]


def test_partial_edge_chunk_get_nchunks() -> None:
    """Test get_nchunks with partial edge chunks."""
    grid = RectilinearChunkGrid(chunk_shapes=[[10, 10, 10], [20, 20]])
    # Array shape smaller than chunk sum
    nchunks = grid.get_nchunks((25, 35))
    assert nchunks == 6


def test_partial_edge_chunk_get_chunk_shape() -> None:
    """Test get_chunk_shape returns full chunk size even for partial chunks."""
    grid = RectilinearChunkGrid(chunk_shapes=[[10, 10, 10], [20, 20]])
    # The chunk shape is defined by the grid, not truncated to array shape
    shape = grid.get_chunk_shape((25, 35), (2, 1))
    assert shape == (10, 20)


def test_partial_edge_chunk_get_chunk_start() -> None:
    """Test get_chunk_start with partial edge chunks."""
    grid = RectilinearChunkGrid(chunk_shapes=[[10, 10, 10], [20, 20]])
    # Last chunk starts at (20, 20), even though array is only (25, 35)
    start = grid.get_chunk_start((25, 35), (2, 1))
    assert start == (20, 20)


async def test_api_create_array_with_partial_edge_chunks() -> None:
    """Test creating an array with partial edge chunks via API."""
    store = MemoryStore()

    # Array shape (55, 45) with chunks that sum to (60, 50)
    arr = await zarr.api.asynchronous.create_array(
        store=store,
        shape=(55, 45),
        chunks=[[10, 20, 30], [25, 25]],
        dtype="f4",
        zarr_format=3,
    )

    assert isinstance(arr.metadata.chunk_grid, RectilinearChunkGrid)
    assert arr.metadata.chunk_grid.chunk_shapes == ((10, 20, 30), (25, 25))

    # Write and read data
    data = np.arange(55 * 45, dtype="f4").reshape(55, 45)
    await arr.setitem(slice(None), data)
    result = await arr.getitem(slice(None))
    np.testing.assert_array_equal(result, data)


async def test_api_partial_edge_chunks_slicing() -> None:
    """Test that slicing works correctly with partial edge chunks."""
    store = MemoryStore()

    # Chunks sum to (30, 30), array is (28, 27)
    arr = await zarr.api.asynchronous.create_array(
        store=store,
        shape=(28, 27),
        chunks=[[10, 10, 10], [15, 15]],
        dtype="i4",
        zarr_format=3,
    )

    # Write full array
    data = np.arange(28 * 27, dtype="i4").reshape(28, 27)
    await arr.setitem(slice(None), data)

    # Read slices that span the partial edge chunk
    result = await arr.getitem((slice(25, 28), slice(20, 27)))
    expected = data[25:28, 20:27]
    np.testing.assert_array_equal(result, expected)


def test_sync_api_partial_edge_chunks() -> None:
    """Test partial edge chunks with synchronous API."""
    store = MemoryStore()

    # 95x95 array with 100x100 worth of chunks
    arr = zarr.create_array(
        store=store,
        shape=(95, 95),
        chunks=[[50, 50], [50, 50]],
        dtype="u2",
        zarr_format=3,
    )

    assert isinstance(arr.metadata.chunk_grid, RectilinearChunkGrid)

    # Write and read
    data = np.arange(95 * 95, dtype="u2").reshape(95, 95)
    arr[:] = data
    np.testing.assert_array_equal(arr[:], data)


async def test_api_partial_edge_single_chunk() -> None:
    """Test array smaller than a single chunk."""
    store = MemoryStore()

    # Array is 50x50, but single chunk is 100x100
    arr = await zarr.api.asynchronous.create_array(
        store=store,
        shape=(50, 50),
        chunks=[[100], [100]],
        dtype="f8",
        zarr_format=3,
    )

    assert isinstance(arr.metadata.chunk_grid, RectilinearChunkGrid)
    assert arr.metadata.chunk_grid.chunk_shapes == ((100,), (100,))

    # Should work fine
    data = np.random.random((50, 50))
    await arr.setitem(slice(None), data)
    result = await arr.getitem(slice(None))
    np.testing.assert_array_almost_equal(result, data)  # type: ignore[arg-type]


# =============================================================================
# Tests for Resizing Variable Chunked Arrays
# =============================================================================


def test_update_shape_no_change() -> None:
    """Test update_shape when shape doesn't change."""
    grid = RectilinearChunkGrid(chunk_shapes=[[10, 20, 30], [25, 25]])
    new_grid = grid.update_shape((60, 50))

    assert new_grid.chunk_shapes == grid.chunk_shapes


def test_update_shape_grow_single_dim() -> None:
    """Test update_shape when growing a single dimension."""
    grid = RectilinearChunkGrid(chunk_shapes=[[10, 20, 30], [25, 25]])
    # Grow first dimension from 60 to 80
    new_grid = grid.update_shape((80, 50))

    # Should add a new chunk of size 20 (80 - 60 = 20)
    assert new_grid.chunk_shapes[0] == (10, 20, 30, 20)
    assert new_grid.chunk_shapes[1] == (25, 25)


def test_update_shape_grow_multiple_dims() -> None:
    """Test update_shape when growing multiple dimensions."""
    grid = RectilinearChunkGrid(chunk_shapes=[[10, 10], [20, 20]])
    # Grow from (20, 40) to (35, 55)
    new_grid = grid.update_shape((35, 55))

    # First dim: 20 + 15 = 35, adds chunk of size 15
    # Second dim: 40 + 15 = 55, adds chunk of size 15
    assert new_grid.chunk_shapes[0] == (10, 10, 15)
    assert new_grid.chunk_shapes[1] == (20, 20, 15)


def test_update_shape_shrink_single_dim() -> None:
    """Test update_shape when shrinking a single dimension."""
    grid = RectilinearChunkGrid(chunk_shapes=[[10, 20, 30, 40], [25, 25]])
    # Shrink first dimension from 100 to 35 (keeps first two full chunks + partial)
    new_grid = grid.update_shape((35, 50))

    # Should keep chunks that contain data (10 + 20 = 30, need 35, so keep 10, 20, 30)
    assert new_grid.chunk_shapes[0] == (10, 20, 30)
    assert new_grid.chunk_shapes[1] == (25, 25)


def test_update_shape_shrink_to_single_chunk() -> None:
    """Test update_shape when shrinking to fit in first chunk."""
    grid = RectilinearChunkGrid(chunk_shapes=[[10, 20, 30], [25, 25]])
    # Shrink to fit in first chunk
    new_grid = grid.update_shape((5, 50))

    assert new_grid.chunk_shapes[0] == (10,)
    assert new_grid.chunk_shapes[1] == (25, 25)


def test_update_shape_shrink_multiple_dims() -> None:
    """Test update_shape when shrinking multiple dimensions."""
    grid = RectilinearChunkGrid(chunk_shapes=[[10, 10, 10, 10], [20, 20, 20]])
    # Shrink from (40, 60) to (25, 35)
    new_grid = grid.update_shape((25, 35))

    # First dim: 25 falls within third chunk (10+10+10=30)
    # Second dim: 35 falls within second chunk (20+20=40)
    assert new_grid.chunk_shapes[0] == (10, 10, 10)
    assert new_grid.chunk_shapes[1] == (20, 20)


def test_update_shape_dimension_mismatch_error() -> None:
    """Test that update_shape raises error on dimension mismatch."""
    grid = RectilinearChunkGrid(chunk_shapes=[[10, 20], [30, 40]])

    with pytest.raises(ValueError, match="dimensions"):
        grid.update_shape((30, 70, 100))


async def test_api_resize_grow_array() -> None:
    """Test resizing (growing) an array with variable chunks."""
    store = MemoryStore()

    # Create initial array
    arr = await zarr.api.asynchronous.create_array(
        store=store,
        shape=(30, 40),
        chunks=[[10, 20], [20, 20]],
        dtype="i4",
        zarr_format=3,
    )

    # Write initial data
    data = np.arange(30 * 40, dtype="i4").reshape(30, 40)
    await arr.setitem(slice(None), data)

    # Resize to larger shape
    await arr.resize((50, 60))

    # Verify new shape
    assert arr.shape == (50, 60)

    # Verify chunk grid was updated
    assert isinstance(arr.metadata.chunk_grid, RectilinearChunkGrid)
    # Should have added chunks for the extra space
    assert arr.metadata.chunk_grid.chunk_shapes[0] == (10, 20, 20)  # 30 + 20 = 50
    assert arr.metadata.chunk_grid.chunk_shapes[1] == (20, 20, 20)  # 40 + 20 = 60

    # Verify original data is preserved
    result = await arr.getitem((slice(0, 30), slice(0, 40)))
    np.testing.assert_array_equal(result, data)


async def test_api_resize_shrink_array() -> None:
    """Test resizing (shrinking) an array with variable chunks."""
    store = MemoryStore()

    # Create initial array
    arr = await zarr.api.asynchronous.create_array(
        store=store,
        shape=(60, 50),
        chunks=[[10, 20, 30], [25, 25]],
        dtype="f4",
        zarr_format=3,
    )

    # Write initial data
    data = np.arange(60 * 50, dtype="f4").reshape(60, 50)
    await arr.setitem(slice(None), data)

    # Resize to smaller shape (within first two chunks of first dim)
    await arr.resize((25, 30))

    # Verify new shape
    assert arr.shape == (25, 30)

    # Verify chunk grid was updated - keeps chunks that cover the new shape
    assert isinstance(arr.metadata.chunk_grid, RectilinearChunkGrid)
    # First dim: 25 <= 10+20=30, so keeps (10, 20, 30) but only uses partial
    # Actually the implementation keeps the minimal set of chunks
    assert sum(arr.metadata.chunk_grid.chunk_shapes[0]) >= 25

    # Verify preserved data
    result = await arr.getitem(slice(None))
    np.testing.assert_array_equal(result, data[:25, :30])


def test_sync_api_resize_grow() -> None:
    """Test resizing with synchronous API."""
    store = MemoryStore()

    arr = zarr.create_array(
        store=store,
        shape=(20, 30),
        chunks=[[10, 10], [15, 15]],
        dtype="u1",
        zarr_format=3,
    )

    # Write data
    data = np.arange(20 * 30, dtype="u1").reshape(20, 30)
    arr[:] = data

    # Resize
    arr.resize((35, 45))

    assert arr.shape == (35, 45)
    assert isinstance(arr.metadata.chunk_grid, RectilinearChunkGrid)

    # Verify original data preserved
    np.testing.assert_array_equal(arr[:20, :30], data)


def test_sync_api_resize_shrink() -> None:
    """Test shrinking with synchronous API."""
    store = MemoryStore()

    arr = zarr.create_array(
        store=store,
        shape=(40, 50),
        chunks=[[10, 10, 10, 10], [25, 25]],
        dtype="i2",
        zarr_format=3,
    )

    data = np.arange(40 * 50, dtype="i2").reshape(40, 50)
    arr[:] = data

    # Shrink
    arr.resize((15, 30))

    assert arr.shape == (15, 30)
    np.testing.assert_array_equal(arr[:], data[:15, :30])


# =============================================================================
# Tests for Appending to Variable Chunked Arrays
# =============================================================================


async def test_api_append_to_first_axis() -> None:
    """Test appending data along the first axis."""
    store = MemoryStore()

    # Create initial array
    arr = await zarr.api.asynchronous.create_array(
        store=store,
        shape=(30, 20),
        chunks=[[10, 20], [10, 10]],
        dtype="i4",
        zarr_format=3,
    )

    # Write initial data
    initial_data = np.arange(30 * 20, dtype="i4").reshape(30, 20)
    await arr.setitem(slice(None), initial_data)

    # Append data along first axis
    append_data = np.arange(30 * 20, 45 * 20, dtype="i4").reshape(15, 20)
    await arr.append(append_data, axis=0)

    # Verify new shape
    assert arr.shape == (45, 20)

    # Verify chunk grid was updated
    assert isinstance(arr.metadata.chunk_grid, RectilinearChunkGrid)
    # Should have added a chunk for the appended data
    assert sum(arr.metadata.chunk_grid.chunk_shapes[0]) >= 45

    # Verify all data
    result = await arr.getitem(slice(None))
    expected = np.vstack([initial_data, append_data])
    np.testing.assert_array_equal(result, expected)


async def test_api_append_to_second_axis() -> None:
    """Test appending data along the second axis."""
    store = MemoryStore()

    # Create initial array
    arr = await zarr.api.asynchronous.create_array(
        store=store,
        shape=(20, 30),
        chunks=[[10, 10], [10, 20]],
        dtype="f4",
        zarr_format=3,
    )

    # Write initial data
    initial_data = np.arange(20 * 30, dtype="f4").reshape(20, 30)
    await arr.setitem(slice(None), initial_data)

    # Append data along second axis
    append_data = np.arange(20 * 30, 20 * 45, dtype="f4").reshape(20, 15)
    await arr.append(append_data, axis=1)

    # Verify new shape
    assert arr.shape == (20, 45)

    # Verify all data
    result = await arr.getitem(slice(None))
    expected = np.hstack([initial_data, append_data])
    np.testing.assert_array_equal(result, expected)


def test_sync_api_append() -> None:
    """Test appending with synchronous API."""
    store = MemoryStore()

    arr = zarr.create_array(
        store=store,
        shape=(20, 20),
        chunks=[[10, 10], [10, 10]],
        dtype="u2",
        zarr_format=3,
    )

    # Write initial data
    initial_data = np.arange(20 * 20, dtype="u2").reshape(20, 20)
    arr[:] = initial_data

    # Append
    append_data = np.arange(20 * 20, 25 * 20, dtype="u2").reshape(5, 20)
    arr.append(append_data, axis=0)

    assert arr.shape == (25, 20)
    np.testing.assert_array_equal(arr[:20, :], initial_data)
    np.testing.assert_array_equal(arr[20:, :], append_data)


async def test_api_multiple_appends() -> None:
    """Test multiple successive appends."""
    store = MemoryStore()

    arr = await zarr.api.asynchronous.create_array(
        store=store,
        shape=(10, 10),
        chunks=[[5, 5], [5, 5]],
        dtype="i4",
        zarr_format=3,
    )

    initial_data = np.arange(10 * 10, dtype="i4").reshape(10, 10)
    await arr.setitem(slice(None), initial_data)

    # Multiple appends
    all_data = [initial_data]
    for i in range(3):
        append_data = np.full((5, 10), i + 100, dtype="i4")
        await arr.append(append_data, axis=0)
        all_data.append(append_data)

    assert arr.shape == (25, 10)

    result = await arr.getitem(slice(None))
    expected = np.vstack(all_data)
    np.testing.assert_array_equal(result, expected)


async def test_api_append_with_partial_edge_chunks() -> None:
    """Test appending to array that already has partial edge chunks."""
    store = MemoryStore()

    # Create array with partial edge chunk
    arr = await zarr.api.asynchronous.create_array(
        store=store,
        shape=(25, 30),  # Less than chunk sum of (30, 30)
        chunks=[[15, 15], [15, 15]],
        dtype="f8",
        zarr_format=3,
    )

    initial_data = np.random.random((25, 30))
    await arr.setitem(slice(None), initial_data)

    # Append to extend beyond original chunk boundary
    append_data = np.random.random((10, 30))
    await arr.append(append_data, axis=0)

    assert arr.shape == (35, 30)

    result = await arr.getitem(slice(None))
    expected = np.vstack([initial_data, append_data])
    np.testing.assert_array_almost_equal(result, expected)  # type: ignore[arg-type]


# =============================================================================
# Tests for Edge Cases in Resize and Append
# =============================================================================


def test_update_shape_boundary_cases() -> None:
    """Test update_shape at exact chunk boundaries."""
    grid = RectilinearChunkGrid(chunk_shapes=[[10, 20, 30], [25, 25]])

    # Grow to exact chunk boundary
    new_grid = grid.update_shape((60, 75))
    assert new_grid.chunk_shapes[0] == (10, 20, 30)  # No change (60 == sum)
    assert new_grid.chunk_shapes[1] == (25, 25, 25)  # Added chunk of 25

    # Shrink to exact chunk boundary - keeps minimal chunks that cover the shape
    grid2 = RectilinearChunkGrid(chunk_shapes=[[10, 20, 30], [25, 25, 25]])
    new_grid2 = grid2.update_shape((30, 50))
    # First dim: 30 is covered by (10, 20) since 10+20=30
    # Second dim: 50 is covered by (25, 25) since 25+25=50
    assert new_grid2.chunk_shapes[0] == (10, 20)
    assert new_grid2.chunk_shapes[1] == (25, 25)


async def test_append_small_data() -> None:
    """Test appending small amounts of data."""
    store = MemoryStore()

    arr = await zarr.api.asynchronous.create_array(
        store=store,
        shape=(20, 20),
        chunks=[[10, 10], [10, 10]],
        dtype="i4",
        zarr_format=3,
    )

    data = np.arange(20 * 20, dtype="i4").reshape(20, 20)
    await arr.setitem(slice(None), data)

    # Append small array (less than one chunk)
    small = np.full((3, 20), 999, dtype="i4")
    await arr.append(small, axis=0)

    assert arr.shape == (23, 20)
    result = await arr.getitem((slice(20, 23), slice(None)))
    np.testing.assert_array_equal(result, small)


async def test_roundtrip_after_resize() -> None:
    """Test that array can be reopened after resize."""
    store = MemoryStore()

    arr = await zarr.api.asynchronous.create_array(
        store=store,
        name="resizable",
        shape=(30, 30),
        chunks=[[10, 10, 10], [15, 15]],
        dtype="f4",
        zarr_format=3,
    )

    data = np.random.random((30, 30)).astype("f4")
    await arr.setitem(slice(None), data)

    # Resize
    await arr.resize((45, 40))
    new_data = np.random.random((45, 40)).astype("f4")
    new_data[:30, :30] = data
    await arr.setitem(slice(None), new_data)

    # Reopen and verify
    arr2 = await zarr.api.asynchronous.open_array(store=store, path="resizable")

    assert arr2.shape == (45, 40)
    assert isinstance(arr2.metadata.chunk_grid, RectilinearChunkGrid)

    result = await arr2.getitem(slice(None))
    np.testing.assert_array_almost_equal(result, new_data)  # type: ignore[arg-type]


async def test_roundtrip_after_append() -> None:
    """Test that array can be reopened after append."""
    store = MemoryStore()

    arr = await zarr.api.asynchronous.create_array(
        store=store,
        name="appendable",
        shape=(20, 20),
        chunks=[[10, 10], [10, 10]],
        dtype="i2",
        zarr_format=3,
    )

    data = np.arange(20 * 20, dtype="i2").reshape(20, 20)
    await arr.setitem(slice(None), data)

    # Append
    append_data = np.arange(20 * 20, 25 * 20, dtype="i2").reshape(5, 20)
    await arr.append(append_data, axis=0)

    # Reopen and verify
    arr2 = await zarr.api.asynchronous.open_array(store=store, path="appendable")

    assert arr2.shape == (25, 20)
    result = await arr2.getitem(slice(None))
    expected = np.vstack([data, append_data])
    np.testing.assert_array_equal(result, expected)


async def test_resize_preserve_single_chunk() -> None:
    """Test resizing preserves at least one chunk."""
    store = MemoryStore()

    arr = await zarr.api.asynchronous.create_array(
        store=store,
        shape=(100, 100),
        chunks=[[25, 25, 25, 25], [50, 50]],
        dtype="u1",
        zarr_format=3,
    )

    data = np.arange(100 * 100, dtype="u1").reshape(100, 100)
    await arr.setitem(slice(None), data)

    # Shrink to fit in first chunk
    await arr.resize((10, 30))

    assert arr.shape == (10, 30)
    result = await arr.getitem(slice(None))
    np.testing.assert_array_equal(result, data[:10, :30])


# =============================================================================
# Tests for Array.chunks Property with RectilinearChunkGrid
# =============================================================================


async def test_array_chunks_property_rectilinear() -> None:
    """Test that Array.chunks returns tuple of tuples for RectilinearChunkGrid."""
    store = MemoryStore()

    arr = await zarr.api.asynchronous.create_array(
        store=store,
        shape=(60, 50),
        chunks=[[10, 20, 30], [25, 25]],
        dtype="f4",
        zarr_format=3,
    )

    # chunks should return the expanded tuple of tuples (not RLE)
    chunks = arr.chunks
    assert chunks == ((10, 20, 30), (25, 25))
    assert isinstance(chunks, tuple)
    assert all(isinstance(dim_chunks, tuple) for dim_chunks in chunks)
    assert all(isinstance(size, int) for dim_chunks in chunks for size in dim_chunks)


async def test_array_chunks_property_rle_expanded() -> None:
    """Test that Array.chunks returns expanded (not RLE) chunks."""
    store = MemoryStore()

    # Create with RLE format
    arr = await zarr.api.asynchronous.create_array(
        store=store,
        shape=(60, 60),
        chunks=[[[10, 6]], [[10, 6]]],  # RLE: 6 chunks of size 10
        dtype="i2",
        zarr_format=3,
    )

    # chunks property should return expanded format
    chunks = arr.chunks
    assert chunks == ((10, 10, 10, 10, 10, 10), (10, 10, 10, 10, 10, 10))
    # Verify it's not RLE - each element is an int, not a list
    assert len(chunks[0]) == 6
    assert len(chunks[1]) == 6


def test_sync_array_chunks_property_rectilinear() -> None:
    """Test that sync Array.chunks returns tuple of tuples for RectilinearChunkGrid."""
    store = MemoryStore()

    arr = zarr.create_array(
        store=store,
        shape=(100, 100),
        chunks=[[10, 20, 30, 40], [25, 25, 50]],
        dtype="u4",
        zarr_format=3,
    )

    chunks = arr.chunks
    assert chunks == ((10, 20, 30, 40), (25, 25, 50))


async def test_array_chunks_property_regular() -> None:
    """Test that Array.chunks still returns tuple[int, ...] for RegularChunkGrid."""
    store = MemoryStore()

    arr = await zarr.api.asynchronous.create_array(
        store=store,
        shape=(100, 100),
        chunks=(10, 20),  # Regular chunks
        dtype="f8",
        zarr_format=3,
    )

    chunks = arr.chunks
    # For regular chunks, it's tuple[int, ...]
    assert chunks == (10, 20)
    assert isinstance(chunks[0], int)
    assert isinstance(chunks[1], int)


async def test_array_chunks_property_after_resize() -> None:
    """Test that Array.chunks reflects updated chunks after resize."""
    store = MemoryStore()

    arr = await zarr.api.asynchronous.create_array(
        store=store,
        shape=(30, 30),
        chunks=[[10, 20], [15, 15]],
        dtype="i4",
        zarr_format=3,
    )

    assert arr.chunks == ((10, 20), (15, 15))

    # Resize to grow
    await arr.resize((50, 45))

    # Chunks should be updated
    chunks = arr.chunks
    assert chunks == ((10, 20, 20), (15, 15, 15))  # New chunk added


async def test_metadata_chunks_property_matches_array() -> None:
    """Test that metadata.chunks matches arr.chunks."""
    store = MemoryStore()

    arr = await zarr.api.asynchronous.create_array(
        store=store,
        shape=(60, 50),
        chunks=[[10, 20, 30], [25, 25]],
        dtype="f4",
        zarr_format=3,
    )

    # Both should return the same value
    assert arr.chunks == arr.metadata.chunks
    assert arr.chunks == ((10, 20, 30), (25, 25))


# ===================================================================
# Tests for array_index_to_chunk_coord() method
# ===================================================================


def test_array_index_to_chunk_coord_basic() -> None:
    """Test array_index_to_chunk_coord with basic cases."""
    grid = RectilinearChunkGrid(chunk_shapes=[[2, 3, 1], [4, 2]])
    array_shape = (6, 6)

    # First element
    assert grid.array_index_to_chunk_coord(array_shape, (0, 0)) == (0, 0)

    # Last element
    assert grid.array_index_to_chunk_coord(array_shape, (5, 5)) == (2, 1)

    # Middle elements
    assert grid.array_index_to_chunk_coord(array_shape, (2, 0)) == (1, 0)
    assert grid.array_index_to_chunk_coord(array_shape, (0, 4)) == (0, 1)


def test_array_index_to_chunk_coord_boundaries() -> None:
    """Test array_index_to_chunk_coord at chunk boundaries."""
    grid = RectilinearChunkGrid(chunk_shapes=[[10, 20, 30], [25, 25]])
    array_shape = (60, 50)

    # Boundaries between chunks
    assert grid.array_index_to_chunk_coord(array_shape, (9, 24)) == (0, 0)  # Last of first chunk
    assert grid.array_index_to_chunk_coord(array_shape, (10, 25)) == (1, 1)  # First of second chunk
    assert grid.array_index_to_chunk_coord(array_shape, (29, 49)) == (1, 1)
    assert grid.array_index_to_chunk_coord(array_shape, (30, 0)) == (2, 0)


def test_array_index_to_chunk_coord_all_dims() -> None:
    """Test array_index_to_chunk_coord with all dimensions."""
    grid = RectilinearChunkGrid(chunk_shapes=[[2, 2, 2], [3, 3]])
    array_shape = (6, 6)

    # Test all combinations
    test_cases = [
        ((0, 0), (0, 0)),
        ((1, 1), (0, 0)),
        ((2, 3), (1, 1)),
        ((3, 4), (1, 1)),
        ((4, 0), (2, 0)),
        ((5, 5), (2, 1)),
    ]

    for array_idx, expected_chunk in test_cases:
        assert grid.array_index_to_chunk_coord(array_shape, array_idx) == expected_chunk


def test_array_index_to_chunk_coord_irregular() -> None:
    """Test array_index_to_chunk_coord with highly irregular chunks."""
    grid = RectilinearChunkGrid(chunk_shapes=[[1, 5, 2, 12], [3, 1, 6]])
    array_shape = (20, 10)

    # Test various positions
    assert grid.array_index_to_chunk_coord(array_shape, (0, 0)) == (0, 0)  # Chunk (0,0): size 1x3
    assert grid.array_index_to_chunk_coord(array_shape, (1, 3)) == (1, 1)  # Chunk (1,1): size 5x1
    assert grid.array_index_to_chunk_coord(array_shape, (6, 4)) == (2, 2)  # Chunk (2,2): size 2x6
    assert grid.array_index_to_chunk_coord(array_shape, (8, 9)) == (3, 2)  # Chunk (3,2): size 12x6
    assert grid.array_index_to_chunk_coord(array_shape, (19, 9)) == (3, 2)  # Last element


def test_array_index_to_chunk_coord_1d() -> None:
    """Test array_index_to_chunk_coord with 1D array."""
    grid = RectilinearChunkGrid(chunk_shapes=[[5, 10, 15]])
    array_shape = (30,)

    assert grid.array_index_to_chunk_coord(array_shape, (0,)) == (0,)
    assert grid.array_index_to_chunk_coord(array_shape, (4,)) == (0,)
    assert grid.array_index_to_chunk_coord(array_shape, (5,)) == (1,)
    assert grid.array_index_to_chunk_coord(array_shape, (14,)) == (1,)
    assert grid.array_index_to_chunk_coord(array_shape, (15,)) == (2,)
    assert grid.array_index_to_chunk_coord(array_shape, (29,)) == (2,)


def test_array_index_to_chunk_coord_out_of_bounds() -> None:
    """Test that array_index_to_chunk_coord raises IndexError for out of bounds."""
    grid = RectilinearChunkGrid(chunk_shapes=[[2, 3, 1], [4, 2]])
    array_shape = (6, 6)

    with pytest.raises(IndexError, match="array_index\\[0\\] = 6 is out of bounds"):
        grid.array_index_to_chunk_coord(array_shape, (6, 0))

    with pytest.raises(IndexError, match="array_index\\[1\\] = 6 is out of bounds"):
        grid.array_index_to_chunk_coord(array_shape, (0, 6))

    with pytest.raises(IndexError, match="array_index\\[0\\] = -1 is out of bounds"):
        grid.array_index_to_chunk_coord(array_shape, (-1, 0))


# ===================================================================
# Tests for chunks_in_selection() method
# ===================================================================


def test_chunks_in_selection_full_array() -> None:
    """Test chunks_in_selection with full array selection."""
    grid = RectilinearChunkGrid(chunk_shapes=[[2, 2, 2], [3, 3]])
    array_shape = (6, 6)
    selection = (slice(0, 6), slice(0, 6))

    chunks = list(grid.chunks_in_selection(array_shape, selection))
    # Should get all 6 chunks (3 x 2)
    expected = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    assert chunks == expected


def test_chunks_in_selection_single_chunk() -> None:
    """Test chunks_in_selection with selection contained in single chunk."""
    grid = RectilinearChunkGrid(chunk_shapes=[[10, 20], [15, 15]])
    array_shape = (30, 30)

    # Selection entirely within first chunk
    selection = (slice(0, 5), slice(0, 10))
    chunks = list(grid.chunks_in_selection(array_shape, selection))
    assert chunks == [(0, 0)]

    # Selection entirely within last chunk
    selection = (slice(15, 25), slice(20, 28))
    chunks = list(grid.chunks_in_selection(array_shape, selection))
    assert chunks == [(1, 1)]


def test_chunks_in_selection_multi_chunk() -> None:
    """Test chunks_in_selection spanning multiple chunks."""
    grid = RectilinearChunkGrid(chunk_shapes=[[2, 2, 2], [3, 3]])
    array_shape = (6, 6)

    # Selection spanning chunks (1, 5) x (2, 5)
    selection = (slice(1, 5), slice(2, 5))
    chunks = list(grid.chunks_in_selection(array_shape, selection))
    # Should touch chunks at (0,0), (0,1), (1,0), (1,1), (2,0), (2,1)
    expected = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    assert chunks == expected


def test_chunks_in_selection_boundaries() -> None:
    """Test chunks_in_selection at chunk boundaries."""
    grid = RectilinearChunkGrid(chunk_shapes=[[10, 20, 30], [25, 25]])
    array_shape = (60, 50)

    # Selection exactly at chunk boundaries
    selection = (slice(10, 30), slice(0, 25))
    chunks = list(grid.chunks_in_selection(array_shape, selection))
    # Should touch chunks (1,0) and (1,1) but also (2,0) since stop=30 touches it
    expected = [(1, 0)]
    assert chunks == expected

    # Selection crossing multiple boundaries
    selection = (slice(5, 35), slice(20, 30))
    chunks = list(grid.chunks_in_selection(array_shape, selection))
    expected = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    assert chunks == expected


def test_chunks_in_selection_partial() -> None:
    """Test chunks_in_selection with partial chunk overlap."""
    grid = RectilinearChunkGrid(chunk_shapes=[[5, 10, 15], [8, 12]])
    array_shape = (30, 20)

    # Selection partially overlapping chunks
    selection = (slice(3, 18), slice(5, 15))
    chunks = list(grid.chunks_in_selection(array_shape, selection))
    # Touches (0,0), (0,1), (1,0), (1,1), (2,0), (2,1)
    expected = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    assert chunks == expected


def test_chunks_in_selection_empty() -> None:
    """Test chunks_in_selection with empty selection."""
    grid = RectilinearChunkGrid(chunk_shapes=[[2, 2, 2], [3, 3]])
    array_shape = (6, 6)

    # Empty selection (start >= stop)
    selection = (slice(3, 3), slice(0, 6))
    chunks = list(grid.chunks_in_selection(array_shape, selection))
    assert chunks == []

    selection = (slice(5, 2), slice(0, 6))  # start > stop
    chunks = list(grid.chunks_in_selection(array_shape, selection))
    assert chunks == []


def test_chunks_in_selection_with_step_raises() -> None:
    """Test that chunks_in_selection raises ValueError for step != 1."""
    grid = RectilinearChunkGrid(chunk_shapes=[[2, 2, 2], [3, 3]])
    array_shape = (6, 6)

    with pytest.raises(ValueError, match="step=2, only step=1 is supported"):
        list(grid.chunks_in_selection(array_shape, (slice(0, 6, 2), slice(0, 6))))

    with pytest.raises(ValueError, match="step=3, only step=1 is supported"):
        list(grid.chunks_in_selection(array_shape, (slice(0, 6), slice(0, 6, 3))))


# ===================================================================
# Tests for get_chunk_slice() method
# ===================================================================


def test_get_chunk_slice_basic() -> None:
    """Test get_chunk_slice with basic cases."""
    grid = RectilinearChunkGrid(chunk_shapes=[[2, 2, 2], [3, 3]])
    array_shape = (6, 6)

    # First chunk
    assert grid.get_chunk_slice(array_shape, (0, 0)) == (slice(0, 2), slice(0, 3))

    # Middle chunk
    assert grid.get_chunk_slice(array_shape, (1, 1)) == (slice(2, 4), slice(3, 6))

    # Last chunk
    assert grid.get_chunk_slice(array_shape, (2, 1)) == (slice(4, 6), slice(3, 6))


def test_get_chunk_slice_partial_edge() -> None:
    """Test get_chunk_slice with partial edge chunks."""
    grid = RectilinearChunkGrid(chunk_shapes=[[10, 20, 25], [15, 20]])
    array_shape = (55, 35)

    # Full chunks
    assert grid.get_chunk_slice(array_shape, (0, 0)) == (slice(0, 10), slice(0, 15))
    assert grid.get_chunk_slice(array_shape, (1, 1)) == (slice(10, 30), slice(15, 35))

    # Edge chunks (all chunk sizes are exact)
    assert grid.get_chunk_slice(array_shape, (2, 0)) == (slice(30, 55), slice(0, 15))
    assert grid.get_chunk_slice(array_shape, (0, 1)) == (slice(0, 10), slice(15, 35))


def test_get_chunk_slice_irregular() -> None:
    """Test get_chunk_slice with highly irregular chunk sizes."""
    grid = RectilinearChunkGrid(chunk_shapes=[[1, 5, 2, 12], [3, 1, 6]])
    array_shape = (20, 10)

    assert grid.get_chunk_slice(array_shape, (0, 0)) == (slice(0, 1), slice(0, 3))
    assert grid.get_chunk_slice(array_shape, (1, 1)) == (slice(1, 6), slice(3, 4))
    assert grid.get_chunk_slice(array_shape, (2, 2)) == (slice(6, 8), slice(4, 10))
    assert grid.get_chunk_slice(array_shape, (3, 0)) == (slice(8, 20), slice(0, 3))


# ===================================================================
# Tests for chunks_per_dim() and get_chunk_grid_shape() methods
# ===================================================================


def test_chunks_per_dim_2d() -> None:
    """Test chunks_per_dim with 2D array."""
    grid = RectilinearChunkGrid(chunk_shapes=[[10, 20], [5, 5, 5]])
    array_shape = (30, 15)

    assert grid.chunks_per_dim(array_shape, 0) == 2  # 2 chunks along axis 0
    assert grid.chunks_per_dim(array_shape, 1) == 3  # 3 chunks along axis 1


def test_chunks_per_dim_3d() -> None:
    """Test chunks_per_dim with 3D array."""
    grid = RectilinearChunkGrid(chunk_shapes=[[5, 5], [10, 10, 10], [8, 12]])
    array_shape = (10, 30, 20)

    assert grid.chunks_per_dim(array_shape, 0) == 2
    assert grid.chunks_per_dim(array_shape, 1) == 3
    assert grid.chunks_per_dim(array_shape, 2) == 2


def test_get_chunk_grid_shape_various() -> None:
    """Test get_chunk_grid_shape with various shapes."""
    # 2D
    grid = RectilinearChunkGrid(chunk_shapes=[[2, 2, 2], [3, 3]])
    assert grid.get_chunk_grid_shape((6, 6)) == (3, 2)

    # 3D
    grid = RectilinearChunkGrid(chunk_shapes=[[5, 5], [10, 10, 10], [8, 12]])
    assert grid.get_chunk_grid_shape((10, 30, 20)) == (2, 3, 2)

    # 1D
    grid = RectilinearChunkGrid(chunk_shapes=[[1, 2, 3, 4]])
    assert grid.get_chunk_grid_shape((10,)) == (4,)

    # Irregular
    grid = RectilinearChunkGrid(chunk_shapes=[[10, 20, 30], [25, 25]])
    assert grid.get_chunk_grid_shape((60, 50)) == (3, 2)


# ===================================================================
# Edge case and integration tests
# ===================================================================


def test_rectilinear_1d_array() -> None:
    """Test RectilinearChunkGrid with 1D arrays."""
    grid = RectilinearChunkGrid(chunk_shapes=[[5, 10, 15]])
    array_shape = (30,)

    # Test basic properties
    assert grid.get_nchunks(array_shape) == 3
    assert grid.get_chunk_grid_shape(array_shape) == (3,)

    # Test chunk operations
    assert grid.get_chunk_start(array_shape, (0,)) == (0,)
    assert grid.get_chunk_start(array_shape, (1,)) == (5,)
    assert grid.get_chunk_start(array_shape, (2,)) == (15,)

    assert grid.get_chunk_shape(array_shape, (0,)) == (5,)
    assert grid.get_chunk_shape(array_shape, (1,)) == (10,)
    assert grid.get_chunk_shape(array_shape, (2,)) == (15,)

    # Test all coords
    coords = list(grid.all_chunk_coords(array_shape))
    assert coords == [(0,), (1,), (2,)]


def test_rectilinear_3d_array() -> None:
    """Test RectilinearChunkGrid with 3D arrays."""
    grid = RectilinearChunkGrid(chunk_shapes=[[5, 5], [10, 10], [4, 6]])
    array_shape = (10, 20, 10)

    # Test basic properties
    assert grid.get_nchunks(array_shape) == 8  # 2 x 2 x 2
    assert grid.get_chunk_grid_shape(array_shape) == (2, 2, 2)

    # Test some chunk operations
    assert grid.get_chunk_start(array_shape, (0, 0, 0)) == (0, 0, 0)
    assert grid.get_chunk_start(array_shape, (1, 1, 1)) == (5, 10, 4)

    assert grid.get_chunk_shape(array_shape, (0, 0, 0)) == (5, 10, 4)
    assert grid.get_chunk_shape(array_shape, (1, 1, 1)) == (5, 10, 6)

    # Test index to coord
    assert grid.array_index_to_chunk_coord(array_shape, (0, 0, 0)) == (0, 0, 0)
    assert grid.array_index_to_chunk_coord(array_shape, (7, 15, 8)) == (1, 1, 1)


def test_rectilinear_4d_array() -> None:
    """Test RectilinearChunkGrid with 4D arrays."""
    grid = RectilinearChunkGrid(chunk_shapes=[[3, 3], [4, 4], [5], [2, 3]])
    array_shape = (6, 8, 5, 5)

    assert grid.get_nchunks(array_shape) == 8  # 2 x 2 x 1 x 2
    assert grid.get_chunk_grid_shape(array_shape) == (2, 2, 1, 2)

    # Test index to coord in 4D
    assert grid.array_index_to_chunk_coord(array_shape, (0, 0, 0, 0)) == (0, 0, 0, 0)
    assert grid.array_index_to_chunk_coord(array_shape, (5, 7, 4, 4)) == (1, 1, 0, 1)


def test_out_of_bounds_chunk_coords() -> None:
    """Test that out-of-bounds chunk coordinates raise IndexError."""
    grid = RectilinearChunkGrid(chunk_shapes=[[2, 2, 2], [3, 3]])
    array_shape = (6, 6)

    # Out of bounds chunk coordinates
    with pytest.raises(IndexError, match="chunk_coord\\[0\\] = 3 is out of bounds"):
        grid.get_chunk_start(array_shape, (3, 0))

    with pytest.raises(IndexError, match="chunk_coord\\[1\\] = 2 is out of bounds"):
        grid.get_chunk_shape(array_shape, (0, 2))

    with pytest.raises(IndexError, match="chunk_coord\\[0\\] = -1 is out of bounds"):
        grid.get_chunk_slice(array_shape, (-1, 0))


def test_dimension_mismatch_raises() -> None:
    """Test that dimension mismatches raise ValueError."""
    grid = RectilinearChunkGrid(chunk_shapes=[[2, 2, 2], [3, 3]])

    # 1D shape with 2D chunks
    with pytest.raises(ValueError, match="array_shape has 1 dimensions"):
        grid.all_chunk_coords((6,))

    # 3D shape with 2D chunks
    with pytest.raises(ValueError, match="array_shape has 3 dimensions"):
        grid.get_nchunks((6, 6, 6))

    # Wrong shape sum
    with pytest.raises(
        ValueError, match="Sum of chunk sizes along axis 0 is 6 but array shape is 10"
    ):
        grid.get_chunk_start((10, 6), (0, 0))


def test_invalid_chunk_coordinates_raise() -> None:
    """Test that invalid chunk coordinates raise appropriate errors."""
    grid = RectilinearChunkGrid(chunk_shapes=[[2, 2, 2], [3, 3]])
    array_shape = (6, 6)

    # Negative chunk coordinates
    with pytest.raises(IndexError):
        grid.get_chunk_start(array_shape, (-1, 0))

    # Chunk coordinates beyond grid
    with pytest.raises(IndexError):
        grid.get_chunk_start(array_shape, (3, 0))  # Only 3 chunks in dim 0 (0,1,2)

    with pytest.raises(IndexError):
        grid.get_chunk_start(array_shape, (0, 2))  # Only 2 chunks in dim 1 (0,1)
