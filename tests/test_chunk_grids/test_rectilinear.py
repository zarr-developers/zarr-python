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

# Run-length encoding tests


def test_expand_run_length_encoding_simple_integers() -> None:
    """Test with simple integer values"""
    assert _expand_run_length_encoding([2, 3, 1]) == (2, 3, 1)


def test_expand_run_length_encoding_single_run_length() -> None:
    """Test with single run-length encoded value"""
    assert _expand_run_length_encoding([[2, 3]]) == (2, 2, 2)  # type: ignore[list-item]


def test_expand_run_length_encoding_mixed() -> None:
    """Test with mix of integers and run-length encoded values"""
    assert _expand_run_length_encoding([1, [2, 1], 3]) == (1, 2, 3)  # type: ignore[list-item]
    assert _expand_run_length_encoding([[1, 3], 3]) == (1, 1, 1, 3)  # type: ignore[list-item]


def test_expand_run_length_encoding_zero_count() -> None:
    """Test with zero count in run-length encoding"""
    assert _expand_run_length_encoding([[2, 0], 3]) == (3,)  # type: ignore[list-item]


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
        _expand_run_length_encoding([[2, -1]])  # type: ignore[list-item]


# Parse chunk shapes tests


def test_parse_chunk_shapes_simple_2d() -> None:
    """Test parsing simple 2D chunk shapes"""
    result = _parse_chunk_shapes([[2, 2, 2], [3, 3]])
    assert result == ((2, 2, 2), (3, 3))


def test_parse_chunk_shapes_with_run_length_encoding() -> None:
    """Test parsing with run-length encoding"""
    result = _parse_chunk_shapes([[[2, 3]], [[1, 6]]])  # type: ignore[list-item]
    assert result == ((2, 2, 2), (1, 1, 1, 1, 1, 1))


def test_parse_chunk_shapes_mixed_encoding() -> None:
    """Test parsing with mixed encoding styles"""
    result = _parse_chunk_shapes(
        [
            [1, [2, 1], 3],  # type: ignore[list-item]
            [[1, 3], 3],  # type: ignore[list-item]
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


# RectilinearChunkGrid class tests


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


# RLE compression tests


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
    assert metadata["configuration"]["chunk_shapes"] == [[[10, 6]], [[5, 5]]]  # type: ignore[call-overload, index]

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


# RLE in top-level API tests


async def test_api_create_array_with_rle_simple() -> None:
    """Test creating an array using simple RLE format."""
    store = MemoryStore()

    # [[10, 6]] means 6 chunks of size 10 each (RLE format)
    arr = await zarr.api.asynchronous.create_array(
        store=store,
        shape=(60, 60),
        chunks=[[[10, 6]], [[10, 6]]],  # type: ignore[list-item]
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
        chunks=[[[2, 3]], [1, [2, 1], 3]],  # type: ignore[list-item]
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
        chunks=[[[10, 10]], [[10, 5]]],  # type: ignore[list-item]
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
            [[2, 3]],  # type: ignore[list-item]
            [[1, 6]],  # type: ignore[list-item]
            [1, [2, 1], 3],  # type: ignore[list-item]
            [[1, 3], 1],  # type: ignore[list-item]
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
        chunks=[[[10, 3]], [[10, 4]]],  # type: ignore[list-item]
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
        chunks=[[[5, 0], 5, 5], [[5, 2]]],  # type: ignore[list-item]
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
        chunks=[[[10, 5]], [[10, 5]]],  # type: ignore[list-item]
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
        chunks=[[[10, 100]], [[10, 100]]],  # type: ignore[list-item]
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
        chunks=[[[10, 5], 50], [25, 30, 20, 25]],  # type: ignore[list-item]
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
            chunks=[[[10, 6]], [[10, 6]]],  # type: ignore[list-item]
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
