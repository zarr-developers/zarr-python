"""Tests for RectilinearChunkGrid implementation."""

import pytest

from zarr.core.chunk_grids import (
    RectilinearChunkGrid,
    _expand_run_length_encoding,
    _parse_chunk_shapes,
)

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
    """Test serialization to dict"""
    grid = RectilinearChunkGrid(chunk_shapes=[[2, 2, 2], [3, 3]])
    result = grid.to_dict()

    assert result == {
        "name": "rectilinear",
        "configuration": {
            "kind": "inline",
            "chunk_shapes": [[2, 2, 2], [3, 3]],
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
