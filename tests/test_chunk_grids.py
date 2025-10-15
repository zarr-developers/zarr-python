from typing import Any

import numpy as np
import pytest

from zarr.core.chunk_grids import (
    RectilinearChunkGrid,
    _expand_run_length_encoding,
    _guess_chunks,
    _parse_chunk_shapes,
    normalize_chunks,
)


@pytest.mark.parametrize(
    "shape", [(0,), (0,) * 2, (1, 2, 0, 4, 5), (10, 0), (10,), (100,) * 3, (1000000,), (10000,) * 2]
)
@pytest.mark.parametrize("itemsize", [1, 2, 4])
def test_guess_chunks(shape: tuple[int, ...], itemsize: int) -> None:
    chunks = _guess_chunks(shape, itemsize)
    chunk_size = np.prod(chunks) * itemsize
    assert isinstance(chunks, tuple)
    assert len(chunks) == len(shape)
    assert chunk_size < (64 * 1024 * 1024)
    # doesn't make any sense to allow chunks to have zero length dimension
    assert all(0 < c <= max(s, 1) for c, s in zip(chunks, shape, strict=False))


@pytest.mark.parametrize(
    ("chunks", "shape", "typesize", "expected"),
    [
        ((10,), (100,), 1, (10,)),
        ([10], (100,), 1, (10,)),
        (10, (100,), 1, (10,)),
        ((10, 10), (100, 10), 1, (10, 10)),
        (10, (100, 10), 1, (10, 10)),
        ((10, None), (100, 10), 1, (10, 10)),
        (30, (100, 20, 10), 1, (30, 30, 30)),
        ((30,), (100, 20, 10), 1, (30, 20, 10)),
        ((30, None), (100, 20, 10), 1, (30, 20, 10)),
        ((30, None, None), (100, 20, 10), 1, (30, 20, 10)),
        ((30, 20, None), (100, 20, 10), 1, (30, 20, 10)),
        ((30, 20, 10), (100, 20, 10), 1, (30, 20, 10)),
        # auto chunking
        (None, (100,), 1, (100,)),
        (-1, (100,), 1, (100,)),
        ((30, -1, None), (100, 20, 10), 1, (30, 20, 10)),
    ],
)
def test_normalize_chunks(
    chunks: Any, shape: tuple[int, ...], typesize: int, expected: tuple[int, ...]
) -> None:
    assert expected == normalize_chunks(chunks, shape, typesize)


def test_normalize_chunks_errors() -> None:
    with pytest.raises(ValueError):
        normalize_chunks("foo", (100,), 1)
    with pytest.raises(ValueError):
        normalize_chunks((100, 10), (100,), 1)


# RectilinearChunkGrid tests


class TestExpandRunLengthEncoding:
    """Tests for _expand_run_length_encoding function"""

    def test_simple_integers(self) -> None:
        """Test with simple integer values"""
        assert _expand_run_length_encoding([2, 3, 1]) == (2, 3, 1)

    def test_single_run_length(self) -> None:
        """Test with single run-length encoded value"""
        assert _expand_run_length_encoding([[2, 3]]) == (2, 2, 2)  # type: ignore[list-item]

    def test_mixed(self) -> None:
        """Test with mix of integers and run-length encoded values"""
        assert _expand_run_length_encoding([1, [2, 1], 3]) == (1, 2, 3)  # type: ignore[list-item]
        assert _expand_run_length_encoding([[1, 3], 3]) == (1, 1, 1, 3)  # type: ignore[list-item]

    def test_zero_count(self) -> None:
        """Test with zero count in run-length encoding"""
        assert _expand_run_length_encoding([[2, 0], 3]) == (3,)  # type: ignore[list-item]

    def test_empty(self) -> None:
        """Test with empty input"""
        assert _expand_run_length_encoding([]) == ()

    def test_invalid_run_length_type(self) -> None:
        """Test error handling for invalid run-length encoding types"""
        with pytest.raises(TypeError, match="must be \\[int, int\\]"):
            _expand_run_length_encoding([["a", 2]])  # type: ignore[list-item]

    def test_invalid_item_type(self) -> None:
        """Test error handling for invalid item types"""
        with pytest.raises(TypeError, match="must be int or \\[int, int\\]"):
            _expand_run_length_encoding(["string"])  # type: ignore[list-item]

    def test_negative_count(self) -> None:
        """Test error handling for negative count"""
        with pytest.raises(ValueError, match="must be non-negative"):
            _expand_run_length_encoding([[2, -1]])  # type: ignore[list-item]


class TestParseChunkShapes:
    """Tests for _parse_chunk_shapes function"""

    def test_simple_2d(self) -> None:
        """Test parsing simple 2D chunk shapes"""
        result = _parse_chunk_shapes([[2, 2, 2], [3, 3]])
        assert result == ((2, 2, 2), (3, 3))

    def test_with_run_length_encoding(self) -> None:
        """Test parsing with run-length encoding"""
        result = _parse_chunk_shapes([[[2, 3]], [[1, 6]]])  # type: ignore[list-item]
        assert result == ((2, 2, 2), (1, 1, 1, 1, 1, 1))

    def test_mixed_encoding(self) -> None:
        """Test parsing with mixed encoding styles"""
        result = _parse_chunk_shapes(
            [
                [1, [2, 1], 3],  # type: ignore[list-item]
                [[1, 3], 3],  # type: ignore[list-item]
            ]
        )
        assert result == ((1, 2, 3), (1, 1, 1, 3))

    def test_invalid_type(self) -> None:
        """Test error handling for invalid types"""
        with pytest.raises(TypeError, match="must be a sequence"):
            _parse_chunk_shapes("not a sequence")  # type: ignore[arg-type]

    def test_invalid_axis_type(self) -> None:
        """Test error handling for invalid axis type"""
        with pytest.raises(TypeError, match="chunk_shapes\\[0\\] must be a sequence"):
            _parse_chunk_shapes([123])  # type: ignore[list-item]


class TestRectilinearChunkGrid:
    """Tests for RectilinearChunkGrid class"""

    def test_init_simple(self) -> None:
        """Test simple initialization"""
        grid = RectilinearChunkGrid(chunk_shapes=[[2, 2, 2], [3, 3]])
        assert grid.chunk_shapes == ((2, 2, 2), (3, 3))

    def test_init_validation_non_positive(self) -> None:
        """Test validation rejects non-positive chunk sizes"""
        with pytest.raises(ValueError, match="must be positive"):
            RectilinearChunkGrid(chunk_shapes=[[2, 0, 2], [3, 3]])

    def test_init_validation_non_integer(self) -> None:
        """Test validation rejects non-integer chunk sizes"""
        with pytest.raises(TypeError, match="must be an int"):
            RectilinearChunkGrid(chunk_shapes=[[2, 2.5, 2], [3, 3]])  # type: ignore[list-item]

    def test_from_dict_spec_example(self) -> None:
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

    def test_from_dict_invalid_kind(self) -> None:
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

    def test_from_dict_missing_chunk_shapes(self) -> None:
        """Test error handling for missing chunk_shapes"""
        metadata = {
            "name": "rectilinear",
            "configuration": {
                "kind": "inline",
            },
        }
        with pytest.raises(ValueError, match="must contain 'chunk_shapes'"):
            RectilinearChunkGrid._from_dict(metadata)  # type: ignore[arg-type]

    def test_to_dict(self) -> None:
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

    def test_all_chunk_coords_2d(self) -> None:
        """Test generating all chunk coordinates for 2D array"""
        grid = RectilinearChunkGrid(chunk_shapes=[[2, 2, 2], [3, 3]])
        array_shape = (6, 6)

        coords = list(grid.all_chunk_coords(array_shape))

        # Should have 3 chunks along first axis, 2 along second
        assert len(coords) == 6
        assert coords == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]

    def test_all_chunk_coords_validation_mismatch(self) -> None:
        """Test validation when array shape doesn't match chunk shapes"""
        grid = RectilinearChunkGrid(chunk_shapes=[[2, 2, 2], [3, 3]])

        # Wrong sum
        with pytest.raises(ValueError, match="Sum of chunk sizes"):
            list(grid.all_chunk_coords((7, 6)))

        # Wrong dimensions
        with pytest.raises(ValueError, match="dimensions"):
            list(grid.all_chunk_coords((6, 6, 6)))

    def test_get_nchunks(self) -> None:
        """Test getting total number of chunks"""
        grid = RectilinearChunkGrid(chunk_shapes=[[2, 2, 2], [3, 3], [1, 1, 1, 1, 1, 1]])
        array_shape = (6, 6, 6)

        nchunks = grid.get_nchunks(array_shape)

        # 3 chunks x 2 chunks x 6 chunks = 36 chunks
        assert nchunks == 36

    def test_get_nchunks_validation(self) -> None:
        """Test validation in get_nchunks"""
        grid = RectilinearChunkGrid(chunk_shapes=[[2, 2, 2], [3, 3]])

        # Wrong sum
        with pytest.raises(ValueError, match="Sum of chunk sizes"):
            grid.get_nchunks((7, 6))

        # Wrong dimensions
        with pytest.raises(ValueError, match="dimensions"):
            grid.get_nchunks((6, 6, 6))

    def test_roundtrip(self) -> None:
        """Test that to_dict and from_dict are inverses"""
        original = RectilinearChunkGrid(chunk_shapes=[[1, 2, 3], [4, 5]])
        metadata = original.to_dict()
        reconstructed = RectilinearChunkGrid._from_dict(metadata)

        assert reconstructed.chunk_shapes == original.chunk_shapes
