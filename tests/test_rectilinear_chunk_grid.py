"""
Comprehensive test suite for RectilinearChunkGrid functionality.

This test suite is written ahead of implementation to define expected behaviors
for variable-sized chunk grids.
"""

import numpy as np
import pytest

from zarr.core.chunk_grids import ChunkGrid, RectilinearChunkGrid


class TestRectilinearChunkGridBasics:
    """Test basic RectilinearChunkGrid functionality"""

    def test_simple_2d_grid(self) -> None:
        """Test a simple 2D rectilinear grid"""
        grid = RectilinearChunkGrid(chunk_shapes=[[2, 3, 1], [4, 2]])
        array_shape = (6, 6)

        # Should have 3 chunks along axis 0, 2 chunks along axis 1
        assert grid.get_nchunks(array_shape) == 6

        # All chunk coordinates
        coords = list(grid.all_chunk_coords(array_shape))
        assert len(coords) == 6
        assert (0, 0) in coords
        assert (2, 1) in coords

    def test_from_dict_integration(self) -> None:
        """Test that RectilinearChunkGrid works with ChunkGrid.from_dict"""
        metadata = {
            "name": "rectilinear",
            "configuration": {
                "kind": "inline",
                "chunk_shapes": [[2, 4], [3, 3]],
            },
        }

        grid = ChunkGrid.from_dict(metadata)  # type: ignore[arg-type]
        assert isinstance(grid, RectilinearChunkGrid)
        assert grid.chunk_shapes == ((2, 4), (3, 3))


class TestChunkBoundaries:
    """Test computing chunk boundaries and slices"""

    def test_get_chunk_slice_2d(self) -> None:
        """Test getting the slice for a specific chunk in 2D"""
        grid = RectilinearChunkGrid(chunk_shapes=[[2, 2, 2], [3, 3]])
        array_shape = (6, 6)

        # Chunk (0, 0): rows [0:2], cols [0:3]
        slice_00 = grid.get_chunk_slice(array_shape, (0, 0))
        assert slice_00 == (slice(0, 2), slice(0, 3))

        # Chunk (1, 0): rows [2:4], cols [0:3]
        slice_10 = grid.get_chunk_slice(array_shape, (1, 0))
        assert slice_10 == (slice(2, 4), slice(0, 3))

        # Chunk (2, 1): rows [4:6], cols [3:6]
        slice_21 = grid.get_chunk_slice(array_shape, (2, 1))
        assert slice_21 == (slice(4, 6), slice(3, 6))

    def test_get_chunk_shape_2d(self) -> None:
        """Test getting the shape of a specific chunk"""
        grid = RectilinearChunkGrid(chunk_shapes=[[2, 3, 1], [4, 2]])
        array_shape = (6, 6)

        # Chunk (0, 0): shape (2, 4)
        assert grid.get_chunk_shape(array_shape, (0, 0)) == (2, 4)

        # Chunk (1, 0): shape (3, 4)
        assert grid.get_chunk_shape(array_shape, (1, 0)) == (3, 4)

        # Chunk (2, 1): shape (1, 2)
        assert grid.get_chunk_shape(array_shape, (2, 1)) == (1, 2)

    def test_get_chunk_start_3d(self) -> None:
        """Test getting the start position of a chunk in 3D"""
        grid = RectilinearChunkGrid(chunk_shapes=[[2, 2], [3, 3], [1, 2, 1]])
        array_shape = (4, 6, 4)

        # Chunk (0, 0, 0): starts at (0, 0, 0)
        assert grid.get_chunk_start(array_shape, (0, 0, 0)) == (0, 0, 0)

        # Chunk (1, 1, 2): starts at (2, 3, 3)
        assert grid.get_chunk_start(array_shape, (1, 1, 2)) == (2, 3, 3)

    def test_chunk_boundaries_all_chunks(self) -> None:
        """Test that all chunks tile the array without gaps or overlaps"""
        grid = RectilinearChunkGrid(chunk_shapes=[[2, 3, 1], [4, 2]])
        array_shape = (6, 6)

        # Collect all indices covered by chunks
        covered = np.zeros(array_shape, dtype=bool)

        for chunk_coord in grid.all_chunk_coords(array_shape):
            chunk_slice = grid.get_chunk_slice(array_shape, chunk_coord)
            chunk_covered = np.zeros(array_shape, dtype=bool)
            chunk_covered[chunk_slice] = True

            # Check no overlap
            assert not np.any(covered & chunk_covered), f"Overlap at chunk {chunk_coord}"

            covered |= chunk_covered

        # Check complete coverage
        assert np.all(covered), "Not all array elements are covered by chunks"


class TestArrayIndexToChunk:
    """Test mapping array indices to chunk coordinates"""

    def test_index_to_chunk_coord_2d(self) -> None:
        """Test finding which chunk contains a given array index"""
        grid = RectilinearChunkGrid(chunk_shapes=[[2, 3, 1], [4, 2]])
        array_shape = (6, 6)

        # Index (0, 0) is in chunk (0, 0)
        assert grid.array_index_to_chunk_coord(array_shape, (0, 0)) == (0, 0)

        # Index (1, 3) is in chunk (0, 0)
        assert grid.array_index_to_chunk_coord(array_shape, (1, 3)) == (0, 0)

        # Index (2, 0) is in chunk (1, 0)
        assert grid.array_index_to_chunk_coord(array_shape, (2, 0)) == (1, 0)

        # Index (5, 5) is in chunk (2, 1)
        assert grid.array_index_to_chunk_coord(array_shape, (5, 5)) == (2, 1)

    def test_index_to_chunk_coord_3d(self) -> None:
        """Test array index to chunk coordinate in 3D"""
        grid = RectilinearChunkGrid(chunk_shapes=[[2, 2], [3, 3], [1, 2, 1]])
        array_shape = (4, 6, 4)

        # Index (0, 0, 0) is in chunk (0, 0, 0)
        assert grid.array_index_to_chunk_coord(array_shape, (0, 0, 0)) == (0, 0, 0)

        # Index (3, 5, 3) is in chunk (1, 1, 2)
        assert grid.array_index_to_chunk_coord(array_shape, (3, 5, 3)) == (1, 1, 2)

    def test_all_indices_map_correctly(self) -> None:
        """Test that all indices map to the correct chunk"""
        grid = RectilinearChunkGrid(chunk_shapes=[[2, 2, 2], [3, 3]])
        array_shape = (6, 6)

        for i in range(array_shape[0]):
            for j in range(array_shape[1]):
                chunk_coord = grid.array_index_to_chunk_coord(array_shape, (i, j))
                chunk_slice = grid.get_chunk_slice(array_shape, chunk_coord)

                # Verify the index is within the chunk slice
                assert chunk_slice[0].start <= i < chunk_slice[0].stop
                assert chunk_slice[1].start <= j < chunk_slice[1].stop


class TestChunkIterators:
    """Test iterating over chunks"""

    def test_iter_chunks_in_selection_2d(self) -> None:
        """Test getting chunks that intersect with a selection"""
        grid = RectilinearChunkGrid(chunk_shapes=[[2, 2, 2], [3, 3]])
        array_shape = (6, 6)

        # Selection that spans multiple chunks: [1:5, 2:5]
        # Should intersect chunks: (0,0), (0,1), (1,0), (1,1), (2,0), (2,1)
        selection = (slice(1, 5), slice(2, 5))
        chunks = list(grid.chunks_in_selection(array_shape, selection))

        # Should have 6 chunks
        assert len(chunks) == 6
        assert (0, 0) in chunks
        assert (1, 1) in chunks
        assert (2, 1) in chunks

    def test_iter_chunks_single_chunk(self) -> None:
        """Test selection within a single chunk"""
        grid = RectilinearChunkGrid(chunk_shapes=[[2, 3, 1], [4, 2]])
        array_shape = (6, 6)

        # Selection within chunk (1, 0): [2:4, 1:3]
        selection = (slice(2, 4), slice(1, 3))
        chunks = list(grid.chunks_in_selection(array_shape, selection))

        # Should only touch chunk (1, 0)
        assert len(chunks) == 1
        assert chunks[0] == (1, 0)


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_single_chunk_per_axis(self) -> None:
        """Test grid with single chunk per axis"""
        grid = RectilinearChunkGrid(chunk_shapes=[[10], [10]])
        array_shape = (10, 10)

        assert grid.get_nchunks(array_shape) == 1
        assert list(grid.all_chunk_coords(array_shape)) == [(0, 0)]
        assert grid.get_chunk_shape(array_shape, (0, 0)) == (10, 10)

    def test_many_small_chunks(self) -> None:
        """Test grid with many small chunks"""
        # 10 chunks of size 1 each
        grid = RectilinearChunkGrid(chunk_shapes=[[1] * 10, [1] * 10])
        array_shape = (10, 10)

        assert grid.get_nchunks(array_shape) == 100
        assert grid.get_chunk_shape(array_shape, (5, 5)) == (1, 1)

    def test_uneven_chunks(self) -> None:
        """Test grid with very uneven chunk sizes"""
        grid = RectilinearChunkGrid(chunk_shapes=[[1, 5, 10], [2, 14]])
        array_shape = (16, 16)

        assert grid.get_nchunks(array_shape) == 6
        assert grid.get_chunk_shape(array_shape, (0, 0)) == (1, 2)
        assert grid.get_chunk_shape(array_shape, (2, 1)) == (10, 14)

    def test_1d_array(self) -> None:
        """Test rectilinear grid with 1D array"""
        grid = RectilinearChunkGrid(chunk_shapes=[[2, 3, 1]])
        array_shape = (6,)

        assert grid.get_nchunks(array_shape) == 3
        assert grid.get_chunk_slice(array_shape, (0,)) == (slice(0, 2),)
        assert grid.get_chunk_slice(array_shape, (1,)) == (slice(2, 5),)
        assert grid.get_chunk_slice(array_shape, (2,)) == (slice(5, 6),)

    def test_high_dimensional(self) -> None:
        """Test rectilinear grid with 4D array"""
        grid = RectilinearChunkGrid(
            chunk_shapes=[
                [2, 2],  # axis 0: 2 chunks
                [3, 3],  # axis 1: 2 chunks
                [1, 1, 1, 1],  # axis 2: 4 chunks
                [5],  # axis 3: 1 chunk
            ]
        )
        array_shape = (4, 6, 4, 5)

        assert grid.get_nchunks(array_shape) == 16  # 2*2*4*1
        assert grid.get_chunk_shape(array_shape, (0, 0, 0, 0)) == (2, 3, 1, 5)
        assert grid.get_chunk_shape(array_shape, (1, 1, 3, 0)) == (2, 3, 1, 5)


class TestInvalidUsage:
    """Test error handling for invalid usage"""

    def test_invalid_chunk_coord(self) -> None:
        """Test error when requesting invalid chunk coordinate"""
        grid = RectilinearChunkGrid(chunk_shapes=[[2, 2, 2], [3, 3]])
        array_shape = (6, 6)

        # Chunk coordinate out of bounds
        with pytest.raises((IndexError, ValueError)):
            grid.get_chunk_slice(array_shape, (3, 0))

        with pytest.raises((IndexError, ValueError)):
            grid.get_chunk_slice(array_shape, (0, 2))

    def test_invalid_array_index(self) -> None:
        """Test error when array index is out of bounds"""
        grid = RectilinearChunkGrid(chunk_shapes=[[2, 2, 2], [3, 3]])
        array_shape = (6, 6)

        # Array index out of bounds
        with pytest.raises((IndexError, ValueError)):
            grid.array_index_to_chunk_coord(array_shape, (6, 0))

        with pytest.raises((IndexError, ValueError)):
            grid.array_index_to_chunk_coord(array_shape, (0, 6))


class TestChunkGridShape:
    """Test computing the shape of the chunk grid itself"""

    def test_chunk_grid_shape_2d(self) -> None:
        """Test getting the shape of the chunk grid (number of chunks per axis)"""
        grid = RectilinearChunkGrid(chunk_shapes=[[2, 2, 2], [3, 3]])
        array_shape = (6, 6)

        # 3 chunks along axis 0, 2 chunks along axis 1
        assert grid.get_chunk_grid_shape(array_shape) == (3, 2)

    def test_chunk_grid_shape_3d(self) -> None:
        """Test chunk grid shape in 3D"""
        grid = RectilinearChunkGrid(chunk_shapes=[[2, 2], [3, 3], [1, 2, 1]])
        array_shape = (4, 6, 4)

        # 2 chunks along axis 0, 2 along axis 1, 3 along axis 2
        assert grid.get_chunk_grid_shape(array_shape) == (2, 2, 3)


class TestSpecialCases:
    """Test special cases from the specification"""

    def test_spec_example_array(self) -> None:
        """Test using the exact example from the specification"""
        grid = RectilinearChunkGrid(
            chunk_shapes=[
                [2, 2, 2],  # axis 0: 3 chunks
                [1, 1, 1, 1, 1, 1],  # axis 1: 6 chunks
                [1, 2, 3],  # axis 2: 3 chunks
                [1, 1, 1, 3],  # axis 3: 4 chunks
                [6],  # axis 4: 1 chunk
            ]
        )
        array_shape = (6, 6, 6, 6, 6)

        # Total chunks: 3*6*3*4*1 = 216
        assert grid.get_nchunks(array_shape) == 216

        # Test specific chunk shapes
        assert grid.get_chunk_shape(array_shape, (0, 0, 0, 0, 0)) == (2, 1, 1, 1, 6)
        assert grid.get_chunk_shape(array_shape, (1, 2, 1, 2, 0)) == (2, 1, 2, 1, 6)
        assert grid.get_chunk_shape(array_shape, (2, 5, 2, 3, 0)) == (2, 1, 3, 3, 6)

        # Test chunk positions
        assert grid.get_chunk_start(array_shape, (0, 0, 0, 0, 0)) == (0, 0, 0, 0, 0)
        assert grid.get_chunk_start(array_shape, (1, 2, 1, 2, 0)) == (2, 2, 1, 2, 0)
        assert grid.get_chunk_start(array_shape, (2, 5, 2, 3, 0)) == (4, 5, 3, 3, 0)


class TestComparisonsWithRegularGrid:
    """Test that RectilinearChunkGrid can represent regular grids"""

    def test_equivalent_to_regular_grid(self) -> None:
        """Test that uniform chunks behave like RegularChunkGrid"""
        from zarr.core.chunk_grids import RegularChunkGrid

        # Create equivalent grids
        rectilinear = RectilinearChunkGrid(chunk_shapes=[[2, 2, 2], [3, 3]])
        regular = RegularChunkGrid(chunk_shape=(2, 3))

        array_shape = (6, 6)

        # Should have same number of chunks
        assert rectilinear.get_nchunks(array_shape) == regular.get_nchunks(array_shape)

        # Should have same chunk coordinates
        rect_coords = set(rectilinear.all_chunk_coords(array_shape))
        reg_coords = set(regular.all_chunk_coords(array_shape))
        assert rect_coords == reg_coords

        # Should have same chunk shapes for all chunks
        for coord in rect_coords:
            assert rectilinear.get_chunk_shape(array_shape, coord) == (2, 3)


class TestRoundTrip:
    """Test serialization round-trips with full grid functionality"""

    def test_roundtrip_preserves_behavior(self) -> None:
        """Test that to_dict/from_dict preserves grid behavior"""
        original = RectilinearChunkGrid(chunk_shapes=[[2, 3, 1], [4, 2]])
        array_shape = (6, 6)

        # Serialize and deserialize
        metadata = original.to_dict()
        reconstructed = RectilinearChunkGrid._from_dict(metadata)

        # Should have same behavior
        assert reconstructed.get_nchunks(array_shape) == original.get_nchunks(array_shape)
        assert list(reconstructed.all_chunk_coords(array_shape)) == list(
            original.all_chunk_coords(array_shape)
        )

        # Test specific chunk operations
        for coord in original.all_chunk_coords(array_shape):
            assert reconstructed.get_chunk_shape(array_shape, coord) == original.get_chunk_shape(
                array_shape, coord
            )
            assert reconstructed.get_chunk_slice(array_shape, coord) == original.get_chunk_slice(
                array_shape, coord
            )
