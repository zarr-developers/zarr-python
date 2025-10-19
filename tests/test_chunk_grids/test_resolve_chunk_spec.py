"""Tests for the resolve_chunk_spec() function."""

import pytest

from zarr.core.chunk_grids import (
    RectilinearChunkGrid,
    RegularChunkGrid,
    ResolvedChunkSpec,
    resolve_chunk_spec,
)

# Basic functionality tests


def test_resolve_chunk_spec_regular_chunks_no_sharding() -> None:
    """Test regular chunks without sharding."""
    spec = resolve_chunk_spec(
        chunks=(10, 10),
        shards=None,
        shape=(100, 100),
        dtype_itemsize=4,
        zarr_format=3,
    )
    assert spec.chunks == (10, 10)
    assert spec.shards is None
    assert isinstance(spec.chunk_grid, RegularChunkGrid)


def test_resolve_chunk_spec_regular_chunks_with_sharding() -> None:
    """Test regular chunks with sharding."""
    spec = resolve_chunk_spec(
        chunks=(5, 5),
        shards=(20, 20),
        shape=(100, 100),
        dtype_itemsize=4,
        zarr_format=3,
    )
    assert spec.chunks == (5, 5)
    assert spec.shards == (20, 20)
    assert spec.chunk_grid is None  # sharding uses init_array's _auto_partition


def test_resolve_chunk_spec_auto_chunks_no_sharding() -> None:
    """Test auto chunking without sharding."""
    spec = resolve_chunk_spec(
        chunks="auto",
        shards=None,
        shape=(100, 100),
        dtype_itemsize=4,
        zarr_format=3,
    )
    assert isinstance(spec.chunks, tuple)
    assert len(spec.chunks) == 2
    assert spec.shards is None
    assert isinstance(spec.chunk_grid, RegularChunkGrid)


def test_resolve_chunk_spec_auto_chunks_with_sharding() -> None:
    """Test auto chunking with sharding."""
    spec = resolve_chunk_spec(
        chunks="auto",
        shards=(20, 20),
        shape=(100, 100),
        dtype_itemsize=4,
        zarr_format=3,
    )
    assert spec.chunks == "auto"
    assert spec.shards == (20, 20)
    assert spec.chunk_grid is None


def test_resolve_chunk_spec_single_int_chunks() -> None:
    """Test single integer for chunks (applied to all dimensions)."""
    spec = resolve_chunk_spec(
        chunks=10,
        shards=None,
        shape=(100, 100),
        dtype_itemsize=4,
        zarr_format=3,
    )
    assert spec.chunks == (10, 10)
    assert spec.shards is None
    assert isinstance(spec.chunk_grid, RegularChunkGrid)


def test_resolve_chunk_spec_variable_chunks_no_sharding() -> None:
    """Test variable chunks (RectilinearChunkGrid) without sharding."""
    spec = resolve_chunk_spec(
        chunks=[[10, 20, 30], [25, 25, 25, 25]],
        shards=None,
        shape=(60, 100),
        dtype_itemsize=4,
        zarr_format=3,
    )
    assert spec.chunks == "auto"
    assert spec.shards is None
    assert isinstance(spec.chunk_grid, RectilinearChunkGrid)


def test_resolve_chunk_spec_chunk_grid_instance() -> None:
    """Test passing a ChunkGrid instance."""
    grid = RegularChunkGrid(chunk_shape=(15, 15))
    spec = resolve_chunk_spec(
        chunks=grid,
        shards=None,
        shape=(100, 100),
        dtype_itemsize=4,
        zarr_format=3,
    )
    assert spec.chunks == (15, 15)
    assert spec.shards is None
    assert spec.chunk_grid is grid


def test_resolve_chunk_spec_zarr_v2_regular_chunks() -> None:
    """Test Zarr v2 with regular chunks."""
    spec = resolve_chunk_spec(
        chunks=(10, 10),
        shards=None,
        shape=(100, 100),
        dtype_itemsize=4,
        zarr_format=2,
    )
    assert spec.chunks == (10, 10)
    assert spec.shards is None
    assert spec.chunk_grid is None  # Zarr v2 doesn't use chunk_grid


def test_resolve_chunk_spec_result_is_dataclass() -> None:
    """Test that result is a ResolvedChunkSpec dataclass."""
    spec = resolve_chunk_spec(
        chunks=(10, 10),
        shards=None,
        shape=(100, 100),
        dtype_itemsize=4,
        zarr_format=3,
    )
    assert isinstance(spec, ResolvedChunkSpec)
    assert hasattr(spec, "chunk_grid")
    assert hasattr(spec, "chunks")
    assert hasattr(spec, "shards")


# Zarr format compatibility error tests


def test_resolve_chunk_spec_error_variable_chunks_with_zarr_v2() -> None:
    """Test that variable chunks raise error with Zarr v2."""
    with pytest.raises(ValueError, match="only supported in Zarr format 3"):
        resolve_chunk_spec(
            chunks=[[10, 20], [5, 5]],
            shards=None,
            shape=(30, 10),
            dtype_itemsize=4,
            zarr_format=2,
        )


def test_resolve_chunk_spec_error_chunk_grid_with_zarr_v2() -> None:
    """Test that ChunkGrid raises error with Zarr v2."""
    grid = RegularChunkGrid(chunk_shape=(10, 10))
    with pytest.raises(ValueError, match="only supported in Zarr format 3"):
        resolve_chunk_spec(
            chunks=grid,
            shards=None,
            shape=(100, 100),
            dtype_itemsize=4,
            zarr_format=2,
        )


def test_resolve_chunk_spec_error_sharding_with_zarr_v2() -> None:
    """Test that sharding raises error with Zarr v2."""
    with pytest.raises(ValueError, match="only supported in Zarr format 3"):
        resolve_chunk_spec(
            chunks=(10, 10),
            shards=(20, 20),
            shape=(100, 100),
            dtype_itemsize=4,
            zarr_format=2,
        )


# Sharding compatibility error tests


def test_resolve_chunk_spec_error_variable_chunks_with_sharding() -> None:
    """Test that variable chunks + sharding raises error."""
    with pytest.raises(ValueError, match="Cannot use variable chunks.*with sharding"):
        resolve_chunk_spec(
            chunks=[[10, 20], [5, 5]],
            shards=(30, 10),
            shape=(30, 10),
            dtype_itemsize=4,
            zarr_format=3,
        )


def test_resolve_chunk_spec_error_chunk_grid_with_sharding() -> None:
    """Test that ChunkGrid + sharding raises error."""
    grid = RegularChunkGrid(chunk_shape=(10, 10))
    with pytest.raises(ValueError, match="Cannot use ChunkGrid.*with sharding"):
        resolve_chunk_spec(
            chunks=grid,
            shards=(20, 20),
            shape=(100, 100),
            dtype_itemsize=4,
            zarr_format=3,
        )


def test_resolve_chunk_spec_error_rectilinear_chunk_grid_with_sharding() -> None:
    """Test that RectilinearChunkGrid + sharding raises error."""
    grid = RectilinearChunkGrid(chunk_shapes=((10, 20), (5, 5)))
    with pytest.raises(ValueError, match="Cannot use ChunkGrid.*with sharding"):
        resolve_chunk_spec(
            chunks=grid,
            shards=(30, 10),
            shape=(30, 10),
            dtype_itemsize=4,
            zarr_format=3,
        )


# Data compatibility error tests


def test_resolve_chunk_spec_error_variable_chunks_with_data() -> None:
    """Test that variable chunks + has_data raises error."""
    with pytest.raises(
        ValueError, match="Cannot use RectilinearChunkGrid.*when creating array from data"
    ):
        resolve_chunk_spec(
            chunks=[[10, 20, 30], [25, 25, 25, 25]],
            shards=None,
            shape=(60, 100),
            dtype_itemsize=4,
            zarr_format=3,
            has_data=True,
        )


def test_resolve_chunk_spec_error_rectilinear_chunk_grid_with_data() -> None:
    """Test that RectilinearChunkGrid + has_data raises error."""
    grid = RectilinearChunkGrid(chunk_shapes=((10, 20, 30), (25, 25, 25, 25)))
    with pytest.raises(
        ValueError, match="Cannot use RectilinearChunkGrid.*when creating array from data"
    ):
        resolve_chunk_spec(
            chunks=grid,
            shards=None,
            shape=(60, 100),
            dtype_itemsize=4,
            zarr_format=3,
            has_data=True,
        )


def test_resolve_chunk_spec_regular_chunks_with_data_ok() -> None:
    """Test that regular chunks with has_data works fine."""
    spec = resolve_chunk_spec(
        chunks=(10, 10),
        shards=None,
        shape=(100, 100),
        dtype_itemsize=4,
        zarr_format=3,
        has_data=True,
    )
    assert spec.chunks == (10, 10)
    assert spec.shards is None


# Invalid chunk specification error tests


def test_resolve_chunk_spec_error_chunks_dont_sum_to_shape() -> None:
    """Test that variable chunks that don't sum to shape raise error."""
    with pytest.raises(ValueError, match="sum to.*but array shape"):
        resolve_chunk_spec(
            chunks=[[10, 20], [5, 5]],  # sums to 30
            shards=None,
            shape=(40, 10),  # shape is 40
            dtype_itemsize=4,
            zarr_format=3,
        )


def test_resolve_chunk_spec_error_wrong_dimensionality() -> None:
    """Test that variable chunks with wrong dimensionality raise error."""
    with pytest.raises(ValueError, match="dimensionality.*must match"):
        resolve_chunk_spec(
            chunks=[[10, 20, 30]],  # 1D
            shards=None,
            shape=(60, 100),  # 2D
            dtype_itemsize=4,
            zarr_format=3,
        )


# Edge case tests


def test_resolve_chunk_spec_empty_array_shape() -> None:
    """Test with empty array shape."""
    spec = resolve_chunk_spec(
        chunks=(1,),
        shards=None,
        shape=(0,),
        dtype_itemsize=4,
        zarr_format=3,
    )
    # normalize_chunks may adjust chunk size for empty arrays
    assert isinstance(spec.chunks, tuple)
    assert spec.shards is None


def test_resolve_chunk_spec_1d_array() -> None:
    """Test with 1D array."""
    spec = resolve_chunk_spec(
        chunks=(10,),
        shards=None,
        shape=(100,),
        dtype_itemsize=4,
        zarr_format=3,
    )
    assert spec.chunks == (10,)
    assert spec.shards is None


def test_resolve_chunk_spec_high_dimensional_array() -> None:
    """Test with high-dimensional array."""
    spec = resolve_chunk_spec(
        chunks=(10, 10, 10, 10),
        shards=None,
        shape=(100, 100, 100, 100),
        dtype_itemsize=4,
        zarr_format=3,
    )
    assert spec.chunks == (10, 10, 10, 10)
    assert spec.shards is None


def test_resolve_chunk_spec_single_int_with_sharding() -> None:
    """Test single int for chunks with sharding."""
    spec = resolve_chunk_spec(
        chunks=5,
        shards=(20, 20),
        shape=(100, 100),
        dtype_itemsize=4,
        zarr_format=3,
    )
    assert spec.chunks == (5, 5)  # Converted to tuple
    assert spec.shards == (20, 20)


# Backward compatibility tests


def test_resolve_chunk_spec_maintains_chunk_normalization() -> None:
    """Test that chunk normalization still works."""
    # Test with -1 (should use full dimension)
    spec = resolve_chunk_spec(
        chunks=(-1, 10),
        shards=None,
        shape=(100, 100),
        dtype_itemsize=4,
        zarr_format=3,
    )
    assert spec.chunks == (100, 10)  # -1 replaced with full dimension


def test_resolve_chunk_spec_maintains_auto_chunking_heuristics() -> None:
    """Test that auto-chunking heuristics still work."""
    spec = resolve_chunk_spec(
        chunks="auto",
        shards=None,
        shape=(1000, 1000),
        dtype_itemsize=8,
        zarr_format=3,
    )
    # Auto-chunking should produce reasonable chunk sizes
    assert isinstance(spec.chunks, tuple)
    assert len(spec.chunks) == 2
    assert all(c > 0 for c in spec.chunks)
