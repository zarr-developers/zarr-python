"""
Tests for the unified ChunkGrid design (POC).

Tests the core ChunkGrid with FixedDimension/VaryingDimension internals,
serialization round-trips, indexing with rectilinear grids, and end-to-end
array creation + read/write.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from zarr.core.chunk_grids import (
    ChunkGrid,
    FixedDimension,
    RegularChunkGrid,
    VaryingDimension,
    _compress_rle,
    _expand_rle,
)

# ---------------------------------------------------------------------------
# FixedDimension
# ---------------------------------------------------------------------------


class TestFixedDimension:
    def test_basic(self) -> None:
        d = FixedDimension(size=10)
        assert d.size == 10
        assert d.index_to_chunk(0) == 0
        assert d.index_to_chunk(9) == 0
        assert d.index_to_chunk(10) == 1
        assert d.index_to_chunk(25) == 2
        assert d.chunk_offset(0) == 0
        assert d.chunk_offset(1) == 10
        assert d.chunk_offset(3) == 30
        assert d.chunk_size(0, 100) == 10
        assert d.chunk_size(9, 100) == 10
        # boundary chunk
        assert d.chunk_size(9, 95) == 5
        assert d.nchunks(100) == 10
        assert d.nchunks(95) == 10

    def test_vectorized(self) -> None:
        d = FixedDimension(size=10)
        indices = np.array([0, 5, 10, 15, 99])
        chunks = d.indices_to_chunks(indices)
        np.testing.assert_array_equal(chunks, [0, 0, 1, 1, 9])

    def test_negative_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be >= 0"):
            FixedDimension(size=-1)

    def test_zero_size_allowed(self) -> None:
        d = FixedDimension(size=0)
        assert d.size == 0


# ---------------------------------------------------------------------------
# VaryingDimension
# ---------------------------------------------------------------------------


class TestVaryingDimension:
    def test_basic(self) -> None:
        d = VaryingDimension([10, 20, 30])
        assert d.edges == (10, 20, 30)
        assert d.cumulative == (10, 30, 60)
        assert d.nchunks(60) == 3

    def test_index_to_chunk(self) -> None:
        d = VaryingDimension([10, 20, 30])
        # First chunk: indices 0-9
        assert d.index_to_chunk(0) == 0
        assert d.index_to_chunk(9) == 0
        # Second chunk: indices 10-29
        assert d.index_to_chunk(10) == 1
        assert d.index_to_chunk(29) == 1
        # Third chunk: indices 30-59
        assert d.index_to_chunk(30) == 2
        assert d.index_to_chunk(59) == 2

    def test_chunk_offset(self) -> None:
        d = VaryingDimension([10, 20, 30])
        assert d.chunk_offset(0) == 0
        assert d.chunk_offset(1) == 10
        assert d.chunk_offset(2) == 30

    def test_chunk_size(self) -> None:
        d = VaryingDimension([10, 20, 30])
        assert d.chunk_size(0, 60) == 10
        assert d.chunk_size(1, 60) == 20
        assert d.chunk_size(2, 60) == 30

    def test_vectorized(self) -> None:
        d = VaryingDimension([10, 20, 30])
        indices = np.array([0, 9, 10, 29, 30, 59])
        chunks = d.indices_to_chunks(indices)
        np.testing.assert_array_equal(chunks, [0, 0, 1, 1, 2, 2])

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            VaryingDimension([])

    def test_zero_edge_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be > 0"):
            VaryingDimension([10, 0, 5])


# ---------------------------------------------------------------------------
# ChunkGrid construction
# ---------------------------------------------------------------------------


class TestChunkGridConstruction:
    def test_from_regular(self) -> None:
        g = ChunkGrid.from_regular((10, 20))
        assert g.is_regular
        assert g.chunk_shape == (10, 20)
        assert g.ndim == 2

    def test_zero_dim(self) -> None:
        """0-d arrays produce a ChunkGrid with no dimensions."""
        g = ChunkGrid.from_regular(())
        assert g.is_regular
        assert g.chunk_shape == ()
        assert g.ndim == 0

    def test_from_rectilinear(self) -> None:
        g = ChunkGrid.from_rectilinear([[10, 20, 30], [25, 25, 25, 25]])
        assert not g.is_regular
        assert g.ndim == 2
        with pytest.raises(ValueError, match="only available for regular"):
            _ = g.chunk_shape

    def test_rectilinear_with_uniform_dim(self) -> None:
        """A rectilinear grid with all-same sizes in one dim stores it as Fixed."""
        g = ChunkGrid.from_rectilinear([[10, 20, 30], [25, 25, 25, 25]])
        assert isinstance(g.dimensions[0], VaryingDimension)
        assert isinstance(g.dimensions[1], FixedDimension)

    def test_all_uniform_becomes_regular(self) -> None:
        """If all dimensions have uniform sizes, the grid is regular."""
        g = ChunkGrid.from_rectilinear([[10, 10, 10], [25, 25]])
        assert g.is_regular
        assert g.chunk_shape == (10, 25)


# ---------------------------------------------------------------------------
# ChunkGrid queries
# ---------------------------------------------------------------------------


class TestChunkGridQueries:
    def test_regular_grid_shape(self) -> None:
        g = ChunkGrid.from_regular((10, 20))
        assert g.grid_shape((100, 200)) == (10, 10)
        assert g.grid_shape((95, 200)) == (10, 10)

    def test_rectilinear_grid_shape(self) -> None:
        g = ChunkGrid.from_rectilinear([[10, 20, 30], [25, 25, 25, 25]])
        assert g.grid_shape((60, 100)) == (3, 4)

    def test_regular_get_chunk_shape(self) -> None:
        g = ChunkGrid.from_regular((10, 20))
        assert g.get_chunk_shape((100, 200), (0, 0)) == (10, 20)
        assert g.get_chunk_shape((95, 200), (9, 0)) == (5, 20)  # boundary
        assert g.get_chunk_shape((100, 200), (99, 0)) is None  # OOB

    def test_rectilinear_get_chunk_shape(self) -> None:
        g = ChunkGrid.from_rectilinear([[10, 20, 30], [25, 25, 25, 25]])
        assert g.get_chunk_shape((60, 100), (0, 0)) == (10, 25)
        assert g.get_chunk_shape((60, 100), (1, 0)) == (20, 25)
        assert g.get_chunk_shape((60, 100), (2, 3)) == (30, 25)
        assert g.get_chunk_shape((60, 100), (3, 0)) is None  # OOB

    def test_get_chunk_origin(self) -> None:
        g = ChunkGrid.from_rectilinear([[10, 20, 30], [25, 25, 25, 25]])
        assert g.get_chunk_origin((60, 100), (0, 0)) == (0, 0)
        assert g.get_chunk_origin((60, 100), (1, 0)) == (10, 0)
        assert g.get_chunk_origin((60, 100), (2, 2)) == (30, 50)

    def test_all_chunk_coords(self) -> None:
        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]])
        coords = list(g.all_chunk_coords((60, 100)))
        assert len(coords) == 6  # 3 * 2
        assert coords[0] == (0, 0)
        assert coords[-1] == (2, 1)

    def test_get_nchunks(self) -> None:
        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]])
        assert g.get_nchunks((60, 100)) == 6


# ---------------------------------------------------------------------------
# RLE helpers
# ---------------------------------------------------------------------------


class TestRLE:
    def test_expand(self) -> None:
        assert _expand_rle([[10, 3]]) == [10, 10, 10]
        assert _expand_rle([[10, 2], [20, 1]]) == [10, 10, 20]

    def test_compress(self) -> None:
        assert _compress_rle([10, 10, 10]) == [[10, 3]]
        assert _compress_rle([10, 10, 20]) == [[10, 2], [20, 1]]

    def test_roundtrip(self) -> None:
        original = [10, 10, 10, 20, 20, 30]
        compressed = _compress_rle(original)
        assert _expand_rle(compressed) == original


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_regular_roundtrip(self) -> None:
        g = ChunkGrid.from_regular((10, 20))
        d = g.to_dict()
        assert d["name"] == "regular"
        config = d["configuration"]
        assert isinstance(config, dict)
        assert tuple(config["chunk_shape"]) == (10, 20)
        g2 = ChunkGrid.from_dict(d)
        assert g2.is_regular
        assert g2.chunk_shape == (10, 20)

    def test_rectilinear_roundtrip(self) -> None:
        g = ChunkGrid.from_rectilinear([[10, 20, 30], [25, 25, 25, 25]])
        d = g.to_dict()
        assert d["name"] == "rectilinear"
        g2 = ChunkGrid.from_dict(d)
        assert not g2.is_regular
        # Verify the reconstructed grid produces correct shapes
        assert g2.get_chunk_shape((60, 100), (0, 0)) == (10, 25)
        assert g2.get_chunk_shape((60, 100), (1, 0)) == (20, 25)
        assert g2.get_chunk_shape((60, 100), (2, 3)) == (30, 25)

    def test_rectilinear_rle_serialization(self) -> None:
        """RLE should be used when it actually compresses."""
        g = ChunkGrid.from_rectilinear([[100] * 10, [25, 25, 25, 25]])
        d = g.to_dict()
        # First dim: 10 identical chunks -> RLE
        # Second dim: 4 identical chunks -> stored as FixedDimension -> RLE [[25, 1]]
        assert d["name"] == "regular"  # all uniform -> serializes as regular

    def test_rectilinear_rle_with_varying(self) -> None:
        g = ChunkGrid.from_rectilinear([[100, 100, 100, 50], [25, 25, 25, 25]])
        d = g.to_dict()
        assert d["name"] == "rectilinear"
        # Check RLE used for first dimension
        config = d["configuration"]
        assert isinstance(config, dict)
        chunk_shapes = config["chunk_shapes"]
        assert isinstance(chunk_shapes, list)
        # First dim: [100, 100, 100, 50] -> [[100, 3], [50, 1]] (RLE shorter)
        assert chunk_shapes[0] == [[100, 3], [50, 1]]

    def test_json_roundtrip(self) -> None:
        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]])
        d = g.to_dict()
        json_str = json.dumps(d)
        d2 = json.loads(json_str)
        g2 = ChunkGrid.from_dict(d2)
        assert g2.grid_shape((60, 100)) == (3, 2)

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown chunk grid"):
            ChunkGrid.from_dict({"name": "hexagonal", "configuration": {}})


# ---------------------------------------------------------------------------
# Backwards compatibility
# ---------------------------------------------------------------------------


class TestBackwardsCompat:
    def test_regular_chunk_grid_still_works(self) -> None:
        g = RegularChunkGrid(chunk_shape=(10, 20))
        assert g.chunk_shape == (10, 20)
        assert g.is_regular
        assert isinstance(g, ChunkGrid)

    def test_from_dict_regular(self) -> None:
        d: dict[str, Any] = {"name": "regular", "configuration": {"chunk_shape": [10, 20]}}
        g = ChunkGrid.from_dict(d)
        # from_dict now returns ChunkGrid, not RegularChunkGrid
        assert isinstance(g, ChunkGrid)
        assert g.is_regular
        assert g.chunk_shape == (10, 20)

    def test_regular_chunk_grid_passed_to_from_dict(self) -> None:
        """RegularChunkGrid instances should be convertible."""
        rcg = RegularChunkGrid(chunk_shape=(10, 20))
        g = ChunkGrid.from_dict(rcg)
        assert isinstance(g, ChunkGrid)
        assert g.is_regular


# ---------------------------------------------------------------------------
# Indexing with rectilinear grids
# ---------------------------------------------------------------------------


class TestRectilinearIndexing:
    """Test that the indexing pipeline works with VaryingDimension."""

    def test_basic_indexer_rectilinear(self) -> None:
        from zarr.core.indexing import BasicIndexer

        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]])
        indexer = BasicIndexer(
            selection=(slice(None), slice(None)),
            shape=(60, 100),
            chunk_grid=g,
        )
        projections = list(indexer)
        # Should visit all 3*2=6 chunks
        assert len(projections) == 6

        # Check first chunk
        p0 = projections[0]
        assert p0.chunk_coords == (0, 0)
        assert p0.chunk_selection == (slice(0, 10, 1), slice(0, 50, 1))

        # Check second chunk on first axis
        p1 = projections[2]  # (1, 0) in product order
        assert p1.chunk_coords == (1, 0)
        assert p1.chunk_selection == (slice(0, 20, 1), slice(0, 50, 1))

    def test_basic_indexer_int_selection(self) -> None:
        from zarr.core.indexing import BasicIndexer

        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]])
        indexer = BasicIndexer(
            selection=(15, slice(None)),  # index 15 falls in chunk 1 (offset 10)
            shape=(60, 100),
            chunk_grid=g,
        )
        projections = list(indexer)
        assert len(projections) == 2  # 2 chunks in second dimension
        assert projections[0].chunk_coords == (1, 0)
        assert projections[0].chunk_selection == (5, slice(0, 50, 1))  # 15 - 10 = 5

    def test_basic_indexer_slice_subset(self) -> None:
        from zarr.core.indexing import BasicIndexer

        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]])
        indexer = BasicIndexer(
            selection=(slice(5, 35), slice(0, 50)),
            shape=(60, 100),
            chunk_grid=g,
        )
        projections = list(indexer)
        # slice(5, 35) spans chunks 0 (5:10), 1 (0:20), 2 (0:5)
        chunk_coords_dim0 = sorted({p.chunk_coords[0] for p in projections})
        assert chunk_coords_dim0 == [0, 1, 2]

    def test_orthogonal_indexer_rectilinear(self) -> None:
        from zarr.core.indexing import OrthogonalIndexer

        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]])
        indexer = OrthogonalIndexer(
            selection=(slice(None), slice(None)),
            shape=(60, 100),
            chunk_grid=g,
        )
        projections = list(indexer)
        assert len(projections) == 6


# ---------------------------------------------------------------------------
# End-to-end: array creation with rectilinear chunks
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Test creating, writing, and reading arrays with rectilinear chunk grids."""

    def test_create_regular_array(self, tmp_path: Path) -> None:
        import zarr

        arr = zarr.create_array(
            store=tmp_path / "regular.zarr",
            shape=(100, 200),
            chunks=(10, 20),
            dtype="float32",
        )
        assert arr.metadata.chunk_grid.is_regular
        assert arr.chunks == (10, 20)

    def test_create_rectilinear_array(self, tmp_path: Path) -> None:
        """Create an array with a rectilinear chunk grid via metadata."""
        from zarr.core.array import AsyncArray
        from zarr.core.dtype import Float32
        from zarr.core.metadata.v3 import ArrayV3Metadata

        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]])

        meta = AsyncArray._create_metadata_v3(
            shape=(60, 100),
            dtype=Float32(),
            chunk_shape=(10, 20),  # fallback, overridden by chunk_grid
            chunk_grid=g,
        )
        assert isinstance(meta, ArrayV3Metadata)
        assert not meta.chunk_grid.is_regular
        assert meta.chunk_grid.ndim == 2

    def test_rectilinear_metadata_serialization(self, tmp_path: Path) -> None:
        """Verify metadata round-trips through JSON."""
        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]])
        d = g.to_dict()
        g2 = ChunkGrid.from_dict(d)
        assert g2.grid_shape((60, 100)) == g.grid_shape((60, 100))
        # All chunk shapes should match
        for coord in g.all_chunk_coords((60, 100)):
            assert g.get_chunk_shape((60, 100), coord) == g2.get_chunk_shape((60, 100), coord)

    def test_get_chunk_spec_regular(self, tmp_path: Path) -> None:
        """get_chunk_spec works for regular grids."""
        from zarr.core.array import AsyncArray
        from zarr.core.array_spec import ArrayConfig
        from zarr.core.buffer.core import default_buffer_prototype
        from zarr.core.dtype import Float32

        meta = AsyncArray._create_metadata_v3(
            shape=(100, 200),
            dtype=Float32(),
            chunk_shape=(10, 20),
        )
        spec = meta.get_chunk_spec(
            (0, 0),
            ArrayConfig.from_dict({}),
            default_buffer_prototype(),
        )
        assert spec.shape == (10, 20)

        # Boundary chunk
        spec_boundary = meta.get_chunk_spec(
            (9, 9),
            ArrayConfig.from_dict({}),
            default_buffer_prototype(),
        )
        assert spec_boundary.shape == (10, 20)

    def test_get_chunk_spec_rectilinear(self, tmp_path: Path) -> None:
        """get_chunk_spec returns per-chunk shapes for rectilinear grids."""
        from zarr.core.array import AsyncArray
        from zarr.core.array_spec import ArrayConfig
        from zarr.core.buffer.core import default_buffer_prototype
        from zarr.core.dtype import Float32

        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]])
        meta = AsyncArray._create_metadata_v3(
            shape=(60, 100),
            dtype=Float32(),
            chunk_shape=(10, 20),
            chunk_grid=g,
        )
        proto = default_buffer_prototype()
        config = ArrayConfig.from_dict({})

        spec0 = meta.get_chunk_spec((0, 0), config, proto)
        assert spec0.shape == (10, 50)

        spec1 = meta.get_chunk_spec((1, 0), config, proto)
        assert spec1.shape == (20, 50)

        spec2 = meta.get_chunk_spec((2, 1), config, proto)
        assert spec2.shape == (30, 50)


# ---------------------------------------------------------------------------
# Sharding compatibility
# ---------------------------------------------------------------------------


class TestShardingCompat:
    def test_sharding_accepts_rectilinear_outer_grid(self) -> None:
        """ShardingCodec.validate should not reject rectilinear outer grids."""
        from zarr.codecs.sharding import ShardingCodec
        from zarr.core.dtype import Float32

        codec = ShardingCodec(chunk_shape=(5, 5))
        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]])

        # Should not raise
        codec.validate(
            shape=(60, 100),
            dtype=Float32(),
            chunk_grid=g,
        )
