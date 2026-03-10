"""
Tests for the unified ChunkGrid design (POC).

Tests the core ChunkGrid with FixedDimension/VaryingDimension internals,
ChunkSpec, serialization round-trips, indexing with rectilinear grids,
and end-to-end array creation + read/write.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import zarr

if TYPE_CHECKING:
    from pathlib import Path

    from zarr.core.common import JSON

from zarr.core.chunk_grids import (
    ChunkGrid,
    ChunkSpec,
    FixedDimension,
    RegularChunkGrid,
    VaryingDimension,
    _compress_rle,
    _expand_rle,
    parse_chunk_grid,
    serialize_chunk_grid,
)

# ---------------------------------------------------------------------------
# FixedDimension
# ---------------------------------------------------------------------------


class TestFixedDimension:
    def test_basic(self) -> None:
        d = FixedDimension(size=10, extent=100)
        assert d.size == 10
        assert d.extent == 100
        assert d.index_to_chunk(0) == 0
        assert d.index_to_chunk(9) == 0
        assert d.index_to_chunk(10) == 1
        assert d.index_to_chunk(25) == 2
        assert d.chunk_offset(0) == 0
        assert d.chunk_offset(1) == 10
        assert d.chunk_offset(3) == 30
        # chunk_size is always uniform (codec buffer)
        assert d.chunk_size(0) == 10
        assert d.chunk_size(9) == 10
        # data_size clips at boundary
        assert d.data_size(0) == 10
        assert d.data_size(9) == 10
        assert d.nchunks == 10

    def test_boundary_data_size(self) -> None:
        d = FixedDimension(size=10, extent=95)
        assert d.nchunks == 10
        assert d.chunk_size(9) == 10  # codec buffer always full
        assert d.data_size(9) == 5  # only 5 valid elements at boundary

    def test_vectorized(self) -> None:
        d = FixedDimension(size=10, extent=100)
        indices = np.array([0, 5, 10, 15, 99])
        chunks = d.indices_to_chunks(indices)
        np.testing.assert_array_equal(chunks, [0, 0, 1, 1, 9])

    def test_negative_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be >= 0"):
            FixedDimension(size=-1, extent=100)

    def test_negative_extent_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be >= 0"):
            FixedDimension(size=10, extent=-1)

    def test_zero_size_allowed(self) -> None:
        d = FixedDimension(size=0, extent=0)
        assert d.size == 0
        assert d.nchunks == 1  # 0-size with 0-extent = 1 chunk


# ---------------------------------------------------------------------------
# VaryingDimension
# ---------------------------------------------------------------------------


class TestVaryingDimension:
    def test_basic(self) -> None:
        d = VaryingDimension([10, 20, 30])
        assert d.edges == (10, 20, 30)
        assert d.cumulative == (10, 30, 60)
        assert d.nchunks == 3
        assert d.extent == 60

    def test_index_to_chunk(self) -> None:
        d = VaryingDimension([10, 20, 30])
        assert d.index_to_chunk(0) == 0
        assert d.index_to_chunk(9) == 0
        assert d.index_to_chunk(10) == 1
        assert d.index_to_chunk(29) == 1
        assert d.index_to_chunk(30) == 2
        assert d.index_to_chunk(59) == 2

    def test_chunk_offset(self) -> None:
        d = VaryingDimension([10, 20, 30])
        assert d.chunk_offset(0) == 0
        assert d.chunk_offset(1) == 10
        assert d.chunk_offset(2) == 30

    def test_chunk_size(self) -> None:
        d = VaryingDimension([10, 20, 30])
        assert d.chunk_size(0) == 10
        assert d.chunk_size(1) == 20
        assert d.chunk_size(2) == 30

    def test_data_size(self) -> None:
        d = VaryingDimension([10, 20, 30])
        # data_size == chunk_size for varying dims
        assert d.data_size(0) == 10
        assert d.data_size(1) == 20
        assert d.data_size(2) == 30

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
# ChunkSpec
# ---------------------------------------------------------------------------


class TestChunkSpec:
    def test_basic(self) -> None:
        spec = ChunkSpec(
            slices=(slice(0, 10), slice(0, 20)),
            codec_shape=(10, 20),
        )
        assert spec.shape == (10, 20)
        assert not spec.is_boundary

    def test_boundary(self) -> None:
        spec = ChunkSpec(
            slices=(slice(90, 95), slice(0, 20)),
            codec_shape=(10, 20),
        )
        assert spec.shape == (5, 20)
        assert spec.is_boundary


# ---------------------------------------------------------------------------
# ChunkGrid construction
# ---------------------------------------------------------------------------


class TestChunkGridConstruction:
    def test_from_regular(self) -> None:
        g = ChunkGrid.from_regular((100, 200), (10, 20))
        assert g.is_regular
        assert g.chunk_shape == (10, 20)
        assert g.ndim == 2

    def test_zero_dim(self) -> None:
        """0-d arrays produce a ChunkGrid with no dimensions."""
        g = ChunkGrid.from_regular((), ())
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
    def test_regular_shape(self) -> None:
        g = ChunkGrid.from_regular((100, 200), (10, 20))
        assert g.shape == (10, 10)

    def test_regular_shape_boundary(self) -> None:
        g = ChunkGrid.from_regular((95, 200), (10, 20))
        assert g.shape == (10, 10)  # ceildiv(95, 10) == 10

    def test_rectilinear_shape(self) -> None:
        g = ChunkGrid.from_rectilinear([[10, 20, 30], [25, 25, 25, 25]])
        assert g.shape == (3, 4)

    def test_regular_getitem(self) -> None:
        g = ChunkGrid.from_regular((100, 200), (10, 20))
        spec = g[(0, 0)]
        assert spec is not None
        assert spec.shape == (10, 20)
        assert spec.codec_shape == (10, 20)
        assert not spec.is_boundary

    def test_regular_getitem_boundary(self) -> None:
        g = ChunkGrid.from_regular((95, 200), (10, 20))
        spec = g[(9, 0)]
        assert spec is not None
        assert spec.shape == (5, 20)  # data_size clipped
        assert spec.codec_shape == (10, 20)  # codec always full
        assert spec.is_boundary

    def test_regular_getitem_oob(self) -> None:
        g = ChunkGrid.from_regular((100, 200), (10, 20))
        assert g[(99, 0)] is None

    def test_rectilinear_getitem(self) -> None:
        g = ChunkGrid.from_rectilinear([[10, 20, 30], [25, 25, 25, 25]])
        spec0 = g[(0, 0)]
        assert spec0 is not None
        assert spec0.shape == (10, 25)

        spec1 = g[(1, 0)]
        assert spec1 is not None
        assert spec1.shape == (20, 25)

        spec2 = g[(2, 3)]
        assert spec2 is not None
        assert spec2.shape == (30, 25)

        assert g[(3, 0)] is None  # OOB

    def test_getitem_slices(self) -> None:
        g = ChunkGrid.from_rectilinear([[10, 20, 30], [25, 25, 25, 25]])
        spec = g[(1, 2)]
        assert spec is not None
        assert spec.slices == (slice(10, 30), slice(50, 75))

    def test_all_chunk_coords(self) -> None:
        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]])
        coords = list(g.all_chunk_coords())
        assert len(coords) == 6
        assert coords[0] == (0, 0)
        assert coords[-1] == (2, 1)

    def test_get_nchunks(self) -> None:
        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]])
        assert g.get_nchunks() == 6

    def test_iter(self) -> None:
        g = ChunkGrid.from_regular((30, 40), (10, 20))
        specs = list(g)
        assert len(specs) == 6  # 3 * 2
        assert all(isinstance(s, ChunkSpec) for s in specs)


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
        g = ChunkGrid.from_regular((100, 200), (10, 20))
        d = serialize_chunk_grid(g, "regular")
        assert d["name"] == "regular"
        config = d["configuration"]
        assert isinstance(config, dict)
        assert tuple(config["chunk_shape"]) == (10, 20)
        g2 = ChunkGrid.from_dict(d)
        assert g2.is_regular
        assert g2.chunk_shape == (10, 20)

    def test_rectilinear_roundtrip(self) -> None:
        g = ChunkGrid.from_rectilinear([[10, 20, 30], [25, 25, 25, 25]])
        d = serialize_chunk_grid(g, "rectilinear")
        assert d["name"] == "rectilinear"
        g2 = ChunkGrid.from_dict(d)
        assert not g2.is_regular
        # Verify the reconstructed grid has same dimensions
        spec0 = g2[(0, 0)]
        assert spec0 is not None
        assert spec0.shape == (10, 25)
        spec1 = g2[(1, 0)]
        assert spec1 is not None
        assert spec1.shape == (20, 25)

    def test_rectilinear_rle_serialization(self) -> None:
        """RLE should be used when it actually compresses."""
        g = ChunkGrid.from_rectilinear([[100] * 10, [25, 25, 25, 25]])
        # All uniform, but name is chosen by the metadata layer, not the grid.
        # Serializing as "regular" works because is_regular is True.
        d = serialize_chunk_grid(g, "regular")
        assert d["name"] == "regular"

    def test_rectilinear_uniform_stays_rectilinear(self) -> None:
        """A rectilinear grid with uniform edges stays rectilinear if the name says so."""
        g = ChunkGrid.from_rectilinear([[100] * 10, [25, 25, 25, 25]])
        d = serialize_chunk_grid(g, "rectilinear")
        assert d["name"] == "rectilinear"

    def test_rectilinear_rle_with_varying(self) -> None:
        g = ChunkGrid.from_rectilinear([[100, 100, 100, 50], [25, 25, 25, 25]])
        d = serialize_chunk_grid(g, "rectilinear")
        assert d["name"] == "rectilinear"
        config = d["configuration"]
        assert isinstance(config, dict)
        chunk_shapes = config["chunk_shapes"]
        assert isinstance(chunk_shapes, list)
        assert chunk_shapes[0] == [[100, 3], [50, 1]]

    def test_json_roundtrip(self) -> None:
        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]])
        d = serialize_chunk_grid(g, "rectilinear")
        json_str = json.dumps(d)
        d2 = json.loads(json_str)
        g2 = ChunkGrid.from_dict(d2)
        assert g2.shape == (3, 2)

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown chunk grid"):
            ChunkGrid.from_dict({"name": "hexagonal", "configuration": {}})

    def test_serialize_non_regular_as_regular_raises(self) -> None:
        g = ChunkGrid.from_rectilinear([[10, 20, 30], [25, 25, 25, 25]])
        with pytest.raises(ValueError, match="Cannot serialize a non-regular chunk grid"):
            serialize_chunk_grid(g, "regular")

    def test_serialize_unknown_name_raises(self) -> None:
        g = ChunkGrid.from_regular((100,), (10,))
        with pytest.raises(ValueError, match="Unknown chunk grid name for serialization"):
            serialize_chunk_grid(g, "hexagonal")


class TestParseChunkGridValidation:
    def test_varying_extent_mismatch_raises(self) -> None:
        from zarr.core.chunk_grids import parse_chunk_grid

        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]])
        # VaryingDimension extent is 60, but array_shape says 100
        with pytest.raises(ValueError, match="extent"):
            parse_chunk_grid(g, (100, 100))

    def test_varying_extent_match_ok(self) -> None:
        from zarr.core.chunk_grids import parse_chunk_grid

        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]])
        # Matching extents should work fine
        g2 = parse_chunk_grid(g, (60, 100))
        assert g2.dimensions[0].extent == 60

    def test_rectilinear_extent_mismatch_raises(self) -> None:
        """sum(edges) must match the array shape for each dimension."""
        data: dict[str, Any] = {
            "name": "rectilinear",
            "configuration": {"chunk_shapes": [[10, 20, 30], [25, 25]]},
        }
        # sum([10,20,30])=60, sum([25,25])=50 — array shape (100, 50) mismatches dim 0
        with pytest.raises(ValueError, match="sum to 60 but array shape extent is 100"):
            parse_chunk_grid(data, (100, 50))

    def test_rectilinear_extent_mismatch_second_dim(self) -> None:
        data: dict[str, Any] = {
            "name": "rectilinear",
            "configuration": {"chunk_shapes": [[50, 50], [10, 20]]},
        }
        # dim 0 OK (100), dim 1: sum([10,20])=30 != 50
        with pytest.raises(ValueError, match="dimension 1 sum to 30 but array shape extent is 50"):
            parse_chunk_grid(data, (100, 50))

    def test_rectilinear_extent_match_passes(self) -> None:
        data: dict[str, Any] = {
            "name": "rectilinear",
            "configuration": {"chunk_shapes": [[10, 20, 30], [25, 25]]},
        }
        g = parse_chunk_grid(data, (60, 50))
        assert g.shape == (3, 2)

    def test_rectilinear_ndim_mismatch_raises(self) -> None:
        data: dict[str, Any] = {
            "name": "rectilinear",
            "configuration": {"chunk_shapes": [[10, 20], [25, 25]]},
        }
        with pytest.raises(ValueError, match="2 dimensions but array shape has 3"):
            parse_chunk_grid(data, (30, 50, 100))

    def test_rectilinear_rle_extent_validated(self) -> None:
        """RLE-encoded edges are expanded before validation."""
        data: dict[str, Any] = {
            "name": "rectilinear",
            "configuration": {"chunk_shapes": [[[10, 5]], [[25, 2]]]},
        }
        # sum = 50 and 50 — match (50, 50)
        g = parse_chunk_grid(data, (50, 50))
        assert g.shape == (5, 2)
        # mismatch
        with pytest.raises(ValueError, match="sum to 50 but array shape extent is 100"):
            parse_chunk_grid(data, (100, 50))

    def test_varying_dimension_extent_mismatch_on_chunkgrid_input(self) -> None:
        """When passing a ChunkGrid directly, VaryingDimension extent is validated."""
        g = ChunkGrid.from_rectilinear([[10, 20, 30], [25, 25]])
        with pytest.raises(ValueError, match="does not match"):
            parse_chunk_grid(g, (100, 50))


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
        d: dict[str, JSON] = {"name": "regular", "configuration": {"chunk_shape": [10, 20]}}
        g = ChunkGrid.from_dict(d)
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
        assert len(projections) == 6

        p0 = projections[0]
        assert p0.chunk_coords == (0, 0)
        assert p0.chunk_selection == (slice(0, 10, 1), slice(0, 50, 1))

        p1 = projections[2]
        assert p1.chunk_coords == (1, 0)
        assert p1.chunk_selection == (slice(0, 20, 1), slice(0, 50, 1))

    def test_basic_indexer_int_selection(self) -> None:
        from zarr.core.indexing import BasicIndexer

        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]])
        indexer = BasicIndexer(
            selection=(15, slice(None)),
            shape=(60, 100),
            chunk_grid=g,
        )
        projections = list(indexer)
        assert len(projections) == 2
        assert projections[0].chunk_coords == (1, 0)
        assert projections[0].chunk_selection == (5, slice(0, 50, 1))

    def test_basic_indexer_slice_subset(self) -> None:
        from zarr.core.indexing import BasicIndexer

        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]])
        indexer = BasicIndexer(
            selection=(slice(5, 35), slice(0, 50)),
            shape=(60, 100),
            chunk_grid=g,
        )
        projections = list(indexer)
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
            chunk_shape=(10, 20),
            chunk_grid=g,
        )
        assert isinstance(meta, ArrayV3Metadata)
        assert not meta.chunk_grid.is_regular
        assert meta.chunk_grid.ndim == 2

    def test_rectilinear_metadata_serialization(self, tmp_path: Path) -> None:
        """Verify metadata round-trips through JSON."""
        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]])
        d = serialize_chunk_grid(g, "rectilinear")
        g2 = ChunkGrid.from_dict(d)
        assert g2.shape == g.shape
        for coord in g.all_chunk_coords():
            orig_spec = g[coord]
            new_spec = g2[coord]
            assert orig_spec is not None
            assert new_spec is not None
            assert orig_spec.shape == new_spec.shape

    def test_chunk_grid_name_regular(self, tmp_path: Path) -> None:
        """Regular arrays store chunk_grid_name='regular'."""
        from zarr.core.array import AsyncArray
        from zarr.core.dtype import Float32

        meta = AsyncArray._create_metadata_v3(
            shape=(100, 200),
            dtype=Float32(),
            chunk_shape=(10, 20),
        )
        assert meta.chunk_grid_name == "regular"
        d = meta.to_dict()
        chunk_grid_dict = d["chunk_grid"]
        assert isinstance(chunk_grid_dict, dict)
        assert chunk_grid_dict["name"] == "regular"

    def test_chunk_grid_name_rectilinear(self, tmp_path: Path) -> None:
        """Rectilinear arrays store chunk_grid_name='rectilinear'."""
        from zarr.core.array import AsyncArray
        from zarr.core.dtype import Float32

        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]])
        meta = AsyncArray._create_metadata_v3(
            shape=(60, 100),
            dtype=Float32(),
            chunk_shape=(10, 20),
            chunk_grid=g,
        )
        assert meta.chunk_grid_name == "rectilinear"
        d = meta.to_dict()
        chunk_grid_dict = d["chunk_grid"]
        assert isinstance(chunk_grid_dict, dict)
        assert chunk_grid_dict["name"] == "rectilinear"

    def test_chunk_grid_name_roundtrip_preserves_rectilinear(self, tmp_path: Path) -> None:
        """A rectilinear grid with uniform edges stays 'rectilinear' through to_dict/from_dict."""
        from zarr.core.metadata.v3 import ArrayV3Metadata

        meta_dict: dict[str, Any] = {
            "zarr_format": 3,
            "node_type": "array",
            "shape": [100, 100],
            "chunk_grid": {
                "name": "rectilinear",
                "configuration": {"chunk_shapes": [[[50, 2]], [[25, 4]]]},
            },
            "chunk_key_encoding": {"name": "default"},
            "data_type": "float32",
            "fill_value": 0.0,
            "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        }
        meta = ArrayV3Metadata.from_dict(meta_dict)
        # Grid is uniform (all Fixed), but name should stay "rectilinear"
        assert meta.chunk_grid.is_regular
        assert meta.chunk_grid_name == "rectilinear"
        d = meta.to_dict()
        chunk_grid_dict = d["chunk_grid"]
        assert isinstance(chunk_grid_dict, dict)
        assert chunk_grid_dict["name"] == "rectilinear"

    def test_chunk_grid_name_regular_from_dict(self, tmp_path: Path) -> None:
        """A 'regular' chunk grid name is preserved through from_dict."""
        from zarr.core.metadata.v3 import ArrayV3Metadata

        meta_dict: dict[str, Any] = {
            "zarr_format": 3,
            "node_type": "array",
            "shape": [100, 100],
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": [50, 25]},
            },
            "chunk_key_encoding": {"name": "default"},
            "data_type": "float32",
            "fill_value": 0.0,
            "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        }
        meta = ArrayV3Metadata.from_dict(meta_dict)
        assert meta.chunk_grid_name == "regular"
        d = meta.to_dict()
        chunk_grid_dict = d["chunk_grid"]
        assert isinstance(chunk_grid_dict, dict)
        assert chunk_grid_dict["name"] == "regular"

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

        codec.validate(
            shape=(60, 100),
            dtype=Float32(),
            chunk_grid=g,
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases around boundary chunks, zero-size dims, direct construction,
    and serialization round-trips."""

    # -- FixedDimension boundary (extent != size * nchunks) --

    def test_fixed_dim_boundary_data_size(self) -> None:
        """Boundary chunk's data_size is clipped to the remainder."""
        d = FixedDimension(size=10, extent=95)
        assert d.nchunks == 10
        assert d.data_size(0) == 10
        assert d.data_size(9) == 5  # 95 - 9*10 = 5
        assert d.chunk_size(9) == 10  # codec buffer always full

    def test_fixed_dim_data_size_out_of_bounds(self) -> None:
        """data_size clamps to 0 for out-of-bounds chunk indices."""
        d = FixedDimension(size=10, extent=100)
        assert d.data_size(10) == 0  # exactly at boundary
        assert d.data_size(11) == 0  # past boundary
        assert d.data_size(999) == 0

    def test_fixed_dim_data_size_boundary_oob(self) -> None:
        """data_size for boundary grid, past last chunk."""
        d = FixedDimension(size=10, extent=95)
        assert d.data_size(10) == 0  # past nchunks=10

    def test_chunk_grid_boundary_getitem(self) -> None:
        """ChunkGrid with boundary FixedDimension via direct construction."""
        g = ChunkGrid(dimensions=(FixedDimension(10, 95), FixedDimension(20, 40)))
        spec = g[(9, 1)]
        assert spec is not None
        assert spec.shape == (5, 20)  # data: (95-90, 40-20)
        assert spec.codec_shape == (10, 20)  # codec buffers are full
        assert spec.is_boundary

    def test_chunk_grid_boundary_iter(self) -> None:
        """Iterating a boundary grid yields correct boundary ChunkSpecs."""
        g = ChunkGrid(dimensions=(FixedDimension(10, 25),))
        specs = list(g)
        assert len(specs) == 3
        assert specs[0].shape == (10,)
        assert specs[1].shape == (10,)
        assert specs[2].shape == (5,)
        assert specs[2].is_boundary
        assert not specs[0].is_boundary

    def test_chunk_grid_boundary_shape(self) -> None:
        """shape property with boundary extent."""
        g = ChunkGrid(dimensions=(FixedDimension(10, 95),))
        assert g.shape == (10,)  # ceildiv(95, 10) = 10

    # -- Boundary FixedDimension in rectilinear serialization --

    def test_boundary_fixed_dim_rectilinear_roundtrip(self) -> None:
        """A rectilinear grid with a boundary FixedDimension preserves extent."""
        g = ChunkGrid(
            dimensions=(
                VaryingDimension([10, 20, 30]),
                FixedDimension(size=10, extent=95),
            )
        )
        assert g.shape == (3, 10)

        d = serialize_chunk_grid(g, "rectilinear")
        assert d["name"] == "rectilinear"
        # Second dim should serialize as edges that sum to 95
        config = d["configuration"]
        assert isinstance(config, dict)
        chunk_shapes = config["chunk_shapes"]
        assert isinstance(chunk_shapes, list)
        # Last edge should be 5, not 10
        dim1_shapes = chunk_shapes[1]
        # Expand RLE to check
        if isinstance(dim1_shapes[0], list):
            expanded = _expand_rle(dim1_shapes)
        else:
            expanded = dim1_shapes
        assert sum(expanded) == 95  # extent preserved
        assert expanded[-1] == 5  # boundary chunk

        g2 = ChunkGrid.from_dict(d)
        assert g2.shape == g.shape
        # Round-tripped grid should have correct extent
        for coord in g.all_chunk_coords():
            orig = g[coord]
            new = g2[coord]
            assert orig is not None
            assert new is not None
            assert orig.shape == new.shape

    def test_exact_extent_fixed_dim_rectilinear_roundtrip(self) -> None:
        """No boundary: extent == size * nchunks round-trips cleanly."""
        g = ChunkGrid(
            dimensions=(
                VaryingDimension([10, 20]),
                FixedDimension(size=25, extent=100),
            )
        )
        d = serialize_chunk_grid(g, "rectilinear")
        g2 = ChunkGrid.from_dict(d)
        assert g2.shape == g.shape
        # All chunks should be uniform
        for coord in g.all_chunk_coords():
            orig = g[coord]
            new = g2[coord]
            assert orig is not None
            assert new is not None
            assert orig.shape == new.shape

    # -- Zero-size and zero-extent --

    def test_zero_size_zero_extent(self) -> None:
        """FixedDimension(size=0, extent=0) => 1 chunk of size 0."""
        d = FixedDimension(size=0, extent=0)
        assert d.nchunks == 1
        assert d.chunk_size(0) == 0
        assert d.data_size(0) == 0

    def test_zero_size_nonzero_extent(self) -> None:
        """FixedDimension(size=0, extent=5) => 0 chunks (can't partition)."""
        d = FixedDimension(size=0, extent=5)
        assert d.nchunks == 0
        assert d.data_size(0) == 0
        assert d.chunk_size(0) == 0

    def test_zero_extent_nonzero_size(self) -> None:
        """FixedDimension(size=10, extent=0) => 0 chunks."""
        d = FixedDimension(size=10, extent=0)
        assert d.nchunks == 0
        assert d.data_size(0) == 0

    # -- 0-d grid --

    def test_0d_grid_getitem(self) -> None:
        """0-d grid has exactly one chunk at coords ()."""
        g = ChunkGrid.from_regular((), ())
        spec = g[()]
        assert spec is not None
        assert spec.shape == ()
        assert spec.codec_shape == ()
        assert not spec.is_boundary

    def test_0d_grid_iter(self) -> None:
        """0-d grid iteration yields a single ChunkSpec."""
        g = ChunkGrid.from_regular((), ())
        specs = list(g)
        assert len(specs) == 1

    def test_0d_grid_all_chunk_coords(self) -> None:
        """0-d grid has one chunk coord: the empty tuple."""
        g = ChunkGrid.from_regular((), ())
        coords = list(g.all_chunk_coords())
        assert coords == [()]

    def test_0d_grid_nchunks(self) -> None:
        g = ChunkGrid.from_regular((), ())
        assert g.get_nchunks() == 1

    # -- parse_chunk_grid edge cases --

    def test_parse_chunk_grid_preserves_varying_extent(self) -> None:
        """parse_chunk_grid does not overwrite VaryingDimension extent."""
        from zarr.core.chunk_grids import parse_chunk_grid

        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]])
        # VaryingDimension extent is 60 (sum of edges)
        assert isinstance(g.dimensions[0], VaryingDimension)
        assert g.dimensions[0].extent == 60

        # Re-binding with a different array shape should not change VaryingDimension
        g2 = parse_chunk_grid(g, (60, 100))
        assert isinstance(g2.dimensions[0], VaryingDimension)
        assert g2.dimensions[0].extent == 60  # unchanged

    def test_parse_chunk_grid_rebinds_fixed_extent(self) -> None:
        """parse_chunk_grid updates FixedDimension extent from array shape."""
        from zarr.core.chunk_grids import parse_chunk_grid

        g = ChunkGrid.from_regular((100, 200), (10, 20))
        assert g.dimensions[0].extent == 100

        g2 = parse_chunk_grid(g, (50, 100))
        assert isinstance(g2.dimensions[0], FixedDimension)
        assert g2.dimensions[0].extent == 50  # re-bound
        assert g2.shape == (5, 5)

    # -- ChunkGrid.__getitem__ validation --

    def test_getitem_negative_index_returns_none(self) -> None:
        g = ChunkGrid.from_regular((100,), (10,))
        assert g[(-1,)] is None

    def test_getitem_oob_returns_none(self) -> None:
        g = ChunkGrid.from_regular((100,), (10,))
        assert g[(10,)] is None
        assert g[(99,)] is None

    # -- ChunkSpec properties --

    def test_chunk_spec_empty_slices(self) -> None:
        """ChunkSpec with zero-width slice."""
        spec = ChunkSpec(slices=(slice(10, 10),), codec_shape=(0,))
        assert spec.shape == (0,)
        assert not spec.is_boundary

    def test_chunk_spec_multidim_boundary(self) -> None:
        """is_boundary only when shape != codec_shape."""
        spec = ChunkSpec(
            slices=(slice(0, 10), slice(0, 5)),
            codec_shape=(10, 10),
        )
        assert spec.shape == (10, 5)
        assert spec.is_boundary  # second dim differs

    # -- Rectilinear with zero-nchunks FixedDimension in serialize_chunk_grid --

    def test_zero_nchunks_fixed_dim_in_rectilinear_serialize(self) -> None:
        """A rectilinear grid with a 0-nchunks FixedDimension serializes."""
        g = ChunkGrid(
            dimensions=(
                VaryingDimension([10, 20]),
                FixedDimension(size=10, extent=0),
            )
        )
        assert g.shape == (2, 0)
        d = serialize_chunk_grid(g, "rectilinear")
        assert d["name"] == "rectilinear"

    # -- VaryingDimension data_size --

    def test_varying_dim_data_size_equals_chunk_size(self) -> None:
        """For VaryingDimension, data_size == chunk_size (no padding)."""
        d = VaryingDimension([10, 20, 5])
        for i in range(3):
            assert d.data_size(i) == d.chunk_size(i)


class TestOrthogonalIndexerRectilinear:
    """OrthogonalIndexer must use correct per-chunk sizes for VaryingDimension,
    not a hardcoded 1. The chunk_shape field is used by ix_() to convert slices
    to ranges for advanced indexing."""

    def test_orthogonal_int_array_selection_rectilinear(self) -> None:
        """Integer array selection with rectilinear grid must produce correct
        chunk-local selections."""
        from zarr.core.indexing import OrthogonalIndexer

        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]])
        indexer = OrthogonalIndexer(
            selection=(np.array([5, 15, 35]), slice(None)),
            shape=(60, 100),
            chunk_grid=g,
        )
        projections = list(indexer)
        # Grid: dim0 chunks [0..10), [10..30), [30..60); dim1 chunks [0..50), [50..100)
        # Indices 5, 15, 35 land in chunks 0, 1, 2 respectively.
        # Combined with slice(None) over 2 dim1 chunks, we get 6 projections.
        chunk_coords = [p.chunk_coords for p in projections]
        assert chunk_coords == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]

    def test_orthogonal_bool_array_selection_rectilinear(self) -> None:
        """Boolean array selection with rectilinear grid."""
        from zarr.core.indexing import OrthogonalIndexer

        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]])
        mask = np.zeros(60, dtype=bool)
        mask[5] = True
        mask[15] = True
        mask[35] = True
        indexer = OrthogonalIndexer(
            selection=(mask, slice(None)),
            shape=(60, 100),
            chunk_grid=g,
        )
        projections = list(indexer)
        assert len(projections) > 0

    def test_orthogonal_advanced_indexing_chunk_shape_not_one(self) -> None:
        """Verify OrthogonalIndexer.chunk_shape reflects actual chunk sizes,
        not a hardcoded 1 for VaryingDimension."""
        from zarr.core.indexing import OrthogonalIndexer

        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]])
        indexer = OrthogonalIndexer(
            selection=(np.array([5, 15]), slice(None)),
            shape=(60, 100),
            chunk_grid=g,
        )
        # chunk_shape should NOT have 1 for the VaryingDimension
        # The first dim has varying chunks [10, 20, 30] — we need a
        # representative size for ix_() to work. Using the max is safe.
        assert indexer.chunk_shape[0] > 1  # was incorrectly 1 before fix
        assert indexer.chunk_shape[1] == 50


class TestShardingValidationRectilinear:
    """ShardingCodec.validate must check divisibility for rectilinear grids too."""

    def test_sharding_rejects_non_divisible_rectilinear(self) -> None:
        """Rectilinear shard sizes not divisible by inner chunk_shape should raise."""
        from zarr.codecs.sharding import ShardingCodec
        from zarr.core.dtype import Float32

        codec = ShardingCodec(chunk_shape=(5, 5))
        # 17 is not divisible by 5
        g = ChunkGrid.from_rectilinear([[10, 20, 17], [50, 50]])

        with pytest.raises(ValueError, match="divisible"):
            codec.validate(
                shape=(47, 100),
                dtype=Float32(),
                chunk_grid=g,
            )

    def test_sharding_accepts_divisible_rectilinear(self) -> None:
        """Rectilinear shard sizes all divisible by inner chunk_shape should pass."""
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


# ---------------------------------------------------------------------------
# Full-pipeline read/write tests with rectilinear grids
# ---------------------------------------------------------------------------


class TestFullPipelineRectilinear:
    """End-to-end read/write tests through the full Array pipeline."""

    @staticmethod
    def _make_1d(tmp_path: Path) -> tuple[zarr.Array[Any], np.ndarray[Any, Any]]:
        a = np.arange(30, dtype="int32")
        z = zarr.create_array(
            store=tmp_path / "arr1d.zarr",
            shape=(30,),
            chunks=[[5, 10, 15]],
            dtype="int32",
        )
        z[:] = a
        return z, a

    @staticmethod
    def _make_2d(tmp_path: Path) -> tuple[zarr.Array[Any], np.ndarray[Any, Any]]:
        a = np.arange(6000, dtype="int32").reshape(60, 100)
        z = zarr.create_array(
            store=tmp_path / "arr2d.zarr",
            shape=(60, 100),
            chunks=[[10, 20, 30], [25, 25, 25, 25]],
            dtype="int32",
        )
        z[:] = a
        return z, a

    # --- Basic selection ---

    def test_basic_selection_1d(self, tmp_path: Path) -> None:
        z, a = self._make_1d(tmp_path)
        sels: list[Any] = [0, 4, 5, 14, 15, 29, -1, slice(None), slice(3, 18), slice(0, 0)]
        for sel in sels:
            np.testing.assert_array_equal(z[sel], a[sel], err_msg=f"sel={sel}")

    def test_basic_selection_1d_strided(self, tmp_path: Path) -> None:
        z, a = self._make_1d(tmp_path)
        for sel in [slice(None, None, 2), slice(1, 25, 3), slice(0, 30, 7)]:
            np.testing.assert_array_equal(z[sel], a[sel], err_msg=f"sel={sel}")

    def test_basic_selection_2d(self, tmp_path: Path) -> None:
        z, a = self._make_2d(tmp_path)
        selections: list[Any] = [
            42,
            -1,
            (9, 24),
            (10, 25),
            (30, 50),
            (59, 99),
            slice(None),
            (slice(5, 35), slice(20, 80)),
            (slice(0, 10), slice(0, 25)),  # within one chunk
            (slice(10, 10), slice(None)),  # empty
            (slice(None, None, 3), slice(None, None, 7)),  # strided
        ]
        for sel in selections:
            np.testing.assert_array_equal(z[sel], a[sel], err_msg=f"sel={sel}")

    # --- Orthogonal selection ---

    def test_orthogonal_selection_1d_bool(self, tmp_path: Path) -> None:
        z, a = self._make_1d(tmp_path)
        ix = np.zeros(30, dtype=bool)
        ix[[0, 4, 5, 14, 15, 29]] = True
        np.testing.assert_array_equal(z.oindex[ix], a[ix])

    def test_orthogonal_selection_1d_int(self, tmp_path: Path) -> None:
        z, a = self._make_1d(tmp_path)
        ix = np.array([0, 4, 5, 14, 15, 29])
        np.testing.assert_array_equal(z.oindex[ix], a[ix])
        ix_neg = np.array([0, -1, -15, -25])
        np.testing.assert_array_equal(z.oindex[ix_neg], a[ix_neg])

    def test_orthogonal_selection_2d_bool(self, tmp_path: Path) -> None:
        z, a = self._make_2d(tmp_path)
        ix0 = np.zeros(60, dtype=bool)
        ix0[[0, 9, 10, 29, 30, 59]] = True
        ix1 = np.zeros(100, dtype=bool)
        ix1[[0, 24, 25, 49, 50, 99]] = True
        np.testing.assert_array_equal(z.oindex[ix0, ix1], a[np.ix_(ix0, ix1)])

    def test_orthogonal_selection_2d_int(self, tmp_path: Path) -> None:
        z, a = self._make_2d(tmp_path)
        ix0 = np.array([0, 9, 10, 29, 30, 59])
        ix1 = np.array([0, 24, 25, 49, 50, 99])
        np.testing.assert_array_equal(z.oindex[ix0, ix1], a[np.ix_(ix0, ix1)])

    def test_orthogonal_selection_2d_mixed(self, tmp_path: Path) -> None:
        z, a = self._make_2d(tmp_path)
        ix = np.array([0, 9, 10, 29, 30, 59])
        np.testing.assert_array_equal(z.oindex[ix, slice(25, 75)], a[np.ix_(ix, np.arange(25, 75))])
        np.testing.assert_array_equal(
            z.oindex[slice(10, 30), ix[:4]], a[np.ix_(np.arange(10, 30), ix[:4])]
        )

    # --- Coordinate (vindex) selection ---

    def test_coordinate_selection_1d(self, tmp_path: Path) -> None:
        z, a = self._make_1d(tmp_path)
        ix = np.array([0, 4, 5, 14, 15, 29])
        np.testing.assert_array_equal(z.vindex[ix], a[ix])

    def test_coordinate_selection_2d(self, tmp_path: Path) -> None:
        z, a = self._make_2d(tmp_path)
        r = np.array([0, 9, 10, 29, 30, 59])
        c = np.array([0, 24, 25, 49, 50, 99])
        np.testing.assert_array_equal(z.vindex[r, c], a[r, c])

    def test_coordinate_selection_2d_bool_mask(self, tmp_path: Path) -> None:
        z, a = self._make_2d(tmp_path)
        mask = a > 3000
        np.testing.assert_array_equal(z.vindex[mask], a[mask])

    # --- Set selection ---

    def test_set_basic_selection(self, tmp_path: Path) -> None:
        z, a = self._make_2d(tmp_path)
        new_data = np.full((20, 50), -1, dtype="int32")
        z[5:25, 10:60] = new_data
        a[5:25, 10:60] = new_data
        np.testing.assert_array_equal(z[:], a)

    def test_set_orthogonal_selection(self, tmp_path: Path) -> None:
        z, a = self._make_2d(tmp_path)
        rows = np.array([0, 10, 30])
        cols = np.array([0, 25, 50, 75])
        val = np.full((3, 4), -99, dtype="int32")
        z.oindex[rows, cols] = val
        a[np.ix_(rows, cols)] = val
        np.testing.assert_array_equal(z[:], a)

    # --- Higher dimensions ---

    def test_3d_array(self, tmp_path: Path) -> None:
        shape = (12, 20, 15)
        chunk_shapes = [[4, 8], [5, 5, 10], [5, 10]]
        a = np.arange(int(np.prod(shape)), dtype="int32").reshape(shape)
        z = zarr.create_array(
            store=tmp_path / "arr3d.zarr",
            shape=shape,
            chunks=chunk_shapes,
            dtype="int32",
        )
        z[:] = a
        np.testing.assert_array_equal(z[:], a)
        np.testing.assert_array_equal(z[2:10, 3:18, 4:14], a[2:10, 3:18, 4:14])

    def test_1d_single_chunk(self, tmp_path: Path) -> None:
        a = np.arange(20, dtype="int32")
        z = zarr.create_array(
            store=tmp_path / "arr1c.zarr",
            shape=(20,),
            chunks=[[20]],
            dtype="int32",
        )
        z[:] = a
        np.testing.assert_array_equal(z[:], a)

    # --- Persistence roundtrip ---

    def test_persistence_roundtrip(self, tmp_path: Path) -> None:
        _, a = self._make_2d(tmp_path)
        z2 = zarr.open_array(store=tmp_path / "arr2d.zarr", mode="r")
        assert not z2.metadata.chunk_grid.is_regular
        np.testing.assert_array_equal(z2[:], a)

    # --- Highly irregular chunks ---

    def test_highly_irregular_chunks(self, tmp_path: Path) -> None:
        shape = (100, 100)
        chunk_shapes = [[5, 10, 15, 20, 50], [100]]
        a = np.arange(10000, dtype="int32").reshape(shape)
        z = zarr.create_array(
            store=tmp_path / "irreg.zarr",
            shape=shape,
            chunks=chunk_shapes,
            dtype="int32",
        )
        z[:] = a
        np.testing.assert_array_equal(z[:], a)
        np.testing.assert_array_equal(z[3:97, 10:90], a[3:97, 10:90])

    # --- API validation ---

    def test_v2_rejects_rectilinear(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Zarr format 2"):
            zarr.create_array(
                store=tmp_path / "v2.zarr",
                shape=(30,),
                chunks=[[10, 20]],
                dtype="int32",
                zarr_format=2,
            )

    def test_sharding_rejects_rectilinear(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="sharding"):
            zarr.create_array(
                store=tmp_path / "shard.zarr",
                shape=(60, 100),
                chunks=[[10, 20, 30], [25, 25, 25, 25]],
                shards=(30, 50),
                dtype="int32",
            )

    def test_nchunks(self, tmp_path: Path) -> None:
        z, _ = self._make_2d(tmp_path)
        assert z.metadata.chunk_grid.get_nchunks() == 12


# ---------------------------------------------------------------------------
# Hypothesis property-based tests
# ---------------------------------------------------------------------------

pytest.importorskip("hypothesis")

import hypothesis.strategies as st  # noqa: E402
from hypothesis import event, given, settings  # noqa: E402


@st.composite
def rectilinear_chunks_st(draw: st.DrawFn, *, shape: tuple[int, ...]) -> list[list[int]]:
    """Generate valid rectilinear chunk shapes for a given array shape."""
    chunk_shapes: list[list[int]] = []
    for size in shape:
        assert size > 0
        max_chunks = min(size, 10)
        nchunks = draw(st.integers(min_value=1, max_value=max_chunks))
        if nchunks == 1:
            chunk_shapes.append([size])
        else:
            dividers = sorted(
                draw(
                    st.lists(
                        st.integers(min_value=1, max_value=size - 1),
                        min_size=nchunks - 1,
                        max_size=nchunks - 1,
                        unique=True,
                    )
                )
            )
            chunk_shapes.append(
                [a - b for a, b in zip(dividers + [size], [0] + dividers, strict=False)]
            )
    return chunk_shapes


@st.composite
def rectilinear_arrays_st(draw: st.DrawFn) -> tuple[zarr.Array[Any], np.ndarray[Any, Any]]:
    """Generate a rectilinear zarr array with random data, shape, and chunks."""
    from zarr.storage import MemoryStore

    ndim = draw(st.integers(min_value=1, max_value=3))
    shape = draw(st.tuples(*[st.integers(min_value=2, max_value=20) for _ in range(ndim)]))
    chunk_shapes = draw(rectilinear_chunks_st(shape=shape))
    event(f"ndim={ndim}, shape={shape}")

    a = np.arange(int(np.prod(shape)), dtype="int32").reshape(shape)
    store = MemoryStore()
    z = zarr.create_array(store=store, shape=shape, chunks=chunk_shapes, dtype="int32")
    z[:] = a
    return z, a


@settings(deadline=None, max_examples=50)
@given(data=st.data())
def test_property_basic_indexing_rectilinear(data: st.DataObject) -> None:
    """Property test: basic indexing on rectilinear arrays matches numpy."""
    z, a = data.draw(rectilinear_arrays_st())
    np.testing.assert_array_equal(z[:], a)

    slicers = []
    for size in a.shape:
        start = data.draw(st.integers(min_value=0, max_value=size - 1))
        stop = data.draw(st.integers(min_value=start, max_value=size))
        slicers.append(slice(start, stop))
    sel = tuple(slicers)
    np.testing.assert_array_equal(z[sel], a[sel], err_msg=f"sel={sel}")


@settings(deadline=None, max_examples=50)
@given(data=st.data())
def test_property_oindex_rectilinear(data: st.DataObject) -> None:
    """Property test: orthogonal int-array indexing matches numpy."""
    z, a = data.draw(rectilinear_arrays_st())

    indexers_z = []
    indexers_np = []
    for size in a.shape:
        n = data.draw(st.integers(min_value=1, max_value=min(size, 5)))
        ix = np.array(
            sorted(
                data.draw(
                    st.lists(
                        st.integers(min_value=0, max_value=size - 1),
                        min_size=n,
                        max_size=n,
                        unique=True,
                    )
                )
            )
        )
        indexers_z.append(ix)
        indexers_np.append(ix)

    result = z.oindex[tuple(indexers_z)]
    expected = a[np.ix_(*indexers_np)]
    np.testing.assert_array_equal(result, expected)


@settings(deadline=None, max_examples=50)
@given(data=st.data())
def test_property_vindex_rectilinear(data: st.DataObject) -> None:
    """Property test: vindex on rectilinear arrays matches numpy."""
    z, a = data.draw(rectilinear_arrays_st())

    n = data.draw(st.integers(min_value=1, max_value=min(min(a.shape), 5)))
    indexers = tuple(
        np.array(
            data.draw(
                st.lists(
                    st.integers(min_value=0, max_value=size - 1),
                    min_size=n,
                    max_size=n,
                )
            )
        )
        for size in a.shape
    )
    np.testing.assert_array_equal(z.vindex[indexers], a[indexers])


@settings(deadline=None, max_examples=50)
@given(data=st.data())
def test_property_roundtrip_rectilinear(data: st.DataObject) -> None:
    """Property test: write then read matches original data."""
    z, a = data.draw(rectilinear_arrays_st())
    np.testing.assert_array_equal(z[:], a)


# ---------------------------------------------------------------------------
# Bug #3: _resize with rectilinear grids
# ---------------------------------------------------------------------------


class TestResizeRectilinear:
    def test_resize_regular_preserves_chunk_grid(self, tmp_path: Path) -> None:
        """Resize a regular array — chunk_grid extents must match new shape."""
        z = zarr.create_array(
            store=tmp_path / "regular.zarr",
            shape=(100,),
            chunks=(10,),
            dtype="int32",
        )
        z[:] = np.arange(100, dtype="int32")
        z.resize(50)
        assert z.shape == (50,)
        # The chunk grid's extent must agree with the new shape
        assert z.metadata.chunk_grid.dimensions[0].extent == 50

    def test_resize_rectilinear_raises(self, tmp_path: Path) -> None:
        """Resize should raise for rectilinear grids (not yet supported)."""
        z = zarr.create_array(
            store=tmp_path / "rect.zarr",
            shape=(30,),
            chunks=[[5, 10, 15]],
            dtype="int32",
        )
        z[:] = np.arange(30, dtype="int32")
        with pytest.raises((ValueError, NotImplementedError)):
            z.resize(20)


# ---------------------------------------------------------------------------
# Bug #4: extent=0 placeholder in RegularChunkGrid / from_dict
# ---------------------------------------------------------------------------


class TestExtentPlaceholder:
    def test_regular_chunk_grid_chunk_shape_preserved(self) -> None:
        """RegularChunkGrid preserves chunk_shape."""
        g = RegularChunkGrid(chunk_shape=(10, 20))
        assert g.chunk_shape == (10, 20)

    def test_regular_chunk_grid_nchunks_raises(self) -> None:
        """RegularChunkGrid raises on get_nchunks() (no extent info)."""
        g = RegularChunkGrid(chunk_shape=(10, 20))
        with pytest.raises(ValueError, match="array shape"):
            g.get_nchunks()

    def test_regular_chunk_grid_shape_raises(self) -> None:
        """RegularChunkGrid raises on .shape (no extent info)."""
        g = RegularChunkGrid(chunk_shape=(10, 20))
        with pytest.raises(ValueError, match="array shape"):
            _ = g.shape

    def test_regular_chunk_grid_all_chunk_coords_raises(self) -> None:
        """RegularChunkGrid raises on all_chunk_coords() (no extent info)."""
        g = RegularChunkGrid(chunk_shape=(10, 20))
        with pytest.raises(ValueError, match="array shape"):
            list(g.all_chunk_coords())

    def test_from_dict_regular_raises_on_extent_ops(self) -> None:
        """ChunkGrid.from_dict for regular grids raises on extent-dependent ops."""
        g = ChunkGrid.from_dict({"name": "regular", "configuration": {"chunk_shape": [10, 20]}})
        assert g.chunk_shape == (10, 20)
        with pytest.raises(ValueError, match="array shape"):
            g.get_nchunks()

    def test_from_dict_regular_is_regular_chunk_grid(self) -> None:
        """ChunkGrid.from_dict for regular grids returns a RegularChunkGrid."""
        g = ChunkGrid.from_dict({"name": "regular", "configuration": {"chunk_shape": [10, 20]}})
        assert isinstance(g, RegularChunkGrid)
