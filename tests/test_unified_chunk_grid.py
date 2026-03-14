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
from zarr.core.chunk_grids import (
    ChunkGrid,
    ChunkSpec,
    FixedDimension,
    VaryingDimension,
    _compress_rle,
    _decode_dim_spec,
    _expand_rle,
    _infer_chunk_grid_name,
    _is_rectilinear_chunks,
    parse_chunk_grid,
    serialize_chunk_grid,
)
from zarr.errors import BoundsCheckError
from zarr.storage import MemoryStore

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


@pytest.fixture(autouse=True)
def _enable_rectilinear_chunks() -> Generator[None, None, None]:
    """Enable rectilinear chunks for all tests in this module."""
    with zarr.config.set({"array.rectilinear_chunks": True}):
        yield


def _edges(grid: ChunkGrid, dim: int) -> tuple[int, ...]:
    """Extract the per-chunk edge lengths for *dim* from a ChunkGrid."""
    d = grid.dimensions[dim]
    if isinstance(d, FixedDimension):
        return tuple(d.size for _ in range(d.nchunks))
    if isinstance(d, VaryingDimension):
        return tuple(d.edges)
    raise TypeError(f"Unexpected dimension type: {type(d)}")


# ---------------------------------------------------------------------------
# Index to chunk
# ---------------------------------------------------------------------------


class TestVaryingDimensionIndexToChunkBounds:
    def test_index_at_extent_raises(self) -> None:
        """index_to_chunk(extent) should raise since extent is out of bounds."""
        dim = VaryingDimension([10, 20, 30], extent=60)
        with pytest.raises(IndexError, match="out of bounds"):
            dim.index_to_chunk(60)

    def test_index_past_extent_raises(self) -> None:
        dim = VaryingDimension([10, 20, 30], extent=60)
        with pytest.raises(IndexError, match="out of bounds"):
            dim.index_to_chunk(100)

    def test_last_valid_index_works(self) -> None:
        dim = VaryingDimension([10, 20, 30], extent=60)
        assert dim.index_to_chunk(59) == 2


class TestFixedDimensionIndexToChunkBounds:
    def test_negative_index_raises(self) -> None:
        """index_to_chunk(-1) should raise, not silently return -1."""
        dim = FixedDimension(size=10, extent=95)
        with pytest.raises(IndexError, match="Negative"):
            dim.index_to_chunk(-1)

    def test_index_at_extent_raises(self) -> None:
        dim = FixedDimension(size=10, extent=95)
        with pytest.raises(IndexError, match="out of bounds"):
            dim.index_to_chunk(95)

    def test_last_valid_index_works(self) -> None:
        dim = FixedDimension(size=10, extent=95)
        assert dim.index_to_chunk(94) == 9


# ---------------------------------------------------------------------------
# Feature flag gating
# ---------------------------------------------------------------------------


class TestRectilinearFeatureFlag:
    """Test that rectilinear chunks are gated behind the config flag."""

    def test_disabled_by_default(self) -> None:
        with zarr.config.set({"array.rectilinear_chunks": False}):
            with pytest.raises(ValueError, match="experimental and disabled by default"):
                ChunkGrid.from_rectilinear([[10, 20], [25, 25]], array_shape=(30, 50))

    def test_enabled_via_config(self) -> None:
        with zarr.config.set({"array.rectilinear_chunks": True}):
            g = ChunkGrid.from_rectilinear([[10, 20], [25, 25]], array_shape=(30, 50))
            assert g.ndim == 2

    def test_create_array_blocked(self) -> None:
        with zarr.config.set({"array.rectilinear_chunks": False}):
            store = MemoryStore()
            with pytest.raises(ValueError, match="experimental and disabled by default"):
                zarr.create_array(store, shape=(30,), chunks=[[10, 20]], dtype="int32")

    def test_parse_chunk_grid_blocked(self) -> None:
        """Opening a rectilinear array from metadata is also gated."""
        with zarr.config.set({"array.rectilinear_chunks": False}):
            with pytest.raises(ValueError, match="experimental and disabled by default"):
                parse_chunk_grid(
                    {
                        "name": "rectilinear",
                        "configuration": {
                            "kind": "inline",
                            "chunk_shapes": [[10, 20, 30], [50, 50]],
                        },
                    },
                    array_shape=(60, 100),
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

    def test_chunk_offset_oob_raises(self) -> None:
        d = FixedDimension(size=10, extent=100)
        with pytest.raises(IndexError, match="out of bounds"):
            d.chunk_offset(10)
        with pytest.raises(IndexError, match="out of bounds"):
            d.chunk_offset(-1)

    def test_chunk_size_oob_raises(self) -> None:
        d = FixedDimension(size=10, extent=100)
        with pytest.raises(IndexError, match="out of bounds"):
            d.chunk_size(10)
        with pytest.raises(IndexError, match="out of bounds"):
            d.chunk_size(-1)

    def test_data_size_oob_raises(self) -> None:
        d = FixedDimension(size=10, extent=100)
        with pytest.raises(IndexError, match="out of bounds"):
            d.data_size(10)
        with pytest.raises(IndexError, match="out of bounds"):
            d.data_size(-1)


# ---------------------------------------------------------------------------
# VaryingDimension
# ---------------------------------------------------------------------------


class TestVaryingDimension:
    def test_basic(self) -> None:
        d = VaryingDimension([10, 20, 30], extent=60)
        assert d.edges == (10, 20, 30)
        assert d.cumulative == (10, 30, 60)
        assert d.nchunks == 3
        assert d.extent == 60

    def test_index_to_chunk(self) -> None:
        d = VaryingDimension([10, 20, 30], extent=60)
        assert d.index_to_chunk(0) == 0
        assert d.index_to_chunk(9) == 0
        assert d.index_to_chunk(10) == 1
        assert d.index_to_chunk(29) == 1
        assert d.index_to_chunk(30) == 2
        assert d.index_to_chunk(59) == 2

    def test_chunk_offset(self) -> None:
        d = VaryingDimension([10, 20, 30], extent=60)
        assert d.chunk_offset(0) == 0
        assert d.chunk_offset(1) == 10
        assert d.chunk_offset(2) == 30

    def test_chunk_size(self) -> None:
        d = VaryingDimension([10, 20, 30], extent=60)
        assert d.chunk_size(0) == 10
        assert d.chunk_size(1) == 20
        assert d.chunk_size(2) == 30

    def test_data_size(self) -> None:
        d = VaryingDimension([10, 20, 30], extent=60)
        # data_size == chunk_size when extent == sum(edges) (no boundary)
        assert d.data_size(0) == 10
        assert d.data_size(1) == 20
        assert d.data_size(2) == 30

    def test_vectorized(self) -> None:
        d = VaryingDimension([10, 20, 30], extent=60)
        indices = np.array([0, 9, 10, 29, 30, 59])
        chunks = d.indices_to_chunks(indices)
        np.testing.assert_array_equal(chunks, [0, 0, 1, 1, 2, 2])

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            VaryingDimension([], extent=0)

    def test_zero_edge_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be > 0"):
            VaryingDimension([10, 0, 5], extent=15)


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
        g = ChunkGrid.from_rectilinear([[10, 20, 30], [25, 25, 25, 25]], array_shape=(60, 100))
        assert not g.is_regular
        assert g.ndim == 2
        with pytest.raises(ValueError, match="only available for regular"):
            _ = g.chunk_shape

    def test_rectilinear_with_uniform_dim(self) -> None:
        """A rectilinear grid with all-same sizes in one dim stores it as Fixed."""
        g = ChunkGrid.from_rectilinear([[10, 20, 30], [25, 25, 25, 25]], array_shape=(60, 100))
        assert isinstance(g.dimensions[0], VaryingDimension)
        assert isinstance(g.dimensions[1], FixedDimension)

    def test_all_uniform_becomes_regular(self) -> None:
        """If all dimensions have uniform sizes, the grid is regular."""
        g = ChunkGrid.from_rectilinear([[10, 10, 10], [25, 25]], array_shape=(30, 50))
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
        g = ChunkGrid.from_rectilinear([[10, 20, 30], [25, 25, 25, 25]], array_shape=(60, 100))
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
        g = ChunkGrid.from_rectilinear([[10, 20, 30], [25, 25, 25, 25]], array_shape=(60, 100))
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
        g = ChunkGrid.from_rectilinear([[10, 20, 30], [25, 25, 25, 25]], array_shape=(60, 100))
        spec = g[(1, 2)]
        assert spec is not None
        assert spec.slices == (slice(10, 30, 1), slice(50, 75, 1))

    def test_all_chunk_coords(self) -> None:
        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]], array_shape=(60, 100))
        coords = list(g.all_chunk_coords())
        assert len(coords) == 6
        assert coords[0] == (0, 0)
        assert coords[-1] == (2, 1)

    def test_all_chunk_coords_with_origin(self) -> None:
        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]], array_shape=(60, 100))
        coords = list(g.all_chunk_coords(origin=(1, 0)))
        assert len(coords) == 4  # 2 remaining in dim0 * 2 in dim1
        assert coords[0] == (1, 0)
        assert coords[-1] == (2, 1)

    def test_all_chunk_coords_with_selection_shape(self) -> None:
        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]], array_shape=(60, 100))
        coords = list(g.all_chunk_coords(selection_shape=(2, 1)))
        assert len(coords) == 2
        assert coords == [(0, 0), (1, 0)]

    def test_all_chunk_coords_with_origin_and_selection_shape(self) -> None:
        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]], array_shape=(60, 100))
        coords = list(g.all_chunk_coords(origin=(1, 1), selection_shape=(2, 1)))
        assert coords == [(1, 1), (2, 1)]

    def test_all_chunk_coords_origin_at_last_chunk(self) -> None:
        g = ChunkGrid.from_regular((30, 40), (10, 20))
        coords = list(g.all_chunk_coords(origin=(2, 1)))
        assert coords == [(2, 1)]

    def test_all_chunk_coords_selection_shape_zero(self) -> None:
        g = ChunkGrid.from_regular((30, 40), (10, 20))
        coords = list(g.all_chunk_coords(selection_shape=(0, 0)))
        assert coords == []

    def test_all_chunk_coords_single_dim_slice(self) -> None:
        """Origin shifts one dim, selection_shape restricts the other."""
        g = ChunkGrid.from_regular((60, 80), (20, 20))  # 3x4
        coords = list(g.all_chunk_coords(origin=(0, 2), selection_shape=(3, 1)))
        assert coords == [(0, 2), (1, 2), (2, 2)]

    def test_get_nchunks(self) -> None:
        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]], array_shape=(60, 100))
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
        assert _compress_rle([10, 10, 20]) == [[10, 2], 20]
        assert _compress_rle([5]) == [5]
        assert _compress_rle([10, 20, 30]) == [10, 20, 30]

    def test_roundtrip(self) -> None:
        original = [10, 10, 10, 20, 20, 30]
        compressed = _compress_rle(original)
        assert _expand_rle(compressed) == original


class TestExpandRleHandlesJsonFloats:
    def test_bare_integer_floats_accepted(self) -> None:
        """JSON parsers may emit 10.0 for the integer 10; _expand_rle should handle it."""
        result = _expand_rle([10.0, 20.0])
        assert result == [10, 20]

    def test_rle_pair_with_float_count(self) -> None:
        result = _expand_rle([[10, 3.0]])
        assert result == [10, 10, 10]


# ---------------------------------------------------------------------------
# _decode_dim_spec edge cases
# ---------------------------------------------------------------------------


class TestDecodeDimSpec:
    """Edge cases for _decode_dim_spec: floats, empty lists, negatives, missing extent."""

    def test_bare_integer(self) -> None:
        assert _decode_dim_spec(10, array_extent=25) == [10, 10, 10]

    def test_bare_integer_exact_fit(self) -> None:
        assert _decode_dim_spec(5, array_extent=10) == [5, 5]

    def test_bare_integer_no_extent_raises(self) -> None:
        with pytest.raises(ValueError, match="requires array shape"):
            _decode_dim_spec(10, array_extent=None)

    def test_bare_integer_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="must be > 0"):
            _decode_dim_spec(0, array_extent=10)

    def test_bare_integer_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="must be > 0"):
            _decode_dim_spec(-5, array_extent=10)

    def test_bare_float_raises(self) -> None:
        """A bare float (not in a list) is not int or list — should raise."""
        with pytest.raises(ValueError, match="Invalid chunk_shapes entry"):
            _decode_dim_spec(10.0, array_extent=10)

    def test_explicit_integer_list(self) -> None:
        assert _decode_dim_spec([10, 20, 30]) == [10, 20, 30]

    def test_empty_list(self) -> None:
        """An empty list has no sub-lists, so it returns an empty explicit list."""
        assert _decode_dim_spec([]) == []

    def test_list_with_rle(self) -> None:
        assert _decode_dim_spec([[5, 3], 10]) == [5, 5, 5, 10]

    def test_string_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid chunk_shapes entry"):
            _decode_dim_spec("auto", array_extent=10)

    def test_none_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid chunk_shapes entry"):
            _decode_dim_spec(None, array_extent=10)


# ---------------------------------------------------------------------------
# _is_rectilinear_chunks edge cases
# ---------------------------------------------------------------------------


class TestIsRectilinearChunks:
    """Edge cases for _is_rectilinear_chunks."""

    def test_nested_lists(self) -> None:
        assert _is_rectilinear_chunks([[10, 20], [5, 5]]) is True

    def test_nested_tuples(self) -> None:
        assert _is_rectilinear_chunks(((10, 20), (5, 5))) is True

    def test_flat_tuple(self) -> None:
        assert _is_rectilinear_chunks((10, 20)) is False

    def test_flat_list(self) -> None:
        assert _is_rectilinear_chunks([10, 20]) is False

    def test_single_int(self) -> None:
        assert _is_rectilinear_chunks(10) is False

    def test_string(self) -> None:
        assert _is_rectilinear_chunks("auto") is False

    def test_empty_list(self) -> None:
        assert _is_rectilinear_chunks([]) is False

    def test_empty_nested_list(self) -> None:
        """First element is an empty list — it's iterable and not str/int."""
        assert _is_rectilinear_chunks([[]]) is True

    def test_chunk_grid_instance(self) -> None:
        g = ChunkGrid.from_regular((10,), (5,))
        assert _is_rectilinear_chunks(g) is False

    def test_none(self) -> None:
        assert _is_rectilinear_chunks(None) is False

    def test_float(self) -> None:
        assert _is_rectilinear_chunks(3.14) is False


# ---------------------------------------------------------------------------
# _infer_chunk_grid_name edge cases
# ---------------------------------------------------------------------------


class TestInferChunkGridName:
    """Edge cases for _infer_chunk_grid_name."""

    def test_regular_grid(self) -> None:
        g = ChunkGrid.from_regular((100,), (10,))
        assert _infer_chunk_grid_name(g, g) == "regular"

    @pytest.fixture(autouse=True)
    def _enable_rectilinear(self) -> Any:
        with zarr.config.set({"array.rectilinear_chunks": True}):
            yield

    def test_rectilinear_grid(self) -> None:
        g = ChunkGrid.from_rectilinear([[10, 20, 30]], array_shape=(60,))
        assert _infer_chunk_grid_name(g, g) == "rectilinear"

    def test_dict_with_regular_name(self) -> None:
        g = ChunkGrid.from_regular((100,), (10,))
        d: dict[str, Any] = {"name": "regular", "configuration": {"chunk_shape": [10]}}
        assert _infer_chunk_grid_name(d, g) == "regular"

    def test_dict_with_rectilinear_name(self) -> None:
        g = ChunkGrid.from_regular((100,), (10,))
        d: dict[str, Any] = {
            "name": "rectilinear",
            "configuration": {"kind": "inline", "chunk_shapes": [10]},
        }
        assert _infer_chunk_grid_name(d, g) == "rectilinear"


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
        g2 = parse_chunk_grid(d, (100, 200))
        assert g2.is_regular
        assert g2.chunk_shape == (10, 20)

    def test_rectilinear_roundtrip(self) -> None:
        g = ChunkGrid.from_rectilinear([[10, 20, 30], [25, 25, 25, 25]], array_shape=(60, 100))
        d = serialize_chunk_grid(g, "rectilinear")
        assert d["name"] == "rectilinear"
        g2 = parse_chunk_grid(d, (60, 100))
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
        g = ChunkGrid.from_rectilinear([[100] * 10, [25, 25, 25, 25]], array_shape=(1000, 100))
        # All uniform, but name is chosen by the metadata layer, not the grid.
        # Serializing as "regular" works because is_regular is True.
        d = serialize_chunk_grid(g, "regular")
        assert d["name"] == "regular"

    def test_rectilinear_uniform_stays_rectilinear(self) -> None:
        """A rectilinear grid with uniform edges stays rectilinear if the name says so."""
        g = ChunkGrid.from_rectilinear([[100] * 10, [25, 25, 25, 25]], array_shape=(1000, 100))
        d = serialize_chunk_grid(g, "rectilinear")
        assert d["name"] == "rectilinear"

    def test_rectilinear_rle_with_varying(self) -> None:
        g = ChunkGrid.from_rectilinear(
            [[100, 100, 100, 50], [25, 25, 25, 25]], array_shape=(350, 100)
        )
        d = serialize_chunk_grid(g, "rectilinear")
        assert d["name"] == "rectilinear"
        config = d["configuration"]
        assert isinstance(config, dict)
        chunk_shapes = config["chunk_shapes"]
        assert isinstance(chunk_shapes, list)
        assert chunk_shapes[0] == [[100, 3], 50]

    def test_json_roundtrip(self) -> None:
        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]], array_shape=(60, 100))
        d = serialize_chunk_grid(g, "rectilinear")
        json_str = json.dumps(d)
        d2 = json.loads(json_str)
        g2 = parse_chunk_grid(d2, (60, 100))
        assert g2.shape == (3, 2)

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown chunk grid"):
            parse_chunk_grid({"name": "hexagonal", "configuration": {}}, (10,))

    def test_serialize_non_regular_as_regular_raises(self) -> None:
        g = ChunkGrid.from_rectilinear([[10, 20, 30], [25, 25, 25, 25]], array_shape=(60, 100))
        with pytest.raises(ValueError, match="Cannot serialize a non-regular chunk grid"):
            serialize_chunk_grid(g, "regular")

    def test_serialize_unknown_name_raises(self) -> None:
        g = ChunkGrid.from_regular((100,), (10,))
        with pytest.raises(ValueError, match="Unknown chunk grid name for serialization"):
            serialize_chunk_grid(g, "hexagonal")

    def test_zero_extent_rectilinear_raises(self) -> None:
        """Zero-extent grids cannot be serialized as rectilinear (spec requires positive edges)."""
        grid = ChunkGrid.from_regular((0,), (10,))
        with pytest.raises(ValueError, match="zero-extent"):
            serialize_chunk_grid(grid, "rectilinear")


class TestSpecCompliance:
    """Tests for compliance with the rectilinear chunk grid extension spec
    (zarr-extensions PR #25)."""

    def test_kind_inline_required_on_deserialize(self) -> None:
        """Deserialization requires kind: 'inline'."""
        data: dict[str, Any] = {
            "name": "rectilinear",
            "configuration": {"chunk_shapes": [[10, 20], [15, 15]]},
        }
        with pytest.raises(ValueError, match="requires a 'kind' field"):
            parse_chunk_grid(data, (30, 30))

    def test_kind_unknown_rejected(self) -> None:
        data: dict[str, Any] = {
            "name": "rectilinear",
            "configuration": {"kind": "reference", "chunk_shapes": [[10, 20], [15, 15]]},
        }
        with pytest.raises(ValueError, match="Unsupported rectilinear chunk grid kind"):
            parse_chunk_grid(data, (30, 30))

    def test_kind_inline_in_serialized_output(self) -> None:
        """Serialization includes kind: 'inline'."""
        g = ChunkGrid.from_rectilinear([[10, 20, 30], [25, 25]], array_shape=(60, 50))
        d = serialize_chunk_grid(g, "rectilinear")
        config = d["configuration"]
        assert isinstance(config, dict)
        assert config["kind"] == "inline"

    def test_integer_shorthand_per_dimension(self) -> None:
        """A bare integer in chunk_shapes means repeat until >= extent."""
        data: dict[str, Any] = {
            "name": "rectilinear",
            "configuration": {"kind": "inline", "chunk_shapes": [4, [1, 2, 3]]},
        }
        g = parse_chunk_grid(data, (6, 6))
        # 4 repeated: ceildiv(6, 4) = 2 → [4, 4]
        assert _edges(g, 0) == (4, 4)
        assert _edges(g, 1) == (1, 2, 3)

    def test_mixed_rle_and_bare_integers(self) -> None:
        """An array can mix bare integers and [value, count] RLE pairs."""
        data: dict[str, Any] = {
            "name": "rectilinear",
            "configuration": {"kind": "inline", "chunk_shapes": [[[1, 3], 3]]},
        }
        # [[1, 3], 3] → [1, 1, 1, 3] → sum = 6
        g = parse_chunk_grid(data, (6,))
        assert _edges(g, 0) == (1, 1, 1, 3)

    def test_overflow_chunks_allowed(self) -> None:
        """Edge sum >= extent is valid (overflow chunks permitted)."""
        data: dict[str, Any] = {
            "name": "rectilinear",
            "configuration": {"kind": "inline", "chunk_shapes": [[4, 4, 4]]},
        }
        # sum = 12 > extent = 6 — allowed per spec
        g = parse_chunk_grid(data, (6,))
        assert _edges(g, 0) == (4, 4, 4)

    def test_spec_example(self) -> None:
        """The full example from the spec README."""
        data: dict[str, Any] = {
            "name": "rectilinear",
            "configuration": {
                "kind": "inline",
                "chunk_shapes": [
                    4,  # integer shorthand → [4, 4]
                    [1, 2, 3],  # explicit list
                    [[4, 2]],  # pure RLE → [4, 4]
                    [[1, 3], 3],  # mixed RLE + bare → [1, 1, 1, 3]
                    [4, 4, 4],  # explicit list with overflow
                ],
            },
        }
        g = parse_chunk_grid(data, (6, 6, 6, 6, 6))
        assert _edges(g, 0) == (4, 4)
        assert _edges(g, 1) == (1, 2, 3)
        assert _edges(g, 2) == (4, 4)
        assert _edges(g, 3) == (1, 1, 1, 3)
        assert _edges(g, 4) == (4, 4, 4)


class TestParseChunkGridValidation:
    def test_varying_extent_mismatch_raises(self) -> None:
        from zarr.core.chunk_grids import parse_chunk_grid

        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]], array_shape=(60, 100))
        # VaryingDimension extent is 60, but array_shape says 100
        with pytest.raises(ValueError, match="extent"):
            parse_chunk_grid(g, (100, 100))

    def test_varying_extent_match_ok(self) -> None:
        from zarr.core.chunk_grids import parse_chunk_grid

        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]], array_shape=(60, 100))
        # Matching extents should work fine
        g2 = parse_chunk_grid(g, (60, 100))
        assert g2.dimensions[0].extent == 60

    def test_rectilinear_extent_mismatch_raises(self) -> None:
        """sum(edges) must match the array shape for each dimension."""
        data: dict[str, Any] = {
            "name": "rectilinear",
            "configuration": {"kind": "inline", "chunk_shapes": [[10, 20, 30], [25, 25]]},
        }
        # sum([10,20,30])=60, sum([25,25])=50 — array shape (100, 50) mismatches dim 0
        with pytest.raises(ValueError, match="sum to 60 but array shape extent is 100"):
            parse_chunk_grid(data, (100, 50))

    def test_rectilinear_extent_mismatch_second_dim(self) -> None:
        data: dict[str, Any] = {
            "name": "rectilinear",
            "configuration": {"kind": "inline", "chunk_shapes": [[50, 50], [10, 20]]},
        }
        # dim 0 OK (100), dim 1: sum([10,20])=30 != 50
        with pytest.raises(ValueError, match="dimension 1 sum to 30 but array shape extent is 50"):
            parse_chunk_grid(data, (100, 50))

    def test_rectilinear_extent_match_passes(self) -> None:
        data: dict[str, Any] = {
            "name": "rectilinear",
            "configuration": {"kind": "inline", "chunk_shapes": [[10, 20, 30], [25, 25]]},
        }
        g = parse_chunk_grid(data, (60, 50))
        assert g.shape == (3, 2)

    def test_rectilinear_ndim_mismatch_raises(self) -> None:
        data: dict[str, Any] = {
            "name": "rectilinear",
            "configuration": {"kind": "inline", "chunk_shapes": [[10, 20], [25, 25]]},
        }
        with pytest.raises(ValueError, match="2 dimensions but array shape has 3"):
            parse_chunk_grid(data, (30, 50, 100))

    def test_rectilinear_rle_extent_validated(self) -> None:
        """RLE-encoded edges are expanded before validation."""
        data: dict[str, Any] = {
            "name": "rectilinear",
            "configuration": {"kind": "inline", "chunk_shapes": [[[10, 5]], [[25, 2]]]},
        }
        # sum = 50 and 50 — match (50, 50)
        g = parse_chunk_grid(data, (50, 50))
        assert g.shape == (5, 2)
        # mismatch
        with pytest.raises(ValueError, match="sum to 50 but array shape extent is 100"):
            parse_chunk_grid(data, (100, 50))

    def test_varying_dimension_extent_mismatch_on_chunkgrid_input(self) -> None:
        """When passing a ChunkGrid directly, VaryingDimension extent is validated.

        After resize, extent >= array_shape is allowed (last chunk extends past
        boundary). But extent < array_shape is still an error.
        """
        g = ChunkGrid.from_rectilinear([[10, 20, 30], [25, 25]], array_shape=(60, 50))
        with pytest.raises(ValueError, match="less than"):
            parse_chunk_grid(g, (100, 50))


class TestRectilinearRoundTripPreservesCodecShape:
    def test_boundary_chunk_codec_size_preserved(self) -> None:
        """Round-tripping through rectilinear should not change codec buffer sizes."""
        grid = ChunkGrid.from_regular((95,), (10,))
        original_codec_size = grid.dimensions[0].chunk_size(9)
        assert original_codec_size == 10

        serialized = serialize_chunk_grid(grid, "rectilinear")
        parsed = parse_chunk_grid(serialized, (95,))

        roundtripped_codec_size = parsed.dimensions[0].chunk_size(9)
        assert roundtripped_codec_size == original_codec_size, (
            f"codec buffer changed from {original_codec_size} to "
            f"{roundtripped_codec_size} after round-trip"
        )

    def test_single_chunk_boundary_codec_size_preserved(self) -> None:
        """shape=7, chunk_size=10: single chunk's codec buffer should stay 10."""
        grid = ChunkGrid.from_regular((7,), (10,))
        assert grid.dimensions[0].chunk_size(0) == 10

        serialized = serialize_chunk_grid(grid, "rectilinear")
        parsed = parse_chunk_grid(serialized, (7,))

        assert parsed.dimensions[0].chunk_size(0) == 10


# ---------------------------------------------------------------------------
# Indexing with rectilinear grids
# ---------------------------------------------------------------------------


class TestRectilinearIndexing:
    """Test that the indexing pipeline works with VaryingDimension."""

    def test_basic_indexer_rectilinear(self) -> None:
        from zarr.core.indexing import BasicIndexer

        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]], array_shape=(60, 100))
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

        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]], array_shape=(60, 100))
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

        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]], array_shape=(60, 100))
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

        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]], array_shape=(60, 100))
        indexer = OrthogonalIndexer(
            selection=(slice(None), slice(None)),
            shape=(60, 100),
            chunk_grid=g,
        )
        projections = list(indexer)
        assert len(projections) == 6

    def test_oob_block_raises_bounds_check_error(self) -> None:
        """Out-of-bounds block index should raise BoundsCheckError, not IndexError."""
        store = MemoryStore()
        a = zarr.create_array(store, shape=(30,), chunks=[[10, 20]], dtype="int32")
        with pytest.raises(BoundsCheckError):
            a.get_block_selection((2,))


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

        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]], array_shape=(60, 100))

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
        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]], array_shape=(60, 100))
        d = serialize_chunk_grid(g, "rectilinear")
        g2 = parse_chunk_grid(d, (60, 100))
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

        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]], array_shape=(60, 100))
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
                "configuration": {"kind": "inline", "chunk_shapes": [[[50, 2]], [[25, 4]]]},
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

        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]], array_shape=(60, 100))
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
        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]], array_shape=(60, 100))

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
        """data_size raises IndexError for out-of-bounds chunk indices."""
        d = FixedDimension(size=10, extent=100)
        with pytest.raises(IndexError, match="out of bounds"):
            d.data_size(10)  # exactly at boundary
        with pytest.raises(IndexError, match="out of bounds"):
            d.data_size(11)  # past boundary
        with pytest.raises(IndexError, match="out of bounds"):
            d.data_size(999)
        with pytest.raises(IndexError, match="out of bounds"):
            d.data_size(-1)

    def test_fixed_dim_data_size_boundary_oob(self) -> None:
        """data_size raises IndexError past last chunk."""
        d = FixedDimension(size=10, extent=95)
        with pytest.raises(IndexError, match="out of bounds"):
            d.data_size(10)  # past nchunks=10

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
                VaryingDimension([10, 20, 30], extent=60),
                FixedDimension(size=10, extent=95),
            )
        )
        assert g.shape == (3, 10)

        d = serialize_chunk_grid(g, "rectilinear")
        assert d["name"] == "rectilinear"
        # Second dim serializes as bare integer (per rectilinear spec,
        # a bare integer repeats until sum >= extent, preserving full
        # codec buffer size for boundary chunks).
        config = d["configuration"]
        assert isinstance(config, dict)
        chunk_shapes = config["chunk_shapes"]
        assert isinstance(chunk_shapes, list)
        assert chunk_shapes[1] == 10  # bare integer shorthand

        g2 = parse_chunk_grid(d, (60, 95))
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
                VaryingDimension([10, 20], extent=30),
                FixedDimension(size=25, extent=100),
            )
        )
        d = serialize_chunk_grid(g, "rectilinear")
        g2 = parse_chunk_grid(d, (30, 100))
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
        # No valid chunk index exists on a 0-chunk dimension
        with pytest.raises(IndexError, match="out of bounds"):
            d.data_size(0)
        with pytest.raises(IndexError, match="out of bounds"):
            d.chunk_size(0)

    def test_zero_extent_nonzero_size(self) -> None:
        """FixedDimension(size=10, extent=0) => 0 chunks."""
        d = FixedDimension(size=10, extent=0)
        assert d.nchunks == 0
        # No valid chunk index exists on a 0-chunk dimension
        with pytest.raises(IndexError, match="out of bounds"):
            d.data_size(0)

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

        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]], array_shape=(60, 100))
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

    def test_getitem_int_1d_regular(self) -> None:
        """Integer indexing works for 1-d regular grids."""
        g = ChunkGrid.from_regular((100,), (10,))
        spec = g[0]
        assert spec is not None
        assert spec.shape == (10,)
        assert spec.slices == (slice(0, 10, 1),)
        # Boundary chunk
        spec = g[9]
        assert spec is not None
        assert spec.shape == (10,)

    def test_getitem_int_1d_rectilinear(self) -> None:
        """Integer indexing works for 1-d rectilinear grids."""
        g = ChunkGrid.from_rectilinear([[20, 30, 50]], array_shape=(100,))
        spec = g[0]
        assert spec is not None
        assert spec.shape == (20,)
        spec = g[1]
        assert spec is not None
        assert spec.shape == (30,)
        spec = g[2]
        assert spec is not None
        assert spec.shape == (50,)

    def test_getitem_int_0d_raises(self) -> None:
        """Integer indexing raises ValueError for 0-d grids (ndim mismatch)."""
        g = ChunkGrid.from_regular((), ())
        with pytest.raises(ValueError, match="Expected 0 coordinate.*got 1"):
            g[0]

    def test_getitem_int_2d_raises(self) -> None:
        """Integer indexing raises ValueError for 2-d grids (ndim mismatch)."""
        g = ChunkGrid.from_regular((100, 200), (10, 20))
        with pytest.raises(ValueError, match="Expected 2 coordinate.*got 1"):
            g[0]

    def test_getitem_int_oob_returns_none(self) -> None:
        """Integer OOB returns None for 1-d grid."""
        g = ChunkGrid.from_regular((100,), (10,))
        assert g[10] is None
        assert g[99] is None

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

    def test_zero_nchunks_fixed_dim_in_rectilinear_serialize_raises(self) -> None:
        """A rectilinear grid with a 0-extent dimension cannot be serialized."""
        g = ChunkGrid(
            dimensions=(
                VaryingDimension([10, 20], extent=30),
                FixedDimension(size=10, extent=0),
            )
        )
        assert g.shape == (2, 0)
        with pytest.raises(ValueError, match="zero-extent"):
            serialize_chunk_grid(g, "rectilinear")

    # -- VaryingDimension data_size --

    def test_varying_dim_data_size_equals_chunk_size(self) -> None:
        """For VaryingDimension, data_size == chunk_size (no padding)."""
        d = VaryingDimension([10, 20, 5], extent=35)
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

        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]], array_shape=(60, 100))
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

        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]], array_shape=(60, 100))
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

    def test_orthogonal_advanced_indexing_produces_correct_projections(self) -> None:
        """Verify OrthogonalIndexer produces correct chunk projections
        for advanced indexing with VaryingDimension."""
        from zarr.core.indexing import OrthogonalIndexer

        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]], array_shape=(60, 100))
        indexer = OrthogonalIndexer(
            selection=(np.array([5, 15]), slice(None)),
            shape=(60, 100),
            chunk_grid=g,
        )
        projections = list(indexer)
        # index 5 is in chunk 0 (edges [10,...]), index 15 is in chunk 1 (edges [...,20,...])
        # dim 1 slice(None) covers both chunks [50, 50]
        # cartesian product: 2 chunks in dim 0 x 2 chunks in dim 1 = 4 projections
        assert len(projections) == 4
        coords = [p.chunk_coords for p in projections]
        assert (0, 0) in coords
        assert (0, 1) in coords
        assert (1, 0) in coords
        assert (1, 1) in coords


class TestShardingValidationRectilinear:
    """ShardingCodec.validate must check divisibility for rectilinear grids too."""

    def test_sharding_rejects_non_divisible_rectilinear(self) -> None:
        """Rectilinear shard sizes not divisible by inner chunk_shape should raise."""
        from zarr.codecs.sharding import ShardingCodec
        from zarr.core.dtype import Float32

        codec = ShardingCodec(chunk_shape=(5, 5))
        # 17 is not divisible by 5
        g = ChunkGrid.from_rectilinear([[10, 20, 17], [50, 50]], array_shape=(47, 100))

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
        g = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]], array_shape=(60, 100))

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

    # --- Block selection ---

    def test_block_selection_1d(self, tmp_path: Path) -> None:
        z, a = self._make_1d(tmp_path)
        # chunks: [5, 10, 15] -> offsets 0, 5, 15
        # block 0: a[0:5], block 1: a[5:15], block 2: a[15:30]
        np.testing.assert_array_equal(z.blocks[0], a[0:5])
        np.testing.assert_array_equal(z.blocks[1], a[5:15])
        np.testing.assert_array_equal(z.blocks[2], a[15:30])
        np.testing.assert_array_equal(z.blocks[-1], a[15:30])
        # slice of blocks
        np.testing.assert_array_equal(z.blocks[0:2], a[0:15])
        np.testing.assert_array_equal(z.blocks[1:3], a[5:30])
        np.testing.assert_array_equal(z.blocks[:], a[:])

    def test_block_selection_2d(self, tmp_path: Path) -> None:
        z, a = self._make_2d(tmp_path)
        # dim0 chunks: [10, 20, 30] -> offsets 0, 10, 30
        # dim1 chunks: [25, 25, 25, 25] -> offsets 0, 25, 50, 75
        np.testing.assert_array_equal(z.blocks[0, 0], a[0:10, 0:25])
        np.testing.assert_array_equal(z.blocks[1, 2], a[10:30, 50:75])
        np.testing.assert_array_equal(z.blocks[2, 3], a[30:60, 75:100])
        np.testing.assert_array_equal(z.blocks[-1, -1], a[30:60, 75:100])
        # slice of blocks
        np.testing.assert_array_equal(z.blocks[0:2, 1:3], a[0:30, 25:75])
        np.testing.assert_array_equal(z.blocks[:, :], a[:, :])

    def test_set_block_selection_1d(self, tmp_path: Path) -> None:
        z, a = self._make_1d(tmp_path)
        # overwrite block 1 (a[5:15])
        val = np.full(10, -1, dtype="int32")
        z.blocks[1] = val
        a[5:15] = val
        np.testing.assert_array_equal(z[:], a)

    def test_set_block_selection_2d(self, tmp_path: Path) -> None:
        z, a = self._make_2d(tmp_path)
        # overwrite blocks [0:2, 1:3] -> a[0:30, 25:75]
        val = np.full((30, 50), -99, dtype="int32")
        z.blocks[0:2, 1:3] = val
        a[0:30, 25:75] = val
        np.testing.assert_array_equal(z[:], a)

    # --- Set coordinate selection ---

    def test_set_coordinate_selection_1d(self, tmp_path: Path) -> None:
        z, a = self._make_1d(tmp_path)
        ix = np.array([0, 4, 5, 14, 15, 29])
        val = np.full(len(ix), -7, dtype="int32")
        z.vindex[ix] = val
        a[ix] = val
        np.testing.assert_array_equal(z[:], a)

    def test_set_coordinate_selection_2d(self, tmp_path: Path) -> None:
        z, a = self._make_2d(tmp_path)
        r = np.array([0, 9, 10, 29, 30, 59])
        c = np.array([0, 24, 25, 49, 50, 99])
        val = np.full(len(r), -42, dtype="int32")
        z.vindex[r, c] = val
        a[r, c] = val
        np.testing.assert_array_equal(z[:], a)

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

    def test_sharding_rejects_rectilinear_chunks_with_shards(self, tmp_path: Path) -> None:
        """Rectilinear chunks (inner) with sharding is not supported."""
        with pytest.raises(ValueError, match="Rectilinear chunks with sharding"):
            zarr.create_array(
                store=tmp_path / "shard.zarr",
                shape=(60, 100),
                chunks=[[10, 20, 30], [25, 25, 25, 25]],
                shards=(30, 50),
                dtype="int32",
            )

    def test_rectilinear_shards_roundtrip(self, tmp_path: Path) -> None:
        """Rectilinear shards with uniform inner chunks: full write/read roundtrip."""
        import numpy as np

        data = np.arange(120 * 100, dtype="int32").reshape(120, 100)
        arr = zarr.create_array(
            store=tmp_path / "rect_shards.zarr",
            shape=(120, 100),
            chunks=(10, 10),  # uniform inner chunks
            shards=[[60, 40, 20], [50, 50]],  # rectilinear shard boundaries
            dtype="int32",
        )
        arr[:] = data
        result = arr[:]
        np.testing.assert_array_equal(result, data)

    def test_rectilinear_shards_partial_read(self, tmp_path: Path) -> None:
        """Partial reads across rectilinear shard boundaries."""
        import numpy as np

        data = np.arange(120 * 100, dtype="float64").reshape(120, 100)
        arr = zarr.create_array(
            store=tmp_path / "rect_shards.zarr",
            shape=(120, 100),
            chunks=(10, 10),
            shards=[[60, 40, 20], [50, 50]],
            dtype="float64",
        )
        arr[:] = data
        # Read a slice crossing shard boundaries
        result = arr[50:70, 40:60]
        np.testing.assert_array_equal(result, data[50:70, 40:60])

    def test_rectilinear_shards_validates_divisibility(self, tmp_path: Path) -> None:
        """Inner chunk_shape must divide every shard's dimensions."""
        with pytest.raises(ValueError, match="divisible"):
            zarr.create_array(
                store=tmp_path / "bad.zarr",
                shape=(120, 100),
                chunks=(10, 10),
                shards=[[60, 45, 15], [50, 50]],  # 45 not divisible by 10
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
def test_property_block_indexing_rectilinear(data: st.DataObject) -> None:
    """Property test: block indexing on rectilinear arrays matches numpy."""
    z, a = data.draw(rectilinear_arrays_st())
    grid = z.metadata.chunk_grid

    # Pick a random block per dimension and verify it matches the expected slice
    for dim in range(a.ndim):
        dim_grid = grid.dimensions[dim]
        block_ix = data.draw(st.integers(min_value=0, max_value=dim_grid.nchunks - 1))
        sel = [slice(None)] * a.ndim
        start = dim_grid.chunk_offset(block_ix)
        stop = start + dim_grid.data_size(block_ix)
        sel[dim] = slice(start, stop)
        block_sel: list[slice | int] = [slice(None)] * a.ndim
        block_sel[dim] = block_ix
        np.testing.assert_array_equal(
            z.blocks[tuple(block_sel)],
            a[tuple(sel)],
            err_msg=f"dim={dim}, block={block_ix}",
        )


# ---------------------------------------------------------------------------
# V2 regression tests
# ---------------------------------------------------------------------------


class TestV2Regression:
    """Verify V2 arrays still work correctly after the ChunkGrid refactor.

    V2 only supports regular chunks. These tests ensure the V2 metadata
    round-trip (create → write → read) and chunk_grid property work as
    expected with the unified ChunkGrid infrastructure.
    """

    def test_v2_create_and_readback(self, tmp_path: Path) -> None:
        """Basic V2 array: create, write, read back."""
        data = np.arange(60, dtype="float64").reshape(6, 10)
        a = zarr.create_array(
            store=tmp_path / "v2.zarr",
            shape=data.shape,
            chunks=(3, 5),
            dtype=data.dtype,
            zarr_format=2,
        )
        a[:] = data
        np.testing.assert_array_equal(a[:], data)

    def test_v2_chunk_grid_is_regular(self, tmp_path: Path) -> None:
        """V2 metadata.chunk_grid produces a regular ChunkGrid with FixedDimensions."""
        a = zarr.create_array(
            store=tmp_path / "v2.zarr",
            shape=(20, 30),
            chunks=(10, 15),
            dtype="int32",
            zarr_format=2,
        )
        grid = a.metadata.chunk_grid
        assert grid.is_regular
        assert grid.chunk_shape == (10, 15)
        assert grid.shape == (2, 2)
        assert all(isinstance(d, FixedDimension) for d in grid.dimensions)

    def test_v2_boundary_chunks(self, tmp_path: Path) -> None:
        """V2 boundary chunks: codec buffer size stays full, data is clipped."""
        a = zarr.create_array(
            store=tmp_path / "v2.zarr",
            shape=(25,),
            chunks=(10,),
            dtype="int32",
            zarr_format=2,
        )
        grid = a.metadata.chunk_grid
        assert grid.dimensions[0].nchunks == 3
        assert grid.dimensions[0].chunk_size(2) == 10  # full codec buffer
        assert grid.dimensions[0].data_size(2) == 5  # clipped to extent

    def test_v2_slicing_with_boundary(self, tmp_path: Path) -> None:
        """V2 array slicing across boundary chunks returns correct data."""
        data = np.arange(25, dtype="int32")
        a = zarr.create_array(
            store=tmp_path / "v2.zarr",
            shape=(25,),
            chunks=(10,),
            dtype="int32",
            zarr_format=2,
        )
        a[:] = data
        np.testing.assert_array_equal(a[18:25], data[18:25])
        np.testing.assert_array_equal(a[:], data)

    def test_v2_metadata_roundtrip(self, tmp_path: Path) -> None:
        """V2 metadata survives store close and reopen."""
        store_path = tmp_path / "v2.zarr"
        data = np.arange(12, dtype="float32").reshape(3, 4)
        a = zarr.create_array(
            store=store_path,
            shape=data.shape,
            chunks=(2, 2),
            dtype=data.dtype,
            zarr_format=2,
        )
        a[:] = data

        # Reopen from store
        b = zarr.open_array(store=store_path, mode="r")
        assert b.metadata.zarr_format == 2
        assert b.chunks == (2, 2)
        assert b.metadata.chunk_grid.chunk_shape == (2, 2)
        np.testing.assert_array_equal(b[:], data)

    def test_v2_chunk_spec_via_grid(self, tmp_path: Path) -> None:
        """ChunkSpec from V2 grid has correct slices and codec_shape."""
        a = zarr.create_array(
            store=tmp_path / "v2.zarr",
            shape=(15, 20),
            chunks=(10, 10),
            dtype="int32",
            zarr_format=2,
        )
        grid = a.metadata.chunk_grid
        # Interior chunk
        spec = grid[(0, 0)]
        assert spec is not None
        assert spec.shape == (10, 10)
        assert spec.codec_shape == (10, 10)
        # Boundary chunk
        spec = grid[(1, 1)]
        assert spec is not None
        assert spec.shape == (5, 10)  # clipped data
        assert spec.codec_shape == (10, 10)  # full buffer


# ---------------------------------------------------------------------------
# .chunk_sizes property
# ---------------------------------------------------------------------------


class TestChunkSizes:
    """Tests for ChunkGrid.chunk_sizes and Array.chunk_sizes."""

    def test_regular_grid(self) -> None:
        grid = ChunkGrid.from_regular((100, 80), (30, 40))
        assert grid.chunk_sizes == ((30, 30, 30, 10), (40, 40))

    def test_regular_grid_exact(self) -> None:
        grid = ChunkGrid.from_regular((90, 80), (30, 40))
        assert grid.chunk_sizes == ((30, 30, 30), (40, 40))

    def test_rectilinear_grid(self) -> None:
        grid = ChunkGrid.from_rectilinear([[10, 20, 30], [50, 50]], array_shape=(60, 100))
        assert grid.chunk_sizes == ((10, 20, 30), (50, 50))

    def test_single_chunk(self) -> None:
        grid = ChunkGrid.from_regular((10,), (10,))
        assert grid.chunk_sizes == ((10,),)

    def test_array_property_regular(self) -> None:
        store = zarr.storage.MemoryStore()
        arr = zarr.create_array(
            store=store, shape=(100, 80), chunks=(30, 40), dtype="i4", zarr_format=3
        )
        assert arr.chunk_sizes == ((30, 30, 30, 10), (40, 40))

    def test_array_property_rectilinear(self) -> None:
        store = zarr.storage.MemoryStore()
        arr = zarr.create_array(
            store=store, shape=(60, 100), chunks=[[10, 20, 30], [50, 50]], dtype="i4", zarr_format=3
        )
        assert arr.chunk_sizes == ((10, 20, 30), (50, 50))


# ---------------------------------------------------------------------------
# .info display for rectilinear grids
# ---------------------------------------------------------------------------


def test_info_display_rectilinear() -> None:
    """Array.info should not crash for rectilinear grids."""
    store = zarr.storage.MemoryStore()
    arr = zarr.create_array(
        store=store,
        shape=(30,),
        chunks=[[10, 20]],
        dtype="i4",
        zarr_format=3,
    )
    info = arr.info
    text = repr(info)
    assert "<variable>" in text
    assert "Array" in text


# ---------------------------------------------------------------------------
# Resize / append for rectilinear grids
# ---------------------------------------------------------------------------


class TestUpdateShape:
    """Unit tests for ChunkGrid.update_shape()."""

    def test_no_change(self) -> None:
        grid = ChunkGrid.from_rectilinear([[10, 20, 30], [25, 25]], array_shape=(60, 50))
        new_grid = grid.update_shape((60, 50))
        assert _edges(new_grid, 0) == (10, 20, 30)
        assert _edges(new_grid, 1) == (25, 25)

    def test_grow_single_dim(self) -> None:
        grid = ChunkGrid.from_rectilinear([[10, 20, 30], [25, 25]], array_shape=(60, 50))
        new_grid = grid.update_shape((80, 50))
        assert _edges(new_grid, 0) == (10, 20, 30, 20)
        assert _edges(new_grid, 1) == (25, 25)

    def test_grow_multiple_dims(self) -> None:
        grid = ChunkGrid.from_rectilinear([[10, 20], [20, 30]], array_shape=(30, 50))
        # from (30, 50) to (45, 65)
        new_grid = grid.update_shape((45, 65))
        assert _edges(new_grid, 0) == (10, 20, 15)
        assert _edges(new_grid, 1) == (20, 30, 15)

    def test_shrink_single_dim(self) -> None:
        grid = ChunkGrid.from_rectilinear([[10, 20, 30, 40], [25, 25]], array_shape=(100, 50))
        new_grid = grid.update_shape((35, 50))
        # 10+20=30 < 35, 10+20+30=60 >= 35 → keep (10, 20, 30)
        assert _edges(new_grid, 0) == (10, 20, 30)
        assert _edges(new_grid, 1) == (25, 25)

    def test_shrink_to_single_chunk(self) -> None:
        grid = ChunkGrid.from_rectilinear([[10, 20, 30], [25, 25]], array_shape=(60, 50))
        new_grid = grid.update_shape((5, 50))
        assert _edges(new_grid, 0) == (10,)
        assert _edges(new_grid, 1) == (25, 25)

    def test_shrink_multiple_dims(self) -> None:
        grid = ChunkGrid.from_rectilinear([[10, 10, 15, 5], [20, 25, 15]], array_shape=(40, 60))
        # from (40, 60) to (25, 35)
        new_grid = grid.update_shape((25, 35))
        # dim 0: 10+10=20 < 25, 10+10+15=35 >= 25 → keep (10, 10, 15)
        assert _edges(new_grid, 0) == (10, 10, 15)
        # dim 1: 20 < 35, 20+25=45 >= 35 → keep (20, 25)
        assert _edges(new_grid, 1) == (20, 25)

    def test_dimension_mismatch_error(self) -> None:
        grid = ChunkGrid.from_rectilinear([[10, 20], [30, 40]], array_shape=(30, 70))
        with pytest.raises(ValueError, match="dimensions"):
            grid.update_shape((30, 70, 100))

    def test_boundary_cases(self) -> None:
        grid = ChunkGrid.from_rectilinear([[10, 20, 30], [15, 25]], array_shape=(60, 40))
        # Grow to exact chunk boundary on dim 0, add 25 to dim 1
        new_grid = grid.update_shape((60, 65))
        assert _edges(new_grid, 0) == (10, 20, 30)  # no change (60 == sum)
        assert _edges(new_grid, 1) == (15, 25, 25)  # added chunk of 25

        # Shrink to exact chunk boundary
        grid2 = ChunkGrid.from_rectilinear([[10, 20, 30], [15, 25, 10]], array_shape=(60, 50))
        new_grid2 = grid2.update_shape((30, 40))
        # dim 0: 10+20=30 >= 30 → keep (10, 20)
        assert _edges(new_grid2, 0) == (10, 20)
        # dim 1: 15+25=40 >= 40 → keep (15, 25)
        assert _edges(new_grid2, 1) == (15, 25)

    def test_regular_preserves_extents(self, tmp_path: Path) -> None:
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
        assert z.metadata.chunk_grid.dimensions[0].extent == 50


class TestResizeRectilinear:
    """End-to-end resize tests on rectilinear arrays."""

    async def test_async_resize_grow(self) -> None:
        store = zarr.storage.MemoryStore()
        arr = await zarr.api.asynchronous.create_array(
            store=store,
            shape=(30, 40),
            chunks=[[10, 20], [20, 20]],
            dtype="i4",
            zarr_format=3,
        )
        data = np.arange(30 * 40, dtype="i4").reshape(30, 40)
        await arr.setitem(slice(None), data)

        await arr.resize((50, 60))
        assert arr.shape == (50, 60)
        assert _edges(arr.metadata.chunk_grid, 0) == (10, 20, 20)
        assert _edges(arr.metadata.chunk_grid, 1) == (20, 20, 20)
        result = await arr.getitem((slice(0, 30), slice(0, 40)))
        np.testing.assert_array_equal(result, data)

    async def test_async_resize_shrink(self) -> None:
        store = zarr.storage.MemoryStore()
        arr = await zarr.api.asynchronous.create_array(
            store=store,
            shape=(60, 50),
            chunks=[[10, 20, 30], [25, 25]],
            dtype="f4",
            zarr_format=3,
        )
        data = np.arange(60 * 50, dtype="f4").reshape(60, 50)
        await arr.setitem(slice(None), data)

        await arr.resize((25, 30))
        assert arr.shape == (25, 30)
        result = await arr.getitem(slice(None))
        np.testing.assert_array_equal(result, data[:25, :30])

    def test_sync_resize_grow(self) -> None:
        store = zarr.storage.MemoryStore()
        arr = zarr.create_array(
            store=store,
            shape=(20, 30),
            chunks=[[8, 12], [10, 20]],
            dtype="u1",
            zarr_format=3,
        )
        data = np.arange(20 * 30, dtype="u1").reshape(20, 30)
        arr[:] = data
        arr.resize((35, 45))
        assert arr.shape == (35, 45)
        np.testing.assert_array_equal(arr[:20, :30], data)

    def test_sync_resize_shrink(self) -> None:
        store = zarr.storage.MemoryStore()
        arr = zarr.create_array(
            store=store,
            shape=(40, 50),
            chunks=[[10, 15, 15], [20, 30]],
            dtype="i2",
            zarr_format=3,
        )
        data = np.arange(40 * 50, dtype="i2").reshape(40, 50)
        arr[:] = data
        arr.resize((15, 30))
        assert arr.shape == (15, 30)
        np.testing.assert_array_equal(arr[:], data[:15, :30])


class TestAppendRectilinear:
    """End-to-end append tests on rectilinear arrays."""

    async def test_append_first_axis(self) -> None:
        store = zarr.storage.MemoryStore()
        arr = await zarr.api.asynchronous.create_array(
            store=store,
            shape=(30, 20),
            chunks=[[10, 20], [10, 10]],
            dtype="i4",
            zarr_format=3,
        )
        initial = np.arange(30 * 20, dtype="i4").reshape(30, 20)
        await arr.setitem(slice(None), initial)

        append_data = np.arange(30 * 20, 45 * 20, dtype="i4").reshape(15, 20)
        await arr.append(append_data, axis=0)
        assert arr.shape == (45, 20)

        result = await arr.getitem(slice(None))
        np.testing.assert_array_equal(result, np.vstack([initial, append_data]))

    async def test_append_second_axis(self) -> None:
        store = zarr.storage.MemoryStore()
        arr = await zarr.api.asynchronous.create_array(
            store=store,
            shape=(20, 30),
            chunks=[[10, 10], [10, 20]],
            dtype="f4",
            zarr_format=3,
        )
        initial = np.arange(20 * 30, dtype="f4").reshape(20, 30)
        await arr.setitem(slice(None), initial)

        append_data = np.arange(20 * 30, 20 * 45, dtype="f4").reshape(20, 15)
        await arr.append(append_data, axis=1)
        assert arr.shape == (20, 45)

        result = await arr.getitem(slice(None))
        np.testing.assert_array_equal(result, np.hstack([initial, append_data]))

    def test_sync_append(self) -> None:
        store = zarr.storage.MemoryStore()
        arr = zarr.create_array(
            store=store,
            shape=(20, 20),
            chunks=[[8, 12], [7, 13]],
            dtype="u2",
            zarr_format=3,
        )
        initial = np.arange(20 * 20, dtype="u2").reshape(20, 20)
        arr[:] = initial

        append_data = np.arange(20 * 20, 25 * 20, dtype="u2").reshape(5, 20)
        arr.append(append_data, axis=0)
        assert arr.shape == (25, 20)
        np.testing.assert_array_equal(arr[:20, :], initial)
        np.testing.assert_array_equal(arr[20:, :], append_data)

    async def test_multiple_appends(self) -> None:
        store = zarr.storage.MemoryStore()
        arr = await zarr.api.asynchronous.create_array(
            store=store,
            shape=(10, 10),
            chunks=[[3, 7], [4, 6]],
            dtype="i4",
            zarr_format=3,
        )
        initial = np.arange(10 * 10, dtype="i4").reshape(10, 10)
        await arr.setitem(slice(None), initial)

        all_data = [initial]
        for i in range(3):
            chunk = np.full((5, 10), i + 100, dtype="i4")
            await arr.append(chunk, axis=0)
            all_data.append(chunk)

        assert arr.shape == (25, 10)
        result = await arr.getitem(slice(None))
        np.testing.assert_array_equal(result, np.vstack(all_data))

    async def test_append_with_partial_edge_chunks(self) -> None:
        store = zarr.storage.MemoryStore()
        arr = await zarr.api.asynchronous.create_array(
            store=store,
            shape=(25, 30),
            chunks=[[10, 15], [12, 18]],
            dtype="f8",
            zarr_format=3,
        )
        initial = np.random.default_rng(42).random((25, 30))
        await arr.setitem(slice(None), initial)

        append_data = np.random.default_rng(43).random((10, 30))
        await arr.append(append_data, axis=0)
        assert arr.shape == (35, 30)

        result = np.asarray(await arr.getitem(slice(None)))
        np.testing.assert_array_almost_equal(result, np.vstack([initial, append_data]))

    async def test_append_small_data(self) -> None:
        store = zarr.storage.MemoryStore()
        arr = await zarr.api.asynchronous.create_array(
            store=store,
            shape=(20, 20),
            chunks=[[8, 12], [7, 13]],
            dtype="i4",
            zarr_format=3,
        )
        data = np.arange(20 * 20, dtype="i4").reshape(20, 20)
        await arr.setitem(slice(None), data)

        small = np.full((3, 20), 999, dtype="i4")
        await arr.append(small, axis=0)
        assert arr.shape == (23, 20)
        result = await arr.getitem((slice(20, 23), slice(None)))
        np.testing.assert_array_equal(result, small)

    def test_parse_chunk_grid_regular_from_dict(self) -> None:
        """parse_chunk_grid constructs a regular grid from a metadata dict."""
        d: dict[str, Any] = {"name": "regular", "configuration": {"chunk_shape": [10, 20]}}
        g = parse_chunk_grid(d, (100, 200))
        assert g.is_regular
        assert g.chunk_shape == (10, 20)
        assert g.shape == (10, 10)
        assert g.get_nchunks() == 100


# ---------------------------------------------------------------------------
# Boundary chunk tests
# ---------------------------------------------------------------------------


class TestVaryingDimensionBoundary:
    """VaryingDimension with extent < sum(edges), mirroring how FixedDimension
    handles boundary chunks."""

    def test_extent_parameter(self) -> None:
        d = VaryingDimension([10, 20, 30], extent=50)
        assert d.extent == 50
        assert d.chunk_size(2) == 30  # codec buffer: full edge
        assert d.data_size(2) == 20  # valid data: clipped to extent

    def test_extent_equals_sum_no_clipping(self) -> None:
        d = VaryingDimension([10, 20, 30], extent=60)
        assert d.extent == 60
        assert d.data_size(2) == 30  # no clipping when extent == sum(edges)

    def test_data_size_interior_chunks_unaffected(self) -> None:
        d = VaryingDimension([10, 20, 30], extent=50)
        assert d.data_size(0) == 10  # fully within extent
        assert d.data_size(1) == 20  # fully within extent (offset 10, ends at 30)

    def test_data_size_at_exact_boundary(self) -> None:
        d = VaryingDimension([10, 20, 30], extent=60)
        # extent == sum(edges), so no clipping
        assert d.data_size(2) == 30

    def test_data_size_single_element_boundary(self) -> None:
        d = VaryingDimension([10, 20, 30], extent=31)
        assert d.data_size(0) == 10
        assert d.data_size(1) == 20
        assert d.data_size(2) == 1  # only 1 element in last chunk

    def test_extent_exceeds_sum_rejected(self) -> None:
        with pytest.raises(ValueError, match="exceeds sum of edges"):
            VaryingDimension([10, 20], extent=50)

    def test_negative_extent_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be >= 0"):
            VaryingDimension([10, 20], extent=-1)

    def test_chunk_spec_boundary_varying(self) -> None:
        """ChunkGrid with a boundary VaryingDimension produces correct ChunkSpec."""
        g = ChunkGrid(dimensions=(VaryingDimension([10, 20, 30], extent=50),))
        spec = g[(2,)]
        assert spec is not None
        assert spec.codec_shape == (30,)  # full edge
        assert spec.shape == (20,)  # clipped to extent
        assert spec.is_boundary is True

    def test_chunk_spec_interior_varying(self) -> None:
        g = ChunkGrid(dimensions=(VaryingDimension([10, 20, 30], extent=50),))
        spec = g[(0,)]
        assert spec is not None
        assert spec.codec_shape == (10,)
        assert spec.shape == (10,)
        assert spec.is_boundary is False


class TestMultipleOverflowChunks:
    """Rectilinear grids where multiple chunks extend past the array extent."""

    def test_multiple_chunks_past_extent(self) -> None:
        """Chunks 2 is partial, chunk 3 is entirely past the extent."""
        g = ChunkGrid.from_rectilinear([[10, 20, 30, 40]], array_shape=(50,))
        d = g.dimensions[0]
        assert d.nchunks == 4
        assert d.data_size(0) == 10  # fully within
        assert d.data_size(1) == 20  # fully within
        assert d.data_size(2) == 20  # partial: 50 - 30 = 20
        assert d.data_size(3) == 0  # entirely past
        assert d.chunk_size(2) == 30  # codec buffer: full edge
        assert d.chunk_size(3) == 40  # codec buffer: full edge

    def test_chunk_spec_entirely_past_extent(self) -> None:
        """ChunkSpec for a chunk entirely past the extent has zero-size shape."""
        g = ChunkGrid.from_rectilinear([[10, 20, 30, 40]], array_shape=(50,))
        spec = g[(3,)]
        assert spec is not None
        assert spec.shape == (0,)
        assert spec.codec_shape == (40,)
        assert spec.is_boundary is True

    def test_chunk_spec_partial_overflow(self) -> None:
        """ChunkSpec for a partially-overflowing chunk clips correctly."""
        g = ChunkGrid.from_rectilinear([[10, 20, 30, 40]], array_shape=(50,))
        spec = g[(2,)]
        assert spec is not None
        assert spec.shape == (20,)
        assert spec.codec_shape == (30,)
        assert spec.is_boundary is True
        assert spec.slices == (slice(30, 50, 1),)

    def test_chunk_sizes_with_overflow(self) -> None:
        """chunk_sizes returns clipped data sizes including zero for past-extent chunks."""
        g = ChunkGrid.from_rectilinear([[10, 20, 30, 40]], array_shape=(50,))
        assert g.chunk_sizes == ((10, 20, 20, 0),)

    def test_multidim_overflow(self) -> None:
        """Overflow in multiple dimensions simultaneously."""
        g = ChunkGrid.from_rectilinear([[10, 20, 30], [40, 40, 40]], array_shape=(45, 100))
        # dim 0: edges sum to 60, extent 45 → chunk 2 partial (45-30=15)
        # dim 1: edges sum to 120, extent 100 → chunk 2 partial (100-80=20)
        assert g.chunk_sizes == ((10, 20, 15), (40, 40, 20))
        spec = g[(2, 2)]
        assert spec is not None
        assert spec.shape == (15, 20)
        assert spec.codec_shape == (30, 40)

    def test_uniform_edges_with_overflow_collapses_to_fixed(self) -> None:
        """Uniform edges where len == ceildiv(extent, edge) collapse to FixedDimension."""
        g = ChunkGrid.from_rectilinear([[10, 10, 10, 10]], array_shape=(35,))
        assert isinstance(g.dimensions[0], FixedDimension)
        assert g.is_regular
        assert g.chunk_sizes == ((10, 10, 10, 5),)
        assert g.dimensions[0].nchunks == 4

    def test_serialization_roundtrip_overflow(self) -> None:
        """Overflow chunks survive serialization round-trip."""
        g = ChunkGrid.from_rectilinear([[10, 20, 30, 40]], array_shape=(50,))
        serialized = serialize_chunk_grid(g, "rectilinear")
        assert serialized == {
            "name": "rectilinear",
            "configuration": {"kind": "inline", "chunk_shapes": [[10, 20, 30, 40]]},
        }
        g2 = parse_chunk_grid(serialized, (50,))
        assert g2.dimensions[0].nchunks == 4
        assert g2.chunk_sizes == ((10, 20, 20, 0),)

    def test_index_to_chunk_near_extent(self) -> None:
        """Index lookup near and at the extent boundary."""
        d = VaryingDimension([10, 20, 30, 40], extent=50)
        assert d.index_to_chunk(29) == 1  # last index in chunk 1
        assert d.index_to_chunk(30) == 2  # first index in chunk 2
        assert d.index_to_chunk(49) == 2  # last valid index


class TestBoundaryIndexing:
    """Indexing operations on boundary chunks for both FixedDimension and
    VaryingDimension, ensuring the isinstance cleanup works correctly."""

    def test_bool_indexer_fixed_boundary(self) -> None:
        """BoolArrayDimIndexer pads to codec size for FixedDimension boundary."""
        from zarr.core.indexing import BoolArrayDimIndexer

        # array extent 7, chunk size 5 → 2 chunks, last has data_size=2
        dim = FixedDimension(size=5, extent=7)
        mask = np.array([False, False, False, False, False, True, True])
        indexer = BoolArrayDimIndexer(mask, 7, dim)
        projections = list(indexer)
        assert len(projections) == 1
        p = projections[0]
        assert p.dim_chunk_ix == 1
        # boolean selection should be padded to chunk_size (5)
        sel = p.dim_chunk_sel
        assert isinstance(sel, np.ndarray)
        assert sel.shape[0] == 5
        assert sel[0] is np.True_
        assert sel[1] is np.True_
        assert sel[2] is np.False_  # padding

    def test_bool_indexer_varying_boundary(self) -> None:
        """BoolArrayDimIndexer pads to codec size for VaryingDimension boundary."""
        from zarr.core.indexing import BoolArrayDimIndexer

        # edges [5, 10], extent=7 -> last chunk has data_size=2, chunk_size=10
        dim = VaryingDimension([5, 10], extent=7)
        mask = np.array([False, False, False, False, False, True, True])
        indexer = BoolArrayDimIndexer(mask, 7, dim)
        projections = list(indexer)
        assert len(projections) == 1
        p = projections[0]
        assert p.dim_chunk_ix == 1
        # boolean selection should be padded to chunk_size (10)
        sel = p.dim_chunk_sel
        assert isinstance(sel, np.ndarray)
        assert sel.shape[0] == 10
        assert sel[0] is np.True_
        assert sel[1] is np.True_
        assert sel[2] is np.False_  # padding

    def test_bool_indexer_no_padding_interior(self) -> None:
        """No padding needed for interior chunks."""
        from zarr.core.indexing import BoolArrayDimIndexer

        dim = FixedDimension(size=5, extent=10)
        mask = np.array([True, False, False, False, False, False, False, False, False, False])
        indexer = BoolArrayDimIndexer(mask, 10, dim)
        projections = list(indexer)
        assert len(projections) == 1
        p = projections[0]
        assert p.dim_chunk_ix == 0
        sel = p.dim_chunk_sel
        assert isinstance(sel, np.ndarray)
        assert sel.shape[0] == 5  # equals chunk_size, no padding needed

    def test_slice_indexer_varying_boundary(self) -> None:
        """SliceDimIndexer clips to data_size at boundary for VaryingDimension."""
        from zarr.core.indexing import SliceDimIndexer

        dim = VaryingDimension([5, 10], extent=7)
        # select all elements
        indexer = SliceDimIndexer(slice(None), 7, dim)
        projections = list(indexer)
        assert len(projections) == 2
        # chunk 0: full chunk
        assert projections[0].dim_chunk_sel == slice(0, 5, 1)
        # chunk 1: clipped to data_size (2), not chunk_size (10)
        assert projections[1].dim_chunk_sel == slice(0, 2, 1)

    def test_int_array_indexer_varying_boundary(self) -> None:
        """IntArrayDimIndexer handles indices near boundary correctly."""
        from zarr.core.indexing import IntArrayDimIndexer

        dim = VaryingDimension([5, 10], extent=7)
        indices = np.array([6])  # in chunk 1, offset 5, so chunk-local = 1
        indexer = IntArrayDimIndexer(indices, 7, dim)
        projections = list(indexer)
        assert len(projections) == 1
        assert projections[0].dim_chunk_ix == 1
        sel = projections[0].dim_chunk_sel
        assert isinstance(sel, np.ndarray)
        np.testing.assert_array_equal(sel, [1])

    def test_orthogonal_indexer_varying_boundary_advanced(self) -> None:
        """OrthogonalIndexer with advanced indexing uses per-chunk chunk_size
        for ix_() conversion, not a precomputed max."""
        from zarr.core.indexing import OrthogonalIndexer

        # 2D: dim 0 has boundary chunk, dim 1 is regular
        g = ChunkGrid(
            dimensions=(
                VaryingDimension([5, 10], extent=7),
                FixedDimension(size=4, extent=8),
            )
        )
        indexer = OrthogonalIndexer(
            selection=(np.array([0, 6]), slice(None)),
            shape=(7, 8),
            chunk_grid=g,
        )
        projections = list(indexer)
        # index 0 → chunk 0, index 6 → chunk 1; dim 1 has 2 chunks
        assert len(projections) == 4
        coords = {p.chunk_coords for p in projections}
        assert coords == {(0, 0), (0, 1), (1, 0), (1, 1)}


class TestUpdateShapeBoundary:
    """Resize creates boundary VaryingDimensions with correct extent."""

    def test_shrink_creates_boundary(self) -> None:
        grid = ChunkGrid.from_rectilinear([[10, 20, 30]], array_shape=(60,))
        new_grid = grid.update_shape((45,))
        dim = new_grid.dimensions[0]
        assert isinstance(dim, VaryingDimension)
        assert dim.edges == (10, 20, 30)  # last chunk kept (cumulative 60 >= 45)
        assert dim.extent == 45
        assert dim.chunk_size(2) == 30  # codec buffer
        assert dim.data_size(2) == 15  # clipped: 45 - 30 = 15

    def test_shrink_to_exact_boundary(self) -> None:
        grid = ChunkGrid.from_rectilinear([[10, 20, 30]], array_shape=(60,))
        new_grid = grid.update_shape((30,))
        dim = new_grid.dimensions[0]
        assert isinstance(dim, VaryingDimension)
        assert dim.edges == (10, 20)  # chunk 2 dropped entirely
        assert dim.extent == 30
        assert dim.data_size(1) == 20  # no clipping needed

    def test_shrink_chunk_spec(self) -> None:
        """After shrink, ChunkSpec reflects boundary correctly."""
        grid = ChunkGrid.from_rectilinear([[10, 20, 30]], array_shape=(60,))
        new_grid = grid.update_shape((45,))
        spec = new_grid[(2,)]
        assert spec is not None
        assert spec.codec_shape == (30,)
        assert spec.shape == (15,)
        assert spec.is_boundary is True

    def test_parse_chunk_grid_rebinds_extent(self) -> None:
        """parse_chunk_grid re-binds VaryingDimension extent to array shape."""
        g = ChunkGrid.from_rectilinear([[10, 20, 30]], array_shape=(60,))
        # sum(edges)=60, array_shape=50 → re-bind extent
        g2 = parse_chunk_grid(g, (50,))
        dim = g2.dimensions[0]
        assert isinstance(dim, VaryingDimension)
        assert dim.extent == 50
        assert dim.data_size(2) == 20  # 50 - 30 = 20


class TestNchunksWorksForRectilinear:
    def test_nchunks_returns_correct_count(self) -> None:
        """nchunks should work for rectilinear arrays."""
        store = MemoryStore()
        a = zarr.create_array(store, shape=(30,), chunks=[[10, 20]], dtype="int32")
        assert a.nchunks == 2

    def test_nchunks_2d_rectilinear(self) -> None:
        store = MemoryStore()
        a = zarr.create_array(store, shape=(30, 40), chunks=[[10, 20], [15, 25]], dtype="int32")
        assert a.nchunks == 4  # 2 chunks x 2 chunks


class TestIterChunkRegionsWorksForRectilinear:
    def test_iter_chunk_regions_rectilinear(self) -> None:
        """_iter_chunk_regions should work for rectilinear arrays."""
        from zarr.core.array import _iter_chunk_regions

        store = MemoryStore()
        a = zarr.create_array(store, shape=(30,), chunks=[[10, 20]], dtype="int32")
        regions = list(_iter_chunk_regions(a))
        assert len(regions) == 2
        assert regions[0] == (slice(0, 10, 1),)
        assert regions[1] == (slice(10, 30, 1),)
