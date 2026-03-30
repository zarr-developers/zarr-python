"""
Tests for the unified ChunkGrid design (POC).

Tests the core ChunkGrid with FixedDimension/VaryingDimension internals,
ChunkSpec, serialization round-trips, indexing with rectilinear grids,
and end-to-end array creation + read/write.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import zarr
from zarr.core.chunk_grids import (
    ChunkGrid,
    ChunkSpec,
    FixedDimension,
    VaryingDimension,
    _is_rectilinear_chunks,
)
from zarr.core.common import compress_rle, expand_rle
from zarr.core.metadata.v3 import (
    RectilinearChunkGrid,
    parse_chunk_grid,
)
from zarr.core.metadata.v3 import (
    RegularChunkGrid as RegularChunkGridMeta,
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
# Dimension index_to_chunk bounds tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dim", "index", "match"),
    [
        (VaryingDimension([10, 20, 30], extent=60), 60, "out of bounds"),
        (VaryingDimension([10, 20, 30], extent=60), 100, "out of bounds"),
        (FixedDimension(size=10, extent=95), 95, "out of bounds"),
        (FixedDimension(size=10, extent=95), -1, "Negative"),
    ],
    ids=[
        "varying-at-extent",
        "varying-past-extent",
        "fixed-at-extent",
        "fixed-negative",
    ],
)
def test_dimension_index_to_chunk_bounds(
    dim: FixedDimension | VaryingDimension, index: int, match: str
) -> None:
    """Out-of-bounds or negative indices raise IndexError for both dimension types"""
    with pytest.raises(IndexError, match=match):
        dim.index_to_chunk(index)


@pytest.mark.parametrize(
    ("dim", "index", "expected"),
    [
        (VaryingDimension([10, 20, 30], extent=60), 59, 2),
        (FixedDimension(size=10, extent=95), 94, 9),
    ],
    ids=["varying-last-valid", "fixed-last-valid"],
)
def test_dimension_index_to_chunk_last_valid(
    dim: FixedDimension | VaryingDimension, index: int, expected: int
) -> None:
    """Last valid index maps to the correct chunk for both dimension types"""
    assert dim.index_to_chunk(index) == expected


# ---------------------------------------------------------------------------
# Rectilinear feature flag tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "action",
    [
        lambda: RectilinearChunkGrid(chunk_shapes=((10, 20), (25, 25))),
        lambda: RectilinearChunkGrid.from_dict(
            {
                "name": "rectilinear",
                "configuration": {"kind": "inline", "chunk_shapes": [[10, 20, 30], [50, 50]]},  # type: ignore[typeddict-item]
            }
        ),
        lambda: zarr.create_array(MemoryStore(), shape=(30,), chunks=[[10, 20]], dtype="int32"),
    ],
    ids=["constructor", "from_dict", "create_array"],
)
def test_rectilinear_feature_flag_blocked(action: Any) -> None:
    """Rectilinear chunk operations raise ValueError when the feature flag is disabled"""
    with zarr.config.set({"array.rectilinear_chunks": False}):
        with pytest.raises(ValueError, match="experimental and disabled by default"):
            action()


def test_rectilinear_feature_flag_enabled() -> None:
    """Rectilinear chunk grid construction succeeds when the feature flag is enabled"""
    with zarr.config.set({"array.rectilinear_chunks": True}):
        grid = RectilinearChunkGrid(chunk_shapes=((10, 20), (25, 25)))
        assert grid.ndim == 2


# ---------------------------------------------------------------------------
# FixedDimension tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    (
        "size",
        "extent",
        "chunk_ix",
        "expected_nchunks",
        "expected_chunk_size",
        "expected_data_size",
        "expected_offset",
    ),
    [
        (10, 100, 0, 10, 10, 10, 0),
        (10, 100, 1, 10, 10, 10, 10),
        (10, 100, 9, 10, 10, 10, 90),
        (10, 95, 9, 10, 10, 5, 90),  # boundary chunk
        (0, 0, None, 0, None, None, None),  # zero-size
    ],
    ids=["start", "middle", "end", "boundary", "zero-size"],
)
def test_fixed_dimension(
    size: int,
    extent: int,
    chunk_ix: int | None,
    expected_nchunks: int,
    expected_chunk_size: int | None,
    expected_data_size: int | None,
    expected_offset: int | None,
) -> None:
    """FixedDimension properties match expected values for various chunk/extent combinations"""
    d = FixedDimension(size=size, extent=extent)
    assert d.nchunks == expected_nchunks
    if chunk_ix is not None:
        assert d.chunk_size(chunk_ix) == expected_chunk_size
        assert d.data_size(chunk_ix) == expected_data_size
        assert d.chunk_offset(chunk_ix) == expected_offset


@pytest.mark.parametrize(
    ("idx", "expected"),
    [(0, 0), (9, 0), (10, 1), (25, 2)],
)
def test_fixed_dimension_index_to_chunk(idx: int, expected: int) -> None:
    """FixedDimension.index_to_chunk maps element indices to correct chunk indices"""
    d = FixedDimension(size=10, extent=100)
    assert d.index_to_chunk(idx) == expected


def test_fixed_dimension_indices_to_chunks() -> None:
    """FixedDimension.indices_to_chunks vectorizes index-to-chunk mapping over an array"""
    d = FixedDimension(size=10, extent=100)
    indices = np.array([0, 5, 10, 15, 99])
    np.testing.assert_array_equal(d.indices_to_chunks(indices), [0, 0, 1, 1, 9])


@pytest.mark.parametrize(
    ("size", "extent", "match"),
    [(-1, 100, "must be >= 0"), (10, -1, "must be >= 0")],
    ids=["negative-size", "negative-extent"],
)
def test_fixed_dimension_rejects_negative(size: int, extent: int, match: str) -> None:
    """FixedDimension raises ValueError for negative size or extent"""
    with pytest.raises(ValueError, match=match):
        FixedDimension(size=size, extent=extent)


# ---------------------------------------------------------------------------
# VaryingDimension tests
# ---------------------------------------------------------------------------


def test_varying_dimension_construction() -> None:
    """VaryingDimension stores edges, cumulative sums, nchunks, and extent correctly"""
    d = VaryingDimension([10, 20, 30], extent=60)
    assert d.edges == (10, 20, 30)
    assert d.cumulative == (10, 30, 60)
    assert d.nchunks == 3
    assert d.extent == 60


@pytest.mark.parametrize(
    (
        "chunk_idx",
        "expected_offset",
        "expected_size",
        "expected_data",
        "expected_chunk_for_first_idx",
    ),
    [
        (0, 0, 10, 10, 0),
        (1, 10, 20, 20, 1),
        (2, 30, 30, 30, 2),
    ],
)
def test_varying_dimension(
    chunk_idx: int,
    expected_offset: int,
    expected_size: int,
    expected_data: int,
    expected_chunk_for_first_idx: int,
) -> None:
    """VaryingDimension chunk_offset, chunk_size, data_size, and index_to_chunk return correct values"""
    d = VaryingDimension([10, 20, 30], extent=60)
    assert d.chunk_offset(chunk_idx) == expected_offset
    assert d.chunk_size(chunk_idx) == expected_size
    assert d.data_size(chunk_idx) == expected_data
    assert d.index_to_chunk(expected_offset) == expected_chunk_for_first_idx


def test_varying_dimension_indices_to_chunks() -> None:
    """VaryingDimension.indices_to_chunks vectorizes index-to-chunk mapping over an array"""
    d = VaryingDimension([10, 20, 30], extent=60)
    indices = np.array([0, 9, 10, 29, 30, 59])
    np.testing.assert_array_equal(d.indices_to_chunks(indices), [0, 0, 1, 1, 2, 2])


@pytest.mark.parametrize(
    ("edges", "extent", "match"),
    [
        ([], 0, "must not be empty"),
        ([10, 0, 5], 15, "must be > 0"),
    ],
    ids=["empty", "zero-edge"],
)
def test_varying_dimension_rejects_invalid(edges: list[int], extent: int, match: str) -> None:
    """VaryingDimension raises ValueError for empty edges or zero-length edges"""
    with pytest.raises(ValueError, match=match):
        VaryingDimension(edges, extent=extent)


# ---------------------------------------------------------------------------
# ChunkSpec tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("slices", "codec_shape", "expected_shape", "expected_boundary"),
    [
        ((slice(0, 10), slice(0, 20)), (10, 20), (10, 20), False),
        ((slice(90, 95), slice(0, 20)), (10, 20), (5, 20), True),
        ((slice(10, 10),), (0,), (0,), False),
        ((slice(0, 10), slice(0, 5)), (10, 10), (10, 5), True),
    ],
    ids=["basic", "boundary", "empty-slices", "multidim-boundary"],
)
def test_chunk_spec(
    slices: tuple[slice, ...],
    codec_shape: tuple[int, ...],
    expected_shape: tuple[int, ...],
    expected_boundary: bool,
) -> None:
    """ChunkSpec reports correct shape and boundary status from slices and codec_shape"""
    spec = ChunkSpec(slices=slices, codec_shape=codec_shape)
    assert spec.shape == expected_shape
    assert spec.is_boundary == expected_boundary


# ---------------------------------------------------------------------------
# ChunkGrid construction tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("array_shape", "chunk_sizes", "expected_regular", "expected_ndim", "expected_chunk_shape"),
    [
        ((100, 200), (10, 20), True, 2, (10, 20)),
        ((), (), True, 0, ()),
        ((60, 100), [[10, 20, 30], [25, 25, 25, 25]], False, 2, None),
        ((30, 50), [[10, 10, 10], [25, 25]], True, 2, (10, 25)),  # uniform edges → regular
    ],
    ids=["regular", "zero-dim", "rectilinear", "uniform-becomes-regular"],
)
def test_chunk_grid_construction(
    array_shape: tuple[int, ...],
    chunk_sizes: Any,
    expected_regular: bool,
    expected_ndim: int,
    expected_chunk_shape: tuple[int, ...] | None,
) -> None:
    """ChunkGrid.from_sizes produces grids with correct regularity, ndim, and chunk_shape"""
    g = ChunkGrid.from_sizes(array_shape, chunk_sizes)
    assert g.is_regular == expected_regular
    assert g.ndim == expected_ndim
    if expected_chunk_shape is not None:
        assert g.chunk_shape == expected_chunk_shape
    else:
        with pytest.raises(ValueError, match="only available for regular"):
            _ = g.chunk_shape


def test_chunk_grid_rectilinear_uniform_dim_is_fixed() -> None:
    """A rectilinear grid with all-same sizes in one dim stores it as Fixed."""
    g = ChunkGrid.from_sizes((60, 100), [[10, 20, 30], [25, 25, 25, 25]])
    assert isinstance(g.dimensions[0], VaryingDimension)
    assert isinstance(g.dimensions[1], FixedDimension)


# ---------------------------------------------------------------------------
# ChunkGrid query tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("shape", "chunks", "expected_grid_shape"),
    [
        ((100, 200), (10, 20), (10, 10)),
        ((95, 200), (10, 20), (10, 10)),
        ((60, 100), [[10, 20, 30], [25, 25, 25, 25]], (3, 4)),
    ],
    ids=["regular", "regular-boundary", "rectilinear"],
)
def test_chunk_grid_shape(
    shape: tuple[int, ...],
    chunks: Any,
    expected_grid_shape: tuple[int, ...],
) -> None:
    """ChunkGrid.grid_shape returns the expected number of chunks per dimension"""
    g = ChunkGrid.from_sizes(shape, chunks)
    assert g.grid_shape == expected_grid_shape


@pytest.mark.parametrize(
    (
        "array_shape",
        "chunk_sizes",
        "coords",
        "expected_shape",
        "expected_codec_shape",
        "expected_boundary",
    ),
    [
        # regular interior
        ((100, 200), (10, 20), (0, 0), (10, 20), (10, 20), False),
        # regular boundary
        ((95, 200), (10, 20), (9, 0), (5, 20), (10, 20), True),
        # rectilinear
        ((60, 100), [[10, 20, 30], [25, 25, 25, 25]], (0, 0), (10, 25), (10, 25), False),
        ((60, 100), [[10, 20, 30], [25, 25, 25, 25]], (1, 0), (20, 25), (20, 25), False),
        ((60, 100), [[10, 20, 30], [25, 25, 25, 25]], (2, 3), (30, 25), (30, 25), False),
    ],
    ids=["regular", "regular-boundary", "rectilinear-0,0", "rectilinear-1,0", "rectilinear-2,3"],
)
def test_chunk_grid_getitem(
    array_shape: tuple[int, ...],
    chunk_sizes: Any,
    coords: tuple[int, ...],
    expected_shape: tuple[int, ...],
    expected_codec_shape: tuple[int, ...],
    expected_boundary: bool,
) -> None:
    """ChunkGrid.__getitem__ returns a ChunkSpec with correct shape, codec_shape, and boundary flag"""
    g = ChunkGrid.from_sizes(array_shape, chunk_sizes)
    spec = g[coords]
    assert spec is not None
    assert spec.shape == expected_shape
    assert spec.codec_shape == expected_codec_shape
    assert spec.is_boundary == expected_boundary


@pytest.mark.parametrize(
    ("array_shape", "chunk_sizes", "coords"),
    [
        ((100, 200), (10, 20), (99, 0)),
        ((60, 100), [[10, 20, 30], [25, 25, 25, 25]], (3, 0)),
    ],
    ids=["regular-oob", "rectilinear-oob"],
)
def test_chunk_grid_getitem_oob(
    array_shape: tuple[int, ...], chunk_sizes: Any, coords: tuple[int, ...]
) -> None:
    """Out-of-bounds chunk coordinates return None"""
    g = ChunkGrid.from_sizes(array_shape, chunk_sizes)
    assert g[coords] is None


def test_chunk_grid_getitem_slices() -> None:
    """ChunkSpec.slices reflect the correct start/stop for a rectilinear chunk"""
    g = ChunkGrid.from_sizes((60, 100), [[10, 20, 30], [25, 25, 25, 25]])
    spec = g[(1, 2)]
    assert spec is not None
    assert spec.slices == (slice(10, 30, 1), slice(50, 75, 1))


# -- all_chunk_coords tests --


@pytest.mark.parametrize(
    ("array_shape", "chunk_sizes", "origin", "selection_shape", "expected_coords"),
    [
        # rectilinear grid
        (
            (60, 100),
            [[10, 20, 30], [50, 50]],
            None,
            None,
            [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],
        ),
        ((60, 100), [[10, 20, 30], [50, 50]], (1, 0), None, [(1, 0), (1, 1), (2, 0), (2, 1)]),
        ((60, 100), [[10, 20, 30], [50, 50]], None, (2, 1), [(0, 0), (1, 0)]),
        ((60, 100), [[10, 20, 30], [50, 50]], (1, 1), (2, 1), [(1, 1), (2, 1)]),
        # regular grid
        ((30, 40), (10, 20), (2, 1), None, [(2, 1)]),
        ((30, 40), (10, 20), None, (0, 0), []),
        ((60, 80), (20, 20), (0, 2), (3, 1), [(0, 2), (1, 2), (2, 2)]),
    ],
    ids=[
        "all",
        "with-origin",
        "with-sel-shape",
        "origin+sel",
        "last-chunk",
        "zero-sel",
        "single-dim",
    ],
)
def test_all_chunk_coords(
    array_shape: tuple[int, ...],
    chunk_sizes: Any,
    origin: tuple[int, ...] | None,
    selection_shape: tuple[int, ...] | None,
    expected_coords: list[tuple[int, ...]],
) -> None:
    """all_chunk_coords yields the expected coordinates with optional origin and selection_shape"""
    g = ChunkGrid.from_sizes(array_shape, chunk_sizes)
    kwargs: dict[str, Any] = {}
    if origin is not None:
        kwargs["origin"] = origin
    if selection_shape is not None:
        kwargs["selection_shape"] = selection_shape
    assert list(g.all_chunk_coords(**kwargs)) == expected_coords


def test_chunk_grid_get_nchunks() -> None:
    """get_nchunks returns the total number of chunks across all dimensions"""
    g = ChunkGrid.from_sizes((60, 100), [[10, 20, 30], [50, 50]])
    assert g.get_nchunks() == 6


def test_chunk_grid_iter() -> None:
    """Iterating a ChunkGrid yields the correct number of ChunkSpec objects"""
    g = ChunkGrid.from_sizes((30, 40), (10, 20))
    specs = list(g)
    assert len(specs) == 6
    assert all(isinstance(s, ChunkSpec) for s in specs)


# ---------------------------------------------------------------------------
# RLE tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("compressed", "expected"),
    [
        ([[10, 3]], [10, 10, 10]),
        ([[10, 2], [20, 1]], [10, 10, 20]),
    ],
)
def test_rle_expand(compressed: list[Any], expected: list[int]) -> None:
    """RLE-encoded edges expand correctly"""
    assert expand_rle(compressed) == expected


@pytest.mark.parametrize(
    ("original", "expected"),
    [
        ([10, 10, 10], [[10, 3]]),
        ([10, 10, 20], [[10, 2], 20]),
        ([5], [5]),
        ([10, 20, 30], [10, 20, 30]),
    ],
)
def test_rle_compress(original: list[int], expected: list[Any]) -> None:
    """compress_rle produces the expected RLE encoding for various input sequences"""
    assert compress_rle(original) == expected


def test_rle_roundtrip() -> None:
    """compress_rle followed by expand_rle recovers the original sequence"""
    original = [10, 10, 10, 20, 20, 30]
    compressed = compress_rle(original)
    assert expand_rle(compressed) == original


@pytest.mark.parametrize(
    ("rle_input", "match"),
    [
        ([0], "Chunk edge length must be >= 1"),
        ([-5], "Chunk edge length must be >= 1"),
        ([[0, 3]], "Chunk edge length must be >= 1"),
        ([[-10, 2]], "Chunk edge length must be >= 1"),
        ([[5, 0]], "RLE repeat count must be >= 1"),
        ([[5, -1]], "RLE repeat count must be >= 1"),
    ],
    ids=[
        "zero-edge",
        "negative-edge",
        "zero-rle-size",
        "negative-rle-size",
        "zero-rle-count",
        "negative-rle-count",
    ],
)
def test_rle_expand_rejects_invalid(rle_input: list[Any], match: str) -> None:
    """expand_rle raises ValueError for zero/negative edge lengths or repeat counts"""
    with pytest.raises(ValueError, match=match):
        expand_rle(rle_input)


# -- expand_rle handles JSON floats --


def test_expand_rle_bare_integer_floats_accepted() -> None:
    """JSON parsers may emit 10.0 for the integer 10; expand_rle should handle it."""
    result = expand_rle([10.0, 20.0])  # type: ignore[list-item]
    assert result == [10, 20]


def test_expand_rle_pair_with_float_count() -> None:
    """expand_rle accepts float repeat counts that are integer-valued"""
    result = expand_rle([[10, 3.0]])  # type: ignore[list-item]
    assert result == [10, 10, 10]


# ---------------------------------------------------------------------------
# _is_rectilinear_chunks tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ([[10, 20], [5, 5]], True),
        (((10, 20), (5, 5)), True),
        ((10, 20), False),
        ([10, 20], False),
        (10, False),
        ("auto", False),
        ([], False),
        ([[]], True),
        (ChunkGrid.from_sizes((10,), (5,)), False),
        (None, False),
        (3.14, False),
    ],
    ids=[
        "nested-lists",
        "nested-tuples",
        "flat-tuple",
        "flat-list",
        "single-int",
        "string",
        "empty-list",
        "empty-nested-list",
        "chunk-grid-instance",
        "none",
        "float",
    ],
)
def test_is_rectilinear_chunks(value: Any, expected: bool) -> None:
    """_is_rectilinear_chunks correctly identifies nested sequences as rectilinear"""
    assert _is_rectilinear_chunks(value) is expected


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------


def test_serialization_error_non_regular_chunk_shape() -> None:
    """Accessing chunk_shape on a non-regular grid raises ValueError."""
    grid = ChunkGrid.from_sizes((60, 100), [[10, 20, 30], [25, 25, 25, 25]])
    with pytest.raises(ValueError, match="only available for regular"):
        grid.chunk_shape  # noqa: B018


def test_serialization_error_zero_extent_rectilinear() -> None:
    """RectilinearChunkGrid rejects empty edge tuples."""
    with pytest.raises(ValueError, match="has no chunk edges"):
        RectilinearChunkGrid(chunk_shapes=((),))


def test_serialization_unknown_name_parse() -> None:
    """Parsing metadata with an unknown chunk grid name raises ValueError"""
    with pytest.raises(ValueError, match="Unknown chunk grid"):
        parse_chunk_grid({"name": "hexagonal", "configuration": {}})


# ---------------------------------------------------------------------------
# Spec compliance tests
# ---------------------------------------------------------------------------


def test_spec_kind_inline_required_on_deserialize() -> None:
    """Deserialization requires kind: 'inline'."""
    data: dict[str, Any] = {
        "name": "rectilinear",
        "configuration": {"chunk_shapes": [[10, 20], [15, 15]]},
    }
    with pytest.raises(ValueError, match="requires a 'kind' field"):
        parse_chunk_grid(data)


def test_spec_kind_unknown_rejected() -> None:
    """Unsupported rectilinear chunk grid kind raises ValueError on parse"""
    data: dict[str, Any] = {
        "name": "rectilinear",
        "configuration": {"kind": "reference", "chunk_shapes": [[10, 20], [15, 15]]},
    }
    with pytest.raises(ValueError, match="Unsupported rectilinear chunk grid kind"):
        parse_chunk_grid(data)


def test_spec_integer_shorthand_per_dimension() -> None:
    """A bare integer in chunk_shapes means repeat until >= extent."""
    data: dict[str, Any] = {
        "name": "rectilinear",
        "configuration": {"kind": "inline", "chunk_shapes": [4, [1, 2, 3]]},
    }
    meta = parse_chunk_grid(data)
    g = ChunkGrid.from_sizes((6, 6), meta.chunk_shapes)  # type: ignore[union-attr]
    assert _edges(g, 0) == (4, 4)
    assert _edges(g, 1) == (1, 2, 3)


def test_spec_mixed_rle_and_bare_integers() -> None:
    """An array can mix bare integers and [value, count] RLE pairs."""
    data: dict[str, Any] = {
        "name": "rectilinear",
        "configuration": {"kind": "inline", "chunk_shapes": [[[1, 3], 3]]},
    }
    meta = parse_chunk_grid(data)
    g = ChunkGrid.from_sizes((6,), meta.chunk_shapes)  # type: ignore[union-attr]
    assert _edges(g, 0) == (1, 1, 1, 3)


def test_spec_overflow_chunks_allowed() -> None:
    """Edge sum >= extent is valid (overflow chunks permitted)."""
    data: dict[str, Any] = {
        "name": "rectilinear",
        "configuration": {"kind": "inline", "chunk_shapes": [[4, 4, 4]]},
    }
    meta = parse_chunk_grid(data)
    g = ChunkGrid.from_sizes((6,), meta.chunk_shapes)  # type: ignore[union-attr]
    assert _edges(g, 0) == (4, 4, 4)


def test_spec_example() -> None:
    """The full example from the spec README."""
    data: dict[str, Any] = {
        "name": "rectilinear",
        "configuration": {
            "kind": "inline",
            "chunk_shapes": [
                4,
                [1, 2, 3],
                [[4, 2]],
                [[1, 3], 3],
                [4, 4, 4],
            ],
        },
    }
    meta = parse_chunk_grid(data)
    g = ChunkGrid.from_sizes((6, 6, 6, 6, 6), meta.chunk_shapes)  # type: ignore[union-attr]
    assert _edges(g, 0) == (4, 4)
    assert _edges(g, 1) == (1, 2, 3)
    assert _edges(g, 2) == (4, 4)
    assert _edges(g, 3) == (1, 1, 1, 3)
    assert _edges(g, 4) == (4, 4, 4)


# ---------------------------------------------------------------------------
# parse_chunk_grid validation tests
# ---------------------------------------------------------------------------


def test_parse_chunk_grid_varying_extent_mismatch_raises() -> None:
    """Reconstructing a ChunkGrid with mismatched extents raises ValueError"""
    g = ChunkGrid.from_sizes((60, 100), [[10, 20, 30], [50, 50]])
    with pytest.raises(ValueError, match="extent"):
        ChunkGrid(
            dimensions=tuple(
                dim.with_extent(ext) for dim, ext in zip(g.dimensions, (100, 100), strict=True)
            )
        )


def test_parse_chunk_grid_varying_extent_match_ok() -> None:
    """Reconstructing a ChunkGrid with matching extents succeeds"""
    g = ChunkGrid.from_sizes((60, 100), [[10, 20, 30], [50, 50]])
    g2 = ChunkGrid(
        dimensions=tuple(
            dim.with_extent(ext) for dim, ext in zip(g.dimensions, (60, 100), strict=True)
        )
    )
    assert g2.dimensions[0].extent == 60


@pytest.mark.parametrize(
    ("chunk_shapes", "array_shape", "match"),
    [
        ([[10, 20, 30], [25, 25]], (100, 50), "extent 100 exceeds sum of edges 60"),
        ([[50, 50], [10, 20]], (100, 50), "extent 50 exceeds sum of edges 30"),
    ],
    ids=["first-dim-mismatch", "second-dim-mismatch"],
)
def test_parse_chunk_grid_rectilinear_extent_mismatch_raises(
    chunk_shapes: list[list[int]], array_shape: tuple[int, ...], match: str
) -> None:
    """Rectilinear grid raises ValueError when array extent exceeds sum of edges"""
    data: dict[str, Any] = {
        "name": "rectilinear",
        "configuration": {"kind": "inline", "chunk_shapes": chunk_shapes},
    }
    meta = parse_chunk_grid(data)
    with pytest.raises(ValueError, match=match):
        ChunkGrid.from_sizes(array_shape, meta.chunk_shapes)  # type: ignore[union-attr]


def test_parse_chunk_grid_rectilinear_extent_match_passes() -> None:
    """Rectilinear grid with matching extents parses and builds successfully"""
    data: dict[str, Any] = {
        "name": "rectilinear",
        "configuration": {"kind": "inline", "chunk_shapes": [[10, 20, 30], [25, 25]]},
    }
    meta = parse_chunk_grid(data)
    g = ChunkGrid.from_sizes((60, 50), meta.chunk_shapes)  # type: ignore[union-attr]
    assert g.grid_shape == (3, 2)


def test_parse_chunk_grid_rectilinear_ndim_mismatch_raises() -> None:
    """Mismatched ndim between array shape and chunk_sizes raises ValueError"""
    data: dict[str, Any] = {
        "name": "rectilinear",
        "configuration": {"kind": "inline", "chunk_shapes": [[10, 20], [25, 25]]},
    }
    meta = parse_chunk_grid(data)
    with pytest.raises(ValueError, match="3 dimensions but chunk_sizes has 2"):
        ChunkGrid.from_sizes((30, 50, 100), meta.chunk_shapes)  # type: ignore[union-attr]


def test_parse_chunk_grid_rectilinear_rle_extent_validated() -> None:
    """RLE-encoded edges are expanded before validation."""
    data: dict[str, Any] = {
        "name": "rectilinear",
        "configuration": {"kind": "inline", "chunk_shapes": [[[10, 5]], [[25, 2]]]},
    }
    meta = parse_chunk_grid(data)
    g = ChunkGrid.from_sizes((50, 50), meta.chunk_shapes)  # type: ignore[union-attr]
    assert g.grid_shape == (5, 2)
    with pytest.raises(ValueError, match="extent 100 exceeds sum of edges 50"):
        ChunkGrid.from_sizes((100, 50), meta.chunk_shapes)  # type: ignore[union-attr]


def test_parse_chunk_grid_varying_dimension_extent_mismatch_on_chunkgrid_input() -> None:
    """ChunkGrid constructor rejects VaryingDimension with extent exceeding sum of edges"""
    g = ChunkGrid.from_sizes((60, 50), [[10, 20, 30], [25, 25]])
    with pytest.raises(ValueError, match="less than"):
        ChunkGrid(
            dimensions=tuple(
                dim.with_extent(ext) for dim, ext in zip(g.dimensions, (100, 50), strict=True)
            )
        )


# ---------------------------------------------------------------------------
# Rectilinear indexing tests
# ---------------------------------------------------------------------------


def test_basic_indexer_rectilinear() -> None:
    """BasicIndexer produces correct projections for a full-slice rectilinear selection"""
    from zarr.core.indexing import BasicIndexer

    g = ChunkGrid.from_sizes((60, 100), [[10, 20, 30], [50, 50]])
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


def test_basic_indexer_int_selection() -> None:
    """BasicIndexer with integer selection maps to the correct chunk and local offset"""
    from zarr.core.indexing import BasicIndexer

    g = ChunkGrid.from_sizes((60, 100), [[10, 20, 30], [50, 50]])
    indexer = BasicIndexer(
        selection=(15, slice(None)),
        shape=(60, 100),
        chunk_grid=g,
    )
    projections = list(indexer)
    assert len(projections) == 2
    assert projections[0].chunk_coords == (1, 0)
    assert projections[0].chunk_selection == (5, slice(0, 50, 1))


def test_basic_indexer_slice_subset() -> None:
    """BasicIndexer with partial slices spans the expected chunk dimensions"""
    from zarr.core.indexing import BasicIndexer

    g = ChunkGrid.from_sizes((60, 100), [[10, 20, 30], [50, 50]])
    indexer = BasicIndexer(
        selection=(slice(5, 35), slice(0, 50)),
        shape=(60, 100),
        chunk_grid=g,
    )
    projections = list(indexer)
    chunk_coords_dim0 = sorted({p.chunk_coords[0] for p in projections})
    assert chunk_coords_dim0 == [0, 1, 2]


def test_orthogonal_indexer_rectilinear() -> None:
    """OrthogonalIndexer produces the expected number of projections for a rectilinear grid"""
    from zarr.core.indexing import OrthogonalIndexer

    g = ChunkGrid.from_sizes((60, 100), [[10, 20, 30], [50, 50]])
    indexer = OrthogonalIndexer(
        selection=(slice(None), slice(None)),
        shape=(60, 100),
        chunk_grid=g,
    )
    projections = list(indexer)
    assert len(projections) == 6


def test_oob_block_raises_bounds_check_error() -> None:
    """Out-of-bounds block index should raise BoundsCheckError, not IndexError."""
    store = MemoryStore()
    a = zarr.create_array(store, shape=(30,), chunks=[[10, 20]], dtype="int32")
    with pytest.raises(BoundsCheckError):
        a.get_block_selection((2,))


# ---------------------------------------------------------------------------
# End-to-end tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("shape", "chunks", "expected_regular"),
    [
        ((100, 200), (10, 20), True),
        ((60, 100), [[10, 20, 30], [50, 50]], False),
    ],
    ids=["regular", "rectilinear"],
)
def test_e2e_create_array(
    tmp_path: Path, shape: tuple[int, ...], chunks: Any, expected_regular: bool
) -> None:
    """End-to-end array creation sets correct regularity and ndim on chunk_grid"""
    arr = zarr.create_array(
        store=tmp_path / "arr.zarr",
        shape=shape,
        chunks=chunks,
        dtype="float32",
    )
    assert ChunkGrid.from_metadata(arr.metadata).is_regular == expected_regular
    assert ChunkGrid.from_metadata(arr.metadata).ndim == len(shape)


@pytest.mark.parametrize(
    ("shape", "chunks", "grid_type_name", "grid_name"),
    [
        ((100, 200), (10, 20), "RegularChunkGrid", "regular"),
        ((60, 100), [[10, 20, 30], [50, 50]], "RectilinearChunkGrid", "rectilinear"),
    ],
    ids=["regular", "rectilinear"],
)
def test_e2e_chunk_grid_serializes(
    tmp_path: Path, shape: tuple[int, ...], chunks: Any, grid_type_name: str, grid_name: str
) -> None:
    """Array metadata serializes chunk_grid with the correct type and name"""
    from zarr.core.metadata.v3 import ArrayV3Metadata, RectilinearChunkGrid, RegularChunkGrid

    grid_type = RegularChunkGrid if grid_type_name == "RegularChunkGrid" else RectilinearChunkGrid
    arr = zarr.create_array(
        store=tmp_path / "arr.zarr",
        shape=shape,
        chunks=chunks,
        dtype="float32",
    )
    assert isinstance(arr.metadata, ArrayV3Metadata)
    assert isinstance(arr.metadata.chunk_grid, grid_type)
    d = arr.metadata.to_dict()
    chunk_grid_dict = d["chunk_grid"]
    assert isinstance(chunk_grid_dict, dict)
    assert chunk_grid_dict["name"] == grid_name


def test_e2e_chunk_grid_name_roundtrip_preserves_rectilinear(tmp_path: Path) -> None:
    """A rectilinear grid with uniform edges stays 'rectilinear' through to_dict/from_dict."""
    from zarr.core.metadata.v3 import ArrayV3Metadata, RectilinearChunkGrid

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
    assert isinstance(meta.chunk_grid, RectilinearChunkGrid)
    d = meta.to_dict()
    chunk_grid_dict = d["chunk_grid"]
    assert isinstance(chunk_grid_dict, dict)
    assert chunk_grid_dict["name"] == "rectilinear"


def test_e2e_chunk_grid_name_regular_from_dict(tmp_path: Path) -> None:
    """A 'regular' chunk grid name is preserved through from_dict."""
    from zarr.core.metadata.v3 import ArrayV3Metadata, RegularChunkGrid

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
    assert isinstance(meta.chunk_grid, RegularChunkGrid)
    d = meta.to_dict()
    chunk_grid_dict = d["chunk_grid"]
    assert isinstance(chunk_grid_dict, dict)
    assert chunk_grid_dict["name"] == "regular"


# ---------------------------------------------------------------------------
# Sharding compatibility tests
# ---------------------------------------------------------------------------


def test_sharding_accepts_rectilinear_outer_grid() -> None:
    """ShardingCodec.validate should not reject rectilinear outer grids."""
    from zarr.codecs.sharding import ShardingCodec
    from zarr.core.dtype import Float32
    from zarr.core.metadata.v3 import RectilinearChunkGrid

    codec = ShardingCodec(chunk_shape=(5, 5))
    grid_meta = RectilinearChunkGrid(chunk_shapes=((10, 20, 30), (50, 50)))

    codec.validate(
        shape=(60, 100),
        dtype=Float32(),
        chunk_grid=grid_meta,
    )


def test_sharding_rejects_non_divisible_rectilinear() -> None:
    """Rectilinear shard sizes not divisible by inner chunk_shape should raise."""
    from zarr.codecs.sharding import ShardingCodec
    from zarr.core.dtype import Float32
    from zarr.core.metadata.v3 import RectilinearChunkGrid

    codec = ShardingCodec(chunk_shape=(5, 5))
    grid_meta = RectilinearChunkGrid(chunk_shapes=((10, 20, 17), (50, 50)))

    with pytest.raises(ValueError, match="divisible"):
        codec.validate(
            shape=(47, 100),
            dtype=Float32(),
            chunk_grid=grid_meta,
        )


def test_sharding_accepts_divisible_rectilinear() -> None:
    """Rectilinear shard sizes all divisible by inner chunk_shape should pass."""
    from zarr.codecs.sharding import ShardingCodec
    from zarr.core.dtype import Float32
    from zarr.core.metadata.v3 import RectilinearChunkGrid

    codec = ShardingCodec(chunk_shape=(5, 5))
    grid_meta = RectilinearChunkGrid(chunk_shapes=((10, 20, 30), (50, 50)))

    codec.validate(
        shape=(60, 100),
        dtype=Float32(),
        chunk_grid=grid_meta,
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_edge_case_chunk_grid_boundary_getitem() -> None:
    """ChunkGrid with boundary FixedDimension via direct construction."""
    g = ChunkGrid(dimensions=(FixedDimension(10, 95), FixedDimension(20, 40)))
    spec = g[(9, 1)]
    assert spec is not None
    assert spec.shape == (5, 20)
    assert spec.codec_shape == (10, 20)
    assert spec.is_boundary


def test_edge_case_chunk_grid_boundary_iter() -> None:
    """Iterating a boundary grid yields correct boundary ChunkSpecs."""
    g = ChunkGrid(dimensions=(FixedDimension(10, 25),))
    specs = list(g)
    assert len(specs) == 3
    assert specs[0].shape == (10,)
    assert specs[1].shape == (10,)
    assert specs[2].shape == (5,)
    assert specs[2].is_boundary
    assert not specs[0].is_boundary


def test_edge_case_chunk_grid_boundary_shape() -> None:
    """shape property with boundary extent."""
    g = ChunkGrid(dimensions=(FixedDimension(10, 95),))
    assert g.grid_shape == (10,)


# -- Zero-size and zero-extent --


@pytest.mark.parametrize(
    ("size", "extent"),
    [(0, 0), (0, 5), (10, 0)],
    ids=["zero-size-zero-extent", "zero-size-nonzero-extent", "zero-extent-nonzero-size"],
)
def test_edge_case_zero_size_or_extent(size: int, extent: int) -> None:
    """FixedDimension with zero size or extent has zero chunks and getitem returns None"""
    d = FixedDimension(size=size, extent=extent)
    assert d.nchunks == 0
    g = ChunkGrid(dimensions=(d,))
    assert g[0] is None


# -- 0-d grid --


def test_0d_grid_getitem() -> None:
    """0-d grid has exactly one chunk at coords ()."""
    g = ChunkGrid.from_sizes((), ())
    spec = g[()]
    assert spec is not None
    assert spec.shape == ()
    assert spec.codec_shape == ()
    assert not spec.is_boundary


def test_0d_grid_iter() -> None:
    """0-d grid iteration yields a single ChunkSpec."""
    g = ChunkGrid.from_sizes((), ())
    specs = list(g)
    assert len(specs) == 1


def test_0d_grid_all_chunk_coords() -> None:
    """0-d grid has one chunk coord: the empty tuple."""
    g = ChunkGrid.from_sizes((), ())
    coords = list(g.all_chunk_coords())
    assert coords == [()]


def test_0d_grid_nchunks() -> None:
    """0-d grid reports exactly one chunk"""
    g = ChunkGrid.from_sizes((), ())
    assert g.get_nchunks() == 1


# -- parse_chunk_grid edge cases --


def test_parse_chunk_grid_preserves_varying_extent() -> None:
    """parse_chunk_grid does not overwrite VaryingDimension extent."""
    g = ChunkGrid.from_sizes((60, 100), [[10, 20, 30], [50, 50]])
    assert isinstance(g.dimensions[0], VaryingDimension)
    assert g.dimensions[0].extent == 60

    g2 = ChunkGrid(
        dimensions=tuple(
            dim.with_extent(ext) for dim, ext in zip(g.dimensions, (60, 100), strict=True)
        )
    )
    assert isinstance(g2.dimensions[0], VaryingDimension)
    assert g2.dimensions[0].extent == 60


def test_parse_chunk_grid_rebinds_fixed_extent() -> None:
    """parse_chunk_grid updates FixedDimension extent from array shape."""
    g = ChunkGrid.from_sizes((100, 200), (10, 20))
    assert g.dimensions[0].extent == 100

    g2 = ChunkGrid(
        dimensions=tuple(
            dim.with_extent(ext) for dim, ext in zip(g.dimensions, (50, 100), strict=True)
        )
    )
    assert isinstance(g2.dimensions[0], FixedDimension)
    assert g2.dimensions[0].extent == 50
    assert g2.grid_shape == (5, 5)


# -- ChunkGrid.__getitem__ validation --


def test_getitem_int_1d_regular() -> None:
    """Integer indexing works for 1-d regular grids."""
    g = ChunkGrid.from_sizes((100,), (10,))
    spec = g[0]
    assert spec is not None
    assert spec.shape == (10,)
    assert spec.slices == (slice(0, 10, 1),)
    spec = g[9]
    assert spec is not None
    assert spec.shape == (10,)


def test_getitem_int_1d_rectilinear() -> None:
    """Integer indexing works for 1-d rectilinear grids."""
    g = ChunkGrid.from_sizes((100,), [[20, 30, 50]])
    spec = g[0]
    assert spec is not None
    assert spec.shape == (20,)
    spec = g[1]
    assert spec is not None
    assert spec.shape == (30,)
    spec = g[2]
    assert spec is not None
    assert spec.shape == (50,)


@pytest.mark.parametrize(
    ("shape", "chunks", "match"),
    [
        ((), (), "Expected 0 coordinate.*got 1"),
        ((100, 200), (10, 20), "Expected 2 coordinate.*got 1"),
    ],
    ids=["0d", "2d"],
)
def test_getitem_int_ndim_mismatch_raises(
    shape: tuple[int, ...], chunks: tuple[int, ...], match: str
) -> None:
    """Integer indexing on a multi-dim or 0-d grid raises ValueError for ndim mismatch"""
    g = ChunkGrid.from_sizes(shape, chunks)
    with pytest.raises(ValueError, match=match):
        g[0]


@pytest.mark.parametrize(
    "index",
    [(10,), (99,), (-1,)],
    ids=["oob-10", "oob-99", "negative"],
)
def test_getitem_oob_returns_none(index: tuple[int, ...]) -> None:
    """Out-of-bounds or negative chunk indices return None"""
    g = ChunkGrid.from_sizes((100,), (10,))
    assert g[index] is None


# -- Rectilinear with zero-nchunks FixedDimension --


def test_zero_nchunks_fixed_dim_in_rectilinear() -> None:
    """A rectilinear grid with a 0-extent FixedDimension still has valid size."""
    g = ChunkGrid(
        dimensions=(
            VaryingDimension([10, 20], extent=30),
            FixedDimension(size=10, extent=0),
        )
    )
    assert g.grid_shape == (2, 0)


# -- VaryingDimension data_size --


def test_varying_dim_data_size_equals_chunk_size() -> None:
    """For VaryingDimension, data_size == chunk_size (no padding)."""
    d = VaryingDimension([10, 20, 5], extent=35)
    for i in range(3):
        assert d.data_size(i) == d.chunk_size(i)


# ---------------------------------------------------------------------------
# OrthogonalIndexer rectilinear tests
# ---------------------------------------------------------------------------


def test_orthogonal_int_array_selection_rectilinear() -> None:
    """Integer array selection with rectilinear grid must produce correct
    chunk-local selections."""
    from zarr.core.indexing import OrthogonalIndexer

    g = ChunkGrid.from_sizes((60, 100), [[10, 20, 30], [50, 50]])
    indexer = OrthogonalIndexer(
        selection=(np.array([5, 15, 35]), slice(None)),
        shape=(60, 100),
        chunk_grid=g,
    )
    projections = list(indexer)
    chunk_coords = [p.chunk_coords for p in projections]
    assert chunk_coords == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]


def test_orthogonal_bool_array_selection_rectilinear() -> None:
    """Boolean array selection with rectilinear grid produces correct chunk projections."""
    from zarr.core.indexing import OrthogonalIndexer

    g = ChunkGrid.from_sizes((60, 100), [[10, 20, 30], [50, 50]])
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
    assert len(projections) == 6
    chunk_coords = [p.chunk_coords for p in projections]
    assert (0, 0) in chunk_coords
    assert (1, 0) in chunk_coords
    assert (2, 0) in chunk_coords
    assert (0, 1) in chunk_coords
    assert (1, 1) in chunk_coords
    assert (2, 1) in chunk_coords


def test_orthogonal_advanced_indexing_produces_correct_projections() -> None:
    """Verify OrthogonalIndexer produces correct chunk projections
    for advanced indexing with VaryingDimension."""
    from zarr.core.indexing import OrthogonalIndexer

    g = ChunkGrid.from_sizes((60, 100), [[10, 20, 30], [50, 50]])
    indexer = OrthogonalIndexer(
        selection=(np.array([5, 15]), slice(None)),
        shape=(60, 100),
        chunk_grid=g,
    )
    projections = list(indexer)
    assert len(projections) == 4
    coords = [p.chunk_coords for p in projections]
    assert (0, 0) in coords
    assert (0, 1) in coords
    assert (1, 0) in coords
    assert (1, 1) in coords


# ---------------------------------------------------------------------------
# Full pipeline rectilinear tests (helpers)
# ---------------------------------------------------------------------------


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


def test_pipeline_basic_selection_1d(tmp_path: Path) -> None:
    """1D rectilinear basic selections match numpy for ints, slices, and full-array reads"""
    z, a = _make_1d(tmp_path)
    sels: list[Any] = [0, 4, 5, 14, 15, 29, -1, slice(None), slice(3, 18), slice(0, 0)]
    for sel in sels:
        np.testing.assert_array_equal(z[sel], a[sel], err_msg=f"sel={sel}")


def test_pipeline_basic_selection_1d_strided(tmp_path: Path) -> None:
    """1D rectilinear strided slice selections match numpy"""
    z, a = _make_1d(tmp_path)
    for sel in [slice(None, None, 2), slice(1, 25, 3), slice(0, 30, 7)]:
        np.testing.assert_array_equal(z[sel], a[sel], err_msg=f"sel={sel}")


def test_pipeline_basic_selection_2d(tmp_path: Path) -> None:
    """2D rectilinear basic selections match numpy across chunk boundaries"""
    z, a = _make_2d(tmp_path)
    selections: list[Any] = [
        42,
        -1,
        (9, 24),
        (10, 25),
        (30, 50),
        (59, 99),
        slice(None),
        (slice(5, 35), slice(20, 80)),
        (slice(0, 10), slice(0, 25)),
        (slice(10, 10), slice(None)),
        (slice(None, None, 3), slice(None, None, 7)),
    ]
    for sel in selections:
        np.testing.assert_array_equal(z[sel], a[sel], err_msg=f"sel={sel}")


# --- Orthogonal selection ---


def test_pipeline_orthogonal_selection_1d_bool(tmp_path: Path) -> None:
    """1D boolean orthogonal indexing on rectilinear arrays matches numpy"""
    z, a = _make_1d(tmp_path)
    ix = np.zeros(30, dtype=bool)
    ix[[0, 4, 5, 14, 15, 29]] = True
    np.testing.assert_array_equal(z.oindex[ix], a[ix])


def test_pipeline_orthogonal_selection_1d_int(tmp_path: Path) -> None:
    """1D integer and negative-index orthogonal selection on rectilinear arrays matches numpy"""
    z, a = _make_1d(tmp_path)
    ix = np.array([0, 4, 5, 14, 15, 29])
    np.testing.assert_array_equal(z.oindex[ix], a[ix])
    ix_neg = np.array([0, -1, -15, -25])
    np.testing.assert_array_equal(z.oindex[ix_neg], a[ix_neg])


def test_pipeline_orthogonal_selection_2d_bool(tmp_path: Path) -> None:
    """2D boolean orthogonal selection on rectilinear arrays matches numpy"""
    z, a = _make_2d(tmp_path)
    ix0 = np.zeros(60, dtype=bool)
    ix0[[0, 9, 10, 29, 30, 59]] = True
    ix1 = np.zeros(100, dtype=bool)
    ix1[[0, 24, 25, 49, 50, 99]] = True
    np.testing.assert_array_equal(z.oindex[ix0, ix1], a[np.ix_(ix0, ix1)])


def test_pipeline_orthogonal_selection_2d_int(tmp_path: Path) -> None:
    """2D integer orthogonal selection on rectilinear arrays matches numpy"""
    z, a = _make_2d(tmp_path)
    ix0 = np.array([0, 9, 10, 29, 30, 59])
    ix1 = np.array([0, 24, 25, 49, 50, 99])
    np.testing.assert_array_equal(z.oindex[ix0, ix1], a[np.ix_(ix0, ix1)])


def test_pipeline_orthogonal_selection_2d_mixed(tmp_path: Path) -> None:
    """2D mixed int-array and slice orthogonal selection on rectilinear arrays matches numpy"""
    z, a = _make_2d(tmp_path)
    ix = np.array([0, 9, 10, 29, 30, 59])
    np.testing.assert_array_equal(z.oindex[ix, slice(25, 75)], a[np.ix_(ix, np.arange(25, 75))])
    np.testing.assert_array_equal(
        z.oindex[slice(10, 30), ix[:4]], a[np.ix_(np.arange(10, 30), ix[:4])]
    )


# --- Coordinate (vindex) selection ---


def test_pipeline_coordinate_selection_1d(tmp_path: Path) -> None:
    """1D coordinate (vindex) selection on rectilinear arrays matches numpy"""
    z, a = _make_1d(tmp_path)
    ix = np.array([0, 4, 5, 14, 15, 29])
    np.testing.assert_array_equal(z.vindex[ix], a[ix])


def test_pipeline_coordinate_selection_2d(tmp_path: Path) -> None:
    """2D coordinate (vindex) selection on rectilinear arrays matches numpy"""
    z, a = _make_2d(tmp_path)
    r = np.array([0, 9, 10, 29, 30, 59])
    c = np.array([0, 24, 25, 49, 50, 99])
    np.testing.assert_array_equal(z.vindex[r, c], a[r, c])


def test_pipeline_coordinate_selection_2d_bool_mask(tmp_path: Path) -> None:
    """2D boolean mask vindex selection on rectilinear arrays matches numpy"""
    z, a = _make_2d(tmp_path)
    mask = a > 3000
    np.testing.assert_array_equal(z.vindex[mask], a[mask])


# --- Block selection ---


def test_pipeline_block_selection_1d(tmp_path: Path) -> None:
    """1D block selection on rectilinear arrays returns correct chunk data"""
    z, a = _make_1d(tmp_path)
    np.testing.assert_array_equal(z.blocks[0], a[0:5])
    np.testing.assert_array_equal(z.blocks[1], a[5:15])
    np.testing.assert_array_equal(z.blocks[2], a[15:30])
    np.testing.assert_array_equal(z.blocks[-1], a[15:30])
    np.testing.assert_array_equal(z.blocks[0:2], a[0:15])
    np.testing.assert_array_equal(z.blocks[1:3], a[5:30])
    np.testing.assert_array_equal(z.blocks[:], a[:])


def test_pipeline_block_selection_2d(tmp_path: Path) -> None:
    """2D block selection on rectilinear arrays returns correct chunk data"""
    z, a = _make_2d(tmp_path)
    np.testing.assert_array_equal(z.blocks[0, 0], a[0:10, 0:25])
    np.testing.assert_array_equal(z.blocks[1, 2], a[10:30, 50:75])
    np.testing.assert_array_equal(z.blocks[2, 3], a[30:60, 75:100])
    np.testing.assert_array_equal(z.blocks[-1, -1], a[30:60, 75:100])
    np.testing.assert_array_equal(z.blocks[0:2, 1:3], a[0:30, 25:75])
    np.testing.assert_array_equal(z.blocks[:, :], a[:, :])


def test_pipeline_set_block_selection_1d(tmp_path: Path) -> None:
    """Writing via 1D block selection on rectilinear arrays persists correctly"""
    z, a = _make_1d(tmp_path)
    val = np.full(10, -1, dtype="int32")
    z.blocks[1] = val
    a[5:15] = val
    np.testing.assert_array_equal(z[:], a)


def test_pipeline_set_block_selection_2d(tmp_path: Path) -> None:
    """Writing via 2D block selection on rectilinear arrays persists correctly"""
    z, a = _make_2d(tmp_path)
    val = np.full((30, 50), -99, dtype="int32")
    z.blocks[0:2, 1:3] = val
    a[0:30, 25:75] = val
    np.testing.assert_array_equal(z[:], a)


def test_pipeline_block_selection_slice_stop_at_nchunks(tmp_path: Path) -> None:
    """Block slice with stop == nchunks exercises the dim_len fallback."""
    z, a = _make_1d(tmp_path)
    np.testing.assert_array_equal(z.blocks[1:3], a[5:30])
    np.testing.assert_array_equal(z.blocks[0:10], a[:])


def test_pipeline_block_selection_slice_stop_at_nchunks_2d(tmp_path: Path) -> None:
    """Same fallback test for 2D rectilinear arrays."""
    z, a = _make_2d(tmp_path)
    np.testing.assert_array_equal(z.blocks[2:3, 3:4], a[30:60, 75:100])
    np.testing.assert_array_equal(z.blocks[0:99, 0:99], a[:, :])


# --- Set coordinate selection ---


def test_pipeline_set_coordinate_selection_1d(tmp_path: Path) -> None:
    """Writing via 1D coordinate selection on rectilinear arrays persists correctly"""
    z, a = _make_1d(tmp_path)
    ix = np.array([0, 4, 5, 14, 15, 29])
    val = np.full(len(ix), -7, dtype="int32")
    z.vindex[ix] = val
    a[ix] = val
    np.testing.assert_array_equal(z[:], a)


def test_pipeline_set_coordinate_selection_2d(tmp_path: Path) -> None:
    """Writing via 2D coordinate selection on rectilinear arrays persists correctly"""
    z, a = _make_2d(tmp_path)
    r = np.array([0, 9, 10, 29, 30, 59])
    c = np.array([0, 24, 25, 49, 50, 99])
    val = np.full(len(r), -42, dtype="int32")
    z.vindex[r, c] = val
    a[r, c] = val
    np.testing.assert_array_equal(z[:], a)


# --- Set selection ---


def test_pipeline_set_basic_selection(tmp_path: Path) -> None:
    """Writing via basic slice selection on rectilinear arrays persists correctly"""
    z, a = _make_2d(tmp_path)
    new_data = np.full((20, 50), -1, dtype="int32")
    z[5:25, 10:60] = new_data
    a[5:25, 10:60] = new_data
    np.testing.assert_array_equal(z[:], a)


def test_pipeline_set_orthogonal_selection(tmp_path: Path) -> None:
    """Writing via orthogonal selection on rectilinear arrays persists correctly"""
    z, a = _make_2d(tmp_path)
    rows = np.array([0, 10, 30])
    cols = np.array([0, 25, 50, 75])
    val = np.full((3, 4), -99, dtype="int32")
    z.oindex[rows, cols] = val
    a[np.ix_(rows, cols)] = val
    np.testing.assert_array_equal(z[:], a)


# --- Higher dimensions ---


def test_pipeline_3d_array(tmp_path: Path) -> None:
    """3D rectilinear array write and read-back match numpy"""
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


def test_pipeline_1d_single_chunk(tmp_path: Path) -> None:
    """Single-chunk rectilinear array write and read-back match numpy"""
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


def test_pipeline_persistence_roundtrip(tmp_path: Path) -> None:
    """Rectilinear array survives close and reopen with correct data"""
    _, a = _make_2d(tmp_path)
    z2 = zarr.open_array(store=tmp_path / "arr2d.zarr", mode="r")
    assert not ChunkGrid.from_metadata(z2.metadata).is_regular
    np.testing.assert_array_equal(z2[:], a)


# --- Highly irregular chunks ---


def test_pipeline_highly_irregular_chunks(tmp_path: Path) -> None:
    """Highly irregular chunk sizes produce correct write and partial-read results"""
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


def test_pipeline_v2_rejects_rectilinear(tmp_path: Path) -> None:
    """Creating a rectilinear array with zarr_format=2 raises ValueError"""
    with pytest.raises(ValueError, match="Zarr format 2"):
        zarr.create_array(
            store=tmp_path / "v2.zarr",
            shape=(30,),
            chunks=[[10, 20]],
            dtype="int32",
            zarr_format=2,
        )


def test_pipeline_sharding_rejects_rectilinear_chunks_with_shards(tmp_path: Path) -> None:
    """Rectilinear chunks (inner) with sharding is not supported."""
    with pytest.raises(ValueError, match="Rectilinear chunks with sharding"):
        zarr.create_array(
            store=tmp_path / "shard.zarr",
            shape=(60, 100),
            chunks=[[10, 20, 30], [25, 25, 25, 25]],
            shards=(30, 50),
            dtype="int32",
        )


def test_pipeline_rectilinear_shards_roundtrip(tmp_path: Path) -> None:
    """Rectilinear shards with uniform inner chunks: full write/read roundtrip."""
    data = np.arange(120 * 100, dtype="int32").reshape(120, 100)
    arr = zarr.create_array(
        store=tmp_path / "rect_shards.zarr",
        shape=(120, 100),
        chunks=(10, 10),
        shards=[[60, 40, 20], [50, 50]],
        dtype="int32",
    )
    arr[:] = data
    result = arr[:]
    np.testing.assert_array_equal(result, data)


def test_pipeline_rectilinear_shards_partial_read(tmp_path: Path) -> None:
    """Partial reads across rectilinear shard boundaries."""
    data = np.arange(120 * 100, dtype="float64").reshape(120, 100)
    arr = zarr.create_array(
        store=tmp_path / "rect_shards.zarr",
        shape=(120, 100),
        chunks=(10, 10),
        shards=[[60, 40, 20], [50, 50]],
        dtype="float64",
    )
    arr[:] = data
    result = arr[50:70, 40:60]
    np.testing.assert_array_equal(result, data[50:70, 40:60])


def test_pipeline_rectilinear_shards_validates_divisibility(tmp_path: Path) -> None:
    """Inner chunk_shape must divide every shard's dimensions."""
    with pytest.raises(ValueError, match="divisible"):
        zarr.create_array(
            store=tmp_path / "bad.zarr",
            shape=(120, 100),
            chunks=(10, 10),
            shards=[[60, 45, 15], [50, 50]],
            dtype="int32",
        )


def test_pipeline_nchunks(tmp_path: Path) -> None:
    """Rectilinear array reports the correct total number of chunks"""
    z, _ = _make_2d(tmp_path)
    assert ChunkGrid.from_metadata(z.metadata).get_nchunks() == 12


def test_pipeline_parse_chunk_grid_regular_from_dict() -> None:
    """parse_chunk_grid constructs a regular grid from a metadata dict."""
    d: dict[str, Any] = {"name": "regular", "configuration": {"chunk_shape": [10, 20]}}
    meta = parse_chunk_grid(d)
    assert isinstance(meta, RegularChunkGridMeta)
    g = ChunkGrid.from_sizes((100, 200), tuple(meta.chunk_shape))
    assert g.is_regular
    assert g.chunk_shape == (10, 20)
    assert g.grid_shape == (10, 10)
    assert g.get_nchunks() == 100


# ---------------------------------------------------------------------------
# VaryingDimension boundary tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("edges", "extent", "chunk_idx", "expected_data_size"),
    [
        ([10, 20, 30], 50, 0, 10),
        ([10, 20, 30], 50, 1, 20),
        ([10, 20, 30], 50, 2, 20),
        ([10, 20, 30], 60, 2, 30),
        ([10, 20, 30], 31, 0, 10),
        ([10, 20, 30], 31, 1, 20),
        ([10, 20, 30], 31, 2, 1),
    ],
    ids=[
        "interior-0",
        "interior-1",
        "boundary-clipped",
        "exact-no-clip",
        "single-element-boundary-0",
        "single-element-boundary-1",
        "single-element-boundary-2",
    ],
)
def test_varying_dimension_boundary_data_size(
    edges: list[int], extent: int, chunk_idx: int, expected_data_size: int
) -> None:
    """VaryingDimension.data_size clips correctly at boundary chunks"""
    d = VaryingDimension(edges, extent=extent)
    assert d.data_size(chunk_idx) == expected_data_size


def test_varying_dimension_boundary_extent_parameter() -> None:
    """VaryingDimension preserves extent and full chunk_size even when extent < sum of edges"""
    d = VaryingDimension([10, 20, 30], extent=50)
    assert d.extent == 50
    assert d.chunk_size(2) == 30


def test_varying_dimension_extent_exceeds_sum_rejected() -> None:
    """VaryingDimension rejects extent greater than sum of edges"""
    with pytest.raises(ValueError, match="exceeds sum of edges"):
        VaryingDimension([10, 20], extent=50)


def test_varying_dimension_negative_extent_rejected() -> None:
    """VaryingDimension rejects negative extent"""
    with pytest.raises(ValueError, match="must be >= 0"):
        VaryingDimension([10, 20], extent=-1)


def test_varying_dimension_boundary_chunk_spec() -> None:
    """ChunkGrid with a boundary VaryingDimension produces correct ChunkSpec."""
    g = ChunkGrid(dimensions=(VaryingDimension([10, 20, 30], extent=50),))
    spec = g[(2,)]
    assert spec is not None
    assert spec.codec_shape == (30,)
    assert spec.shape == (20,)
    assert spec.is_boundary is True


def test_varying_dimension_interior_chunk_spec() -> None:
    """Interior VaryingDimension chunk has matching codec_shape and shape with no boundary"""
    g = ChunkGrid(dimensions=(VaryingDimension([10, 20, 30], extent=50),))
    spec = g[(0,)]
    assert spec is not None
    assert spec.codec_shape == (10,)
    assert spec.shape == (10,)
    assert spec.is_boundary is False


# ---------------------------------------------------------------------------
# Multiple overflow chunks tests
# ---------------------------------------------------------------------------


def test_overflow_multiple_chunks_past_extent() -> None:
    """Edges past extent are structural; nchunks counts active only."""
    g = ChunkGrid.from_sizes((50,), [[10, 20, 30, 40]])
    d = g.dimensions[0]
    assert d.ngridcells == 4
    assert d.nchunks == 3
    assert d.data_size(0) == 10
    assert d.data_size(1) == 20
    assert d.data_size(2) == 20
    assert d.chunk_size(2) == 30


def test_overflow_chunk_spec_past_extent_is_oob() -> None:
    """Chunk entirely past the extent is out of bounds (not active)."""
    g = ChunkGrid.from_sizes((50,), [[10, 20, 30, 40]])
    spec = g[(3,)]
    assert spec is None


def test_overflow_chunk_spec_partial() -> None:
    """ChunkSpec for a partially-overflowing chunk clips correctly."""
    g = ChunkGrid.from_sizes((50,), [[10, 20, 30, 40]])
    spec = g[(2,)]
    assert spec is not None
    assert spec.shape == (20,)
    assert spec.codec_shape == (30,)
    assert spec.is_boundary is True
    assert spec.slices == (slice(30, 50, 1),)


def test_overflow_chunk_sizes() -> None:
    """chunk_sizes only includes active chunks."""
    g = ChunkGrid.from_sizes((50,), [[10, 20, 30, 40]])
    assert g.chunk_sizes == ((10, 20, 20),)


def test_overflow_multidim() -> None:
    """Overflow in multiple dimensions simultaneously."""
    g = ChunkGrid.from_sizes((45, 100), [[10, 20, 30], [40, 40, 40]])
    assert g.chunk_sizes == ((10, 20, 15), (40, 40, 20))
    spec = g[(2, 2)]
    assert spec is not None
    assert spec.shape == (15, 20)
    assert spec.codec_shape == (30, 40)


def test_overflow_uniform_edges_collapses_to_fixed() -> None:
    """Uniform edges where len == ceildiv(extent, edge) collapse to FixedDimension."""
    g = ChunkGrid.from_sizes((35,), [[10, 10, 10, 10]])
    assert isinstance(g.dimensions[0], FixedDimension)
    assert g.is_regular
    assert g.chunk_sizes == ((10, 10, 10, 5),)
    assert g.dimensions[0].nchunks == 4


def test_overflow_index_to_chunk_near_extent() -> None:
    """Index lookup near and at the extent boundary."""
    d = VaryingDimension([10, 20, 30, 40], extent=50)
    assert d.index_to_chunk(29) == 1
    assert d.index_to_chunk(30) == 2
    assert d.index_to_chunk(49) == 2


# ---------------------------------------------------------------------------
# Boundary indexing tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    (
        "dim",
        "mask",
        "dim_len",
        "expected_chunk_ix",
        "expected_sel_len",
        "expected_first_two",
        "expected_third",
    ),
    [
        (
            FixedDimension(size=5, extent=7),
            np.array([False, False, False, False, False, True, True]),
            7,
            1,
            5,
            (np.True_, np.True_),
            np.False_,
        ),
        (
            VaryingDimension([5, 10], extent=7),
            np.array([False, False, False, False, False, True, True]),
            7,
            1,
            10,
            (np.True_, np.True_),
            np.False_,
        ),
    ],
    ids=["fixed-boundary", "varying-boundary"],
)
def test_bool_indexer_boundary(
    dim: FixedDimension | VaryingDimension,
    mask: np.ndarray[Any, Any],
    dim_len: int,
    expected_chunk_ix: int,
    expected_sel_len: int,
    expected_first_two: tuple[Any, Any],
    expected_third: Any,
) -> None:
    """BoolArrayDimIndexer pads to codec size for boundary chunks."""
    from zarr.core.indexing import BoolArrayDimIndexer

    indexer = BoolArrayDimIndexer(mask, dim_len, dim)
    projections = list(indexer)
    assert len(projections) == 1
    p = projections[0]
    assert p.dim_chunk_ix == expected_chunk_ix
    sel = p.dim_chunk_sel
    assert isinstance(sel, np.ndarray)
    assert sel.shape[0] == expected_sel_len
    assert sel[0] is expected_first_two[0]
    assert sel[1] is expected_first_two[1]
    assert sel[2] is expected_third


def test_bool_indexer_no_padding_interior() -> None:
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
    assert sel.shape[0] == 5


def test_slice_indexer_varying_boundary() -> None:
    """SliceDimIndexer clips to data_size at boundary for VaryingDimension."""
    from zarr.core.indexing import SliceDimIndexer

    dim = VaryingDimension([5, 10], extent=7)
    indexer = SliceDimIndexer(slice(None), 7, dim)
    projections = list(indexer)
    assert len(projections) == 2
    assert projections[0].dim_chunk_sel == slice(0, 5, 1)
    assert projections[1].dim_chunk_sel == slice(0, 2, 1)


def test_int_array_indexer_varying_boundary() -> None:
    """IntArrayDimIndexer handles indices near boundary correctly."""
    from zarr.core.indexing import IntArrayDimIndexer

    dim = VaryingDimension([5, 10], extent=7)
    indices = np.array([6])
    indexer = IntArrayDimIndexer(indices, 7, dim)
    projections = list(indexer)
    assert len(projections) == 1
    assert projections[0].dim_chunk_ix == 1
    sel = projections[0].dim_chunk_sel
    assert isinstance(sel, np.ndarray)
    np.testing.assert_array_equal(sel, [1])


@pytest.mark.parametrize(
    "dim",
    [FixedDimension(size=2, extent=10), VaryingDimension([5, 5], extent=10)],
    ids=["fixed", "varying"],
)
def test_slice_indexer_empty_slice_at_boundary(dim: FixedDimension | VaryingDimension) -> None:
    """SliceDimIndexer yields no projections for an empty slice at the dimension boundary."""
    from zarr.core.indexing import SliceDimIndexer

    indexer = SliceDimIndexer(slice(10, 10), 10, dim)
    projections = list(indexer)
    assert len(projections) == 0


def test_orthogonal_indexer_varying_boundary_advanced() -> None:
    """OrthogonalIndexer with advanced indexing uses per-chunk chunk_size."""
    from zarr.core.indexing import OrthogonalIndexer

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
    assert len(projections) == 4
    coords = {p.chunk_coords for p in projections}
    assert coords == {(0, 0), (0, 1), (1, 0), (1, 1)}


# ---------------------------------------------------------------------------
# update_shape tests
# ---------------------------------------------------------------------------


def test_update_shape_no_change() -> None:
    """update_shape with the same shape preserves edges unchanged"""
    grid = ChunkGrid.from_sizes((60, 50), [[10, 20, 30], [25, 25]])
    new_grid = grid.update_shape((60, 50))
    assert _edges(new_grid, 0) == (10, 20, 30)
    assert _edges(new_grid, 1) == (25, 25)


def test_update_shape_grow_single_dim() -> None:
    """Growing a single dimension appends a new edge chunk"""
    grid = ChunkGrid.from_sizes((60, 50), [[10, 20, 30], [25, 25]])
    new_grid = grid.update_shape((80, 50))
    assert _edges(new_grid, 0) == (10, 20, 30, 20)
    assert _edges(new_grid, 1) == (25, 25)


def test_update_shape_grow_multiple_dims() -> None:
    """Growing multiple dimensions appends correctly sized edge chunks"""
    grid = ChunkGrid.from_sizes((30, 50), [[10, 20], [20, 30]])
    new_grid = grid.update_shape((45, 65))
    assert _edges(new_grid, 0) == (10, 20, 15)
    assert _edges(new_grid, 1) == (20, 30, 15)


def test_update_shape_shrink_single_dim() -> None:
    """Shrinking a single dimension reduces nchunks while preserving edges"""
    grid = ChunkGrid.from_sizes((100, 50), [[10, 20, 30, 40], [25, 25]])
    new_grid = grid.update_shape((35, 50))
    assert _edges(new_grid, 0) == (10, 20, 30, 40)
    assert new_grid.dimensions[0].nchunks == 3
    assert _edges(new_grid, 1) == (25, 25)


def test_update_shape_shrink_to_single_chunk() -> None:
    """Shrinking to fit within the first chunk reduces nchunks to 1"""
    grid = ChunkGrid.from_sizes((60, 50), [[10, 20, 30], [25, 25]])
    new_grid = grid.update_shape((5, 50))
    assert _edges(new_grid, 0) == (10, 20, 30)
    assert new_grid.dimensions[0].nchunks == 1
    assert _edges(new_grid, 1) == (25, 25)


def test_update_shape_shrink_multiple_dims() -> None:
    """Shrinking multiple dimensions reduces nchunks in each dimension"""
    grid = ChunkGrid.from_sizes((40, 60), [[10, 10, 15, 5], [20, 25, 15]])
    new_grid = grid.update_shape((25, 35))
    assert _edges(new_grid, 0) == (10, 10, 15, 5)
    assert new_grid.dimensions[0].nchunks == 3
    assert _edges(new_grid, 1) == (20, 25, 15)
    assert new_grid.dimensions[1].nchunks == 2


def test_update_shape_dimension_mismatch_error() -> None:
    """update_shape raises ValueError when new shape has different ndim"""
    grid = ChunkGrid.from_sizes((30, 70), [[10, 20], [30, 40]])
    with pytest.raises(ValueError, match="dimensions"):
        grid.update_shape((30, 70, 100))


def test_update_shape_boundary_cases() -> None:
    """update_shape handles grow-one-dim and shrink-both-dims edge cases correctly"""
    grid = ChunkGrid.from_sizes((60, 40), [[10, 20, 30], [15, 25]])
    new_grid = grid.update_shape((60, 65))
    assert _edges(new_grid, 0) == (10, 20, 30)
    assert _edges(new_grid, 1) == (15, 25, 25)

    grid2 = ChunkGrid.from_sizes((60, 50), [[10, 20, 30], [15, 25, 10]])
    new_grid2 = grid2.update_shape((30, 40))
    assert _edges(new_grid2, 0) == (10, 20, 30)
    assert new_grid2.dimensions[0].nchunks == 2
    assert _edges(new_grid2, 1) == (15, 25, 10)
    assert new_grid2.dimensions[1].nchunks == 2


def test_update_shape_regular_preserves_extents(tmp_path: Path) -> None:
    """Resize a regular array -- chunk_grid extents must match new shape."""
    z = zarr.create_array(
        store=tmp_path / "regular.zarr",
        shape=(100,),
        chunks=(10,),
        dtype="int32",
    )
    z[:] = np.arange(100, dtype="int32")
    z.resize(50)
    assert z.shape == (50,)
    assert ChunkGrid.from_metadata(z.metadata).dimensions[0].extent == 50


# ---------------------------------------------------------------------------
# update_shape boundary tests
# ---------------------------------------------------------------------------


def test_update_shape_shrink_creates_boundary() -> None:
    """Shrinking extent into a chunk creates a boundary with clipped data_size"""
    grid = ChunkGrid.from_sizes((60,), [[10, 20, 30]])
    new_grid = grid.update_shape((45,))
    dim = new_grid.dimensions[0]
    assert isinstance(dim, VaryingDimension)
    assert dim.edges == (10, 20, 30)
    assert dim.extent == 45
    assert dim.chunk_size(2) == 30
    assert dim.data_size(2) == 15


def test_update_shape_shrink_to_exact_boundary() -> None:
    """Shrinking to an exact chunk boundary reduces nchunks without partial data"""
    grid = ChunkGrid.from_sizes((60,), [[10, 20, 30]])
    new_grid = grid.update_shape((30,))
    dim = new_grid.dimensions[0]
    assert isinstance(dim, VaryingDimension)
    assert dim.edges == (10, 20, 30)
    assert dim.nchunks == 2
    assert dim.ngridcells == 3
    assert dim.extent == 30
    assert dim.data_size(1) == 20


def test_update_shape_shrink_chunk_spec() -> None:
    """After shrink, ChunkSpec reflects boundary correctly."""
    grid = ChunkGrid.from_sizes((60,), [[10, 20, 30]])
    new_grid = grid.update_shape((45,))
    spec = new_grid[(2,)]
    assert spec is not None
    assert spec.codec_shape == (30,)
    assert spec.shape == (15,)
    assert spec.is_boundary is True


def test_update_shape_parse_chunk_grid_rebinds_extent() -> None:
    """parse_chunk_grid re-binds VaryingDimension extent to array shape."""
    g = ChunkGrid.from_sizes((60,), [[10, 20, 30]])
    g2 = ChunkGrid(
        dimensions=tuple(dim.with_extent(ext) for dim, ext in zip(g.dimensions, (50,), strict=True))
    )
    dim = g2.dimensions[0]
    assert isinstance(dim, VaryingDimension)
    assert dim.extent == 50
    assert dim.data_size(2) == 20


# ---------------------------------------------------------------------------
# Resize rectilinear tests
# ---------------------------------------------------------------------------


async def test_async_resize_grow() -> None:
    """Async resize grow appends new edge chunks and preserves existing data"""
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
    assert _edges(ChunkGrid.from_metadata(arr.metadata), 0) == (10, 20, 20)
    assert _edges(ChunkGrid.from_metadata(arr.metadata), 1) == (20, 20, 20)
    result = await arr.getitem((slice(0, 30), slice(0, 40)))
    np.testing.assert_array_equal(result, data)


async def test_async_resize_shrink() -> None:
    """Async resize shrink truncates data to the new shape"""
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


def test_sync_resize_grow() -> None:
    """Sync resize grow expands the array and preserves existing data"""
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


def test_sync_resize_shrink() -> None:
    """Sync resize shrink truncates the array and returns correct data"""
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


# ---------------------------------------------------------------------------
# Append rectilinear tests
# ---------------------------------------------------------------------------


async def test_append_first_axis() -> None:
    """Appending along axis 0 grows the array and concatenates data correctly"""
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


async def test_append_second_axis() -> None:
    """Appending along axis 1 grows the array and concatenates data correctly"""
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


def test_sync_append() -> None:
    """Sync append grows the array and preserves both initial and appended data"""
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


async def test_multiple_appends() -> None:
    """Multiple sequential appends accumulate data correctly"""
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


async def test_append_with_partial_edge_chunks() -> None:
    """Appending data that creates partial edge chunks preserves all data"""
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


async def test_append_small_data() -> None:
    """Appending a small amount of data smaller than a chunk works correctly"""
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


# ---------------------------------------------------------------------------
# V2 regression tests
# ---------------------------------------------------------------------------


def test_v2_create_and_readback(tmp_path: Path) -> None:
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


def test_v2_chunk_grid_is_regular(tmp_path: Path) -> None:
    """V2 chunk_grid produces a regular ChunkGrid with FixedDimensions."""
    a = zarr.create_array(
        store=tmp_path / "v2.zarr",
        shape=(20, 30),
        chunks=(10, 15),
        dtype="int32",
        zarr_format=2,
    )
    grid = ChunkGrid.from_metadata(a.metadata)
    assert grid.is_regular
    assert grid.chunk_shape == (10, 15)
    assert grid.grid_shape == (2, 2)
    assert all(isinstance(d, FixedDimension) for d in grid.dimensions)


def test_v2_boundary_chunks(tmp_path: Path) -> None:
    """V2 boundary chunks: codec buffer size stays full, data is clipped."""
    a = zarr.create_array(
        store=tmp_path / "v2.zarr",
        shape=(25,),
        chunks=(10,),
        dtype="int32",
        zarr_format=2,
    )
    grid = ChunkGrid.from_metadata(a.metadata)
    assert grid.dimensions[0].nchunks == 3
    assert grid.dimensions[0].chunk_size(2) == 10
    assert grid.dimensions[0].data_size(2) == 5


def test_v2_slicing_with_boundary(tmp_path: Path) -> None:
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


def test_v2_metadata_roundtrip(tmp_path: Path) -> None:
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

    b = zarr.open_array(store=store_path, mode="r")
    assert b.metadata.zarr_format == 2
    assert b.chunks == (2, 2)
    assert ChunkGrid.from_metadata(b.metadata).chunk_shape == (2, 2)
    np.testing.assert_array_equal(b[:], data)


def test_v2_chunk_spec_via_grid(tmp_path: Path) -> None:
    """ChunkSpec from V2 grid has correct slices and codec_shape."""
    a = zarr.create_array(
        store=tmp_path / "v2.zarr",
        shape=(15, 20),
        chunks=(10, 10),
        dtype="int32",
        zarr_format=2,
    )
    grid = ChunkGrid.from_metadata(a.metadata)
    spec = grid[(0, 0)]
    assert spec is not None
    assert spec.shape == (10, 10)
    assert spec.codec_shape == (10, 10)
    spec = grid[(1, 1)]
    assert spec is not None
    assert spec.shape == (5, 10)
    assert spec.codec_shape == (10, 10)


# ---------------------------------------------------------------------------
# ChunkSizes tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("shape", "chunks", "expected"),
    [
        ((100, 80), (30, 40), ((30, 30, 30, 10), (40, 40))),
        ((90, 80), (30, 40), ((30, 30, 30), (40, 40))),
        ((60, 100), [[10, 20, 30], [50, 50]], ((10, 20, 30), (50, 50))),
        ((10,), (10,), ((10,),)),
    ],
    ids=["regular", "regular-exact", "rectilinear", "single-chunk"],
)
def test_chunk_sizes(
    shape: tuple[int, ...], chunks: Any, expected: tuple[tuple[int, ...], ...]
) -> None:
    """chunk_sizes returns the per-dimension tuple of actual data sizes"""
    grid = ChunkGrid.from_sizes(shape, chunks)
    assert grid.chunk_sizes == expected


def test_array_read_chunk_sizes_regular() -> None:
    """Regular array exposes correct read_chunk_sizes and write_chunk_sizes"""
    store = zarr.storage.MemoryStore()
    arr = zarr.create_array(
        store=store, shape=(100, 80), chunks=(30, 40), dtype="i4", zarr_format=3
    )
    assert arr.read_chunk_sizes == ((30, 30, 30, 10), (40, 40))
    assert arr.write_chunk_sizes == ((30, 30, 30, 10), (40, 40))


def test_array_read_chunk_sizes_rectilinear() -> None:
    """Rectilinear array exposes correct read_chunk_sizes and write_chunk_sizes"""
    store = zarr.storage.MemoryStore()
    arr = zarr.create_array(
        store=store, shape=(60, 100), chunks=[[10, 20, 30], [50, 50]], dtype="i4", zarr_format=3
    )
    assert arr.read_chunk_sizes == ((10, 20, 30), (50, 50))
    assert arr.write_chunk_sizes == ((10, 20, 30), (50, 50))


def test_array_sharded_chunk_sizes() -> None:
    """Sharded array read_chunk_sizes reflects inner chunks and write_chunk_sizes reflects shards"""
    store = zarr.storage.MemoryStore()
    arr = zarr.create_array(
        store=store,
        shape=(120, 80),
        chunks=(60, 40),
        shards=(120, 80),
        dtype="i4",
        zarr_format=3,
    )
    assert arr.read_chunk_sizes == ((60, 60), (40, 40))
    assert arr.write_chunk_sizes == ((120,), (80,))


# ---------------------------------------------------------------------------
# Info display test
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
# nchunks tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("shape", "chunks", "expected"),
    [
        ((30,), [[10, 20]], 2),
        ((30, 40), [[10, 20], [15, 25]], 4),
    ],
    ids=["1d", "2d"],
)
def test_nchunks_rectilinear(
    shape: tuple[int, ...], chunks: list[list[int]], expected: int
) -> None:
    """Array.nchunks reports correct total chunk count for rectilinear arrays"""
    store = MemoryStore()
    a = zarr.create_array(store, shape=shape, chunks=chunks, dtype="int32")
    assert a.nchunks == expected


# ---------------------------------------------------------------------------
# iter_chunk_regions test
# ---------------------------------------------------------------------------


def test_iter_chunk_regions_rectilinear() -> None:
    """_iter_chunk_regions should work for rectilinear arrays."""
    from zarr.core.array import _iter_chunk_regions

    store = MemoryStore()
    a = zarr.create_array(store, shape=(30,), chunks=[[10, 20]], dtype="int32")
    regions = list(_iter_chunk_regions(a))
    assert len(regions) == 2
    assert regions[0] == (slice(0, 10, 1),)
    assert regions[1] == (slice(10, 30, 1),)


# ---------------------------------------------------------------------------
# RectilinearChunkGrid metadata object tests (already parametrized)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("json_input", "expected_chunk_shapes"),
    [
        (
            {
                "name": "rectilinear",
                "configuration": {"kind": "inline", "chunk_shapes": [4, 8]},
            },
            (4, 8),
        ),
        (
            {
                "name": "rectilinear",
                "configuration": {"kind": "inline", "chunk_shapes": [[1, 2, 3], [10, 20]]},
            },
            ((1, 2, 3), (10, 20)),
        ),
        (
            {
                "name": "rectilinear",
                "configuration": {"kind": "inline", "chunk_shapes": [[[4, 3]], [10, 20]]},
            },
            ((4, 4, 4), (10, 20)),
        ),
        (
            {
                "name": "rectilinear",
                "configuration": {"kind": "inline", "chunk_shapes": [[[1, 3], 3], [5]]},
            },
            ((1, 1, 1, 3), (5,)),
        ),
        (
            {
                "name": "rectilinear",
                "configuration": {"kind": "inline", "chunk_shapes": [4, [10, 20]]},
            },
            (4, (10, 20)),
        ),
    ],
)
def test_rectilinear_from_dict(
    json_input: dict[str, Any], expected_chunk_shapes: tuple[int | tuple[int, ...], ...]
) -> None:
    """RectilinearChunkGrid.from_dict correctly parses all spec forms."""
    grid = RectilinearChunkGrid.from_dict(json_input)  # type: ignore[arg-type]
    assert grid.chunk_shapes == expected_chunk_shapes


@pytest.mark.parametrize(
    ("chunk_shapes", "expected_json_shapes"),
    [
        ((4, 8), [4, 8]),
        (((4,), (8,)), [[4], [8]]),
        (((10, 20), (5, 5)), [[10, 20], [[5, 2]]]),
        (((4, 4, 4), (10, 20)), [[[4, 3]], [10, 20]]),
        ((4, (10, 20)), [4, [10, 20]]),
    ],
)
def test_rectilinear_to_dict(
    chunk_shapes: tuple[int | tuple[int, ...], ...],
    expected_json_shapes: list[Any],
) -> None:
    """RectilinearChunkGrid.to_dict serializes back to spec-compliant JSON."""
    grid = RectilinearChunkGrid(chunk_shapes=chunk_shapes)
    result = grid.to_dict()
    assert result["name"] == "rectilinear"
    assert result["configuration"]["kind"] == "inline"
    assert list(result["configuration"]["chunk_shapes"]) == expected_json_shapes


@pytest.mark.parametrize(
    "json_input",
    [
        {"name": "rectilinear", "configuration": {"kind": "inline", "chunk_shapes": [4, 8]}},
        {
            "name": "rectilinear",
            "configuration": {"kind": "inline", "chunk_shapes": [[1, 2, 3], [10, 20]]},
        },
        {
            "name": "rectilinear",
            "configuration": {"kind": "inline", "chunk_shapes": [[[4, 3]], [[5, 2]]]},
        },
    ],
)
def test_rectilinear_roundtrip(json_input: dict[str, Any]) -> None:
    """from_dict -> to_dict -> from_dict produces the same grid."""
    grid1 = RectilinearChunkGrid.from_dict(json_input)  # type: ignore[arg-type]
    grid2 = RectilinearChunkGrid.from_dict(grid1.to_dict())
    assert grid1.chunk_shapes == grid2.chunk_shapes


# ---------------------------------------------------------------------------
# Hypothesis property tests
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
    grid = ChunkGrid.from_metadata(z.metadata)

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
