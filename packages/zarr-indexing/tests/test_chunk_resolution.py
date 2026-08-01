from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from zarr.core.chunk_grids import ChunkGrid, FixedDimension, VaryingDimension

from zarr_indexing import ChunkPlan, ChunkProjection, chunk_resolution, plan_chunks
from zarr_indexing.chunk_resolution import iter_chunk_transforms, sub_transform_to_selections
from zarr_indexing.domain import IndexDomain
from zarr_indexing.grid import dimension_grids_from_chunks
from zarr_indexing.output_map import ArrayMap, ConstantMap, DimensionMap
from zarr_indexing.transform import IndexTransform


def test_basic_plan_is_reiterable_and_projects_both_spaces() -> None:
    """A plan can be revisited without losing either side of each projection."""
    transform = IndexTransform.from_shape((6,))[1:6]
    grids = dimension_grids_from_chunks((3,), (6,))

    plan = plan_chunks(transform, grids)
    first = list(plan)
    second = list(plan.projections())

    assert isinstance(plan, ChunkPlan)
    assert all(isinstance(projection, ChunkProjection) for projection in first)
    assert [projection.chunk_coords for projection in first] == [(0,), (1,)]
    assert [projection.chunk_domain for projection in first] == [
        IndexDomain((0,), (3,)),
        IndexDomain((3,), (6,)),
    ]
    assert [projection.coverage for projection in first] == ["partial", "full"]
    assert first == second
    assert all(
        projection.chunk_transform.domain == projection.cell_transform.domain
        for projection in first
    )


def test_plan_rejects_grid_rank_different_from_transform_output_rank() -> None:
    """A missing storage grid dimension is rejected before iteration."""
    transform = IndexTransform.from_shape((2, 3))

    with pytest.raises(ValueError, match="1 grids for output rank 2"):
        plan_chunks(transform, dimension_grids_from_chunks((2,), (2,)))


class TestChunkResolutionIdentity:
    def test_single_chunk(self) -> None:
        """Array fits in one chunk."""
        t = IndexTransform.from_shape((10,))
        grid = ChunkGrid(dimensions=(FixedDimension(size=10, extent=10),))
        results = list(iter_chunk_transforms(t, grid._dimensions))
        assert len(results) == 1
        coords, sub_t, _ = results[0]
        assert coords == (0,)
        assert sub_t.domain.shape == (10,)

    def test_multiple_chunks_1d(self) -> None:
        """1D array spanning 3 chunks."""
        t = IndexTransform.from_shape((30,))
        grid = ChunkGrid(dimensions=(FixedDimension(size=10, extent=30),))
        results = list(iter_chunk_transforms(t, grid._dimensions))
        assert len(results) == 3
        coords_list = [r[0] for r in results]
        assert (0,) in coords_list
        assert (1,) in coords_list
        assert (2,) in coords_list

    def test_multiple_chunks_2d(self) -> None:
        """2D array spanning 2x3 chunks."""
        t = IndexTransform.from_shape((20, 30))
        grid = ChunkGrid(
            dimensions=(
                FixedDimension(size=10, extent=20),
                FixedDimension(size=10, extent=30),
            )
        )
        results = list(iter_chunk_transforms(t, grid._dimensions))
        assert len(results) == 6
        coords_list = [r[0] for r in results]
        assert (0, 0) in coords_list
        assert (1, 2) in coords_list


class TestChunkResolutionSliced:
    def test_slice_within_chunk(self) -> None:
        """Slice that falls within a single chunk."""
        # Chunk resolution consumes zero-origin transforms: the I/O layer
        # normalizes preserved (user-facing) domains via translate_domain_to
        # before resolving, so mirror that contract here.
        t = IndexTransform.from_shape((100,))[5:8].translate_domain_to((0,))
        grid = ChunkGrid(dimensions=(FixedDimension(size=10, extent=100),))
        results = list(iter_chunk_transforms(t, grid._dimensions))
        assert len(results) == 1
        coords, sub_t, _ = results[0]
        assert coords == (0,)
        assert isinstance(sub_t.output[0], DimensionMap)
        assert sub_t.output[0].offset == 5

    def test_slice_across_chunks(self) -> None:
        """Slice that spans two chunks."""
        t = IndexTransform.from_shape((100,))[8:15]
        grid = ChunkGrid(dimensions=(FixedDimension(size=10, extent=100),))
        results = list(iter_chunk_transforms(t, grid._dimensions))
        assert len(results) == 2
        coords_list = [r[0] for r in results]
        assert (0,) in coords_list
        assert (1,) in coords_list


class TestChunkResolutionConstant:
    def test_integer_index(self) -> None:
        """Integer index produces constant map — single chunk per constant dim."""
        t = IndexTransform.from_shape((100, 100))[25, :]
        grid = ChunkGrid(
            dimensions=(
                FixedDimension(size=10, extent=100),
                FixedDimension(size=10, extent=100),
            )
        )
        results = list(iter_chunk_transforms(t, grid._dimensions))
        assert len(results) == 10
        for coords, _, _ in results:
            assert coords[0] == 2


class TestChunkResolutionArray:
    def test_array_index(self) -> None:
        """Array index map — chunks determined by array values."""
        idx = np.array([5, 15, 25], dtype=np.intp)
        t = IndexTransform(
            domain=IndexDomain.from_shape((3,)),
            output=(ArrayMap(index_array=idx),),
        )
        grid = ChunkGrid(dimensions=(FixedDimension(size=10, extent=30),))
        results = list(iter_chunk_transforms(t, grid._dimensions))
        coords_list = [r[0] for r in results]
        assert (0,) in coords_list
        assert (1,) in coords_list
        assert (2,) in coords_list


class TestChunkResolutionSorted1D:
    def test_matches_general_resolution_for_randomized_sorted_selections(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Direct partitioning matches the original resolver across varied inputs."""
        rng = np.random.default_rng(0)
        grids = (
            ChunkGrid(dimensions=(FixedDimension(size=7, extent=30),)),
            ChunkGrid(dimensions=(VaryingDimension(edges=(3, 4, 8, 5, 10), extent=30),)),
        )

        for grid in grids:
            for _ in range(50):
                idx = np.sort(rng.integers(0, 30, size=int(rng.integers(1, 80)))).astype(np.intp)
                transform = IndexTransform.from_shape((30,)).vindex[idx]
                direct = list(iter_chunk_transforms(transform, grid._dimensions))

                with monkeypatch.context() as context:
                    context.setattr(
                        chunk_resolution,
                        "_one_dimensional_correlated_array_map",
                        lambda _transform: None,
                    )
                    general = list(iter_chunk_transforms(transform, grid._dimensions))

                assert [result[0] for result in direct] == [result[0] for result in general]
                for direct_result, general_result in zip(direct, general, strict=True):
                    _, direct_t, direct_out = direct_result
                    _, general_t, general_out = general_result
                    assert direct_t.domain == general_t.domain

                    direct_chunk_sel, direct_out_sel, direct_drop = sub_transform_to_selections(
                        direct_t, direct_out
                    )
                    general_chunk_sel, general_out_sel, general_drop = sub_transform_to_selections(
                        general_t, general_out
                    )
                    assert direct_drop == general_drop
                    np.testing.assert_array_equal(direct_chunk_sel[0], general_chunk_sel[0])
                    np.testing.assert_array_equal(direct_out_sel[0], general_out_sel[0])

    def test_sorted_vindex_partitions_chunks_without_intersection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sorted vectorized coordinates are sliced directly per touched chunk."""
        idx = np.array([0, 3, 4, 4, 9, 11], dtype=np.intp)
        t = IndexTransform.from_shape((12,)).vindex[idx]
        grid = ChunkGrid(dimensions=(FixedDimension(size=4, extent=12),))

        calls = _count_intersect_calls(monkeypatch)
        results = list(iter_chunk_transforms(t, grid._dimensions))

        assert [result[0] for result in results] == [(0,), (1,), (2,)]
        assert calls["n"] == 0

        expected_chunk_indices = ([0, 3], [0, 0], [1, 3])
        expected_out_indices = ([0, 1], [2, 3], [4, 5])
        for result, expected_chunk, expected_out in zip(
            results, expected_chunk_indices, expected_out_indices, strict=True
        ):
            _, sub_t, out_indices = result
            chunk_sel, out_sel, drop_axes = sub_transform_to_selections(sub_t, out_indices)
            np.testing.assert_array_equal(chunk_sel[0], expected_chunk)
            np.testing.assert_array_equal(out_sel[0], expected_out)
            assert drop_axes == ()

    def test_sorted_array_map_preserves_offset_and_stride(self) -> None:
        """Storage partitioning retains the ArrayMap's offset and stride."""
        t = IndexTransform(
            domain=IndexDomain.from_shape((3,)),
            output=(
                ArrayMap(
                    index_array=np.array([0, 1, 2], dtype=np.intp),
                    offset=1,
                    stride=3,
                ),
            ),
        )
        grid = ChunkGrid(dimensions=(FixedDimension(size=4, extent=8),))

        results = list(iter_chunk_transforms(t, grid._dimensions))

        assert [result[0] for result in results] == [(0,), (1,)]
        expected_chunk_indices = ([1], [0, 3])
        expected_out_indices = ([0], [1, 2])
        for result, expected_chunk, expected_out in zip(
            results, expected_chunk_indices, expected_out_indices, strict=True
        ):
            _, sub_t, out_indices = result
            chunk_sel, out_sel, _ = sub_transform_to_selections(sub_t, out_indices)
            np.testing.assert_array_equal(chunk_sel[0], expected_chunk)
            np.testing.assert_array_equal(out_sel[0], expected_out)

    def test_sorted_vindex_with_varying_chunks(self) -> None:
        """Touched-boundary searches also support a non-uniform 1-D grid."""
        idx = np.array([0, 1, 2, 3, 5, 9], dtype=np.intp)
        t = IndexTransform.from_shape((10,)).vindex[idx]
        grid = ChunkGrid(dimensions=(VaryingDimension(edges=(2, 3, 5), extent=10),))

        results = list(iter_chunk_transforms(t, grid._dimensions))

        assert [result[0] for result in results] == [(0,), (1,), (2,)]
        expected_chunk_indices = ([0, 1], [0, 1], [0, 4])
        for result, expected_chunk in zip(results, expected_chunk_indices, strict=True):
            _, sub_t, out_indices = result
            chunk_sel, _, _ = sub_transform_to_selections(sub_t, out_indices)
            np.testing.assert_array_equal(chunk_sel[0], expected_chunk)

    def test_sorted_vindex_with_zero_sized_dimension_uses_general_resolution(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A zero-sized grid cannot be partitioned by touched boundaries."""
        t = IndexTransform.from_shape((10,)).vindex[np.array([1], dtype=np.intp)]
        grid = ChunkGrid(dimensions=(FixedDimension(size=0, extent=10),))

        calls = _count_intersect_calls(monkeypatch)
        results = list(iter_chunk_transforms(t, grid._dimensions))

        assert results == []
        assert calls["n"] == 1

    def test_unsorted_vindex_uses_general_resolution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unsorted coordinates continue through the general intersection logic."""
        t = IndexTransform.from_shape((12,)).vindex[np.array([9, 0, 4], dtype=np.intp)]
        grid = ChunkGrid(dimensions=(FixedDimension(size=4, extent=12),))

        calls = _count_intersect_calls(monkeypatch)
        results = list(iter_chunk_transforms(t, grid._dimensions))

        assert [result[0] for result in results] == [(0,), (1,), (2,)]
        assert calls["n"] == 3

    def test_sorted_oindex_uses_general_resolution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Orthogonal ArrayMaps retain their existing domain-aware resolution."""
        t = IndexTransform.from_shape((12,)).oindex[np.array([0, 4, 9], dtype=np.intp)]
        grid = ChunkGrid(dimensions=(FixedDimension(size=4, extent=12),))

        calls = _count_intersect_calls(monkeypatch)
        results = list(iter_chunk_transforms(t, grid._dimensions))

        assert [result[0] for result in results] == [(0,), (1,), (2,)]
        assert calls["n"] == 3


def _count_intersect_calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Wrap `IndexTransform.intersect` with a call counter.

    Returns a mutable dict whose `"n"` entry is the number of times
    `intersect` is invoked. Used to assert that candidate-chunk enumeration is
    proportional to the *touched* chunks, not the dense bounding box between the
    min and max touched chunk.
    """
    calls = {"n": 0}
    original = IndexTransform.intersect

    def counting(self: IndexTransform, output_domain: IndexDomain) -> object:
        calls["n"] += 1
        return original(self, output_domain)

    monkeypatch.setattr(IndexTransform, "intersect", counting)
    return calls


class TestChunkResolutionTouchedOnly:
    """`iter_chunk_transforms` must enumerate only the chunks a fancy selection
    actually touches — never the dense `range(min_chunk, max_chunk + 1)` bounding
    box. These guard against a regression to bounding-box enumeration, whose cost
    scales with grid size rather than with the number of selected coordinates.
    """

    def test_1d_sparse_vindex_enumerates_only_touched_chunks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two far-apart coordinates on a 1000-chunk grid touch exactly 2 chunks.

        A dense bounding-box enumeration would intersect ~1000 candidate chunks;
        touched-only enumeration intersects exactly 2.
        """
        # 4000 elements, chunk size 4 -> 1000 chunks. coords 1 and 3997 land in
        # chunk 0 and chunk 999 respectively (998 empty chunks between them).
        grid = ChunkGrid(dimensions=(FixedDimension(size=4, extent=4000),))
        t = IndexTransform.from_shape((4000,)).vindex[np.array([1, 3997], dtype=np.intp)]

        calls = _count_intersect_calls(monkeypatch)
        results = list(iter_chunk_transforms(t, grid._dimensions))

        coords = sorted(r[0] for r in results)
        assert coords == [(0,), (999,)]
        # Sorted 1-D coordinates are partitioned directly, without intersecting
        # either the touched chunks or the 998 empty chunks between them.
        assert calls["n"] == 0

    def test_2d_orthogonal_enumerates_only_touched_chunks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Orthogonal outer product of two 2-coordinate arrays touches 2x2 chunks.

        Per-dimension distinct touched chunks: {0, 999} on each axis. The outer
        product is 2*2 = 4 candidate chunks (all survive), versus ~1e6 for a
        dense 1000x1000 bounding box.
        """
        grid = ChunkGrid(
            dimensions=(
                FixedDimension(size=4, extent=4000),
                FixedDimension(size=4, extent=4000),
            )
        )
        t = IndexTransform.from_shape((4000, 4000)).oindex[
            np.array([1, 3997], dtype=np.intp), np.array([2, 3998], dtype=np.intp)
        ]

        calls = _count_intersect_calls(monkeypatch)
        results = list(iter_chunk_transforms(t, grid._dimensions))

        coords = sorted(r[0] for r in results)
        assert coords == [(0, 0), (0, 999), (999, 0), (999, 999)]
        assert calls["n"] == 4

    def test_2d_correlated_vindex_enumerates_joint_touched_chunks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two correlated (vindex) coordinate arrays scatter to 2 diagonal chunks.

        The two points (1, 2) and (3997, 3998) touch chunks (0, 0) and
        (999, 999). Correlated coordinate arrays are grouped *jointly*, so
        enumeration intersects exactly the 2 touched chunks — never the 2x2
        cartesian product of per-dimension distinct chunks, and never the dense
        1e6 grid.
        """
        grid = ChunkGrid(
            dimensions=(
                FixedDimension(size=4, extent=4000),
                FixedDimension(size=4, extent=4000),
            )
        )
        t = IndexTransform.from_shape((4000, 4000)).vindex[
            np.array([1, 3997], dtype=np.intp), np.array([2, 3998], dtype=np.intp)
        ]

        calls = _count_intersect_calls(monkeypatch)
        results = list(iter_chunk_transforms(t, grid._dimensions))

        coords = sorted(r[0] for r in results)
        assert coords == [(0, 0), (999, 999)]
        assert calls["n"] == 2

    def test_2d_correlated_vindex_diagonal_is_linear_in_points(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A diagonal of P correlated points touches P chunks with O(P) intersections.

        Enumerating the cartesian product of per-dimension distinct chunk sets
        would cost P**2 intersections (2500 here) — quadratic in the number of
        selected points for the scattered selections of zarr-python gh-4174.
        Joint grouping keeps resolution work proportional to the touched chunks.
        """
        p = 50
        grid = ChunkGrid(
            dimensions=(
                FixedDimension(size=4, extent=4000),
                FixedDimension(size=4, extent=4000),
            )
        )
        # point i lands in chunk (2i, 2i): all per-dimension chunks distinct
        coords_1d = np.arange(p, dtype=np.intp) * 8
        t = IndexTransform.from_shape((4000, 4000)).vindex[coords_1d, coords_1d]

        calls = _count_intersect_calls(monkeypatch)
        results = list(iter_chunk_transforms(t, grid._dimensions))

        assert sorted(r[0] for r in results) == [(2 * i, 2 * i) for i in range(p)]
        assert calls["n"] == p


class TestSubTransformToSelections:
    def test_constant_map(self) -> None:
        """ConstantMap reads one cell through a length-one slice, and says so.

        Not the bare integer this used to emit. NumPy counts an integer among
        the *advanced* indices whenever the tuple also holds an index array, and
        moves the broadcast axis to the front when a slice separates them, while
        `out_selection` is built positionally — so the two sides described
        transposed blocks. A slice is basic wherever it sits, and the size-one
        axis it leaves behind is named in `drop_axes` for the caller to squeeze.

        The domain dimension here is referenced by no output map, so its
        out-selection cannot come from a map: it is the whole axis, taken from
        the domain. One entry per domain dimension is what `out[out_sel] = value`
        needs — an empty tuple would leave the buffer's rank unaddressed and
        place a lower-rank part against the leading axes.
        """
        t = IndexTransform(
            domain=IndexDomain.from_shape((10,)),
            output=(ConstantMap(offset=5),),
        )
        chunk_sel, out_sel, drop_axes = sub_transform_to_selections(t)
        assert chunk_sel == (slice(5, 6),)
        assert out_sel == (slice(0, 10),)
        assert drop_axes == (0,)

    def test_dimension_map_stride_1(self) -> None:
        """DimensionMap with stride=1 produces contiguous slice."""
        t = IndexTransform(
            domain=IndexDomain.from_shape((10,)),
            output=(DimensionMap(input_dimension=0, offset=3, stride=1),),
        )
        chunk_sel, out_sel, drop_axes = sub_transform_to_selections(t)
        assert chunk_sel == (slice(3, 13, 1),)
        assert out_sel == (slice(0, 10),)
        assert drop_axes == ()

    def test_dimension_map_strided(self) -> None:
        """DimensionMap with stride>1 produces strided slice."""
        t = IndexTransform(
            domain=IndexDomain.from_shape((5,)),
            output=(DimensionMap(input_dimension=0, offset=2, stride=3),),
        )
        chunk_sel, out_sel, drop_axes = sub_transform_to_selections(t)
        assert chunk_sel == (slice(2, 17, 3),)
        assert out_sel == (slice(0, 5),)
        assert drop_axes == ()

    def test_array_map(self) -> None:
        """ArrayMap produces integer array selection."""
        arr = np.array([1, 5, 9], dtype=np.intp)
        t = IndexTransform(
            domain=IndexDomain.from_shape((3,)),
            output=(ArrayMap(index_array=arr, offset=0, stride=1),),
        )
        chunk_sel, out_sel, drop_axes = sub_transform_to_selections(t)
        assert isinstance(chunk_sel[0], np.ndarray)
        np.testing.assert_array_equal(chunk_sel[0], arr)
        # Without chunk_mask, out_sel falls back to domain-based slices
        assert out_sel == (slice(0, 3),)
        assert drop_axes == ()

    def test_array_map_with_offset_stride(self) -> None:
        """ArrayMap with offset and stride computes storage coords."""
        arr = np.array([0, 1, 2], dtype=np.intp)
        t = IndexTransform(
            domain=IndexDomain.from_shape((3,)),
            output=(ArrayMap(index_array=arr, offset=10, stride=5),),
        )
        chunk_sel, _out_sel, drop_axes = sub_transform_to_selections(t)
        assert isinstance(chunk_sel[0], np.ndarray)
        np.testing.assert_array_equal(chunk_sel[0], np.array([10, 15, 20]))
        assert drop_axes == ()

    def test_dimension_map_reversed(self) -> None:
        """A negative stride walks the axis backwards, stopping off the front.

        `slice(0, 8, -1)` — the endpoints swapped while the step stays negative —
        selects nothing; the axis reversed is `slice(7, None, -1)`.
        """
        t = IndexTransform.identity(IndexDomain.from_shape((8,)))[::-1].translate_domain_to((0,))
        chunk_sel, out_sel, drop_axes = sub_transform_to_selections(t)
        assert chunk_sel == (slice(7, None, -1),)
        assert out_sel == (slice(0, 8),)
        assert drop_axes == ()
        np.testing.assert_array_equal(np.arange(8)[chunk_sel[0]], np.arange(8)[::-1])

    def test_dimension_map_reversed_bounded(self) -> None:
        """A reversed walk that stops short keeps an ordinary integer stop."""
        t = IndexTransform.identity(IndexDomain.from_shape((8,)))[6:2:-1].translate_domain_to((0,))
        chunk_sel, _out_sel, _drop = sub_transform_to_selections(t)
        np.testing.assert_array_equal(np.arange(8)[chunk_sel[0]], np.arange(8)[6:2:-1])

    def test_dimension_map_reversed_strided(self) -> None:
        """The same, with a stride whose walk does not land on the origin."""
        t = IndexTransform.identity(IndexDomain.from_shape((9,)))[::-3].translate_domain_to((0,))
        chunk_sel, _out_sel, _drop = sub_transform_to_selections(t)
        np.testing.assert_array_equal(np.arange(9)[chunk_sel[0]], np.arange(9)[::-3])

    def test_dimension_map_slice_reads_the_coordinates_the_map_names(self) -> None:
        """Over every offset/stride/extent in range, the emitted slice picks out
        exactly `offset + stride * i` for `i` in the input range."""
        extent = 12
        cells = np.arange(extent)
        for stride in (-4, -3, -2, -1, 1, 2, 3, 4):
            for size in range(extent + 1):
                span = 0 if size == 0 else abs(stride) * (size - 1)
                offsets = range(extent - span) if span < extent else ()
                for offset in offsets:
                    start = offset if stride > 0 else offset + span
                    t = IndexTransform(
                        domain=IndexDomain.from_shape((size,)),
                        output=(DimensionMap(input_dimension=0, offset=start, stride=stride),),
                    )
                    chunk_sel, _out_sel, _drop = sub_transform_to_selections(t)
                    expected = np.array([start + stride * i for i in range(size)])
                    np.testing.assert_array_equal(
                        cells[chunk_sel[0]], expected, err_msg=f"{stride=} {size=} {start=}"
                    )

    def test_correlated_residual_dimension_map_reversed(self) -> None:
        """The correlated branch builds the same reversed slice for its residual dim."""
        t = IndexTransform.from_shape((4, 3, 5)).vindex[np.array([1, 3]), np.array([2, 0])]
        reversed_slice = t[:, ::-1].translate_domain_to((0, 0))
        chunk_sel, _out_sel, _drop = sub_transform_to_selections(reversed_slice)
        assert chunk_sel[2] == slice(4, None, -1)
        np.testing.assert_array_equal(np.arange(5)[chunk_sel[2]], np.arange(5)[::-1])

    def test_mixed_maps_2d(self) -> None:
        """Mix of ConstantMap and DimensionMap."""
        t = IndexTransform(
            domain=IndexDomain.from_shape((10,)),
            output=(
                ConstantMap(offset=5),
                DimensionMap(input_dimension=0, offset=0, stride=1),
            ),
        )
        chunk_sel, _out_sel, drop_axes = sub_transform_to_selections(t)
        assert chunk_sel[0] == slice(5, 6)
        assert chunk_sel[1] == slice(0, 10, 1)
        # The constant's axis survives the read at size one and is squeezed by
        # the caller; letting NumPy drop it through an integer index put the
        # constant in the advanced-index group, where it reordered the block.
        assert drop_axes == (0,)


class TestChunkResolutionArrayMapFlavors:
    """Chunk resolution must yield outer-product (np.ix_) selectors for
    orthogonal ArrayMaps and shared flat-scatter selectors for correlated ones,
    and must return early for empty fancy selections."""

    def test_empty_array_selection_yields_nothing(self) -> None:
        """An empty ArrayMap selection produces no chunk transforms (no crash)."""
        t = IndexTransform(
            domain=IndexDomain.from_shape((0,)),
            output=(ArrayMap(index_array=np.array([], dtype=np.intp)),),
        )
        grid = ChunkGrid(dimensions=(FixedDimension(size=3, extent=10),))
        assert list(iter_chunk_transforms(t, grid._dimensions)) == []

    def test_orthogonal_outer_product_selectors(self) -> None:
        """Two independent arrays produce np.ix_-style (mesh) chunk/out selectors."""
        t = IndexTransform.from_shape((10, 10)).oindex[np.array([1, 3]), np.array([2, 4, 6])]
        grid = ChunkGrid(
            dimensions=(FixedDimension(size=10, extent=10), FixedDimension(size=10, extent=10))
        )
        results = list(iter_chunk_transforms(t, grid._dimensions))
        assert len(results) == 1
        _coords, sub_t, out_indices = results[0]
        chunk_sel, out_sel, drop_axes = sub_transform_to_selections(sub_t, out_indices)
        # np.ix_ produces one 2-D open-mesh selector per axis, for both sides.
        assert len(chunk_sel) == 2
        assert len(out_sel) == 2
        assert isinstance(chunk_sel[0], np.ndarray)
        assert isinstance(chunk_sel[1], np.ndarray)
        assert chunk_sel[0].shape == (2, 1)
        assert chunk_sel[1].shape == (1, 3)
        assert drop_axes == ()

    def test_correlated_scatter_with_residual_slice(self) -> None:
        """Correlated arrays + a residual slice dim scatter through a single flat
        index whose shape matches the (points, slice) block read from the chunk."""
        t = IndexTransform.from_shape((4, 3, 5)).vindex[np.array([1, 3]), np.array([2, 0])]
        grid = ChunkGrid(
            dimensions=(
                FixedDimension(size=4, extent=4),
                FixedDimension(size=3, extent=3),
                FixedDimension(size=5, extent=5),
            )
        )
        # One chunk holds everything: both points survive, slice dim spans [0,5).
        results = list(iter_chunk_transforms(t, grid._dimensions))
        assert len(results) == 1
        _coords, sub_t, out_indices = results[0]
        chunk_sel, out_sel, _drop = sub_transform_to_selections(sub_t, out_indices)
        # Chunk side: flat coordinate arrays for the two correlated dims plus a
        # slice for the residual dim.
        assert len(chunk_sel) == 3
        np.testing.assert_array_equal(np.asarray(chunk_sel[0]), [1, 3])
        np.testing.assert_array_equal(np.asarray(chunk_sel[1]), [2, 0])
        assert chunk_sel[2] == slice(0, 5, 1)
        # Output side: a single flat scatter index of shape (points, slice) = (2, 5).
        assert len(out_sel) == 1
        assert np.asarray(out_sel[0]).shape == (2, 5)


class TestBridgeBlockOrder:
    """`out[out_sel] = chunk[chunk_sel]` must address the same block on both sides.

    Nothing exercised this pairing before: `LazyArray` resolves a part by
    lowering its own transform and only ever reads `out_selection`, so a
    `chunk_selection` describing a differently-ordered block went unnoticed by
    the whole suite while corrupting any consumer that followed the contract.
    """

    @staticmethod
    def _assemble(
        data: np.ndarray[Any, Any], chunks: tuple[int, ...], transform: IndexTransform
    ) -> np.ndarray[Any, Any]:
        grids = dimension_grids_from_chunks(chunks, data.shape)
        out = np.zeros(transform.domain.shape, dtype=data.dtype)
        for coords, local, out_indices in iter_chunk_transforms(transform, grids):
            chunk_sel, out_sel, drop_axes = sub_transform_to_selections(local, out_indices)
            block_slice = tuple(
                slice(c * size, c * size + grid.chunk_size(c))
                for c, size, grid in zip(coords, chunks, grids, strict=True)
            )
            block = data[block_slice][chunk_sel]
            if len(drop_axes) > 0:
                block = np.squeeze(block, axis=drop_axes)
            if len(out_sel) == 1 and isinstance(out_sel[0], np.ndarray):
                out.reshape(-1)[out_sel[0]] = block
            else:
                out[out_sel] = block
        return out

    @pytest.mark.parametrize("chunks", [(2, 3, 3), (1, 2, 2)], ids=["one-chunk", "many-chunks"])
    def test_a_constant_before_an_array_keeps_the_blocks_axis_order(
        self, chunks: tuple[int, int, int]
    ) -> None:
        """`oindex[int, :, array]` — a constant, a slice, then an index array.

        NumPy folds an integer into the advanced-index group, and moves the
        broadcast axis to the front when a slice separates the two. Emitting the
        constant as an integer therefore transposed the chunk block against an
        `out_selection` built positionally.
        """
        data = np.arange(18).reshape(2, 3, 3)
        transform = (
            IndexTransform.from_shape(data.shape)[1]
            .translate_domain_to((0, 0))
            .oindex[:, np.array([0, 1, 2])]
        )
        np.testing.assert_array_equal(
            self._assemble(data, chunks, transform), data[1][:, [0, 1, 2]]
        )

    def test_a_residual_slice_before_the_points_keeps_the_scatter_in_step(self) -> None:
        """A correlated read whose residual dimension precedes its coordinates.

        The scatter enumerates points-major, but NumPy leaves the points axis
        where the coordinate arrays sit, so the block arrived slice-major. Equal
        extents transposed the values silently; unequal ones raised.
        """
        data = np.arange(12).reshape(3, 4)
        transform = IndexTransform.from_shape(data.shape).vindex[..., np.array([0, 1, 2])]
        np.testing.assert_array_equal(self._assemble(data, (3, 4), transform), data[..., [0, 1, 2]])

    def test_a_residual_slice_before_the_points_of_a_different_length(self) -> None:
        """The same shape, with extents that cannot coincide — this one raised."""
        data = np.arange(8).reshape(2, 4)
        transform = IndexTransform.from_shape(data.shape).vindex[..., np.array([0, 1, 2])]
        np.testing.assert_array_equal(self._assemble(data, (2, 4), transform), data[..., [0, 1, 2]])
