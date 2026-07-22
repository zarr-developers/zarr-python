from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from zarr.core.chunk_grids import ChunkGrid, FixedDimension, VaryingDimension

import zarr_indexing.chunk_resolution as chunk_resolution
from zarr_indexing.chunk_resolution import iter_chunk_transforms, sub_transform_to_selections
from zarr_indexing.domain import IndexDomain
from zarr_indexing.output_map import ArrayMap, ConstantMap, DimensionMap
from zarr_indexing.transform import IndexTransform

if TYPE_CHECKING:
    import pytest


class TestChunkResolutionIdentity:
    def test_single_chunk(self) -> None:
        """Array fits in one chunk."""
        t = IndexTransform.from_shape((10,))
        grid = ChunkGrid(dimensions=(FixedDimension(size=10, extent=10),))
        results = list(iter_chunk_transforms(t, grid))
        assert len(results) == 1
        coords, sub_t, _ = results[0]
        assert coords == (0,)
        assert sub_t.domain.shape == (10,)

    def test_multiple_chunks_1d(self) -> None:
        """1D array spanning 3 chunks."""
        t = IndexTransform.from_shape((30,))
        grid = ChunkGrid(dimensions=(FixedDimension(size=10, extent=30),))
        results = list(iter_chunk_transforms(t, grid))
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
        results = list(iter_chunk_transforms(t, grid))
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
        results = list(iter_chunk_transforms(t, grid))
        assert len(results) == 1
        coords, sub_t, _ = results[0]
        assert coords == (0,)
        assert isinstance(sub_t.output[0], DimensionMap)
        assert sub_t.output[0].offset == 5

    def test_slice_across_chunks(self) -> None:
        """Slice that spans two chunks."""
        t = IndexTransform.from_shape((100,))[8:15]
        grid = ChunkGrid(dimensions=(FixedDimension(size=10, extent=100),))
        results = list(iter_chunk_transforms(t, grid))
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
        results = list(iter_chunk_transforms(t, grid))
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
        results = list(iter_chunk_transforms(t, grid))
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
                direct = list(iter_chunk_transforms(transform, grid))

                with monkeypatch.context() as context:
                    context.setattr(
                        chunk_resolution,
                        "_one_dimensional_correlated_array_map",
                        lambda _transform: None,
                    )
                    general = list(iter_chunk_transforms(transform, grid))

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
        results = list(iter_chunk_transforms(t, grid))

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

        results = list(iter_chunk_transforms(t, grid))

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

        results = list(iter_chunk_transforms(t, grid))

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
        results = list(iter_chunk_transforms(t, grid))

        assert results == []
        assert calls["n"] == 1

    def test_unsorted_vindex_uses_general_resolution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unsorted coordinates continue through the general intersection logic."""
        t = IndexTransform.from_shape((12,)).vindex[np.array([9, 0, 4], dtype=np.intp)]
        grid = ChunkGrid(dimensions=(FixedDimension(size=4, extent=12),))

        calls = _count_intersect_calls(monkeypatch)
        results = list(iter_chunk_transforms(t, grid))

        assert [result[0] for result in results] == [(0,), (1,), (2,)]
        assert calls["n"] == 3

    def test_sorted_oindex_uses_general_resolution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Orthogonal ArrayMaps retain their existing domain-aware resolution."""
        t = IndexTransform.from_shape((12,)).oindex[np.array([0, 4, 9], dtype=np.intp)]
        grid = ChunkGrid(dimensions=(FixedDimension(size=4, extent=12),))

        calls = _count_intersect_calls(monkeypatch)
        results = list(iter_chunk_transforms(t, grid))

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
        results = list(iter_chunk_transforms(t, grid))

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
        results = list(iter_chunk_transforms(t, grid))

        coords = sorted(r[0] for r in results)
        assert coords == [(0, 0), (0, 999), (999, 0), (999, 999)]
        assert calls["n"] == 4

    def test_2d_correlated_vindex_enumerates_per_dim_distinct_chunks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two correlated (vindex) coordinate arrays scatter to 2 diagonal chunks.

        The two points (1, 2) and (3997, 3998) touch chunks (0, 0) and
        (999, 999). Per-dimension distinct touched chunks are {0, 999} on each
        axis, so enumeration intersects the 2x2 = 4 combinations; the two
        off-diagonal combinations are filtered out by `intersect`, leaving 2
        surviving chunks. The key guarantee is that the work is bounded by the
        per-dimension distinct touched chunks (4), not the dense 1e6 grid.
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
        results = list(iter_chunk_transforms(t, grid))

        coords = sorted(r[0] for r in results)
        assert coords == [(0, 0), (999, 999)]
        # 2x2 per-dim-distinct combinations enumerated; 2 survive intersection.
        assert calls["n"] == 4


class TestSubTransformToSelections:
    def test_constant_map(self) -> None:
        """ConstantMap produces int selection + drop axis."""
        t = IndexTransform(
            domain=IndexDomain.from_shape((10,)),
            output=(ConstantMap(offset=5),),
        )
        chunk_sel, out_sel, drop_axes = sub_transform_to_selections(t)
        assert chunk_sel == (5,)
        assert out_sel == ()
        assert drop_axes == ()

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
        assert chunk_sel[0] == 5
        assert chunk_sel[1] == slice(0, 10, 1)
        # drop_axes is empty — integer in chunk_sel naturally drops the dim via numpy
        assert drop_axes == ()


class TestChunkResolutionArrayMapFlavours:
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
        assert list(iter_chunk_transforms(t, grid)) == []

    def test_orthogonal_outer_product_selectors(self) -> None:
        """Two independent arrays produce np.ix_-style (mesh) chunk/out selectors."""
        t = IndexTransform.from_shape((10, 10)).oindex[np.array([1, 3]), np.array([2, 4, 6])]
        grid = ChunkGrid(
            dimensions=(FixedDimension(size=10, extent=10), FixedDimension(size=10, extent=10))
        )
        results = list(iter_chunk_transforms(t, grid))
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
        results = list(iter_chunk_transforms(t, grid))
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
