from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

import zarr_indexing
from zarr_indexing import (
    ChunkGrid,
    ChunkPlan,
    ChunkProjection,
    FixedDimension,
    VaryingDimension,
    chunk_resolution,
    plan_chunks,
)
from zarr_indexing.domain import IndexDomain
from zarr_indexing.grid import dimension_grids_from_chunks
from zarr_indexing.output_map import ConstantMap, DimensionMap
from zarr_indexing.transform import IndexTransform


def _storage_of(transform: IndexTransform, point: tuple[int, ...]) -> tuple[int, ...]:
    """Evaluate the three map forms at one point, independently of planning."""
    result: list[int] = []
    for output_map in transform.output:
        if isinstance(output_map, ConstantMap):
            result.append(output_map.offset)
        elif isinstance(output_map, DimensionMap):
            result.append(output_map.offset + output_map.stride * point[output_map.input_dimension])
        else:
            index = tuple(
                0
                if output_map.index_array.shape[axis] == 1
                else point[axis] - transform.domain.inclusive_min[axis]
                for axis in range(output_map.index_array.ndim)
            )
            result.append(
                output_map.offset + output_map.stride * int(output_map.index_array[index])
            )
    return tuple(result)


def _points(domain: IndexDomain) -> list[tuple[int, ...]]:
    """Enumerate a small finite domain in its own coordinates."""
    return [
        tuple(
            coordinate + origin
            for coordinate, origin in zip(position, domain.inclusive_min, strict=True)
        )
        for position in np.ndindex(*domain.shape)
    ]


def _count_intersect_calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Count real intersections to protect touched-only candidate enumeration."""
    calls = {"n": 0}
    original = IndexTransform.intersect

    def counting(self: IndexTransform, output_domain: IndexDomain) -> object:
        calls["n"] += 1
        return original(self, output_domain)

    monkeypatch.setattr(IndexTransform, "intersect", counting)
    return calls


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
    assert all(projection.chunk_transform.domain.origin == (0,) for projection in first)


def test_projection_requires_one_shared_synthetic_domain() -> None:
    """Paired transforms with different cell domains are rejected as incoherent."""
    with pytest.raises(ValueError, match="must share an input domain"):
        ChunkProjection(
            chunk_coords=(0,),
            chunk_domain=IndexDomain.from_shape((3,)),
            chunk_transform=IndexTransform.identity(IndexDomain.from_shape((2,))),
            cell_transform=IndexTransform.identity(IndexDomain.from_shape((1,))),
            coverage="partial",
        )


def test_projection_plan_is_the_only_public_chunk_resolution_surface() -> None:
    """The greenfield API does not retain tuple or NumPy-selector bridges."""
    assert {"ChunkCoverage", "ChunkPlan", "ChunkProjection", "plan_chunks"} <= set(
        zarr_indexing.__all__
    )
    assert "iter_chunk_transforms" not in zarr_indexing.__all__
    assert "sub_transform_to_selections" not in zarr_indexing.__all__


def test_plan_rejects_grid_rank_different_from_transform_output_rank() -> None:
    """A missing storage grid dimension is rejected before iteration."""
    transform = IndexTransform.from_shape((2, 3))

    with pytest.raises(ValueError, match="1 grids for output rank 2"):
        plan_chunks(transform, dimension_grids_from_chunks((2,), (2,)))


@pytest.mark.parametrize(
    ("transform", "expected"),
    [
        (IndexTransform.from_shape((5,)), ["full", "full"]),
        (IndexTransform.from_shape((5,))[::-1], ["full", "full"]),
        (IndexTransform.from_shape((5,))[::2], ["partial", "partial"]),
        (IndexTransform.from_shape((5,))[2], ["partial"]),
        (
            IndexTransform.from_shape((5,)).oindex[np.array([0, 1, 2, 3, 4])],
            ["unknown", "unknown"],
        ),
    ],
    ids=["clipped-edge", "reverse", "strided", "scalar", "fancy-is-conservative"],
)
def test_coverage_classification(transform: IndexTransform, expected: list[str]) -> None:
    """Coverage is exact for affine requests and conservative for gathers."""
    grids = dimension_grids_from_chunks((3,), (5,))

    assert [projection.coverage for projection in plan_chunks(transform, grids)] == expected


def test_repeated_input_dependency_is_not_full_coverage() -> None:
    transform = IndexTransform(
        domain=IndexDomain.from_shape((2,)),
        output=(DimensionMap(input_dimension=0), DimensionMap(input_dimension=0)),
    )
    grids = dimension_grids_from_chunks((2, 2), (2, 2))

    assert [projection.coverage for projection in plan_chunks(transform, grids)] == ["partial"]


def test_unused_input_axis_is_not_full_coverage() -> None:
    transform = IndexTransform(
        domain=IndexDomain.from_shape((2, 2)),
        output=(DimensionMap(input_dimension=0),),
    )
    grids = dimension_grids_from_chunks((2,), (2,))

    assert [projection.coverage for projection in plan_chunks(transform, grids)] == ["partial"]


@pytest.mark.parametrize(
    ("transform", "grids"),
    [
        (
            IndexTransform.from_shape((2, 3)),
            dimension_grids_from_chunks((2, 3), (2, 3)),
        ),
        (
            IndexTransform(
                domain=IndexDomain.from_shape((2, 3)),
                output=(DimensionMap(input_dimension=1), DimensionMap(input_dimension=0)),
            ),
            dimension_grids_from_chunks((3, 2), (3, 2)),
        ),
        (
            IndexTransform.from_shape((2, 3))[::-1, ::-1],
            dimension_grids_from_chunks((2, 3), (2, 3)),
        ),
        (
            IndexTransform(
                domain=IndexDomain((4, 7), (6, 10)),
                output=(
                    DimensionMap(input_dimension=0, offset=-4),
                    DimensionMap(input_dimension=1, offset=-7),
                ),
            ),
            dimension_grids_from_chunks((2, 3), (2, 3)),
        ),
    ],
    ids=["identity", "axis-permutation", "reversal", "translated-unit-affine"],
)
def test_bijective_unit_affine_transforms_retain_full_coverage(
    transform: IndexTransform, grids: tuple[Any, ...]
) -> None:
    assert [projection.coverage for projection in plan_chunks(transform, grids)] == ["full"]


def test_rank_zero_transform_has_full_coverage() -> None:
    transform = IndexTransform.identity(IndexDomain((), ()))

    assert [projection.coverage for projection in plan_chunks(transform, ())] == ["full"]


@pytest.mark.parametrize(
    ("transform", "grids", "expected_coords"),
    [
        (
            IndexTransform.from_shape((30,)),
            dimension_grids_from_chunks((10,), (30,)),
            [(0,), (1,), (2,)],
        ),
        (
            IndexTransform.from_shape((20, 30)),
            dimension_grids_from_chunks((10, 10), (20, 30)),
            [(i, j) for i in range(2) for j in range(3)],
        ),
        (
            IndexTransform.from_shape((100, 100))[25, :],
            dimension_grids_from_chunks((10, 10), (100, 100)),
            [(2, j) for j in range(10)],
        ),
        (
            IndexTransform.from_shape((100,))[8:15],
            dimension_grids_from_chunks((10,), (100,)),
            [(0,), (1,)],
        ),
    ],
    ids=["one-dimensional", "two-dimensional", "constant-map", "slice"],
)
def test_affine_plans_touch_the_expected_chunks(
    transform: IndexTransform,
    grids: tuple[Any, ...],
    expected_coords: list[tuple[int, ...]],
) -> None:
    """Identity, constant, and sliced transforms enumerate literal grid cells."""
    assert [projection.chunk_coords for projection in plan_chunks(transform, grids)] == (
        expected_coords
    )


@pytest.mark.parametrize(
    ("transform", "grids"),
    [
        (
            IndexTransform.from_shape((6,)).oindex[np.array([4, 0, 4, 2])],
            dimension_grids_from_chunks((3,), (6,)),
        ),
        (
            IndexTransform.from_shape((4, 5)).oindex[np.array([3, 0]), np.array([4, 1, 1])],
            dimension_grids_from_chunks(((1, 3), (2, 3)), (4, 5)),
        ),
        (
            IndexTransform.from_shape((2, 4, 5)).vindex[
                ..., np.array([3, 0, 3]), np.array([4, 1, 1])
            ],
            dimension_grids_from_chunks((1, 2, 3), (2, 4, 5)),
        ),
    ],
    ids=["repeated-oindex", "irregular-oindex", "vindex-with-residual"],
)
def test_projection_invariants_for_fancy_selections(
    transform: IndexTransform, grids: tuple[Any, ...]
) -> None:
    """Both transforms agree pointwise and cell ranges tile request space once."""
    plan = plan_chunks(transform, grids)
    request_points: list[tuple[int, ...]] = []

    for projection in plan:
        assert projection.coverage == "unknown"
        assert projection.chunk_transform.domain == projection.cell_transform.domain
        for cell_point in _points(projection.cell_transform.domain):
            request_point = _storage_of(projection.cell_transform, cell_point)
            chunk_point = _storage_of(projection.chunk_transform, cell_point)
            storage_point = _storage_of(plan.transform, request_point)
            chunk_origin = projection.chunk_domain.inclusive_min
            assert chunk_point == tuple(
                value - origin for value, origin in zip(storage_point, chunk_origin, strict=True)
            )
            assert all(
                0 <= value < extent
                for value, extent in zip(chunk_point, projection.chunk_domain.shape, strict=True)
            )
            request_points.append(request_point)

    assert sorted(request_points) == sorted(_points(transform.domain))


@given(
    origin=st.integers(min_value=-4, max_value=4),
    extent=st.integers(min_value=0, max_value=8),
    stride=st.integers(min_value=-3, max_value=3),
)
def test_affine_projection_pairs_reconstruct_independent_source_coordinates(
    origin: int, extent: int, stride: int
) -> None:
    """Bounded literal-domain examples preserve every request/storage pair."""
    anchor = extent - 1 if stride < 0 else 0
    offset = anchor - stride * origin
    expected_pairs = [
        ((coordinate,), (source_coordinate,))
        for coordinate in range(origin, origin + extent)
        if 0 <= (source_coordinate := offset + stride * coordinate) < extent
    ]
    assume(expected_pairs)

    unrestricted = IndexTransform(
        domain=IndexDomain((origin,), (origin + extent,)),
        output=(DimensionMap(input_dimension=0, offset=offset, stride=stride),),
    )
    intersection = unrestricted.intersect(IndexDomain.from_shape((extent,)))
    assume(intersection is not None)
    transform, _ = intersection
    grids = dimension_grids_from_chunks((min(3, extent),), (extent,))

    reconstructed_pairs = [
        (
            _storage_of(projection.cell_transform, cell_coordinate),
            tuple(
                local_coordinate + chunk_origin
                for local_coordinate, chunk_origin in zip(
                    _storage_of(projection.chunk_transform, cell_coordinate),
                    projection.chunk_domain.inclusive_min,
                    strict=True,
                )
            ),
        )
        for projection in plan_chunks(transform, grids)
        for cell_coordinate in _points(projection.cell_transform.domain)
    ]

    assert sorted(reconstructed_pairs) == sorted(expected_pairs)


def test_correlated_projection_preserves_nonzero_request_coordinates() -> None:
    base = IndexTransform.identity(IndexDomain((2, 5), (4, 8)))
    transform = base.vindex[np.array([2, 3], dtype=np.intp), :]
    grids = dimension_grids_from_chunks((2, 4), (4, 8))

    points = [
        transform.apply(projection.cell_transform.apply(cell))
        for projection in plan_chunks(transform, grids)
        for cell in _points(projection.cell_transform.domain)
    ]

    assert sorted(points) == [(2, 5), (2, 6), (2, 7), (3, 5), (3, 6), (3, 7)]


def test_correlated_projection_preserves_translated_advanced_axis_coordinates() -> None:
    transform = (
        IndexTransform.from_shape((4,))
        .vindex[np.array([0, 3], dtype=np.intp)]
        .translate_domain_by((5,))
    )
    grids = dimension_grids_from_chunks((2,), (4,))

    request_points = [
        projection.cell_transform.apply(cell)
        for projection in plan_chunks(transform, grids)
        for cell in _points(projection.cell_transform.domain)
    ]

    assert sorted(request_points) == [(5,), (6,)]


def test_empty_request_has_no_projections() -> None:
    """An empty fancy selection does not fabricate a touched chunk."""
    transform = IndexTransform.from_shape((10,)).oindex[np.array([], dtype=np.intp)]
    grids = dimension_grids_from_chunks((3,), (10,))

    assert list(plan_chunks(transform, grids)) == []


class TestSortedOneDimensionalPlan:
    def test_matches_general_resolution_for_randomized_selections(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The direct sorted path has the same paired transforms as intersection."""
        rng = np.random.default_rng(0)
        grids = (
            ChunkGrid(dimensions=(FixedDimension(size=7, extent=30),)),
            ChunkGrid(dimensions=(VaryingDimension(edges=(3, 4, 8, 5, 10), extent=30),)),
        )
        for grid in grids:
            for _ in range(50):
                indices = np.sort(rng.integers(0, 30, size=int(rng.integers(1, 80)))).astype(
                    np.intp
                )
                transform = IndexTransform.from_shape((30,)).vindex[indices]
                direct = list(plan_chunks(transform, grid.dimensions))
                with monkeypatch.context() as context:
                    context.setattr(
                        chunk_resolution,
                        "_one_dimensional_correlated_array_map",
                        lambda _transform: None,
                    )
                    general = list(plan_chunks(transform, grid.dimensions))
                assert direct == general

    def test_sorted_coordinates_bypass_intersection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Sorted coordinates partition directly at touched chunk boundaries."""
        transform = IndexTransform.from_shape((12,)).vindex[
            np.array([0, 3, 4, 4, 9, 11], dtype=np.intp)
        ]
        grid = ChunkGrid(dimensions=(FixedDimension(size=4, extent=12),))
        calls = _count_intersect_calls(monkeypatch)

        projections = list(plan_chunks(transform, grid.dimensions))

        assert [projection.chunk_coords for projection in projections] == [(0,), (1,), (2,)]
        assert calls["n"] == 0
        assert [
            [
                _storage_of(projection.cell_transform, point)[0]
                for point in _points(projection.cell_transform.domain)
            ]
            for projection in projections
        ] == [[0, 1], [2, 3], [4, 5]]

    def test_unsorted_coordinates_use_intersection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unsorted coordinates retain the general intersection path."""
        transform = IndexTransform.from_shape((12,)).vindex[np.array([9, 0, 4], dtype=np.intp)]
        grid = ChunkGrid(dimensions=(FixedDimension(size=4, extent=12),))
        calls = _count_intersect_calls(monkeypatch)

        projections = list(plan_chunks(transform, grid.dimensions))

        assert [projection.chunk_coords for projection in projections] == [(0,), (1,), (2,)]
        assert calls["n"] == 3


class CountingUnitGrid:
    """A real unit grid that counts every planner-grid operation."""

    def __init__(self, extent: int) -> None:
        self._grid = FixedDimension(size=1, extent=extent)
        self.calls = 0

    def index_to_chunk(self, idx: int) -> int:
        self.calls += 1
        return self._grid.index_to_chunk(idx)

    def chunk_offset(self, chunk_ix: int) -> int:
        self.calls += 1
        return self._grid.chunk_offset(chunk_ix)

    def chunk_size(self, chunk_ix: int) -> int:
        self.calls += 1
        return self._grid.chunk_size(chunk_ix)

    def indices_to_chunks(
        self, indices: np.ndarray[Any, np.dtype[np.intp]]
    ) -> np.ndarray[Any, np.dtype[np.intp]]:
        self.calls += 1
        return self._grid.indices_to_chunks(indices)


def test_sparse_affine_plan_does_not_visit_intervening_chunks() -> None:
    grid = CountingUnitGrid(extent=100_001)
    transform = IndexTransform.from_shape((100_001,))[::100_000]

    assert [projection.chunk_coords for projection in plan_chunks(transform, (grid,))] == [
        (0,),
        (100_000,),
    ]
    assert grid.calls <= 12


def test_sparse_affine_plan_handles_large_origin_cancellation() -> None:
    origin = int(np.iinfo(np.intp).max)
    transform = IndexTransform(
        domain=IndexDomain((origin,), (origin + 2,)),
        output=(DimensionMap(input_dimension=0, offset=-2 * origin, stride=2),),
    )
    grids = dimension_grids_from_chunks((1,), (3,))

    assert [projection.chunk_coords for projection in plan_chunks(transform, grids)] == [
        (0,),
        (2,),
    ]


class TestTouchedOnlyCandidateEnumeration:
    def test_sparse_one_dimensional_selection_skips_the_dense_span(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two sorted points on a 1000-cell grid require no intersections."""
        transform = IndexTransform.from_shape((4000,)).vindex[np.array([1, 3997], dtype=np.intp)]
        grid = ChunkGrid(dimensions=(FixedDimension(size=4, extent=4000),))
        calls = _count_intersect_calls(monkeypatch)

        projections = list(plan_chunks(transform, grid.dimensions))

        assert [projection.chunk_coords for projection in projections] == [(0,), (999,)]
        assert calls["n"] == 0

    @pytest.mark.parametrize(
        ("mode", "expected_coords", "expected_calls"),
        [
            ("orthogonal", [(0, 0), (0, 999), (999, 0), (999, 999)], 4),
            ("correlated", [(0, 0), (999, 999)], 2),
        ],
    )
    def test_sparse_two_dimensional_selection_uses_only_touched_combinations(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mode: str,
        expected_coords: list[tuple[int, int]],
        expected_calls: int,
    ) -> None:
        """Orthogonal points use their outer product; correlated points remain paired."""
        base = IndexTransform.from_shape((4000, 4000))
        first = np.array([1, 3997], dtype=np.intp)
        second = np.array([2, 3998], dtype=np.intp)
        transform = (
            base.oindex[first, second] if mode == "orthogonal" else base.vindex[first, second]
        )
        grid = ChunkGrid(
            dimensions=(
                FixedDimension(size=4, extent=4000),
                FixedDimension(size=4, extent=4000),
            )
        )
        calls = _count_intersect_calls(monkeypatch)

        projections = list(plan_chunks(transform, grid.dimensions))

        assert sorted(projection.chunk_coords for projection in projections) == expected_coords
        assert calls["n"] == expected_calls

    def test_correlated_diagonal_scales_with_points_not_their_product(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fifty diagonal points require fifty, rather than 2500, intersections."""
        n_points = 50
        coordinates = np.arange(n_points, dtype=np.intp) * 8
        transform = IndexTransform.from_shape((4000, 4000)).vindex[coordinates, coordinates]
        grid = ChunkGrid(
            dimensions=(
                FixedDimension(size=4, extent=4000),
                FixedDimension(size=4, extent=4000),
            )
        )
        calls = _count_intersect_calls(monkeypatch)

        projections = list(plan_chunks(transform, grid.dimensions))

        assert sorted(projection.chunk_coords for projection in projections) == [
            (2 * index, 2 * index) for index in range(n_points)
        ]
        assert calls["n"] == n_points
