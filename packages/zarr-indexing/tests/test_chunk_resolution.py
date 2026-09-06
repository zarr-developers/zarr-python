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
    EdgeDimensionGrid,
    FixedDimension,
    VaryingDimension,
    chunk_resolution,
    plan_chunks,
)
from zarr_indexing.domain import IndexDomain
from zarr_indexing.grid import dimension_grids_from_chunks
from zarr_indexing.output_map import ArrayMap, ConstantMap, DimensionMap
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


def test_repeated_input_dependency_partition_is_rejected() -> None:
    """The per-output-axis table API cannot represent a diagonal."""
    transform = IndexTransform(
        domain=IndexDomain.from_shape((2,)),
        output=(DimensionMap(input_dimension=0), DimensionMap(input_dimension=0)),
    )
    grids = dimension_grids_from_chunks((2, 2), (2, 2))

    with pytest.raises(ValueError, match="read input axis 0"):
        plan_chunks(transform, grids).partition()


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


@pytest.mark.parametrize(
    "grid",
    [
        pytest.param(FixedDimension(size=2, extent=4), id="fixed"),
        pytest.param(VaryingDimension(edges=(1, 3), extent=4), id="varying"),
    ],
)
def test_orthogonal_array_map_plan_rejects_coordinate_below_grid(grid: Any) -> None:
    transform = IndexTransform(
        domain=IndexDomain.from_shape((2,)),
        output=(ArrayMap(np.array([-1, 1], dtype=np.intp)),),
    )

    # The sorted 1-D fast path reports the first offending coordinate.
    with pytest.raises(IndexError, match=r"index -1 is out of bounds"):
        list(plan_chunks(transform, (grid,)))


@pytest.mark.parametrize(
    "grid",
    [
        pytest.param(FixedDimension(size=2, extent=4), id="fixed"),
        pytest.param(VaryingDimension(edges=(1, 3), extent=4), id="varying"),
    ],
)
def test_orthogonal_array_map_plan_rejects_coordinate_above_grid(grid: Any) -> None:
    transform = IndexTransform(
        domain=IndexDomain.from_shape((2,)),
        output=(ArrayMap(np.array([1, 4], dtype=np.intp)),),
    )

    # The sorted 1-D fast path reports the first offending coordinate.
    with pytest.raises(IndexError, match=r"index 4 is out of bounds"):
        list(plan_chunks(transform, (grid,)))


def test_nonempty_identity_plan_rejects_zero_size_fixed_dimension() -> None:
    transform = IndexTransform.from_shape((4,))

    with pytest.raises(ValueError, match="size must be > 0 when extent is nonzero"):
        list(plan_chunks(transform, (FixedDimension(size=0, extent=4),)))


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
    def test_sorted_coordinates_bypass_intersection(self) -> None:
        """Sorted coordinates partition directly at touched chunk boundaries."""
        transform = IndexTransform.from_shape((12,)).vindex[
            np.array([0, 3, 4, 4, 9, 11], dtype=np.intp)
        ]
        grid = ChunkGrid(dimensions=(FixedDimension(size=4, extent=12),))

        projections = list(plan_chunks(transform, grid.dimensions))

        assert [projection.chunk_coords for projection in projections] == [(0,), (1,), (2,)]
        assert [
            [
                _storage_of(projection.cell_transform, point)[0]
                for point in _points(projection.cell_transform.domain)
            ]
            for projection in projections
        ] == [[0, 1], [2, 3], [4, 5]]

    def test_unsorted_coordinates_use_intersection(self) -> None:
        """Unsorted coordinates are grouped by chunk, in chunk order."""
        transform = IndexTransform.from_shape((12,)).vindex[np.array([9, 0, 4], dtype=np.intp)]
        grid = ChunkGrid(dimensions=(FixedDimension(size=4, extent=12),))

        projections = list(plan_chunks(transform, grid.dimensions))

        assert [projection.chunk_coords for projection in projections] == [(0,), (1,), (2,)]


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
        self,
    ) -> None:
        """Two sorted points on a 1000-cell grid touch two chunks."""
        transform = IndexTransform.from_shape((4000,)).vindex[np.array([1, 3997], dtype=np.intp)]
        grid = ChunkGrid(dimensions=(FixedDimension(size=4, extent=4000),))

        projections = list(plan_chunks(transform, grid.dimensions))

        assert [projection.chunk_coords for projection in projections] == [(0,), (999,)]

    @pytest.mark.parametrize(
        ("mode", "expected_coords"),
        [
            ("orthogonal", [(0, 0), (0, 999), (999, 0), (999, 999)]),
            ("correlated", [(0, 0), (999, 999)]),
        ],
    )
    def test_sparse_two_dimensional_selection_uses_only_touched_combinations(
        self,
        mode: str,
        expected_coords: list[tuple[int, int]],
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

        projections = list(plan_chunks(transform, grid.dimensions))

        assert sorted(projection.chunk_coords for projection in projections) == expected_coords

    def test_correlated_diagonal_scales_with_points_not_their_product(
        self,
    ) -> None:
        """Fifty diagonal points touch fifty chunks, not 2500."""
        n_points = 50
        coordinates = np.arange(n_points, dtype=np.intp) * 8
        transform = IndexTransform.from_shape((4000, 4000)).vindex[coordinates, coordinates]
        grid = ChunkGrid(
            dimensions=(
                FixedDimension(size=4, extent=4000),
                FixedDimension(size=4, extent=4000),
            )
        )

        projections = list(plan_chunks(transform, grid.dimensions))

        assert sorted(projection.chunk_coords for projection in projections) == [
            (2 * index, 2 * index) for index in range(n_points)
        ]


# ---------------------------------------------------------------------------
# GridPartition: the factored form
# ---------------------------------------------------------------------------


def _partition_cases() -> list[tuple[str, IndexTransform, tuple[Any, ...]]]:
    base = IndexTransform.from_shape((7, 9, 5))
    fixed = dimension_grids_from_chunks((3, 4, 2), shape=(7, 9, 5))
    varying = (
        VaryingDimension(edges=(2, 5), extent=7),
        VaryingDimension(edges=(1, 4, 4), extent=9),
        FixedDimension(size=2, extent=5),
    )
    # Declared edges past the extent: the last chunk's data extent is shorter
    # than its declared size, which is where `data_size` and `chunk_size` part.
    clipped = (
        VaryingDimension(edges=(5, 10), extent=7),
        VaryingDimension(edges=(4, 4, 4), extent=9),
        FixedDimension(size=3, extent=5),
    )
    # The minimal grid protocol: no `data_size`, so resolution falls back to `chunk_size`.
    edges = (EdgeDimensionGrid([3, 4]), EdgeDimensionGrid([4, 5]), EdgeDimensionGrid([2, 3]))
    return [
        ("identity", base, fixed),
        ("strided", base[1:6:2, 5:, ::3], fixed),
        ("strided varying", base[1:6:2, 5:, ::3], varying),
        ("scalars", base[3, :, 4], fixed),
        ("reversed", base[::-1, 8:1:-1, :], fixed),
        ("empty", base[3:3, :, :], fixed),
        ("oindex arrays", base.oindex[np.array([6, 0, 0, 2]), :, np.array([4, 1])], fixed),
        ("oindex mixed", base.oindex[np.array([1, 5]), 2, 1:5:2], varying),
        ("oindex one element", base.oindex[np.array([2]), :, :], fixed),
        ("vindex", base.vindex[np.array([0, 6, 6, 1]), np.array([8, 0, 1, 8]), :], fixed),
        ("vindex varying", base.vindex[np.array([0, 6, 6, 1]), np.array([8, 0, 1, 8]), 2], varying),
        ("strided clipped", base[1:7:2, 5:, ::3], clipped),
        ("oindex clipped", base.oindex[np.array([6, 0, 6]), 8, 1:5:2], clipped),
        ("vindex clipped", base.vindex[np.array([6, 6, 1]), np.array([8, 0, 8]), :], clipped),
        ("strided edge grid", base[1:7:2, 5:, ::3], edges),
        ("vindex edge grid", base.vindex[np.array([6, 6, 1]), np.array([8, 0, 8]), :], edges),
        (
            "vindex 2-d block",
            base.vindex[
                np.array([[0, 6], [3, 1]]), np.array([[8, 0], [2, 8]]), np.array([[4, 0], [1, 1]])
            ],
            fixed,
        ),
        (
            "vindex 1-d sorted",
            IndexTransform.from_shape((20,)).vindex[np.array([1, 5, 9, 17])],
            dimension_grids_from_chunks((4,), shape=(20,)),
        ),
        ("rank 0", IndexTransform.from_shape(()), ()),
    ]


def _check_projections(transform: IndexTransform, projections: list[ChunkProjection]) -> None:
    """The evaluation oracle: both transforms agree pointwise, cells tile the
    request exactly once, chunk-local coordinates lie in the chunk, and
    `coverage` is `full` exactly when a chunk's cells are each read once."""
    has_array = any(isinstance(m, ArrayMap) for m in transform.output)
    request_points: list[tuple[int, ...]] = []
    for projection in projections:
        assert projection.chunk_transform.domain == projection.cell_transform.domain
        chunk_points: list[tuple[int, ...]] = []
        for cell_point in _points(projection.cell_transform.domain):
            request_point = _storage_of(projection.cell_transform, cell_point)
            chunk_point = _storage_of(projection.chunk_transform, cell_point)
            storage_point = _storage_of(transform, request_point)
            chunk_origin = projection.chunk_domain.inclusive_min
            assert chunk_point == tuple(
                value - origin for value, origin in zip(storage_point, chunk_origin, strict=True)
            )
            assert all(
                0 <= value < extent
                for value, extent in zip(chunk_point, projection.chunk_domain.shape, strict=True)
            )
            request_points.append(request_point)
            chunk_points.append(chunk_point)
        if has_array:
            assert projection.coverage == "unknown"
        else:
            covers = sorted(chunk_points) == sorted(
                tuple(int(c) for c in cell) for cell in np.ndindex(*projection.chunk_domain.shape)
            )
            assert (projection.coverage == "full") == covers, projection
    assert sorted(request_points) == sorted(_points(transform.domain))


@pytest.mark.parametrize("case", _partition_cases(), ids=lambda case: case[0])
def test_partition_rows_are_the_plan_and_satisfy_the_oracle(
    case: tuple[str, IndexTransform, tuple[Any, ...]],
) -> None:
    _, transform, grids = case
    plan = plan_chunks(transform, grids)
    partition = plan.partition()
    rows = list(partition)
    assert rows == list(plan)
    assert len(partition) == len(rows)
    _check_projections(transform, rows)


@pytest.mark.parametrize("case", _partition_cases(), ids=lambda case: case[0])
def test_partition_chunk_coords_are_vectorized_rows(
    case: tuple[str, IndexTransform, tuple[Any, ...]],
) -> None:
    """`chunk_coords` reads the tables without materializing a row per chunk."""
    _, transform, grids = case
    partition = plan_chunks(transform, grids).partition()
    coords = partition.chunk_coords()
    assert coords.shape == (len(partition), transform.output_rank)
    assert coords.tolist() == [list(p.chunk_coords) for p in partition]


def test_partition_tables_describe_chunk_local_coordinates() -> None:
    """The columnar tables carry exactly what each row's chunk transform maps to."""
    transform = IndexTransform.from_shape((7, 9)).oindex[np.array([6, 0, 0, 2]), 5:]
    grids = dimension_grids_from_chunks((3, 4), shape=(7, 9))
    partition = plan_chunks(transform, grids).partition()
    indexed, strided = partition.sets
    assert isinstance(indexed, chunk_resolution.IndexedSet)
    assert isinstance(strided, chunk_resolution.StridedSet)
    # rows are in chunk order; the array [6, 0, 0, 2] lands in chunks 0, 0, 0, 2
    assert indexed.chunk.tolist() == [0, 2]
    assert indexed.pointer.tolist() == [0, 3, 4]
    assert indexed.local.tolist() == [0, 0, 2, 0]
    assert indexed.positions.tolist() == [1, 2, 3, 0]
    assert strided.chunk.tolist() == [1, 2]
    assert strided.local_start.tolist() == [1, 0]
    assert strided.extent.tolist() == [3, 1]
    assert strided.origin.tolist() == [0, 3]  # positions along the request axis
    assert strided.full.tolist() == [False, True]
    for row, projection in enumerate(partition):
        table_rows = np.unravel_index(row, partition.row_shape)
        run = indexed.run(int(table_rows[0]))
        storage = projection.chunk_transform.apply_many(
            np.array(list(np.ndindex(*projection.chunk_transform.domain.shape)))
        )
        assert sorted(set(storage[:, 0].tolist())) == sorted(set(indexed.local[run].tolist()))


def test_joint_set_groups_points_by_chunk() -> None:
    transform = IndexTransform.from_shape((7, 9)).vindex[
        np.array([0, 6, 6, 1]), np.array([8, 0, 1, 8])
    ]
    grids = dimension_grids_from_chunks((3, 4), shape=(7, 9))
    joint = plan_chunks(transform, grids).partition().joint_sets[0]
    assert joint is not None
    assert joint.chunk.tolist() == [[0, 2], [2, 0]]
    assert joint.pointer.tolist() == [0, 2, 4]
    assert joint.positions.tolist() == [0, 3, 1, 2]
    assert joint.local.tolist() == [[0, 0], [1, 0], [0, 0], [0, 1]]


def test_partition_rejects_correlated_residual_diagonal() -> None:
    """A diagonal among the residual slice axes of a correlated transform is rejected too."""
    transform = IndexTransform(
        domain=IndexDomain.from_shape((2, 3)),
        output=(
            ArrayMap(np.array([[0], [2]])),
            ArrayMap(np.array([[1], [0]])),
            DimensionMap(input_dimension=1),
            DimensionMap(input_dimension=1),
        ),
    )
    grids = dimension_grids_from_chunks((2, 2, 2, 2), shape=(3, 3, 3, 3))
    with pytest.raises(ValueError, match="read input axis 1"):
        plan_chunks(transform, grids).partition()
    with pytest.raises(ValueError, match="read input axis 1"):
        list(plan_chunks(transform, grids))


def test_correlated_plan_checks_storage_bounds_with_any_grid() -> None:
    """Out-of-range coordinates raise even on grids whose vectorized lookup does not check."""

    class UncheckedGrid:  # like zarr's FixedDimension: only the scalar lookup validates
        def __init__(self, size: int, extent: int) -> None:
            self.size, self.extent = size, extent

        def index_to_chunk(self, index: int) -> int:
            if not 0 <= index < self.extent:
                raise IndexError(f"index {index} is out of bounds for extent {self.extent}")
            return index // self.size

        def chunk_offset(self, chunk: int) -> int:
            return chunk * self.size

        def chunk_size(self, chunk: int) -> int:
            return self.size

        def indices_to_chunks(self, indices: Any) -> Any:
            return indices // self.size

    transform = IndexTransform.from_shape((10, 10)).vindex[np.array([1, 9]), np.array([2, 9])]
    with pytest.raises(IndexError, match="out of bounds"):
        list(plan_chunks(transform, (UncheckedGrid(2, 5), UncheckedGrid(2, 5))))


def test_wide_request_extent_is_preserved() -> None:
    """A zero-stride map over a domain wider than np.intp touches one cell and keeps its extent."""
    width = 2**63
    transform = IndexTransform(
        domain=IndexDomain((0,), (width,)),
        output=(DimensionMap(input_dimension=0, offset=3, stride=0),),
    )
    (projection,) = plan_chunks(transform, dimension_grids_from_chunks((2,), shape=(5,)))
    assert projection.chunk_coords == (1,)
    assert projection.chunk_transform.domain.shape == (width,)
    assert projection.coverage == "partial"


@pytest.mark.parametrize("width", [3, 10_000])
def test_correlated_projection_keeps_residual_slices_compact(width: int) -> None:
    indices = np.array([0, 2, 1])
    transform = IndexTransform.from_shape((3, 3, width)).vindex[indices, indices, :]
    grids = dimension_grids_from_chunks((3, 3, width), shape=(3, 3, width))
    (projection,) = plan_chunks(transform, grids)
    assert projection.cell_transform.apply((2, width - 1)) == (2, width - 1)
    assert projection.chunk_transform.apply((2, width - 1)) == (1, 1, width - 1)
    coordinate_bytes = sum(
        m.index_array.nbytes for m in projection.cell_transform.output if isinstance(m, ArrayMap)
    )
    # Request placement needs only the three point positions, regardless of slab width.
    assert coordinate_bytes <= indices.nbytes


def test_mixed_affine_array_dependency_is_rejected() -> None:
    """A slice map and an index array reading one input axis is a diagonal too."""
    transform = IndexTransform(
        IndexDomain.from_shape((4,)), (DimensionMap(0), ArrayMap(np.array([3, 2, 1, 0])))
    )
    grids = dimension_grids_from_chunks((2, 2), shape=(4, 4))
    with pytest.raises(ValueError, match="read input axis 0"):
        list(plan_chunks(transform, grids))


def test_partition_tables_are_memoized_and_immutable() -> None:
    """The plan builds its tables once, and nothing a consumer does to a column can change them."""
    base = IndexTransform.from_shape((7, 9))
    indices = np.array([6, 0, 6, 1])
    for transform in (base.oindex[indices, 5:], base.vindex[indices, indices]):
        plan = plan_chunks(transform, dimension_grids_from_chunks((3, 4), (7, 9)))
        partition = plan.partition()
        assert plan.partition() is partition
        tables: list[Any] = [*partition.sets, *partition.joint_sets]
        for table in tables:
            columns = [table.chunk, table.chunk_start, table.chunk_extent]
            if hasattr(table, "pointer"):
                assert table.local is table.local
                columns += [table.local, table.pointer, table.index, table.positions]
            else:
                columns += [table.origin, table.extent]
            for column in columns:
                assert not column.flags.writeable
                with pytest.raises(ValueError):
                    column.setflags(write=True)


def test_independent_correlated_groups_stay_compact() -> None:
    n = 1000
    indices = np.arange(n)
    transform = IndexTransform.from_shape((n, n, n)).vindex[
        indices[:, None], indices[:, None], indices[None, :]
    ]
    assert sum(m.index_array.size for m in transform.output if isinstance(m, ArrayMap)) == 3 * n
    partition = plan_chunks(
        transform, dimension_grids_from_chunks((n, n, n), (n, n, n))
    ).partition()
    assert [j.output_dimensions for j in partition.joint_sets] == [(0, 1), (2,)]
    assert sum(j.index.size for j in partition.joint_sets) == 3 * n
    assert partition.row_shape == (1, 1)
    (projection,) = partition
    assert projection.chunk_transform.domain.shape == (n, n)
    assert projection.chunk_transform.apply((5, 9)) == (5, 5, 9)
    assert projection.cell_transform.apply((5, 9)) == (5, 9)


@pytest.mark.parametrize("origin", [0, 5])
@pytest.mark.parametrize("transpose", [False, True])
def test_component_projections_preserve_request_cells(origin: int, transpose: bool) -> None:
    maps = (
        ArrayMap(np.array([2, 0, 2]).reshape(3, 1, 1, 1)),
        ArrayMap(np.array([0, 3, 1]).reshape(3, 1, 1, 1)),
        ArrayMap(np.array([4, 0, 4, 2]).reshape(1, 4, 1, 1)),
        DimensionMap(2, offset=-origin),
    )
    transform = IndexTransform(
        IndexDomain((origin,) * 4, (origin + 3, origin + 4, origin + 2, origin + 1)), maps
    )
    if transpose:
        transform = IndexTransform(transform.domain, (maps[2], maps[0], maps[3], maps[1]))
    shape = (5, 3, 2, 4) if transpose else (3, 4, 5, 2)
    grids = dimension_grids_from_chunks((2, 2, 2, 2), shape)
    partition = plan_chunks(transform, grids).partition()
    assert len(partition.joint_sets) == 2
    rows = list(partition)
    _check_projections(transform, rows)
    assert partition.chunk_coords().tolist() == [list(p.chunk_coords) for p in rows]


@pytest.mark.parametrize("origin", [0, 5])
@pytest.mark.parametrize("shape", [(6,), (2, 3)])
@pytest.mark.parametrize("reverse_outputs", [False, True])
def test_single_component_projection_coordinates(
    origin: int, shape: tuple[int, ...], reverse_outputs: bool
) -> None:
    """A sole component preserves repeated points, affine maps, and input origins."""
    values = np.array([4, 0, 4, 2, 1, 3]).reshape(shape)
    maps = (ArrayMap(values, offset=1, stride=2), ArrayMap(values, offset=9, stride=-2))
    transform = IndexTransform(
        IndexDomain((origin,) * len(shape), tuple(origin + size for size in shape)),
        maps[::-1] if reverse_outputs else maps,
    )
    partition = plan_chunks(
        transform, dimension_grids_from_chunks(((2, 3, 5), (4, 6)), (10, 10))
    ).partition()
    assert len(partition.joint_sets) == 1
    assert not partition.sets
    projections = list(partition)
    _check_projections(transform, projections)
    assert partition.chunk_coords().tolist() == [list(p.chunk_coords) for p in projections]


def test_index_array_components_merge_transitively() -> None:
    transform = IndexTransform(
        IndexDomain.from_shape((2, 3, 4)),
        (
            ArrayMap(np.zeros((2, 1, 1), dtype=np.intp)),
            ArrayMap(np.zeros((1, 1, 4), dtype=np.intp)),
            ArrayMap(np.zeros((2, 3, 1), dtype=np.intp)),
            ArrayMap(np.zeros((1, 3, 4), dtype=np.intp)),
        ),
    )
    partition = plan_chunks(transform, dimension_grids_from_chunks((1,) * 4, (1,) * 4)).partition()
    assert len(partition.joint_sets) == 1
    _check_projections(transform, list(partition))


@pytest.mark.parametrize("reader_kind", ["basic", "numpy"])
def test_independent_components_scatter_through_lazy_array(reader_kind: str) -> None:
    from zarr_indexing import LazyArray

    source = np.arange(5 * 6 * 7 * 3).reshape(5, 6, 7, 3)
    a = np.array([4, 0, 4])[:, None]
    b = np.array([5, 1, 3])[:, None]
    c = np.array([6, 0, 6, 2])[None, :]
    wrapped = LazyArray.from_numpy(source) if reader_kind == "numpy" else LazyArray(source)
    view = wrapped.with_parts((2, 3, 2, 2)).lazy.vindex[a, b, c, ...]
    assert (
        len(
            plan_chunks(view.transform, dimension_grids_from_chunks((2, 3, 2, 2), source.shape))
            .partition()
            .joint_sets
        )
        == 2
    )
    np.testing.assert_array_equal(view.result(), source[a, b, c, :])


@given(data=st.data())
def test_component_dependency_graph_matches_pointwise_oracle(data: st.DataObject) -> None:
    shape = tuple(data.draw(st.lists(st.integers(1, 3), min_size=0, max_size=3)))
    output_rank = data.draw(st.integers(1, 5))
    maps = []
    for _ in range(output_rank):
        dependencies = data.draw(st.lists(st.booleans(), min_size=len(shape), max_size=len(shape)))
        array_shape = tuple(
            size if dependent else 1 for size, dependent in zip(shape, dependencies, strict=True)
        )
        count = int(np.prod(array_shape))
        values = data.draw(st.lists(st.integers(0, 3), min_size=count, max_size=count))
        maps.append(ArrayMap(np.array(values, dtype=np.intp).reshape(array_shape)))
    transform = IndexTransform(IndexDomain.from_shape(shape), tuple(maps))
    grids = dimension_grids_from_chunks((2,) * output_rank, (4,) * output_rank)
    partition = plan_chunks(transform, grids).partition()
    rows = list(partition)
    _check_projections(transform, rows)
    assert partition.chunk_coords().tolist() == [list(row.chunk_coords) for row in rows]


@pytest.mark.parametrize("stride", [0, 1, 2, -2])
@pytest.mark.parametrize("count", [1, 2])
def test_singleton_coverage(stride: int, count: int) -> None:
    transform = IndexTransform(IndexDomain.from_shape((count,)), (DimensionMap(0, stride=stride),))
    shape = (5,) * transform.output_rank

    class SingletonGrid:
        def index_to_chunk(self, index: int) -> int:
            return index

        def chunk_offset(self, chunk: int) -> int:
            return chunk

        def chunk_size(self, chunk: int) -> int:
            return 1

        def indices_to_chunks(self, indices: Any) -> Any:
            return indices

    grids = (SingletonGrid(),) * len(shape)
    expected = "partial" if stride == 0 and count > 1 else "full"
    assert all(p.coverage == expected for p in plan_chunks(transform, grids))
    partition = plan_chunks(transform, grids).partition()
    assert all(bool(full) == (expected == "full") for full in partition.sets[0].full)
