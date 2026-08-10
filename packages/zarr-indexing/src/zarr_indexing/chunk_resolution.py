"""Chunk resolution — mapping transforms to chunk-level I/O.

Given an `IndexTransform` (which coordinates a user wants to access) and a
`ChunkGrid` (how storage is divided into chunks), chunk resolution answers:

    For each chunk, which storage coordinates does this transform touch,
    and where do those values land in the output buffer?

The algorithm is:

1. **Enumerate candidate chunks** — determine which chunks could possibly
   be touched by the transform's output coordinate ranges.

2. **Intersect** — for each candidate chunk, call
   `transform.intersect(chunk_domain)` to restrict the transform to
   coordinates within that chunk. If the intersection is empty, skip it.

3. **Translate** — shift the restricted transform to chunk-local coordinates
   via `transform.translate(-chunk_origin)`.

4. **Project** — pair the chunk-local storage transform with a transform back
   to the request's cells. Both use the same compact, zero-origin domain.

Sorted one-dimensional correlated array maps can be partitioned directly
because every touched chunk owns a contiguous slice of the index array. That
case bypasses candidate enumeration and repeated intersection.

The public result is a lazy, reusable `ChunkPlan`. Each `ChunkProjection` is
source-independent: it identifies the chunk and expresses both sides of the
gather without assuming NumPy selectors, a codec pipeline, or an execution
scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from zarr_indexing.affine import checked_affine
from zarr_indexing.domain import IndexDomain
from zarr_indexing.output_map import ArrayMap, ConstantMap, DimensionMap
from zarr_indexing.transform import (
    IndexTransform,
    array_map_dependent_axis,
    index_array_structure,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from zarr_indexing.grid import DimensionGridLike

_OutIndices = (
    dict[int, np.ndarray[Any, np.dtype[np.intp]]] | np.ndarray[Any, np.dtype[np.intp]] | None
)

_ChunkTransformResult = tuple[
    tuple[int, ...],
    IndexTransform,
    _OutIndices,
]

type ChunkCoverage = Literal["full", "partial", "unknown"]


def _data_size(dim_grid: DimensionGridLike, chunk_ix: int) -> int:
    """Return a chunk's data extent, falling back for narrow-protocol grids."""
    data_size = getattr(dim_grid, "data_size", None)
    if data_size is None:
        return dim_grid.chunk_size(chunk_ix)
    return int(data_size(chunk_ix))


@dataclass(frozen=True, slots=True)
class ChunkProjection:
    """One source-independent projection of a request through a chunk.

    Both transforms share a synthetic input domain. ``chunk_transform`` maps
    that domain to chunk-local storage coordinates; ``cell_transform`` maps it
    to the original request domain.

    Attributes
    ----------
    chunk_coords
        Coordinates of the selected cell in the caller's grid.
    chunk_domain
        Bounds of that grid cell in global storage coordinates.
    chunk_transform
        Mapping from the shared synthetic domain to chunk-local storage.
    cell_transform
        Mapping from the shared synthetic domain to request coordinates.
    coverage
        Whether the request is proven to cover the whole grid cell exactly
        once. Fancy selections are conservatively ``"unknown"``.

    Examples
    --------
    Row 1 of a `(3, 4)` array with `(2, 2)` chunks touches only part of the
    first chunk, whose domain spans rows `[0, 2)` and columns `[0, 2)`:

    >>> from zarr_indexing import IndexTransform
    >>> from zarr_indexing.grid import dimension_grids_from_chunks
    >>> grids = dimension_grids_from_chunks((2, 2), shape=(3, 4))
    >>> plan = plan_chunks(IndexTransform.from_shape((3, 4))[1, :], grids)
    >>> first = next(iter(plan))
    >>> first.chunk_coords
    (0, 0)
    >>> first.chunk_domain.shape
    (2, 2)
    >>> first.coverage
    'partial'
    """

    chunk_coords: tuple[int, ...]
    chunk_domain: IndexDomain
    chunk_transform: IndexTransform
    cell_transform: IndexTransform
    coverage: ChunkCoverage

    def __post_init__(self) -> None:
        if self.chunk_transform.domain != self.cell_transform.domain:
            raise ValueError(
                "chunk_transform and cell_transform must share an input domain; "
                f"got {self.chunk_transform.domain!r} and {self.cell_transform.domain!r}"
            )


@dataclass(frozen=True, slots=True)
class ChunkPlan:
    """A reusable, lazy partition of an index transform over a chunk grid.

    Construct plans with `plan_chunks`; iterating either the plan or
    `projections()` performs a fresh chunk walk.

    Examples
    --------
    Row 1 of a `(3, 4)` array with `(2, 2)` chunks crosses two chunks, and
    the plan can be walked again after it is exhausted:

    >>> from zarr_indexing import IndexTransform
    >>> from zarr_indexing.grid import dimension_grids_from_chunks
    >>> grids = dimension_grids_from_chunks((2, 2), shape=(3, 4))
    >>> plan = plan_chunks(IndexTransform.from_shape((3, 4))[1, :], grids)
    >>> [p.chunk_coords for p in plan]
    [(0, 0), (0, 1)]
    >>> [p.chunk_coords for p in plan.projections()]
    [(0, 0), (0, 1)]
    """

    transform: IndexTransform
    """The composed request this plan partitions."""

    dimension_grids: tuple[DimensionGridLike, ...]
    """One grid per storage dimension, defining the chunk layout the plan walks."""

    def projections(self) -> Iterator[ChunkProjection]:
        """Return a fresh iterator over the chunks touched by this plan."""
        return _iter_chunk_projections(self.transform, self.dimension_grids)

    def __iter__(self) -> Iterator[ChunkProjection]:
        """Equivalent to `projections()`: each iteration performs a fresh chunk walk."""
        return self.projections()


def plan_chunks(
    transform: IndexTransform,
    dimension_grids: Sequence[DimensionGridLike],
) -> ChunkPlan:
    """Plan a transform against a caller-selected chunk grid.

    Parameters
    ----------
    transform
        Mapping from the request domain to storage coordinates.
    dimension_grids
        One storage grid per transform output dimension.

    Returns
    -------
    ChunkPlan
        A reusable plan whose projections are computed lazily.

    Examples
    --------
    Row 1 of a `(3, 4)` array with `(2, 2)` chunks touches the two chunks in
    the top grid row, each contributing a `(2, 2)` chunk domain:

    >>> from zarr_indexing import IndexTransform
    >>> from zarr_indexing.grid import dimension_grids_from_chunks
    >>> grids = dimension_grids_from_chunks((2, 2), shape=(3, 4))
    >>> plan = plan_chunks(IndexTransform.from_shape((3, 4))[1, :], grids)
    >>> [p.chunk_coords for p in plan]
    [(0, 0), (0, 1)]
    >>> [p.chunk_domain.shape for p in plan]
    [(2, 2), (2, 2)]
    """
    grids = tuple(dimension_grids)
    if len(grids) != transform.output_rank:
        raise ValueError(
            "dimension_grids must have one entry per transform output dimension; "
            f"got {len(grids)} grids for output rank {transform.output_rank}"
        )
    return ChunkPlan(transform=transform, dimension_grids=grids)


def _one_dimensional_array_map(
    transform: IndexTransform,
) -> tuple[ArrayMap, np.ndarray[Any, np.dtype[np.intp]]] | None:
    """Return a nonempty 1-D single-ArrayMap transform's map and storage coords.

    A one-dimensional array selection has no cross-dimensional correlation to
    preserve — the orthogonal and vectorized flavors coincide there — so the
    sorted fast path applies to either spelling. The computed storage
    coordinates are also reused by general resolution when they are unsorted.
    """
    if transform.input_rank != 1 or transform.output_rank != 1:
        return None

    m = transform.output[0]
    if not isinstance(m, ArrayMap) or m.index_array.ndim != 1 or m.index_array.size == 0:
        return None

    return m, checked_affine(m.offset, m.stride, m.index_array)


def _iter_sorted_1d_array_map(
    m: ArrayMap,
    storage: np.ndarray[Any, np.dtype[np.intp]],
    dim_grid: DimensionGridLike,
) -> Iterator[_ChunkTransformResult]:
    """Resolve a sorted 1-D ArrayMap one touched chunk at a time."""
    start = 0
    while start < storage.size:
        chunk = dim_grid.index_to_chunk(int(storage[start]))
        chunk_start = dim_grid.chunk_offset(chunk)
        chunk_stop = chunk_start + _data_size(dim_grid, chunk)
        stop = int(np.searchsorted(storage, chunk_stop, side="left"))

        restricted = IndexTransform(
            domain=IndexDomain(inclusive_min=(0,), exclusive_max=(stop - start,)),
            output=(
                ArrayMap(
                    index_array=m.index_array[start:stop],
                    offset=m.offset,
                    stride=m.stride,
                ),
            ),
        )
        local = restricted.translate((-chunk_start,))
        surviving = np.arange(start, stop, dtype=np.intp)

        yield (chunk,), local, surviving
        start = stop


def _iter_chunk_transform_results(
    transform: IndexTransform,
    dim_grids: Sequence[DimensionGridLike],
) -> Iterator[_ChunkTransformResult]:
    """Resolve a transform into private intersection bookkeeping.

    The survivor arrays are an implementation detail immediately converted to
    a public `cell_transform` by `_iter_chunk_projections`.
    """

    if any(size == 0 for size in transform.domain.shape):
        # An empty view touches no chunk. Checked on the domain rather than on
        # the index arrays: an axis of genuine extent 1 is stored as a broadcast
        # singleton, so a slice that empties the domain does not shrink the
        # array, and the emptiness shows only here.
        return

    array_map_1d = _one_dimensional_array_map(transform)
    if array_map_1d is not None:
        sorted_map, storage = array_map_1d
        if storage[0] <= storage[-1] and bool(np.all(storage[1:] >= storage[:-1])):
            dim_grid = dim_grids[0]
            first_chunk = dim_grid.index_to_chunk(int(storage[0]))
            if dim_grid.chunk_size(first_chunk) > 0:
                yield from _iter_sorted_1d_array_map(sorted_map, storage, dim_grid)
                return

    # Enumerate candidate chunks via the cartesian product of per-slot candidate
    # chunk ids, then for each candidate intersect the transform with the chunk
    # domain (`transform.intersect` handles orthogonal and vectorized cases
    # alike, filtering out combinations it does not actually touch).
    #
    # A slot covers one or more output dimensions and contributes exactly the
    # chunk-coordinate tuples those dimensions can touch:
    #
    # - `ConstantMap`/`DimensionMap` dims each form their own slot with a
    #   contiguous range — a single chunk for a constant, and the span between
    #   the first and last chunk for a slice. These are already tight (or
    #   nearly so).
    # - Orthogonal `ArrayMap` (fancy) dims each form their own slot with only
    #   the *distinct* chunk ids the index array actually lands in
    #   (`np.unique`), never the dense `range(min_chunk, max_chunk + 1)`
    #   between them. A sparse fancy selection (e.g. two far-apart coordinates)
    #   would otherwise enumerate every chunk in the bounding box, making
    #   resolution scale with grid size instead of with the number of selected
    #   coordinates.
    # - Correlated (vindex) `ArrayMap` dims share one *joint* slot holding the
    #   distinct chunk-coordinate tuples the points actually land in. The
    #   cartesian product of their per-dimension distinct sets would include
    #   combinations no point touches — quadratic in the number of selected
    #   points for a diagonal selection — while the joint distinct set is
    #   bounded by the point count (see zarr-python gh-4174).
    structure = index_array_structure(transform)
    correlated_dims: list[int] = []
    correlated_chunk_ids: list[np.ndarray[Any, np.dtype[np.intp]]] = []
    slot_dims: list[tuple[int, ...]] = []
    slot_candidates: list[Sequence[tuple[int, ...]]] = []
    for out_dim, m in enumerate(transform.output):
        dg = dim_grids[out_dim]
        if isinstance(m, ConstantMap):
            # Single chunk
            coordinate = checked_affine(m.offset, 0, 0)
            c = dg.index_to_chunk(coordinate)
            slot_dims.append((out_dim,))
            slot_candidates.append(((c,),))
        elif isinstance(m, DimensionMap):
            d = m.input_dimension
            dim_lo = transform.domain.inclusive_min[d]
            dim_hi = transform.domain.exclusive_max[d]
            if dim_lo >= dim_hi:
                return  # empty domain
            first_storage = checked_affine(m.offset, m.stride, dim_lo)
            if m.stride > 0:
                s_min = first_storage
                s_max = checked_affine(m.offset, m.stride, dim_hi - 1)
            elif m.stride < 0:
                s_min = checked_affine(m.offset, m.stride, dim_hi - 1)
                s_max = first_storage
            else:
                s_min = s_max = first_storage
            first = dg.index_to_chunk(s_min)
            last = dg.index_to_chunk(s_max)
            slot_dims.append((out_dim,))
            point_count = dim_hi - dim_lo
            chunk_count = last - first + 1
            if point_count < chunk_count:
                steps = np.arange(point_count, dtype=np.intp)
                storage = checked_affine(first_storage, m.stride, steps)
                chunk_ids = dg.indices_to_chunks(storage)
                slot_candidates.append([(int(c),) for c in np.unique(chunk_ids)])
            else:
                slot_candidates.append([(c,) for c in range(first, last + 1)])
        else:
            # m: ArrayMap (OutputIndexMap = ConstantMap | DimensionMap | ArrayMap).
            # Storage coordinates were already computed for a correlated 1-D map.
            storage = (
                array_map_1d[1]
                if array_map_1d is not None
                else checked_affine(m.offset, m.stride, m.index_array)
            )
            if storage.size == 0:
                # Empty fancy selection: no coordinates, so no chunks are touched.
                return
            # Keep the index-array shape: correlated maps broadcast against each
            # other below, and raveling first would lose the singleton axes.
            chunk_ids = dg.indices_to_chunks(storage)
            if structure == "orthogonal":
                slot_dims.append((out_dim,))
                slot_candidates.append([(int(c),) for c in np.unique(chunk_ids)])
            else:
                # Every index array of a general transform joins one joint
                # slot: their chunk ids broadcast over the shared block, so the
                # distinct tuples enumerate only combinations some point
                # actually touches.
                correlated_dims.append(out_dim)
                correlated_chunk_ids.append(chunk_ids)

    if len(correlated_dims) == 1:
        slot_dims.append((correlated_dims[0],))
        slot_candidates.append([(int(c),) for c in np.unique(correlated_chunk_ids[0])])
    elif len(correlated_dims) >= 2:
        # Group the points jointly: distinct rows of the per-point chunk
        # coordinates, O(points log points) regardless of grid size.
        broadcast = np.broadcast_arrays(*correlated_chunk_ids)
        stacked = np.stack([b.ravel() for b in broadcast], axis=1)
        joint = np.unique(stacked, axis=0)
        slot_dims.append(tuple(correlated_dims))
        slot_candidates.append([tuple(int(c) for c in row) for row in joint])

    import itertools

    output_rank = len(transform.output)
    for combo in itertools.product(*slot_candidates):
        chunk_coords_list = [0] * output_rank
        for dims, part in zip(slot_dims, combo, strict=True):
            for d, c in zip(dims, part, strict=True):
                chunk_coords_list[d] = c
        chunk_coords = tuple(chunk_coords_list)

        # Build the chunk domain in storage space
        chunk_min: list[int] = []
        chunk_max: list[int] = []
        chunk_shift: list[int] = []
        for out_dim, c in enumerate(chunk_coords):
            dg = dim_grids[out_dim]
            c_start = dg.chunk_offset(c)
            c_size = _data_size(dg, c)
            chunk_min.append(c_start)
            chunk_max.append(c_start + c_size)
            chunk_shift.append(-c_start)

        chunk_domain = IndexDomain(
            inclusive_min=tuple(chunk_min),
            exclusive_max=tuple(chunk_max),
        )

        # Intersect transform with chunk domain
        result = transform.intersect(chunk_domain)
        if result is None:
            continue

        restricted, surviving = result

        # Translate to chunk-local coordinates
        local = restricted.translate(tuple(chunk_shift))

        yield (chunk_coords, local, surviving)


def _covers_whole_chunk(transform: IndexTransform, chunk_shape: tuple[int, ...]) -> bool:
    """Whether an affine chunk-local transform bijects onto every chunk cell."""
    domain = transform.domain
    used_nontrivial_inputs: set[int] = set()
    for out_dim, m in enumerate(transform.output):
        extent = chunk_shape[out_dim]
        if isinstance(m, ConstantMap):
            if extent != 1 or m.offset != 0:
                return False
        elif isinstance(m, DimensionMap):
            if abs(m.stride) != 1:
                return False
            lo = domain.inclusive_min[m.input_dimension]
            hi = domain.exclusive_max[m.input_dimension]
            if hi <= lo:
                if extent != 0:
                    return False
                continue
            first = m.offset + m.stride * lo
            last = m.offset + m.stride * (hi - 1)
            if min(first, last) != 0 or max(first, last) != extent - 1:
                return False
            if extent > 1:
                if m.input_dimension in used_nontrivial_inputs:
                    return False
                used_nontrivial_inputs.add(m.input_dimension)
        else:
            return False
    nontrivial_inputs = {dimension for dimension, extent in enumerate(domain.shape) if extent > 1}
    return used_nontrivial_inputs == nontrivial_inputs


def _orthogonal_cell_transform(
    original: IndexTransform,
    restricted: IndexTransform,
    survivors: dict[int, np.ndarray[Any, np.dtype[np.intp]]] | np.ndarray[Any, np.dtype[np.intp]],
) -> IndexTransform:
    """Map a compacted orthogonal intersection back to request coordinates."""
    by_input_dimension: dict[int, np.ndarray[Any, np.dtype[np.intp]]] = {}
    if isinstance(survivors, dict):
        survivor_items = survivors.items()
    else:
        array_output_dimensions = [
            output_dimension
            for output_dimension, output_map in enumerate(original.output)
            if isinstance(output_map, ArrayMap)
        ]
        if len(array_output_dimensions) != 1:
            raise ValueError(
                "one survivor array requires exactly one orthogonal ArrayMap; "
                f"found output dimensions {array_output_dimensions}"
            )
        survivor_items = ((array_output_dimensions[0], survivors),)

    for output_dimension, positions in survivor_items:
        output_map = original.output[output_dimension]
        if not isinstance(output_map, ArrayMap):
            raise TypeError(
                f"survivors for output dimension {output_dimension} do not describe an ArrayMap"
            )
        input_dimension = array_map_dependent_axis(output_map)
        if input_dimension is None:
            raise ValueError(
                f"output dimension {output_dimension} has no orthogonal input dimension"
            )
        by_input_dimension[input_dimension] = np.asarray(positions, dtype=np.intp)

    output: list[ConstantMap | DimensionMap | ArrayMap] = []
    rank = original.input_rank
    for input_dimension in range(rank):
        positions = by_input_dimension.get(input_dimension)
        if positions is None:
            output.append(DimensionMap(input_dimension=input_dimension))
            continue
        shape = (1,) * input_dimension + (positions.size,) + (1,) * (rank - input_dimension - 1)
        output.append(
            ArrayMap(
                index_array=positions.reshape(shape),
                offset=original.domain.inclusive_min[input_dimension],
            )
        )
    return IndexTransform(domain=restricted.domain, output=tuple(output))


def _correlated_cell_transform(
    original: IndexTransform,
    restricted: IndexTransform,
    survivors: np.ndarray[Any, np.dtype[np.intp]],
) -> IndexTransform:
    """Map compacted correlated points back through the request's row-major domain."""
    positions = np.asarray(survivors, dtype=np.intp)
    # Correlated broadcast axes already contribute positional survivor offsets;
    # residual affine axes still contribute literal coordinates. Remove only
    # the latter origins before unraveling the fully positional flat offsets.
    literal_axes = {
        output_map.input_dimension
        for output_map in original.output
        if isinstance(output_map, DimensionMap)
    }
    origin_offset = 0
    flat_stride = 1
    for input_dimension in range(original.input_rank - 1, -1, -1):
        if input_dimension in literal_axes:
            origin_offset += original.domain.inclusive_min[input_dimension] * flat_stride
        extent = original.domain.shape[input_dimension]
        flat_stride *= extent
    coordinates = np.unravel_index(
        checked_affine(-origin_offset, 1, positions), original.domain.shape
    )
    output = tuple(
        ArrayMap(
            index_array=np.asarray(coordinate, dtype=np.intp),
            offset=origin,
        )
        for coordinate, origin in zip(coordinates, original.domain.inclusive_min, strict=True)
    )
    return IndexTransform(domain=restricted.domain, output=output)


def _cell_transform(
    original: IndexTransform,
    restricted: IndexTransform,
    survivors: _OutIndices,
) -> IndexTransform:
    """Convert private survivor bookkeeping into a direction-neutral transform."""
    if survivors is None:
        return IndexTransform.identity(restricted.domain)
    if index_array_structure(original) == "general":
        if isinstance(survivors, dict):
            raise ValueError("general intersections require one shared survivor array")
        return _correlated_cell_transform(original, restricted, survivors)
    return _orthogonal_cell_transform(original, restricted, survivors)


def _iter_chunk_projections(
    transform: IndexTransform,
    dim_grids: Sequence[DimensionGridLike],
) -> Iterator[ChunkProjection]:
    """Convert private intersection results into public paired projections."""
    for chunk_coords, chunk_transform, survivors in _iter_chunk_transform_results(
        transform, dim_grids
    ):
        chunk_min = tuple(
            grid.chunk_offset(coord) for grid, coord in zip(dim_grids, chunk_coords, strict=True)
        )
        chunk_shape = tuple(
            _data_size(grid, coord) for grid, coord in zip(dim_grids, chunk_coords, strict=True)
        )
        chunk_domain = IndexDomain(
            inclusive_min=chunk_min,
            exclusive_max=tuple(
                origin + extent for origin, extent in zip(chunk_min, chunk_shape, strict=True)
            ),
        )
        cell_transform = _cell_transform(transform, chunk_transform, survivors)
        synthetic_origin = (0,) * chunk_transform.input_rank
        chunk_transform = chunk_transform.translate_domain_to(synthetic_origin)
        cell_transform = cell_transform.translate_domain_to(synthetic_origin)
        if survivors is not None or any(isinstance(m, ArrayMap) for m in chunk_transform.output):
            coverage: ChunkCoverage = "unknown"
        elif _covers_whole_chunk(chunk_transform, chunk_shape):
            coverage = "full"
        else:
            coverage = "partial"
        yield ChunkProjection(
            chunk_coords=chunk_coords,
            chunk_domain=chunk_domain,
            chunk_transform=chunk_transform,
            cell_transform=cell_transform,
            coverage=coverage,
        )
