"""Chunk resolution — mapping transforms to chunk-level I/O.

Given an `IndexTransform` (which coordinates a request reads) and one grid per
storage dimension (how storage is divided into chunks), chunk resolution
answers:

    For each chunk, which storage coordinates does this transform touch,
    and where do those values land in the request?

The public result is a lazy, reusable `ChunkPlan` whose rows are
`ChunkProjection`s. Each identifies a chunk and pairs a chunk-local transform
with a transform back to the request's cells, over one shared zero-origin
cell domain, without assuming NumPy selectors, a codec pipeline, or a
scheduler.

The plan is computed in factored form, the `GridPartition`. Restricting a
transform to a chunk box distributes over output dimensions whenever each
output map reads its own input axis — every basic and orthogonal selection —
so each axis is resolved once against its grid into a table:

- `StridedSet` — a `ConstantMap` or `DimensionMap` axis: one row per touched
  chunk, holding the chunk-local start, the extent, the request position of
  the first cell, and whether the row covers its chunk exactly once.
- `IndexedSet` — an orthogonal `ArrayMap` axis: its coordinates grouped by
  chunk in CSR form, with the request positions they fill.
- `JointSet` — one connected index-array component. Arrays sharing input
  axes are grouped together; independent components remain separate tables.

A projection is one row of each table combined. Building the tables costs the
sum of the touched chunks per axis rather than their product, rows are
materialized only on request, and a consumer may read the tables directly
instead. Two output maps that read one input axis through a `DimensionMap`
(a diagonal, which no selection produces) have no factored form and are
rejected with `ValueError`; a correlated index array varying over an axis a
`DimensionMap` also reads is rejected with `NotImplementedError`.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from zarr_indexing._affine import checked_affine
from zarr_indexing.domain import IndexDomain
from zarr_indexing.output_map import ArrayMap, ConstantMap, DimensionMap, OutputIndexMap
from zarr_indexing.transform import (
    IndexTransform,
    _intersect_dimension_map,  # pyright: ignore[reportPrivateUsage]
    _prepare_correlated,  # pyright: ignore[reportPrivateUsage]
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from zarr_indexing.grid import DimensionGridLike

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

    Construct plans with `plan_chunks`. The plan's factored form, a
    `GridPartition`, is built on first use and memoized; iterating the plan
    or `projections()` materializes fresh `ChunkProjection` rows from it.

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

    # Memoized factored form; derived state, excluded from identity (see
    # `IndexDomain._shape` for the same pattern).
    _partition: GridPartition | None = field(default=None, init=False, repr=False, compare=False)

    def partition(self) -> GridPartition:
        """The plan in factored, columnar form: one table per axis plus connected index-array tables.

        Built once per plan and memoized. Raises `ValueError` if two output
        maps read one input axis through a `DimensionMap` (a diagonal, which
        no selection produces): its tables are per output dimension, and a
        diagonal needs a strided set spanning several.
        """
        cached = self._partition
        if cached is None:
            cached = _partition_transform(self.transform, self.dimension_grids)
            object.__setattr__(self, "_partition", cached)
        return cached

    def projections(self) -> Iterator[ChunkProjection]:
        """Return a fresh iterator over the chunks touched by this plan."""
        return iter(self.partition())

    def __iter__(self) -> Iterator[ChunkProjection]:
        """Equivalent to `projections()`."""
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
        A reusable plan whose projections are computed lazily; its
        `partition()` is the factored form they are derived from.

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


# --------------------------------------------------------------------------- #
# Grid partition: the factored form of a plan
# --------------------------------------------------------------------------- #


def _dimension_map_candidates(
    m: DimensionMap, dim_lo: int, dim_hi: int, dg: DimensionGridLike
) -> Sequence[tuple[int, ...]]:
    """The chunks a nonempty `DimensionMap` over input `[dim_lo, dim_hi)` can touch, ascending."""
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
    point_count = dim_hi - dim_lo
    chunk_count = last - first + 1
    if point_count < chunk_count:
        steps = np.arange(point_count, dtype=np.intp)
        storage = checked_affine(first_storage, m.stride, steps)
        chunk_ids = dg.indices_to_chunks(storage)
        return [(int(c),) for c in np.unique(chunk_ids)]
    return [(c,) for c in range(first, last + 1)]


def _own(column: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """A read-only column that cannot be made writable again.

    A memoized partition must not drift under a consumer. Clearing the
    writeable flag is advisory (`setflags(write=True)` undoes it), so a
    column is re-homed over immutable bytes, the way `ArrayMap` owns its
    index array; a view of such bytes is kept as is. Object columns (exact
    ints beyond ``np.intp``) hold references and cannot live in bytes, so
    they keep the flag only.
    """
    if column.dtype == object:
        column.setflags(write=False)
        return column
    owner: Any = column.base
    while isinstance(owner, np.ndarray):
        owner = owner.base
    if isinstance(owner, bytes) and not column.flags.writeable:
        return column
    contiguous = np.ascontiguousarray(column)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(contiguous.shape)


def _own_columns(table: Any, *names: str) -> None:
    for name in names:
        object.__setattr__(table, name, _own(getattr(table, name)))


def _wide_column(values: Sequence[int]) -> np.ndarray[Any, np.dtype[np.intp]]:
    """An ``intp`` column, or an object column of exact ints when a value does not fit.

    Request-domain extents are unbounded Python ints (a zero-stride map over
    a `(2**63,)` domain is valid and touches one storage cell), so the two
    columns measured along the request axis keep exact integers rather than
    imposing a domain-size limit the algebra does not have.
    """
    try:
        return np.array(values, dtype=np.intp)
    except OverflowError:
        return cast("np.ndarray[Any, np.dtype[np.intp]]", np.array(values, dtype=object))


def _chunk_bounds(
    dg: DimensionGridLike, chunks: np.ndarray[Any, np.dtype[np.intp]]
) -> tuple[np.ndarray[Any, np.dtype[np.intp]], np.ndarray[Any, np.dtype[np.intp]]]:
    """Storage origin and data extent of each chunk, one grid call per touched chunk."""
    n = int(chunks.size)
    starts = np.fromiter((dg.chunk_offset(int(c)) for c in chunks), dtype=np.intp, count=n)
    extents = np.fromiter((_data_size(dg, int(c)) for c in chunks), dtype=np.intp, count=n)
    return starts, extents


@dataclass(frozen=True, slots=True)
class StridedSet:
    """One output dimension read through a `ConstantMap` or `DimensionMap`, one row per chunk.

    Row ``i`` is the map restricted to chunk ``chunk[i]`` and re-based to
    chunk-local, zero-origin coordinates: the chunk-local map is
    `DimensionMap(input_dimension, offset=local_start[i], stride=stride)` over
    ``[0, extent[i])`` (a `ConstantMap(local_start[i])` for a constant), and
    its cells are request positions ``[origin[i], origin[i] + extent[i])``
    along the input axis, counted from the request domain's lower bound.

    Columns are read-only NumPy arrays. `extent` and `origin` are measured
    along the request axis, whose bounds are arbitrary Python ints; they are
    ``intp`` unless a value does not fit, in which case they hold exact ints
    (``dtype=object``).

    Examples
    --------
    >>> from zarr_indexing import IndexTransform, plan_chunks
    >>> from zarr_indexing.grid import dimension_grids_from_chunks
    >>> grids = dimension_grids_from_chunks((4,), shape=(10,))
    >>> (axis,) = plan_chunks(IndexTransform.from_shape((10,))[1:9:2], grids).partition().sets
    >>> axis.chunk.tolist(), axis.local_start.tolist(), axis.extent.tolist(), axis.origin.tolist()
    ([0, 1], [1, 1], [2, 2], [0, 2])
    """

    output_dimension: int
    """The storage axis this table describes."""

    input_dimension: int | None
    """The request axis the map reads, or `None` for a constant."""

    stride: int
    """Storage step per request cell; ``0`` for a constant."""

    chunk: np.ndarray[Any, np.dtype[np.intp]]
    """Chunk index along the axis, one per row, ascending."""

    chunk_start: np.ndarray[Any, np.dtype[np.intp]]
    """Storage origin of each chunk."""

    chunk_extent: np.ndarray[Any, np.dtype[np.intp]]
    """Data extent of each chunk (clipped at the array boundary)."""

    local_start: np.ndarray[Any, np.dtype[np.intp]]
    """Chunk-local storage coordinate of the row's first cell."""

    extent: np.ndarray[Any, np.dtype[np.intp]]
    """Cells the row selects along the request axis (``1`` for a constant)."""

    origin: np.ndarray[Any, np.dtype[np.intp]]
    """Position of the row's first cell along the request axis (``0`` for a constant)."""

    full: np.ndarray[Any, np.dtype[np.bool_]]
    """Whether the row covers its chunk's data extent exactly once (in either direction)."""

    def __post_init__(self) -> None:
        _own_columns(
            self, "chunk", "chunk_start", "chunk_extent", "local_start", "extent", "origin", "full"
        )

    def __len__(self) -> int:
        return int(self.chunk.size)


@dataclass(frozen=True, slots=True)
class IndexedSet:
    """One output dimension read through an orthogonal `ArrayMap`, one row per chunk.

    The map's coordinates are grouped by chunk in CSR form: row ``i`` owns
    ``index[pointer[i]:pointer[i + 1]]`` (the index-array values, in request
    order) and ``positions[pointer[i]:pointer[i + 1]]`` (their positions along
    the request axis, ascending). `local` gives the same values as chunk-local
    storage coordinates. Columns are read-only NumPy arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from zarr_indexing import IndexTransform, plan_chunks
    >>> from zarr_indexing.grid import dimension_grids_from_chunks
    >>> grids = dimension_grids_from_chunks((4,), shape=(10,))
    >>> transform = IndexTransform.from_shape((10,)).oindex[np.array([9, 1, 2, 8])]
    >>> (axis,) = plan_chunks(transform, grids).partition().sets
    >>> axis.chunk.tolist(), axis.pointer.tolist()
    ([0, 2], [0, 2, 4])
    >>> axis.local.tolist(), axis.positions.tolist()
    ([1, 2, 1, 0], [1, 2, 0, 3])
    """

    output_dimension: int
    """The storage axis this table describes."""

    input_dimension: int
    """The request axis the index array varies over."""

    offset: int
    """The map's affine offset: storage is ``offset + stride * index``."""

    stride: int
    """The map's affine stride."""

    chunk: np.ndarray[Any, np.dtype[np.intp]]
    """Chunk index along the axis, one per row, ascending."""

    chunk_start: np.ndarray[Any, np.dtype[np.intp]]
    """Storage origin of each chunk."""

    chunk_extent: np.ndarray[Any, np.dtype[np.intp]]
    """Data extent of each chunk."""

    pointer: np.ndarray[Any, np.dtype[np.intp]]
    """CSR row pointer: row ``i`` owns entries ``pointer[i]`` to ``pointer[i + 1]``."""

    index: np.ndarray[Any, np.dtype[np.intp]]
    """Index-array values grouped by chunk."""

    positions: np.ndarray[Any, np.dtype[np.intp]]
    """Positions along the request axis, grouped by chunk, ascending within a row."""

    def __post_init__(self) -> None:
        _own_columns(self, "chunk", "chunk_start", "chunk_extent", "pointer", "index", "positions")

    def __len__(self) -> int:
        return int(self.chunk.size)

    @property
    def counts(self) -> np.ndarray[Any, np.dtype[np.intp]]:
        """Entries per row."""
        return np.diff(self.pointer)

    _local: np.ndarray[Any, np.dtype[np.intp]] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @property
    def local(self) -> np.ndarray[Any, np.dtype[np.intp]]:
        """Memoized, read-only chunk-local coordinates, grouped like `index`."""
        cached = self._local
        if cached is None:
            storage = checked_affine(self.offset, self.stride, self.index)
            cached = storage - np.repeat(self.chunk_start, self.counts)
            cached = _own(cached)
            object.__setattr__(self, "_local", cached)
        return cached

    def run(self, row: int) -> slice:
        """The slice of `index` / `positions` a row owns."""
        return slice(int(self.pointer[row]), int(self.pointer[row + 1]))


@dataclass(frozen=True, slots=True)
class JointSet:
    """One connected component of index arrays, grouped by the chunk each point lands in.

    Arrays connected by shared input axes, possibly transitively, constrain
    each other and are sorted into chunks together.
    Row ``i`` is one touched chunk, `chunk[i]` its coordinates on the
    `output_dimensions`, and CSR range ``pointer[i]:pointer[i + 1]`` its
    points: `index` holds their index-array values per output dimension,
    `positions` their flat positions in this component's input block, and
    `block_coordinates` those positions unravelled over the block. Columns are
    read-only NumPy arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from zarr_indexing import IndexTransform, plan_chunks
    >>> from zarr_indexing.grid import dimension_grids_from_chunks
    >>> grids = dimension_grids_from_chunks((3, 4), shape=(7, 9))
    >>> transform = IndexTransform.from_shape((7, 9)).vindex[
    ...     np.array([0, 6, 6, 1]), np.array([8, 0, 1, 8])
    ... ]
    >>> (joint,) = plan_chunks(transform, grids).partition().joint_sets
    >>> joint.chunk.tolist(), joint.pointer.tolist(), joint.positions.tolist()
    ([[0, 2], [2, 0]], [0, 2, 4], [0, 3, 1, 2])
    """

    output_dimensions: tuple[int, ...]
    """The storage axes read by correlated index arrays."""

    offsets: tuple[int, ...]
    """Affine offset of each array's map, aligned with `output_dimensions`."""

    strides: tuple[int, ...]
    """Affine stride of each array's map."""

    broadcast_axes: tuple[int, ...]
    """The request axes the arrays broadcast over."""

    broadcast_shape: tuple[int, ...]
    """The extent of those axes."""

    chunk: np.ndarray[Any, np.dtype[np.intp]]
    """Chunk coordinates on `output_dimensions`, shape ``(rows, k)``, lexicographic."""

    chunk_start: np.ndarray[Any, np.dtype[np.intp]]
    """Storage origin of each chunk on `output_dimensions`, shape ``(rows, k)``."""

    chunk_extent: np.ndarray[Any, np.dtype[np.intp]]
    """Data extent of each chunk on `output_dimensions`, shape ``(rows, k)``."""

    pointer: np.ndarray[Any, np.dtype[np.intp]]
    """CSR row pointer into `index`, `positions` and `block_coordinates`."""

    index: np.ndarray[Any, np.dtype[np.intp]]
    """Index-array values per point and output dimension, shape ``(points, k)``."""

    positions: np.ndarray[Any, np.dtype[np.intp]]
    """Flat block position of each point, ascending within a row."""

    _block_coordinates: np.ndarray[Any, np.dtype[np.intp]] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        _own_columns(self, "chunk", "chunk_start", "chunk_extent", "pointer", "index", "positions")

    def __len__(self) -> int:
        return int(self.chunk.shape[0])

    @property
    def counts(self) -> np.ndarray[Any, np.dtype[np.intp]]:
        """Points per row."""
        return np.diff(self.pointer)

    _local: np.ndarray[Any, np.dtype[np.intp]] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @property
    def local(self) -> np.ndarray[Any, np.dtype[np.intp]]:
        """Memoized, read-only chunk-local coordinates, shape ``(points, k)``."""
        cached = self._local
        if cached is None:
            storage = np.stack(
                [
                    checked_affine(offset, stride, self.index[:, column])
                    for column, (offset, stride) in enumerate(
                        zip(self.offsets, self.strides, strict=True)
                    )
                ],
                axis=1,
            )
            cached = storage - np.repeat(self.chunk_start, self.counts, axis=0)
            cached = _own(cached)
            object.__setattr__(self, "_local", cached)
        return cached

    @property
    def block_coordinates(self) -> np.ndarray[Any, np.dtype[np.intp]]:
        """Memoized `positions` unravelled over `broadcast_shape`, shape ``(points, len(broadcast_axes))``."""
        cached = self._block_coordinates
        if cached is None:
            if self.broadcast_shape:
                cached = np.stack(np.unravel_index(self.positions, self.broadcast_shape), axis=1)
            else:
                cached = np.empty((self.positions.size, 0), dtype=np.intp)
            cached = _own(np.asarray(cached, dtype=np.intp))
            object.__setattr__(self, "_block_coordinates", cached)
        return cached

    def run(self, row: int) -> slice:
        """The slice of the point arrays a row owns."""
        return slice(int(self.pointer[row]), int(self.pointer[row + 1]))


def _strided_set(
    transform: IndexTransform, out_dim: int, m: ConstantMap | DimensionMap, dg: DimensionGridLike
) -> StridedSet:
    domain = transform.domain
    if isinstance(m, ConstantMap):
        c = dg.index_to_chunk(checked_affine(m.offset, 0, 0))
        chunks = np.array([c], dtype=np.intp)
        starts, extents = _chunk_bounds(dg, chunks)
        local = np.array([m.offset - int(starts[0])], dtype=np.intp)
        return StridedSet(
            output_dimension=out_dim,
            input_dimension=None,
            stride=0,
            chunk=chunks,
            chunk_start=starts,
            chunk_extent=extents,
            local_start=local,
            extent=np.ones(1, dtype=np.intp),
            origin=np.zeros(1, dtype=np.intp),
            full=(extents == 1) & (local == 0),
        )
    k = m.input_dimension
    lo = domain.inclusive_min[k]
    hi = domain.exclusive_max[k]
    stride = m.stride
    unit = abs(stride) == 1
    rows: list[tuple[int, int, int, int, int, int, bool]] = []
    # Exact Python-int arithmetic per touched chunk: the request's literal
    # coordinates can exceed np.intp before cancellation, and the number of
    # touched chunks along one axis is a sum, not a product. Chunk-local
    # columns fit np.intp; `extent` and `origin` scale with the request
    # domain, which has arbitrary Python-int bounds, so they keep exact ints
    # when they must (`_wide_column`).
    for (c,) in _dimension_map_candidates(m, lo, hi, dg):
        c_start = dg.chunk_offset(c)
        c_extent = _data_size(dg, c)
        narrowed = _intersect_dimension_map(m, lo, hi, c_start, c_start + c_extent)
        if narrowed is None:
            continue
        nlo, nhi = narrowed
        extent = nhi - nlo
        # Chunk-local, then re-based to a zero-origin input axis.
        local_start = m.offset - c_start + stride * nlo
        if unit or (extent == 1 and c_extent == 1):
            last = local_start + stride * (extent - 1)
            full = min(local_start, last) == 0 and max(local_start, last) == c_extent - 1
        else:
            full = False
        rows.append((c, c_start, c_extent, local_start, extent, nlo - lo, full))
    columns = list(zip(*rows, strict=True)) if rows else [()] * 7
    return StridedSet(
        output_dimension=out_dim,
        input_dimension=k,
        stride=stride,
        chunk=np.array(columns[0], dtype=np.intp),
        chunk_start=np.array(columns[1], dtype=np.intp),
        chunk_extent=np.array(columns[2], dtype=np.intp),
        local_start=np.array(columns[3], dtype=np.intp),
        extent=_wide_column(columns[4]),
        origin=_wide_column(columns[5]),
        full=np.array(columns[6], dtype=np.bool_),
    )


def _indexed_set(out_dim: int, m: ArrayMap, dg: DimensionGridLike) -> IndexedSet:
    dependent = m.dependent_axis
    assert dependent is not None
    flat = m.index_array.reshape(-1)
    n = int(flat.size)
    storage = checked_affine(m.offset, m.stride, flat)
    # Probe the extreme coordinates with the scalar lookup first: a grid's
    # scalar error names the offending coordinate, where the vectorized one
    # reports a range.
    dg.index_to_chunk(int(storage.min()))
    dg.index_to_chunk(int(storage.max()))
    chunk_ids = dg.indices_to_chunks(storage)
    # Already-sorted coordinates (the common case) need no sort.
    if bool((chunk_ids[1:] >= chunk_ids[:-1]).all()):
        positions = np.arange(n, dtype=np.intp)
        sorted_ids = chunk_ids
        index = flat
    else:
        positions = np.argsort(chunk_ids, kind="stable")
        sorted_ids = chunk_ids[positions]
        index = flat[positions]
    boundaries = np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]) + 1
    pointer = np.concatenate([[0], boundaries, [n]]).astype(np.intp)
    chunks = np.asarray(sorted_ids[pointer[:-1]], dtype=np.intp)
    starts, extents = _chunk_bounds(dg, chunks)
    return IndexedSet(
        output_dimension=out_dim,
        input_dimension=dependent,
        offset=m.offset,
        stride=m.stride,
        chunk=chunks,
        chunk_start=starts,
        chunk_extent=extents,
        pointer=pointer,
        index=np.asarray(index, dtype=np.intp),
        positions=np.asarray(positions, dtype=np.intp),
    )


def _chunk_keys(
    chunk_ids: Sequence[np.ndarray[Any, np.dtype[np.intp]]],
) -> np.ndarray[Any, np.dtype[np.intp]]:
    """One sortable key per point whose order is the lexicographic chunk order."""
    n = int(chunk_ids[0].size)
    if len(chunk_ids) == 1:
        return np.asarray(chunk_ids[0], dtype=np.intp)
    # Mixed-radix key with the first dimension most significant.
    keys = np.zeros(n, dtype=np.intp)
    multiplier = 1
    for ids in reversed(chunk_ids):
        radix = int(ids.max()) + 1
        if multiplier * radix >= 2**62:
            stacked = np.stack([np.asarray(i, dtype=np.intp).ravel() for i in chunk_ids], axis=1)
            _, inverse = np.unique(stacked, axis=0, return_inverse=True)
            return np.asarray(inverse, dtype=np.intp).reshape(-1)
        keys += np.asarray(ids, dtype=np.intp) * multiplier
        multiplier *= radix
    return keys


def _joint_set(
    transform: IndexTransform,
    dim_grids: Sequence[DimensionGridLike],
    *,
    output_dimensions: tuple[int, ...] | None = None,
    input_dimensions: tuple[int, ...] | None = None,
) -> JointSet:
    block = _prepare_correlated(
        transform, output_dimensions=output_dimensions, input_dimensions=input_dimensions
    )
    dims = block.correlated_dims
    # Probe the extreme coordinates with the scalar lookup first: a grid's
    # vectorized lookup need not check bounds (zarr's does not), and its
    # scalar error names the offending coordinate.
    for d in dims:
        storage = block.flat_storage[d]
        dim_grids[d].index_to_chunk(int(storage.min()))
        dim_grids[d].index_to_chunk(int(storage.max()))
    chunk_ids = [dim_grids[d].indices_to_chunks(block.flat_storage[d]) for d in dims]
    n_points = int(chunk_ids[0].size)
    keys = _chunk_keys(chunk_ids)
    positions = np.argsort(keys, kind="stable")
    sorted_keys = keys[positions]
    boundaries = np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1
    pointer = np.concatenate([[0], boundaries, [n_points]]).astype(np.intp)
    first = positions[pointer[:-1]]
    chunk = np.stack([np.asarray(ids, dtype=np.intp)[first] for ids in chunk_ids], axis=1)
    index = np.stack([block.flat_index[d][positions] for d in dims], axis=1)
    starts = np.empty_like(chunk)
    extents = np.empty_like(chunk)
    for column, d in enumerate(dims):
        starts[:, column], extents[:, column] = _chunk_bounds(dim_grids[d], chunk[:, column])
    maps = [cast("ArrayMap", transform.output[d]) for d in dims]
    return JointSet(
        output_dimensions=dims,
        offsets=tuple(m.offset for m in maps),
        strides=tuple(m.stride for m in maps),
        broadcast_axes=block.broadcast_axes,
        broadcast_shape=block.broadcast_shape,
        chunk=chunk,
        chunk_start=starts,
        chunk_extent=extents,
        pointer=pointer,
        index=index,
        positions=np.asarray(positions, dtype=np.intp),
    )


# One table row's contribution to a projection: (chunk index, chunk start,
# chunk data extent, bound input axis or None, restricted extent of that axis,
# the rebased chunk-local map, the cell map for the axis, whole-chunk cover).
_AxisPiece = tuple[int, int, int, int | None, int, OutputIndexMap, OutputIndexMap | None, bool]


def _strided_piece(
    axis: StridedSet, row: int, origin: int, input_dimension: int | None = None
) -> _AxisPiece:
    """One row of a strided table as maps; ``origin`` is the request axis's lower bound.

    ``input_dimension`` renumbers the chunk-local map's axis for the
    correlated walk, whose restricted domain collapses the broadcast axes
    into one leading axis.
    """
    offset = int(axis.local_start[row])
    k = axis.input_dimension
    chunk_map: OutputIndexMap
    cell_map: OutputIndexMap | None
    if k is None:
        chunk_map = ConstantMap(offset=offset)
        cell_map = None
    else:
        chunk_map = DimensionMap(
            input_dimension=k if input_dimension is None else input_dimension,
            offset=offset,
            stride=axis.stride,
        )
        cell_map = DimensionMap(input_dimension=k, offset=origin + int(axis.origin[row]))
    return (
        int(axis.chunk[row]),
        int(axis.chunk_start[row]),
        int(axis.chunk_extent[row]),
        k,
        int(axis.extent[row]),
        chunk_map,
        cell_map,
        bool(axis.full[row]),
    )


def _indexed_piece(axis: IndexedSet, row: int, rank: int, origin: int) -> _AxisPiece:
    run = axis.run(row)
    k = axis.input_dimension
    count = run.stop - run.start
    shape = (1,) * k + (count,) + (1,) * (rank - k - 1)
    c_start = int(axis.chunk_start[row])
    return (
        int(axis.chunk[row]),
        c_start,
        int(axis.chunk_extent[row]),
        k,
        count,
        ArrayMap(
            index_array=axis.index[run].reshape(shape),
            offset=axis.offset - c_start,
            stride=axis.stride,
        ),
        ArrayMap(index_array=axis.positions[run].reshape(shape), offset=origin),
        False,
    )


@dataclass(frozen=True, slots=True)
class GridPartition:
    """A plan in factored form: per-axis tables whose product is the chunk walk.

    `sets` holds one `StridedSet` or `IndexedSet` per output dimension the
    transform reads independently, in output-dimension order; `joint_sets`
    holds the connected index-array components. A projection is one row of each
    table, so the partition has ``prod(row_shape)`` rows, walked
    in row-major order over `row_shape` (the component tables after `sets`). Rows are materialized into
    `ChunkProjection` objects only on request; a vectorized consumer can read
    the tables directly.

    Take one from `ChunkPlan.partition`.

    Examples
    --------
    `arr[1:6:2, 5:]` on a `(7, 9)` array with `(3, 4)` chunks touches two
    chunks along each axis, so the partition has four rows:

    >>> from zarr_indexing import IndexTransform, plan_chunks
    >>> from zarr_indexing.grid import dimension_grids_from_chunks
    >>> grids = dimension_grids_from_chunks((3, 4), shape=(7, 9))
    >>> partition = plan_chunks(IndexTransform.from_shape((7, 9))[1:6:2, 5:], grids).partition()
    >>> partition.row_shape, len(partition)
    ((2, 2), 4)
    >>> partition.chunk_coords().tolist()
    [[0, 1], [0, 2], [1, 1], [1, 2]]
    >>> [projection.chunk_transform.selection_repr for projection in partition][3]
    '{ [0, 4) step 2, [0, 1) }'
    """

    transform: IndexTransform
    """The transform this partition factors."""

    dimension_grids: tuple[DimensionGridLike, ...]
    """One grid per storage dimension."""

    sets: tuple[StridedSet | IndexedSet, ...]
    """Independent per-axis tables, in output-dimension order."""

    joint_sets: tuple[JointSet, ...]
    """Connected index-array components, ordered by their first output dimension."""

    row_shape: tuple[int, ...]
    """Rows per table: `sets` followed by `joint_sets`, in row-major iteration order."""

    def __len__(self) -> int:
        return math.prod(self.row_shape)

    def chunk_coords(self) -> np.ndarray[Any, np.dtype[np.intp]]:
        """Chunk coordinates of every row, shape ``(len(self), output rank)``, without materializing rows."""
        count = len(self)
        out = np.empty((count, self.transform.output_rank), dtype=np.intp)
        if count == 0 or not self.row_shape:
            return out
        indices = np.unravel_index(np.arange(count, dtype=np.intp), self.row_shape)
        for axis, rows in zip(self.sets, indices, strict=False):
            out[:, axis.output_dimension] = axis.chunk[rows]
        for joint, rows in zip(self.joint_sets, indices[len(self.sets) :], strict=True):
            out[:, list(joint.output_dimensions)] = joint.chunk[rows]
        return out

    def __iter__(self) -> Iterator[ChunkProjection]:
        """Materialize every row in order."""
        if math.prod(self.row_shape) == 0:
            return
        if not self.joint_sets:
            yield from self._iter_factorized()
        else:
            yield from self._iter_correlated()

    # -- factorized assembly ------------------------------------------------

    def _pieces(self, axis: StridedSet | IndexedSet, row: int) -> _AxisPiece:
        domain = self.transform.domain
        if isinstance(axis, StridedSet):
            origin = (
                0 if axis.input_dimension is None else domain.inclusive_min[axis.input_dimension]
            )
            return _strided_piece(axis, row, origin)
        return _indexed_piece(axis, row, domain.ndim, domain.inclusive_min[axis.input_dimension])

    def _unbound(self) -> tuple[list[OutputIndexMap | None], bool]:
        """Cell maps for request axes no output reads, and whether they permit full coverage.

        A whole-chunk cover must biject onto the chunk, so an unread axis can
        only be a singleton.
        """
        domain = self.transform.domain
        bound = {axis.input_dimension for axis in self.sets if axis.input_dimension is not None}
        cell_maps: list[OutputIndexMap | None] = [None] * domain.ndim
        unbound_ok = True
        for axis in range(domain.ndim):
            if axis not in bound:
                cell_maps[axis] = DimensionMap(
                    input_dimension=axis, offset=domain.inclusive_min[axis]
                )
                unbound_ok = unbound_ok and domain.shape[axis] <= 1
        return cell_maps, unbound_ok

    def _iter_factorized(self) -> Iterator[ChunkProjection]:
        pieces_per_set = [
            [self._pieces(axis, row) for row in range(len(axis))] for axis in self.sets
        ]
        base_cell_maps, unbound_ok = self._unbound()
        for combo in itertools.product(*pieces_per_set):
            yield self._factorized_projection(list(combo), base_cell_maps, unbound_ok)

    def _factorized_projection(
        self,
        pieces: list[_AxisPiece],
        base_cell_maps: list[OutputIndexMap | None],
        unbound_ok: bool,
    ) -> ChunkProjection:
        domain = self.transform.domain
        shape = list(domain.shape)
        cell_maps = list(base_cell_maps)
        chunk_maps: list[OutputIndexMap] = []
        chunk_coords: list[int] = []
        chunk_min: list[int] = []
        chunk_max: list[int] = []
        has_array = False
        full = unbound_ok
        for c, c_start, c_extent, k, extent, chunk_map, cell_map, piece_full in pieces:
            chunk_coords.append(c)
            chunk_min.append(c_start)
            chunk_max.append(c_start + c_extent)
            chunk_maps.append(chunk_map)
            if isinstance(chunk_map, ArrayMap):
                has_array = True
            if k is not None:
                shape[k] = extent
                cell_maps[k] = cell_map
            full = full and piece_full
        synthetic = IndexDomain._unchecked((0,) * domain.ndim, tuple(shape))  # pyright: ignore[reportPrivateUsage]
        if has_array:
            coverage: ChunkCoverage = "unknown"
        elif full:
            coverage = "full"
        else:
            coverage = "partial"
        return ChunkProjection(
            chunk_coords=tuple(chunk_coords),
            chunk_domain=IndexDomain._unchecked(tuple(chunk_min), tuple(chunk_max)),  # pyright: ignore[reportPrivateUsage]
            chunk_transform=IndexTransform._unchecked(synthetic, tuple(chunk_maps)),  # pyright: ignore[reportPrivateUsage]
            cell_transform=IndexTransform._unchecked(  # pyright: ignore[reportPrivateUsage]
                synthetic, tuple(cast("list[OutputIndexMap]", cell_maps))
            ),
            coverage=coverage,
        )

    # -- correlated assembly ------------------------------------------------

    def _slots(self) -> tuple[tuple[int | None, ...], dict[int, int]]:
        """One synthetic axis per nonconstant component, followed by residual axes."""
        component_slots: list[int | None] = []
        bound: set[int] = set()
        n_lead = 0
        for joint in self.joint_sets:
            bound.update(joint.broadcast_axes)
            component_slots.append(n_lead if joint.broadcast_axes else None)
            n_lead += bool(joint.broadcast_axes)
        residual_axes = [axis for axis in range(self.transform.input_rank) if axis not in bound]
        return tuple(component_slots), {
            axis: n_lead + slot for slot, axis in enumerate(residual_axes)
        }

    def _iter_correlated(self) -> Iterator[ChunkProjection]:
        component_slots, slot_of = self._slots()
        lo_all = self.transform.domain.inclusive_min
        pieces_per_set = [
            [
                _strided_piece(
                    cast("StridedSet", axis),
                    row,
                    0 if axis.input_dimension is None else lo_all[axis.input_dimension],
                    None if axis.input_dimension is None else slot_of[axis.input_dimension],
                )
                for row in range(len(axis))
            ]
            for axis in self.sets
        ]
        for residual in itertools.product(*pieces_per_set):
            for rows in itertools.product(*(range(len(joint)) for joint in self.joint_sets)):
                yield self._correlated_projection(list(residual), rows, component_slots, slot_of)

    def _correlated_projection(
        self,
        residual: list[_AxisPiece],
        rows: tuple[int, ...],
        component_slots: tuple[int | None, ...],
        slot_of: dict[int, int],
    ) -> ChunkProjection:
        transform = self.transform
        domain = transform.domain
        lo_all = domain.inclusive_min
        output_rank = transform.output_rank
        runs = tuple(joint.run(row) for joint, row in zip(self.joint_sets, rows, strict=True))
        points_shape = tuple(
            run.stop - run.start
            for run, slot in zip(runs, component_slots, strict=True)
            if slot is not None
        )
        n_lead = len(points_shape)
        rank = n_lead + len(slot_of)
        chunk_coords = [0] * output_rank
        chunk_min = [0] * output_rank
        chunk_max = [0] * output_rank
        chunk_maps: list[OutputIndexMap | None] = [None] * output_rank
        cell_maps: list[OutputIndexMap | None] = [None] * domain.ndim
        # Unread input axes are independent repetitions, not part of an index
        # array component. Retain their extent without expanding any arrays.
        extents = [domain.shape[axis] for axis in slot_of]
        for axis, slot in slot_of.items():
            cell_maps[axis] = DimensionMap(input_dimension=slot, offset=lo_all[axis])
        for out_dim, (c, c_start, c_extent, k, extent, chunk_map, cell_map, _full) in zip(
            (axis.output_dimension for axis in self.sets), residual, strict=True
        ):
            chunk_coords[out_dim] = c
            chunk_min[out_dim] = c_start
            chunk_max[out_dim] = c_start + c_extent
            chunk_maps[out_dim] = chunk_map
            if k is not None:
                slot = slot_of[k]
                extents[slot - n_lead] = extent
                cell_maps[k] = DimensionMap(
                    input_dimension=slot, offset=cast("DimensionMap", cell_map).offset
                )
        for joint, row, run, component_slot in zip(
            self.joint_sets, rows, runs, component_slots, strict=True
        ):
            component_shape = [1] * rank
            if component_slot is not None:
                component_shape[component_slot] = run.stop - run.start
            for column, out_dim in enumerate(joint.output_dimensions):
                c_start = int(joint.chunk_start[row, column])
                chunk_coords[out_dim] = int(joint.chunk[row, column])
                chunk_min[out_dim] = c_start
                chunk_max[out_dim] = c_start + int(joint.chunk_extent[row, column])
                chunk_maps[out_dim] = ArrayMap(
                    index_array=joint.index[run, column].reshape(component_shape),
                    offset=joint.offsets[column] - c_start,
                    stride=joint.strides[column],
                )
            for column, axis in enumerate(joint.broadcast_axes):
                cell_maps[axis] = ArrayMap(
                    index_array=joint.block_coordinates[run, column].reshape(component_shape),
                    offset=lo_all[axis],
                )
        synthetic = IndexDomain._unchecked((0,) * rank, points_shape + tuple(extents))  # pyright: ignore[reportPrivateUsage]
        return ChunkProjection(
            chunk_coords=tuple(chunk_coords),
            chunk_domain=IndexDomain._unchecked(tuple(chunk_min), tuple(chunk_max)),  # pyright: ignore[reportPrivateUsage]
            chunk_transform=IndexTransform._unchecked(  # pyright: ignore[reportPrivateUsage]
                synthetic, tuple(cast("list[OutputIndexMap]", chunk_maps))
            ),
            cell_transform=IndexTransform._unchecked(  # pyright: ignore[reportPrivateUsage]
                synthetic, tuple(cast("list[OutputIndexMap]", cell_maps))
            ),
            coverage="unknown",
        )


def _component_joint_sets(
    transform: IndexTransform, grids: tuple[DimensionGridLike, ...]
) -> tuple[JointSet, ...]:
    """Partition the input/index-array dependency graph into connected components."""
    affine_axes = {m.input_dimension for m in transform.output if isinstance(m, DimensionMap)}
    # Accumulated components have disjoint input axes. A new map can bridge
    # several of them; merging every overlap maintains that invariant.
    components: list[tuple[set[int], list[int]]] = []
    for dimension, m in enumerate(transform.output):
        if not isinstance(m, ArrayMap):
            continue
        axes = set(m.dependency_axes)
        if axes.intersection(affine_axes):
            raise NotImplementedError(
                "index array varies over an input dimension also bound by a slice map"
            )
        dimensions = [dimension]
        separate = []
        for other_axes, other_dimensions in components:
            if axes.intersection(other_axes):
                axes.update(other_axes)
                dimensions.extend(other_dimensions)
            else:
                separate.append((other_axes, other_dimensions))
        separate.append((axes, sorted(dimensions)))
        components = separate
    return tuple(
        _joint_set(
            transform,
            grids,
            output_dimensions=tuple(dimensions),
            input_dimensions=tuple(sorted(axes)),
        )
        for axes, dimensions in sorted(components, key=lambda component: component[1][0])
    )


def _shared_input_axis(transform: IndexTransform) -> int | None:
    """An input axis read by two output maps in a way that does not factor, if any.

    Two `DimensionMap`s on one axis, or a `DimensionMap` and an orthogonal
    `ArrayMap` on one axis, make a diagonal: neither restricts independently
    of the other. Correlated `ArrayMap`s sharing axes are the `JointSet` case
    and factor fine; an `ArrayMap` varying over a `DimensionMap`'s axis is
    rejected separately by `_prepare_correlated`.
    """
    seen: set[int] = set()
    correlated = transform.index_array_structure == "general"
    for m in transform.output:
        if isinstance(m, DimensionMap):
            axis: int | None = m.input_dimension
        elif isinstance(m, ArrayMap) and not correlated:
            axis = m.dependent_axis
        else:
            continue
        if axis is None:
            continue
        if axis in seen:
            return axis
        seen.add(axis)
    return None


def _partition_transform(
    transform: IndexTransform, grids: tuple[DimensionGridLike, ...]
) -> GridPartition:
    """Factor a transform over its grids; `ChunkPlan.partition` is the public entry."""
    shared = _shared_input_axis(transform)
    if shared is not None:
        raise ValueError(
            f"two output maps read input axis {shared} (a diagonal, which no selection "
            "produces); it has no per-output-axis factored form"
        )
    if any(size == 0 for size in transform.domain.shape):
        return GridPartition(
            transform=transform, dimension_grids=grids, sets=(), joint_sets=(), row_shape=(0,)
        )
    correlated = transform.index_array_structure == "general"
    sets: list[StridedSet | IndexedSet] = []
    for out_dim, m in enumerate(transform.output):
        if isinstance(m, ArrayMap):
            if correlated:
                continue
            sets.append(_indexed_set(out_dim, m, grids[out_dim]))
        else:
            sets.append(_strided_set(transform, out_dim, m, grids[out_dim]))
    joint_sets = _component_joint_sets(transform, grids) if correlated else ()
    row_shape = tuple(len(axis) for axis in sets) + tuple(len(joint) for joint in joint_sets)
    return GridPartition(
        transform=transform,
        dimension_grids=grids,
        sets=tuple(sets),
        joint_sets=joint_sets,
        row_shape=row_shape,
    )
