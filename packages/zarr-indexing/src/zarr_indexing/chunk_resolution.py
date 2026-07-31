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

4. **Yield** — produce `(chunk_coords, local_transform, surviving_indices)`
   triples that the codec pipeline consumes.

Sorted one-dimensional correlated array maps can be partitioned directly
because every touched chunk owns a contiguous slice of the index array. That
case bypasses candidate enumeration and repeated intersection.

`sub_transform_to_selections` bridges from the transform representation
back to the raw `(chunk_selection, out_selection, drop_axes)` tuples that
the current codec pipeline expects. This bridge will go away when the codec
pipeline accepts transforms natively.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from zarr_indexing.domain import IndexDomain
from zarr_indexing.output_map import ArrayMap, ConstantMap, DimensionMap
from zarr_indexing.transform import (
    IndexTransform,
    _positional_slice,  # pyright: ignore[reportPrivateUsage]
)

# `_positional_slice` is a leading-underscore helper in `transform.py`, shared
# with this module on purpose: the slice an `ArrayMap` axis is reindexed by and
# the slice a `DimensionMap` is lowered to are the same walk, and the two must
# agree on where a downward one stops. See `json.py` for the same suppression
# and the same open question about promoting these helpers out of "private".

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from zarr_indexing.grid import DimensionGridLike

OutIndices = (
    dict[int, np.ndarray[Any, np.dtype[np.intp]]] | np.ndarray[Any, np.dtype[np.intp]] | None
)

ChunkTransformResult = tuple[
    tuple[int, ...],
    IndexTransform,
    OutIndices,
]


def _one_dimensional_correlated_array_map(
    transform: IndexTransform,
) -> tuple[ArrayMap, np.ndarray[Any, np.dtype[np.intp]]] | None:
    """Return a nonempty correlated 1-D ArrayMap and its storage coordinates.

    A one-dimensional array selection has no cross-dimensional correlation to
    preserve. The computed storage coordinates are also reused by general
    resolution when they are unsorted.
    """
    if transform.input_rank != 1 or transform.output_rank != 1:
        return None

    m = transform.output[0]
    if (
        not isinstance(m, ArrayMap)
        or m.input_dimension is not None
        or m.index_array.ndim != 1
        or m.index_array.size == 0
    ):
        return None

    return m, m.offset + m.stride * m.index_array


def _iter_sorted_1d_array_map(
    m: ArrayMap,
    storage: np.ndarray[Any, np.dtype[np.intp]],
    dim_grid: DimensionGridLike,
) -> Iterator[ChunkTransformResult]:
    """Resolve a sorted 1-D ArrayMap one touched chunk at a time."""
    start = 0
    while start < storage.size:
        chunk = dim_grid.index_to_chunk(int(storage[start]))
        chunk_start = dim_grid.chunk_offset(chunk)
        chunk_stop = chunk_start + dim_grid.chunk_size(chunk)
        stop = int(np.searchsorted(storage, chunk_stop, side="left"))

        restricted = IndexTransform(
            domain=IndexDomain(inclusive_min=(0,), exclusive_max=(stop - start,)),
            output=(
                ArrayMap(
                    index_array=m.index_array[start:stop],
                    offset=m.offset,
                    stride=m.stride,
                    input_dimension=m.input_dimension,
                ),
            ),
        )
        local = restricted.translate((-chunk_start,))
        surviving = np.arange(start, stop, dtype=np.intp)

        yield (chunk,), local, surviving
        start = stop


def iter_chunk_transforms(
    transform: IndexTransform,
    dim_grids: Sequence[DimensionGridLike],
) -> Iterator[ChunkTransformResult]:
    """Resolve a composed IndexTransform against per-dimension chunk grids.

    `dim_grids` holds one `DimensionGridLike` per output (storage) dimension —
    for zarr this is the chunk grid's per-dimension sequence. Yields
    `(chunk_coords, sub_transform, out_indices)` triples:

    - `chunk_coords`: which chunk to access.
    - `sub_transform`: maps output buffer coords to chunk-local coords.
    - `out_indices`: for vectorized/array indexing, the output scatter
      indices (integer array). `None` for basic/slice indexing.
    """

    if any(size == 0 for size in transform.domain.shape):
        # An empty view touches no chunk. Checked on the domain rather than on
        # the index arrays: an axis of genuine extent 1 is stored as a broadcast
        # singleton, so a slice that empties the domain does not shrink the
        # array, and the emptiness shows only here.
        return

    array_map_1d = _one_dimensional_correlated_array_map(transform)
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
    correlated_dims: list[int] = []
    correlated_chunk_ids: list[np.ndarray[Any, np.dtype[np.intp]]] = []
    slot_dims: list[tuple[int, ...]] = []
    slot_candidates: list[Sequence[tuple[int, ...]]] = []
    for out_dim, m in enumerate(transform.output):
        dg = dim_grids[out_dim]
        if isinstance(m, ConstantMap):
            # Single chunk
            c = dg.index_to_chunk(m.offset)
            slot_dims.append((out_dim,))
            slot_candidates.append(((c,),))
        elif isinstance(m, DimensionMap):
            d = m.input_dimension
            dim_lo = transform.domain.inclusive_min[d]
            dim_hi = transform.domain.exclusive_max[d]
            if dim_lo >= dim_hi:
                return  # empty domain
            if m.stride > 0:
                s_min = m.offset + m.stride * dim_lo
                s_max = m.offset + m.stride * (dim_hi - 1)
            else:
                s_min = m.offset + m.stride * (dim_hi - 1)
                s_max = m.offset + m.stride * dim_lo
            first = dg.index_to_chunk(s_min)
            last = dg.index_to_chunk(s_max)
            slot_dims.append((out_dim,))
            slot_candidates.append([(c,) for c in range(first, last + 1)])
        else:
            # m: ArrayMap (OutputIndexMap = ConstantMap | DimensionMap | ArrayMap).
            # Storage coordinates were already computed for a correlated 1-D map.
            storage = (
                array_map_1d[1] if array_map_1d is not None else m.offset + m.stride * m.index_array
            )
            if storage.size == 0:
                # Empty fancy selection: no coordinates, so no chunks are touched.
                return
            # Keep the index-array shape: correlated maps broadcast against each
            # other below, and raveling first would lose the singleton axes.
            chunk_ids = dg.indices_to_chunks(storage.astype(np.intp))
            if m.input_dimension is None:
                correlated_dims.append(out_dim)
                correlated_chunk_ids.append(chunk_ids)
            else:
                slot_dims.append((out_dim,))
                slot_candidates.append([(int(c),) for c in np.unique(chunk_ids)])

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
            c_size = dg.chunk_size(c)
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


def _dimension_map_slice(m: DimensionMap, dim_lo: int, dim_hi: int) -> slice:
    """The NumPy slice a `DimensionMap` reads over input range `[dim_lo, dim_hi)`.

    The walk starts at the storage coordinate of `dim_lo` and takes `dim_hi -
    dim_lo` steps of `m.stride`. A downward walk that reaches the start of the
    axis must stop at `None`: an integer stop below zero counts from the end
    instead, and swapping the endpoints while keeping the negative step (which
    this did) produces a slice that selects nothing at all.
    """
    return _positional_slice(m.offset + m.stride * dim_lo, dim_hi - dim_lo, m.stride)


def sub_transform_to_selections(
    sub_transform: IndexTransform,
    out_indices: OutIndices = None,
) -> tuple[
    tuple[int | slice | np.ndarray[tuple[int, ...], np.dtype[np.intp]], ...],
    tuple[slice | np.ndarray[tuple[int, ...], np.dtype[np.intp]], ...],
    tuple[int, ...],
]:
    """Convert a chunk-local sub-transform to raw selections for the codec pipeline.

    Parameters
    ----------
    sub_transform
        A chunk-local IndexTransform (output maps already translated to
        chunk-local coordinates).
    out_indices
        For vectorized indexing: the output scatter indices for this chunk.
        None for orthogonal/basic indexing.

    Returns
    -------
    tuple
        `(chunk_selection, out_selection, drop_axes)`
    """
    inclusive_min = sub_transform.domain.inclusive_min
    exclusive_max = sub_transform.domain.exclusive_max

    # Orthogonal outer product: >= 2 ArrayMaps each bound to a distinct input
    # dimension. out_indices is a per-output-dim dict of surviving positions. The
    # codec applies chunk_array[chunk_sel] / out[out_sel] with NumPy semantics, so
    # build np.ix_-style selections (mirroring the legacy OrthogonalIndexer): one
    # 1-D selector per dimension, expanded to an open mesh. ConstantMap dims are
    # size-1 in chunk space and squeezed out via drop_axes.
    if isinstance(out_indices, dict):
        chunk_arrays: list[np.ndarray[Any, np.dtype[np.intp]]] = []
        out_arrays: list[np.ndarray[Any, np.dtype[np.intp]]] = []
        drop_axes: list[int] = []
        for out_dim, m in enumerate(sub_transform.output):
            if isinstance(m, ConstantMap):
                chunk_arrays.append(np.array([m.offset], dtype=np.intp))
                drop_axes.append(out_dim)
            elif isinstance(m, DimensionMap):
                rng = np.arange(inclusive_min[m.input_dimension], exclusive_max[m.input_dimension])
                chunk_arrays.append((m.offset + m.stride * rng).astype(np.intp))
                out_arrays.append(rng.astype(np.intp))
            else:  # ArrayMap
                idx = m.index_array.ravel()
                chunk_arrays.append((m.offset + m.stride * idx).astype(np.intp))
                out_arrays.append(out_indices[out_dim])
        return np.ix_(*chunk_arrays), np.ix_(*out_arrays), tuple(drop_axes)

    # Correlated (vindex) sub-transforms carry ArrayMaps with `input_dimension`
    # None. They scatter through a single flat index (`out_indices`) into the
    # row-major-flattened output buffer; the chunk selection reads a
    # (points, residual-slice) block via the raveled coordinate arrays and any
    # residual DimensionMap slices.
    correlated = any(
        isinstance(m, ArrayMap) and m.input_dimension is None for m in sub_transform.output
    )
    if correlated:
        chunk_sel: list[int | slice | np.ndarray[tuple[int, ...], np.dtype[np.intp]]] = []
        for m in sub_transform.output:
            if isinstance(m, ConstantMap):
                chunk_sel.append(m.offset)
            elif isinstance(m, DimensionMap):
                d = m.input_dimension
                chunk_sel.append(_dimension_map_slice(m, inclusive_min[d], exclusive_max[d]))
            else:  # ArrayMap
                idx = m.index_array.reshape(-1)
                chunk_sel.append((m.offset + m.stride * idx).astype(np.intp))
        # Chunk resolution always supplies the flat scatter index for a
        # correlated transform. Absent one (a bare sub-transform), fall back to an
        # identity scatter over the whole flattened output buffer.
        # `out_indices` is narrowed to a flat scatter array or None here (the
        # per-dimension dict is an orthogonal outer product, handled above).
        out_scatter: slice | np.ndarray[Any, np.dtype[np.intp]]
        if out_indices is None:
            n = 1
            for s in sub_transform.domain.shape:
                n *= s
            out_scatter = slice(0, n)
        else:
            out_scatter = out_indices
        return tuple(chunk_sel), (out_scatter,), ()

    chunk_sel = []  # annotated in the correlated branch above (same function scope)
    out_sel: list[slice | np.ndarray[tuple[int, ...], np.dtype[np.intp]]] = []

    # Single-pass build for the basic / single-orthogonal-array cases.
    # ConstantMap dims are dropped (no out_sel entry).
    for m in sub_transform.output:
        if isinstance(m, ConstantMap):
            chunk_sel.append(m.offset)
        elif isinstance(m, DimensionMap):
            d = m.input_dimension
            dim_lo = inclusive_min[d]
            dim_hi = exclusive_max[d]
            chunk_sel.append(_dimension_map_slice(m, dim_lo, dim_hi))
            out_sel.append(slice(dim_lo, dim_hi))
        else:  # ArrayMap (orthogonal: full-rank, raveled to its 1-D fancy coords)
            idx = m.index_array.reshape(-1)
            if m.offset == 0 and m.stride == 1:
                chunk_sel.append(idx)
            else:
                chunk_sel.append((m.offset + m.stride * idx).astype(np.intp))
            # Orthogonal ArrayMap: out_indices holds the surviving positions.
            out_sel.append(out_indices if out_indices is not None else slice(0, idx.size))

    return tuple(chunk_sel), tuple(out_sel), ()
