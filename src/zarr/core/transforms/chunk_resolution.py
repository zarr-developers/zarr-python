"""Chunk resolution — mapping transforms to chunk-level I/O.

Given an ``IndexTransform`` (which coordinates a user wants to access) and a
``ChunkGrid`` (how storage is divided into chunks), chunk resolution answers:

    For each chunk, which storage coordinates does this transform touch,
    and where do those values land in the output buffer?

The algorithm is:

1. **Enumerate candidate chunks** — determine which chunks could possibly
   be touched by the transform's output coordinate ranges.

2. **Intersect** — for each candidate chunk, call
   ``transform.intersect(chunk_domain)`` to restrict the transform to
   coordinates within that chunk. If the intersection is empty, skip it.

3. **Translate** — shift the restricted transform to chunk-local coordinates
   via ``transform.translate(-chunk_origin)``.

4. **Yield** — produce ``(chunk_coords, local_transform, surviving_indices)``
   triples that the codec pipeline consumes.

``sub_transform_to_selections`` bridges from the transform representation
back to the raw ``(chunk_selection, out_selection, drop_axes)`` tuples that
the current codec pipeline expects. This bridge will go away when the codec
pipeline accepts transforms natively.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from zarr.core.transforms.domain import IndexDomain
from zarr.core.transforms.output_map import ArrayMap, ConstantMap, DimensionMap
from zarr.core.transforms.transform import IndexTransform

if TYPE_CHECKING:
    from collections.abc import Iterator

    from zarr.core.chunk_grids import ChunkGrid

ChunkTransformResult = tuple[
    tuple[int, ...],
    IndexTransform,
    np.ndarray[Any, np.dtype[np.intp]] | None,
]


def iter_chunk_transforms(
    transform: IndexTransform,
    chunk_grid: ChunkGrid,
) -> Iterator[ChunkTransformResult]:
    """Resolve a composed IndexTransform against a ChunkGrid.

    Yields ``(chunk_coords, sub_transform, out_indices)`` triples:

    - ``chunk_coords``: which chunk to access.
    - ``sub_transform``: maps output buffer coords to chunk-local coords.
    - ``out_indices``: for vectorized/array indexing, the output scatter
      indices (integer array). ``None`` for basic/slice indexing.
    """
    dim_grids = chunk_grid._dimensions

    # Enumerate all possible chunks via cartesian product of per-dim chunk ranges
    # For each candidate chunk, intersect the transform with the chunk domain.
    # The transform.intersect method handles both orthogonal and vectorized cases.
    chunk_ranges: list[range] = []
    for out_dim, m in enumerate(transform.output):
        dg = dim_grids[out_dim]
        if isinstance(m, ConstantMap):
            # Single chunk
            c = dg.index_to_chunk(m.offset)
            chunk_ranges.append(range(c, c + 1))
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
            chunk_ranges.append(range(first, last + 1))
        elif isinstance(m, ArrayMap):
            storage = m.offset + m.stride * m.index_array
            flat = storage.ravel().astype(np.intp)
            chunk_ids = dg.indices_to_chunks(flat)
            first = int(chunk_ids.min())
            last = int(chunk_ids.max())
            chunk_ranges.append(range(first, last + 1))

    import itertools

    for chunk_coords_tuple in itertools.product(*chunk_ranges):
        chunk_coords = tuple(int(c) for c in chunk_coords_tuple)

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


def sub_transform_to_selections(
    sub_transform: IndexTransform,
    out_indices: np.ndarray[Any, np.dtype[np.intp]] | None = None,
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
        ``(chunk_selection, out_selection, drop_axes)``
    """
    chunk_sel: list[int | slice | np.ndarray[tuple[int, ...], np.dtype[np.intp]]] = []
    out_sel: list[slice | np.ndarray[tuple[int, ...], np.dtype[np.intp]]] = []

    # Hoist the per-dimension domain bounds out of the loop (attribute-chain +
    # tuple indexing per output dim otherwise). Build chunk_sel and out_sel in a
    # single pass; ConstantMap dims are dropped (no out_sel entry).
    inclusive_min = sub_transform.domain.inclusive_min
    exclusive_max = sub_transform.domain.exclusive_max
    n_array_maps = 0

    for m in sub_transform.output:
        t = type(m)
        if t is ConstantMap:
            chunk_sel.append(m.offset)
        elif t is DimensionMap:
            d = m.input_dimension
            dim_lo = inclusive_min[d]
            dim_hi = exclusive_max[d]
            start = m.offset + m.stride * dim_lo
            stop = m.offset + m.stride * dim_hi
            if m.stride < 0:
                start, stop = stop + 1, start + 1
            chunk_sel.append(slice(start, stop, m.stride))
            out_sel.append(slice(dim_lo, dim_hi))
        else:  # ArrayMap
            n_array_maps += 1
            if m.offset == 0 and m.stride == 1:
                chunk_sel.append(m.index_array)
            else:
                chunk_sel.append((m.offset + m.stride * m.index_array).astype(np.intp))
            # Orthogonal ArrayMap: out_indices holds the surviving positions.
            out_sel.append(out_indices if out_indices is not None else slice(0, len(m.index_array)))

    # Vectorized: ≥2 correlated ArrayMaps scatter through a single shared index.
    if out_indices is not None and n_array_maps >= 2:
        out_sel = [out_indices]

    return tuple(chunk_sel), tuple(out_sel), ()
