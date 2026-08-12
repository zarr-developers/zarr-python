"""Resolve a `SelectionRequest` against a zarrista array.

Reads go through `zarr_indexing.LazyArray`, which grafts the whole NumPy
indexing dialect onto a backend that natively offers only step-1 boxes.

Writes have no `LazyArray` equivalent — it describes reads only — so they are
resolved here, in two tiers:

- A *dense forward box* (the shape of an ordinary `arr[a:b, c:d] = v`) is one
  `store_array_subset` call. zarrista does any partial-chunk read-modify-write
  internally, in Rust.
- Anything else — strided, reversed, orthogonal, vectorized, masked — is
  scattered pointwise: the transform is evaluated over its whole domain to get
  one storage coordinate per value, the points are grouped by chunk, and each
  touched chunk is read, patched, and written back once. Grouping by chunk is
  what keeps this from degenerating into the hull rewrite that a bounding-box
  strategy would perform (`oindex[[0, 999999]]` touches two chunks, not a
  million rows).

Both tiers take their transform from the `LazyArray` view rather than building
one directly, because `LazyArray` re-bases every view's domain to origin 0.
A transform built straight from `IndexTransform.from_shape(shape)[1:5]` has
domain origin 1, and scattering a zero-origin value buffer through it would
silently write to the wrong offsets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from zarr_indexing import LazyArray, unit_step_reader
from zarr_indexing.output_map import ConstantMap, DimensionMap

if TYPE_CHECKING:
    import numpy.typing as npt
    from zarr_indexing import IndexTransform

    from zarr.abc.engine import SelectionRequest
    from zarr.core.chunk_grids import ChunkGrid

__all__ = ["dense_forward_box", "lazy_view", "scatter_points"]


def lazy_view(source: Any, request: SelectionRequest, chunk_shape: tuple[int, ...] | None) -> Any:
    """Build the `LazyArray` view `request` selects from `source`.

    `chunk_shape`, when given, partitions the read so a fancy or strided
    selection is covered by per-chunk boxes rather than one hull read.
    """
    lazy = LazyArray(source).with_reader(unit_step_reader)
    if chunk_shape is not None:
        lazy = lazy.with_parts(list(chunk_shape))
    kind = request.kind
    selection = request.selection
    if kind == "basic":
        return lazy.lazy[selection]
    if kind == "orthogonal":
        return lazy.lazy.oindex[selection]
    if kind in ("coordinate", "mask"):
        return lazy.lazy.vindex[selection]
    if kind == "block":
        # A block selection is per-axis slices once expanded against the grid;
        # the indexer has already done that expansion.
        return lazy.lazy[_block_slices(request)]
    raise NotImplementedError(f"the zarrista engine cannot serve a {kind!r} selection")


def _block_slices(request: SelectionRequest) -> tuple[slice, ...]:
    """The element-space slices a block selection expands to."""
    return tuple(
        slice(d.start, d.stop, d.step)  # BlockIndexer yields only SliceDimIndexers
        for d in request.indexer.dim_indexers  # type: ignore[attr-defined]
    )


def dense_forward_box(
    transform: IndexTransform,
) -> tuple[tuple[slice, ...], tuple[int, ...]] | None:
    """The step-1 box `transform` selects, or `None` if it is not one.

    Returns `(key, ndim_shape)` where `key` is a per-storage-axis step-1 slice
    and `ndim_shape` is the ndim-preserving shape to reshape a value to (an
    axis the selection dropped with an integer has extent 1). `None` means the
    selection strides, reverses, transposes, broadcasts, or gathers, and so
    must be scattered instead.
    """
    key: list[slice] = []
    shape: list[int] = []
    previous = -1
    domain = transform.domain
    for m in transform.output:
        if isinstance(m, ConstantMap):
            key.append(slice(m.offset, m.offset + 1))
            shape.append(1)
        elif isinstance(m, DimensionMap):
            # stride 1 only: a reversal (-1) or a stride selects a sublattice
            # of its hull, so the hull is not what the value fills. Requiring
            # ascending input dimensions rejects a transposing selection,
            # whose value layout does not match the box.
            if m.stride != 1 or m.input_dimension <= previous:
                return None
            previous = m.input_dimension
            d = m.input_dimension
            lo = m.offset + domain.inclusive_min[d]
            hi = m.offset + domain.exclusive_max[d]
            key.append(slice(lo, hi))
            shape.append(hi - lo)
        else:
            return None
    # Every domain axis must be consumed exactly once; otherwise the value is
    # broadcast across an axis rather than laid into a box.
    dims = sorted(m.input_dimension for m in transform.output if isinstance(m, DimensionMap))
    if dims != list(range(transform.input_rank)):
        return None
    return tuple(key), tuple(shape)


def _domain_points(transform: IndexTransform) -> npt.NDArray[np.intp]:
    """Every point of `transform`'s domain, in C order, as `(n, input_rank)`."""
    domain = transform.domain
    axes = [
        np.arange(lo, hi) for lo, hi in zip(domain.inclusive_min, domain.exclusive_max, strict=True)
    ]
    if not axes:
        return np.zeros((1, 0), dtype=np.intp)
    grid = np.meshgrid(*axes, indexing="ij")
    return np.stack(grid, axis=-1).reshape(-1, len(axes)).astype(np.intp, copy=False)


def _chunk_indices(storage: npt.NDArray[np.intp], chunk_grid: ChunkGrid) -> npt.NDArray[np.intp]:
    """Map storage coordinates to chunk-grid coordinates, per axis."""
    dims = chunk_grid._dimensions
    return np.stack(
        [dim.indices_to_chunks(storage[:, axis]) for axis, dim in enumerate(dims)],
        axis=-1,
    )


def scatter_points(
    read_chunk: Any,
    write_chunk: Any,
    transform: IndexTransform,
    value: npt.NDArray[Any],
    chunk_grid: ChunkGrid,
    shape: tuple[int, ...],
) -> None:
    """Write `value` through `transform`, one touched chunk at a time.

    `read_chunk(key)` returns the decoded box at `key` as a writable, C-order
    NumPy array; `write_chunk(key, data)` stores it back.

    Duplicate coordinates (legal in a vectorized write) resolve last-wins,
    matching `numpy.ndarray.__setitem__`, because the points keep their
    original order within a chunk and NumPy's fancy assignment is last-wins.
    """
    points = _domain_points(transform)
    if points.size == 0 and transform.input_rank > 0:
        return
    storage = np.asarray(transform.apply_many(points), dtype=np.intp)
    values = np.asarray(value).reshape(-1)
    if values.size != storage.shape[0]:
        raise ValueError(
            f"value has {values.size} elements but the selection covers {storage.shape[0]}"
        )
    if storage.shape[0] == 0:
        return

    chunk_ix = _chunk_indices(storage, chunk_grid)
    # Group points by chunk. `lexsort` on the reversed key columns sorts by the
    # leading axis first; a stable sort keeps each chunk's points in the user's
    # original order, which is what makes last-wins match NumPy.
    order = np.lexsort(tuple(chunk_ix[:, axis] for axis in range(chunk_ix.shape[1] - 1, -1, -1)))
    storage, values, chunk_ix = storage[order], values[order], chunk_ix[order]

    boundaries = np.flatnonzero(np.any(np.diff(chunk_ix, axis=0) != 0, axis=1)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(storage)]))

    dims = chunk_grid._dimensions
    for start, end in zip(starts, ends, strict=True):
        coords = chunk_ix[start]
        origin = np.array(
            [dim.chunk_offset(int(c)) for dim, c in zip(dims, coords, strict=True)],
            dtype=np.intp,
        )
        extent = np.array(
            [dim.data_size(int(c)) for dim, c in zip(dims, coords, strict=True)],
            dtype=np.intp,
        )
        # `data_size` is the chunk's *clipped* extent, so an edge chunk is read
        # and written at its valid size rather than its nominal one.
        stop = np.minimum(origin + extent, np.array(shape, dtype=np.intp))
        key = tuple(slice(int(a), int(b)) for a, b in zip(origin, stop, strict=True))
        buffer = read_chunk(key)
        local = storage[start:end] - origin
        buffer[tuple(local.T)] = values[start:end]
        write_chunk(key, buffer)
