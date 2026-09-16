"""Synchronous writes through coordinate transforms using basic source assignment."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from zarr_indexing.chunk_resolution import plan_chunks
from zarr_indexing.output_map import ArrayMap, ConstantMap, DimensionMap

if TYPE_CHECKING:
    from collections.abc import Sequence

    from zarr_indexing.chunk_resolution import ChunkProjection
    from zarr_indexing.grid import DimensionGridLike
    from zarr_indexing.transform import IndexTransform


def write_into(
    source: Any,
    transform: IndexTransform,
    values: Any,
    write_grid: Sequence[DimensionGridLike] | None = None,
) -> None:
    """Write broadcast values to precisely the cells addressed by ``transform``.

    The source must expose ``shape``, a NumPy-compatible ``dtype``, and basic
    integer/slice assignment. Values are snapshotted and converted before the
    first mutation, so source aliases and conversion failures are safe. Extra
    leading singleton value axes are accepted, as in NumPy assignment. Repeated
    source coordinates receive the last value in C order of the view.

    Independent affine maps use one basic assignment. Other selections are
    scattered in bulk: a NumPy source receives one fancy assignment, and a
    readable source with a ``write_grid`` is written one grid cell at a time —
    the cell's touched hull is read once, updated in memory, and written back
    with one basic assignment, so storage round trips are bounded by touched
    cells. Without a grid, or for the few transforms the planner cannot factor
    (such as one input axis feeding two output maps), or for a source that
    cannot be read, the fallback is one integer assignment per element, which
    never reads. Backend assignment may itself read storage units as part of
    updating them.

    Source assignment failures propagate and may leave preceding writes
    applied; this operation is not transactional.
    """
    from zarr_indexing.lazy_array import LazyArray

    shape = transform.domain.shape
    source_shape = tuple(int(extent) for extent in source.shape)
    if transform.output_rank != len(source_shape):
        raise ValueError("transform output rank must match the source rank")
    if isinstance(values, LazyArray):
        # Its NumPy conversion protocol may discard MaskedArray metadata.
        values = values.result()
    if isinstance(source, np.ma.MaskedArray):
        prepared = np.ma.array(values, dtype=source.dtype, copy=True)
    else:
        prepared = np.array(values, dtype=source.dtype, copy=True)
    while prepared.ndim > len(shape) and prepared.shape[0] == 1:
        prepared = prepared[0]
    if isinstance(prepared, np.ma.MaskedArray):
        broadcast = np.ma.array(
            np.broadcast_to(prepared.data, shape),
            mask=np.broadcast_to(np.ma.getmaskarray(prepared), shape),
            copy=False,
        )
    else:
        broadcast = np.broadcast_to(prepared, shape)
    if any(extent == 0 for extent in shape):
        # Values were still converted and broadcast, so a mis-sized RHS is
        # reported even when the selection happens to be empty.
        return

    # Check every map's extremal outputs before any assignment. Python integer
    # arithmetic avoids overflow while computing bounds of affine maps.
    for extent, output in zip(source_shape, transform.output, strict=True):
        if isinstance(output, ConstantMap):
            low = high = output.offset
        else:
            if isinstance(output, DimensionMap):
                axis = output.input_dimension
                first = transform.domain.inclusive_min[axis]
                last = transform.domain.exclusive_max[axis] - 1
            else:
                first = int(output.index_array.min())
                last = int(output.index_array.max())
            endpoints = (
                output.offset + output.stride * first,
                output.offset + output.stride * last,
            )
            low, high = min(endpoints), max(endpoints)
        if low < 0 or high >= extent:
            raise IndexError("transform output coordinates are outside the source shape")

    if _write_affine(source, transform, broadcast):
        return
    if len(source_shape) == 0:
        # A zero-rank source has one cell; no grid or scatter applies.
        source[()] = broadcast[(0,) * len(shape)]
        return
    if isinstance(source, np.ndarray):
        _scatter_numpy(source, transform, broadcast)
        return
    if write_grid is not None and hasattr(source, "__getitem__"):
        try:
            # Materialized before any write: an unfactorable transform is
            # reported by planning, never after some cells were rewritten.
            projections = list(plan_chunks(transform, tuple(write_grid)))
        except (ValueError, NotImplementedError):
            projections = None
        if projections is not None:
            _scatter_planned(source, transform, broadcast, projections)
            return
    _scatter_elementwise(source, transform, broadcast)


def _write_affine(source: Any, transform: IndexTransform, broadcast: Any) -> bool:
    """Write with one basic assignment when every map is an independent affine map."""
    shape = transform.domain.shape
    axes = [
        output.input_dimension for output in transform.output if isinstance(output, DimensionMap)
    ]
    if (
        any(isinstance(output, ArrayMap) for output in transform.output)
        or len(set(axes)) != len(axes)
        or any(shape[axis] != 1 for axis in range(len(shape)) if axis not in axes)
        or any(
            isinstance(output, DimensionMap) and output.stride == 0 for output in transform.output
        )
    ):
        return False
    selection: list[int | slice] = []
    reverse_axes: list[int] = []
    for output in transform.output:
        if isinstance(output, ConstantMap):
            selection.append(output.offset)
        else:
            assert isinstance(output, DimensionMap)
            axis = output.input_dimension
            start = output.offset + output.stride * transform.domain.inclusive_min[axis]
            if output.stride < 0:
                reverse_axes.append(axes.index(axis))
                selection.append(
                    slice(start + output.stride * (shape[axis] - 1), start + 1, -output.stride)
                )
            else:
                selection.append(slice(start, start + output.stride * shape[axis], output.stride))
    unused = tuple(axis for axis in range(len(shape)) if axis not in axes)
    squeezed = np.squeeze(broadcast, axis=unused)
    remaining = sorted(axes)
    ordered = np.transpose(squeezed, tuple(remaining.index(axis) for axis in axes))
    if reverse_axes:
        # `np.flip` with no axes would hand a zero-rank masked value back as
        # the shared `masked` singleton, discarding its payload.
        ordered = np.flip(ordered, axis=tuple(reverse_axes))
    source[tuple(selection)] = ordered
    return True


def _domain_cells(shape: tuple[int, ...]) -> np.ndarray[Any, np.dtype[np.intp]]:
    """Every coordinate of a zero-origin domain, one row per cell in C order."""
    if len(shape) == 0:
        return np.zeros((1, 0), dtype=np.intp)
    return np.indices(shape, dtype=np.intp).reshape(len(shape), -1).T


def _last_occurrences(flat: np.ndarray[Any, Any]) -> np.ndarray[Any, np.dtype[np.intp]]:
    """Positions to keep so that repeated flat indices take their last value."""
    _, first_in_reversed = np.unique(flat[::-1], return_index=True)
    return np.asarray(len(flat) - 1 - first_in_reversed, dtype=np.intp)


def _scatter_numpy(source: np.ndarray[Any, Any], transform: IndexTransform, broadcast: Any) -> None:
    """One fancy assignment: NumPy sources need no read-modify-write."""
    positions = _domain_cells(transform.domain.shape)
    coords = transform.apply_many(positions + np.asarray(transform.domain.inclusive_min))
    flat = np.asarray(np.ravel_multi_index(tuple(coords.T), source.shape), dtype=np.intp)
    keep = _last_occurrences(flat)
    source[tuple(coords[keep].T)] = broadcast[tuple(positions[keep].T)]


def _scatter_planned(
    source: Any,
    transform: IndexTransform,
    broadcast: Any,
    projections: Sequence[ChunkProjection],
) -> None:
    """One read-modify-write per touched grid cell, through basic slices only."""
    origin = np.asarray(transform.domain.inclusive_min, dtype=np.intp)
    for projection in projections:
        cells = _domain_cells(projection.chunk_transform.domain.shape)
        local = projection.chunk_transform.apply_many(cells)
        positions = projection.cell_transform.apply_many(cells) - origin
        cell_origin = np.asarray(projection.chunk_domain.inclusive_min, dtype=np.intp)
        hull_min = local.min(axis=0)
        hull_shape = tuple(int(n) for n in local.max(axis=0) + 1 - hull_min)
        box = tuple(
            slice(int(lo), int(lo + n))
            for lo, n in zip(cell_origin + hull_min, hull_shape, strict=True)
        )
        block = np.array(source[box], dtype=source.dtype, copy=True)
        flat = np.asarray(
            np.ravel_multi_index(tuple((local - hull_min).T), hull_shape), dtype=np.intp
        )
        keep = _last_occurrences(flat)
        np.put(block, flat[keep], np.asarray(broadcast[tuple(positions[keep].T)]))
        source[box] = block


def _scatter_elementwise(source: Any, transform: IndexTransform, broadcast: Any) -> None:
    """One integer assignment per element, for sources that cannot be read."""
    origin = transform.domain.inclusive_min
    for position in np.ndindex(transform.domain.shape):
        point = tuple(start + offset for start, offset in zip(origin, position, strict=True))
        if isinstance(broadcast, np.ma.MaskedArray):
            # Masked scalar indexing returns the shared ``masked`` singleton,
            # discarding the payload. A zero-dimensional array retains both.
            value = np.ma.array(
                broadcast.data[position], mask=np.ma.getmaskarray(broadcast)[position]
            )
        else:
            value = broadcast[position]
        source[transform.apply(point)] = value
