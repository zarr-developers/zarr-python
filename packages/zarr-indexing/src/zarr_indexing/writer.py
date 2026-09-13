"""Synchronous writes through coordinate transforms using basic source assignment."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from zarr_indexing.output_map import ArrayMap, ConstantMap, DimensionMap

if TYPE_CHECKING:
    from zarr_indexing.transform import IndexTransform


def write_into(source: Any, transform: IndexTransform, values: Any) -> None:
    """Write broadcast values to precisely the cells addressed by ``transform``.

    The source must expose ``shape``, a NumPy-compatible ``dtype``, and basic
    integer/slice assignment. Values are snapshotted and converted before the
    first mutation, so source aliases and conversion failures are safe. Extra
    leading singleton value axes are accepted, as in NumPy assignment.

    Independent affine maps use one basic assignment. Other maps use one
    integer assignment per result cell, in C order: repeated destinations get
    the last value. This conservative fallback can be expensive for remote or
    chunked sources. It does not read the destination to fill unselected cells;
    a lazy RHS is materialized before assignment. Backend assignment may itself
    read storage units as part of updating them.
    Coordinate storage is bounded by rank; the RHS snapshot occupies its
    original, unbroadcast size. Source assignment failures propagate and may
    leave preceding writes applied; this operation is not transactional.
    """
    from zarr_indexing.lazy_array import LazyArray

    if isinstance(values, LazyArray):
        # Its NumPy conversion protocol may discard MaskedArray metadata.
        values = values.result()
    shape = transform.domain.shape
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
    if transform.output_rank != len(source.shape):
        raise ValueError("transform output rank must match the source rank")
    if any(extent == 0 for extent in shape):
        return

    # Check every map's extremal outputs before any assignment. Python integer
    # arithmetic avoids overflow while computing bounds of affine maps.
    for extent, output in zip(source.shape, transform.output, strict=True):
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

    axes = [
        output.input_dimension for output in transform.output if isinstance(output, DimensionMap)
    ]
    if (
        not any(isinstance(output, ArrayMap) for output in transform.output)
        and len(set(axes)) == len(axes)
        and all(shape[axis] == 1 for axis in range(len(shape)) if axis not in axes)
        and all(
            not isinstance(output, DimensionMap) or output.stride != 0
            for output in transform.output
        )
    ):
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
                    selection.append(
                        slice(start, start + output.stride * shape[axis], output.stride)
                    )
        unused = tuple(axis for axis in range(len(shape)) if axis not in axes)
        squeezed = np.squeeze(broadcast, axis=unused)
        remaining = sorted(axes)
        ordered = np.transpose(squeezed, tuple(remaining.index(axis) for axis in axes))
        source[tuple(selection)] = np.flip(ordered, axis=tuple(reverse_axes))
        return

    origin = transform.domain.inclusive_min
    for position in np.ndindex(shape):
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
