"""Composition — chaining two transforms into one.

`compose(outer, inner)` is the operation that makes views stack. `outer` maps
user coordinates to intermediate coordinates, `inner` maps those intermediate
coordinates to storage, and the result maps user coordinates straight to
storage — so a view of a view of an array is still a single
`IndexTransform`, and indexing never accumulates layers to walk at read time.

Composition works one output map at a time, and each case reduces to
substituting the outer map into the inner one:

- A `ConstantMap` inner map ignores its input, so it survives unchanged.
- A `DimensionMap` inner map is affine, so composing it with an outer
  `ConstantMap` or `DimensionMap` folds into new `offset`/`stride` values;
  composing it with an outer `ArrayMap` leaves the index array alone and
  rescales around it.
- An `ArrayMap` inner map must be *evaluated* at the coordinates the outer
  transform produces, which is the only case that touches array data.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from zarr_indexing.errors import BoundsCheckError
from zarr_indexing.output_map import ArrayMap, ConstantMap, DimensionMap, OutputIndexMap
from zarr_indexing.transform import IndexTransform


def compose(outer: IndexTransform, inner: IndexTransform) -> IndexTransform:
    """Compose two IndexTransforms.

    `outer` maps user coords (rank m) to intermediate coords (rank n).
    `inner` maps intermediate coords (rank n) to storage coords (rank p).
    The result maps user coords (rank m) to storage coords (rank p).

    Precondition: `outer.output_rank == inner.domain.ndim`.
    """
    if outer.output_rank != inner.domain.ndim:
        raise ValueError(
            f"outer output rank ({outer.output_rank}) must match inner input rank "
            f"({inner.domain.ndim})"
        )

    result_output = [
        _compose_single(outer, inner_map, inner.domain.inclusive_min) for inner_map in inner.output
    ]

    return IndexTransform(domain=outer.domain, output=tuple(result_output))


def _compose_single(
    outer: IndexTransform, inner_map: OutputIndexMap, inner_origin: tuple[int, ...]
) -> OutputIndexMap:
    """Compose a single inner output map with the full outer transform."""
    if isinstance(inner_map, ConstantMap):
        return ConstantMap(offset=inner_map.offset)

    if isinstance(inner_map, DimensionMap):
        return _compose_dimension(outer, inner_map)

    # inner_map: ArrayMap (OutputIndexMap = ConstantMap | DimensionMap | ArrayMap)
    return _compose_array(outer, inner_map, inner_origin)


def _compose_dimension(outer: IndexTransform, inner_map: DimensionMap) -> OutputIndexMap:
    """Compose when inner is a DimensionMap.

    storage = offset_i + stride_i * intermediate[dim_i]
    where intermediate[dim_i] = outer.output[dim_i](user_input)
    """
    dim_i = inner_map.input_dimension
    offset_i = inner_map.offset
    stride_i = inner_map.stride
    outer_map = outer.output[dim_i]

    if isinstance(outer_map, ConstantMap):
        return ConstantMap(offset=offset_i + stride_i * outer_map.offset)

    if isinstance(outer_map, DimensionMap):
        return DimensionMap(
            input_dimension=outer_map.input_dimension,
            offset=offset_i + stride_i * outer_map.offset,
            stride=stride_i * outer_map.stride,
        )

    # outer_map: ArrayMap (OutputIndexMap = ConstantMap | DimensionMap | ArrayMap)
    # Affine post-composition leaves the index array (and hence its full
    # input rank and dependency axes) untouched; carry the orthogonal
    # binding through unchanged.
    return ArrayMap(
        index_array=outer_map.index_array,
        offset=offset_i + stride_i * outer_map.offset,
        stride=stride_i * outer_map.stride,
        input_dimension=outer_map.input_dimension,
    )


def _checked_position(position: int, size: int, axis: int) -> int:
    """A single positional index into an inner index array, bounds-checked.

    The intermediate coordinate comes from the outer transform's outputs, which
    nothing here has established lie inside the inner domain. Unchecked, a
    negative one wraps NumPy-style and reads a cell at the far end of the axis —
    silently, and with no way for a caller to tell.
    """
    if position < 0 or position >= size:
        raise BoundsCheckError(
            f"composing reads position {position} on axis {axis} of an index "
            f"array of extent {size}; the outer transform's output lies outside "
            f"the inner transform's domain"
        )
    return position


def _checked_positions(
    positions: np.ndarray[Any, np.dtype[np.intp]], size: int
) -> np.ndarray[Any, np.dtype[np.intp]]:
    """The same check, for a whole array of positional indices."""
    if positions.size > 0 and (int(positions.min()) < 0 or int(positions.max()) >= size):
        raise BoundsCheckError(
            f"composing reads positions in "
            f"[{int(positions.min())}, {int(positions.max())}] of an index array "
            f"of extent {size}; the outer transform's output lies outside the "
            f"inner transform's domain"
        )
    return positions


def _compose_array(
    outer: IndexTransform, inner_map: ArrayMap, inner_origin: tuple[int, ...]
) -> OutputIndexMap:
    """Compose when inner is an ArrayMap.

    storage = offset_i + stride_i * arr_i[intermediate]
    We need to evaluate arr_i at the intermediate coordinates produced by outer.

    Both domains carry their own origin, and neither is necessarily 0 — a step-1
    slice keeps its literal bounds and a negative step produces a negative
    origin, so non-zero origins are the ordinary case here rather than the exotic
    one. The intermediate coordinates are read over the *outer* domain's own
    range, and the inner array is addressed positionally from the *inner*
    domain's origin.
    """
    arr_i = inner_map.index_array
    offset_i = inner_map.offset
    stride_i = inner_map.stride

    # Check if all outer outputs are constant
    all_constant = all(isinstance(m, ConstantMap) for m in outer.output)

    if all_constant:
        # Evaluate arr_i at the single constant point. A non-dependency axis of
        # the inner array is a singleton that broadcasts over the whole domain,
        # so the coordinate there is always 0 whatever the intermediate names —
        # indexing such an axis by the raw coordinate walked off the end of it.
        idx = tuple(
            0 if size == 1 else _checked_position(m.offset - lo, size, axis)
            for axis, (m, lo, size) in enumerate(
                zip(
                    (m for m in outer.output if isinstance(m, ConstantMap)),
                    inner_origin,
                    arr_i.shape,
                    strict=True,
                )
            )
        )
        value = int(arr_i[idx])
        return ConstantMap(offset=offset_i + stride_i * value)

    # For 1D inner array with a single outer output (simple case)
    if arr_i.ndim == 1 and len(outer.output) == 1:
        outer_map = outer.output[0]
        rank = outer.input_rank

        if isinstance(outer_map, DimensionMap):
            dim = outer_map.input_dimension
            user_indices = np.arange(
                outer.domain.inclusive_min[dim], outer.domain.exclusive_max[dim], dtype=np.intp
            )
            intermediate_vals = outer_map.offset + outer_map.stride * user_indices
            positions = _checked_positions(intermediate_vals - inner_origin[0], arr_i.shape[0])
            # Shaped to the *outer* input rank, not left 1-D. The gate above is on
            # the output rank, so a rank-2 outer reached here and built an array
            # of the wrong rank for the domain it belongs to.
            shape = (1,) * dim + (len(positions),) + (1,) * (rank - dim - 1)
            return ArrayMap(
                index_array=arr_i[positions].reshape(shape),
                offset=offset_i,
                stride=stride_i,
                input_dimension=dim,
            )

        if isinstance(outer_map, ArrayMap):
            intermediate_vals = outer_map.offset + outer_map.stride * outer_map.index_array
            positions = _checked_positions(intermediate_vals - inner_origin[0], arr_i.shape[0])
            # The outer array already carries the outer input rank, so indexing
            # with it keeps that rank; the dependency travels with it.
            return ArrayMap(
                index_array=arr_i[positions],
                offset=offset_i,
                stride=stride_i,
                input_dimension=outer_map.input_dimension,
            )

    # General multi-dim case: not yet implemented
    raise NotImplementedError(
        "Composing a multi-dimensional inner array map with non-constant outer maps "
        "is not yet supported."
    )
