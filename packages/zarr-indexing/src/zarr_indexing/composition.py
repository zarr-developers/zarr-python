"""Composition — chaining two transforms into one.

`compose(outer, inner)` is the operation that makes views stack. `outer` maps
user coordinates to intermediate coordinates, `inner` maps those intermediate
coordinates to output coordinates, and the result maps user coordinates
straight through — so a view of a view of an array is still a single
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

from zarr_indexing.affine import checked_affine
from zarr_indexing.errors import BoundsCheckError
from zarr_indexing.output_map import (
    ArrayMap,
    ConstantMap,
    DimensionMap,
    OutputIndexMap,
    array_map_or_constant,
)
from zarr_indexing.transform import IndexTransform


def compose(outer: IndexTransform, inner: IndexTransform) -> IndexTransform:
    """Compose two IndexTransforms.

    `outer` maps user coords (rank m) to intermediate coords (rank n).
    `inner` maps intermediate coords (rank n) to output coords (rank p).
    The result maps user coords (rank m) to output coords (rank p).

    Precondition: `outer.output_rank == inner.domain.ndim`.

    Examples
    --------
    Chained indexing — `source[2:5]`, then `[::-1]` on the result — collapses
    to a single transform (a reversed axis keeps literal coordinates, so the
    composed domain is `[-4, -1)`):

    >>> inner = IndexTransform.from_shape((10,))[2:5]
    >>> outer = IndexTransform.identity(inner.domain)[::-1]
    >>> chained = compose(outer, inner)
    >>> chained == inner[::-1]
    True
    >>> [chained.apply((i,)) for i in (-4, -3, -2)]
    [(4,), (3,), (2,)]
    >>> np.arange(10)[2:5][::-1].tolist()
    [4, 3, 2]
    """
    if outer.output_rank != inner.domain.ndim:
        raise ValueError(
            f"outer output rank ({outer.output_rank}) must match inner input rank "
            f"({inner.domain.ndim})"
        )

    _validate_outer_outputs(outer, inner)

    result_output = [
        _compose_single(outer, inner_map, inner.domain.inclusive_min) for inner_map in inner.output
    ]

    return IndexTransform(domain=outer.domain, output=tuple(result_output))


def _validate_outer_outputs(outer: IndexTransform, inner: IndexTransform) -> None:
    """Prove that every intermediate coordinate is in the inner domain.

    An empty outer domain has no points, so containment is vacuously true. For
    a nonempty domain, each map form has an exact, constant-space range proof:
    constants are singletons, dimension maps are affine intervals, and array
    maps need only their extrema.
    """
    if any(extent == 0 for extent in outer.domain.shape):
        return

    for axis, (outer_map, inner_lo, inner_hi) in enumerate(
        zip(
            outer.output,
            inner.domain.inclusive_min,
            inner.domain.exclusive_max,
            strict=True,
        )
    ):
        output_lo, output_hi = _output_bounds(outer, outer_map)
        if output_lo < inner_lo or output_hi >= inner_hi:
            raise BoundsCheckError(
                f"outer output dimension {axis} produces coordinates "
                f"[{output_lo}, {output_hi}] outside the inner input domain "
                f"[{inner_lo}, {inner_hi})"
            )


def _output_bounds(outer: IndexTransform, output_map: OutputIndexMap) -> tuple[int, int]:
    """Return the exact inclusive bounds of one output map on a nonempty domain."""
    if isinstance(output_map, ConstantMap):
        return output_map.offset, output_map.offset

    if isinstance(output_map, DimensionMap):
        input_lo = outer.domain.inclusive_min[output_map.input_dimension]
        input_hi = outer.domain.exclusive_max[output_map.input_dimension]
        first = output_map.offset + output_map.stride * input_lo
        last = output_map.offset + output_map.stride * (input_hi - 1)
        return min(first, last), max(first, last)

    index_lo = int(output_map.index_array.min())
    index_hi = int(output_map.index_array.max())
    first = output_map.offset + output_map.stride * index_lo
    last = output_map.offset + output_map.stride * index_hi
    return min(first, last), max(first, last)


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
        return ConstantMap(offset=checked_affine(offset_i, stride_i, outer_map.offset))

    if isinstance(outer_map, DimensionMap):
        return DimensionMap(
            input_dimension=outer_map.input_dimension,
            offset=offset_i + stride_i * outer_map.offset,
            stride=stride_i * outer_map.stride,
        )

    # outer_map: ArrayMap (OutputIndexMap = ConstantMap | DimensionMap | ArrayMap)
    # Affine post-composition leaves the index array (and hence its full
    # input rank and dependency axes) untouched.
    return ArrayMap(
        index_array=outer_map.index_array,
        offset=offset_i + stride_i * outer_map.offset,
        stride=stride_i * outer_map.stride,
    )


def _dimension_positions(
    outer: IndexTransform, outer_map: DimensionMap, inner_origin: int
) -> np.ndarray[Any, np.dtype[np.intp]]:
    """Build exact positional indices for an affine outer map."""
    dimension = outer_map.input_dimension
    extent = outer.domain.shape[dimension]
    start = checked_affine(
        outer_map.offset - inner_origin,
        outer_map.stride,
        outer.domain.inclusive_min[dimension],
    )
    shape = (1,) * dimension + (extent,) + (1,) * (outer.input_rank - dimension - 1)
    if extent == 0:
        return np.empty(shape, dtype=np.intp)
    if extent == 1:
        return np.full(shape, start, dtype=np.intp)
    steps = np.arange(extent, dtype=np.intp)
    return checked_affine(start, outer_map.stride, steps).reshape(shape)


def _array_positions(outer_map: ArrayMap, inner_origin: int) -> np.ndarray[Any, np.dtype[np.intp]]:
    """Build exact positional indices without fixed-width affine overflow."""
    return checked_affine(outer_map.offset - inner_origin, outer_map.stride, outer_map.index_array)


def _positions_for_axis(
    outer: IndexTransform, outer_map: OutputIndexMap, inner_origin: int
) -> int | np.ndarray[Any, np.dtype[np.intp]]:
    """Convert one intermediate coordinate map to inner-array positions."""
    if isinstance(outer_map, ConstantMap):
        return outer_map.offset - inner_origin
    if isinstance(outer_map, DimensionMap):
        return _dimension_positions(outer, outer_map, inner_origin)
    return _array_positions(outer_map, inner_origin)


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
    if any(extent == 0 for extent in outer.domain.shape):
        # The empty map is singleton on every non-empty axis: it varies over no
        # axis at all, and the emptiness lives in the domain emitted alongside.
        empty_shape = tuple(0 if extent == 0 else 1 for extent in outer.domain.shape)
        return ArrayMap(
            index_array=np.empty(empty_shape, dtype=arr_i.dtype),
            offset=inner_map.offset,
            stride=inner_map.stride,
        )

    positions = tuple(
        0 if size == 1 else _positions_for_axis(outer, outer_map, origin)
        for outer_map, origin, size in zip(outer.output, inner_origin, arr_i.shape, strict=True)
    )
    # A gather narrowed to one coordinate — scalar or all-singleton — is the
    # ConstantMap it equals; `array_map_or_constant` normalizes both.
    gathered = np.asarray(arr_i[positions])
    if gathered.ndim == 0:
        return ConstantMap(offset=checked_affine(inner_map.offset, inner_map.stride, int(gathered)))
    return array_map_or_constant(gathered, offset=inner_map.offset, stride=inner_map.stride)
