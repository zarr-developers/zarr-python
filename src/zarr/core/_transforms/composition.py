from __future__ import annotations

import numpy as np

from zarr.core._transforms.output_map import ArrayMap, ConstantMap, DimensionMap, OutputIndexMap
from zarr.core._transforms.transform import IndexTransform


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

    result_output = [_compose_single(outer, inner_map) for inner_map in inner.output]

    return IndexTransform(domain=outer.domain, output=tuple(result_output))


def _compose_single(outer: IndexTransform, inner_map: OutputIndexMap) -> OutputIndexMap:
    """Compose a single inner output map with the full outer transform."""
    if isinstance(inner_map, ConstantMap):
        return ConstantMap(offset=inner_map.offset)

    if isinstance(inner_map, DimensionMap):
        return _compose_dimension(outer, inner_map)

    if isinstance(inner_map, ArrayMap):
        return _compose_array(outer, inner_map)

    raise TypeError(f"Unknown output map type: {type(inner_map)}")  # pragma: no cover


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

    if isinstance(outer_map, ArrayMap):
        return ArrayMap(
            index_array=outer_map.index_array,
            input_dimensions=outer_map.input_dimensions,
            offset=offset_i + stride_i * outer_map.offset,
            stride=stride_i * outer_map.stride,
        )

    raise TypeError(f"Unknown output map type: {type(outer_map)}")  # pragma: no cover


def _compose_array(outer: IndexTransform, inner_map: ArrayMap) -> OutputIndexMap:
    """Compose when inner is an ArrayMap.

    storage = offset_i + stride_i * arr_i[intermediate[input_dimensions[0]],
                                          intermediate[input_dimensions[1]], ...]

    For each axis k of arr_i, the corresponding intermediate dim is
    inner_map.input_dimensions[k] = d. We need to evaluate arr_i over the
    product of `outer.output[d]` for each such d.

    All-constant outer: collapse to a single ConstantMap.

    Single 1-D inner array, single outer output: evaluate arr_i along the
    one outer output's parameterization.
    """
    arr_i = inner_map.index_array
    offset_i = inner_map.offset
    stride_i = inner_map.stride
    in_dims_i = inner_map.input_dimensions

    # All-constant outer: arr_i is evaluated at a single fixed point.
    if all(isinstance(m, ConstantMap) for m in outer.output):
        idx = tuple(outer.output[d].offset for d in in_dims_i)
        value = int(arr_i[idx])
        return ConstantMap(offset=offset_i + stride_i * value)

    # 1-D inner array, single referenced outer output.
    if len(in_dims_i) == 1:
        dim_i = in_dims_i[0]
        outer_map = outer.output[dim_i]

        if isinstance(outer_map, DimensionMap):
            # Evaluate arr_i at the outer DimensionMap's range.
            input_d = outer_map.input_dimension
            input_lo = outer.domain.inclusive_min[input_d]
            input_hi = outer.domain.exclusive_max[input_d]
            user_indices = np.arange(input_lo, input_hi, dtype=np.intp)
            intermediate_vals = outer_map.offset + outer_map.stride * user_indices
            new_arr = arr_i[intermediate_vals]
            return ArrayMap(
                index_array=new_arr,
                input_dimensions=(input_d,),
                offset=offset_i,
                stride=stride_i,
            )

        if isinstance(outer_map, ArrayMap):
            # Evaluate arr_i at outer's array values; new array inherits outer's
            # parameterization.
            intermediate_vals = outer_map.offset + outer_map.stride * outer_map.index_array
            new_arr = arr_i[intermediate_vals]
            return ArrayMap(
                index_array=new_arr,
                input_dimensions=outer_map.input_dimensions,
                offset=offset_i,
                stride=stride_i,
            )

    # General multi-dim case: not yet implemented.
    raise NotImplementedError(
        "Composing a multi-dimensional inner array map with non-constant outer maps "
        "is not yet supported."
    )
