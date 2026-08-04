"""Backend reader protocol and built-in system-memory implementations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Protocol

import numpy as np

from zarr_indexing.affine import checked_affine
from zarr_indexing.output_map import ArrayMap, ConstantMap, DimensionMap, OutputIndexMap
from zarr_indexing.transform import IndexTransform, array_map_dependent_axis

if TYPE_CHECKING:
    from zarr_indexing.chunk_resolution import ChunkProjection

__all__ = [
    "BasicReader",
    "NumPyReader",
    "ReadContext",
    "Reader",
    "basic_reader",
    "numpy_reader",
]


@dataclass(frozen=True, slots=True)
class ReadContext:
    """The complete source-global transform and optional local chunk projection."""

    transform: IndexTransform
    projection: ChunkProjection | None = None


class Reader(Protocol):
    """Backend adapter that fills supplied system-memory result buffers.

    A reader may be shared by every view and part derived from one
    [`LazyArray`][zarr_indexing.lazy_array.LazyArray]. Part reads may run
    concurrently, so a stateful implementation must synchronize its own
    mutable state. `LazyArray` deliberately adds no serialization.
    """

    def read_into(
        self,
        source: Any,
        context: ReadContext,
        out: np.ndarray[Any, Any],
        /,
    ) -> None:
        """Fill `out` with the exact source values selected by `context`.

        `context.transform` maps zero-origin coordinates in the output buffer
        to global coordinates in `source`, and its domain shape equals
        `out.shape`. `context.projection`, when present, is the corresponding
        chunk-local plan. Fill every cell in place, preserving the transform's
        exact values, order, and dtype, then return `None`. Do not replace or
        retain `out`; it may be a strided writable view rather than an owning
        array.

        Backend exceptions propagate unchanged. Because callers may resolve
        parts concurrently through the same reader object, stateful readers
        are responsible for synchronizing their own state.
        """
        ...


class BasicReader:
    """Reader for system-memory sources exposing basic integer/slice indexing.

    Each transform is decomposed into the smallest enclosing positive-slice
    slab and a residual transform. The slab is read once with basic indexing,
    so fancy or negative-step selections may over-read, and the residual is
    then lowered through NumPy system-memory operations into the supplied
    buffer.

    Slice results must permit conversion to NumPy system memory. Device arrays
    that reject implicit conversion require a custom reader responsible for
    transferring values into the supplied system-memory output buffer.
    """

    __slots__ = ()

    def read_into(self, source: Any, context: ReadContext, out: Any, /) -> None:
        """Read one transform through a positive-slice slab and residual lowering."""
        transform = context.transform
        key, residual = _decompose_basic(transform)
        block = np.asanyarray(source[key])
        out[...] = _lower(block, residual)


class NumPyReader:
    """Reader optimized for NumPy system-memory arrays.

    This is the reader selected by
    [`LazyArray.from_numpy`][zarr_indexing.lazy_array.LazyArray.from_numpy]. It
    applies the complete transform with NumPy operations and is applicable to
    `numpy.ndarray` sources, including `numpy.ma.MaskedArray`.
    """

    __slots__ = ()

    def read_into(self, source: Any, context: ReadContext, out: Any, /) -> None:
        """Read one transform through a narrowed slab into `out`."""
        transform = context.transform
        key, residual = _decompose_basic(transform)
        block = np.asanyarray(source[key])
        out[...] = _lower(block, residual)


basic_reader: Final = BasicReader()
numpy_reader: Final = NumPyReader()


def _take(array: Any, indices: np.ndarray[Any, np.dtype[np.intp]], axis: int) -> Any:
    return np.take(array, indices, axis=axis)


def _reshape(array: Any, shape: tuple[int, ...]) -> Any:
    return np.reshape(array, shape)


def _transpose(array: Any, permutation: tuple[int, ...]) -> Any:
    return np.transpose(array, permutation)


def _expand_dims(array: Any, axis: int) -> Any:
    return np.expand_dims(array, axis)


def _dimension_map_coords(
    m: DimensionMap, transform: IndexTransform
) -> np.ndarray[Any, np.dtype[np.intp]]:
    """The storage coordinates a DimensionMap enumerates, in view order."""
    d = m.input_dimension
    lo = transform.domain.inclusive_min[d]
    hi = transform.domain.exclusive_max[d]
    extent = hi - lo
    if extent == 0:
        return np.empty((0,), dtype=np.intp)
    first = checked_affine(m.offset, m.stride, lo)
    return checked_affine(first, m.stride, np.arange(extent, dtype=np.intp))


def _array_map_coords(m: ArrayMap) -> np.ndarray[Any, np.dtype[np.intp]]:
    """The storage coordinates an ArrayMap enumerates, flattened."""
    return checked_affine(m.offset, m.stride, m.index_array).reshape(-1)


def _correlated_map_coords(
    m: ArrayMap, broadcast_axes: list[int], broadcast_shape: tuple[int, ...], input_rank: int
) -> np.ndarray[Any, np.dtype[np.intp]]:
    """One storage coordinate per point of the correlated block, flattened.

    A correlated `ArrayMap`'s index array carries the transform's full input
    rank, with a singleton on every axis it does not vary over — including
    broadcast axes it shares with the *other* correlated maps but is itself
    constant along. Flattening it directly would then yield fewer coordinates
    than there are points, so it is reduced to the broadcast block and
    broadcast up to it explicitly.
    """
    coords = checked_affine(m.offset, m.stride, m.index_array)
    if coords.ndim == input_rank:
        # Drop the axes bound by a slice, which the map is singleton along.
        # Removing size-1 axes by reshape preserves element order wherever they
        # sit, so no transpose is needed.
        coords = coords.reshape(tuple(coords.shape[axis] for axis in broadcast_axes))
    return np.ascontiguousarray(np.broadcast_to(coords, broadcast_shape)).reshape(-1)


def _restore_domain_axis_order(
    result: Any, axis_input_dims: list[int], domain_shape: tuple[int, ...]
) -> Any:
    """Permute `result`'s axes into input-domain order, restoring dropped axes.

    `axis_input_dims[k]` is the input (domain) dimension that axis `k` of
    `result` corresponds to. Axes are permuted so that they appear in increasing
    domain-dimension order, and any domain dimension no output map depends on is
    reinserted at **its own extent**.

    An unreferenced dimension is not always a singleton. A `vindex` coordinate
    array with a broadcast axis it does not vary over leaves that axis in the
    domain; a later basic index that consumes the axis the array *does* vary
    over collapses the map to a `ConstantMap` and leaves the broadcast axis
    behind, with whatever extent the basic index gave it — including 0. Every
    position along such an axis holds the same values, so it is restored by
    repeating the block, and an extent of 0 restores an empty result rather than
    fabricating a row.

    This is also where NumPy's advanced-index placement rules are absorbed:
    whatever order the gather produced, the lowered result always comes back in
    the view's own axis order.
    """
    if len(set(axis_input_dims)) != len(axis_input_dims):
        raise NotImplementedError(
            "resolving a transform whose output maps share an input dimension "
            "(a diagonal view) is not supported"
        )
    order = sorted(range(len(axis_input_dims)), key=lambda k: axis_input_dims[k])
    if order != list(range(len(order))):
        result = _transpose(result, tuple(order))
    covered = set(axis_input_dims)
    for dim, extent in enumerate(domain_shape):
        if dim in covered:
            continue
        result = _expand_dims(result, dim)
        if extent != 1:
            result = _take(result, np.zeros(extent, dtype=np.intp), axis=dim)
    return result


def _lower(array: Any, transform: IndexTransform) -> Any:
    """Lower a transform to one pass of array operations over `array`.

    Every read, partitioned or not, goes through this function. The result is
    always in the transform's own domain axis order and of exactly its domain
    shape.
    """
    if any(isinstance(m, ArrayMap) and m.input_dimension is None for m in transform.output):
        result = _lower_correlated(array, transform)
    else:
        result = _lower_orthogonal(array, transform)
    if isinstance(result, np.generic):
        # Basic indexing every axis of a NumPy array yields a scalar; `result()`
        # documents a zero-dimensional array.
        return np.asarray(result)
    return result


def _lower_orthogonal(array: Any, transform: IndexTransform) -> Any:
    """Basic slicing plus one `take` per fancy-indexed axis (an outer product).

    Orthogonal `ArrayMap`s vary over distinct input axes, so gathering them one
    axis at a time is exact — a `take` along one storage axis leaves every other
    axis's coordinates untouched.
    """
    outputs = transform.output
    gathered: dict[int, np.ndarray[Any, np.dtype[np.intp]]] = {}
    for out_dim, m in enumerate(outputs):
        if isinstance(m, ArrayMap):
            gathered[out_dim] = _array_map_coords(m)
        elif isinstance(m, DimensionMap) and m.stride <= 0:
            # Reversing and repeating maps have no positive-step slice; gather them.
            gathered[out_dim] = _dimension_map_coords(m, transform)

    result = array
    for out_dim, coords in gathered.items():
        result = _take(result, coords, axis=out_dim)

    selection: list[Any] = []
    axis_input_dims: list[int] = []
    for out_dim, m in enumerate(outputs):
        if isinstance(m, ConstantMap):
            selection.append(m.offset)
            continue
        if out_dim in gathered:
            selection.append(slice(None))
        else:
            assert isinstance(m, DimensionMap)
            d = m.input_dimension
            lo = transform.domain.inclusive_min[d]
            hi = transform.domain.exclusive_max[d]
            selection.append(slice(m.offset + m.stride * lo, m.offset + m.stride * hi, m.stride))
        if isinstance(m, ArrayMap):
            axis = array_map_dependent_axis(m)
            if axis is None:
                raise NotImplementedError(
                    "resolving an orthogonal ArrayMap that varies over no input "
                    "dimension is not supported; such a map should have been "
                    "collapsed to a ConstantMap"
                )
            axis_input_dims.append(axis)
        else:
            axis_input_dims.append(m.input_dimension)
    result = result[tuple(selection)]
    return _restore_domain_axis_order(result, axis_input_dims, transform.domain.shape)


def _lower_correlated(array: Any, transform: IndexTransform) -> Any:
    """Flatten the correlated axes, gather the points once, reshape back.

    Correlated (`vindex`) `ArrayMap`s address a list of *points*, not an outer
    product, so they cannot be gathered one axis at a time. The correlated
    storage axes are moved to the front and flattened, the per-point coordinates
    are converted to offsets into that flat axis with row-major strides, and a
    single `take` collects them.
    """
    outputs = transform.output
    correlated_dims = [
        d for d, m in enumerate(outputs) if isinstance(m, ArrayMap) and m.input_dimension is None
    ]
    if any(isinstance(m, ArrayMap) and m.input_dimension is not None for m in outputs):
        raise NotImplementedError(
            "resolving a transform with both correlated and orthogonal ArrayMaps is not supported"
        )

    slice_input_dims = {m.input_dimension for m in outputs if isinstance(m, DimensionMap)}
    broadcast_axes = [d for d in range(transform.input_rank) if d not in slice_input_dims]
    broadcast_shape = tuple(transform.domain.shape[d] for d in broadcast_axes)

    # Gather any reversing or repeating slice axis first, then take the basic-slice
    # cut. The correlated axes keep their full extent: their coordinates are absolute.
    gathered: dict[int, np.ndarray[Any, np.dtype[np.intp]]] = {
        d: _dimension_map_coords(m, transform)
        for d, m in enumerate(outputs)
        if isinstance(m, DimensionMap) and m.stride <= 0
    }
    result = array
    for out_dim, coords in gathered.items():
        result = _take(result, coords, axis=out_dim)

    selection: list[Any] = []
    residual_axis_dims: list[int] = []
    correlated_positions: list[int] = []
    residual_positions: list[int] = []
    axis = 0
    for out_dim, m in enumerate(outputs):
        if isinstance(m, ConstantMap):
            selection.append(m.offset)
            continue
        if out_dim in correlated_dims:
            selection.append(slice(None))
            correlated_positions.append(axis)
        elif out_dim in gathered:
            selection.append(slice(None))
            residual_positions.append(axis)
            assert isinstance(m, DimensionMap)
            residual_axis_dims.append(m.input_dimension)
        else:
            assert isinstance(m, DimensionMap)
            d = m.input_dimension
            lo = transform.domain.inclusive_min[d]
            hi = transform.domain.exclusive_max[d]
            selection.append(slice(m.offset + m.stride * lo, m.offset + m.stride * hi, m.stride))
            residual_positions.append(axis)
            residual_axis_dims.append(d)
        axis += 1
    result = result[tuple(selection)]

    # Correlated axes to the front, in output order, so the flattening strides
    # below match the order the coordinates are combined in.
    perm = tuple(correlated_positions) + tuple(residual_positions)
    if perm != tuple(range(len(perm))):
        result = _transpose(result, perm)

    n_corr = len(correlated_dims)
    corr_sizes = tuple(int(s) for s in result.shape[:n_corr])
    tail_shape = tuple(int(s) for s in result.shape[n_corr:])
    result = _reshape(result, (math.prod(corr_sizes), *tail_shape))

    flat_index = np.zeros(math.prod(broadcast_shape), dtype=np.intp)
    stride = 1
    for position in range(n_corr - 1, -1, -1):
        m = outputs[correlated_dims[position]]
        assert isinstance(m, ArrayMap)
        flat_index = flat_index + (
            _correlated_map_coords(m, broadcast_axes, broadcast_shape, transform.input_rank)
            * stride
        )
        stride *= corr_sizes[position]

    result = _take(result, flat_index, axis=0)
    result = _reshape(result, broadcast_shape + tail_shape)
    return _restore_domain_axis_order(
        result, list(broadcast_axes) + residual_axis_dims, transform.domain.shape
    )


def _push_slice_for_dimension_map(
    m: DimensionMap, transform: IndexTransform
) -> tuple[slice, DimensionMap]:
    """The positive-step slice covering a `DimensionMap`, and its block-local map.

    A negative step is read forwards and reversed by the residual: a source is
    only ever asked for a slice that walks upwards, which is the one form every
    array-like agrees on.
    """
    d = m.input_dimension
    lo = transform.domain.inclusive_min[d]
    hi = max(transform.domain.exclusive_max[d], lo)
    if hi == lo:
        return slice(0, 0, 1), DimensionMap(input_dimension=d, offset=-lo, stride=1)
    first = checked_affine(m.offset, m.stride, lo)
    last = checked_affine(m.offset, m.stride, hi - 1)
    if m.stride > 0:
        return (
            slice(first, last + 1, m.stride),
            DimensionMap(input_dimension=d, offset=-lo, stride=1),
        )
    if m.stride == 0:
        return (
            slice(first, first + 1, 1),
            DimensionMap(input_dimension=d, offset=0, stride=0),
        )
    # Descending: the block holds the same coordinates in ascending order, so
    # the residual walks it backwards from the last block position.
    return (
        slice(last, first + 1, -m.stride),
        DimensionMap(input_dimension=d, offset=hi - 1, stride=-1),
    )


def _decompose_basic(transform: IndexTransform) -> tuple[tuple[slice, ...], IndexTransform]:
    key: list[slice] = []
    residual: list[OutputIndexMap] = []
    for output_map in transform.output:
        if isinstance(output_map, ConstantMap):
            coordinate = checked_affine(output_map.offset, 0, 0)
            key.append(slice(coordinate, coordinate + 1, 1))
            residual.append(ConstantMap(offset=0))
        elif isinstance(output_map, DimensionMap):
            pushed, local = _push_slice_for_dimension_map(output_map, transform)
            key.append(pushed)
            residual.append(local)
        else:
            coordinates = checked_affine(
                output_map.offset, output_map.stride, output_map.index_array
            )
            if coordinates.size == 0:
                key.append(slice(0, 0, 1))
                local_index = coordinates
            else:
                origin = int(coordinates.min())
                key.append(slice(origin, int(coordinates.max()) + 1, 1))
                local_index = checked_affine(-origin, 1, coordinates)
            residual.append(
                ArrayMap(
                    index_array=local_index,
                    input_dimension=output_map.input_dimension,
                )
            )
    return tuple(key), IndexTransform(domain=transform.domain, output=tuple(residual))
