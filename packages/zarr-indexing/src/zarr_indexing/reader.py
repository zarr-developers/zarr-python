"""Backend reader protocol and built-in system-memory implementations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np

from zarr_indexing._affine import checked_affine
from zarr_indexing.chunk_resolution import ChunkProjection  # noqa: TC001 (runtime annotation)
from zarr_indexing.output_map import ArrayMap, ConstantMap, DimensionMap, OutputIndexMap
from zarr_indexing.transform import (
    IndexTransform,
)

__all__ = [
    "BasicReader",
    "NumPyReader",
    "ReadContext",
    "Reader",
    "UnitStepReader",
    "basic_reader",
    "numpy_reader",
    "unit_step_reader",
]


@dataclass(frozen=True, slots=True)
class ReadContext:
    """A source-global transform and optional projection for a partitioned read.

    Examples
    --------
    >>> transform = IndexTransform.from_shape((6,))[1:5:2]
    >>> context = ReadContext(transform)
    >>> context.transform.domain.shape
    (2,)
    >>> context.projection is None
    True
    """

    transform: IndexTransform
    """Maps zero-origin output-buffer coordinates to global coordinates in the source."""

    projection: ChunkProjection | None = None
    """The partition plan when this read is one part of a partitioned view, else `None`."""


class Reader(Protocol):
    """Backend adapter that fills supplied system-memory result buffers.

    A reader may be shared by every view and part derived from one
    [`LazyArray`][zarr_indexing.lazy_array.LazyArray]. Part reads may run
    concurrently, so a stateful implementation must synchronize its own
    mutable state. `LazyArray` deliberately adds no serialization.

    Examples
    --------
    The protocol is not `runtime_checkable`; an object satisfies it by
    exposing a conforming `read_into`, as `basic_reader` does:

    >>> transform = IndexTransform.from_shape((6,))[1:5:2]
    >>> source = np.arange(6)
    >>> out = np.empty(transform.domain.shape, dtype=source.dtype)
    >>> basic_reader.read_into(source, ReadContext(transform), out)
    >>> out.tolist()
    [1, 3]
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
        partition plan: its `chunk_transform` is chunk-local, its
        `cell_transform` describes result placement, and its `chunk_domain`
        describes the grid cell. Fill every cell in place, preserving the
        transform's exact values, order, and dtype, then return `None`. Do not
        replace or retain `out`; it may be a strided writable view rather than
        an owning array.

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

    Examples
    --------
    >>> transform = IndexTransform.from_shape((6,))[1:5:2]
    >>> source = np.arange(6)
    >>> out = np.empty(transform.domain.shape, dtype=source.dtype)
    >>> BasicReader().read_into(source, ReadContext(transform), out)
    >>> out.tolist() == source[1:5:2].tolist()
    True
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

    Examples
    --------
    >>> transform = IndexTransform.from_shape((3, 4))[::2, 1:3]
    >>> source = np.arange(12).reshape(3, 4)
    >>> out = np.empty(transform.domain.shape, dtype=source.dtype)
    >>> NumPyReader().read_into(source, ReadContext(transform), out)
    >>> out.tolist()
    [[1, 2], [9, 10]]
    """

    __slots__ = ()

    def read_into(self, source: Any, context: ReadContext, out: Any, /) -> None:
        """Read one transform through a narrowed slab into `out`."""
        transform = context.transform
        key, residual = _decompose_basic(transform)
        block = np.asanyarray(source[key])
        out[...] = _lower(block, residual)


class UnitStepReader:
    """Reader for sources whose basic indexing accepts only step-1 slices.

    Each transform is decomposed into the smallest enclosing ascending
    unit-step slab and a residual transform, so the source only ever receives
    `slice(start, stop, 1)` on every axis — the one form an API without
    general strided reads (an FFI binding, an HTTP range endpoint) supports.
    `BasicReader` instead pushes strided and reversed slices down, which
    reads less but asks more of the source.

    The residual lowering applies strides, reversals, and gathers to the
    in-memory block, so a strided selection over-reads its cover by the
    stride factor. Partitioning the wrapping
    [`LazyArray`][zarr_indexing.lazy_array.LazyArray] (`with_parts`) bounds
    each cover by a part.

    Slice results must permit conversion to NumPy system memory, exactly as
    for `BasicReader`.

    Examples
    --------
    The source below is only ever asked for step-1 slices — here the cover
    `slice(1, 4, 1)` — and the stride is replayed against the block:

    >>> transform = IndexTransform.from_shape((6,))[1:5:2]
    >>> source = np.arange(6)
    >>> out = np.empty(transform.domain.shape, dtype=source.dtype)
    >>> UnitStepReader().read_into(source, ReadContext(transform), out)
    >>> out.tolist()
    [1, 3]
    """

    __slots__ = ()

    def read_into(self, source: Any, context: ReadContext, out: Any, /) -> None:
        """Read one transform through an ascending unit-step slab into `out`."""
        transform = context.transform
        key, residual = _decompose_unit_step(transform)
        block = np.asanyarray(source[key])
        out[...] = _lower(block, residual)


basic_reader: Final = BasicReader()
numpy_reader: Final = NumPyReader()
unit_step_reader: Final = UnitStepReader()


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
    if math.prod(broadcast_shape) == 0:
        # A zero-extent broadcast axis makes the correlated block empty — for
        # example an ArrayMap composed over an empty domain, which the package
        # promises resolves like any other. The per-axis reshape below cannot
        # express that block (a 0-size array does not reshape to the non-zero
        # singleton axes), and there is no coordinate to produce anyway.
        return np.empty(0, dtype=np.intp)
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
    if math.prod(transform.domain.shape) == 0:
        # An empty domain selects nothing, and its maps may legitimately be
        # empty along the vanished axes (an ArrayMap composed over an empty
        # domain, which the package promises resolves like any other). The
        # resolvers below cannot evaluate such maps — and have no reason to.
        return np.empty(transform.domain.shape, dtype=np.asanyarray(array).dtype)
    if transform.index_array_structure == "general":
        result = _lower_general(array, transform)
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
            axis = m.dependent_axis
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


def _lower_general(array: Any, transform: IndexTransform) -> Any:
    """Flatten the index-array axes, gather the points once, reshape back.

    The general path for every index-array structure the orthogonal resolver
    cannot take: correlated (`vindex`) maps, maps sharing an input axis (a
    diagonal gather), and mixtures of correlated and orthogonal maps. All
    index arrays are treated as lookup tables over the joint block of
    non-slice axes: the corresponding storage axes are moved to the front and
    flattened, the per-point coordinates are converted to offsets into that
    flat axis with row-major strides, and a single `take` collects them.
    """
    outputs = transform.output
    correlated_dims = [d for d, m in enumerate(outputs) if isinstance(m, ArrayMap)]

    slice_input_dims = {m.input_dimension for m in outputs if isinstance(m, DimensionMap)}
    broadcast_axes = [d for d in range(transform.input_rank) if d not in slice_input_dims]
    broadcast_shape = tuple(transform.domain.shape[d] for d in broadcast_axes)

    for d in correlated_dims:
        arr_map = outputs[d]
        assert isinstance(arr_map, ArrayMap)
        # The axes the array varies over (its non-singleton axes; see
        # transform._array_map_dependency_axes) must all live in the block.
        dependency = (axis for axis, size in enumerate(arr_map.index_array.shape) if size > 1)
        if any(a not in broadcast_axes for a in dependency):
            # Reachable only by hand-building a transform: no selection binds
            # the same input axis to both a slice map and an index array.
            raise NotImplementedError(
                "resolving a transform whose index array varies over an input "
                "dimension also bound by a slice map is not supported"
            )

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


def _push_unit_slice_for_dimension_map(
    m: DimensionMap, transform: IndexTransform
) -> tuple[slice, DimensionMap]:
    """The unit-step slice covering a `DimensionMap`, and its block-local map.

    Strides and reversals stay in the residual: the source is only ever asked
    for a contiguous ascending slice, and the original stride is replayed
    against the in-memory block. The cover therefore over-reads a strided
    selection by its stride factor, which is the price of a source that
    accepts nothing but `slice(start, stop, 1)`.
    """
    d = m.input_dimension
    lo = transform.domain.inclusive_min[d]
    hi = max(transform.domain.exclusive_max[d], lo)
    if hi == lo:
        return slice(0, 0, 1), DimensionMap(input_dimension=d, offset=-lo, stride=1)
    first = checked_affine(m.offset, m.stride, lo)
    if m.stride == 0:
        return (
            slice(first, first + 1, 1),
            DimensionMap(input_dimension=d, offset=0, stride=0),
        )
    last = checked_affine(m.offset, m.stride, hi - 1)
    origin = min(first, last)
    return (
        slice(origin, max(first, last) + 1, 1),
        DimensionMap(input_dimension=d, offset=m.offset - origin, stride=m.stride),
    )


def _decompose_basic(transform: IndexTransform) -> tuple[tuple[slice, ...], IndexTransform]:
    return _decompose(transform, _push_slice_for_dimension_map)


def _decompose_unit_step(transform: IndexTransform) -> tuple[tuple[slice, ...], IndexTransform]:
    return _decompose(transform, _push_unit_slice_for_dimension_map)


def _decompose(
    transform: IndexTransform,
    push_dimension_map: Callable[[DimensionMap, IndexTransform], tuple[slice, DimensionMap]],
) -> tuple[tuple[slice, ...], IndexTransform]:
    key: list[slice] = []
    residual: list[OutputIndexMap] = []
    for output_map in transform.output:
        if isinstance(output_map, ConstantMap):
            coordinate = checked_affine(output_map.offset, 0, 0)
            key.append(slice(coordinate, coordinate + 1, 1))
            residual.append(ConstantMap(offset=0))
        elif isinstance(output_map, DimensionMap):
            pushed, local = push_dimension_map(output_map, transform)
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
            residual.append(ArrayMap(index_array=local_index))
    return tuple(key), IndexTransform(domain=transform.domain, output=tuple(residual))
