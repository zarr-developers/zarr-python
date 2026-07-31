"""`LazyArray` — TensorStore-style lazy indexing over any array-API-like array.

`LazyArray` wraps an existing array — NumPy, zarr, CuPy, anything with `shape`,
`dtype`, and `__getitem__` — and adds a `.lazy` accessor whose indexing
operations build up an [`IndexTransform`](transform.md) instead of reading data:

```python
view = LazyArray(source).lazy[10:50, ::2].lazy.oindex[[3, 1, 1], :]
view.shape          # known without touching the data
values = view.result()
```

Nothing is read until `result()` (or `__array__`, or an eager `__getitem__`).
Composition does not accumulate layers: a view of a view is still a single
transform.

Parts
-----
A `LazyArray` carries a **partitioning** of the array it wraps: a grid of boxes
that a read is broken into. `parts()` walks those boxes as they fall through the
view, yielding a [`Partition`](#zarr_indexing.lazy_array.Partition) per box —
the base coordinates of the box, a `LazyArray` covering exactly the cells of the
view that live in it, and where those cells belong in the result. `result()` is
built on that walk:

```python
for part in view.parts():
    out[part.out_selection] = part.view.result()
```

Each box is therefore read once, and whatever the source could not do itself is
applied to the block in memory.

The partitioning is discovered from the wrapped array at construction — first
`read_chunk_sizes` (zarr's clipped per-axis sizes, sharding-aware), then
`chunks`, read as per-axis sizes if its entries are sequences and as a uniform
box shape if they are integers. Those attribute names belong to the wrapped
array; this API refers only to parts. An array that advertises neither gets a
single whole-array part, and resolving it lowers the whole view in one pass
rather than materializing a block first.

`with_parts` replaces the partitioning without touching the data or the view:

```python
view.with_parts((64, 64))       # uniform boxes, tail clipped
view.with_parts_per_axis(((3, 3, 1),))   # explicit per-axis sizes
view.unpartitioned()           # one whole-array part; resolve in one shot
```

Repartitioning changes how the read is divided, not what `result()` returns.
Parts that do not align with the source's own boxes are permitted and can be
useful (to bound peak memory, or to batch small reads); they cost extra I/O but
do not affect correctness.

What the source is asked for
---------------------------
Sources differ in what their indexing surface accepts, so a read is negotiated
rather than assumed. Each wrapper carries an
[`IndexingSupport`](support.md) level, and every request is split in two: the
largest part of the selection that level can express is asked of the source in a
single call — through `oindex` or `vindex` when the source has them — and the
remainder is applied to the block that comes back, with NumPy.

The level is detected at construction (a NumPy array or zarr's
`oindex`/`vindex` pair reads as `VECTORIZED`, a source's own
`__zarr_indexing_support__` declaration is honored, and anything else is
`BASIC`, which is the only assumption that is always safe).
`with_indexing_support` overrides the detection:

```python
view.with_indexing_support(IndexingSupport.OUTER)
```

The level changes how much data crosses the boundary, never what `result()`
returns. Declaring `BASIC` on a NumPy array answers exactly what `VECTORIZED`
answers, having read the enclosing slab and gathered from it instead.

Boxes and queries
-----------------
A selection is either **rectangular** — an interval and a stride per dimension,
which is what basic indexing composes to at any depth — or a **query**, an
explicit list of coordinates, which is what `oindex`, `vindex`, and masks
produce and which subsequent basic indexing cannot undo. `is_box` reports the
category and `bounding_box()` reports the storage region touched: the exact
interval per dimension for a box, a hull for a query. A box is only *dense* in
that interval when every entry of `strides()` is 1. The distinction is
structural rather than an optimization; [the design
notes](../design-notes.md) describe why it matters to consumers of a selection.

The positional dialect
----------------------
Selections on `LazyArray` are **positional, NumPy-style**: index 0 is the first
element of the current view, `-1` is the last, boolean masks must match the
view's shape, and every index is bounds-checked against the view.

This differs deliberately from `zarr.Array.lazy[...]`, which exposes the
**literal** TensorStore dialect: a zarr view keeps the coordinate system of the
array it came from, so after `v = arr.lazy[10:50]` the first element of `v` is
`v[10]` and a negative index is out of bounds rather than counted from the end.
That dialect suits zarr, where a view's coordinates stay comparable with the
parent array's. `LazyArray` is a duck array and has to behave like the array it
wraps to be usable as a NumPy drop-in or as a dask source, so it re-zeroes its
coordinates on every view and uses positions. `zarr_indexing.boundary` performs
the translation between the two.

Two more NumPy rules the dialect keeps, in every mode:

- A scalar integer drops its axis. It is a basic index wherever it appears,
  applied before any advanced index rather than broadcast against one. So
  `lazy.oindex[0]` has the shape of `x[0]`, `lazy.oindex[0, [1, 2], :]` means
  `x[0][numpy.ix_([1, 2], ...)]`, and `lazy.oindex[0, 1, 2]` and
  `lazy.vindex[0, 1, 2]` are both zero-rank. Use a length-1 list to keep an
  axis.
- Advanced indices are placed as NumPy places them. For a `vindex` selection
  that leaves some axes unindexed, the gathered dimensions sit where the
  coordinate arrays sat when those arrays are adjacent, and lead when a slice
  separates them — so `lazy.vindex[..., i, j]` has shape
  `(x.shape[0], *broadcast)`, matching `x[..., i, j]`.

Materializing on fallback
-------------------------
`LazyArray` implements `__array__` but deliberately implements neither
`__array_ufunc__`/`__array_function__` nor `__array_namespace__`. A NumPy
*function* given a view therefore materializes the whole thing through
`__array__` and works on the resulting array: `numpy.sum(view)`,
`numpy.add(view, 1)` and `numpy.stack([view, view])` all do, and so does
`numpy.ones(view.shape) + view`, where the ndarray on the left dispatches.

Python's arithmetic *operators* do not: `view + 1` raises `TypeError`, because
the wrapper defines no arithmetic dunders and an `int` has nothing to dispatch
to. Both facts follow from the same intent — laziness here applies to indexing,
not to building a deferred compute graph — and a `LazyArray` is not a drop-in
for arithmetic on a large array either way. Use `.lazy[...]` to narrow the view
first, or pass the wrapper to `dask.array.from_array` so that dask owns the
compute graph.

Device caveat
-------------
Resolving a **partitioned** view builds its output buffer with NumPy, because
the resolver's scatter indices are host-side integer arrays. Wrapping a device
array that advertises parts therefore returns host memory, and any information
carried by the wrapped array's own type is lost with it — a `numpy.ma` mask is
the exception, kept by allocating a masked buffer. A wrapper with **no**
partitioning (`unpartitioned()`) skips that buffer and stays in the wrapped
array's own namespace.

`result()` never returns memory shared with the wrapped array. That is settled
from the arrays' memory bounds, which only NumPy exposes: a foreign namespace
whose basic indexing returns views is beyond reach here, and an unpartitioned
read of such a source can hand back one of its views.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from zarr_indexing.boundary import (
    SelectionMode,
    normalize_positional_selection,
    split_scalar_axes,
)
from zarr_indexing.chunk_resolution import (
    iter_chunk_transforms,
    sub_transform_to_selections,
)
from zarr_indexing.grid import DimensionGridLike, EdgeDimensionGrid, dimension_grids_from_chunks
from zarr_indexing.json import transform_to_canonical
from zarr_indexing.output_map import ArrayMap, ConstantMap, DimensionMap, OutputIndexMap
from zarr_indexing.support import (
    IndexingSupport,
    has_accessor,
    resolve_indexing_support,
)
from zarr_indexing.transform import (
    IndexTransform,
    array_map_dependent_axis,
    selection_to_transform,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    SelectFn = Callable[[Any, SelectionMode], "LazyArray"]

__all__ = ["LazyArray", "Partition"]

# Above this many bytes, the no-dask token fallback describes an array
# structurally instead of digesting its contents. See `_wrapped_token`.
_TOKEN_DIGEST_LIMIT = 1 << 20


class ArrayLike(Protocol):
    """The surface `LazyArray` needs from the array it wraps."""

    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def dtype(self) -> Any: ...
    def __getitem__(self, key: Any) -> Any: ...


# --------------------------------------------------------------------------- #
# Array-namespace helpers
# --------------------------------------------------------------------------- #


def _namespace(x: Any) -> Any:
    """Return `x`'s array-API namespace, or None if it does not declare one."""
    getter = getattr(x, "__array_namespace__", None)
    if getter is None:
        return None
    try:
        return getter()
    # A foreign namespace may fail in arbitrary ways; discovery must degrade
    # to "no information", never raise.
    except Exception:  # pragma: no cover - a namespace that refuses to load
        return None


def _take(x: Any, indices: np.ndarray[Any, np.dtype[np.intp]], axis: int) -> Any:
    """Gather along one axis, preferring the wrapped array's own namespace."""
    xp = _namespace(x)
    take = getattr(xp, "take", None)
    if take is not None:
        return take(x, xp.asarray(indices), axis=axis)
    return x[(slice(None),) * axis + (indices,)]


def _reshape(x: Any, shape: tuple[int, ...]) -> Any:
    xp = _namespace(x)
    reshape = getattr(xp, "reshape", None)
    if reshape is not None:
        return reshape(x, shape)
    return x.reshape(shape)


def _transpose(x: Any, perm: tuple[int, ...]) -> Any:
    xp = _namespace(x)
    for name in ("permute_dims", "transpose"):
        fn = getattr(xp, name, None)
        if fn is not None:
            return fn(x, perm)
    return np.transpose(x, perm)


def _expand_dims(x: Any, axis: int) -> Any:
    xp = _namespace(x)
    fn = getattr(xp, "expand_dims", None)
    if fn is not None:
        return fn(x, axis=axis)
    return np.expand_dims(x, axis=axis)


# --------------------------------------------------------------------------- #
# Partition discovery
# --------------------------------------------------------------------------- #


def _read_source_attribute(array: Any, name: str) -> Any:
    """Read a partition-describing attribute, treating any failure as "absent".

    Discovery inspects an object we did not write. A missing attribute is the
    common case, but zarr raises an `AttributeError` subclass from
    `read_chunk_sizes` on a lazy view, and other backends compute the attribute
    lazily and may fail for their own reasons. Any failure here means "this
    array does not advertise a partitioning", never a hard error.
    """
    try:
        return getattr(array, name, None)
    # Guarded properties (e.g. zarr's LazyViewError) and broken foreign
    # attributes may raise anything; discovery must degrade to None.
    except Exception:
        return None


def _discover_parts(array: Any, shape: tuple[int, ...]) -> tuple[EdgeDimensionGrid, ...] | None:
    """Resolve the partitioning advertised by `array`, or None for one whole part.

    Discovery parses external input: an attribute that does not describe a
    partitioning of `shape` means "this object does not advertise one I
    understand", and the array is treated as unpartitioned rather than rejected.
    A partitioning is an I/O strategy, so reading the whole array is always a
    correct fallback. `with_parts` is a public API and validates strictly.
    """
    declared = _read_source_attribute(array, "read_chunk_sizes")
    if declared is None:
        declared = _read_source_attribute(array, "chunks")
    if declared is None:
        return None
    try:
        return dimension_grids_from_chunks(declared, shape)
    except (ValueError, TypeError):
        return None


def _whole_array_grids(shape: tuple[int, ...]) -> tuple[EdgeDimensionGrid, ...]:
    """A partitioning with a single part covering the whole array."""
    return tuple(EdgeDimensionGrid((extent,) if extent > 0 else ()) for extent in shape)


# --------------------------------------------------------------------------- #
# The lowering engine
# --------------------------------------------------------------------------- #


def _is_identity_transform(transform: IndexTransform, shape: tuple[int, ...]) -> bool:
    """True when `transform` maps every coordinate of `shape` to itself.

    Structural rather than an `==` against `IndexTransform.from_shape`: an
    `ArrayMap` holds an ndarray, so equality on two transforms that both carry
    one would try to take the truth value of an array.
    """
    domain = transform.domain
    if domain.inclusive_min != (0,) * len(shape) or domain.exclusive_max != shape:
        return False
    if len(transform.output) != len(shape):
        return False
    return all(
        isinstance(m, DimensionMap) and m.input_dimension == i and m.offset == 0 and m.stride == 1
        for i, m in enumerate(transform.output)
    )


def _is_correlated(transform: IndexTransform) -> bool:
    """True when the transform gathers a list of points rather than an outer product."""
    return any(isinstance(m, ArrayMap) and m.input_dimension is None for m in transform.output)


def _dimension_map_coords(
    m: DimensionMap, transform: IndexTransform
) -> np.ndarray[Any, np.dtype[np.intp]]:
    """The storage coordinates a DimensionMap enumerates, in view order."""
    d = m.input_dimension
    lo = transform.domain.inclusive_min[d]
    hi = transform.domain.exclusive_max[d]
    return (m.offset + m.stride * np.arange(lo, hi, dtype=np.intp)).astype(np.intp)


def _array_map_coords(m: ArrayMap) -> np.ndarray[Any, np.dtype[np.intp]]:
    """The storage coordinates an ArrayMap enumerates, flattened."""
    return (m.offset + m.stride * m.index_array).astype(np.intp).reshape(-1)


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
    coords = (m.offset + m.stride * m.index_array).astype(np.intp)
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
    if _is_correlated(transform):
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
        elif isinstance(m, DimensionMap) and m.stride < 0:
            # A reversing map has no positive-step slice; gather it explicitly.
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

    # Gather any reversing slice axis first, then take the basic-slice cut. The
    # correlated axes keep their full extent: their coordinates are absolute.
    gathered: dict[int, np.ndarray[Any, np.dtype[np.intp]]] = {
        d: _dimension_map_coords(m, transform)
        for d, m in enumerate(outputs)
        if isinstance(m, DimensionMap) and m.stride < 0
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


# --------------------------------------------------------------------------- #
# Source-capability negotiation
# --------------------------------------------------------------------------- #
#
# `_lower` is exact but assumes the array under it accepts anything NumPy
# accepts. Most sources do not, and the ones that do would rather be asked for
# what is wanted in one request than be walked axis by axis. So a read is split
# in two: a *pushed key*, the largest part of the selection the source can
# express, and a *residual transform*, whatever is left, which `_lower` applies
# to the block that comes back. The split is the same idea as xarray's
# `decompose_indexer`, expressed over `IndexTransform` rather than over
# xarray's explicit indexer classes.
#
# The pushed key holds exactly one selector per storage dimension, always a
# slice or a one-dimensional integer array and never a bare integer, so the
# block has one axis per storage dimension whatever the level. That invariant is
# what lets the residual be built by rewriting output maps in place: only the
# coordinates change, from storage coordinates to positions within the block.


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
    first = m.offset + m.stride * lo
    last = m.offset + m.stride * (hi - 1)
    if m.stride > 0:
        return (
            slice(first, last + 1, m.stride),
            DimensionMap(input_dimension=d, offset=-lo, stride=1),
        )
    # Descending: the block holds the same coordinates in ascending order, so
    # the residual walks it backwards from the last block position.
    return (
        slice(last, first + 1, -m.stride),
        DimensionMap(input_dimension=d, offset=hi - 1, stride=-1),
    )


def _push_gain(coords: np.ndarray[Any, np.dtype[np.intp]]) -> float:
    """How much a slab read of `coords` would over-read, as a ratio.

    xarray's heuristic for choosing which axis keeps its array when only one
    can: the span of the coordinates divided by how many distinct ones there
    are. An axis whose coordinates are dense and contiguous scores 1 and loses
    nothing to a slab; a sparse axis scores high and is the one worth spending
    the single array slot on.
    """
    if coords.size == 0:
        return 0.0
    span = int(coords.max()) - int(coords.min()) + 1
    return span / int(np.unique(coords).size)


def _array_push_capacity(support: IndexingSupport, has_oindex: bool, wanted: int) -> int:
    """How many axes may carry an integer array in one request to the source."""
    if support is IndexingSupport.BASIC:
        return 0
    if support is IndexingSupport.OUTER_1VECTOR:
        return 1
    # OUTER and VECTORIZED both accept several array axes, but only through an
    # `oindex` accessor: a bare `__getitem__` key with two arrays means an outer
    # product to HDF5 and a correlated gather to NumPy, and nothing about the
    # source says which reading applies. One array is the key both readings
    # agree on, so that is the fallback.
    return wanted if has_oindex else 1


def _decompose(
    transform: IndexTransform, support: IndexingSupport, has_oindex: bool
) -> tuple[tuple[Any, ...], IndexTransform, int]:
    """Split `transform` into a key for the source and a transform for the block.

    Returns `(key, residual, n_arrays)`. `key` has one selector per storage
    dimension; `n_arrays` is how many of them are integer arrays, which decides
    whether the request goes to `oindex` or to `__getitem__`. Applying `residual`
    to the block the key returns is equivalent to applying `transform` to the
    source.
    """
    outputs = transform.output
    coords: dict[int, np.ndarray[Any, np.dtype[np.intp]]] = {
        d: (m.offset + m.stride * m.index_array).astype(np.intp)
        for d, m in enumerate(outputs)
        if isinstance(m, ArrayMap)
    }
    # An empty axis carries no coordinates to push and needs no array slot.
    array_axes = [d for d, c in coords.items() if c.size > 0]
    capacity = _array_push_capacity(support, has_oindex, len(array_axes))
    if capacity >= len(array_axes):
        pushed_axes = set(array_axes)
    else:
        ranked = sorted(array_axes, key=lambda d: _push_gain(coords[d]), reverse=True)
        pushed_axes = set(ranked[:capacity])

    key: list[Any] = []
    residual: list[OutputIndexMap] = []
    for d, m in enumerate(outputs):
        if isinstance(m, ConstantMap):
            # A slice rather than the integer: every axis survives into the
            # block, so residual output `d` addresses block axis `d`.
            key.append(slice(m.offset, m.offset + 1, 1))
            residual.append(ConstantMap(offset=0))
        elif isinstance(m, DimensionMap):
            pushed, local = _push_slice_for_dimension_map(m, transform)
            key.append(pushed)
            residual.append(local)
        else:
            full = coords[d]
            if full.size == 0:
                key.append(slice(0, 0, 1))
                local_index = full
            elif d in pushed_axes:
                # Sorted and deduplicated: a source is never asked for the same
                # coordinate twice, and the residual looks positions back up.
                unique = np.unique(full)
                key.append(unique)
                local_index = np.searchsorted(unique, full).astype(np.intp)
            else:
                # No array slot for this axis: read the slab it spans and
                # gather from that instead.
                origin = int(full.min())
                key.append(slice(origin, int(full.max()) + 1, 1))
                local_index = (full - origin).astype(np.intp)
            residual.append(ArrayMap(index_array=local_index, input_dimension=m.input_dimension))
    return (
        tuple(key),
        IndexTransform(domain=transform.domain, output=tuple(residual)),
        len(pushed_axes),
    )


def _point_key(transform: IndexTransform) -> tuple[Any, ...] | None:
    """The whole selection as one flat coordinate array per axis, or None.

    Defined only when every storage dimension is addressed by a correlated
    `ArrayMap` or fixed by a `ConstantMap`, which is what a plain `vindex[i, j,
    k]` composes to. The points are then exactly the cells the view wants, in
    the view's own order, so a source that can gather points has nothing left to
    do — whereas the outer superset those same points span can be very much
    larger than the points themselves.

    A selection with a sliced axis is not expressible this way without
    enumerating the slice, so it goes through `_decompose` instead.
    """
    shape = transform.domain.shape
    if math.prod(shape) == 0:
        return None
    key: list[Any] = []
    for m in transform.output:
        if isinstance(m, ConstantMap):
            key.append(np.full(math.prod(shape), m.offset, dtype=np.intp))
        elif isinstance(m, ArrayMap) and m.input_dimension is None:
            coords = (m.offset + m.stride * m.index_array).astype(np.intp)
            key.append(np.broadcast_to(coords, shape).reshape(-1))
        else:
            return None
    return tuple(key)


def _shift_key(key: tuple[Any, ...], window: tuple[slice, ...] | None) -> tuple[Any, ...]:
    """Rebase a key built in part-local coordinates onto the wrapped array."""
    if window is None:
        return key
    shifted: list[Any] = []
    for selector, w in zip(key, window, strict=True):
        if isinstance(selector, slice):
            shifted.append(slice(w.start + selector.start, w.start + selector.stop, selector.step))
        else:
            shifted.append(selector + w.start)
    return tuple(shifted)


def _gatherable(block: Any) -> Any:
    """The block the residual is applied to, as something that can be gathered from.

    The residual finishes the read with `take`, `reshape` and `transpose`, which
    is more than the `BASIC` floor asks of a *source*: that floor is `shape`,
    `dtype` and a `__getitem__` understanding integers and slices, and a source
    meeting exactly it may hand back a block that understands no more. Such a
    block is converted, so that the floor is a promise about the source alone.

    A block that is already a NumPy array, or that answers with an array-API
    namespace of its own, is left where it is: the namespace is how a device
    array keeps a read on its own device (see the module docstring's device
    caveat), and converting it would defeat that.
    """
    if isinstance(block, np.ndarray) or _namespace(block) is not None:
        return block
    return np.asarray(block)


def _read(
    source: Any,
    window: tuple[slice, ...] | None,
    transform: IndexTransform,
    support: IndexingSupport,
) -> Any:
    """Materialize `transform` over `source`, asking for no more than `support` allows.

    `window` restricts the read to one part's box, in the wrapped array's
    coordinates; it is folded into the key rather than materialized first, so a
    part costs one request whatever its selection looks like.
    """
    has_vindex = has_accessor(source, "vindex")
    has_oindex = has_accessor(source, "oindex")

    if support is IndexingSupport.VECTORIZED and _is_correlated(transform):
        points = _point_key(transform)
        if points is not None:
            gathered = source.vindex if has_vindex else source
            # The key is flat so that a `vindex` accepting only one-dimensional
            # coordinate lists is served too; the shape is restored here.
            block = _gatherable(gathered[_shift_key(points, window)])
            return _reshape(block, transform.domain.shape)

    key, residual, n_arrays = _decompose(transform, support, has_oindex)
    key = _shift_key(key, window)
    block = source.oindex[key] if n_arrays > 0 and has_oindex else source[key]
    return _lower(_gatherable(block), residual)


# --------------------------------------------------------------------------- #
# Partitions
# --------------------------------------------------------------------------- #


def _wrapped_base(array: Any) -> Any:
    """The concrete array at the bottom of a (possibly nested) wrapper."""
    while isinstance(array, LazyArray):
        array = array.array
    return array


def _detach(value: Any, source: Any) -> Any:
    """Return `value` sharing no memory with the array `source` reads from.

    An unpartitioned read of a selection that lowers to basic slicing hands back
    a *view* of the wrapped array, so writing to it would reach through and
    change the source — while the same read under any partitioning returns a
    fresh buffer. `result()` is answering "what does this view hold", not "lend
    me the storage", and its answer must not depend on how the read was divided.

    Sharing is settled two ways, because the source is not always something
    `may_share_memory` can be pointed at. When the wrapped array is itself a
    NumPy array the two are compared directly, and the copy is made only when
    they really overlap. When it is not — a duck array that stores its data in
    NumPy and returns views from `__getitem__`, which is exactly the `BASIC`
    source this package invites people to wrap — there is nothing to compare
    against, so the question becomes whether the result owns its buffer. A NumPy
    array that does not own its buffer is a view of something, and the only
    thing it can be a view of here is storage the source lent us.

    Deciding it that way means the answer never depends on knowing what the
    source is: memory is released only when sharing is *disproved*, never merely
    because it could not be established. An array whose namespace is foreign
    enough that it is not a NumPy array at all is still handed back as it comes
    (see the module docstring's device caveat).
    """
    if not isinstance(value, np.ndarray):
        return value
    base = _wrapped_base(source)
    if isinstance(base, np.ndarray):
        return value.copy() if np.may_share_memory(value, base) else value
    return value if value.flags.owndata else value.copy()


def _partition_out_selection(
    sub_transform: IndexTransform,
    out_indices: Any,
    out_shape: tuple[int, ...],
) -> tuple[Any, ...]:
    """Where a partition's values belong in an array of shape `out_shape`.

    A correlated sub-transform scatters through flat offsets into the row-major
    result; unravelling them turns that into an index tuple for the result's own
    shape, so callers never need a flat working buffer.

    The correlated scatter is taken directly rather than through
    `sub_transform_to_selections`, whose business is to match a scatter to the
    block a *NumPy chunk read* returns — which is not the block a part produces.
    A part is resolved by lowering its own transform, and that lowering keeps the
    points axis first; the bridge has to answer for NumPy's placement rules
    instead, and permutes its scatter to suit. Reading the bridge's answer here
    would apply a correction for a block this path never sees.
    """
    if not _is_correlated(sub_transform):
        return sub_transform_to_selections(sub_transform, out_indices)[1]
    if len(out_shape) == 0:
        return ()
    if out_indices is None:
        flat = np.arange(math.prod(sub_transform.domain.shape), dtype=np.intp)
    else:
        flat = np.asarray(out_indices, dtype=np.intp)
    return tuple(np.unravel_index(flat, out_shape))


def _out_selection_cell_count(selection: tuple[Any, ...], out_shape: tuple[int, ...]) -> int:
    """How many cells of an array of shape `out_shape` a `Partition.out_selection` writes.

    Counted from the selectors' own shapes, so nothing is read and no index
    array is materialized. The selectors are slices and integer arrays: the
    slices contribute their lengths, and the arrays broadcast against each other
    exactly as NumPy's advanced indexing broadcasts them, whether they arrive as
    an open mesh (`numpy.ix_`) or as parallel coordinates (`unravel_index`).
    """
    total = 1
    array_shapes: list[tuple[int, ...]] = []
    if len(selection) != len(out_shape):
        raise AssertionError(
            f"a partition addressed {len(selection)} of the view's {len(out_shape)} "
            "dimensions; this is a bug in zarr-indexing's partition walk"
        )
    for selector, extent in zip(selection, out_shape, strict=True):
        if isinstance(selector, slice):
            start: Any = selector.start
            stop: Any = selector.stop
            step: Any = selector.step
            if step is None and start is not None and stop is not None and 0 <= start <= stop:
                # The shape a partition walk actually produces: a concrete,
                # forward, in-bounds interval. Sized directly, so the common
                # path allocates neither a tuple nor a range.
                total *= int(stop) - int(start)
            else:
                total *= len(range(*selector.indices(extent)))
        else:
            array_shapes.append(tuple(int(s) for s in np.shape(selector)))
    if len(array_shapes) > 0:
        total *= math.prod(np.broadcast_shapes(*array_shapes))
    return total


def _covers_whole_part(local_transform: IndexTransform, part_shape: tuple[int, ...]) -> bool:
    """Whether a part-local transform addresses every cell of its part.

    Direction does not matter: a reversing view (`lazy[::-1]`) reads every cell
    of the box, just back to front, so a stride of -1 covers a part exactly as a
    stride of 1 does. Only the set of coordinates is asked about here, which is
    the same question `bounding_box()` and `strides()` answer.

    Conservative in one place: an `ArrayMap` could in principle enumerate a whole
    part, but checking that costs more than the answer is worth, so a
    fancy-indexed axis always reports incomplete.
    """
    domain = local_transform.domain
    for out_dim, m in enumerate(local_transform.output):
        extent = part_shape[out_dim]
        if isinstance(m, ConstantMap):
            if extent != 1 or m.offset != 0:
                return False
        elif isinstance(m, DimensionMap):
            if abs(m.stride) != 1:
                return False
            d = m.input_dimension
            lo = domain.inclusive_min[d]
            hi = domain.exclusive_max[d]
            if hi <= lo:
                # An empty walk covers only an empty part.
                if extent != 0:
                    return False
                continue
            first = m.offset + m.stride * lo
            last = m.offset + m.stride * (hi - 1)
            if min(first, last) != 0 or max(first, last) != extent - 1:
                return False
        else:
            return False
    return True


@dataclass(frozen=True, kw_only=True)
class Partition:
    """One box of a `LazyArray`'s partitioning, as it falls through the view.

    Yielded by [`LazyArray.parts`][zarr_indexing.lazy_array.LazyArray.parts].
    The parts of a view tile it exactly and disjointly: assembling every
    `view.result()` at its `out_selection` reproduces the whole view's
    `result()`, and each part can be resolved independently and concurrently.

    Attributes
    ----------
    base_coords
        Which box of the base partitioning this is, one coordinate per dimension
        of the wrapped array.
    box
        The box itself, in the global storage coordinates of the wrapped
        array: one `[inclusive_min, exclusive_max)` interval per dimension.
        `view.bounding_box()` is part-local and so cannot distinguish two
        parts; this attribute can, and together with `is_complete` it decides a
        read-modify-write.
    view
        A `LazyArray` covering exactly the cells of the view that live in this
        box. Resolving it reads the box once with basic slicing and applies the
        selection to the block in memory. Named `view` rather than `array`
        because `LazyArray.array` is the opposite thing — the raw wrapped source
        — and the two sat next to each other meaning inverses.
    out_selection
        Where `view.result()` belongs in an array of the whole view's shape — a
        NumPy index tuple with one entry per dimension of the view, usable
        directly as `out[part.out_selection] = ...`.
    is_complete
        Whether the view covers the whole box. Useful to a writer deciding
        between a blind overwrite and a read-modify-write. True for a reversing
        view, which reads every cell of the box back to front; false for every
        fancy-indexed axis, which is conservative rather than exact — so it may
        be `False` for a part it does in fact cover, but never `True` for one it
        does not.
    """

    base_coords: tuple[int, ...]
    box: tuple[tuple[int, int], ...]
    view: LazyArray
    out_selection: tuple[Any, ...]
    is_complete: bool


# --------------------------------------------------------------------------- #
# Tokenization
# --------------------------------------------------------------------------- #


def _wrapped_token(array: Any) -> Any:
    """A token for the wrapped array.

    In order of preference: the array's own `__dask_tokenize__`;
    `dask.base.tokenize` when dask is importable (imported lazily — this package
    never requires it); otherwise a local fallback that digests the contents of
    a small array.

    The two environments do not agree, and neither is a translation of the
    other: a token taken with dask installed is meaningless to a process without
    it, and the reverse. A token is an identifier within one process, not a
    portable name.

    Above `_TOKEN_DIGEST_LIMIT` the local fallback has nothing left to identify
    the contents with — reading them is exactly what a token call must not do —
    so it declines to claim equality at all and returns a value that matches
    nothing, including itself. A cache keyed on it misses; the alternative, a
    structural description, is a cache that hands one array's result to a
    different array of the same shape and dtype.
    """
    hook = getattr(array, "__dask_tokenize__", None)
    if hook is not None:
        try:
            return hook()
        # A token must never raise; fall through to the structural fallback.
        except Exception:  # pragma: no cover - a hook that refuses to run
            pass
    try:
        # dask is an optional peer, never a dependency of this package, so it is
        # imported here and its absence is ordinary.
        from dask.base import tokenize  # pyright: ignore[reportMissingImports]
    except ImportError:
        pass
    else:
        return tokenize(array)

    shape = tuple(int(s) for s in getattr(array, "shape", ()))
    dtype = getattr(array, "dtype", None)
    structural = (type(array).__qualname__, shape, str(dtype))
    # A token nothing can equal, for when the contents cannot be identified. It
    # is the shape and dtype that would otherwise be mistaken for an identity,
    # so they are kept alongside it for a reader looking at a graph.
    unidentified = (*structural, "unidentified", uuid.uuid4().hex)

    # Decide whether to digest the contents from the *declared* size. Measuring
    # it by converting first would read the whole array — a multi-gigabyte store
    # pulled into memory by a token call, which is the opposite of the point.
    itemsize = getattr(dtype, "itemsize", None)
    if not isinstance(itemsize, int) or itemsize * math.prod(shape) > _TOKEN_DIGEST_LIMIT:
        return unidentified
    try:
        contents = np.ascontiguousarray(array)
    # A token must never raise; an unreadable source is simply unidentified.
    except Exception:
        return unidentified
    return (*structural, hashlib.sha256(contents.tobytes()).hexdigest())


# --------------------------------------------------------------------------- #
# The wrapper
# --------------------------------------------------------------------------- #


class LazyArray:
    """A lazily-indexable view over an array-API-like array.

    Wrapping neither copies nor reads the wrapped array at construction time.
    Indexing through `.lazy` composes an `IndexTransform` and returns another
    `LazyArray`; `result()` materializes.

    Selections use the **positional NumPy dialect**; reads are broken up along a
    **partitioning** discovered from the wrapped array; and how much of a
    selection each read hands to that array is **negotiated** from an
    [`IndexingSupport`](support.md) level, also detected here. See the module
    docstring, which also covers how the dialect differs from `zarr.Array.lazy`
    and why every non-indexing NumPy operation materializes the view.

    Parameters
    ----------
    array
        The array to wrap. It must expose `shape`, `dtype`, and `__getitem__`
        with basic (integer/slice) indexing; anything beyond that is used only
        if detected or declared. Its partitioning, if it advertises one, is
        discovered here; use `with_parts` to choose a different one, and
        `with_indexing_support` to choose a different level.

    Examples
    --------
    >>> import numpy as np
    >>> source = np.arange(12).reshape(3, 4)
    >>> view = LazyArray(source).with_parts((2, 2)).lazy[1:, ::2]
    >>> view.shape
    (2, 2)
    >>> view.result()
    array([[ 4,  6],
           [ 8, 10]])
    """

    __slots__ = ("_array", "_parts", "_support", "_transform", "_window")

    def __init__(self, array: ArrayLike) -> None:
        if isinstance(array, np.matrix):
            # `np.matrix` keeps every result two-dimensional, so `m[1]` has shape
            # `(1, n)` where every other array-like gives `(n,)`. A view's shape
            # comes from the transform, which follows NumPy's rule, so the two
            # disagree on every rank-reducing selection. Refused at the door
            # rather than resolved into a shape the view did not promise.
            raise TypeError(
                "numpy.matrix cannot be wrapped: it never reduces rank, so a "
                "view's shape and its result would disagree. Convert it first, "
                "with numpy.asarray(m)."
            )
        shape = tuple(int(s) for s in array.shape)
        self._array = array
        self._window: tuple[slice, ...] | None = None
        self._transform = IndexTransform.from_shape(shape)
        self._parts = _discover_parts(array, shape)
        self._support = resolve_indexing_support(array)

    @classmethod
    def _derive(
        cls,
        array: ArrayLike,
        transform: IndexTransform,
        parts: tuple[EdgeDimensionGrid, ...] | None,
        window: tuple[slice, ...] | None,
        support: IndexingSupport,
    ) -> LazyArray:
        """Build a wrapper sharing `array` but carrying a new transform or partitioning."""
        view = cls.__new__(cls)
        view._array = array
        # Views re-zero their coordinate system: the positional dialect means a
        # view's first element is at position 0 whatever it was sliced from.
        view._transform = transform.translate_domain_to((0,) * transform.input_rank)
        view._parts = parts
        view._window = window
        view._support = support
        return view

    @property
    def _base_shape(self) -> tuple[int, ...]:
        """The shape of what this wrapper treats as its base array."""
        if self._window is None:
            return tuple(int(s) for s in self._array.shape)
        return tuple(s.stop - s.start for s in self._window)

    # -- array-API surface --------------------------------------------------

    @property
    def array(self) -> ArrayLike:
        """The wrapped array."""
        return self._array

    @property
    def base_shape(self) -> tuple[int, ...]:
        """The shape the partitioning is expressed in — not this view's shape.

        `with_parts` and `with_parts_per_axis` describe boxes of the array being
        read, not of the view reading it, so a narrowed view still partitions
        the extents named here. For a part's own `array`, this is the part's
        box, which is why the same call means different sizes there. Without
        somewhere to read it, the frame in force could only be inferred from an
        error message.
        """
        return self._base_shape

    @property
    def transform(self) -> IndexTransform:
        """The composed transform from this view's coordinates to storage."""
        return self._transform

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of this view — the transform's input domain, not the source's."""
        return self._transform.domain.shape

    @property
    def ndim(self) -> int:
        return self._transform.input_rank

    @property
    def size(self) -> int:
        return math.prod(self.shape)

    @property
    def dtype(self) -> Any:
        """The wrapped array's dtype; views never change it."""
        return self._array.dtype

    # -- source capability --------------------------------------------------

    @property
    def indexing_support(self) -> IndexingSupport:
        """How much of a selection this wrapper hands to the array it wraps.

        Read-only; use
        [`with_indexing_support`][zarr_indexing.lazy_array.LazyArray.with_indexing_support]
        to change it. Resolved once, at construction, in this order:

        1. an explicit `with_indexing_support` on this wrapper or the one it was
           derived from;
        2. the wrapped array's own `__zarr_indexing_support__` declaration;
        3. inference from the wrapped array — `VECTORIZED` for a NumPy array or
           for anything carrying zarr's `oindex`/`vindex` pair,
           [`BASIC`][zarr_indexing.support.IndexingSupport] otherwise.

        See [`zarr_indexing.support`](support.md) for what each level means and
        why the default is the most restrictive one.

        Examples
        --------
        >>> import numpy as np
        >>> LazyArray(np.arange(6)).indexing_support
        <IndexingSupport.VECTORIZED: 'vectorized'>
        """
        return self._support

    def with_indexing_support(self, support: IndexingSupport) -> LazyArray:
        """Return the same view, negotiating differently with the wrapped array.

        The transform, the wrapped array, the partitioning, and therefore
        `result()` are all unchanged; only how much of a selection is asked of
        the source, and how much is finished in memory, differs. Nothing is
        copied and nothing is read.

        Use this when the wrapped array supports more than can be detected from
        the outside, or when it supports less than it appears to and a request
        it cannot answer must not be made.

        Parameters
        ----------
        support
            The level to use, overriding both the source's own declaration and
            inference.

        Returns
        -------
        LazyArray
            The same view, read through a differently-negotiated source.

        Raises
        ------
        TypeError
            If `support` is not an `IndexingSupport`. Unlike detection, which
            treats a malformed declaration as absent, this is a public API and
            validates strictly.

        Examples
        --------
        >>> import numpy as np
        >>> from zarr_indexing.support import IndexingSupport
        >>> view = LazyArray(np.arange(6)).with_indexing_support(IndexingSupport.BASIC)
        >>> view.indexing_support
        <IndexingSupport.BASIC: 'basic'>
        >>> view.lazy.oindex[[4, 1, 1]].result()
        array([4, 1, 1])
        """
        # The annotation already says this, but the check is for callers who are
        # not type-checked: a bare string is the obvious mistake to make here,
        # and it would otherwise be read as "no arrays" and silently over-read.
        if not isinstance(support, IndexingSupport):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(
                f"indexing support must be an IndexingSupport, got {type(support).__name__}"
            )
        return LazyArray._derive(self._array, self._transform, self._parts, self._window, support)

    # -- shape of the selection ---------------------------------------------

    @property
    def is_box(self) -> bool:
        """Whether this view selects a rectangular region rather than a point list.

        True exactly when the composed transform's output maps are all
        `ConstantMap` or `DimensionMap` — no `ArrayMap`. Such a selection is
        affine and monotone along every axis, so it is described completely by
        an interval and a stride per dimension:
        [`bounding_box`][zarr_indexing.lazy_array.LazyArray.bounding_box]
        together with
        [`strides`][zarr_indexing.lazy_array.LazyArray.strides]. Basic indexing,
        at any depth of composition, stays a box; one `oindex`, `vindex`, or
        mask anywhere in the chain makes the selection a query permanently.

        A box is dense — every cell of its bounding box selected — only when
        every stride is 1. A strided box covers its hull sparsely:
        `lazy[10:50, ::4]` selects 40x20 cells out of a 40x77 hull, so a
        consumer that reads the whole hull and discards the rest transfers 3.85x
        the data it needs. Check `strides` before treating a box as a single
        slab read.

        The distinction lets a consumer decide between a slab read and a
        gather; see [the design notes](../design-notes.md) for why it is a
        category rather than an optimization.

        Examples
        --------
        >>> import numpy as np
        >>> array = LazyArray(np.arange(12).reshape(3, 4))
        >>> (array.lazy[1:, ::2].is_box, array.lazy.oindex[[2, 0], :].is_box)
        (True, False)
        """
        return not any(isinstance(m, ArrayMap) for m in self._transform.output)

    def bounding_box(self) -> tuple[tuple[int, int], ...] | None:
        """The storage region this view touches, one interval per storage dimension.

        Defined for any selection, box or not, as the hull: the smallest
        `[inclusive_min, exclusive_max)` interval per dimension of the array
        this view reads from that contains every coordinate the selection
        reaches.

        The hull is dense — every cell in it selected — only for a box whose
        every stride is 1. A strided box selects a sublattice of its hull (pair
        this with [`strides`][zarr_indexing.lazy_array.LazyArray.strides] to
        describe it fully), and a query's hull is a superset that can be
        arbitrarily loose: `oindex[[0, 999]]` has a 1000-wide hull over two
        rows.

        Returns
        -------
        tuple of (int, int), or None
            One interval per storage dimension, or `None` when the view is
            empty (`size == 0`) and so touches no coordinate at all, leaving no
            interval to report.

        Notes
        -----
        The coordinates are those of the array this view reads from, which for
        the `array` of a [`Partition`][zarr_indexing.lazy_array.Partition] is
        part-local — relative to that part's own box, not the wrapped array.
        Two parts of the same view can therefore report identical bounding
        boxes; [`Partition.box`][zarr_indexing.lazy_array.Partition] gives the
        global box they sit in.

        Examples
        --------
        >>> import numpy as np
        >>> array = LazyArray(np.arange(12).reshape(3, 4))
        >>> array.lazy[1:, ::2].bounding_box()
        ((1, 3), (0, 3))
        >>> array.lazy.oindex[[2, 0], :].bounding_box()
        ((0, 3), (0, 4))
        >>> array.lazy[1:1].bounding_box() is None
        True
        """
        if self.size == 0:
            return None
        domain = self._transform.domain
        bounds: list[tuple[int, int]] = []
        for m in self._transform.output:
            if isinstance(m, ConstantMap):
                bounds.append((m.offset, m.offset + 1))
            elif isinstance(m, DimensionMap):
                d = m.input_dimension
                first = m.offset + m.stride * domain.inclusive_min[d]
                last = m.offset + m.stride * (domain.exclusive_max[d] - 1)
                bounds.append((min(first, last), max(first, last) + 1))
            else:
                coords = m.offset + m.stride * m.index_array
                bounds.append((int(coords.min()), int(coords.max()) + 1))
        return tuple(bounds)

    def strides(self) -> tuple[int, ...] | None:
        """The step between selected coordinates, one per storage dimension.

        Together with `bounding_box()`, this fully describes a box selection:
        `bounding_box()` gives the interval per dimension, `strides()` gives the
        step per dimension. A stride of 1 means every cell of the hull along
        that dimension is selected; `k` means every `k`-th. Dimensions fixed by
        an integer index report 1 — they span a single coordinate.

        Returns
        -------
        tuple of int, or None
            One positive stride per storage dimension, or `None` when
            [`is_box`][zarr_indexing.lazy_array.LazyArray.is_box] is false: a
            query's coordinates are a lookup table and have no step. An empty
            box still reports its strides even though
            [`bounding_box`][zarr_indexing.lazy_array.LazyArray.bounding_box]
            returns `None`, because the step is a property of the selection's
            shape, not of the (empty) region it touches.

        Notes
        -----
        Magnitudes only. A reversing view (`lazy[::-1]`) selects the same set of
        coordinates as the equivalent forward view, so it reports the same
        bounding box and the same strides. The traversal direction is recorded
        in the transform, not in this description of the region touched. A
        consumer that needs the order reads the transform, or reverses the block
        it gets back.

        Examples
        --------
        >>> import numpy as np
        >>> array = LazyArray(np.arange(24).reshape(4, 6))
        >>> (array.lazy[1:, ::2].bounding_box(), array.lazy[1:, ::2].strides())
        (((1, 4), (0, 5)), (1, 2))
        >>> array.lazy[2, ::3].strides()
        (1, 3)
        >>> array.lazy.oindex[[2, 0], :].strides() is None
        True
        """
        if not self.is_box:
            return None
        return tuple(
            1 if isinstance(m, ConstantMap) else abs(m.stride) for m in self._transform.output
        )

    # -- partitioning -------------------------------------------------------

    def with_parts(self, parts: Sequence[int]) -> LazyArray:
        """Return the same view, read in uniform boxes of shape `parts`.

        One integer per dimension of `base_shape`, with the trailing box in each
        dimension clipped to the extent. The transform, the wrapped array, and
        therefore `result()` are all unchanged; only the boxes the read is
        broken into differ. Nothing is copied and nothing is read.

        For per-axis sizes see
        [`with_parts_per_axis`][zarr_indexing.lazy_array.LazyArray.with_parts_per_axis],
        and to read in one pass see
        [`unpartitioned`][zarr_indexing.lazy_array.LazyArray.unpartitioned].
        The three were one parameter whose meaning was decided by inspecting the
        type of what it was given, which left no way to ask for one of them and
        be told when you had spelled it wrong.

        Parameters
        ----------
        parts
            The box shape, one integer per dimension of `base_shape`.

        Returns
        -------
        LazyArray
            The same view with a new partitioning.

        Raises
        ------
        ValueError
            If `parts` has the wrong length or contains a non-positive extent.
            Unlike partition discovery, this is a public API and validates
            strictly.

        Examples
        --------
        >>> import numpy as np
        >>> view = LazyArray(np.arange(12).reshape(3, 4))
        >>> [part.base_coords for part in view.with_parts((2, 3)).parts()]
        [(0, 0), (0, 1), (1, 0), (1, 1)]
        """
        if any(isinstance(entry, Sequence) for entry in parts):
            raise ValueError(
                "with_parts takes one integer per dimension; for per-axis box "
                "sizes use with_parts_per_axis"
            )
        return self._with_grids(dimension_grids_from_chunks(parts, self._base_shape))

    def with_parts_per_axis(self, sizes: Sequence[Sequence[int]]) -> LazyArray:
        """Return the same view, read in boxes of explicitly listed sizes.

        The dask convention: one sequence of box extents per dimension of
        `base_shape`, each summing to that dimension's extent. Use it when the
        boxes are not uniform — a partitioning discovered from a store, or one
        whose last box differs by more than clipping.

        Parameters
        ----------
        sizes
            One sequence of box extents per dimension of `base_shape`.

        Returns
        -------
        LazyArray
            The same view with a new partitioning.

        Raises
        ------
        ValueError
            If `sizes` has the wrong length, contains a non-positive extent, or
            declares sizes that do not sum to `base_shape`.

        Examples
        --------
        >>> import numpy as np
        >>> view = LazyArray(np.arange(12).reshape(3, 4))
        >>> [part.box for part in view.with_parts_per_axis(((1, 2), (4,))).parts()]
        [((0, 1), (0, 4)), ((1, 3), (0, 4))]
        """
        return self._with_grids(dimension_grids_from_chunks(sizes, self._base_shape))

    def unpartitioned(self) -> LazyArray:
        """Return the same view, read in one pass.

        `result()` then lowers the whole view to array operations directly
        rather than allocating an output buffer and assembling it box by box.
        `parts()` still yields a single part covering everything.

        Returns
        -------
        LazyArray
            The same view with no partitioning.
        """
        return self._with_grids(None)

    def _with_grids(self, grids: tuple[DimensionGridLike, ...] | None) -> LazyArray:
        return LazyArray._derive(self._array, self._transform, grids, self._window, self._support)

    def parts(self) -> Iterator[Partition]:
        """Iterate the base partitioning, projected through this view.

        Single-use: this is a generator, so it is consumed by the first walk and
        a second `for` over the same object yields nothing. Call `parts()` again
        for a fresh walk, or keep a `list` of it if you need to revisit.

        Yields one [`Partition`][zarr_indexing.lazy_array.Partition] per box the
        view actually touches. The parts tile the view exactly and disjointly,
        and each carries a `LazyArray` that can be resolved on its own: in
        another thread, in another order, or not at all.

        A wrapper with no partitioning (see `with_parts`) yields a single part
        covering the whole array.

        Yields
        ------
        Partition
            One per touched box, in the resolver's own order.

        Examples
        --------
        >>> import numpy as np
        >>> view = LazyArray(np.arange(12).reshape(3, 4)).with_parts((2, 2))
        >>> part = next(view.lazy[:, 1:].parts())
        >>> (part.base_coords, part.view.shape, part.is_complete)
        ((0, 0), (2, 1), False)
        """
        base_shape = self._base_shape
        grids = self._parts if self._parts is not None else _whole_array_grids(base_shape)
        out_shape = self.shape
        rank = len(base_shape)

        for base_coords, local, out_indices in iter_chunk_transforms(self._transform, grids):
            origin = tuple(grid.chunk_offset(c) for grid, c in zip(grids, base_coords, strict=True))
            extent = tuple(grid.chunk_size(c) for grid, c in zip(grids, base_coords, strict=True))
            if origin == (0,) * rank and extent == base_shape:
                # The part is the whole base: lowering directly against the
                # source beats materializing a block that is the source.
                window = self._window
            elif self._window is None:
                window = tuple(slice(o, o + e) for o, e in zip(origin, extent, strict=True))
            else:
                window = tuple(
                    slice(w.start + o, w.start + o + e)
                    for w, o, e in zip(self._window, origin, extent, strict=True)
                )
            # The global box, computed from the origin directly rather than from
            # `window`: a part covering the whole base carries no window (so
            # nothing is pre-materialized) but still sits somewhere concrete.
            if self._window is None:
                global_origin = origin
            else:
                global_origin = tuple(
                    w.start + o for w, o in zip(self._window, origin, strict=True)
                )
            yield Partition(
                base_coords=base_coords,
                box=tuple((o, o + e) for o, e in zip(global_origin, extent, strict=True)),
                view=LazyArray._derive(self._array, local, None, window, self._support),
                out_selection=_partition_out_selection(local, out_indices, out_shape),
                is_complete=_covers_whole_part(local, extent),
            )

    # -- indexing -----------------------------------------------------------

    @property
    def lazy(self) -> _LazyIndexer:
        """Lazy indexing: `lazy[...]`, `lazy.oindex[...]`, `lazy.vindex[...]`.

        Each returns a new `LazyArray` view; no data is read.
        """
        return _LazyIndexer(self._select)

    def _select(self, selection: Any, mode: SelectionMode) -> LazyArray:
        transform = self._transform
        if mode != "basic":
            # NumPy applies scalar integers as basic indices before the advanced
            # ones, dropping their axes. Split them into their own step.
            scalar_selection, selection = split_scalar_axes(selection, transform.domain, mode)
            if scalar_selection is not None:
                transform = selection_to_transform(scalar_selection, transform, "basic")
                transform = transform.translate_domain_to((0,) * transform.input_rank)
        literal = normalize_positional_selection(selection, transform.domain, mode)
        composed = selection_to_transform(literal, transform, mode)
        return LazyArray._derive(self._array, composed, self._parts, self._window, self._support)

    def __getitem__(self, selection: Any) -> Any:
        """Read a basic selection eagerly, like `numpy.ndarray.__getitem__`.

        Reads here are eager, not lazy, so that a `LazyArray` works as a duck
        array for consumers (dask's `from_array`, `numpy.asarray`) that expect
        indexing to produce data. Use `.lazy[...]` for the lazy form.
        """
        return self._select(selection, "basic").result()

    def result(self) -> Any:
        """Materialize this view.

        Assembles the view from its `parts()`, reading each box once. A wrapper
        with **no partitioning** — `unpartitioned()`, and the default for a
        source that advertises none — skips the output buffer and lowers the
        whole view directly to array operations. A partitioning with a single
        whole-array box still goes through the buffer: the branch is on whether
        a partitioning is in force, not on how many boxes it has.

        The result never shares memory with the wrapped array, so writing to it
        is safe whichever of those two routes it came by. That holds for a
        source that stores its data in NumPy whether or not the source is itself
        a NumPy array; a source whose namespace is foreign enough that its
        blocks are not NumPy arrays is out of reach (see the module docstring's
        device caveat).

        Returns
        -------
        array
            An array of shape `self.shape`, identical whatever partitioning is
            in force. A partitioned view produces a NumPy array (see the module
            docstring's device caveat); an unpartitioned one produces whatever
            the wrapped array's namespace produces. A view with a zero-rank
            domain returns a zero-dimensional array, not a scalar.

        Raises
        ------
        AssertionError
            If the partition walk does not cover the view. The output buffer is
            uninitialized where nothing was written, so an incomplete walk is
            reported rather than returned.
        """
        if self._parts is None:
            block = _read(self._array, self._window, self._transform, self._support)
            return _detach(block, self._array)

        out_shape = self.shape
        out = self._output_buffer(out_shape)
        size = math.prod(out_shape)
        if size > 0:
            written = 0
            for part in self.parts():
                # Counted before the scatter, so a part addressing the wrong
                # number of axes is named rather than reported as a broadcast
                # failure against the buffer.
                written += _out_selection_cell_count(part.out_selection, out_shape)
                out[part.out_selection] = np.asanyarray(part.view.result())
            if written != size:
                # The buffer is uninitialized where no part wrote, so a partition
                # walk that does not tile the view exactly would otherwise hand
                # back process memory dressed as data. The parts are disjoint by
                # contract, so counting the cells each addresses is enough:
                # a gap undercounts and an overlap overcounts.
                raise AssertionError(
                    f"the partition walk addressed {written} of the view's {size} "
                    "cells; this is a bug in zarr-indexing's partition walk"
                )
        return out

    def _output_buffer(self, out_shape: tuple[int, ...]) -> Any:
        """The buffer `result()` scatters parts into.

        Deliberately uninitialized: every cell is written by exactly one part,
        and `result()` verifies that before returning. A masked source gets a
        masked buffer so that scattering preserves the mask; nothing else about
        the wrapped array's own type survives a partitioned read (see the module
        docstring's device caveat).
        """
        dtype = np.dtype(self.dtype)
        if isinstance(self._array, np.ma.MaskedArray):
            return np.ma.masked_all(out_shape, dtype=dtype)
        return np.empty(out_shape, dtype=dtype)

    # -- protocols ----------------------------------------------------------

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> Any:
        """Materialize the view as a NumPy array.

        The result never shares memory with the wrapped array, whatever `copy`
        asks for: `result()` already detaches, so `copy=True` gets an array the
        caller owns and `copy=None` gets the same one rather than a second
        allocation. `copy=False` is refused, because materializing means reading
        — the values do not exist as a NumPy array until this call makes them.
        """
        if copy is False:
            raise ValueError(
                "a LazyArray cannot be converted to a NumPy array without a "
                "copy: a view is a description of a read, and the values only "
                "exist once the read is made"
            )
        return np.asarray(self.result(), dtype=dtype)

    def __dask_tokenize__(self) -> Any:
        """A deterministic token: the wrapped array and the view.

        Two wrappers produce equal tokens when they wrap the same data and
        address the same cells. The view contributes its canonical ndsel body,
        so transforms that differ only in representation produce the same
        token. See `_wrapped_token` for the determinism scope of the wrapped
        array's contribution; dask is imported lazily and is never a
        requirement of this package.

        The partitioning and the `IndexingSupport` level are deliberately
        absent. Both decide how the data is read — in which boxes, and through
        which requests to the source — and neither changes the values that come
        back, so two wrappers differing only in those describe the same data.
        A token identifies data, so they token alike and a consumer that caches
        on tokens reuses one result for both.
        """
        return (
            type(self).__qualname__,
            _wrapped_token(self._array),
            None if self._window is None else tuple((s.start, s.stop) for s in self._window),
            json.dumps(transform_to_canonical(self._transform), sort_keys=True),
        )

    def __len__(self) -> int:
        if self.ndim == 0:
            raise TypeError("len() of unsized object")
        return self.shape[0]

    def __iter__(self) -> Iterator[Any]:
        """Iterate eagerly over the first axis, like a NumPy array.

        The rank check happens in `__iter__` itself rather than in the
        generator, so `iter(view)` on a zero-rank view raises immediately as
        NumPy's does, instead of waiting for the first `next`.
        """
        if self.ndim == 0:
            raise TypeError("iteration over a 0-d array")
        return (self[position] for position in range(self.shape[0]))

    # NumPy's own conversions decide what a size-1 (or wrong-sized) view means,
    # including which exception it raises, so these delegate rather than
    # reimplement. Each materializes the view first.
    def __bool__(self) -> bool:
        return bool(self.result())

    def __int__(self) -> int:
        return int(self.result())

    def __float__(self) -> float:
        return float(self.result())

    def __index__(self) -> int:
        return operator.index(self.result())

    def __repr__(self) -> str:
        wrapped = type(self._array).__name__
        described = [f"{wrapped} shape={self.shape} dtype={self.dtype}"]
        if not _is_identity_transform(self._transform, self._base_shape):
            described.append(f"view={self._transform.selection_repr}")
        return f"<LazyArray {' '.join(described)}>"


class _LazyIndexer:
    """The `.lazy` accessor: builds views instead of reading data.

    Holds the owning view's bound `_select` rather than the view itself, so the
    accessor classes never reach into another object's internals.
    """

    __slots__ = ("_select",)

    def __init__(self, select: SelectFn) -> None:
        self._select = select

    def __getitem__(self, selection: Any) -> LazyArray:
        """Basic (integer / slice / ellipsis) indexing, lazily."""
        return self._select(selection, "basic")

    @property
    def oindex(self) -> _LazyOIndex:
        """Orthogonal (outer-product) indexing, lazily."""
        return _LazyOIndex(self._select)

    @property
    def vindex(self) -> _LazyVIndex:
        """Vectorized (coordinate / mask) indexing, lazily."""
        return _LazyVIndex(self._select)


class _LazyOIndex:
    """`lazy.oindex[...]` — one selection per axis, combined as an outer product."""

    __slots__ = ("_select",)

    def __init__(self, select: SelectFn) -> None:
        self._select = select

    def __getitem__(self, selection: Any) -> LazyArray:
        return self._select(selection, "orthogonal")


class _LazyVIndex:
    """`lazy.vindex[...]` — correlated coordinate arrays, or a single mask."""

    __slots__ = ("_select",)

    def __init__(self, select: SelectFn) -> None:
        self._select = select

    def __getitem__(self, selection: Any) -> LazyArray:
        return self._select(selection, "vectorized")
