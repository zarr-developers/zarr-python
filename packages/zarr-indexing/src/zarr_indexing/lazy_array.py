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
Composition is free: a view of a view is still a single transform.

Parts
-----
A `LazyArray` carries a **partitioning** of the array it wraps: a grid of boxes
that a read is broken into. `parts()` walks those boxes as they fall through the
view, yielding a [`Partition`](#zarr_indexing.lazy_array.Partition) per box —
the base coordinates of the box, a `LazyArray` covering exactly the cells of the
view that live in it, and where those cells belong in the result. `result()` is
built on that walk and nothing else:

```python
for part in view.parts():
    out[part.out_selection] = part.array.result()
```

so each box is read once, with plain basic slicing, and the selection is applied
to the block in memory.

The partitioning is discovered from the wrapped array at construction — first
`read_chunk_sizes` (zarr's clipped per-axis sizes, sharding-aware), then
`chunks`, read as per-axis sizes if its entries are sequences and as a uniform
box shape if they are integers. Those attribute names are the *source's*
vocabulary; the wrapper's own surface speaks only of parts. An array that
advertises neither is a single whole-array part, and resolving it lowers the
whole view to one pass of array operations (`take`, basic slicing, a flat
gather) rather than materializing a block first.

`with_parts` replaces the partitioning without touching the data or the view:

```python
view.with_parts((64, 64))       # uniform boxes, tail clipped
view.with_parts(((3, 3, 1),))   # explicit per-axis sizes
view.with_parts(None)           # one whole-array part; resolve in one shot
```

Repartitioning never changes what `result()` returns — only how the read is cut
up. Deliberately misaligning the parts with the source's own boxes is legal and
useful (to bound peak memory, or to batch small reads); it costs I/O, not
correctness.

Boxes and queries
-----------------
A selection is either **rectangular** — an interval and a stride per dimension,
which is what basic indexing composes to, however deeply — or a **query**, an
explicit list of coordinates, which is what `oindex`, `vindex`, and masks
produce and which no amount of subsequent basic indexing undoes. `is_box`
reports the category and `bounding_box()` the storage region touched, exact for
a box and a hull for a query. The distinction is structural rather than an
optimization; [the design notes](../design-notes.md) say why it matters to
anything downstream of a selection.

The positional dialect
----------------------
Selections on `LazyArray` are **positional, NumPy-style**: index 0 is the first
element of the current view, `-1` is the last, boolean masks must match the
view's shape, and every index is bounds-checked against the view.

This is deliberately different from `zarr.Array.lazy[...]`, which exposes the
**literal** TensorStore dialect: a zarr view keeps the coordinate system of the
array it came from, so after `v = arr.lazy[10:50]` the first element of `v` is
`v[10]` and a negative index is out of bounds rather than counted from the end.
That dialect is the right one for zarr, where a view's coordinates are meant to
stay comparable with the parent array's. `LazyArray` is a duck array — it has to
behave like the thing it wraps to be usable as a drop-in for NumPy or as a dask
source — so it re-zeroes its coordinates on every view and speaks positions.
`zarr_indexing.boundary` performs the translation between the two.

Two more NumPy rules the dialect keeps, in every mode:

- **A scalar integer drops its axis.** It is a basic index wherever it appears,
  applied before any advanced index rather than broadcast against one. So
  `lazy.oindex[0]` has the shape of `x[0]`, `lazy.oindex[0, [1, 2], :]` means
  `x[0][numpy.ix_([1, 2], ...)]`, and `lazy.oindex[0, 1, 2]` and
  `lazy.vindex[0, 1, 2]` are both zero-rank. Use a length-1 list to keep an
  axis.
- **Advanced indices land where NumPy puts them.** For a `vindex` selection that
  leaves some axes unindexed, the gathered dimensions sit where the coordinate
  arrays sat when those arrays are adjacent, and lead when a slice separates
  them — so `lazy.vindex[..., i, j]` has shape `(x.shape[0], *broadcast)`,
  matching `x[..., i, j]`.

Materializing on fallback
-------------------------
`LazyArray` implements `__array__` but deliberately implements neither
`__array_ufunc__`/`__array_function__` nor `__array_namespace__`. **Every NumPy
operation other than indexing therefore materializes the whole view**:
`numpy.sum(view)`, `view + 1`, `numpy.stack([view, view])` all convert through
`__array__` first and give you a plain NumPy array back. That is intentional —
laziness here is about *indexing*, not about building a deferred compute graph —
but it means a `LazyArray` is not a drop-in for arithmetic on a large array. Use
`.lazy[...]` to narrow first, or hand the wrapper to `dask.array.from_array` and
let dask own the compute graph.

Device caveat
-------------
Resolving a partitioned view builds its output buffer with NumPy, because the
resolver's scatter indices are host-side integer arrays. Wrapping a device array
that advertises parts therefore returns host memory. A single whole-array part
lowers directly and stays in the wrapped array's own namespace.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
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
from zarr_indexing.grid import EdgeDimensionGrid, dimension_grids_from_chunks
from zarr_indexing.json import transform_to_canonical
from zarr_indexing.output_map import ArrayMap, ConstantMap, DimensionMap
from zarr_indexing.transform import (
    IndexTransform,
    array_map_dependent_axis,
    selection_to_transform,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

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
    except Exception:
        return None


def _discover_parts(array: Any, shape: tuple[int, ...]) -> tuple[EdgeDimensionGrid, ...] | None:
    """Resolve the partitioning advertised by `array`, or None for one whole part.

    Discovery parses external input: an attribute that does not describe a
    partitioning of `shape` means "this object does not advertise one I
    understand", and the array is treated as unpartitioned rather than rejected.
    A partitioning is an I/O strategy, so reading the whole array is always a
    correct fallback. `with_parts` is *our* API, and validates strictly.
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


def _restore_domain_axis_order(result: Any, axis_input_dims: list[int], rank: int) -> Any:
    """Permute `result`'s axes into input-domain order, restoring dropped axes.

    `axis_input_dims[k]` is the input (domain) dimension that axis `k` of
    `result` corresponds to. Axes are permuted so that they appear in increasing
    domain-dimension order, and any domain dimension no output map depends on
    (only reachable via `newaxis`) is reinserted as a singleton.

    This is where NumPy's advanced-index placement rules are absorbed: whatever
    order the gather produced, the lowered result always comes back in the
    view's own axis order.
    """
    if len(set(axis_input_dims)) != len(axis_input_dims):
        raise NotImplementedError(
            "resolving a transform whose output maps share an input dimension "
            "(a diagonal view) is not supported"
        )
    order = sorted(range(len(axis_input_dims)), key=lambda k: axis_input_dims[k])
    if order != list(range(len(order))):
        result = _transpose(result, tuple(order))
    covered = sorted(axis_input_dims)
    for dim in range(rank):
        if dim not in covered:
            result = _expand_dims(result, dim)
    return result


def _lower(array: Any, transform: IndexTransform) -> Any:
    """Lower a transform to one pass of array operations over `array`.

    The single lowering engine: every read, partitioned or not, ends here. The
    result is always in the transform's own domain axis order and of exactly its
    domain shape.
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
    return _restore_domain_axis_order(result, axis_input_dims, transform.input_rank)


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
        flat_index = flat_index + _array_map_coords(m) * stride
        stride *= corr_sizes[position]

    result = _take(result, flat_index, axis=0)
    result = _reshape(result, broadcast_shape + tail_shape)
    return _restore_domain_axis_order(
        result, list(broadcast_axes) + residual_axis_dims, transform.input_rank
    )


# --------------------------------------------------------------------------- #
# Partitions
# --------------------------------------------------------------------------- #


def _partition_out_selection(
    sub_transform: IndexTransform,
    out_indices: Any,
    out_shape: tuple[int, ...],
) -> tuple[Any, ...]:
    """Where a partition's values belong in an array of shape `out_shape`.

    A correlated sub-transform scatters through flat offsets into the row-major
    result; unravelling them turns that into an index tuple for the result's own
    shape, so callers never need a flat working buffer.
    """
    _chunk_selection, out_selection, _drop_axes = sub_transform_to_selections(
        sub_transform, out_indices
    )
    if not _is_correlated(sub_transform):
        return out_selection
    if len(out_shape) == 0:
        return ()
    flat = out_selection[0]
    if isinstance(flat, slice):
        flat = np.arange(flat.start, flat.stop, dtype=np.intp)
    return tuple(np.unravel_index(np.asarray(flat, dtype=np.intp), out_shape))


def _covers_whole_part(local_transform: IndexTransform, part_shape: tuple[int, ...]) -> bool:
    """Whether a part-local transform addresses every cell of its part.

    Conservative: an `ArrayMap` could in principle enumerate a whole part, but
    proving it costs more than the answer is worth, so a fancy-indexed axis
    always reports incomplete.
    """
    domain = local_transform.domain
    for out_dim, m in enumerate(local_transform.output):
        extent = part_shape[out_dim]
        if isinstance(m, ConstantMap):
            if extent != 1 or m.offset != 0:
                return False
        elif isinstance(m, DimensionMap):
            if m.stride != 1:
                return False
            d = m.input_dimension
            if m.offset + domain.inclusive_min[d] != 0:
                return False
            if m.offset + domain.exclusive_max[d] != extent:
                return False
        else:
            return False
    return True


@dataclass(frozen=True)
class Partition:
    """One box of a `LazyArray`'s partitioning, as it falls through the view.

    Yielded by [`LazyArray.parts`][zarr_indexing.lazy_array.LazyArray.parts].
    The parts of a view tile it exactly and disjointly: assembling every
    `array.result()` at its `out_selection` reproduces the view's `result()`,
    and each part can be resolved independently and concurrently.

    Attributes
    ----------
    base_coords
        Which box of the base partitioning this is, one coordinate per dimension
        of the wrapped array.
    box
        The box itself, in the **global** storage coordinates of the wrapped
        array: one `[inclusive_min, exclusive_max)` interval per dimension.
        `array.bounding_box()` is part-*local* and so cannot tell two parts
        apart; this can, and it is what pairs with `is_complete` to decide a
        read-modify-write.
    array
        A `LazyArray` covering exactly the cells of the view that live in this
        box. Resolving it reads the box once with basic slicing and applies the
        selection to the block in memory.
    out_selection
        Where `array.result()` belongs in an array of the view's shape — a
        NumPy index tuple, usable directly as `out[part.out_selection] = ...`.
    is_complete
        Whether the view covers the whole box. Useful to a writer deciding
        between a blind overwrite and a read-modify-write.
    """

    base_coords: tuple[int, ...]
    box: tuple[tuple[int, int], ...]
    array: LazyArray
    out_selection: tuple[Any, ...]
    is_complete: bool


# --------------------------------------------------------------------------- #
# Tokenization
# --------------------------------------------------------------------------- #


def _wrapped_token(array: Any) -> Any:
    """A deterministic token for the wrapped array.

    Determinism scope, in order of preference: the array's own
    `__dask_tokenize__`; `dask.base.tokenize` when dask is importable (imported
    lazily — this package never requires it); otherwise a local fallback that
    digests the contents of a small array and, above `_TOKEN_DIGEST_LIMIT`,
    falls back to a structural description that is stable across processes but
    **not** sensitive to the array's contents.
    """
    hook = getattr(array, "__dask_tokenize__", None)
    if hook is not None:
        try:
            return hook()
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

    # Decide whether to digest the contents from the *declared* size. Measuring
    # it by converting first would read the whole array — a multi-gigabyte store
    # pulled into memory by a token call, which is the opposite of the point.
    itemsize = getattr(dtype, "itemsize", None)
    if not isinstance(itemsize, int) or itemsize * math.prod(shape) > _TOKEN_DIGEST_LIMIT:
        return structural
    try:
        contents = np.ascontiguousarray(array)
    except Exception:
        return structural
    return (*structural, hashlib.sha256(contents.tobytes()).hexdigest())


# --------------------------------------------------------------------------- #
# The wrapper
# --------------------------------------------------------------------------- #


class LazyArray:
    """A lazily-indexable view over an array-API-like array.

    Wrapping is cheap and non-destructive: the wrapped array is never copied and
    never read at construction time. Indexing through `.lazy` composes an
    `IndexTransform` and returns another `LazyArray`; `result()` materializes.

    Selections use the **positional NumPy dialect**, and reads are broken up
    along a **partitioning** discovered from the wrapped array — see the module
    docstring, which also covers how the dialect differs from `zarr.Array.lazy`
    and why every non-indexing NumPy operation materializes the view.

    Parameters
    ----------
    array
        The array to wrap. It must expose `shape`, `dtype`, and `__getitem__`
        with basic (integer/slice) indexing. Its partitioning, if it advertises
        one, is discovered here; use `with_parts` to choose a different one.

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

    __slots__ = ("_array", "_parts", "_transform", "_window")

    def __init__(self, array: ArrayLike) -> None:
        shape = tuple(int(s) for s in array.shape)
        self._array = array
        self._window: tuple[slice, ...] | None = None
        self._transform = IndexTransform.from_shape(shape)
        self._parts = _discover_parts(array, shape)

    @classmethod
    def _derive(
        cls,
        array: ArrayLike,
        transform: IndexTransform,
        parts: tuple[EdgeDimensionGrid, ...] | None,
        window: tuple[slice, ...] | None,
    ) -> LazyArray:
        """Build a wrapper sharing `array` but carrying a new transform or partitioning."""
        view = cls.__new__(cls)
        view._array = array
        # Views re-zero their coordinate system: the positional dialect means a
        # view's first element is at position 0 whatever it was sliced from.
        view._transform = transform.translate_domain_to((0,) * transform.input_rank)
        view._parts = parts
        view._window = window
        return view

    @property
    def _base_shape(self) -> tuple[int, ...]:
        """The shape of what this wrapper treats as its base array."""
        if self._window is None:
            return tuple(int(s) for s in self._array.shape)
        return tuple(s.stop - s.start for s in self._window)

    def _read_base(self) -> Any:
        """The base array, materializing the window if this wrapper has one."""
        if self._window is None:
            return self._array
        return np.asarray(self._array[self._window])

    # -- array-API surface --------------------------------------------------

    @property
    def array(self) -> ArrayLike:
        """The wrapped array."""
        return self._array

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

    # -- shape of the selection ---------------------------------------------

    @property
    def is_box(self) -> bool:
        """Whether this view selects a rectangular region rather than a point list.

        True exactly when the composed transform's output maps are all
        `ConstantMap` or `DimensionMap` — no `ArrayMap`. Such a selection is
        affine and monotone along every axis, so it is described completely by
        an interval **and a stride** per dimension:
        [`bounding_box`][zarr_indexing.lazy_array.LazyArray.bounding_box]
        together with
        [`strides`][zarr_indexing.lazy_array.LazyArray.strides]. Basic indexing,
        however deeply composed, stays a box; one `oindex`, `vindex`, or mask
        anywhere in the chain leaves it forever.

        A box is *dense* — every cell of its bounding box selected — only when
        every stride is 1. A strided box covers its hull sparsely:
        `lazy[10:50, ::4]` selects 40x20 cells out of a 40x77 hull, so a
        consumer that slabs the hull and discards the rest reads 3.85x what it
        needs. Check `strides` before treating a box as a single slab read.

        The distinction is what lets a consumer decide between a slab read and a
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

        The hull is only *dense* — every cell in it selected — for a box whose
        every stride is 1. A strided box selects a sublattice of its hull (pair
        this with [`strides`][zarr_indexing.lazy_array.LazyArray.strides] to
        describe it fully), and a query's hull is merely a superset that can be
        arbitrarily loose: `oindex[[0, 999]]` has a 1000-wide hull over two
        rows.

        Returns
        -------
        tuple of (int, int), or None
            One interval per storage dimension, or `None` when the view is
            empty (`size == 0`) and so touches no coordinate at all — an empty
            selection has no meaningful origin to report an interval around.

        Notes
        -----
        The coordinates are those of the array this view reads from, which for
        the `array` of a [`Partition`][zarr_indexing.lazy_array.Partition] is
        **part-local** — relative to that part's own box, not the wrapped array.
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

        Completes the description a box selection needs: `bounding_box()` says
        *where*, `strides()` says *how densely*. A stride of 1 means every cell
        of the hull along that dimension is selected; `k` means every `k`-th.
        Dimensions fixed by an integer index report 1 — they span a single
        coordinate.

        Returns
        -------
        tuple of int, or None
            One positive stride per storage dimension, or `None` when
            [`is_box`][zarr_indexing.lazy_array.LazyArray.is_box] is false: a
            query's coordinates are a lookup table and have no step.

        Notes
        -----
        The magnitude only. A reversing view (`lazy[::-1]`) selects the same
        *set* of coordinates as its forward twin, so it reports the same
        bounding box and the same strides; the traversal direction lives in the
        transform, not in this description of the region touched.

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

    def with_parts(self, parts: Sequence[int] | Sequence[Sequence[int]] | None) -> LazyArray:
        """Return the same view, read in different parts.

        The transform, the wrapped array, and therefore `result()` are all
        unchanged; only the boxes the read is broken into differ. Cheap: nothing
        is copied and nothing is read.

        Parameters
        ----------
        parts
            Either a uniform box shape (one integer per dimension of the base
            array, with the trailing box clipped to the extent), dask-convention
            per-axis sizes (one sequence of box extents per dimension, each
            summing to the base extent), or `None` for a single part covering
            the whole array — which makes `result()` lower the view in one shot
            instead of assembling it block by block.

        Returns
        -------
        LazyArray
            The same view with a new partitioning.

        Raises
        ------
        ValueError
            If `parts` has the wrong length, mixes the two conventions, contains
            a non-positive extent, or declares per-axis sizes that do not sum to
            the base shape. Unlike discovery, this is our own API and validates
            strictly.

        Examples
        --------
        >>> import numpy as np
        >>> view = LazyArray(np.arange(12).reshape(3, 4))
        >>> [part.base_coords for part in view.with_parts((2, 3)).parts()]
        [(0, 0), (0, 1), (1, 0), (1, 1)]
        """
        grids = None if parts is None else dimension_grids_from_chunks(parts, self._base_shape)
        return LazyArray._derive(self._array, self._transform, grids, self._window)

    def parts(self) -> Iterator[Partition]:
        """Iterate the base partitioning, projected through this view.

        Yields one [`Partition`][zarr_indexing.lazy_array.Partition] per box the
        view actually touches. The parts tile the view exactly and disjointly,
        and each carries a `LazyArray` that can be resolved on its own — in
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
        >>> (part.base_coords, part.array.shape, part.is_complete)
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
                array=LazyArray._derive(self._array, local, None, window),
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
        return LazyArray._derive(self._array, composed, self._parts, self._window)

    def __getitem__(self, selection: Any) -> Any:
        """Read a basic selection **eagerly**, like `numpy.ndarray.__getitem__`.

        Eager, not lazy, so that a `LazyArray` is a drop-in duck array for
        consumers (dask's `from_array`, `numpy.asarray`) that expect indexing to
        produce data. Use `.lazy[...]` for the lazy form.
        """
        return self._select(selection, "basic").result()

    def result(self) -> Any:
        """Materialize this view.

        Assembles the view from its `parts()`, reading each box once. A wrapper
        with a single whole-array part skips the buffer entirely and lowers
        straight to array operations.

        Returns
        -------
        array
            An array of shape `self.shape`, identical whatever partitioning is
            in force. A partitioned view produces a NumPy array (see the module
            docstring's device caveat); a single whole-array part produces
            whatever the wrapped array's namespace produces. A view with a
            zero-rank domain returns a zero-dimensional array, not a scalar.
        """
        if self._parts is None:
            return _lower(self._read_base(), self._transform)

        out_shape = self.shape
        out = np.empty(out_shape, dtype=np.dtype(self.dtype))
        if math.prod(out_shape) > 0:
            for part in self.parts():
                value = np.asarray(part.array.result())
                if len(out_shape) == 0:
                    value = value.reshape(())
                out[part.out_selection] = value
        return out

    # -- protocols ----------------------------------------------------------

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> Any:
        if copy is False:
            raise ValueError(
                "a LazyArray cannot be converted to a NumPy array without a "
                "copy: materializing a view always produces new data"
            )
        return np.asarray(self.result(), dtype=dtype)

    def __dask_tokenize__(self) -> Any:
        """A deterministic token: the wrapped array, the view, and the parts.

        Two wrappers token-equal when they wrap the same data, address the same
        cells, and read them in the same boxes. The view contributes its
        canonical ndsel body, so transforms that differ only in representation
        token alike. See `_wrapped_token` for the determinism scope of the
        wrapped array's contribution; dask is imported lazily and is never a
        requirement of this package.
        """
        return (
            type(self).__qualname__,
            _wrapped_token(self._array),
            None if self._window is None else tuple((s.start, s.stop) for s in self._window),
            json.dumps(transform_to_canonical(self._transform), sort_keys=True),
            None if self._parts is None else tuple(grid.sizes for grid in self._parts),
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
