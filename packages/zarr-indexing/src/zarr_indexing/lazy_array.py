"""`LazyArray` — TensorStore-style lazy indexing over array-like sources.

`LazyArray` wraps a source with `shape`, `dtype`, and basic integer/slice
`__getitem__`, whose reads can be lowered through NumPy system memory. It adds
a `.lazy` accessor whose indexing operations build up an
[`IndexTransform`](transform.md) instead of reading data:

```python
view = LazyArray(source).lazy[10:50, ::2].lazy.oindex[[3, 1, 1], :]
view.shape          # known without touching the data
values = view.result()
```

Nothing is read until `result()` (or `__array__`, or an eager `__getitem__`).
Every `.lazy` operation is metadata-only. Composition does not accumulate
layers: a view of a view is still a single transform and retains its reader.

Parts
-----
A `LazyArray` carries a **partitioning** of the array it wraps: a grid of boxes
that a read is broken into. `parts()` walks those boxes as they fall through the
view, yielding a [`Partition`](#zarr_indexing.lazy_array.Partition) per box. Its
paired projection describes the chunk-local read and where its cells land in the
request; `view` carries that partition's transform. `result()` allocates one
fresh output buffer, then reads each partition once through the selected reader
into the final buffer or an owned temporary for fancy placement.

The part view's transform directly addresses its raw wrapped array. The paired
projection deliberately retains the chunk-local frame; both travel together in
the `ReadContext` passed to the reader.

The partitioning is discovered from the wrapped array at construction — first
`read_chunk_sizes` (zarr's clipped per-axis sizes, sharding-aware), then
`chunks`, read as per-axis sizes if its entries are sequences and as a uniform
box shape if they are integers. Those attribute names belong to the wrapped
array; this API refers only to parts. An array that advertises neither gets a
single whole-array part, and resolving it reads the whole view through its
selected reader in one pass.

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

Readers
-------
Every wrapper carries a reader that owns the backend-specific request. The
transform answers **which values?** and is independent of the backend; the
reader answers **how does this backend obtain them?** and must preserve the
complete transform exactly. Readers do not define indexing semantics,
partitioning, scheduling, or result ownership. The conservative
`LazyArray(source)` uses `basic_reader`, which needs only basic slicing.
`LazyArray.from_numpy(array)` explicitly opts into `numpy_reader` for direct
NumPy indexing. `with_reader()` replaces the reader without reading or changing
the view metadata. The reader object is shared by all derived views and their
parts. Consumers may materialize part views concurrently; `LazyArray` does not
serialize calls, so a stateful reader must synchronize its own mutable state.

Both built-in readers lower through NumPy system memory. They do not implicitly
transfer device arrays. A device source requires an explicit custom reader that
performs any needed transfer into the supplied system-memory output buffer.

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

- A scalar integer drops its axis. Any non-boolean object implementing Python's
  `SupportsIndex` protocol is accepted as one, including in slice bounds and
  steps; an `__int__` method alone is deliberately not enough. A scalar is a
  basic index wherever it appears, applied before any advanced index rather
  than broadcast against one. So
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
`__array_ufunc__` nor `__array_function__`. A NumPy *function* given a view
therefore materializes the whole thing through
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

Ownership
---------
`result()` always allocates fresh system memory before reading through the
selected reader. A `numpy.ma` source keeps its mask by receiving a masked
output buffer; other source-specific array types do not survive materializing.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

from zarr_indexing.boundary import (
    SelectionMode,
    normalize_positional_selection,
    split_scalar_axes,
)
from zarr_indexing.chunk_resolution import (
    ChunkProjection,
    plan_chunks,
)
from zarr_indexing.grid import DimensionGrid, FixedDimension, dimension_grids_from_chunks
from zarr_indexing.json import transform_to_canonical
from zarr_indexing.output_map import ArrayMap, ConstantMap, DimensionMap
from zarr_indexing.reader import (
    ReadContext,
    Reader,
    basic_reader,
    numpy_reader,
)
from zarr_indexing.transform import (
    IndexTransform,
    index_array_structure,
    selection_to_transform,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    SelectFn = Callable[[Any, SelectionMode], "LazyArray"]

__all__ = ["LazyArray", "Partition"]

# Above this many bytes, the no-dask token fallback describes an array
# structurally instead of digesting its contents. See `_wrapped_token`.
_TOKEN_DIGEST_LIMIT = 1 << 20


def _invoke_reader(
    reader: Reader,
    source: Any,
    context: ReadContext,
    out: np.ndarray[Any, Any],
) -> None:
    """Invoke a reader and enforce its in-place return contract."""
    returned = reader.read_into(source, context, out)
    if returned is not None:
        raise TypeError(f"reader.read_into must return None, got {type(returned).__name__}")


def _is_correlated(transform: IndexTransform) -> bool:
    """True when the transform gathers a list of points rather than an outer product."""
    return index_array_structure(transform) == "general"


class ArrayLike(Protocol):
    """The surface `LazyArray` needs from the array it wraps."""

    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def dtype(self) -> Any: ...
    def __getitem__(self, key: Any) -> Any: ...


# NumPy's dtype-specialized ``__getitem__`` overloads do not structurally match
# the deliberately broad protocol above under strict type checking.  Accept an
# ndarray explicitly so users do not have to erase its type with ``cast(Any, …)``.
_WrappedArray = ArrayLike | np.ndarray[Any, Any]


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


def _discover_parts(array: Any, shape: tuple[int, ...]) -> tuple[DimensionGrid, ...] | None:
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


def _whole_array_grids(shape: tuple[int, ...]) -> tuple[DimensionGrid, ...]:
    """A partitioning with a single part covering the whole array."""
    return tuple(FixedDimension(size=extent, extent=extent) for extent in shape)


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


# --------------------------------------------------------------------------- #
# Partitions
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True, eq=False)
class _PartOwner:
    """Opaque identity shared only by one view and the parts it prepared."""


def _partition_out_selection(
    cell_transform: IndexTransform,
) -> tuple[Any, ...]:
    """Lower ``cell_transform`` to NumPy selectors on the request buffer."""
    domain = cell_transform.domain
    if _is_correlated(cell_transform):
        correlated_selectors: list[np.ndarray[Any, np.dtype[np.intp]]] = []
        for output_map in cell_transform.output:
            if isinstance(output_map, ConstantMap):
                coordinates = np.full(domain.shape, output_map.offset, dtype=np.intp)
            elif isinstance(output_map, DimensionMap):
                input_dimension = output_map.input_dimension
                axis = np.arange(
                    domain.inclusive_min[input_dimension],
                    domain.exclusive_max[input_dimension],
                    dtype=np.intp,
                )
                shape = (
                    (1,) * input_dimension
                    + (axis.size,)
                    + ((1,) * (domain.ndim - input_dimension - 1))
                )
                coordinates = np.broadcast_to(axis.reshape(shape), domain.shape)
                coordinates = output_map.offset + output_map.stride * coordinates
            else:
                coordinates = output_map.offset + output_map.stride * np.broadcast_to(
                    output_map.index_array, domain.shape
                )
            correlated_selectors.append(np.asarray(coordinates, dtype=np.intp))
        return tuple(correlated_selectors)

    selectors: list[int | slice | np.ndarray[Any, np.dtype[np.intp]]] = []
    n_array_maps = sum(isinstance(output_map, ArrayMap) for output_map in cell_transform.output)
    for output_map in cell_transform.output:
        if isinstance(output_map, ConstantMap):
            selectors.append(output_map.offset)
        elif isinstance(output_map, DimensionMap):
            input_dimension = output_map.input_dimension
            lo = domain.inclusive_min[input_dimension]
            hi = domain.exclusive_max[input_dimension]
            selectors.append(
                slice(
                    output_map.offset + output_map.stride * lo,
                    output_map.offset + output_map.stride * hi,
                    output_map.stride,
                )
            )
        else:
            selectors.append(
                (output_map.offset + output_map.stride * output_map.index_array.ravel()).astype(
                    np.intp
                )
            )
    if n_array_maps > 1:
        axes = [
            np.asarray([selector], dtype=np.intp)
            if isinstance(selector, int)
            else (
                np.arange(selector.start, selector.stop, selector.step, dtype=np.intp)
                if isinstance(selector, slice)
                else selector
            )
            for selector in selectors
        ]
        return np.ix_(*axes)
    return tuple(selectors)


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


@dataclass(frozen=True, kw_only=True)
class Partition:
    """One box of a `LazyArray`'s partitioning, as it falls through the view.

    Yielded by [`LazyArray.parts`][zarr_indexing.lazy_array.LazyArray.parts].
    The parts of a view tile it exactly and disjointly: assembling every
    `view.result()` at its `out_selection` reproduces the whole view's
    `result()`, and each part can be resolved independently and concurrently.
    Derived parts retain the same reader object; a shared stateful reader owns
    synchronization for concurrent calls.

    A consumer that needs the plan before materialization can prepare it once
    and reuse the same immutable parts for both scheduling and assembly:

    ```python
    parts = tuple(view.parts())
    schedule(part.base_coords for part in parts)
    values = view.result(parts=parts)
    ```

    Prepared parts are owned by the exact view that created them and must tile
    it completely. Passing parts from another view, even an equivalent one, is
    rejected without reading; omitting a part is likewise rejected rather than
    returning a partly initialized result.

    Attributes
    ----------
    projection
        The source-independent description of this part. Its paired
        `chunk_transform` and `cell_transform` share one compact synthetic
        domain, mapping each selected cell to chunk-local storage and request
        coordinates respectively. This is the authoritative placement model;
        `base_coords` and `is_complete` are conveniences derived from it.
    base_coords
        Which box of the base partitioning this is, one coordinate per dimension
        of the wrapped array.
    box
        The box itself, in the global storage coordinates of the wrapped
        array: one `[inclusive_min, exclusive_max)` interval per dimension. It
        describes the whole partition cell, while `view.bounding_box()` is the
        global hull of only the selected values in that cell. For a nested or
        repartitioned view this box may be narrower than
        `projection.chunk_domain`.
    view
        A `LazyArray` covering exactly the cells of the view that live in this
        box. Its transform directly addresses its raw wrapped `array`; only the
        projection's `chunk_transform` is chunk-local. Resolving the view reads
        the box once through its selected reader. Named `view` rather than
        `array` because `LazyArray.array` is the opposite thing — the raw
        wrapped source — and the two sat next to each other meaning inverses.
    out_selection
        Where `view.result()` belongs in an array of the whole view's shape — a
        NumPy index tuple with one entry per dimension of the view, usable
        directly as `out[part.out_selection] = ...`.
    is_complete
        Whether the view covers the whole box. Useful to a writer deciding
        between a blind overwrite and a read-modify-write. Fancy projections
        report `False` because their coverage is deliberately `unknown` until
        duplicate-aware proof is added.

    Examples
    --------
    Assembling every part's result at its `out_selection` reproduces the view:

    >>> import numpy as np
    >>> source = np.arange(12).reshape(3, 4)
    >>> view = LazyArray.from_numpy(source).with_parts((2, 2))
    >>> out = np.empty(view.shape, dtype=view.dtype)
    >>> for part in view.parts():
    ...     out[part.out_selection] = part.view.result()
    >>> bool((out == source).all())
    True
    """

    projection: ChunkProjection
    box: tuple[tuple[int, int], ...]
    view: LazyArray
    out_selection: tuple[Any, ...]
    _owner: _PartOwner | None = field(default=None, repr=False, compare=False)

    @property
    def base_coords(self) -> tuple[int, ...]:
        """Coordinates of this partition in the selected base grid."""
        return self.projection.chunk_coords

    @property
    def is_complete(self) -> bool:
        """Whether the projection proves it covers the entire selected cell."""
        return self.projection.coverage == "full"


def _validate_prepared_parts(parts: Sequence[Partition], out_shape: tuple[int, ...]) -> None:
    """Require `parts` to address every output cell exactly once.

    Prepared parts are caller-supplied input, so a plan that does not tile the
    view is a `ValueError`, not an assertion about this library's own walk.
    """
    coverage = np.zeros(out_shape, dtype=np.bool_)
    addressed = 0
    try:
        for part in parts:
            # Name a rank mismatch explicitly instead of letting NumPy treat
            # omitted selectors as implicit full slices.
            addressed += _out_selection_cell_count(part.out_selection, out_shape)
            if all(isinstance(selector, slice) for selector in part.out_selection):
                # The common box part: plain assignment, no pointwise walk.
                coverage[part.out_selection] = True
            else:
                # `logical_or.at` applies duplicate advanced coordinates one by
                # one instead of buffering them as ordinary advanced indexing
                # would.
                np.logical_or.at(coverage, part.out_selection, True)
    except (AssertionError, IndexError, TypeError, ValueError) as error:
        raise ValueError("prepared parts do not tile the view exactly") from error
    if addressed != math.prod(out_shape) or not np.all(coverage):
        raise ValueError("prepared parts do not tile the view exactly")


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
    """A lazily-indexable view over a system-memory/basic-indexing source.

    Wrapping neither copies nor reads the wrapped array at construction time.
    Indexing through `.lazy` composes an `IndexTransform` and returns another
    `LazyArray`; `result()` materializes.

    Selections use the **positional NumPy dialect** and reads are broken up
    along a **partitioning** discovered from the wrapped array. Every derived
    view retains its reader; that reader receives the complete projected
    transform once per part. See the module docstring, which also covers how the
    dialect differs from `zarr.Array.lazy` and why every non-indexing NumPy
    operation materializes the view.

    Parameters
    ----------
    array
        The array to wrap. It must expose `shape`, `dtype`, and `__getitem__`
        with basic (integer/slice) indexing. Its partitioning, if it advertises
        one, is discovered here; use `with_parts` to choose a different one.
        This conservative constructor selects `basic_reader`; use `from_numpy`
        for a NumPy array or `with_reader` to select another backend adapter.

    Examples
    --------
    >>> import numpy as np
    >>> source = np.arange(12).reshape(3, 4)
    >>> view = LazyArray.from_numpy(source).with_parts((2, 2)).lazy[1:, ::2]
    >>> view.shape
    (2, 2)
    >>> view.result()
    array([[ 4,  6],
           [ 8, 10]])
    """

    __slots__ = ("_array", "_part_owner", "_parts", "_reader", "_transform", "_window")

    def __init__(self, array: _WrappedArray) -> None:
        """Wrap `array` without reading it; parameters are documented on the class.

        The only validation here is the `numpy.matrix` rejection (`TypeError`).
        """
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
        self._reader = basic_reader
        self._part_owner = _PartOwner()

    @classmethod
    def from_numpy(cls, array: np.ndarray[Any, Any]) -> LazyArray:
        """Wrap a NumPy array with its explicitly selected optimized reader."""
        if not isinstance(cast(object, array), np.ndarray):
            raise TypeError(
                f"LazyArray.from_numpy requires a numpy.ndarray, got {type(array).__name__}"
            )
        return cls(array).with_reader(numpy_reader)

    @classmethod
    def _derive(
        cls,
        array: _WrappedArray,
        transform: IndexTransform,
        parts: tuple[DimensionGrid, ...] | None,
        window: tuple[slice, ...] | None,
        reader: Reader,
    ) -> LazyArray:
        """Build a wrapper sharing `array` but carrying a new transform or partitioning."""
        view = cls.__new__(cls)
        view._array = array
        # Views re-zero their coordinate system: the positional dialect means a
        # view's first element is at position 0 whatever it was sliced from.
        view._transform = transform.translate_domain_to((0,) * transform.input_rank)
        view._parts = parts
        view._window = window
        view._reader = reader
        view._part_owner = _PartOwner()
        return view

    @property
    def _base_shape(self) -> tuple[int, ...]:
        """The shape of what this wrapper treats as its base array."""
        if self._window is None:
            return tuple(int(s) for s in self._array.shape)
        return tuple(s.stop - s.start for s in self._window)

    # -- array-like surface -------------------------------------------------

    @property
    def array(self) -> _WrappedArray:
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
        """Number of dimensions of this view — the transform's input rank."""
        return self._transform.input_rank

    @property
    def size(self) -> int:
        """Total number of elements in this view (the product of `shape`)."""
        return math.prod(self.shape)

    @property
    def dtype(self) -> Any:
        """The wrapped array's dtype; views never change it."""
        return self._array.dtype

    @property
    def reader(self) -> Reader:
        """The backend adapter used when this view materializes."""
        return self._reader

    def with_reader(self, reader: Reader) -> LazyArray:
        """Return the same metadata view resolved through `reader`."""
        if not callable(getattr(reader, "read_into", None)):
            raise TypeError(f"reader.read_into must be callable, got {type(reader).__name__}")
        return LazyArray._derive(
            self._array,
            self._transform,
            self._parts,
            self._window,
            reader,
        )

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
        >>> array = LazyArray.from_numpy(np.arange(12).reshape(3, 4))
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
        The coordinates directly address the raw `array` this view exposes.
        Consequently, partition views report source-global hulls;
        [`Partition.box`][zarr_indexing.lazy_array.Partition] separately gives
        the whole global partition cell rather than only the selected hull.

        Examples
        --------
        >>> import numpy as np
        >>> array = LazyArray.from_numpy(np.arange(12).reshape(3, 4))
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
        >>> array = LazyArray.from_numpy(np.arange(24).reshape(4, 6))
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
            Uniform part sizes must remain positive even for a zero-length
            axis; use `with_parts_per_axis` for the accepted explicit zero-axis
            spellings.

        Examples
        --------
        >>> import numpy as np
        >>> view = LazyArray.from_numpy(np.arange(12).reshape(3, 4))
        >>> [part.base_coords for part in view.with_parts((2, 3)).parts()]
        [(0, 0), (0, 1), (1, 0), (1, 1)]
        """
        entries = self._part_entries(parts, "with_parts")
        if any(isinstance(entry, Sequence) for entry in entries):
            raise ValueError(
                "with_parts takes one integer per dimension; for per-axis box "
                "sizes use with_parts_per_axis"
            )
        return self._with_grids(dimension_grids_from_chunks(entries, self._base_shape))

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
            If `sizes` has the wrong length, contains a negative extent, uses a
            zero extent on a nonempty axis, or declares sizes that do not sum
            to `base_shape`. On a zero-length axis, `()`, `(0,)`, and repeated
            zeros all describe no chunks.

        Examples
        --------
        >>> import numpy as np
        >>> view = LazyArray.from_numpy(np.arange(12).reshape(3, 4))
        >>> [part.box for part in view.with_parts_per_axis(((1, 2), (4,))).parts()]
        [((0, 1), (0, 4)), ((1, 3), (0, 4))]
        """
        entries = self._part_entries(sizes, "with_parts_per_axis")
        return self._with_grids(dimension_grids_from_chunks(entries, self._base_shape))

    @staticmethod
    def _part_entries(parts: Sequence[Any], method: str) -> tuple[Any, ...]:
        """Materialize a partitioning argument, naming a non-iterable a ValueError.

        Both partitioning methods document ValueError for malformed input; a
        bare integer would otherwise surface as a TypeError from iteration.
        """
        try:
            return tuple(parts)
        except TypeError as error:
            raise ValueError(
                f"{method} takes one entry per dimension of base_shape; got {parts!r}"
            ) from error

    def unpartitioned(self) -> LazyArray:
        """Return the same view, read in one pass.

        `result()` still allocates its owned output buffer first, then calls the
        reader once with the whole projected transform. `parts()` still yields a
        single part covering everything.

        Returns
        -------
        LazyArray
            The same view with no partitioning.
        """
        return self._with_grids(None)

    def _with_grids(self, grids: tuple[DimensionGrid, ...] | None) -> LazyArray:
        return LazyArray._derive(self._array, self._transform, grids, self._window, self._reader)

    def parts(self) -> Iterator[Partition]:
        """Iterate the base partitioning, projected through this view.

        Single-use: this is a generator, so it is consumed by the first walk and
        a second `for` over the same object yields nothing. Call `parts()` again
        for a fresh walk, or keep a `list` of it if you need to revisit.

        Yields one [`Partition`][zarr_indexing.lazy_array.Partition] per box the
        view actually touches. The parts tile the view exactly and disjointly,
        and each carries a `LazyArray` that can be resolved on its own: in
        another thread, in another order, or not at all. Those views share this
        view's reader, and `LazyArray` does not serialize calls, so a stateful
        reader must synchronize its own mutable state.

        A wrapper with no partitioning (see `with_parts`) yields a single part
        covering the whole array.

        Yields
        ------
        Partition
            One per touched box, in the resolver's own order.

        Examples
        --------
        >>> import numpy as np
        >>> view = LazyArray.from_numpy(np.arange(12).reshape(3, 4)).with_parts((2, 2))
        >>> part = next(view.lazy[:, 1:].parts())
        >>> (part.base_coords, part.view.shape, part.is_complete)
        ((0, 0), (2, 1), False)
        """
        base_shape = self._base_shape
        grids = self._parts if self._parts is not None else _whole_array_grids(base_shape)
        rank = len(base_shape)

        if self._window is None:
            plan_transform = self._transform
        else:
            plan_transform = self._transform.translate(tuple(-item.start for item in self._window))

        for projection in plan_chunks(plan_transform, grids):
            base_coords = projection.chunk_coords
            local = projection.chunk_transform
            origin = tuple(grid.chunk_offset(c) for grid, c in zip(grids, base_coords, strict=True))
            extent = tuple(grid.data_size(c) for grid, c in zip(grids, base_coords, strict=True))
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
                projection=projection,
                box=tuple((o, o + e) for o, e in zip(global_origin, extent, strict=True)),
                view=LazyArray._derive(
                    self._array,
                    local.translate(global_origin),
                    None,
                    window,
                    self._reader,
                ),
                out_selection=_partition_out_selection(projection.cell_transform),
                _owner=self._part_owner,
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
        if mode == "basic":
            # IndexTransform's basic path includes NumPy's `None`/newaxis.
            # `selection_to_transform` intentionally exposes a narrower basic
            # selection contract and rejects it.
            composed = transform[literal]
        else:
            composed = selection_to_transform(literal, transform, mode)
        return LazyArray._derive(self._array, composed, self._parts, self._window, self._reader)

    def __getitem__(self, selection: Any) -> Any:
        """Read a basic selection eagerly, like `numpy.ndarray.__getitem__`.

        Reads here are eager, not lazy, so that a `LazyArray` works as a duck
        array for consumers (dask's `from_array`, `numpy.asarray`) that expect
        indexing to produce data. Use `.lazy[...]` for the lazy form.
        """
        return self._select(selection, "basic").result()

    def result(self, *, parts: Sequence[Partition] | None = None) -> Any:
        """Materialize this view.

        Every result starts as a fresh system-memory buffer. Each touched
        partition is read through the selected reader directly into its
        rectangular destination, or into an owned dense temporary before fancy
        placement. Empty views allocate without reading the source.

        Parameters
        ----------
        parts
            A reusable sequence previously returned by this exact view's
            `parts()` method. Supplying it reuses that partition plan instead
            of constructing another one. The parts must tile the view exactly.

        Returns
        -------
        numpy.ndarray
            An array of shape `self.shape`, identical whatever partitioning is
            in force, always in fresh system memory. A view with a zero-rank
            domain returns a zero-dimensional array, not a scalar.

        Raises
        ------
        ValueError
            If supplied parts were prepared by another view, or do not tile
            this view exactly. The output buffer is uninitialized where nothing
            was written, so a bad plan is reported rather than returned.
        AssertionError
            If this library's own partition walk fails to cover the view — a
            bug in zarr-indexing, never a consequence of the caller's input.
        """
        prepared_parts = None if parts is None else tuple(parts)
        if prepared_parts is not None and any(
            # Module-private provenance deliberately crosses the two public
            # wrapper types without becoming part of either public surface.
            part._owner is not self._part_owner  # pyright: ignore[reportPrivateUsage]
            for part in prepared_parts
        ):
            raise ValueError("prepared parts do not belong to this view")

        out_shape = self.shape
        if prepared_parts is not None:
            _validate_prepared_parts(prepared_parts, out_shape)
        out = self._output_buffer(out_shape)
        size = math.prod(out_shape)
        if size == 0:
            return out

        if prepared_parts is None and self._parts is None:
            _invoke_reader(self._reader, self._array, ReadContext(self._transform), out)
            return out

        written = 0
        selected_parts = self.parts() if prepared_parts is None else prepared_parts
        for part in selected_parts:
            # Counted before the scatter, so a part addressing the wrong
            # number of axes is named rather than reported as a broadcast
            # failure against the buffer.
            written += _out_selection_cell_count(part.out_selection, out_shape)
            direct = all(isinstance(selector, slice) for selector in part.out_selection)
            if direct:
                destination = out if len(part.out_selection) == 0 else out[part.out_selection]
            else:
                destination = part.view._output_buffer(part.view.shape)
            _invoke_reader(
                self._reader,
                self._array,
                ReadContext(part.view.transform, part.projection),
                destination,
            )
            if not direct:
                out[part.out_selection] = destination
        if written != size:
            # The buffer is uninitialized where no part wrote, so a partition
            # walk that does not tile the view exactly would otherwise hand
            # back process memory dressed as data. The parts are disjoint by
            # contract, so counting the cells each addresses is enough:
            # a gap undercounts and an overlap overcounts.
            if prepared_parts is not None:
                raise ValueError(
                    "prepared parts do not tile the view exactly: "
                    f"they addressed {written} of the view's {size} cells"
                )
            raise AssertionError(
                f"the partition walk addressed {written} of the view's {size} "
                "cells; this is a bug in zarr-indexing's partition walk"
            )
        return out

    def _output_buffer(self, out_shape: tuple[int, ...]) -> Any:
        """The buffer `result()` scatters parts into.

        Deliberately uninitialized: every cell is written by exactly one part,
        and `result()` verifies that before returning. A masked source gets a
        masked buffer so that reader writes preserve the mask; other source-
        specific array types do not survive materializing.
        """
        dtype = np.dtype(self.dtype)
        if isinstance(self._array, np.ma.MaskedArray):
            return np.ma.masked_all(out_shape, dtype=dtype)
        return np.empty(out_shape, dtype=dtype)

    # -- protocols ----------------------------------------------------------

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> Any:
        """Materialize the view as a NumPy array.

        The result never shares memory with the wrapped array, whatever `copy`
        asks for: `result()` already allocates, so `copy=True` gets an array the
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
        address the same cells. The view contributes a digest of its canonical
        ndsel body, so transforms that differ only in representation produce
        the same token, and a fancy selection with a large index array does not
        embed that array's JSON in the token. See `_wrapped_token` for the
        determinism scope of the wrapped array's contribution; dask is imported
        lazily and is never a requirement of this package.

        The partitioning and reader are deliberately absent. Both decide how
        the data is read — in which boxes, and through which request strategy —
        and neither changes the values that come back, so two wrappers differing
        only in those describe the same data. A token identifies data, so they
        token alike and a consumer that caches on tokens reuses one result for
        both.
        """
        canonical = json.dumps(transform_to_canonical(self._transform), sort_keys=True)
        return (
            type(self).__qualname__,
            _wrapped_token(self._array),
            hashlib.sha256(canonical.encode()).hexdigest(),
        )

    def __len__(self) -> int:
        """The length of the first axis, as for a NumPy array; `TypeError` on a 0-d view."""
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
