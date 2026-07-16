"""Index transforms — composable, lazy coordinate mappings.

An ``IndexTransform`` pairs an **input domain** (the coordinates a user sees)
with a tuple of **output maps** (the storage coordinates those inputs map to).
One output map per storage dimension. See ``output_map.py`` for the three
output map types.

Key operations:

- **Indexing** (``transform[2:8]``, ``.oindex[idx]``, ``.vindex[idx]``) —
  produces a new transform with a narrower input domain and adjusted output
  maps. No I/O occurs. This is how lazy slicing works.

- **intersect(output_domain)** — restrict to storage coordinates within a
  region. This is chunk resolution: "which of my coordinates fall in this
  chunk?"

- **translate(shift)** — shift all output coordinates. This makes coordinates
  chunk-local: "express my coordinates relative to the chunk origin."

- **compose(outer, inner)** — chain two transforms. See ``composition.py``.

The transform is the atomic unit that connects user-facing indexing to
chunk-level I/O. Every ``Array`` holds a transform (identity by default).
``Array.lazy[...]`` composes a new transform lazily. Reading resolves the
transform against the chunk grid via intersect + translate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np

from zarr.core.transforms.domain import IndexDomain
from zarr.core.transforms.output_map import ArrayMap, ConstantMap, DimensionMap, OutputIndexMap
from zarr.errors import BoundsCheckError, VindexInvalidSelectionError


@dataclass(frozen=True, slots=True)
class IndexTransform:
    """A composable mapping from input coordinates to storage coordinates.

    An ``IndexTransform`` has:

    - ``domain``: an ``IndexDomain`` describing the valid input coordinates
      (the user-facing shape, possibly with non-zero origin).
    - ``output``: a tuple of output maps (one per storage dimension), each
      describing which storage coordinates the inputs touch.

    For a freshly opened array, the transform is the identity: input
    coordinate ``i`` maps to storage coordinate ``i``. Indexing operations
    compose new transforms without I/O.
    """

    domain: IndexDomain
    output: tuple[OutputIndexMap, ...]

    def __post_init__(self) -> None:
        for i, m in enumerate(self.output):
            if isinstance(m, DimensionMap):
                if m.input_dimension < 0 or m.input_dimension >= self.domain.ndim:
                    raise ValueError(
                        f"output[{i}].input_dimension = {m.input_dimension} "
                        f"is out of range for input rank {self.domain.ndim}"
                    )
            elif isinstance(m, ArrayMap) and m.index_array.ndim > self.domain.ndim:
                raise ValueError(
                    f"output[{i}].index_array has {m.index_array.ndim} dims "
                    f"but input domain has {self.domain.ndim} dims"
                )

    @property
    def input_rank(self) -> int:
        return self.domain.ndim

    @property
    def output_rank(self) -> int:
        return len(self.output)

    @classmethod
    def identity(cls, domain: IndexDomain) -> IndexTransform:
        output = tuple(DimensionMap(input_dimension=i) for i in range(domain.ndim))
        return cls(domain=domain, output=output)

    @classmethod
    def from_shape(cls, shape: tuple[int, ...]) -> IndexTransform:
        return cls.identity(IndexDomain.from_shape(shape))

    @property
    def selection_repr(self) -> str:
        """Compact domain string, e.g. ``'{ [2, 8), [0, 10) }'``.

        Follows TensorStore's IndexDomain notation: each dimension shown
        as ``[inclusive_min, exclusive_max)`` with stride annotation if not 1.
        Constant (integer-indexed) dimensions show as a single value.
        Array-indexed dimensions show the set of selected coordinates.
        """
        parts: list[str] = []
        for m in self.output:
            if isinstance(m, ConstantMap):
                parts.append(str(m.offset))
            elif isinstance(m, DimensionMap):
                d = m.input_dimension
                lo = self.domain.inclusive_min[d]
                hi = self.domain.exclusive_max[d]
                start = m.offset + m.stride * lo
                stop = m.offset + m.stride * hi
                if m.stride == 1:
                    parts.append(f"[{start}, {stop})")
                else:
                    parts.append(f"[{start}, {stop}) step {m.stride}")
            elif isinstance(m, ArrayMap):
                storage = m.offset + m.stride * m.index_array
                n = int(storage.size)  # .size, not len(): index_array may be 0-d
                if n <= 5:
                    vals = ", ".join(str(int(v)) for v in storage.ravel())
                    parts.append("{" + vals + "}")
                else:
                    parts.append("{" + f"array({n})" + "}")
        return "{ " + ", ".join(parts) + " }"

    def __repr__(self) -> str:
        maps: list[str] = []
        for i, m in enumerate(self.output):
            if isinstance(m, ConstantMap):
                maps.append(f"out[{i}] = {m.offset}")
            elif isinstance(m, DimensionMap):
                maps.append(f"out[{i}] = {m.offset} + {m.stride} * in[{m.input_dimension}]")
            elif isinstance(m, ArrayMap):
                maps.append(f"out[{i}] = {m.offset} + {m.stride} * arr{m.index_array.shape}[in]")
        maps_str = ", ".join(maps)
        return f"IndexTransform(domain={self.domain}, {maps_str})"

    def intersect(
        self, output_domain: IndexDomain
    ) -> (
        tuple[
            IndexTransform,
            dict[int, np.ndarray[Any, np.dtype[np.intp]]]
            | np.ndarray[Any, np.dtype[np.intp]]
            | None,
        ]
        | None
    ):
        """Restrict this transform to storage coordinates within output_domain.

        Returns ``(restricted_transform, out_indices)`` or None if empty.

        ``out_indices`` carries the surviving output positions: ``None`` when all
        positions survive (ConstantMap/DimensionMap only), a single integer array
        for one ArrayMap (or correlated/vectorized ArrayMaps), or a dict keyed by
        output dimension for >= 2 orthogonal ArrayMaps (an outer product).
        """
        return _intersect(self, output_domain)

    def translate(self, shift: tuple[int, ...]) -> IndexTransform:
        """Shift all output coordinates by ``shift``."""
        if len(shift) != self.output_rank:
            raise ValueError(f"shift must have length {self.output_rank}, got {len(shift)}")
        new_output: list[OutputIndexMap] = []
        for m, s in zip(self.output, shift, strict=True):
            if isinstance(m, ConstantMap):
                new_output.append(ConstantMap(offset=m.offset + s))
            elif isinstance(m, DimensionMap):
                new_output.append(
                    DimensionMap(
                        input_dimension=m.input_dimension,
                        offset=m.offset + s,
                        stride=m.stride,
                    )
                )
            elif isinstance(m, ArrayMap):
                new_output.append(
                    ArrayMap(
                        index_array=m.index_array,
                        offset=m.offset + s,
                        stride=m.stride,
                        input_dimension=m.input_dimension,
                    )
                )
        return IndexTransform(domain=self.domain, output=tuple(new_output))

    def __getitem__(self, selection: Any) -> IndexTransform:
        return _apply_basic_indexing(self, selection)

    def translate_domain_by(self, shift: tuple[int, ...]) -> IndexTransform:
        """Shift the *input* domain by ``shift``, preserving which cells are addressed.

        TensorStore's ``translate_by``: the domain moves, and every output map is
        re-offset so that new coordinate ``c`` addresses the cell that ``c - shift``
        addressed before. ArrayMaps are indexed positionally over the domain, so
        their index arrays are unchanged.
        """
        if len(shift) != self.input_rank:
            raise ValueError(f"shift must have length {self.input_rank}, got {len(shift)}")
        new_domain = self.domain.translate(shift)
        new_output: list[OutputIndexMap] = []
        for m in self.output:
            if isinstance(m, DimensionMap):
                s = shift[m.input_dimension]
                new_output.append(
                    DimensionMap(
                        input_dimension=m.input_dimension,
                        offset=m.offset - m.stride * s,
                        stride=m.stride,
                    )
                )
            else:
                # ConstantMap: no input dependence. ArrayMap: positional over
                # the domain, invariant under domain translation.
                new_output.append(m)
        return IndexTransform(domain=new_domain, output=tuple(new_output))

    def translate_domain_to(self, origins: tuple[int, ...]) -> IndexTransform:
        """Move the input domain so its per-dimension origins equal ``origins``.

        TensorStore's ``translate_to``; ``translate_domain_to((0,) * rank)``
        re-zeros a view's coordinate system without changing which cells it
        addresses.
        """
        if len(origins) != self.input_rank:
            raise ValueError(f"origins must have length {self.input_rank}, got {len(origins)}")
        shift = tuple(o - m for o, m in zip(origins, self.domain.inclusive_min, strict=True))
        return self.translate_domain_by(shift)

    @property
    def oindex(self) -> _OIndexHelper:
        return _OIndexHelper(self)

    @property
    def vindex(self) -> _VIndexHelper:
        return _VIndexHelper(self)


def _intersect(
    transform: IndexTransform, output_domain: IndexDomain
) -> (
    tuple[
        IndexTransform,
        dict[int, np.ndarray[Any, np.dtype[np.intp]]] | np.ndarray[Any, np.dtype[np.intp]] | None,
    ]
    | None
):
    """Intersect a transform with an output domain (e.g., a chunk's bounds).

    For each output dimension, restrict to storage coordinates within
    [output_domain.inclusive_min[d], output_domain.exclusive_max[d]).

    For orthogonal transforms (ConstantMap, DimensionMap, independent ArrayMaps),
    each dimension is intersected independently and the input domain is narrowed.

    For vectorized transforms (correlated ArrayMaps), all array dimensions
    must be checked simultaneously — a point survives only if ALL its
    coordinates fall within the output domain.

    Returns None if the intersection is empty.
    """
    if output_domain.ndim != transform.output_rank:
        raise ValueError(
            f"output_domain rank ({output_domain.ndim}) != "
            f"transform output rank ({transform.output_rank})"
        )

    # Correlated ArrayMaps (vindex) are jointly indexed (input_dimension is None);
    # >= 2 of them means a vectorized intersection. Orthogonal ArrayMaps (oindex)
    # each bind a distinct input dimension and are handled per-dimension below.
    array_dims = [i for i, m in enumerate(transform.output) if isinstance(m, ArrayMap)]
    vectorized_dims = [
        i for i in array_dims if cast("ArrayMap", transform.output[i]).input_dimension is None
    ]
    if len(vectorized_dims) >= 2:
        return _intersect_vectorized(transform, output_domain, vectorized_dims)

    # Orthogonal: intersect each output dimension independently. Multiple
    # ArrayMaps bound to distinct input dimensions form an outer product, so each
    # array dimension's surviving *output* positions are tracked separately.
    new_min = list(transform.domain.inclusive_min)
    new_max = list(transform.domain.exclusive_max)
    new_output: list[OutputIndexMap] = []
    out_positions: dict[int, np.ndarray[Any, np.dtype[np.intp]]] = {}

    for out_dim, m in enumerate(transform.output):
        lo = output_domain.inclusive_min[out_dim]
        hi = output_domain.exclusive_max[out_dim]

        if isinstance(m, ConstantMap):
            if lo <= m.offset < hi:
                new_output.append(m)
            else:
                return None

        elif isinstance(m, DimensionMap):
            d = m.input_dimension
            input_lo = new_min[d]
            input_hi = new_max[d]
            if input_lo >= input_hi:
                return None

            # Find input range that produces storage coords in [lo, hi)
            if m.stride > 0:
                new_input_lo = max(input_lo, math.ceil((lo - m.offset) / m.stride))
                new_input_hi = min(input_hi, math.ceil((hi - m.offset) / m.stride))
            elif m.stride < 0:
                new_input_lo = max(input_lo, math.ceil((hi - 1 - m.offset) / m.stride))
                new_input_hi = min(input_hi, math.ceil((lo - 1 - m.offset) / m.stride))
            else:
                if lo <= m.offset < hi:
                    new_input_lo, new_input_hi = input_lo, input_hi
                else:
                    return None

            if new_input_lo >= new_input_hi:
                return None

            new_min[d] = new_input_lo
            new_max[d] = new_input_hi
            new_output.append(m)

        elif isinstance(m, ArrayMap):
            storage = m.offset + m.stride * m.index_array
            mask = (storage >= lo) & (storage < hi)
            survivors = np.nonzero(mask.ravel())[0].astype(np.intp)
            if survivors.size == 0:
                return None
            filtered = m.index_array.ravel()[survivors]
            new_output.append(
                ArrayMap(
                    index_array=filtered,
                    offset=m.offset,
                    stride=m.stride,
                    input_dimension=m.input_dimension,
                )
            )
            # Narrow this array's own input dimension to the surviving count.
            if m.input_dimension is not None:
                new_min[m.input_dimension] = 0
                new_max[m.input_dimension] = int(survivors.size)
            out_positions[out_dim] = survivors

    new_domain = IndexDomain(
        inclusive_min=tuple(new_min),
        exclusive_max=tuple(new_max),
    )
    result = IndexTransform(domain=new_domain, output=tuple(new_output))

    # Hand back the surviving output positions in the shape the bridge expects:
    # None (no arrays), a single vector (one array), or a per-output-dim dict
    # (>= 2 orthogonal arrays → outer product).
    out_indices: (
        dict[int, np.ndarray[Any, np.dtype[np.intp]]] | np.ndarray[Any, np.dtype[np.intp]] | None
    )
    if len(out_positions) == 0:
        out_indices = None
    elif len(out_positions) == 1:
        out_indices = next(iter(out_positions.values()))
    else:
        out_indices = out_positions
    return (result, out_indices)


def _intersect_vectorized(
    transform: IndexTransform,
    output_domain: IndexDomain,
    array_dims: list[int],
) -> tuple[IndexTransform, np.ndarray[Any, np.dtype[np.intp]] | None] | None:
    """Intersect a vectorized transform with an output domain.

    All ArrayMap outputs are correlated — a point survives only if ALL its
    storage coordinates fall within the output domain.
    """
    # Compute storage coords per array dim and check bounds simultaneously
    masks: list[np.ndarray[Any, np.dtype[np.bool_]]] = []

    for out_dim in array_dims:
        m = transform.output[out_dim]
        assert isinstance(m, ArrayMap)
        storage = m.offset + m.stride * m.index_array
        lo = output_domain.inclusive_min[out_dim]
        hi = output_domain.exclusive_max[out_dim]
        masks.append((storage >= lo) & (storage < hi))

    # A point survives only if it's in-bounds on ALL array dims
    combined_mask = masks[0]
    for mask in masks[1:]:
        combined_mask = combined_mask & mask

    if not np.any(combined_mask):
        return None

    surviving = np.nonzero(combined_mask.ravel())[0].astype(np.intp)

    # Build new output maps
    new_output: list[OutputIndexMap] = []
    for out_dim, m in enumerate(transform.output):
        if isinstance(m, ArrayMap):
            filtered = m.index_array.ravel()[surviving]
            new_output.append(
                ArrayMap(
                    index_array=filtered,
                    offset=m.offset,
                    stride=m.stride,
                    input_dimension=m.input_dimension,
                )
            )
        elif isinstance(m, ConstantMap):
            lo = output_domain.inclusive_min[out_dim]
            hi = output_domain.exclusive_max[out_dim]
            if lo <= m.offset < hi:
                new_output.append(m)
            else:
                return None
        elif isinstance(m, DimensionMap):
            new_output.append(m)

    new_domain = IndexDomain.from_shape((len(surviving),))
    result = IndexTransform(domain=new_domain, output=tuple(new_output))
    return (result, surviving)


def _normalize_basic_selection(selection: Any, ndim: int) -> tuple[int | slice | None, ...]:
    """Normalize a selection to a tuple of int, slice, or None (newaxis),
    expanding ellipsis and padding with slice(None) as needed.
    """
    if not isinstance(selection, tuple):
        selection = (selection,)

    # Count non-newaxis, non-ellipsis entries to determine how many real dims are addressed
    n_newaxis = sum(1 for s in selection if s is None)
    has_ellipsis = any(s is Ellipsis for s in selection)
    n_real = len(selection) - n_newaxis - (1 if has_ellipsis else 0)

    if n_real > ndim:
        raise IndexError(
            f"too many indices for array: array has {ndim} dimensions, but {n_real} were indexed"
        )

    result: list[int | slice | None] = []
    ellipsis_seen = False
    for sel in selection:
        if sel is Ellipsis:
            if ellipsis_seen:
                raise IndexError("an index can only have a single ellipsis ('...')")
            ellipsis_seen = True
            num_missing = ndim - n_real
            result.extend([slice(None)] * num_missing)
        elif isinstance(sel, (int, np.integer)):
            result.append(int(sel))
        elif isinstance(sel, slice) or sel is None:
            result.append(sel)
        else:
            raise IndexError(f"unsupported selection type for basic indexing: {type(sel)!r}")

    # Pad remaining dimensions with slice(None)
    while sum(1 for s in result if s is not None) < ndim:
        result.append(slice(None))

    return tuple(result)


def _reindex_array(
    arr: np.ndarray[Any, np.dtype[np.intp]],
    normalized: tuple[int | slice | None, ...],
    domain: IndexDomain,
) -> np.ndarray[Any, np.dtype[np.intp]]:
    """Apply basic indexing operations to an ArrayMap's index_array.

    The array's axes correspond to the transform's input dimensions (0-indexed
    over the domain shape). When input dimensions are dropped (int), sliced,
    or inserted (newaxis), the array must be updated accordingly.
    """
    # Build a numpy indexing tuple: one entry per old input dimension
    idx: list[Any] = []
    old_dim = 0
    newaxis_positions: list[int] = []
    result_axis = 0

    for sel in normalized:
        if sel is None:
            newaxis_positions.append(result_axis)
            result_axis += 1
        elif isinstance(sel, int):
            if old_dim < arr.ndim:
                # Convert absolute domain coordinate to 0-based array index
                array_idx = sel - domain.inclusive_min[old_dim]
                idx.append(array_idx)
            old_dim += 1
        elif isinstance(sel, slice):
            if old_dim < arr.ndim:
                lo = domain.inclusive_min[old_dim]
                hi = domain.exclusive_max[old_dim]
                # Bounds are literal domain coordinates; the stored array is
                # indexed positionally, so shift by the domain origin.
                start, step, _origin, size = _resolve_slice_ts(sel, old_dim, lo, hi)
                pos = start - lo
                idx.append(slice(pos, pos + size * step, step))
            old_dim += 1
            result_axis += 1

    result = arr[tuple(idx)] if idx else arr

    for pos in newaxis_positions:
        result = np.expand_dims(result, axis=pos)

    return np.asarray(result, dtype=np.intp)


def _reindex_array_oindex(
    arr: np.ndarray[Any, np.dtype[np.intp]],
    normalized: tuple[Any, ...] | list[Any],
    domain: IndexDomain,
) -> np.ndarray[Any, np.dtype[np.intp]]:
    """Apply oindex/vindex selection to an existing ArrayMap's index_array.

    Each old input dimension gets either an array (fancy index that axis)
    or a slice applied to the corresponding array axis.
    """
    idx: list[Any] = []
    for old_dim, sel in enumerate(normalized):
        if old_dim >= arr.ndim:
            break
        lo = domain.inclusive_min[old_dim]
        if isinstance(sel, np.ndarray):
            # Values are literal domain coordinates; the stored array is
            # indexed positionally, so shift by the domain origin.
            idx.append(sel - lo)
        elif isinstance(sel, slice):
            hi = domain.exclusive_max[old_dim]
            start, step, _origin, size = _resolve_slice_ts(sel, old_dim, lo, hi)
            pos = start - lo
            idx.append(slice(pos, pos + size * step, step))
        else:
            idx.append(slice(None))

    result = arr[tuple(idx)] if idx else arr
    return np.asarray(result, dtype=np.intp)


def _apply_basic_indexing(transform: IndexTransform, selection: Any) -> IndexTransform:
    """Apply basic indexing (int, slice, ellipsis, newaxis) to an IndexTransform."""
    normalized = _normalize_basic_selection(selection, transform.domain.ndim)

    new_inclusive_min: list[int] = []
    new_exclusive_max: list[int] = []
    old_dim = 0
    new_dim_idx = 0
    old_to_new_dim: dict[int, int] = {}
    dropped_dims: set[int] = set()

    # Per old-dim: the slice parameters (for computing new output maps)
    dim_slice_params: dict[int, tuple[int, int, int]] = {}  # old_dim -> (start, stop, step)
    dim_int_val: dict[int, int] = {}  # old_dim -> integer index value

    for sel in normalized:
        if sel is None:
            # newaxis: add a size-1 dimension
            new_inclusive_min.append(0)
            new_exclusive_max.append(1)
            new_dim_idx += 1
        elif isinstance(sel, int):
            # Integer index: drop this input dimension.
            # Negative indices are literal coordinates (TensorStore convention),
            # NOT "from the end" like NumPy. The Array layer handles conversion.
            lo = transform.domain.inclusive_min[old_dim]
            hi = transform.domain.exclusive_max[old_dim]
            idx = sel
            if idx < lo or idx >= hi:
                hint = _LITERAL_HINT if sel < 0 else ""
                raise BoundsCheckError(
                    f"index {sel} is out of bounds for dimension {old_dim} "
                    f"(valid indices [{lo}, {hi})){hint}"
                )
            dropped_dims.add(old_dim)
            dim_int_val[old_dim] = idx
            old_dim += 1
        elif isinstance(sel, slice):
            lo = transform.domain.inclusive_min[old_dim]
            hi = transform.domain.exclusive_max[old_dim]

            # TensorStore semantics: bounds are literal coordinates; a step-1
            # slice keeps them as the new domain, a strided slice's domain is
            # [trunc(start/step), trunc(start/step) + size).
            start, step, origin, size = _resolve_slice_ts(sel, old_dim, lo, hi)
            new_inclusive_min.append(origin)
            new_exclusive_max.append(origin + size)
            dim_slice_params[old_dim] = (start, step, origin)
            old_to_new_dim[old_dim] = new_dim_idx
            new_dim_idx += 1
            old_dim += 1

    new_domain = IndexDomain(
        inclusive_min=tuple(new_inclusive_min),
        exclusive_max=tuple(new_exclusive_max),
    )

    # Now update output maps
    new_output: list[OutputIndexMap] = []
    for m in transform.output:
        if isinstance(m, ConstantMap):
            new_output.append(m)
        elif isinstance(m, DimensionMap):
            d = m.input_dimension
            if d in dropped_dims:
                # Integer index: this output becomes constant
                new_offset = m.offset + m.stride * dim_int_val[d]
                new_output.append(ConstantMap(offset=new_offset))
            elif d in old_to_new_dim:
                # Slice: new coordinate `origin + k` maps to old coordinate
                # `start + k*step`, i.e. old = start - step*origin + step*new.
                start, step, origin = dim_slice_params[d]
                new_offset = m.offset + m.stride * (start - step * origin)
                new_stride = m.stride * step
                new_input_dim = old_to_new_dim[d]
                new_output.append(
                    DimensionMap(
                        input_dimension=new_input_dim, offset=new_offset, stride=new_stride
                    )
                )
            else:
                raise RuntimeError(f"unexpected: dimension {d} not handled")
        elif isinstance(m, ArrayMap):
            new_arr = _reindex_array(m.index_array, normalized, transform.domain)
            array_input_dim: int | None = None
            if m.input_dimension is not None:
                array_input_dim = old_to_new_dim.get(m.input_dimension, m.input_dimension)
            new_output.append(
                ArrayMap(
                    index_array=new_arr,
                    offset=m.offset,
                    stride=m.stride,
                    input_dimension=array_input_dim,
                )
            )

    return IndexTransform(domain=new_domain, output=tuple(new_output))


def _array_map_dependency_axes(index_array: np.ndarray[Any, Any]) -> tuple[int, ...]:
    """Return the input axes on which a normalized index array varies.

    Normalized `ArrayMap` index arrays carry the full input rank of their
    enclosing transform: an axis the array varies over has its full size, while
    an axis the array is independent of is a singleton (size 1). The dependency
    axes are therefore exactly the non-singleton axes. An orthogonal (`oindex`)
    array depends on a single axis; a vectorized (`vindex`) array depends on all
    of the (shared) broadcast axes.
    """
    return tuple(axis for axis, size in enumerate(index_array.shape) if size != 1)


def _reshape_to_axis(
    values: np.ndarray[Any, np.dtype[np.intp]], axis: int, ndim: int
) -> np.ndarray[Any, np.dtype[np.intp]]:
    """Reshape a 1-D selection to full rank ``ndim`` varying only along ``axis``.

    The result has ``values`` laid out along ``axis`` and singleton (size-1) axes
    everywhere else, so its dependency axis is derivable from its shape.
    """
    flat = np.asarray(values, dtype=np.intp).ravel()
    shape = [1] * ndim
    shape[axis] = flat.shape[0]
    return flat.reshape(shape)


class _OIndexHelper:
    """Helper that provides orthogonal (outer) indexing via ``transform.oindex[...]``."""

    def __init__(self, transform: IndexTransform) -> None:
        self._transform = transform

    def __getitem__(self, selection: Any) -> IndexTransform:
        return _apply_oindex(self._transform, selection)


def _normalize_oindex_selection(
    selection: Any, ndim: int
) -> tuple[np.ndarray[Any, np.dtype[np.intp]] | slice, ...]:
    """Normalize an oindex selection: arrays, slices, booleans, integers."""
    if not isinstance(selection, tuple):
        selection = (selection,)

    # Expand ellipsis
    has_ellipsis = any(s is Ellipsis for s in selection)
    n_ellipsis = 1 if has_ellipsis else 0
    n_real = len(selection) - n_ellipsis

    result: list[np.ndarray[Any, np.dtype[np.intp]] | slice] = []
    for sel in selection:
        if sel is Ellipsis:
            num_missing = ndim - n_real
            result.extend([slice(None)] * num_missing)
        elif isinstance(sel, np.ndarray) and sel.dtype == np.bool_:
            # Boolean array -> integer indices
            (indices,) = np.nonzero(sel)
            result.append(indices.astype(np.intp))
        elif isinstance(sel, np.ndarray):
            result.append(sel.astype(np.intp))
        elif isinstance(sel, slice):
            result.append(sel)
        elif isinstance(sel, (int, np.integer)):
            # Convert integer scalars to 1-element arrays for orthogonal indexing
            result.append(np.array([int(sel)], dtype=np.intp))
        elif isinstance(sel, (list, tuple)):
            result.append(np.asarray(sel, dtype=np.intp))
        else:
            result.append(sel)

    # Pad with slice(None)
    while len(result) < ndim:
        result.append(slice(None))

    return tuple(result)


def _apply_oindex(transform: IndexTransform, selection: Any) -> IndexTransform:
    """Apply orthogonal indexing to an IndexTransform.

    Each index array is applied independently per dimension (outer product).
    """
    normalized = _normalize_oindex_selection(selection, transform.domain.ndim)

    new_inclusive_min: list[int] = []
    new_exclusive_max: list[int] = []
    new_dim_idx = 0
    old_to_new_dim: dict[int, int] = {}

    # Info per old dim
    dim_array: dict[int, np.ndarray[Any, np.dtype[np.intp]]] = {}
    dim_slice_params: dict[int, tuple[int, int, int]] = {}

    for old_dim, sel in enumerate(normalized):
        if isinstance(sel, np.ndarray):
            lo = transform.domain.inclusive_min[old_dim]
            hi = transform.domain.exclusive_max[old_dim]
            # Index-array values are literal domain coordinates; the fancy dim
            # they create gets a fresh zero-origin [0, n) domain (TensorStore).
            _check_array_in_bounds(sel, lo, hi)
            dim_array[old_dim] = sel
            new_inclusive_min.append(0)
            new_exclusive_max.append(len(sel))
            old_to_new_dim[old_dim] = new_dim_idx
            new_dim_idx += 1
        elif isinstance(sel, slice):
            lo = transform.domain.inclusive_min[old_dim]
            hi = transform.domain.exclusive_max[old_dim]
            start, step, origin, size = _resolve_slice_ts(sel, old_dim, lo, hi)
            new_inclusive_min.append(origin)
            new_exclusive_max.append(origin + size)
            dim_slice_params[old_dim] = (start, step, origin)
            old_to_new_dim[old_dim] = new_dim_idx
            new_dim_idx += 1

    new_domain = IndexDomain(
        inclusive_min=tuple(new_inclusive_min),
        exclusive_max=tuple(new_exclusive_max),
    )

    new_output: list[OutputIndexMap] = []
    for m in transform.output:
        if isinstance(m, ConstantMap):
            new_output.append(m)
        elif isinstance(m, DimensionMap):
            d = m.input_dimension
            if d in dim_array:
                new_axis = old_to_new_dim[d]
                # Normalize to full input rank: the selection varies along its own
                # new axis and is singleton on every other axis. The dependency
                # axis is then derivable from the shape (a single non-singleton
                # axis marks the selection orthogonal / outer-product rather than
                # vectorized). `input_dimension` is kept populated as a
                # compatibility shim for consumers not yet migrated to the
                # shape-derived classifier.
                full_arr = _reshape_to_axis(dim_array[d], new_axis, new_dim_idx)
                new_output.append(
                    ArrayMap(
                        index_array=full_arr,
                        offset=m.offset,
                        stride=m.stride,
                        input_dimension=new_axis,
                    )
                )
            elif d in dim_slice_params:
                start, step, origin = dim_slice_params[d]
                new_offset = m.offset + m.stride * (start - step * origin)
                new_stride = m.stride * step
                new_input_dim = old_to_new_dim[d]
                new_output.append(
                    DimensionMap(
                        input_dimension=new_input_dim, offset=new_offset, stride=new_stride
                    )
                )
            else:
                raise RuntimeError(f"unexpected: dimension {d} not handled")
        elif isinstance(m, ArrayMap):
            new_arr = _reindex_array_oindex(m.index_array, normalized, transform.domain)
            array_input_dim: int | None = None
            if m.input_dimension is not None:
                array_input_dim = old_to_new_dim.get(m.input_dimension, m.input_dimension)
            new_output.append(
                ArrayMap(
                    index_array=new_arr,
                    offset=m.offset,
                    stride=m.stride,
                    input_dimension=array_input_dim,
                )
            )

    return IndexTransform(domain=new_domain, output=tuple(new_output))


class _VIndexHelper:
    """Helper that provides vectorized (fancy) indexing via ``transform.vindex[...]``."""

    def __init__(self, transform: IndexTransform) -> None:
        self._transform = transform

    def __getitem__(self, selection: Any) -> IndexTransform:
        return _apply_vindex(self._transform, selection)


def _apply_vindex(transform: IndexTransform, selection: Any) -> IndexTransform:
    """Apply vectorized indexing to an IndexTransform.

    All array indices are broadcast together. Broadcast dimensions are prepended,
    followed by non-array (slice) dimensions.
    """
    if not isinstance(selection, tuple):
        selection = (selection,)

    # Expand ellipsis and count consumed dimensions
    # Boolean arrays with ndim > 1 consume ndim dims
    n_consumed = 0
    for s in selection:
        if s is Ellipsis:
            continue
        if isinstance(s, np.ndarray) and s.dtype == np.bool_ and s.ndim > 1:
            n_consumed += s.ndim
        else:
            n_consumed += 1
    ndim = transform.domain.ndim

    expanded: list[Any] = []
    for sel in selection:
        if sel is Ellipsis:
            num_missing = ndim - n_consumed
            expanded.extend([slice(None)] * num_missing)
        else:
            expanded.append(sel)
    # Count dimensions already consumed by expanded entries
    n_expanded_dims = 0
    for sel in expanded:
        if isinstance(sel, np.ndarray) and sel.dtype == np.bool_ and sel.ndim > 1:
            n_expanded_dims += sel.ndim
        else:
            n_expanded_dims += 1
    while n_expanded_dims < ndim:
        expanded.append(slice(None))
        n_expanded_dims += 1

    # Convert booleans, lists, ints to integer arrays
    processed: list[np.ndarray[Any, np.dtype[np.intp]] | slice] = []
    for sel in expanded:
        if isinstance(sel, np.ndarray) and sel.dtype == np.bool_:
            indices_tuple = np.nonzero(sel)
            processed.extend(indices.astype(np.intp) for indices in indices_tuple)
        elif isinstance(sel, np.ndarray):
            processed.append(sel.astype(np.intp))
        elif isinstance(sel, (list, tuple)):
            processed.append(np.asarray(sel, dtype=np.intp))
        elif isinstance(sel, (int, np.integer)):
            processed.append(np.array([int(sel)], dtype=np.intp))
        else:
            processed.append(sel)

    # Separate array dims and slice dims
    array_dims: list[int] = []
    slice_dims: list[int] = []
    arrays: list[np.ndarray[Any, np.dtype[np.intp]]] = []

    for i, sel in enumerate(processed):
        if isinstance(sel, np.ndarray):
            lo = transform.domain.inclusive_min[i]
            hi = transform.domain.exclusive_max[i]
            _check_array_in_bounds(sel, lo, hi)
            array_dims.append(i)
            arrays.append(sel)
        else:
            slice_dims.append(i)

    # Broadcast all arrays together
    broadcast_arrays: list[np.ndarray[Any, np.dtype[np.intp]]]
    if arrays:
        broadcast_arrays = list(np.broadcast_arrays(*arrays))
        broadcast_shape = broadcast_arrays[0].shape
    else:
        broadcast_arrays = []
        broadcast_shape = ()

    # Build new domain: broadcast dims first, then slice dims
    new_inclusive_min: list[int] = []
    new_exclusive_max: list[int] = []

    # Broadcast dimensions
    for s in broadcast_shape:
        new_inclusive_min.append(0)
        new_exclusive_max.append(s)

    # Slice dimensions (preserved-domain literal semantics, like basic indexing)
    slice_dim_params: dict[int, tuple[int, int, int]] = {}
    for old_dim in slice_dims:
        sel = processed[old_dim]
        assert isinstance(sel, slice)
        lo = transform.domain.inclusive_min[old_dim]
        hi = transform.domain.exclusive_max[old_dim]
        start, step, origin, size = _resolve_slice_ts(sel, old_dim, lo, hi)
        new_inclusive_min.append(origin)
        new_exclusive_max.append(origin + size)
        slice_dim_params[old_dim] = (start, step, origin)

    new_domain = IndexDomain(
        inclusive_min=tuple(new_inclusive_min),
        exclusive_max=tuple(new_exclusive_max),
    )

    # Build output maps
    array_dim_to_broadcast: dict[int, np.ndarray[Any, np.dtype[np.intp]]] = {}
    for i, d in enumerate(array_dims):
        array_dim_to_broadcast[d] = broadcast_arrays[i]

    # New dim index for slice dims starts after broadcast dims
    n_broadcast_dims = len(broadcast_shape)

    new_output: list[OutputIndexMap] = []
    for m in transform.output:
        if isinstance(m, ConstantMap):
            new_output.append(m)
        elif isinstance(m, DimensionMap):
            d = m.input_dimension
            if d in array_dim_to_broadcast:
                # Normalize to full input rank: the broadcast (correlated) axes
                # come first, followed by a singleton axis per slice dimension.
                # Every vectorized array shares the same broadcast axes, so the
                # dependency axes derived from the shape coincide — the signature
                # of a pointwise scatter rather than an outer product.
                broadcast_arr = array_dim_to_broadcast[d]
                full_arr = broadcast_arr.reshape(broadcast_shape + (1,) * len(slice_dims))
                new_output.append(
                    ArrayMap(
                        index_array=full_arr,
                        offset=m.offset,
                        stride=m.stride,
                    )
                )
            else:
                # Slice dim: new coord `origin + k` maps to old `start + k*step`
                start, step, origin = slice_dim_params[d]
                new_offset = m.offset + m.stride * (start - step * origin)
                new_stride = m.stride * step
                new_input_dim = n_broadcast_dims + slice_dims.index(d)
                new_output.append(
                    DimensionMap(
                        input_dimension=new_input_dim, offset=new_offset, stride=new_stride
                    )
                )
        elif isinstance(m, ArrayMap):
            new_arr = _reindex_array_oindex(m.index_array, processed, transform.domain)
            new_output.append(
                ArrayMap(
                    index_array=new_arr,
                    offset=m.offset,
                    stride=m.stride,
                    input_dimension=m.input_dimension,
                )
            )

    return IndexTransform(domain=new_domain, output=tuple(new_output))


_LITERAL_HINT = (
    "; negative indices are literal coordinates in lazy indexing, not from-the-end "
    "(use `shape[dim] - k`, or materialize with `result()`, for NumPy semantics)"
)


def _trunc_div(a: int, b: int) -> int:
    """Integer division rounded toward zero (C semantics), as TensorStore uses
    for strided-slice domain origins — distinct from Python's floor division
    for negative operands (``trunc(-9/2) == -4`` where ``-9 // 2 == -5``)."""
    q = a // b
    if q < 0 and q * b != a:
        q += 1
    return q


def _resolve_slice_ts(sel: slice, dim: int, lo: int, hi: int) -> tuple[int, int, int, int]:
    """Resolve a slice against domain ``[lo, hi)`` with TensorStore semantics.

    Slice bounds are **literal domain coordinates** — never from-the-end, never
    clamped. Rules (each verified against tensorstore 0.1.84):

    - defaults: ``start = lo``, ``stop = hi``;
    - a non-empty interval must be contained in the domain (no clamping — a
      NumPy-style out-of-range or negative bound is an error, not a shorter or
      wrapped result);
    - an **empty** interval (``start == stop``) is valid anywhere;
    - reversed bounds (``start > stop`` with positive step) are an error, not
      an empty result;
    - the result's domain origin is ``trunc(start/step)`` (rounded toward
      zero) and coordinate ``origin + k`` maps to input ``start + k*step``.

    Returns ``(start, step, origin, size)`` in domain coordinates.
    """
    step = 1 if sel.step is None else sel.step
    if step <= 0:
        # Negative steps are valid in TensorStore but not yet supported here;
        # step 0 is invalid everywhere.
        raise IndexError("slice step must be positive")
    start = lo if sel.start is None else sel.start
    stop = hi if sel.stop is None else sel.stop
    if stop < start:
        raise IndexError(
            f"slice interval [{start}, {stop}) with step {step} does not specify "
            f"a valid interval for dimension {dim} (start > stop)"
        )
    size = -(-(stop - start) // step)  # ceil((stop - start) / step)
    if size > 0 and (start < lo or stop > hi):
        hint = _LITERAL_HINT if (start < 0 or stop < 0) and lo >= 0 else ""
        raise BoundsCheckError(
            f"slice interval [{start}, {stop}) is not contained within domain "
            f"[{lo}, {hi}) for dimension {dim}{hint}"
        )
    origin = _trunc_div(start, step)
    return start, step, origin, size


def _check_array_in_bounds(arr: np.ndarray[Any, np.dtype[np.intp]], lo: int, hi: int) -> None:
    """Reject index-array values outside the domain ``[lo, hi)``.

    Index-array values are literal domain coordinates (TensorStore semantics):
    a value below ``inclusive_min`` is out of bounds rather than counting from
    the end. Out-of-range values raise instead of silently wrapping.
    """
    if arr.size == 0:
        return
    lo_val, hi_val = int(arr.min()), int(arr.max())
    if lo_val < lo:
        hint = _LITERAL_HINT if lo_val < 0 and lo >= 0 else ""
        raise BoundsCheckError(
            f"index {lo_val} is out of bounds (valid indices [{lo}, {hi})){hint}"
        )
    if hi_val >= hi:
        raise BoundsCheckError(f"index {hi_val} is out of bounds (valid indices [{lo}, {hi}))")


def _validate_array_selection(selection: Any, shape: tuple[int, ...], mode: str) -> None:
    """Validate array-based selections (orthogonal, vectorized).

    Rejects types that are not valid for coordinate/vectorized indexing.
    Does not check bounds — the transform operations handle that.
    """
    items = selection if isinstance(selection, tuple) else (selection,)
    for sel in items:
        if isinstance(sel, slice):
            # vindex is coordinate-only (matches eager zarr): every axis needs an
            # integer/boolean array, never a slice. Orthogonal (oindex) allows slices.
            if mode == "vectorized":
                raise VindexInvalidSelectionError(
                    "unsupported selection type for vectorized indexing; only "
                    "coordinate selection (tuple of integer arrays) and mask selection "
                    f"(single Boolean array) are supported; got {selection!r}"
                )
            continue
        if sel is Ellipsis or isinstance(sel, (int, np.integer)):
            continue
        if isinstance(sel, (list, np.ndarray)):
            continue
        raise IndexError(f"unsupported selection type for {mode} indexing: {type(sel)!r}")


def _validate_basic_selection(selection: Any) -> None:
    """Validate that a selection only contains basic indexing types (int, slice, Ellipsis).

    Rejects None (newaxis), arrays, lists, floats, strings, etc.
    """
    items = selection if isinstance(selection, tuple) else (selection,)
    for s in items:
        if s is Ellipsis or isinstance(s, (int, np.integer, slice)):
            continue
        raise IndexError(f"unsupported selection type for basic indexing: {type(s)!r}")


def selection_to_transform(
    selection: Any,
    transform: IndexTransform,
    mode: Literal["basic", "orthogonal", "vectorized"],
) -> IndexTransform:
    """Convert a user selection into a composed IndexTransform.

    Negative indices are treated as literal coordinates (TensorStore convention).
    The caller (Array layer) is responsible for converting numpy-style negative
    indices before calling this function.
    """
    if mode == "basic":
        _validate_basic_selection(selection)
        return transform[selection]
    elif mode == "orthogonal":
        _validate_array_selection(selection, transform.domain.shape, mode)
        return transform.oindex[selection]
    elif mode == "vectorized":
        _validate_array_selection(selection, transform.domain.shape, mode)
        return transform.vindex[selection]
    else:
        raise ValueError(f"Unknown mode: {mode!r}")
