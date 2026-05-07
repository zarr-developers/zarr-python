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
``Array.z[...]`` composes a new transform lazily. Reading resolves the
transform against the chunk grid via intersect + translate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from zarr.core.transforms.domain import IndexDomain
from zarr.core.transforms.output_map import ArrayMap, ConstantMap, DimensionMap, OutputIndexMap


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
                n = len(storage)
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
    ) -> tuple[IndexTransform, np.ndarray[Any, np.dtype[np.intp]] | None] | None:
        """Restrict this transform to storage coordinates within output_domain.

        Returns ``(restricted_transform, surviving_indices)`` or None if empty.

        ``surviving_indices`` is an integer array of which input positions
        survived the intersection (for ArrayMap dimensions), or None if all
        positions survived (ConstantMap/DimensionMap only).
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
                    )
                )
        return IndexTransform(domain=self.domain, output=tuple(new_output))

    def __getitem__(self, selection: Any) -> IndexTransform:
        return _apply_basic_indexing(self, selection)

    @property
    def oindex(self) -> _OIndexHelper:
        return _OIndexHelper(self)

    @property
    def vindex(self) -> _VIndexHelper:
        return _VIndexHelper(self)


def _intersect(
    transform: IndexTransform, output_domain: IndexDomain
) -> tuple[IndexTransform, np.ndarray[Any, np.dtype[np.intp]] | None] | None:
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

    # Check if we have correlated ArrayMaps (vectorized)
    array_dims = [i for i, m in enumerate(transform.output) if isinstance(m, ArrayMap)]
    if len(array_dims) >= 2:
        return _intersect_vectorized(transform, output_domain, array_dims)

    # Orthogonal: intersect each output dimension independently
    new_min = list(transform.domain.inclusive_min)
    new_max = list(transform.domain.exclusive_max)
    new_output: list[OutputIndexMap] = []
    surviving_indices: np.ndarray[Any, np.dtype[np.intp]] | None = None

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
            if not np.any(mask):
                return None
            surviving_indices = np.nonzero(mask.ravel())[0].astype(np.intp)
            filtered = m.index_array.ravel()[surviving_indices]
            new_output.append(
                ArrayMap(
                    index_array=filtered,
                    offset=m.offset,
                    stride=m.stride,
                )
            )

    new_domain = IndexDomain(
        inclusive_min=tuple(new_min),
        exclusive_max=tuple(new_max),
    )
    result = IndexTransform(domain=new_domain, output=tuple(new_output))
    return (result, surviving_indices)


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
    n_points: int | None = None
    masks: list[np.ndarray[Any, np.dtype[np.bool_]]] = []

    for out_dim in array_dims:
        m = transform.output[out_dim]
        assert isinstance(m, ArrayMap)
        storage = m.offset + m.stride * m.index_array
        lo = output_domain.inclusive_min[out_dim]
        hi = output_domain.exclusive_max[out_dim]
        masks.append((storage >= lo) & (storage < hi))
        if n_points is None:
            n_points = storage.size

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
                dim_size = domain.shape[old_dim]
                # sel.indices gives 0-based start/stop/step for the array axis
                start, stop, step = sel.indices(dim_size)
                idx.append(slice(start, stop, step))
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
        if isinstance(sel, np.ndarray):
            idx.append(sel)
        elif isinstance(sel, slice):
            dim_size = domain.shape[old_dim]
            start, stop, step = sel.indices(dim_size)
            idx.append(slice(start, stop, step))
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
                raise IndexError(
                    f"index {sel} is out of bounds for dimension {old_dim} with domain [{lo}, {hi})"
                )
            dropped_dims.add(old_dim)
            dim_int_val[old_dim] = idx
            old_dim += 1
        elif isinstance(sel, slice):
            lo = transform.domain.inclusive_min[old_dim]
            hi = transform.domain.exclusive_max[old_dim]
            dim_size = hi - lo

            # Resolve slice relative to the current domain (origin-based)
            start, stop, step = sel.indices(dim_size)
            # start, stop, step are now relative to a 0-based range of size dim_size

            if step <= 0:
                raise IndexError("slice step must be positive")

            new_size = max(0, math.ceil((stop - start) / step))
            new_inclusive_min.append(0)
            new_exclusive_max.append(new_size)

            # Absolute start in the original domain coordinates
            abs_start = lo + start
            dim_slice_params[old_dim] = (abs_start, stop, step)
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
                # Slice: update offset and stride
                abs_start, _, step = dim_slice_params[d]
                new_offset = m.offset + m.stride * abs_start
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
            new_output.append(ArrayMap(index_array=new_arr, offset=m.offset, stride=m.stride))

    return IndexTransform(domain=new_domain, output=tuple(new_output))


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
            dim_array[old_dim] = sel
            new_inclusive_min.append(0)
            new_exclusive_max.append(len(sel))
            old_to_new_dim[old_dim] = new_dim_idx
            new_dim_idx += 1
        elif isinstance(sel, slice):
            lo = transform.domain.inclusive_min[old_dim]
            hi = transform.domain.exclusive_max[old_dim]
            dim_size = hi - lo
            start, stop, step = sel.indices(dim_size)
            if step <= 0:
                raise IndexError("slice step must be positive")
            new_size = max(0, math.ceil((stop - start) / step))
            new_inclusive_min.append(0)
            new_exclusive_max.append(new_size)
            abs_start = lo + start
            dim_slice_params[old_dim] = (abs_start, stop, step)
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
                new_output.append(
                    ArrayMap(
                        index_array=dim_array[d],
                        offset=m.offset,
                        stride=m.stride,
                    )
                )
            elif d in dim_slice_params:
                abs_start, _, step = dim_slice_params[d]
                new_offset = m.offset + m.stride * abs_start
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
            new_output.append(ArrayMap(index_array=new_arr, offset=m.offset, stride=m.stride))

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

    # Slice dimensions
    slice_dim_params: dict[int, tuple[int, int, int]] = {}
    for old_dim in slice_dims:
        sel = processed[old_dim]
        assert isinstance(sel, slice)
        lo = transform.domain.inclusive_min[old_dim]
        hi = transform.domain.exclusive_max[old_dim]
        dim_size = hi - lo
        start, stop, step = sel.indices(dim_size)
        if step <= 0:
            raise IndexError("slice step must be positive")
        new_size = max(0, math.ceil((stop - start) / step))
        new_inclusive_min.append(0)
        new_exclusive_max.append(new_size)
        abs_start = lo + start
        slice_dim_params[old_dim] = (abs_start, stop, step)

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
                new_output.append(
                    ArrayMap(
                        index_array=array_dim_to_broadcast[d],
                        offset=m.offset,
                        stride=m.stride,
                    )
                )
            else:
                # Slice dim
                abs_start, _, step = slice_dim_params[d]
                new_offset = m.offset + m.stride * abs_start
                new_stride = m.stride * step
                new_input_dim = n_broadcast_dims + slice_dims.index(d)
                new_output.append(
                    DimensionMap(
                        input_dimension=new_input_dim, offset=new_offset, stride=new_stride
                    )
                )
        elif isinstance(m, ArrayMap):
            new_arr = _reindex_array_oindex(m.index_array, processed, transform.domain)
            new_output.append(ArrayMap(index_array=new_arr, offset=m.offset, stride=m.stride))

    return IndexTransform(domain=new_domain, output=tuple(new_output))


def _normalize_negative_indices(selection: Any, shape: tuple[int, ...]) -> Any:
    """Convert negative indices to positive ones using the array shape.

    Only normalizes integer and array-like index components; leaves
    slices, Ellipsis, None, etc. untouched.
    """
    if not isinstance(selection, tuple):
        selection_tuple: tuple[Any, ...] = (selection,)
    else:
        selection_tuple = selection

    # Count real dimensions (non-None, non-Ellipsis) to map each entry to a shape dim
    has_ellipsis = any(s is Ellipsis for s in selection_tuple)
    n_non_newaxis = sum(1 for s in selection_tuple if s is not None and s is not Ellipsis)
    n_ellipsis_dims = len(shape) - n_non_newaxis + (1 if has_ellipsis else 0)

    result: list[Any] = []
    dim = 0

    for sel in selection_tuple:
        if sel is Ellipsis:
            result.append(sel)
            dim += max(0, n_ellipsis_dims)
        elif sel is None:
            result.append(sel)
        elif isinstance(sel, (int, np.integer)) and not isinstance(sel, bool):
            idx = int(sel)
            if idx < 0 and dim < len(shape):
                idx = idx + shape[dim]
            result.append(idx)
            dim += 1
        elif isinstance(sel, np.ndarray) and sel.dtype != np.bool_:
            arr = sel.copy()
            if dim < len(shape):
                arr = np.where(arr < 0, arr + shape[dim], arr)
            result.append(arr)
            dim += 1
        elif isinstance(sel, list):
            # Convert lists to arrays with negative index normalization
            arr = np.asarray(sel, dtype=np.intp)
            if dim < len(shape):
                arr = np.where(arr < 0, arr + shape[dim], arr)
            result.append(arr)
            dim += 1
        else:
            # slice, bool array, or anything else: pass through
            result.append(sel)
            if sel is not None and sel is not Ellipsis:
                dim += 1

    if not isinstance(selection, tuple) and len(result) == 1:
        return result[0]
    return tuple(result)


def _validate_array_selection(selection: Any, shape: tuple[int, ...], mode: str) -> None:
    """Validate array-based selections (orthogonal, vectorized).

    Rejects types that are not valid for coordinate/vectorized indexing.
    Does not check bounds — the transform operations handle that.
    """
    items = selection if isinstance(selection, tuple) else (selection,)
    for sel in items:
        if sel is Ellipsis or isinstance(sel, (int, np.integer, slice)):
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
