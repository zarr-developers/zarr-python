"""The positional (NumPy) selection dialect, lowered onto the transform algebra.

The transform algebra uses **literal domain coordinates**: an index is a point
in the view's own coordinate system, which after `view = arr[10:50]` runs from
10 to 49, and a negative index is a negative coordinate rather than an offset
from the end (TensorStore's convention — see `zarr_indexing.transform`).

NumPy uses **positions**: index 0 always means the first element of the object
being indexed, and `-1` means the last. This module translates between the two.
It validates a selection against the view's shape with NumPy semantics, then
shifts every coordinate by the domain's origin so the transform layer sees
literal coordinates.

Note
----
`zarr.Array` currently carries its own copy of this normalization, tuned to a
different boundary contract (`Array.lazy[...]` deliberately exposes the literal
dialect, so a view's coordinates keep their meaning across composition). This
module is the generic, zarr-free version used by `LazyArray`; consolidating
zarr's copy onto it is left to a follow-up.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from zarr_indexing.domain import IndexDomain

SelectionMode = Literal["basic", "orthogonal", "vectorized"]


def _is_bool_scalar(sel: Any) -> bool:
    """Return True for a Python or NumPy boolean scalar.

    `isinstance(True, int)` holds, so booleans have to be rejected before any
    integer handling — `arr[True]` would otherwise silently mean `arr[1]`.
    """
    return isinstance(sel, (bool, np.bool_))


def _as_index_array(sel: Any) -> np.ndarray[Any, np.dtype[Any]] | None:
    """Return `sel` as an ndarray if it is array-like, else None."""
    if isinstance(sel, np.ndarray):
        return sel
    if isinstance(sel, (list, tuple)):
        arr = np.asarray(sel)
        if arr.dtype.kind in "biu":
            return arr
        raise IndexError(
            f"arrays used as indices must be of integer or boolean type; got dtype {arr.dtype}"
        )
    return None


def _axes_consumed(sel: Any, mode: SelectionMode) -> int:
    """How many axes of the view a single selection entry consumes."""
    if sel is None:
        return 0
    arr = _as_index_array(sel)
    # A multidimensional boolean mask consumes one axis per mask dimension.
    # Orthogonal indexing is per-axis by construction, so a mask there is 1-D
    # and consumes exactly one axis.
    if mode == "vectorized" and arr is not None and arr.dtype == np.bool_:
        return arr.ndim
    return 1


def _normalize_int(value: int, size: int, axis: int) -> int:
    """Bounds-check a positional integer index, wrapping negatives NumPy-style."""
    idx = value
    if idx < 0:
        idx += size
    if idx < 0 or idx >= size:
        raise IndexError(f"index {value} is out of bounds for axis {axis} with size {size}")
    return idx


def _normalize_slice(sel: slice, size: int, axis: int) -> tuple[int, int, int]:
    """Resolve a positional slice to `(start, stop, step)`, either direction.

    `slice.indices` already applies NumPy's rules — negative bounds count from
    the end, out-of-range bounds clamp, and a reversed slice runs downward with
    `stop` one *below* the last selected position. The one thing it does not do
    is canonicalize an empty result: it can hand back a stop on the far side of
    the start (`5:2` going up, `2:5` going down), which the transform layer
    reads as a direction error rather than an empty selection. Collapsing it to
    `stop == start` keeps NumPy's "empty, not an error" answer.
    """
    if sel.step is not None and not isinstance(sel.step, (int, np.integer)):
        raise IndexError(f"slice step must be an integer; got {sel.step!r}")
    step = 1 if sel.step is None else int(sel.step)
    if step == 0:
        raise ValueError(f"slice step cannot be zero (axis {axis})")  # ValueError: NumPy parity
    start, stop, step = sel.indices(size)
    stop = max(stop, start) if step > 0 else min(stop, start)
    return start, stop, step


def _normalize_int_array(
    arr: np.ndarray[Any, np.dtype[Any]], size: int, axis: int
) -> np.ndarray[Any, np.dtype[np.intp]]:
    """Bounds-check a positional integer index array, wrapping negatives."""
    if arr.dtype.kind not in "iu":
        raise IndexError(
            f"arrays used as indices must be of integer or boolean type; got dtype {arr.dtype}"
        )
    # Cast before wrapping: an unsigned array cannot represent the intermediate
    # negative values, and `intp` covers every index NumPy can address.
    out = arr.astype(np.intp, copy=True)
    if out.size > 0:
        negative = out < 0
        if bool(negative.any()):
            out = np.where(negative, out + size, out)
        lo, hi = int(out.min()), int(out.max())
        if lo < 0 or hi >= size:
            bad = lo if lo < 0 else hi
            raise IndexError(f"index {bad} is out of bounds for axis {axis} with size {size}")
    return out


def _expanded_axis_walk(entries: tuple[Any, ...], ndim: int, mode: SelectionMode) -> list[int]:
    """The starting axis each entry addresses, with an ellipsis expanded.

    The returned list has one entry per element of `entries`; the value for an
    `Ellipsis` (or a `newaxis`) is the axis it starts at, which is also the axis
    the following entry resumes from once the skipped axes are accounted for.
    """
    for sel in entries:
        if _is_bool_scalar(sel):
            raise IndexError(
                "boolean scalars are not valid indices; use a boolean array "
                "matching the shape of the axes it selects"
            )
    if sum(1 for sel in entries if sel is Ellipsis) > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    consumed = sum(_axes_consumed(sel, mode) for sel in entries if sel is not Ellipsis)
    if consumed > ndim:
        raise IndexError(
            f"too many indices for array: array has {ndim} dimensions, but {consumed} were indexed"
        )

    axes: list[int] = []
    axis = 0
    for sel in entries:
        axes.append(axis)
        axis += (ndim - consumed) if sel is Ellipsis else _axes_consumed(sel, mode)
    return axes


def split_scalar_axes(
    selection: Any,
    domain: IndexDomain,
    mode: SelectionMode,
) -> tuple[tuple[Any, ...] | None, Any]:
    """Peel scalar integer indices out of a fancy selection.

    In NumPy, a scalar integer is a basic index wherever it appears: it drops
    its axis, and is applied before the advanced indices rather than broadcast
    against them. `a[0, [1, 2], :]` is `a[0][[1, 2], :]`. Neither the orthogonal
    nor the vectorized path of the transform algebra models that — both widen a
    scalar into a length-1 index array, which keeps the axis — so the scalars
    are split off here and applied as a separate basic step first.

    Parameters
    ----------
    selection
        A positional orthogonal or vectorized selection.
    domain
        The domain of the view being indexed.
    mode
        `"orthogonal"` or `"vectorized"`; controls how many axes each entry
        covers, which decides where the scalars sit.

    Returns
    -------
    tuple
        `(basic_selection, remaining_selection)`. `basic_selection` is a
        full-rank basic selection in **literal** domain coordinates that drops
        the scalar axes, or `None` when the selection has no scalar entries (in
        which case `remaining_selection` is `selection` unchanged).

    Raises
    ------
    IndexError
        If a boolean scalar is used as an index, an index is out of bounds, or
        too many indices are supplied.
    """
    entries = selection if isinstance(selection, tuple) else (selection,)
    axes = _expanded_axis_walk(entries, domain.ndim, mode)

    scalar_axes: dict[int, int] = {}
    remaining: list[Any] = []
    for sel, axis in zip(entries, axes, strict=True):
        if isinstance(sel, (int, np.integer)) and not _is_bool_scalar(sel):
            scalar_axes[axis] = _normalize_int(int(sel), domain.shape[axis], axis)
        else:
            remaining.append(sel)

    if len(scalar_axes) == 0:
        return None, selection

    basic: list[Any] = []
    for axis in range(domain.ndim):
        lo = domain.inclusive_min[axis]
        if axis in scalar_axes:
            basic.append(lo + scalar_axes[axis])
        else:
            basic.append(slice(lo, domain.exclusive_max[axis]))
    return tuple(basic), tuple(remaining)


def normalize_positional_selection(
    selection: Any,
    domain: IndexDomain,
    mode: SelectionMode,
) -> Any:
    """Translate a positional (NumPy-dialect) selection into literal coordinates.

    Positions are zero-based offsets into the current view; negatives wrap
    from the end. The returned selection addresses the same cells in the
    literal coordinate system `domain` uses, ready for
    `zarr_indexing.transform.selection_to_transform`.

    Parameters
    ----------
    selection
        A NumPy-style selection: integers, slices, `Ellipsis`, integer arrays or
        lists, or boolean arrays.
    domain
        The domain of the view being indexed. Its shape defines the positional
        bounds and its origin the coordinate shift.
    mode
        Which selection dialect the entries follow: `"basic"` (integers and
        slices), `"orthogonal"` (per-axis arrays, outer product), or
        `"vectorized"` (correlated coordinate arrays or a single mask).

    Returns
    -------
    tuple
        The selection with every coordinate expressed in literal domain
        coordinates.

    Raises
    ------
    IndexError
        If a boolean scalar is used as an index, a boolean mask does not match
        the shape of the axes it covers, an index is out of bounds, or too many
        indices are supplied.
    """
    entries = selection if isinstance(selection, tuple) else (selection,)
    shape = domain.shape
    origin = domain.inclusive_min
    ndim = domain.ndim

    for sel in entries:
        if _is_bool_scalar(sel):
            raise IndexError(
                "boolean scalars are not valid indices; use a boolean array "
                "matching the shape of the axes it selects"
            )

    n_ellipsis = sum(1 for sel in entries if sel is Ellipsis)
    if n_ellipsis > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    consumed = sum(_axes_consumed(sel, mode) for sel in entries if sel is not Ellipsis)
    if consumed > ndim:
        raise IndexError(
            f"too many indices for array: array has {ndim} dimensions, but {consumed} were indexed"
        )

    result: list[Any] = []
    axis = 0
    for sel in entries:
        if sel is Ellipsis:
            # Passed through rather than expanded: every mode of
            # `selection_to_transform` expands an ellipsis (and pads short
            # selections) to whole-axis slices itself, and `vectorized` mode
            # rejects an explicit slice, so expanding here would turn a legal
            # partial coordinate selection into an error.
            result.append(Ellipsis)
            axis += ndim - consumed
            continue
        if sel is None:
            # newaxis: no axis of the view is consumed, and there is no
            # coordinate to shift. The transform layer decides whether the mode
            # accepts it.
            result.append(None)
            continue

        arr = _as_index_array(sel)
        if arr is not None and arr.dtype == np.bool_:
            n_axes = _axes_consumed(sel, mode)
            expected = shape[axis : axis + n_axes]
            if arr.shape != tuple(expected):
                raise IndexError(
                    f"boolean index has shape {arr.shape} but the axes it "
                    f"covers have shape {tuple(expected)}"
                )
            for offset, positions in enumerate(np.nonzero(arr)):
                result.append(positions.astype(np.intp) + origin[axis + offset])
            axis += n_axes
            continue

        if axis >= ndim:
            raise IndexError(
                f"too many indices for array: array has {ndim} dimensions, "
                f"but {consumed} were indexed"
            )
        size = shape[axis]
        if arr is not None:
            result.append(_normalize_int_array(arr, size, axis) + origin[axis])
        elif isinstance(sel, slice):
            start, stop, step = _normalize_slice(sel, size, axis)
            result.append(slice(start + origin[axis], stop + origin[axis], step))
        elif isinstance(sel, (int, np.integer)):
            result.append(_normalize_int(int(sel), size, axis) + origin[axis])
        else:
            raise IndexError(f"unsupported selection type: {type(sel)!r}")
        axis += 1

    # Axes the selection did not mention are left to `selection_to_transform`,
    # which pads them with whole-axis slices in every mode.
    return tuple(result)
