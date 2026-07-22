"""Reduce public selection kinds to `(Region, post_index)` pairs.

The engine boundary only speaks contiguous step-1 boxes (`Region`). Each
helper returns the box to transfer plus the numpy index that, applied to the
ndim-preserving box result, yields exactly `array[original_selection]`.
"""

from __future__ import annotations

import operator
import types
from typing import TYPE_CHECKING, Any

import numpy as np

from zarr.abc.engine import Region

if TYPE_CHECKING:
    from zarr.core.indexing import BasicSelection


def _expand(selection: Any, ndim: int) -> tuple[Any, ...]:
    """Expand Ellipsis and pad missing trailing axes with full slices."""
    sel_tuple = selection if isinstance(selection, tuple) else (selection,)
    n_ellipsis = sum(1 for s in sel_tuple if s is Ellipsis)
    if n_ellipsis > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    if n_ellipsis == 1:
        i = sel_tuple.index(Ellipsis)
        n_fill = ndim - (len(sel_tuple) - 1)
        if n_fill < 0:
            raise IndexError(f"too many indices for array: array is {ndim}-dimensional")
        sel_tuple = sel_tuple[:i] + (slice(None),) * n_fill + sel_tuple[i + 1 :]
    if len(sel_tuple) > ndim:
        raise IndexError(f"too many indices for array: array is {ndim}-dimensional")
    return sel_tuple + (slice(None),) * (ndim - len(sel_tuple))


def _normalize_int(sel: Any, size: int, dim: int) -> int:
    idx = operator.index(sel)
    if idx < 0:
        idx += size
    if not 0 <= idx < size:
        raise IndexError(f"index {sel} is out of bounds for axis {dim} with size {size}")
    return idx


def normalize_basic(
    selection: BasicSelection, shape: tuple[int, ...]
) -> tuple[Region, tuple[slice | int, ...]]:
    """Normalize a numpy basic-indexing selection to a step-1 box.

    Supports integers, slices (any step), and `Ellipsis`. Integer axes get a
    length-1 range in the box and `0` in the post index (dropping the axis,
    matching numpy). Fancy elements raise `TypeError`.
    """
    sel_tuple = _expand(selection, len(shape))
    starts: list[int] = []
    ends: list[int] = []
    post: list[slice | int] = []
    for dim, (sel, size) in enumerate(zip(sel_tuple, shape, strict=True)):
        if isinstance(sel, slice):
            start, stop, step = sel.indices(size)
            n = len(range(start, stop, step))
            if n == 0:
                starts.append(0)
                ends.append(0)
                post.append(slice(None))
            elif step > 0:
                starts.append(start)
                ends.append(start + (n - 1) * step + 1)
                post.append(slice(None, None, step))
            else:
                last = start + (n - 1) * step
                starts.append(last)
                ends.append(start + 1)
                post.append(slice(None, None, step))
        elif isinstance(sel, types.EllipsisType):
            raise AssertionError("Ellipsis expanded by _expand")  # noqa: TRY004
        else:
            try:
                idx = _normalize_int(sel, size, dim)
            except TypeError:
                raise TypeError(
                    f"unsupported selection element {sel!r}: only integers, "
                    "slices, and Ellipsis are supported by basic indexing"
                ) from None
            starts.append(idx)
            ends.append(idx + 1)
            post.append(0)
    return Region(start=tuple(starts), end_exclusive=tuple(ends)), tuple(post)


class _Squeeze:
    """Marker appended to an orthogonal post index: squeeze these axes."""

    def __init__(self, axes: tuple[int, ...]) -> None:
        self.axes = axes


def normalize_orthogonal(selection: Any, shape: tuple[int, ...]) -> tuple[Region, tuple[Any, ...]]:
    """Normalize an orthogonal (`oindex`) selection to a box + outer index.

    Each axis selector may be an integer, a slice, an integer array, or a 1-d
    boolean mask for that axis. The post index is the `np.ix_`-broadcastable
    tuple of per-axis integer arrays (relative to the box origin); if any axis
    was an integer, a trailing `_Squeeze` marker records which axes to drop
    afterwards. Use `apply_post_index` to apply the result correctly.
    """
    sel_tuple = _expand(selection, len(shape))
    starts: list[int] = []
    ends: list[int] = []
    axis_indices: list[np.ndarray] = []
    for dim, (sel, size) in enumerate(zip(sel_tuple, shape, strict=True)):
        if isinstance(sel, slice):
            idxs = np.arange(*sel.indices(size))
        elif isinstance(sel, np.ndarray) and sel.dtype == bool:
            if sel.ndim != 1 or sel.shape[0] != size:
                raise IndexError(f"boolean index for axis {dim} must be 1-d with length {size}")
            idxs = np.nonzero(sel)[0]
        elif isinstance(sel, (np.ndarray, list)):
            idxs = np.asarray(sel, dtype=np.intp)
            if idxs.ndim != 1:
                raise IndexError(f"orthogonal index for axis {dim} must be 1-d")
            idxs = np.where(idxs < 0, idxs + size, idxs)
            if idxs.size and (idxs.min() < 0 or idxs.max() >= size):
                raise IndexError(f"index out of bounds for axis {dim} with size {size}")
        else:
            idxs = np.array([_normalize_int(sel, size, dim)], dtype=np.intp)
        if idxs.size == 0:
            starts.append(0)
            ends.append(0)
            axis_indices.append(idxs)
        else:
            lo, hi = int(idxs.min()), int(idxs.max())
            starts.append(lo)
            ends.append(hi + 1)
            axis_indices.append(idxs - lo)
    region = Region(start=tuple(starts), end_exclusive=tuple(ends))
    post = np.ix_(*axis_indices) if axis_indices else ()
    # integer axes were widened to length-1 arrays; the caller squeezes them
    squeeze_axes = tuple(
        i for i, sel in enumerate(sel_tuple) if not isinstance(sel, (slice, np.ndarray, list))
    )
    if squeeze_axes:
        return region, (*post, _Squeeze(squeeze_axes))
    return region, post


def apply_post_index(box: np.ndarray, post: tuple[Any, ...]) -> np.ndarray:
    """Apply a post index produced by a `normalize_*` helper to a box read."""
    if post and isinstance(post[-1], _Squeeze):
        result = box[post[:-1]] if len(post) > 1 else box
        return np.squeeze(result, axis=post[-1].axes)
    if post == ():
        return box
    return np.asarray(box[post])


def strip_squeeze(post: tuple[Any, ...]) -> tuple[Any, ...]:
    """Return `post` without a trailing `_Squeeze` marker, if present.

    Writes apply the post index directly to a target array (not through
    `apply_post_index`), so callers that only need the "real" numpy index
    strip the `_Squeeze` marker first. Returns `post` unchanged when there is
    no marker.
    """
    if post and isinstance(post[-1], _Squeeze):
        return post[:-1]
    return post


def normalize_coordinate(
    selection: tuple[Any, ...], shape: tuple[int, ...]
) -> tuple[Region, tuple[np.ndarray, ...]]:
    """Normalize a coordinate (`vindex`) selection to a box + pointwise index."""
    coords = tuple(np.asarray(c, dtype=np.intp) for c in selection)
    if len(coords) != len(shape):
        raise IndexError(f"coordinate selection needs {len(shape)} axis arrays, got {len(coords)}")
    coords = tuple(np.where(c < 0, c + size, c) for c, size in zip(coords, shape, strict=True))
    for dim, (c, size) in enumerate(zip(coords, shape, strict=True)):
        if c.size and (c.min() < 0 or c.max() >= size):
            raise IndexError(f"index out of bounds for axis {dim} with size {size}")
    starts = tuple(int(c.min()) if c.size else 0 for c in coords)
    ends = tuple(int(c.max()) + 1 if c.size else 0 for c in coords)
    post = tuple(c - s for c, s in zip(coords, starts, strict=True))
    return Region(start=starts, end_exclusive=ends), post


def normalize_block(
    block_coords: tuple[int, ...],
    *,
    chunk_grid_shape: tuple[int, ...],
    chunk_shape: tuple[int, ...],
    shape: tuple[int, ...],
) -> Region:
    """Normalize a block (chunk-grid) selection to its contiguous box."""
    starts = []
    ends = []
    for dim, (b, nblocks, csize, size) in enumerate(
        zip(block_coords, chunk_grid_shape, chunk_shape, shape, strict=True)
    ):
        idx = _normalize_int(b, nblocks, dim)
        starts.append(idx * csize)
        ends.append(min((idx + 1) * csize, size))
    return Region(start=tuple(starts), end_exclusive=tuple(ends))
