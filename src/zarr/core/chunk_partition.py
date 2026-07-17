"""View-aware chunk partitioning (Layer B of the lazy-view chunk-layout design).

A lazy view is an ``IndexTransform`` applied to a backing array — i.e. a stored
selection already resolved. ``chunk_projections`` enumerates the stored chunks that
selection (or a whole identity array) projects onto, reusing the same resolution
machinery the read/write I/O path uses (``iter_chunk_transforms`` +
``sub_transform_to_selections``). Each projection reports the stored chunk, the
region of it this array covers, the region of this array it maps to, and whether the
coverage is partial (a partial write is a read-modify-write).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from zarr.core.transforms.chunk_resolution import (
    iter_chunk_transforms,
    sub_transform_to_selections,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from zarr.core.chunk_grids import ChunkGrid
    from zarr.core.transforms.transform import IndexTransform

__all__ = ["ChunkProjection", "iter_chunk_projections"]


@dataclass(frozen=True)
class ChunkProjection:
    """How one stored chunk contributes to an array (or lazy view).

    Attributes
    ----------
    coord
        Coordinate of the stored chunk in the chunk grid.
    key
        Store key of the stored chunk (which file).
    shape
        Stored size of the chunk, clipped to the array extent.
    chunk_selection
        The region *within* the stored chunk that this array covers.
    array_selection
        The region of this array (or view) that the chunk maps to.
    is_partial
        Whether the chunk is only partially covered — a write to a partial chunk
        is a read-modify-write.
    """

    coord: tuple[int, ...]
    key: str
    shape: tuple[int, ...]
    chunk_selection: tuple[Any, ...]
    array_selection: tuple[Any, ...]
    is_partial: bool


def _covers_full_chunk(chunk_selection: tuple[Any, ...], shape: tuple[int, ...]) -> bool:
    """Whether ``chunk_selection`` selects every element of a chunk of ``shape``.

    A per-dimension full `0:size:1` slice is a full cover. An integer entry is
    also a full cover *iff* the chunk's extent along that dimension is 1 — fixing
    the only element of a length-1 dimension covers it exactly, the same as an
    equivalent length-1 slice would. This mirrors the write path's
    `_is_complete_chunk` (`ConstantMap` over a dimension of size 1 is complete);
    keeping the two in sync matters because a divergence would make
    `chunk_projections` plan needless read-modify-writes for coverage the write
    path already treats as complete. An array selection always touches a subset
    (or requires per-element comparison this cheap check does not attempt), so it
    is never a full cover.
    """
    if len(chunk_selection) != len(shape):
        return False
    for sel, size in zip(chunk_selection, shape, strict=True):
        if isinstance(sel, slice):
            start, stop, step = sel.indices(size)
            if not (start == 0 and stop == size and step == 1):
                return False
        elif isinstance(sel, int):
            if size > 1:
                return False
        else:
            return False
    return True


def iter_chunk_projections(
    transform: IndexTransform,
    chunk_grid: ChunkGrid,
    encode_key: Callable[[tuple[int, ...]], str],
) -> Iterator[ChunkProjection]:
    """Yield a `ChunkProjection` for each stored chunk ``transform`` projects onto.

    `array_selection` is positional (0-based into the view's extent), so the
    transform is normalized to a zero-origin domain first — translation
    preserves which cells are addressed.
    """
    transform = transform.translate_domain_to((0,) * transform.input_rank)
    chunk_sizes = chunk_grid.chunk_sizes  # per-dimension, extent-clipped
    for chunk_coords, sub_transform, out_indices in iter_chunk_transforms(transform, chunk_grid):
        if chunk_grid[chunk_coords] is None:
            raise IndexError(f"Chunk coordinates {chunk_coords} are out of bounds.")
        chunk_selection, array_selection, _drop_axes = sub_transform_to_selections(
            sub_transform, out_indices
        )
        shape = tuple(chunk_sizes[dim][c] for dim, c in enumerate(chunk_coords))
        yield ChunkProjection(
            coord=chunk_coords,
            key=encode_key(chunk_coords),
            shape=shape,
            chunk_selection=chunk_selection,
            array_selection=array_selection,
            is_partial=not _covers_full_chunk(chunk_selection, shape),
        )
