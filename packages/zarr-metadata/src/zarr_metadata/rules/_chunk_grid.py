"""What array a chunk grid governs.

A codec pipeline encodes one chunk, so every rule about a pipeline —
transpose ranks, shard divisibility — needs the shape of that chunk. Three
different pieces of metadata answer that question, and this module is the
one place that reads them:

- a document's `chunk_grid`, which partitions an array of `shape`;
- a `sharding_indexed` codec's `chunk_shape`, which is a regular grid over
  the chunk the codec receives;
- that same codec's shard index, whose shape the spec derives from the
  first two.

Derived per dimension, not per grid
-----------------------------------
The answer is a `GovernedShape`: one entry per dimension, each the extent
every chunk has along it, or `None` where the chunks differ or the metadata
cannot be read. Only the whole result is `None`, and only when not even the
rank is known.

That granularity is the point. A rectilinear grid has no single chunk
shape, but a *dimension* of one may still be uniform — `[[32, 32], [10, 22]]`
pins the first axis at 32 and says nothing about the second — and a shard
inside it can be judged on the axis that is pinned. A grid this package
cannot read at all still pins the rank, because every chunk of an array has
the array's rank whatever partitions it. Collapsing any of this to "shape
unknown" silently retires rules that had enough information to run.

Prior art
---------
zarrs models the same thing and is worth following: its `ChunkGrid` is
built from metadata *and* the array shape (`ChunkGrid::create(metadata,
array_shape)`) because neither alone determines a grid; its
`dimensionality()` is total rather than optional; and it reports
`chunk_edge_lengths(dimension)` per dimension for exactly the reason above.
Its codec chain carries a three-state `ChunkGridMapped` — `Array`,
`ChunkLocal` ("no global grid, but one per chunk"), `None` — keeping
"varies" distinct from "unknown"; a `GovernedShape` entry of `None` is the
per-dimension form of that distinction.

- https://github.com/zarrs/zarrs/blob/main/zarrs_chunk_grid/src/lib.rs
- https://github.com/zarrs/zarrs/blob/main/zarrs_codec/src/lib.rs
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias, cast

from zarr_metadata.rules._entity import entity_configuration
from zarr_metadata.v3._extension_points import CHUNK_GRID
from zarr_metadata.v3._shape import entity_name
from zarr_metadata.v3.chunk_grid.rectilinear import RECTILINEAR_CHUNK_GRID_NAME
from zarr_metadata.v3.chunk_grid.regular import REGULAR_CHUNK_GRID_NAME

if TYPE_CHECKING:
    from collections.abc import Sequence

GovernedShape: TypeAlias = "tuple[int | None, ...]"
"""One entry per dimension: the extent every chunk shares, or `None`.

`None` in an entry means the chunks differ along that axis, or the value
is unusable. A `GovernedShape` always knows its rank; the absence of a
shape entirely is spelled `None` in place of the whole tuple.
"""


def _positive_int(value: object) -> int | None:
    """`value` as a chunk extent, or None if it is not a usable one."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 1 else None


def rank_of(array_shape: object) -> int | None:
    """The number of dimensions `array_shape` declares, if it declares any."""
    if not isinstance(array_shape, tuple):
        return None
    dimensions = cast("tuple[object, ...]", array_shape)
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in dimensions):
        return None
    return len(dimensions)


def uniform_shape(extents: Sequence[object]) -> GovernedShape:
    """A regular grid's chunk shape: one declared extent per dimension.

    An unusable extent costs that dimension, never the rank — the grid's
    own values rule owns the complaint about the value itself.
    """
    return tuple(_positive_int(extent) for extent in extents)


def _rectilinear_extent(spec: object) -> int | None:
    """The extent every chunk shares along one rectilinear dimension.

    A bare integer is a regular step, so it is that extent. An explicit
    list is uniform only if every entry (run-length pairs expanded) names
    the same size. Anything else varies, or cannot be read.
    """
    bare = _positive_int(spec)
    if bare is not None:
        return bare
    if not isinstance(spec, tuple):
        return None
    sizes: set[int] = set()
    for item in cast("tuple[object, ...]", spec):
        size = _positive_int(item)
        if size is None and isinstance(item, tuple):
            pair = cast("tuple[object, ...]", item)
            size = _positive_int(pair[0]) if len(pair) == 2 else None
            if _positive_int(pair[1]) is None:
                return None
        if size is None:
            return None
        sizes.add(size)
    if len(sizes) != 1:
        return None
    return sizes.pop()


def governed_shape(grid: object, array_shape: object) -> GovernedShape | None:
    """The shape of one chunk of `grid`, dimension by dimension.

    `grid` is a document's `chunk_grid` metadata and `array_shape` its
    `shape`; a grid is not interpretable without the array it partitions,
    and the array shape is what pins the rank when the grid itself cannot
    be read. Answers None only when not even the rank is available.
    """
    fallback = rank_of(array_shape)
    name = entity_name(grid)
    configuration = entity_configuration(CHUNK_GRID, grid) if name is not None else None
    if configuration is not None:
        if name == REGULAR_CHUNK_GRID_NAME:
            extents = configuration.get("chunk_shape")
            if isinstance(extents, tuple):
                return uniform_shape(cast("tuple[object, ...]", extents))
        elif name == RECTILINEAR_CHUNK_GRID_NAME:
            dimensions = configuration.get("chunk_shapes")
            if isinstance(dimensions, tuple):
                return tuple(
                    _rectilinear_extent(spec) for spec in cast("tuple[object, ...]", dimensions)
                )
    return None if fallback is None else (None,) * fallback


def shard_index_shape(shard: GovernedShape | None, inner: Sequence[object]) -> GovernedShape:
    """The shape of a shard's index array.

    The spec derives it from the two shapes around it: "The index is an
    array with 64-bit unsigned integers with a shape that matches the
    chunks per shard tuple with an appended dimension of size 2." Chunks
    per shard needs both extents, so a dimension resolves only where the
    shard and inner extents are both known and divide evenly; the rank is
    one more than the inner chunk's, always.
    """
    inner_shape = uniform_shape(inner)
    # The trailing 2 is fixed by the spec, so it is known even when no
    # chunk count is.
    if shard is None or len(shard) != len(inner_shape):
        return (*(None,) * len(inner_shape), 2)
    counts = tuple(
        outer // extent
        if outer is not None and extent is not None and outer % extent == 0
        else None
        for outer, extent in zip(shard, inner_shape, strict=True)
    )
    return (*counts, 2)


__all__ = [
    "GovernedShape",
    "governed_shape",
    "rank_of",
    "shard_index_shape",
    "uniform_shape",
]
