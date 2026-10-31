"""Strategies for each chunk grid this package defines.

Over an array of a known shape, a grid is drawn to fit it: a chunk
length for each dimension, never zero ("Chunk sizes must be greater
than zero", https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/chunk-grids/regular-grid/index.rst#L40),
and rectilinear edges that cover each dimension. Over no shape, what is
drawn is what the grid's rules accept.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from hypothesis import strategies as st

from zarr_metadata.testing._strategy import ChunkGridStrategy
from zarr_metadata.v3.chunk_grid.rectilinear import RECTILINEAR_CHUNK_GRID
from zarr_metadata.v3.chunk_grid.regular import REGULAR_CHUNK_GRID

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr_metadata._common import JSONValue
    from zarr_metadata.v3._definition import ChunkGridDefinition

_MAX_RANK: Final = 5
"""The most dimensions a grid is drawn with, over no shape."""

_MAX_LENGTH: Final = 16
"""The longest chunk drawn along a dimension whose length is unknown."""

_MAX_RUNS: Final = 5
"""The most runs of equal lengths a rectilinear dimension is drawn with."""

_MAX_COUNT: Final = 4
"""The most chunks a run is drawn with."""


def _length(extent: int | None) -> st.SearchStrategy[int]:
    """A chunk length along a dimension of `extent`: at least 1, and up to twice the extent when it is known and not 0."""
    if extent is None or extent == 0:
        return st.integers(1, _MAX_LENGTH)
    return st.integers(1, 2 * extent)


def _regular(shape: tuple[int, ...] | None) -> st.SearchStrategy[Mapping[str, JSONValue]]:
    """A chunk length for each dimension."""
    if shape is None:
        lengths = st.lists(_length(None), max_size=_MAX_RANK).map(tuple)
    else:
        lengths = st.tuples(*(_length(extent) for extent in shape))
    return lengths.map(lambda chunk_shape: {"chunk_shape": chunk_shape})


def _spelled_run(draw: st.DrawFn, size: int, count: int) -> list[JSONValue]:
    """`count` chunks of `size`: one `[size, count]` pair, or the size written once for each."""
    if draw(st.booleans()):
        return [(size, count)]
    return [size] * count


@st.composite
def _edges(draw: st.DrawFn, extent: int | None) -> JSONValue:
    """Edge lengths along a dimension, covering `extent`, in runs spelled as pairs or written out.

    Runs are drawn apart, so two of one size can sit side by side and a
    pair can count one chunk: spellings the canonical form merges. A
    dimension of length zero, or of a length not known, may be covered by
    no edges at all.
    """
    min_runs = 0 if extent is None or extent == 0 else 1
    runs = draw(
        st.lists(
            st.tuples(st.integers(1, _MAX_LENGTH), st.integers(1, _MAX_COUNT)),
            min_size=min_runs,
            max_size=_MAX_RUNS,
        )
    )
    covered = sum(size * count for size, count in runs)
    if extent is not None and covered < extent:
        runs.append((extent - covered, 1))
    spelled: list[JSONValue] = []
    for size, count in runs:
        spelled.extend(_spelled_run(draw, size, count))
    return tuple(spelled)


def _rectilinear_dimension(extent: int | None) -> st.SearchStrategy[JSONValue]:
    """A bare length, which repeats until it covers the dimension, or edges that cover it."""
    return _length(extent) | _edges(extent)


def _rectilinear(shape: tuple[int, ...] | None) -> st.SearchStrategy[Mapping[str, JSONValue]]:
    """Inline chunk shapes, one entry for each dimension."""
    if shape is None:
        dimensions = st.lists(_rectilinear_dimension(None), max_size=_MAX_RANK).map(tuple)
    else:
        dimensions = st.tuples(*(_rectilinear_dimension(extent) for extent in shape))
    return dimensions.map(lambda chunk_shapes: {"kind": "inline", "chunk_shapes": chunk_shapes})


CHUNK_GRID_STRATEGIES: Final[Mapping[ChunkGridDefinition[Any], ChunkGridStrategy]] = {
    REGULAR_CHUNK_GRID: ChunkGridStrategy(configurations=_regular),
    RECTILINEAR_CHUNK_GRID: ChunkGridStrategy(configurations=_rectilinear),
}
"""A strategy for each chunk grid."""

__all__ = ["CHUNK_GRID_STRATEGIES"]
