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

from zarr_metadata.testing._strategy import MAX_RANK, ChunkGridStrategy
from zarr_metadata.v3.chunk_grid.rectilinear import (
    RECTILINEAR_CHUNK_GRID,
    RectilinearChunkGridConfiguration,
    RectilinearDimSpec,
)
from zarr_metadata.v3.chunk_grid.regular import REGULAR_CHUNK_GRID, RegularChunkGridConfiguration

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr_metadata.testing._strategy import Strategies
    from zarr_metadata.v3._definition import ChunkGridDefinition

_MAX_LENGTH: Final = 16
"""The longest chunk drawn along a dimension whose length is unknown, before any factor its edges share."""

_MAX_CHUNK_LENGTH: Final = 2**63 - 1
"""The longest chunk drawn at all: what a signed 64-bit size holds, as readers index with."""

_MAX_RUNS: Final = 5
"""The most runs of equal lengths a rectilinear dimension is drawn with."""

_FEW_CHUNKS: Final = 4
"""The most chunks a short run is drawn with."""

_MAX_WRITTEN_OUT: Final = 64
"""The most chunks a run is written out with, one length for each; a longer run is always a pair."""

_MAX_FACTOR: Final = 16
"""The largest factor all of a dimension's edges are drawn to share."""

_MANY_CHUNKS: Final = 2**20
"""The most chunks a run is drawn with along a dimension whose length is unknown or 0, into which it grows when the array is resized."""


def _length(extent: int | None) -> st.SearchStrategy[int]:
    """A chunk length along a dimension of `extent`: at least 1, and up to twice the extent when it is known and not 0, within a signed 64-bit size."""
    if extent is None or extent == 0:
        return st.integers(1, _MAX_LENGTH)
    return st.integers(1, min(2 * extent, _MAX_CHUNK_LENGTH))


def _regular(
    shape: tuple[int, ...] | None, _strategies: Strategies
) -> st.SearchStrategy[RegularChunkGridConfiguration]:
    """A chunk length for each dimension."""
    if shape is None:
        lengths = st.lists(_length(None), max_size=MAX_RANK).map(tuple)
    else:
        lengths = st.tuples(*(_length(extent) for extent in shape))
    return lengths.map(lambda chunk_shape: RegularChunkGridConfiguration(chunk_shape=chunk_shape))


def _count(extent: int | None, size: int) -> st.SearchStrategy[int]:
    """How many chunks of `size` a run holds: a few, or as many as it takes to cover `extent`."""
    many = _MANY_CHUNKS if extent is None or extent == 0 else max(extent // size, _FEW_CHUNKS)
    return st.integers(1, _FEW_CHUNKS) | st.integers(1, many)


def _spelled_run(draw: st.DrawFn, size: int, count: int) -> list[int | tuple[int, int]]:
    """`count` chunks of `size`: one `[size, count]` pair, or, for a run short enough, the size written once for each."""
    if count > _MAX_WRITTEN_OUT or draw(st.booleans()):
        return [(size, count)]
    return [size] * count


@st.composite
def _edges(draw: st.DrawFn, extent: int | None) -> tuple[int | tuple[int, int], ...]:
    """Edge lengths along a dimension, covering `extent`, in runs spelled as pairs or written out.

    Runs are drawn apart, so two of one size can sit side by side and a
    pair can count one chunk: spellings the canonical form merges. A run
    can be long, since a pair costs the same however many chunks it
    counts, which is what the spelling is for. Its edges often share a
    factor, as they must for a shard to divide them into inner chunks
    longer than 1; edges drawn apart rarely do. A
    dimension of length zero, or of a length not known, may be covered by
    no edges at all.
    """
    min_runs = 0 if extent is None or extent == 0 else 1
    factor = draw(st.just(1) | st.integers(2, _MAX_FACTOR))
    sizes = [
        factor * size
        for size in draw(
            st.lists(st.integers(1, _MAX_LENGTH), min_size=min_runs, max_size=_MAX_RUNS)
        )
    ]
    runs = [(size, draw(_count(extent, size))) for size in sizes]
    covered = sum(size * count for size, count in runs)
    if extent is not None and covered < extent:
        # Rounded up to the factor, so the last edge shares it, and past the dimension if need be.
        runs.append((-(-(extent - covered) // factor) * factor, 1))
    spelled: list[int | tuple[int, int]] = []
    for size, count in runs:
        spelled.extend(_spelled_run(draw, size, count))
    return tuple(spelled)


def _rectilinear_dimension(extent: int | None) -> st.SearchStrategy[RectilinearDimSpec]:
    """A bare length, which repeats until it covers the dimension, or edges that cover it."""
    return _length(extent) | _edges(extent)


def _rectilinear(
    shape: tuple[int, ...] | None, _strategies: Strategies
) -> st.SearchStrategy[RectilinearChunkGridConfiguration]:
    """Inline chunk shapes, one entry for each dimension."""
    if shape is None:
        dimensions = st.lists(_rectilinear_dimension(None), max_size=MAX_RANK).map(tuple)
    else:
        dimensions = st.tuples(*(_rectilinear_dimension(extent) for extent in shape))
    return dimensions.map(
        lambda chunk_shapes: RectilinearChunkGridConfiguration(
            kind="inline", chunk_shapes=chunk_shapes
        )
    )


CHUNK_GRID_STRATEGIES: Final[Mapping[ChunkGridDefinition[Any], ChunkGridStrategy]] = {
    REGULAR_CHUNK_GRID: ChunkGridStrategy(configurations=_regular),
    RECTILINEAR_CHUNK_GRID: ChunkGridStrategy(configurations=_rectilinear),
}
"""A strategy for each chunk grid."""

__all__ = ["CHUNK_GRID_STRATEGIES"]
