"""
Rectilinear chunk grid (zarr-extensions).

See https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/chunk-grids/rectilinear/README.md
"""

from typing import Final, Literal, NotRequired

from typing_extensions import TypedDict

RECTILINEAR_CHUNK_GRID_NAME: Final = "rectilinear"
"""The `name` field value of the rectilinear chunk grid."""

RectilinearChunkGridName = Literal["rectilinear"]
"""Literal type of the `name` field of the rectilinear chunk grid."""

RectilinearDimSpec = int | tuple[int | tuple[int, int], ...]
"""JSON shape for one dimension's rectilinear spec.

Either a bare integer (uniform shorthand for a regular dimension within
a rectilinear grid), or a tuple of integers and/or `[value, count]` RLE
pairs.
"""


class RectilinearChunkGridConfiguration(TypedDict, closed=True):
    """Configuration for the rectilinear chunk grid."""

    kind: Literal["inline"]
    chunk_shapes: tuple[RectilinearDimSpec, ...]


class RectilinearChunkGridObject(TypedDict, closed=True):
    """Rectilinear chunk grid metadata in object form."""

    name: RectilinearChunkGridName
    configuration: RectilinearChunkGridConfiguration
    must_understand: NotRequired[bool]


RectilinearChunkGridMetadata = RectilinearChunkGridObject
"""Permitted JSON shape for rectilinear chunk grid metadata.

`kind` and `chunk_shapes` are required, so only the object form is valid;
the short-hand-name form is not permitted by the spec for this grid.
  https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/chunk-grids/rectilinear/README.md#L59-L62
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1562-L1564
"""


def canonical_dim_spec(spec: RectilinearDimSpec) -> RectilinearDimSpec:
    """One dimension's chunk sizes in their simplest equivalent form.

    Runs of equal sizes collapse to `[size, count]` pairs, because that is
    the spelling that does not grow with the number of chunks: a million
    equal chunks is two numbers, not a million. A run of one stays a bare
    size, and `[size, 1]` collapses to one, since a pair says nothing extra
    there. Adjacent spellings of the same size merge, which is what makes
    this idempotent: `[[32, 2], 32]` and `[32, [32, 2]]` both become
    `[[32, 3]]`.

    A dimension-level bare integer is left alone. It is a *step* that
    repeats until it covers the extent, so it is not equivalent to any
    fixed list — expanding it would pin a grid that currently adapts, and
    the two would diverge the moment the array were resized. For the same
    reason a one-element list is never collapsed to a bare integer:
    `[32]` declares exactly one chunk and `32` declares as many as it takes.

    Assumes a spec the shape validator has already accepted.
    """
    if not isinstance(spec, tuple):
        return spec
    runs: list[tuple[int, int]] = []
    for entry in spec:
        size, count = entry if isinstance(entry, tuple) else (entry, 1)
        if len(runs) != 0 and runs[-1][0] == size:
            runs[-1] = (size, runs[-1][1] + count)
        else:
            runs.append((size, count))
    return tuple(size if count == 1 else (size, count) for size, count in runs)


def canonical_chunk_shapes(
    chunk_shapes: tuple[RectilinearDimSpec, ...],
) -> tuple[RectilinearDimSpec, ...]:
    """Every dimension's chunk sizes in their simplest equivalent form."""
    return tuple(canonical_dim_spec(spec) for spec in chunk_shapes)


__all__ = [
    "RECTILINEAR_CHUNK_GRID_NAME",
    "RectilinearChunkGridConfiguration",
    "RectilinearChunkGridMetadata",
    "RectilinearChunkGridName",
    "RectilinearChunkGridObject",
    "RectilinearDimSpec",
    "canonical_chunk_shapes",
    "canonical_dim_spec",
]
