"""
Rectilinear chunk grid (zarr-extensions).

See https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/chunk-grids/rectilinear/README.md
"""

from collections.abc import Iterator
from typing import Annotated, Final, Literal, NotRequired

from annotated_types import Ge
from typing_extensions import TypedDict

from zarr_metadata._json import ValidationProblem
from zarr_metadata.v3._definition import ChunkGridDefinition, Lengths, Nested

RECTILINEAR_CHUNK_GRID_NAME: Final = "rectilinear"
"""The `name` field value of the rectilinear chunk grid."""

RectilinearChunkGridName = Literal["rectilinear"]
"""Literal type of the `name` field of the rectilinear chunk grid."""

_Positive = Annotated[int, Ge(1)]

RectilinearDimSpec = _Positive | tuple[_Positive | tuple[_Positive, _Positive], ...]
"""JSON shape for one dimension's rectilinear spec.

Either a bare integer (uniform shorthand for a regular dimension within
a rectilinear grid), or a tuple of integers and/or `[value, count]` RLE
pairs. Every extent, and every run's length and count, is at least 1.
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
    fixed list: expanding it would pin a grid that currently adapts, and
    the two would diverge the moment the array were resized. For the same
    reason a one-element list is never collapsed to a bare integer:
    `[32]` declares exactly one chunk and `32` declares as many as it takes.

    Assumes a spec the rules have accepted.
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


def _canonical(
    configuration: RectilinearChunkGridConfiguration,
) -> RectilinearChunkGridConfiguration:
    """Run-length encoded, which is the spelling that does not grow as the array does."""
    return RectilinearChunkGridConfiguration(
        kind=configuration["kind"],
        chunk_shapes=canonical_chunk_shapes(configuration["chunk_shapes"]),
    )


def _shape_rules(
    configuration: RectilinearChunkGridConfiguration, nested: Nested, shape: tuple[int, ...]
) -> Iterator[ValidationProblem]:
    """Chunk lengths for each of the array's dimensions, which cover it.

    "The length of `chunk_shapes` MUST match the number of dimensions of
    the array", and "The sum of the edge lengths MUST equal or exceed `L`"
    (https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/chunk-grids/rectilinear/README.md?plain=1#L62-L91).
    A bare integer repeats until it covers the dimension, so it always does.
    """
    chunk_shapes = configuration["chunk_shapes"]
    if len(chunk_shapes) != len(shape):
        yield ValidationProblem(
            ("chunk_shapes",),
            f"expected one chunk_shapes entry per dimension of shape, got {len(chunk_shapes)}",
            "invalid_value",
        )
        return
    for axis, (spec, extent) in enumerate(zip(chunk_shapes, shape, strict=True)):
        if isinstance(spec, int):
            continue
        covered = sum(entry if isinstance(entry, int) else entry[0] * entry[1] for entry in spec)
        if covered < extent:
            yield ValidationProblem(
                ("chunk_shapes", axis),
                f"expected chunk lengths that cover the dimension's length {extent}, "
                f"got lengths summing to {covered}",
                "invalid_value",
            )


def _chunk_lengths(
    configuration: RectilinearChunkGridConfiguration, nested: Nested, shape: tuple[int, ...]
) -> Lengths:
    """Along each axis, every chunk length its entry lists.

    A bare integer is the length of every chunk along its axis; a list
    gives each chunk's length, a `[length, count]` pair `count` of them
    (https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/chunk-grids/rectilinear/README.md?plain=1#L62-L91).
    Every length listed counts, a chunk past the array's edge too, which
    the array grows into when it is resized.
    """
    return tuple(
        frozenset({spec})
        if isinstance(spec, int)
        else frozenset(entry if isinstance(entry, int) else entry[0] for entry in spec)
        for spec in configuration["chunk_shapes"]
    )


RECTILINEAR_CHUNK_GRID: Final = ChunkGridDefinition(
    name=RECTILINEAR_CHUNK_GRID_NAME,
    configuration=RectilinearChunkGridConfiguration,
    canonical=_canonical,
    shape_rules=_shape_rules,
    chunk_lengths=_chunk_lengths,
)
"""The `rectilinear` chunk grid."""


__all__ = [
    "RECTILINEAR_CHUNK_GRID",
    "RECTILINEAR_CHUNK_GRID_NAME",
    "RectilinearChunkGridConfiguration",
    "RectilinearChunkGridMetadata",
    "RectilinearChunkGridName",
    "RectilinearChunkGridObject",
    "RectilinearDimSpec",
    "canonical_chunk_shapes",
    "canonical_dim_spec",
]
