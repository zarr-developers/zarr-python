"""
Rectilinear chunk grid (zarr-extensions).

See https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/chunk-grids/rectilinear/README.md
"""

from collections.abc import Iterator
from typing import Final

from zarr_metadata._json import ValidationProblem
from zarr_metadata.definition._definition import Nested
from zarr_metadata.v3._definition import (
    ChunkGridDefinition,
    Lengths,
)
from zarr_metadata.v3._types.chunk_grid import (
    RECTILINEAR_CHUNK_GRID_NAME,
    RectilinearChunkGridConfiguration,
    RectilinearChunkGridMetadata,
    RectilinearChunkGridName,
    RectilinearChunkGridObject,
    RectilinearDimSpec,
)


def canonical_dim_spec(spec: RectilinearDimSpec) -> RectilinearDimSpec:
    """Return one dimension's chunk sizes in their simplest equivalent form.

    Runs of equal sizes collapse to `[size, count]` pairs, because that is
    the spelling that does not grow with the number of chunks: a million
    equal chunks is two numbers, not a million. A run of one stays a bare
    size, and `[size, 1]` collapses to a bare size, because a pair adds no
    information there. Adjacent spellings of the same size merge, which makes
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
    """Return every dimension's chunk sizes in their simplest equivalent form."""
    return tuple(canonical_dim_spec(spec) for spec in chunk_shapes)


def _canonical(
    configuration: RectilinearChunkGridConfiguration,
) -> RectilinearChunkGridConfiguration:
    """Run-length encode the chunk sizes, because that spelling does not grow as the array grows."""
    return RectilinearChunkGridConfiguration(
        kind=configuration["kind"],
        chunk_shapes=canonical_chunk_shapes(configuration["chunk_shapes"]),
    )


def _shape_rules(
    configuration: RectilinearChunkGridConfiguration, nested: Nested, shape: tuple[int, ...]
) -> Iterator[ValidationProblem]:
    """Check that there are chunk lengths for each of the array's dimensions, and that they cover the dimension.

    "The length of `chunk_shapes` MUST match the number of dimensions of
    the array", and "The sum of the edge lengths MUST equal or exceed `L`"
    (https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/chunk-grids/rectilinear/README.md?plain=1#L62-L91).
    A bare integer repeats until it covers the dimension, so it always
    passes this check.
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
    """Return, for each axis, every chunk length that its entry lists.

    A bare integer is the length of every chunk along its axis. A list
    gives the length of each chunk, and a `[length, count]` pair gives
    `count` chunks of that length
    (https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/chunk-grids/rectilinear/README.md?plain=1#L62-L91).
    Every listed length counts, including the length of a chunk past the
    array's edge, which the array grows into when it is resized.
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
