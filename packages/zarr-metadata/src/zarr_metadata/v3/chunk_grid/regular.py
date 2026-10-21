"""
Regular chunk grid (Zarr v3 core spec).

See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#regular-grids
"""

from collections.abc import Iterator
from typing import Final, Literal, NotRequired

from typing_extensions import TypedDict

from zarr_metadata._json import ValidationProblem
from zarr_metadata.v3._definition import ChunkGridDefinition, Lengths, Nested

REGULAR_CHUNK_GRID_NAME: Final = "regular"
"""The `name` field value of the regular chunk grid."""

RegularChunkGridName = Literal["regular"]
"""Literal type of the `name` field of the regular chunk grid."""


class RegularChunkGridConfiguration(TypedDict, closed=True):
    """Configuration for the regular chunk grid."""

    chunk_shape: tuple[int, ...]


class RegularChunkGridObject(TypedDict, closed=True):
    """Regular chunk grid metadata in object form."""

    name: RegularChunkGridName
    configuration: RegularChunkGridConfiguration
    must_understand: NotRequired[bool]


RegularChunkGridMetadata = RegularChunkGridObject
"""Permitted JSON shape for regular chunk grid metadata.

`chunk_shape` is required and has no default, so only the object form is
valid; the short-hand-name form is not permitted by the spec for this grid.
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L528-L537 ("must be an object with the names name and configuration")
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1562-L1564
"""


def _rules(
    configuration: RegularChunkGridConfiguration, nested: Nested
) -> Iterator[ValidationProblem]:
    """No chunk extent is negative.

    "The chunk shape elements are non-zero when the corresponding
    dimensions of the arrays have non-zero length": an extent of 0 is
    right for a dimension of length 0, which zarr-python 3.0 and 3.1
    wrote, and which a grid alone cannot tell from one that is not; the
    shape rules can, beside the array's shape.
    """
    for index, extent in enumerate(configuration["chunk_shape"]):
        if extent < 0:
            yield ValidationProblem(
                ("chunk_shape", index), f"expected an integer >= 0, got {extent}", "invalid_value"
            )


def _shape_rules(
    configuration: RegularChunkGridConfiguration, nested: Nested, shape: tuple[int, ...]
) -> Iterator[ValidationProblem]:
    """A chunk length for each of the array's dimensions, and 0 only for a dimension of length 0.

    "The dimensionality of the grid is the same as the dimensionality of
    the array"
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/chunk-grids/regular-grid/index.rst#L29-L31),
    and "The chunk shape elements are non-zero when the corresponding
    dimensions of the arrays have non-zero length"
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L284-L285).
    """
    chunk_shape = configuration["chunk_shape"]
    if len(chunk_shape) != len(shape):
        yield ValidationProblem(
            ("chunk_shape",),
            f"expected one chunk length per dimension of shape, got {len(chunk_shape)}",
            "invalid_value",
        )
        return
    for axis, (length, extent) in enumerate(zip(chunk_shape, shape, strict=True)):
        if length == 0 and extent != 0:
            yield ValidationProblem(
                ("chunk_shape", axis),
                f"expected a chunk length >= 1 for a dimension of length {extent}, got 0",
                "invalid_value",
            )


def _chunk_lengths(
    configuration: RegularChunkGridConfiguration, nested: Nested, shape: tuple[int, ...]
) -> Lengths:
    """One length along each axis: every chunk has the grid's `chunk_shape`.

    "each chunk is a hyperrectangle of the same shape"
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/chunk-grids/regular-grid/index.rst#L28-L29),
    a chunk at the array's edge too, where "the grid will overhang the
    edge of the array space"
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/chunk-grids/regular-grid/index.rst#L43-L46).
    """
    return tuple(frozenset({length}) for length in configuration["chunk_shape"])


REGULAR_CHUNK_GRID: Final = ChunkGridDefinition(
    name=REGULAR_CHUNK_GRID_NAME,
    configuration=RegularChunkGridConfiguration,
    rules=_rules,
    shape_rules=_shape_rules,
    chunk_lengths=_chunk_lengths,
)
"""The `regular` chunk grid."""

__all__ = [
    "REGULAR_CHUNK_GRID",
    "REGULAR_CHUNK_GRID_NAME",
    "RegularChunkGridConfiguration",
    "RegularChunkGridMetadata",
    "RegularChunkGridName",
    "RegularChunkGridObject",
]
