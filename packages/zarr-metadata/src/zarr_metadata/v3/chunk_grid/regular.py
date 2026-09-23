"""
Regular chunk grid (Zarr v3 core spec).

See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#regular-grids
"""

from collections.abc import Iterator
from typing import Final, Literal, NotRequired

from typing_extensions import TypedDict

from zarr_metadata._json import ValidationProblem
from zarr_metadata.v3._definition import ChunkGridDefinition

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


def _rules(configuration: RegularChunkGridConfiguration) -> Iterator[ValidationProblem]:
    """Every chunk extent is at least 1."""
    for index, extent in enumerate(configuration["chunk_shape"]):
        if extent < 1:
            yield ValidationProblem(
                ("chunk_shape", index), f"expected an integer >= 1, got {extent}", "invalid_value"
            )


REGULAR_CHUNK_GRID: Final = ChunkGridDefinition(
    name=REGULAR_CHUNK_GRID_NAME, configuration=RegularChunkGridConfiguration, rules=_rules
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
