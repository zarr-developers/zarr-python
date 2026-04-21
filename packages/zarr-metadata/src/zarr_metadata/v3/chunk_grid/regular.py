"""
Regular chunk grid (Zarr v3 core spec).

See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#regular-grids
"""

from typing import Final, Literal, TypedDict

REGULAR_CHUNK_GRID_NAME: Final = "regular"
"""The `name` field value of the regular chunk grid."""

RegularChunkGridName = Literal["regular"]
"""Literal type of the `name` field of the regular chunk grid."""


class RegularChunkGridConfiguration(TypedDict):
    """Configuration for the regular chunk grid."""

    chunk_shape: tuple[int, ...]


class RegularChunkGrid(TypedDict):
    """Regular chunk grid metadata."""

    name: RegularChunkGridName
    configuration: RegularChunkGridConfiguration


__all__ = [
    "REGULAR_CHUNK_GRID_NAME",
    "RegularChunkGrid",
    "RegularChunkGridConfiguration",
    "RegularChunkGridName",
]
