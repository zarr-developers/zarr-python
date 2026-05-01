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


class RegularChunkGridObject(TypedDict):
    """Regular chunk grid metadata in object form."""

    name: RegularChunkGridName
    configuration: RegularChunkGridConfiguration


RegularChunkGridMetadata = RegularChunkGridObject
"""Permitted JSON shape for regular chunk grid metadata.

`chunk_shape` is required and has no default, so only the object form is
valid; the short-hand-name form is not permitted by the spec for this grid.
"""


__all__ = [
    "REGULAR_CHUNK_GRID_NAME",
    "RegularChunkGridConfiguration",
    "RegularChunkGridMetadata",
    "RegularChunkGridName",
    "RegularChunkGridObject",
]
