"""
Rectilinear chunk grid (zarr-extensions).

See https://github.com/zarr-developers/zarr-extensions/tree/main/chunk-grids/rectilinear
"""

from typing import Final, Literal, TypedDict

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


class RectilinearChunkGridConfiguration(TypedDict):
    """Configuration for the rectilinear chunk grid."""

    kind: Literal["inline"]
    chunk_shapes: tuple[RectilinearDimSpec, ...]


class RectilinearChunkGrid(TypedDict):
    """Rectilinear chunk grid metadata."""

    name: RectilinearChunkGridName
    configuration: RectilinearChunkGridConfiguration


__all__ = [
    "RECTILINEAR_CHUNK_GRID_NAME",
    "RectilinearChunkGrid",
    "RectilinearChunkGridConfiguration",
    "RectilinearChunkGridName",
    "RectilinearDimSpec",
]
