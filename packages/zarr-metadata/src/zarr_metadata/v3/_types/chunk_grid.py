"""The types of the v3 chunk grids.

For each chunk grid: the value and `Literal` type of its name, its
configuration and object forms, and the JSON shapes its metadata may take.

This module holds types only and imports no definition. Each chunk grid's
submodule of `zarr_metadata.v3.chunk_grid` re-exports its names, and
holds the chunk grid's definition and rules.
"""

from typing import Annotated, Final, Literal, NotRequired

from annotated_types import Ge
from typing_extensions import TypedDict

# regular: https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#regular-grids

REGULAR_CHUNK_GRID_NAME: Final = "regular"
"""The `name` field value of the regular chunk grid."""

RegularChunkGridName = Literal["regular"]
"""Literal type of the `name` field of the regular chunk grid."""


class RegularChunkGridConfiguration(TypedDict, closed=True):
    """Configuration for the regular chunk grid.

    Every chunk length is at least 1, even along a dimension of length 0:
    "Chunk sizes must be greater than zero"
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/chunk-grids/regular-grid/index.rst#L40).
    The core spec's "The chunk shape elements are non-zero when the
    corresponding dimensions of the arrays have non-zero length"
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L284-L285)
    says less about an empty dimension, and it does not allow anything
    that the grid spec forbids.
    """

    chunk_shape: tuple[Annotated[int, Ge(1)], ...]
    """The length of every chunk along each dimension of the array.

    One entry per dimension, each at least 1.
    """


class RegularChunkGridObject(TypedDict, closed=True):
    """Regular chunk grid metadata in object form."""

    name: RegularChunkGridName
    """The chunk grid's name, `"regular"`."""
    configuration: RegularChunkGridConfiguration
    """The chunk grid's parameters."""
    must_understand: NotRequired[bool]
    """Whether a reader that does not recognize this chunk grid must refuse the document.

    Absent means `true`. The spec does not permit `false` for a chunk grid
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1580-L1581).
    """


RegularChunkGridMetadata = RegularChunkGridObject
"""Permitted JSON shape for regular chunk grid metadata.

`chunk_shape` is required and has no default, so only the object form is
valid; the short-hand-name form is not permitted by the spec for this grid.
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L528-L537 ("must be an object with the names name and configuration")
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1562-L1564
"""


# rectilinear: https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/chunk-grids/rectilinear/README.md

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
    """How `chunk_shapes` is encoded; `"inline"` is the only kind the spec defines."""
    chunk_shapes: tuple[RectilinearDimSpec, ...]
    """The chunk edge lengths along each dimension of the array, one entry per dimension.

    An entry is a bare integer, a step repeated until it covers the
    dimension, or a list of edge lengths and `[length, count]` runs, whose
    lengths must sum to at least the dimension's length. Every length and
    count is at least 1.
    """


class RectilinearChunkGridObject(TypedDict, closed=True):
    """Rectilinear chunk grid metadata in object form."""

    name: RectilinearChunkGridName
    """The chunk grid's name, `"rectilinear"`."""
    configuration: RectilinearChunkGridConfiguration
    """The chunk grid's parameters."""
    must_understand: NotRequired[bool]
    """Whether a reader that does not recognize this chunk grid must refuse the document.

    Absent means `true`. The spec does not permit `false` for a chunk grid
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1580-L1581).
    """


RectilinearChunkGridMetadata = RectilinearChunkGridObject
"""Permitted JSON shape for rectilinear chunk grid metadata.

`kind` and `chunk_shapes` are required, so only the object form is valid;
the short-hand-name form is not permitted by the spec for this grid.
  https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/chunk-grids/rectilinear/README.md#L59-L62
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1562-L1564
"""


__all__ = [
    "RECTILINEAR_CHUNK_GRID_NAME",
    "REGULAR_CHUNK_GRID_NAME",
    "RectilinearChunkGridConfiguration",
    "RectilinearChunkGridMetadata",
    "RectilinearChunkGridName",
    "RectilinearChunkGridObject",
    "RectilinearDimSpec",
    "RegularChunkGridConfiguration",
    "RegularChunkGridMetadata",
    "RegularChunkGridName",
    "RegularChunkGridObject",
]
