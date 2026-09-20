"""
Regular chunk grid (Zarr v3 core spec).

See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#regular-grids
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Final, Literal, NotRequired, cast

from typing_extensions import TypedDict, Unpack

from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.v3._entity import (
    ChunkGridEntity,
    problem,
)
from zarr_metadata.v3._parts import ChunkGrid

if TYPE_CHECKING:
    from collections.abc import Sequence


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

__all__ = [
    "REGULAR_CHUNK_GRID_NAME",
    "RegularChunkGrid",
    "RegularChunkGridConfiguration",
    "RegularChunkGridMetadata",
    "RegularChunkGridName",
    "RegularChunkGridObject",
]


@dataclass(frozen=True)
class RegularChunkGrid(ChunkGridEntity):
    """The `regular` chunk grid, coerced from its metadata."""

    chunk_shape: tuple[int, ...]

    identifier: ClassVar[str] = REGULAR_CHUNK_GRID_NAME
    configuration_type = RegularChunkGridConfiguration

    @staticmethod
    def value_problems(
        **members: Unpack[RegularChunkGridConfiguration],
    ) -> tuple[ValidationProblem, ...]:
        """Every chunk extent must be at least one element.

        A chunk of zero elements along an axis covers nothing, so no
        finite number of them tiles the axis; a negative one is
        meaningless. Whether there is one extent *per array dimension* is
        a question for the document, and the rules layer asks it.
        """
        return tuple(
            ValidationProblem(
                ("chunk_shape", position),
                f"expected a positive chunk extent, got {extent}",
                "invalid_value",
            )
            for position, extent in enumerate(members["chunk_shape"])
            if extent < 1
        )

    def shape_problems(self, array_shape: object) -> tuple[ValidationProblem, ...]:
        """A regular grid must chunk every array dimension."""
        if not isinstance(array_shape, (list, tuple)):
            return ()
        extents = tuple(cast("Sequence[object]", array_shape))
        if len(self.chunk_shape) == len(extents):
            return ()
        return problem(
            ("chunk_shape",),
            f"chunk_shape has {len(self.chunk_shape)} entries but shape has "
            f"{len(extents)} dimensions",
            "invalid_value",
        )

    def grid(self, array_shape: object) -> ChunkGrid:
        """One extent per axis, the same for every chunk on that axis."""
        return ChunkGrid.regular(self.chunk_shape)

    def to_json(self) -> RegularChunkGridObject:
        return cast("RegularChunkGridObject", super().to_json())
