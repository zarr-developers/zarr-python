"""
Regular chunk grid (Zarr v3 core spec).

See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#regular-grids
"""

from dataclasses import dataclass
from typing import ClassVar, Final, Literal, NotRequired, cast

from typing_extensions import TypedDict

from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.v3._entity import (
    MemberTypes,
    MetadataEntity,
    is_int,
    sequence_of,
)

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
class RegularChunkGrid(MetadataEntity):
    """The `regular` chunk grid, coerced from its metadata."""

    chunk_shape: tuple[int, ...] = ()

    identifier: ClassVar[str] = REGULAR_CHUNK_GRID_NAME

    configuration_required: ClassVar[bool] = True
    member_types: ClassVar[MemberTypes] = {"chunk_shape": (True, sequence_of(is_int))}

    def problems(self) -> tuple[ValidationProblem, ...]:
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
            for position, extent in enumerate(self.chunk_shape)
            if extent < 1
        )

    def to_json(self) -> RegularChunkGridObject:
        return cast("RegularChunkGridObject", super().to_json())
