"""Composition rules for the `regular` chunk grid."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.rules._registry import entity_rule
from zarr_metadata.v3._extension_points import CHUNK_GRID
from zarr_metadata.v3.chunk_grid.regular import REGULAR_CHUNK_GRID_NAME

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr_metadata.rules._spec import ArraySpec

_ARRAY_V3 = "zarr_v3_array"


@entity_rule(_ARRAY_V3, CHUNK_GRID, REGULAR_CHUNK_GRID_NAME)
def chunk_extents_are_positive(
    configuration: Mapping[str, object], document: Mapping[str, object], incoming: ArraySpec
) -> tuple[ValidationProblem, ...]:
    """Every chunk extent must be at least one element.

    A zero extent makes the chunk index `floor(i / 0)` undefined; a
    negative one is meaningless. The shape validator enforces that the
    entries are integers, so this rule judges only their values.
    """
    chunk_shape = cast("tuple[int, ...]", configuration["chunk_shape"])
    return tuple(
        ValidationProblem(
            ("chunk_shape", position),
            f"expected a positive chunk extent, got {extent}",
            "invalid_value",
        )
        for position, extent in enumerate(chunk_shape)
        if extent < 1
    )


@entity_rule(_ARRAY_V3, CHUNK_GRID, REGULAR_CHUNK_GRID_NAME, requires=frozenset({"shape"}))
def chunks_every_dimension(
    configuration: Mapping[str, object], document: Mapping[str, object], incoming: ArraySpec
) -> tuple[ValidationProblem, ...]:
    """A regular grid must chunk every array dimension."""
    shape = document["shape"]
    if not isinstance(shape, (list, tuple)):
        return ()
    chunk_shape = cast("tuple[int, ...]", configuration["chunk_shape"])
    if len(chunk_shape) == len(cast("tuple[object, ...]", shape)):
        return ()
    return (
        ValidationProblem(
            ("chunk_shape",),
            f"chunk_shape has {len(chunk_shape)} entries but shape has "
            f"{len(cast('tuple[object, ...]', shape))} dimensions",
            "invalid_value",
        ),
    )
