"""Composition rules for the `rectilinear` chunk grid."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.rules._registry import entity_rule
from zarr_metadata.v3._extension_points import CHUNK_GRID
from zarr_metadata.v3.chunk_grid.rectilinear import RECTILINEAR_CHUNK_GRID_NAME

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr_metadata.rules._spec import ArraySpec, Sequence

_ARRAY_V3 = "zarr_v3_array"


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _expanded_extent(spec: Sequence[object]) -> int | None:
    """The total extent an explicit dimension spec covers, or None.

    Entries are chunk sizes or `[size, count]` run-length pairs. Answers
    None when any entry is non-positive — the values rule owns that
    complaint, and a sum over bad entries would be noise.
    """
    total = 0
    for item in spec:
        if _is_int(item) and cast(int, item) >= 1:
            total += cast(int, item)
        elif isinstance(item, tuple):
            size, count = cast("tuple[object, object]", item)
            if not (_is_int(size) and _is_int(count)):
                return None
            if cast(int, size) < 1 or cast(int, count) < 1:
                return None
            total += cast(int, size) * cast(int, count)
        else:
            return None
    return total


@entity_rule(_ARRAY_V3, CHUNK_GRID, RECTILINEAR_CHUNK_GRID_NAME)
def chunk_extents_are_positive(
    configuration: Mapping[str, object], document: Mapping[str, object], incoming: ArraySpec
) -> tuple[ValidationProblem, ...]:
    """Every chunk extent, bare or run-length encoded, must be positive."""
    chunk_shapes = cast("tuple[object, ...]", configuration["chunk_shapes"])
    problems: list[ValidationProblem] = []
    for dim, spec in enumerate(chunk_shapes):
        loc: tuple[str | int, ...] = ("chunk_shapes", dim)
        if _is_int(spec):
            if cast(int, spec) < 1:
                problems.append(
                    ValidationProblem(
                        loc, f"expected a positive chunk extent, got {spec}", "invalid_value"
                    )
                )
            continue
        if not isinstance(spec, tuple):
            continue
        for position, item in enumerate(cast("tuple[object, ...]", spec)):
            if _is_int(item) and cast(int, item) < 1:
                problems.append(
                    ValidationProblem(
                        (*loc, position),
                        f"expected a positive chunk extent, got {item}",
                        "invalid_value",
                    )
                )
            elif isinstance(item, tuple):
                size, count = cast("tuple[int, int]", item)
                if size < 1 or count < 1:
                    problems.append(
                        ValidationProblem(
                            (*loc, position),
                            f"expected a positive [size, count] pair, got {item!r}",
                            "invalid_value",
                        )
                    )
    return tuple(problems)


@entity_rule(_ARRAY_V3, CHUNK_GRID, RECTILINEAR_CHUNK_GRID_NAME, requires=frozenset({"shape"}))
def tiles_the_array(
    configuration: Mapping[str, object], document: Mapping[str, object], incoming: ArraySpec
) -> tuple[ValidationProblem, ...]:
    """One spec per dimension, and explicit specs must sum to that extent.

    A bare-integer dimension spec is uniform shorthand and imposes no sum
    constraint; an explicit list of chunk sizes must tile its dimension
    exactly.
    """
    shape = document["shape"]
    if not isinstance(shape, (list, tuple)):
        return ()
    extents = cast("tuple[object, ...]", shape)
    chunk_shapes = cast("tuple[object, ...]", configuration["chunk_shapes"])
    if len(chunk_shapes) != len(extents):
        return (
            ValidationProblem(
                ("chunk_shapes",),
                f"chunk_shapes has {len(chunk_shapes)} entries but shape has "
                f"{len(extents)} dimensions",
                "invalid_value",
            ),
        )
    problems: list[ValidationProblem] = []
    for dim, (spec, extent) in enumerate(zip(chunk_shapes, extents, strict=True)):
        if not _is_int(extent) or _is_int(spec) or not isinstance(spec, tuple):
            continue
        total = _expanded_extent(cast("tuple[object, ...]", spec))
        if total is not None and total < cast("int", extent):
            problems.append(
                ValidationProblem(
                    ("chunk_shapes", dim),
                    f"chunk sizes sum to {total} but must cover shape[{dim}] extent {extent}",
                    "invalid_value",
                )
            )
    return tuple(problems)
