from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Final

from zarr.errors import ZarrUserWarning

if TYPE_CHECKING:
    from collections.abc import Sequence

    from zarr.core.common import JSON

RESAVE_METADATA_HINT: Final = (
    "Re-save the array metadata to store the corrected value: open the array "
    "writable and call `array.update_attributes({})`."
)
"""How to persist metadata that was read under a compatibility policy.

`update_attributes` rewrites the whole metadata document from the parsed
(corrected) metadata, so an empty update is enough.
"""


def parse_attributes(data: dict[str, JSON] | None) -> dict[str, JSON]:
    if data is None:
        return {}

    return dict(data)


def parse_stored_regular_chunk_shape(
    chunk_shape: Sequence[int], shape: Sequence[int], *, legacy_writers: str
) -> tuple[int, ...]:
    """Validate a stored regular chunk grid's chunk shape against the array shape.

    This is for regular chunk grids only: Zarr format 2 `chunks`, and the
    `chunk_shape` of a Zarr format 3 `regular` grid. Another chunk grid, such
    as the rectilinear grid, is free to define its own meaning for a chunk of
    length 0, so its chunk sizes must not be passed here.

    The chunk shape must have one entry per array axis, and every chunk size
    must be at least 1. The exception is a chunk size of 0 on a zero-length
    axis, which `legacy_writers` stored for arrays created empty. An empty axis
    has no chunk for the size to describe, so it is read as 1 with a
    `ZarrUserWarning` that says how to re-save valid metadata. A chunk size of
    0 on a positive-length axis is rejected, because the metadata cannot say
    how the stored chunks were laid out. JSON `false` counts as 0.
    """
    if len(chunk_shape) != len(shape):
        raise ValueError(
            f"The chunk shape {tuple(chunk_shape)} and the array shape {tuple(shape)} "
            "must have the same number of dimensions."
        )
    parsed: list[int] = []
    for dim_idx, (size, extent) in enumerate(zip(chunk_shape, shape, strict=True)):
        if size < 1:
            if size < 0 or extent != 0:
                raise ValueError(
                    f"Dimension {dim_idx}: chunk edge length must be >= 1, got {size!r}"
                )
            warnings.warn(
                f"Dimension {dim_idx}: chunk edge length {size!r} on a zero-length axis "
                f"(as written by {legacy_writers}) is treated as 1. {RESAVE_METADATA_HINT}",
                ZarrUserWarning,
                stacklevel=3,
            )
            size = 1
        parsed.append(size)
    return tuple(parsed)
