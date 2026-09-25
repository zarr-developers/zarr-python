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
    chunk_shape: Sequence[int], shape: Sequence[int]
) -> tuple[int, ...]:
    """Validate a stored regular chunk grid's chunk shape against the array shape.

    This is for regular chunk grids only: Zarr format 2 `chunks`, and the
    `chunk_shape` of a Zarr format 3 `regular` grid. Another chunk grid, such
    as the rectilinear grid, is free to define its own meaning for a chunk of
    length 0, so its chunk sizes must not be passed here.

    The chunk shape must have one entry per array axis, and every chunk size
    must be at least 1, with one exception: a chunk size of 0 (or JSON
    `false`) is read as one chunk spanning its axis, `max(extent, 1)`, with a
    `ZarrUserWarning` that says how to re-save valid metadata. A chunk size of
    0 gives a grid of zero chunks, so no chunk can be stored under it and any
    positive chunk size reads the store correctly; if the axis has positive
    length, the warning also says that it holds only the fill value. A
    negative chunk size is rejected.
    """
    if len(chunk_shape) != len(shape):
        raise ValueError(
            f"The chunk shape {tuple(chunk_shape)} and the array shape {tuple(shape)} "
            "must have the same number of dimensions."
        )
    parsed: list[int] = []
    for dim_idx, (size, extent) in enumerate(zip(chunk_shape, shape, strict=True)):
        if size < 0:
            raise ValueError(f"Dimension {dim_idx}: chunk edge length must be >= 1, got {size!r}")
        if size == 0:
            corrected = max(extent, 1)
            msg = (
                f"Dimension {dim_idx}: chunk edge length {size!r} is invalid and is read "
                f"as one chunk spanning the axis, of size {corrected}."
            )
            if extent > 0:
                msg += (
                    f" No chunk can be stored under a chunk size of 0, so the {extent} "
                    "elements along this axis hold only the fill value."
                )
            warnings.warn(f"{msg} {RESAVE_METADATA_HINT}", ZarrUserWarning, stacklevel=3)
            size = corrected
        parsed.append(size)
    return tuple(parsed)
