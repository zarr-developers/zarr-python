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
    must be at least 1, with one exception. `legacy_writers` stored a chunk
    size of 0 (or JSON `false`) for an array created with a zero-length axis
    and a chunk spec meaning "one chunk spanning the axis". That is how the
    size is read, `max(extent, 1)`, with a `ZarrUserWarning` that says how to
    re-save valid metadata. The axis may have grown since: those writers let
    the array be resized or appended to, but a chunk size of 0 gives a grid of
    zero chunks, so no chunk was ever stored for it and any chunk size reads
    the store correctly. The warning then says that the appended data was not
    saved. A negative chunk size is rejected.
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
                f"Dimension {dim_idx}: chunk edge length {size!r} (as written by "
                f"{legacy_writers} for an array created with a zero-length axis) is read "
                f"as one chunk spanning the axis, of size {corrected}."
            )
            if extent > 0:
                msg += (
                    f" The axis has since grown to {extent}, but no chunk can be stored "
                    "under a chunk size of 0, so data written to it before now was not "
                    "saved and reads as the fill value."
                )
            warnings.warn(f"{msg} {RESAVE_METADATA_HINT}", ZarrUserWarning, stacklevel=3)
            size = corrected
        parsed.append(size)
    return tuple(parsed)
