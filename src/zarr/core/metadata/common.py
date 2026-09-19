from __future__ import annotations

import numbers
import warnings
from typing import TYPE_CHECKING, Any, Final

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


def parse_stored_chunk_shape(
    chunk_shape: Sequence[Any], shape: Sequence[int], *, legacy_writers: str
) -> tuple[Any, ...]:
    """Validate a stored chunk shape against the array shape.

    This is the one place a stored per-axis chunk shape is checked against the
    array it belongs to, for both Zarr formats. The chunk shape must have one
    entry per array axis, and every integer chunk size must be at least 1.

    The exception is a chunk size of 0 on a zero-length axis, which
    `legacy_writers` stored for arrays created empty. An empty axis has no
    chunk for the size to describe, so it is read as 1 with a
    `ZarrUserWarning` that says how to re-save valid metadata. A chunk size of
    0 on a positive-length axis is rejected, because the metadata cannot say
    how the stored chunks were laid out. JSON `false` counts as 0.

    Entries that are not integers, such as explicit chunk edge lists, are
    returned unchanged for the caller's own parser.
    """
    if len(chunk_shape) != len(shape):
        raise ValueError(
            f"The chunk shape {tuple(chunk_shape)} and the array shape {tuple(shape)} "
            "must have the same number of dimensions."
        )
    parsed: list[Any] = []
    for dim_idx, (size, extent) in enumerate(zip(chunk_shape, shape, strict=True)):
        if isinstance(size, numbers.Integral) and size < 1:
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
