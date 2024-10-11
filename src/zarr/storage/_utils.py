from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zarr.core.buffer import Buffer


def _normalize_interval_index(
    data: Buffer, interval: None | tuple[int | None, int | None]
) -> tuple[int, int]:
    """
    Convert an implicit interval into an explicit start and length
    """
    if interval is None:
        start = 0
        length = len(data)
    else:
        maybe_start, maybe_len = interval
        if maybe_start is None:
            start = 0
        else:
            start = maybe_start

        if maybe_len is None:
            length = len(data) - start
        else:
            length = maybe_len

    return (start, length)
