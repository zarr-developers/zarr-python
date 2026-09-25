from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zarr.core.common import JSON


def parse_attributes(data: dict[str, JSON] | None) -> dict[str, JSON]:
    if data is None:
        return {}

    return dict(data)


def parse_chunk_edge(size: object, axis: int) -> int:
    """Check that `size` is a chunk edge length: an `int` (not a `bool`) of at least 1."""
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError(
            f"Dimension {axis}: chunk edge length must be an integer >= 1, got {size!r}"
        )
    return size
