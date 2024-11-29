from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zarr.core.common import JSON


def parse_attributes(data: None | dict[str, JSON]) -> dict[str, JSON]:
    if data is None:
        return {}

    return data
