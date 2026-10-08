from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zarr.core.common import JSON


def parse_attributes(data: object) -> dict[str, JSON]:
    if data is None:
        return {}
    if not isinstance(data, dict) or not all(isinstance(k, str) for k in data):
        raise TypeError(f"Expected dict with string keys. Got {type(data)} instead.")
    return dict(data)
