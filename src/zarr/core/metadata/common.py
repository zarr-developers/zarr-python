from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from zarr.core.common import JSON


def parse_attributes(data: dict[str, JSON] | None) -> dict[str, JSON]:
    if data is None:
        return {}

    return data


def reject_must_understand_metadata(data: dict[str, Any] | None, dict_name: str) -> None:
    if data and not all(
        isinstance(value, dict) and value.get("must_understand") is False for value in data.values()
    ):
        raise ValueError(f"Unexpected {dict_name} keys: {list(data.keys())}")
