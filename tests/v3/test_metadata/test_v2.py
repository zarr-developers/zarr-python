from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

import pytest

from zarr.metadata import parse_zarr_format_v2


def test_parse_zarr_format_valid() -> None:
    assert parse_zarr_format_v2(2) == 2


@pytest.mark.parametrize("data", [None, 1, 3, 4, 5, "3"])
def test_parse_zarr_format_invalid(data: Any) -> None:
    with pytest.raises(ValueError, match=f"Invalid value. Expected 2. Got {data}"):
        parse_zarr_format_v2(data)
