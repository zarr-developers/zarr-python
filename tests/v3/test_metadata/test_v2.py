from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

import pytest

from zarr.metadata.v2 import parse_zarr_format


@pytest.mark.parametrize("data", [None, 3, "3", (1,)])
def test_parse_zarr_format_invalid(data: Any) -> None:
    with pytest.raises(ValueError, match=re.escape(f"Invalid value. Expected 2. Got {data}.")):
        parse_zarr_format(data)


def test_parse_zarr_format_valid() -> None:
    assert parse_zarr_format(2) == 2
