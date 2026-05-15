"""Validate uint8 fill-value fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from zarr_metadata.v3.data_type.uint8 import Uint8FillValue

FILL_VALUES: dict[str, object] = json.loads(
    (Path(__file__).parent / "fill_values.json").read_text()
)


@pytest.mark.parametrize("case", FILL_VALUES.values(), ids=list(FILL_VALUES))
def test_fill_value(case: object) -> None:
    TypeAdapter(Uint8FillValue).validate_python(case)
