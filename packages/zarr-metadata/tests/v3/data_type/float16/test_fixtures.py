"""Validate float16 fill-value fixtures.

A v3 float fill_value is a JSON number, one of the named non-finite
sentinels (`"NaN"`, `"Infinity"`, `"-Infinity"`), or a hex string
(`"0xYYYY"`) encoding the unsigned-integer representation of the IEEE
754 value.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from zarr_metadata.v3.data_type.float16 import Float16FillValue

FILL_VALUES: dict[str, object] = json.loads(
    (Path(__file__).parent / "fill_values.json").read_text()
)


@pytest.mark.parametrize("case", FILL_VALUES.values(), ids=list(FILL_VALUES))
def test_fill_value(case: object) -> None:
    TypeAdapter(Float16FillValue).validate_python(case)
