"""Validate numpy.timedelta64 dtype value and fill-value fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from zarr_metadata.v3.data_type.numpy_timedelta64 import (
    NumpyTimedelta64,
    NumpyTimedelta64FillValue,
)

DIR = Path(__file__).parent
FILL_VALUES: dict[str, object] = json.loads((DIR / "fill_values.json").read_text())


def test_data_type() -> None:
    TypeAdapter(NumpyTimedelta64).validate_python(json.loads((DIR / "data_type.json").read_text()))


@pytest.mark.parametrize("case", FILL_VALUES.values(), ids=list(FILL_VALUES))
def test_fill_value(case: object) -> None:
    TypeAdapter(NumpyTimedelta64FillValue).validate_python(case)
