"""Validate complex64 fill-value fixtures.

A v3 complex fill_value is a two-element JSON array `[real, imag]` where
each component is shaped per the corresponding float's fill value: a
number, one of the named sentinels (`"NaN"`, `"Infinity"`,
`"-Infinity"`), or a hex string of the underlying float's bits.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from zarr_metadata.v3.data_type.complex64 import Complex64FillValue

FILL_VALUES: dict[str, object] = json.loads(
    (Path(__file__).parent / "fill_values.json").read_text()
)


@pytest.mark.parametrize("case", FILL_VALUES.values(), ids=list(FILL_VALUES))
def test_fill_value(case: object) -> None:
    TypeAdapter(Complex64FillValue).validate_python(case)
