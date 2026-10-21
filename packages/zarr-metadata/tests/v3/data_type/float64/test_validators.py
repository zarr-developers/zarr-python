"""Cover the `hex_float64` brand validator.

The pydantic-driven fixture tests don't enforce hex format because
`HexFloat64` is a `NewType`, which pydantic treats as plain `str`.
Direct coverage of the validator function lives here.
"""

from __future__ import annotations

import pytest

from zarr_metadata.v3.data_type.float64 import hex_float64

VALID = [
    "0x0000000000000000",
    "0x7ff8000000000000",  # canonical NaN
    "0x7ff4000000000000",  # signaling NaN
    "0xffffffffffffffff",
    "0xFFFFFFFFFFFFFFFF",
    "0xDeadBeefCafeBabe",
]
INVALID = [
    "",
    "0000000000000000",  # missing 0x
    "0x000000000000000",  # too short
    "0x00000000000000000",  # too long
    "0x000000000000000g",  # non-hex char
    "0X0000000000000000",  # uppercase X
    "  0x0000000000000000  ",  # whitespace
]


@pytest.mark.parametrize("value", VALID)
def test_valid(value: str) -> None:
    assert hex_float64(value) == value


@pytest.mark.parametrize("value", INVALID)
def test_invalid(value: str) -> None:
    with pytest.raises(ValueError, match="Expected '0x'"):
        hex_float64(value)
