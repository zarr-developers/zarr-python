"""Cover the `hex_float32` brand validator.

The pydantic-driven fixture tests don't enforce hex format because
`HexFloat32` is a `NewType`, which pydantic treats as plain `str`.
Direct coverage of the validator function lives here.
"""

from __future__ import annotations

import pytest

from zarr_metadata.v3.data_type.float32 import hex_float32

VALID = [
    "0x00000000",
    "0x7fc00000",  # canonical NaN
    "0x7fa00000",  # signaling NaN
    "0xffffffff",
    "0xFFFFFFFF",
    "0xDeadBeef",
]
INVALID = [
    "",
    "00000000",  # missing 0x
    "0x0000000",  # too short
    "0x000000000",  # too long
    "0x0000000g",  # non-hex char
    "0X00000000",  # uppercase X
    "  0x00000000  ",  # whitespace
]


@pytest.mark.parametrize("value", VALID)
def test_valid(value: str) -> None:
    assert hex_float32(value) == value


@pytest.mark.parametrize("value", INVALID)
def test_invalid(value: str) -> None:
    with pytest.raises(ValueError, match="Expected '0x'"):
        hex_float32(value)
