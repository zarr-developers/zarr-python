"""Cover the `hex_float16` brand validator.

The pydantic-driven fixture tests don't enforce hex format because
`HexFloat16` is a `NewType`, which pydantic treats as plain `str`.
Direct coverage of the validator function lives here.
"""

from __future__ import annotations

import pytest

from zarr_metadata.v3.data_type.float16 import hex_float16

VALID = ["0x0000", "0x7c00", "0x7d00", "0xffff", "0xFFFF", "0xAbCd"]
INVALID = [
    "",
    "0000",  # missing 0x
    "0x000",  # too short
    "0x00000",  # too long
    "0x000g",  # non-hex char
    "0X0000",  # uppercase X
    "  0x0000  ",  # whitespace
]


@pytest.mark.parametrize("value", VALID)
def test_valid(value: str) -> None:
    assert hex_float16(value) == value


@pytest.mark.parametrize("value", INVALID)
def test_invalid(value: str) -> None:
    with pytest.raises(ValueError, match="Expected '0x'"):
        hex_float16(value)
