"""Cover the `raw_bytes_dtype_name` brand validator.

The pydantic-driven fixture tests don't enforce the `r<N>` shape
because `RawBytesDataTypeName` is a `NewType`, which pydantic treats
as plain `str`. Direct coverage of the validator function lives here.
"""

from __future__ import annotations

import pytest

from zarr_metadata.v3.data_type.raw import raw_bytes_dtype_name

VALID = ["r8", "r16", "r24", "r256", "r1024"]
INVALID_FORMAT = ["", "8", "R8", "r", "r-8", "r8 ", " r8", "r8r8"]
INVALID_BITS = ["r0", "r1", "r7", "r9", "r15", "r17"]


@pytest.mark.parametrize("value", VALID)
def test_valid(value: str) -> None:
    assert raw_bytes_dtype_name(value) == value


@pytest.mark.parametrize("value", INVALID_FORMAT)
def test_invalid_format(value: str) -> None:
    with pytest.raises(ValueError, match="Expected 'r' followed by"):
        raw_bytes_dtype_name(value)


@pytest.mark.parametrize("value", INVALID_BITS)
def test_invalid_bit_count(value: str) -> None:
    with pytest.raises(ValueError, match="positive multiple of 8"):
        raw_bytes_dtype_name(value)
