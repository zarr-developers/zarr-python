"""Cover the `base64_bytes` brand validator.

The pydantic-driven fixture tests don't enforce the base64 alphabet or
length-multiple-of-4 constraint because `Base64Bytes` is a `NewType`,
which pydantic treats as plain `str`. Direct coverage of the validator
function lives here.
"""

from __future__ import annotations

import pytest

from zarr_metadata.v3.data_type.bytes import base64_bytes

VALID = [
    "",  # empty is valid base64 (length 0, multiple of 4)
    "AQID",  # [1, 2, 3]
    "AAAA",  # [0, 0, 0]
    "////",  # [255, 255, 255]
    "abcd",
    "AB==",  # padding
    "ABC=",  # padding
]
INVALID = [
    "AB",  # length 2, not multiple of 4
    "ABC",  # length 3, not multiple of 4
    "ABCDE",  # length 5
    "AB-D",  # url-safe alphabet, not standard
    "AB_D",  # url-safe alphabet, not standard
    "AB!D",  # not base64 char
    "AB CD",  # whitespace
]


@pytest.mark.parametrize("value", VALID)
def test_valid(value: str) -> None:
    assert base64_bytes(value) == value


@pytest.mark.parametrize("value", INVALID)
def test_invalid(value: str) -> None:
    with pytest.raises(ValueError, match="standard-alphabet base64"):
        base64_bytes(value)
