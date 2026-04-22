"""
Tests for the runtime validators in zarr-metadata.

The TypedDicts and Final constants in this package have no runtime
behavior to test -- pyright (in CI) verifies their shapes. The only
runtime logic is the small set of validating constructors for spec
strings whose constraints can't be expressed as a Literal type.
"""

from __future__ import annotations

import pytest


def test_hex_float16_validator() -> None:
    from zarr_metadata.v3.data_type.float16 import hex_float16

    assert hex_float16("0x7c00") == "0x7c00"
    with pytest.raises(ValueError):
        hex_float16("0x7c")  # too short
    with pytest.raises(ValueError):
        hex_float16("0X7C00")  # uppercase 0X prefix not accepted
    with pytest.raises(ValueError):
        hex_float16("not hex")


def test_hex_float32_validator() -> None:
    from zarr_metadata.v3.data_type.float32 import hex_float32

    assert hex_float32("0x7fc00000") == "0x7fc00000"
    with pytest.raises(ValueError):
        hex_float32("0x7fc0")  # too short
    with pytest.raises(ValueError):
        hex_float32("not hex")


def test_hex_float64_validator() -> None:
    from zarr_metadata.v3.data_type.float64 import hex_float64

    assert hex_float64("0x7ff8000000000000") == "0x7ff8000000000000"
    with pytest.raises(ValueError):
        hex_float64("0x7ff8")  # too short
    with pytest.raises(ValueError):
        hex_float64("not hex")


def test_base64_bytes_validator() -> None:
    from zarr_metadata.v3.data_type.bytes import base64_bytes

    assert base64_bytes("SGVsbG8=") == "SGVsbG8="
    assert base64_bytes("") == ""
    assert base64_bytes("AAAA") == "AAAA"

    with pytest.raises(ValueError):
        base64_bytes("not!base64")
    with pytest.raises(ValueError):
        base64_bytes("ABC")  # length not a multiple of 4


def test_raw_bytes_dtype_name_validator() -> None:
    from zarr_metadata.v3.data_type.raw import raw_bytes_dtype_name

    assert raw_bytes_dtype_name("r8") == "r8"
    assert raw_bytes_dtype_name("r16") == "r16"
    assert raw_bytes_dtype_name("r256") == "r256"

    with pytest.raises(ValueError):
        raw_bytes_dtype_name("r3")  # not a multiple of 8
    with pytest.raises(ValueError):
        raw_bytes_dtype_name("r0")  # zero not allowed
    with pytest.raises(ValueError):
        raw_bytes_dtype_name("R8")  # uppercase R not accepted
    with pytest.raises(ValueError):
        raw_bytes_dtype_name("8")  # missing prefix
