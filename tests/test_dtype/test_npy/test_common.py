from __future__ import annotations

import base64
import re
import sys
from typing import TYPE_CHECKING, Any, get_args

import numpy as np
import pytest

from tests.conftest import nan_equal
from zarr.core.dtype.common import ENDIANNESS_STR, JSONFloatV2, SpecialFloatStrings
from zarr.core.dtype.npy.common import (
    NumpyEndiannessStr,
    bytes_from_json,
    bytes_to_json,
    check_json_bool,
    check_json_complex_float_v2,
    check_json_complex_float_v3,
    check_json_float_v2,
    check_json_float_v3,
    check_json_int,
    check_json_str,
    complex_float_to_json_v2,
    complex_float_to_json_v3,
    endianness_from_numpy_str,
    endianness_to_numpy_str,
    float_from_json_v2,
    float_from_json_v3,
    float_to_json_v2,
    float_to_json_v3,
)

if TYPE_CHECKING:
    from zarr.core.common import JSON, ZarrFormat


json_float_v2_roundtrip_cases: tuple[tuple[JSONFloatV2, float | np.floating[Any]], ...] = (
    ("Infinity", float("inf")),
    ("Infinity", np.inf),
    ("-Infinity", float("-inf")),
    ("-Infinity", -np.inf),
    ("NaN", float("nan")),
    ("NaN", np.nan),
    (1.0, 1.0),
)

json_float_v3_cases = json_float_v2_roundtrip_cases


@pytest.mark.parametrize(
    ("data", "expected"),
    [(">", "big"), ("<", "little"), ("=", sys.byteorder), ("|", None), ("err", "")],
)
def test_endianness_from_numpy_str(data: str, expected: str | None) -> None:
    """
    Test that endianness_from_numpy_str correctly converts a numpy str literal to a human-readable literal value.
    This test also checks that an invalid string input raises a ``ValueError``
    """
    if data in get_args(NumpyEndiannessStr):
        assert endianness_from_numpy_str(data) == expected  # type: ignore[arg-type]
    else:
        msg = f"Invalid endianness: {data!r}. Expected one of {get_args(NumpyEndiannessStr)}"
        with pytest.raises(ValueError, match=re.escape(msg)):
            endianness_from_numpy_str(data)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("data", "expected"),
    [("big", ">"), ("little", "<"), (None, "|"), ("err", "")],
)
def test_endianness_to_numpy_str(data: str | None, expected: str) -> None:
    """
    Test that endianness_to_numpy_str correctly converts a human-readable literal value to a numpy str literal.
    This test also checks that an invalid string input raises a ``ValueError``
    """
    if data in ENDIANNESS_STR:
        assert endianness_to_numpy_str(data) == expected  # type: ignore[arg-type]
    else:
        msg = f"Invalid endianness: {data!r}. Expected one of {ENDIANNESS_STR}"
        with pytest.raises(ValueError, match=re.escape(msg)):
            endianness_to_numpy_str(data)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("data", "expected"), json_float_v2_roundtrip_cases + (("SHOULD_ERR", ""),)
)
def test_float_from_json_v2(data: JSONFloatV2 | str, expected: float | str) -> None:
    """
    Test that float_from_json_v2 correctly converts a JSON string representation of a float to a float.
    This test also checks that an invalid string input raises a ``ValueError``
    """
    if data != "SHOULD_ERR":
        assert nan_equal(float_from_json_v2(data), expected)  # type: ignore[arg-type]
    else:
        msg = f"could not convert string to float: {data!r}"
        with pytest.raises(ValueError, match=msg):
            float_from_json_v2(data)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("data", "expected"), json_float_v3_cases + (("SHOULD_ERR", ""), ("0x", ""))
)
def test_float_from_json_v3(data: JSONFloatV2 | str, expected: float | str) -> None:
    """
    Test that float_from_json_v3 correctly converts a JSON string representation of a float to a float.
    This test also checks that an invalid string input raises a ``ValueError``
    """
    if data == "SHOULD_ERR":
        msg = (
            f"Invalid float value: {data!r}. Expected a string starting with the hex prefix"
            " '0x', or one of 'NaN', 'Infinity', or '-Infinity'."
        )
        with pytest.raises(ValueError, match=msg):
            float_from_json_v3(data)
    elif data == "0x":
        msg = (
            f"Invalid hexadecimal float value: {data!r}. "
            "Expected the '0x' prefix to be followed by 4, 8, or 16 numeral characters"
        )

        with pytest.raises(ValueError, match=msg):
            float_from_json_v3(data)
    else:
        assert nan_equal(float_from_json_v3(data), expected)


# note the order of parameters relative to the order of the parametrized variable.
@pytest.mark.parametrize(("expected", "data"), json_float_v2_roundtrip_cases)
def test_float_to_json_v2(data: float | np.floating[Any], expected: JSONFloatV2) -> None:
    """
    Test that floats are JSON-encoded properly for zarr v2
    """
    observed = float_to_json_v2(data)
    assert observed == expected


# note the order of parameters relative to the order of the parametrized variable.
@pytest.mark.parametrize(("expected", "data"), json_float_v3_cases)
def test_float_to_json_v3(data: float | np.floating[Any], expected: JSONFloatV2) -> None:
    """
    Test that floats are JSON-encoded properly for zarr v3
    """
    observed = float_to_json_v3(data)
    assert observed == expected


def test_bytes_from_json(zarr_format: ZarrFormat) -> None:
    """
    Test that a string is interpreted as base64-encoded bytes using the ascii alphabet.
    This test takes zarr_format as a parameter but doesn't actually do anything with it, because at
    present there is no zarr-format-specific logic in the code being tested, but such logic may
    exist in the future.
    """
    data = "\00"
    assert bytes_from_json(data, zarr_format=zarr_format) == base64.b64decode(data.encode("ascii"))


def test_bytes_to_json(zarr_format: ZarrFormat) -> None:
    """
    Test that bytes are encoded with base64 using the ascii alphabet.

    This test takes zarr_format as a parameter but doesn't actually do anything with it, because at
    present there is no zarr-format-specific logic in the code being tested, but such logic may
    exist in the future.
    """

    data = b"asdas"
    assert bytes_to_json(data, zarr_format=zarr_format) == base64.b64encode(data).decode("ascii")


# note the order of parameters relative to the order of the parametrized variable.
@pytest.mark.parametrize(("json_expected", "float_data"), json_float_v2_roundtrip_cases)
def test_complex_to_json_v2(
    float_data: float | np.floating[Any], json_expected: JSONFloatV2
) -> None:
    """
    Test that complex numbers are correctly converted to JSON in v2 format.

    This use the same test input as the float tests, but the conversion is tested
    for complex numbers with real and imaginary parts equal to the float
    values provided in the test cases.
    """
    cplx = complex(float_data, float_data)
    cplx_npy = np.complex128(cplx)
    assert complex_float_to_json_v2(cplx) == (json_expected, json_expected)
    assert complex_float_to_json_v2(cplx_npy) == (json_expected, json_expected)


# note the order of parameters relative to the order of the parametrized variable.
@pytest.mark.parametrize(("json_expected", "float_data"), json_float_v3_cases)
def test_complex_to_json_v3(
    float_data: float | np.floating[Any], json_expected: JSONFloatV2
) -> None:
    """
    Test that complex numbers are correctly converted to JSON in v3 format.

    This use the same test input as the float tests, but the conversion is tested
    for complex numbers with real and imaginary parts equal to the float
    values provided in the test cases.
    """
    cplx = complex(float_data, float_data)
    cplx_npy = np.complex128(cplx)
    assert complex_float_to_json_v3(cplx) == (json_expected, json_expected)
    assert complex_float_to_json_v3(cplx_npy) == (json_expected, json_expected)


@pytest.mark.parametrize(("json_expected", "float_data"), json_float_v3_cases)
def test_complex_float_to_json(
    float_data: float | np.floating[Any], json_expected: JSONFloatV2, zarr_format: ZarrFormat
) -> None:
    """
    Test that complex numbers are correctly converted to JSON in v2 or v3 formats, depending
    on the ``zarr_format`` keyword argument.

    This use the same test input as the float tests, but the conversion is tested
    for complex numbers with real and imaginary parts equal to the float
    values provided in the test cases.
    """

    cplx = complex(float_data, float_data)
    cplx_npy = np.complex128(cplx)
    if zarr_format == 2:
        assert complex_float_to_json_v2(cplx) == (json_expected, json_expected)
        assert complex_float_to_json_v2(cplx_npy) == (
            json_expected,
            json_expected,
        )
    elif zarr_format == 3:
        assert complex_float_to_json_v3(cplx) == (json_expected, json_expected)
        assert complex_float_to_json_v3(cplx_npy) == (
            json_expected,
            json_expected,
        )
    else:
        raise ValueError("zarr_format must be 2 or 3")  # pragma: no cover


check_json_float_cases = get_args(SpecialFloatStrings) + (1.0, 2)


@pytest.mark.parametrize("data", check_json_float_cases)
def test_check_json_float_v2_valid(data: JSONFloatV2 | int) -> None:
    assert check_json_float_v2(data)


def test_check_json_float_v2_invalid() -> None:
    assert not check_json_float_v2("invalid")


@pytest.mark.parametrize("data", check_json_float_cases)
def test_check_json_float_v3_valid(data: JSONFloatV2 | int) -> None:
    assert check_json_float_v3(data)


def test_check_json_float_v3_invalid() -> None:
    assert not check_json_float_v3("invalid")


check_json_complex_float_true_cases: tuple[list[JSONFloatV2], ...] = (
    [0.0, 1.0],
    [0.0, 1.0],
    [-1.0, "NaN"],
    ["Infinity", 1.0],
    ["Infinity", "NaN"],
)

check_json_complex_float_false_cases: tuple[object, ...] = (
    0.0,
    "foo",
    [0.0],
    [1.0, 2.0, 3.0],
    [1.0, "_infinity_"],
    {"hello": 1.0},
)


@pytest.mark.parametrize("data", check_json_complex_float_true_cases)
def test_check_json_complex_float_v2_true(data: JSON) -> None:
    assert check_json_complex_float_v2(data)


@pytest.mark.parametrize("data", check_json_complex_float_false_cases)
def test_check_json_complex_float_v2_false(data: JSON) -> None:
    assert not check_json_complex_float_v2(data)


@pytest.mark.parametrize("data", check_json_complex_float_true_cases)
def test_check_json_complex_float_v3_true(data: JSON) -> None:
    assert check_json_complex_float_v3(data)


@pytest.mark.parametrize("data", check_json_complex_float_false_cases)
def test_check_json_complex_float_v3_false(data: JSON) -> None:
    assert not check_json_complex_float_v3(data)


@pytest.mark.parametrize("data", check_json_complex_float_true_cases)
def test_check_json_complex_float_true(data: JSON, zarr_format: ZarrFormat) -> None:
    if zarr_format == 2:
        assert check_json_complex_float_v2(data)
    elif zarr_format == 3:
        assert check_json_complex_float_v3(data)
    else:
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover


@pytest.mark.parametrize("data", check_json_complex_float_false_cases)
def test_check_json_complex_float_false(data: JSON, zarr_format: ZarrFormat) -> None:
    if zarr_format == 2:
        assert not check_json_complex_float_v2(data)
    elif zarr_format == 3:
        assert not check_json_complex_float_v3(data)
    else:
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}")  # pragma: no cover


def test_check_json_int() -> None:
    assert check_json_int(0)
    assert not check_json_int(1.0)


def test_check_json_str() -> None:
    assert check_json_str("0")
    assert not check_json_str(1.0)


def test_check_json_bool() -> None:
    assert check_json_bool(True)
    assert check_json_bool(False)
    assert not check_json_bool(1.0)
    assert not check_json_bool("True")
