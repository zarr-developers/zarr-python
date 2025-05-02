from __future__ import annotations

import base64
import math
import re
import sys
from typing import TYPE_CHECKING, Any, get_args

import numpy as np
import pytest

from zarr.core.dtype.common import Endianness, JSONFloat, SpecialFloats
from zarr.core.dtype.npy.common import (
    EndiannessNumpy,
    bytes_from_json,
    bytes_to_json,
    check_json_float,
    check_json_float_v2,
    check_json_float_v3,
    check_json_int,
    complex_float_to_json,
    complex_float_to_json_v2,
    complex_float_to_json_v3,
    endianness_from_numpy_str,
    endianness_to_numpy_str,
    float_from_json,
    float_from_json_v2,
    float_from_json_v3,
    float_to_json_v2,
    float_to_json_v3,
)

if TYPE_CHECKING:
    from zarr.core.common import ZarrFormat


def nan_equal(a: object, b: object) -> bool:
    """
    Convenience function for equality comparison between two values ``a`` and ``b``, that might both
    be NaN. Returns True if both ``a`` and ``b`` are NaN, otherwise returns a == b
    """
    if math.isnan(a) and math.isnan(b):  # type: ignore[arg-type]
        return True
    return a == b


json_float_v2: list[tuple[JSONFloat, float | np.floating[Any]]] = [
    ("Infinity", float("inf")),
    ("Infinity", np.inf),
    ("-Infinity", float("-inf")),
    ("-Infinity", -np.inf),
    ("NaN", float("nan")),
    ("NaN", np.nan),
    (1.0, 1.0),
]

# exactly the same as v2, for now, until we get support for the special NaN encoding defined in the
# v3 spec
json_float_v3: list[tuple[JSONFloat, float | np.floating[Any]]] = [
    ("Infinity", float("inf")),
    ("Infinity", np.inf),
    ("-Infinity", float("-inf")),
    ("-Infinity", -np.inf),
    ("NaN", float("nan")),
    ("NaN", np.nan),
    (1.0, 1.0),
]


@pytest.mark.parametrize(
    ("data", "expected"),
    [(">", "big"), ("<", "little"), ("=", sys.byteorder), ("|", None), ("err", "")],
)
def test_endianness_from_numpy_str(data: str, expected: str | None) -> None:
    """
    Test that endianness_from_numpy_str correctly converts a numpy str literal to a human-readable literal value.
    This test also checks that an invalid string input raises a ``ValueError``
    """
    if data in get_args(EndiannessNumpy):
        assert endianness_from_numpy_str(data) == expected  # type: ignore[arg-type]
    else:
        msg = f"Invalid endianness: {data!r}. Expected one of {get_args(EndiannessNumpy)}"
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
    if data in get_args(Endianness) + (None,):
        assert endianness_to_numpy_str(data) == expected  # type: ignore[arg-type]
    else:
        msg = f"Invalid endianness: {data!r}. Expected one of {get_args(Endianness)}"
        with pytest.raises(ValueError, match=re.escape(msg)):
            endianness_to_numpy_str(data)  # type: ignore[arg-type]


@pytest.mark.parametrize(("data", "expected"), json_float_v2 + [("SHOULD_ERR", "")])
def test_float_from_json_v2(data: JSONFloat | str, expected: float | str) -> None:
    """
    Test that float_from_json_v2 correctly converts a JSON string representation of a float to a float.
    This test also checks that an invalid string input raises a ``ValueError``
    """
    if data in get_args(SpecialFloats) or isinstance(data, float):
        assert nan_equal(float_from_json_v2(data), expected)  # type: ignore[arg-type]
    else:
        msg = f"could not convert string to float: {data!r}"
        with pytest.raises(ValueError, match=msg):
            float_from_json_v2(data)  # type: ignore[arg-type]


@pytest.mark.parametrize(("data", "expected"), json_float_v3 + [("SHOULD_ERR", "")])
def test_float_from_json_v3(data: JSONFloat | str, expected: float | str) -> None:
    """
    Test that float_from_json_v3 correctly converts a JSON string representation of a float to a float.
    This test also checks that an invalid string input raises a ``ValueError``
    """
    if data in get_args(SpecialFloats) or isinstance(data, float):
        assert nan_equal(float_from_json_v3(data), expected)  # type: ignore[arg-type]
    else:
        msg = f"could not convert string to float: {data!r}"
        with pytest.raises(ValueError, match=msg):
            float_from_json_v3(data)  # type: ignore[arg-type]


@pytest.mark.parametrize(("data", "expected"), json_float_v2)
def test_float_from_json(data: JSONFloat, expected: float | str, zarr_format: ZarrFormat) -> None:
    """
    Test that float_from_json_v3 correctly converts a JSON string representation of a float to a float.
    This test also checks that an invalid string input raises a ``ValueError``
    """
    observed = float_from_json(data, zarr_format=zarr_format)
    if zarr_format == 2:
        expected = float_from_json_v2(data)
    else:
        expected = float_from_json_v3(data)
    assert nan_equal(observed, expected)


# note the order of parameters relative to the order of the parametrized variable.
@pytest.mark.parametrize(("expected", "data"), json_float_v2)
def test_float_to_json_v2(data: float | np.floating[Any], expected: JSONFloat) -> None:
    """
    Test that floats are JSON-encoded properly for zarr v2
    """
    observed = float_to_json_v2(data)
    assert observed == expected


# note the order of parameters relative to the order of the parametrized variable.
@pytest.mark.parametrize(("expected", "data"), json_float_v3)
def test_float_to_json_v3(data: float | np.floating[Any], expected: JSONFloat) -> None:
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
@pytest.mark.parametrize(("json_expected", "float_data"), json_float_v2)
def test_complex_to_json_v2(float_data: float | np.floating[Any], json_expected: JSONFloat) -> None:
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
@pytest.mark.parametrize(("json_expected", "float_data"), json_float_v3)
def test_complex_to_json_v3(float_data: float | np.floating[Any], json_expected: JSONFloat) -> None:
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


@pytest.mark.parametrize(("json_expected", "float_data"), json_float_v3)
def test_complex_float_to_json(
    float_data: float | np.floating[Any], json_expected: JSONFloat, zarr_format: ZarrFormat
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
    assert complex_float_to_json(cplx, zarr_format=zarr_format) == (json_expected, json_expected)
    assert complex_float_to_json(cplx_npy, zarr_format=zarr_format) == (
        json_expected,
        json_expected,
    )


check_json_float_cases = get_args(SpecialFloats) + (1.0, 2)


@pytest.mark.parametrize("data", check_json_float_cases)
def test_check_json_float_v2_valid(data: JSONFloat | int) -> None:
    assert check_json_float_v2(data)


def test_check_json_float_v2_invalid() -> None:
    assert not check_json_float_v2("invalid")


@pytest.mark.parametrize("data", check_json_float_cases)
def test_check_json_float_v3_valid(data: JSONFloat | int) -> None:
    assert check_json_float_v3(data)


def test_check_json_float_v3_invalid() -> None:
    assert not check_json_float_v3("invalid")


@pytest.mark.parametrize("data", check_json_float_cases)
def test_check_json_float(data: JSONFloat | int, zarr_format: ZarrFormat) -> None:
    observed = check_json_float(data, zarr_format=zarr_format)
    if zarr_format == 2:
        expected = check_json_float_v2(data)
    else:
        expected = check_json_float_v3(data)
    assert observed == expected


def test_check_json_int() -> None:
    assert check_json_int(0)
    assert not check_json_int(1.0)
