from __future__ import annotations

from typing import TYPE_CHECKING, Any, get_args

from zarr.core.dtype import (
    DTYPE,
    Bool,
    Complex64,
    Complex128,
    DateTime64,
    FixedLengthAscii,
    FixedLengthBytes,
    FixedLengthUnicode,
    Float16,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    Structured,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    VariableLengthString,
    ZDType,
)

from .conftest import zdtype_examples

if TYPE_CHECKING:
    from zarr.core.common import ZarrFormat
    from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar

import numpy as np
import pytest

from zarr.core.dtype.common import DataTypeValidationError

_NUMPY_SUPPORTS_VLEN_STRING = hasattr(np.dtypes, "StringDType")
VLEN_STRING_DTYPE: np.dtypes.StringDType | np.dtypes.ObjectDType
if _NUMPY_SUPPORTS_VLEN_STRING:
    VLEN_STRING_DTYPE = np.dtypes.StringDType()
    VLEN_STRING_CODE = "T"
else:
    VLEN_STRING_DTYPE = np.dtypes.ObjectDType()
    VLEN_STRING_CODE = "O"


def test_zdtype_examples() -> None:
    """
    Test that all the elements of the exported union type DTYPE have an example in the variable
    zdtype_examples, which we use for testing.

    If this test fails, that means that either there is a data type that does not have an example,
    or there is a data type that is missing from the DTYPE union type.
    """
    assert set(map(type, zdtype_examples)) == set(get_args(DTYPE))


@pytest.mark.parametrize(
    ("wrapper_cls", "np_dtype"),
    [
        (Bool, "bool"),
        (Int8, "int8"),
        (Int16, "int16"),
        (Int32, "int32"),
        (Int64, "int64"),
        (UInt8, "uint8"),
        (UInt16, "uint16"),
        (UInt32, "uint32"),
        (UInt64, "uint64"),
        (Float32, "float32"),
        (Float64, "float64"),
        (Complex64, "complex64"),
        (Complex128, "complex128"),
        (FixedLengthUnicode, "U"),
        (FixedLengthAscii, "S"),
        (FixedLengthBytes, "V"),
        (VariableLengthString, VLEN_STRING_CODE),
        (Structured, np.dtype([("a", np.float64), ("b", np.int8)])),
        (DateTime64, "datetime64[s]"),
    ],
)
def test_wrap(wrapper_cls: type[ZDType[Any, Any]], np_dtype: np.dtype[np.generic] | str) -> None:
    """
    Test that the wrapper class has the correct dtype class bound to the dtype_cls variable
    Test that the ``wrap`` method produces an instance of the wrapper class
    Test that the ``unwrap`` method returns the original dtype
    """
    dt = np.dtype(np_dtype)
    assert wrapper_cls.dtype_cls is type(dt)
    wrapped = wrapper_cls.from_dtype(dt)

    with pytest.raises(DataTypeValidationError, match="Invalid dtype"):
        wrapper_cls.from_dtype("not a dtype")  # type: ignore[arg-type]
    assert isinstance(wrapped, wrapper_cls)
    assert wrapped.to_dtype() == dt


@pytest.mark.parametrize("zdtype", zdtype_examples)
def test_to_json_roundtrip(zdtype: ZDType[Any, Any], zarr_format: ZarrFormat) -> None:
    """
    Test that a zdtype instance can round-trip through its JSON form
    """
    as_dict = zdtype.to_json(zarr_format=zarr_format)
    assert zdtype.from_json(as_dict, zarr_format=zarr_format) == zdtype


@pytest.mark.parametrize(
    ("wrapper", "expected_default"),
    [
        (Bool(), np.False_),
        (Int8(), np.int8(0)),
        (UInt8(), np.uint8(0)),
        (Int16(), np.int16(0)),
        (UInt16(), np.uint16(0)),
        (Int32(), np.int32(0)),
        (UInt32(), np.uint32(0)),
        (Int64(), np.int64(0)),
        (UInt64(), np.uint64(0)),
        (Float16(), np.float16(0)),
        (Float32(), np.float32(0)),
        (Float64(), np.float64(0)),
        (Complex64(), np.complex64(0)),
        (Complex128(), np.complex128(0)),
        (FixedLengthAscii(length=3), np.bytes_(b"")),
        (FixedLengthBytes(length=3), np.void(b"\x00\x00\x00")),
        (FixedLengthUnicode(length=3), np.str_("")),
        (
            Structured(fields=(("a", Float64()), ("b", Int8()))),
            np.array([0], dtype=[("a", np.float64), ("b", np.int8)])[0],
        ),
        (VariableLengthString(), ""),
        (DateTime64(unit="s"), np.datetime64("NaT")),
    ],
)
def test_default_value(wrapper: ZDType[Any, Any], expected_default: Any) -> None:
    """
    Test that the default_value method is correctly set for each dtype wrapper.
    """
    if isinstance(wrapper, DateTime64):
        assert np.isnan(wrapper.default_value())
    else:
        assert wrapper.default_value() == expected_default


@pytest.mark.parametrize(
    ("wrapper", "input_value", "expected_json"),
    [
        (Bool(), np.bool_(True), True),
        (Int8(), np.int8(42), 42),
        (UInt8(), np.uint8(42), 42),
        (Int16(), np.int16(42), 42),
        (UInt16(), np.uint16(42), 42),
        (Int32(), np.int32(42), 42),
        (UInt32(), np.uint32(42), 42),
        (Int64(), np.int64(42), 42),
        (UInt64(), np.uint64(42), 42),
        (Float16(), np.float16(42.0), 42.0),
        (Float32(), np.float32(42.0), 42.0),
        (Float64(), np.float64(42.0), 42.0),
        (Complex64(), np.complex64(42.0 + 1.0j), (42.0, 1.0)),
        (Complex128(), np.complex128(42.0 + 1.0j), (42.0, 1.0)),
        (FixedLengthAscii(length=4), np.bytes_(b"test"), "dGVzdA=="),
        (FixedLengthBytes(length=4), np.void(b"test"), "dGVzdA=="),
        (FixedLengthUnicode(length=4), np.str_("test"), "test"),
        (VariableLengthString(), "test", "test"),
        (DateTime64(unit="s"), np.datetime64("2021-01-01T00:00:00", "s"), 1609459200),
    ],
)
def test_to_json_value_v2(
    wrapper: ZDType[TBaseDType, TBaseScalar], input_value: Any, expected_json: Any
) -> None:
    """
    Test the to_json_value method for each dtype wrapper for zarr v2
    """
    assert wrapper.to_json_value(input_value, zarr_format=2) == expected_json


# NOTE! This test is currently a direct copy of the v2 version. When or if we change JSON serialization
# in a v3-specific manner, this test must be changed.
# TODO: Apply zarr-v3-specific changes to this test as needed
@pytest.mark.parametrize(
    ("wrapper", "input_value", "expected_json"),
    [
        (Bool(), np.bool_(True), True),
        (Int8(), np.int8(42), 42),
        (UInt8(), np.uint8(42), 42),
        (Int16(), np.int16(42), 42),
        (UInt16(), np.uint16(42), 42),
        (Int32(), np.int32(42), 42),
        (UInt32(), np.uint32(42), 42),
        (Int64(), np.int64(42), 42),
        (UInt64(), np.uint64(42), 42),
        (Float16(), np.float16(42.0), 42.0),
        (Float32(), np.float32(42.0), 42.0),
        (Float64(), np.float64(42.0), 42.0),
        (Complex64(), np.complex64(42.0 + 1.0j), (42.0, 1.0)),
        (Complex128(), np.complex128(42.0 + 1.0j), (42.0, 1.0)),
        (FixedLengthAscii(length=4), np.bytes_(b"test"), "dGVzdA=="),
        (FixedLengthBytes(length=4), np.void(b"test"), "dGVzdA=="),
        (FixedLengthUnicode(length=4), np.str_("test"), "test"),
        (VariableLengthString(), "test", "test"),
        (DateTime64(unit="s"), np.datetime64("2021-01-01T00:00:00", "s"), 1609459200),
    ],
)
def test_to_json_value_v3(
    wrapper: ZDType[TBaseDType, TBaseScalar], input_value: Any, expected_json: Any
) -> None:
    """
    Test the to_json_value method for each dtype wrapper for zarr v3
    """
    assert wrapper.to_json_value(input_value, zarr_format=3) == expected_json


@pytest.mark.parametrize(
    ("wrapper", "json_value", "expected_value"),
    [
        (Bool(), True, np.bool_(True)),
        (Int8(), 42, np.int8(42)),
        (UInt8(), 42, np.uint8(42)),
        (Int16(), 42, np.int16(42)),
        (UInt16(), 42, np.uint16(42)),
        (Int32(), 42, np.int32(42)),
        (UInt32(), 42, np.uint32(42)),
        (Int64(), 42, np.int64(42)),
        (UInt64(), 42, np.uint64(42)),
        (Float16(), 42.0, np.float16(42.0)),
        (Float32(), 42.0, np.float32(42.0)),
        (Float64(), 42.0, np.float64(42.0)),
        (Complex64(), (42.0, 1.0), np.complex64(42.0 + 1.0j)),
        (Complex128(), (42.0, 1.0), np.complex128(42.0 + 1.0j)),
        (FixedLengthAscii(length=4), "dGVzdA==", np.bytes_(b"test")),
        (FixedLengthBytes(length=4), "dGVzdA==", np.void(b"test")),
        (FixedLengthUnicode(length=4), "test", np.str_("test")),
        (VariableLengthString(), "test", "test"),
        (DateTime64(unit="s"), 1609459200, np.datetime64("2021-01-01T00:00:00", "s")),
    ],
)
def test_from_json_value(
    wrapper: ZDType[TBaseDType, TBaseScalar], json_value: Any, expected_value: Any
) -> None:
    """
    Test the from_json_value method for each dtype wrapper.
    """
    assert wrapper.from_json_value(json_value, zarr_format=2) == expected_value
