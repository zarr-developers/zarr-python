from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from zarr.core.metadata.dtype import (
    Bool,
    Complex64,
    Complex128,
    DataTypeRegistry,
    DateTime64,
    DTypeWrapper,
    Float16,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    StaticByteString,
    StaticRawBytes,
    StaticUnicodeString,
    Structured,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    VariableLengthString,
)

_NUMPY_SUPPORTS_VLEN_STRING = hasattr(np.dtypes, "StringDType")
if _NUMPY_SUPPORTS_VLEN_STRING:
    VLEN_STRING_DTYPE = np.dtypes.StringDType()
    VLEN_STRING_CODE = "T"
else:
    VLEN_STRING_DTYPE = np.dtypes.ObjectDType()
    VLEN_STRING_CODE = "O"


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
        (StaticUnicodeString, "U"),
        (StaticByteString, "S"),
        (StaticRawBytes, "V"),
        (VariableLengthString, VLEN_STRING_CODE),
        (Structured, np.dtype([("a", np.float64), ("b", np.int8)])),
        (DateTime64, "datetime64[s]"),
    ],
)
def test_wrap(wrapper_cls: type[DTypeWrapper[Any, Any]], np_dtype: np.dtype | str) -> None:
    """
    Test that the wrapper class has the correct dtype class bound to the dtype_cls variable
    Test that the ``wrap`` method produces an instance of the wrapper class
    Test that the ``unwrap`` method returns the original dtype
    """
    dt = np.dtype(np_dtype)
    assert wrapper_cls.dtype_cls is type(dt)
    wrapped = wrapper_cls.wrap(dt)

    with pytest.raises(TypeError, match="Invalid dtype"):
        wrapper_cls.wrap("not a dtype")

    assert isinstance(wrapped, wrapper_cls)
    assert wrapped.unwrap() == dt


def test_registry_match() -> None:
    """
    Test that registering a dtype in a data type registry works
    Test that match_dtype resolves a numpy dtype into the stored dtype
    Test that match_dtype raises an error if the dtype is not registered
    """
    local_registry = DataTypeRegistry()
    local_registry.register(Bool)
    assert isinstance(local_registry.match_dtype(np.dtype("bool")), Bool)
    outside_dtype = "int8"
    with pytest.raises(
        ValueError, match=f"No data type wrapper found that matches {outside_dtype}"
    ):
        local_registry.match_dtype(np.dtype(outside_dtype))


# start writing new tests here


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
        (StaticByteString(length=3), np.bytes_(b"")),
        (StaticRawBytes(length=3), np.void(b"")),
        (StaticUnicodeString(length=3), np.str_("")),
        (
            Structured(fields=(("a", Float64(), 0), ("b", Int8(), 8))),
            np.array([0], dtype=[("a", np.float64), ("b", np.int8)])[0],
        ),
        (VariableLengthString(), ""),
        (DateTime64(unit="s"), np.datetime64("NaT")),
    ],
)
def test_default_value(wrapper: type[DTypeWrapper[Any, Any]], expected_default: Any) -> None:
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
        (StaticByteString(length=4), np.bytes_(b"test"), "dGVzdA=="),
        (StaticRawBytes(length=4), np.void(b"test"), "dGVzdA=="),
        (StaticUnicodeString(length=4), np.str_("test"), "test"),
        (VariableLengthString(), "test", "test"),
        (DateTime64(unit="s"), np.datetime64("2021-01-01T00:00:00", "s"), 1609459200),
    ],
)
def test_to_json_value_v2(
    wrapper: type[DTypeWrapper[Any, Any]], input_value: Any, expected_json: Any
) -> None:
    """
    Test the to_json_value method for each dtype wrapper for zarr v2
    """
    assert wrapper.to_json_value(input_value, zarr_format=2) == expected_json


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
        (StaticByteString(length=4), "dGVzdA==", np.bytes_(b"test")),
        (StaticRawBytes(length=4), "dGVzdA==", np.void(b"test")),
        (StaticUnicodeString(length=4), "test", np.str_("test")),
        (VariableLengthString(), "test", "test"),
        (DateTime64(unit="s"), 1609459200, np.datetime64("2021-01-01T00:00:00", "s")),
    ],
)
def test_from_json_value(
    wrapper: type[DTypeWrapper[Any, Any]], json_value: Any, expected_value: Any
) -> None:
    """
    Test the from_json_value method for each dtype wrapper.
    """
    assert wrapper.from_json_value(json_value, zarr_format=2) == expected_value
