from __future__ import annotations

import numpy as np

from tests.test_dtype.test_wrapper import _TestZDType
from zarr.core.dtype.npy.int import Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64


class TestInt8(_TestZDType):
    test_cls = Int8
    valid_dtype = (np.dtype(np.int8),)
    invalid_dtype = (
        np.dtype(np.int16),
        np.dtype(np.uint16),
        np.dtype(np.float64),
    )
    valid_json_v2 = Int8._zarr_v2_names
    valid_json_v3_cases = (Int8._zarr_v3_name,)
    invalid_json_v2 = (
        ">i1",
        "int8",
        "|f8",
    )
    invalid_json_v3 = (
        "|i1",
        "|f8",
        {"name": "int8", "configuration": {"endianness": "little"}},
    )


class TestInt16(_TestZDType):
    test_cls = Int16
    valid_dtype = (np.dtype(">i2"), np.dtype("<i2"))
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.uint16),
        np.dtype(np.float64),
    )
    valid_json_v2 = Int16._zarr_v2_names
    valid_json_v3_cases = (Int16._zarr_v3_name,)
    invalid_json_v2 = (
        "|i2",
        "int16",
        "|f8",
    )
    invalid_json_v3 = (
        "|i2",
        "|f8",
        {"name": "int16", "configuration": {"endianness": "little"}},
    )


class TestInt32(_TestZDType):
    test_cls = Int32
    valid_dtype = (np.dtype(">i4"), np.dtype("<i4"))
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.uint16),
        np.dtype(np.float64),
    )
    valid_json_v2 = Int32._zarr_v2_names
    valid_json_v3_cases = (Int32._zarr_v3_name,)
    invalid_json_v2 = (
        "|i4",
        "int32",
        "|f8",
    )
    invalid_json_v3 = (
        "|i4",
        "|f8",
        {"name": "int32", "configuration": {"endianness": "little"}},
    )


class TestInt64(_TestZDType):
    test_cls = Int64
    valid_dtype = (np.dtype(">i8"), np.dtype("<i8"))
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.uint16),
        np.dtype(np.float64),
    )
    valid_json_v2 = Int64._zarr_v2_names
    valid_json_v3_cases = (Int64._zarr_v3_name,)
    invalid_json_v2 = (
        "|i8",
        "int64",
        "|f8",
    )
    invalid_json_v3 = (
        "|i8",
        "|f8",
        {"name": "int64", "configuration": {"endianness": "little"}},
    )


class TestUInt8(_TestZDType):
    test_cls = UInt8
    valid_dtype = (np.dtype(np.uint8),)
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.int16),
        np.dtype(np.float64),
    )
    valid_json_v2 = UInt8._zarr_v2_names
    valid_json_v3_cases = (UInt8._zarr_v3_name,)
    invalid_json_v2 = (
        "|u1",
        "uint8",
        "|f8",
    )
    invalid_json_v3 = (
        "|u1",
        "|f8",
        {"name": "uint8", "configuration": {"endianness": "little"}},
    )


class TestUInt16(_TestZDType):
    test_cls = UInt16
    valid_dtype = (np.dtype(">u2"), np.dtype("<u2"))
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.int16),
        np.dtype(np.float64),
    )
    valid_json_v2 = UInt16._zarr_v2_names
    valid_json_v3_cases = (UInt16._zarr_v3_name,)
    invalid_json_v2 = (
        "|u2",
        "uint16",
        "|f8",
    )
    invalid_json_v3 = (
        "|u2",
        "|f8",
        {"name": "uint16", "configuration": {"endianness": "little"}},
    )


class TestUInt32(_TestZDType):
    test_cls = UInt32
    valid_dtype = (np.dtype(">u4"), np.dtype("<u4"))
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.int16),
        np.dtype(np.float64),
    )
    valid_json_v2 = UInt32._zarr_v2_names
    valid_json_v3_cases = (UInt32._zarr_v3_name,)
    invalid_json_v2 = (
        "|u4",
        "uint32",
        "|f8",
    )
    invalid_json_v3 = (
        "|u4",
        "|f8",
        {"name": "uint32", "configuration": {"endianness": "little"}},
    )


class TestUInt64(_TestZDType):
    test_cls = UInt64
    valid_dtype = (np.dtype(">u8"), np.dtype("<u8"))
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.int16),
        np.dtype(np.float64),
    )
    valid_json_v2 = UInt64._zarr_v2_names
    valid_json_v3_cases = (UInt64._zarr_v3_name,)
    invalid_json_v2 = (
        "|u8",
        "uint64",
        "|f8",
    )
    invalid_json_v3 = (
        "|u8",
        "|f8",
        {"name": "uint64", "configuration": {"endianness": "little"}},
    )
