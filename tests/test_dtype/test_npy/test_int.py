from __future__ import annotations

import numpy as np

from tests.test_dtype.test_wrapper import _TestZDType
from zarr.core.dtype.npy.int import Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64


class TestInt8(_TestZDType):
    test_cls = Int8
    scalar_type = np.int8
    valid_dtype = (np.dtype(np.int8),)
    invalid_dtype = (
        np.dtype(np.int16),
        np.dtype(np.uint16),
        np.dtype(np.float64),
    )
    valid_json_v2 = ("|i1",)
    valid_json_v3 = ("int8",)
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

    scalar_v2_params = ((Int8(), 1), (Int8(), -1))
    scalar_v3_params = ((Int8(), 1), (Int8(), -1))
    cast_value_params = (
        (Int8(), 1, np.int8(1)),
        (Int8(), -1, np.int8(-1)),
    )


class TestInt16(_TestZDType):
    test_cls = Int16
    scalar_type = np.int16
    valid_dtype = (np.dtype(">i2"), np.dtype("<i2"))
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.uint16),
        np.dtype(np.float64),
    )
    valid_json_v2 = (">i2", "<i2")
    valid_json_v3 = ("int16",)
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

    scalar_v2_params = ((Int16(), 1), (Int16(), -1))
    scalar_v3_params = ((Int16(), 1), (Int16(), -1))
    cast_value_params = (
        (Int16(), 1, np.int16(1)),
        (Int16(), -1, np.int16(-1)),
    )


class TestInt32(_TestZDType):
    test_cls = Int32
    scalar_type = np.int32
    valid_dtype = (np.dtype(">i4"), np.dtype("<i4"))
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.uint16),
        np.dtype(np.float64),
    )
    valid_json_v2 = (">i4", "<i4")
    valid_json_v3 = ("int32",)
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

    scalar_v2_params = ((Int32(), 1), (Int32(), -1))
    scalar_v3_params = ((Int32(), 1), (Int32(), -1))
    cast_value_params = (
        (Int32(), 1, np.int32(1)),
        (Int32(), -1, np.int32(-1)),
    )


class TestInt64(_TestZDType):
    test_cls = Int64
    scalar_type = np.int64
    valid_dtype = (np.dtype(">i8"), np.dtype("<i8"))
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.uint16),
        np.dtype(np.float64),
    )
    valid_json_v2 = (">i8", "<i8")
    valid_json_v3 = ("int64",)
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

    scalar_v2_params = ((Int64(), 1), (Int64(), -1))
    scalar_v3_params = ((Int64(), 1), (Int64(), -1))
    cast_value_params = (
        (Int64(), 1, np.int64(1)),
        (Int64(), -1, np.int64(-1)),
    )


class TestUInt8(_TestZDType):
    test_cls = UInt8
    scalar_type = np.uint8
    valid_dtype = (np.dtype(np.uint8),)
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.int16),
        np.dtype(np.float64),
    )
    valid_json_v2 = ("|u1",)
    valid_json_v3 = ("uint8",)
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

    scalar_v2_params = ((UInt8(), 1), (UInt8(), 0))
    scalar_v3_params = ((UInt8(), 1), (UInt8(), 0))
    cast_value_params = (
        (UInt8(), 1, np.uint8(1)),
        (UInt8(), 0, np.uint8(0)),
    )


class TestUInt16(_TestZDType):
    test_cls = UInt16
    scalar_type = np.uint16
    valid_dtype = (np.dtype(">u2"), np.dtype("<u2"))
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.int16),
        np.dtype(np.float64),
    )
    valid_json_v2 = (">u2", "<u2")
    valid_json_v3 = ("uint16",)
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

    scalar_v2_params = ((UInt16(), 1), (UInt16(), 0))
    scalar_v3_params = ((UInt16(), 1), (UInt16(), 0))
    cast_value_params = (
        (UInt16(), 1, np.uint16(1)),
        (UInt16(), 0, np.uint16(0)),
    )


class TestUInt32(_TestZDType):
    test_cls = UInt32
    scalar_type = np.uint32
    valid_dtype = (np.dtype(">u4"), np.dtype("<u4"))
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.int16),
        np.dtype(np.float64),
    )
    valid_json_v2 = (">u4", "<u4")
    valid_json_v3 = ("uint32",)
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

    scalar_v2_params = ((UInt32(), 1), (UInt32(), 0))
    scalar_v3_params = ((UInt32(), 1), (UInt32(), 0))
    cast_value_params = (
        (UInt32(), 1, np.uint32(1)),
        (UInt32(), 0, np.uint32(0)),
    )


class TestUInt64(_TestZDType):
    test_cls = UInt64
    scalar_type = np.uint64
    valid_dtype = (np.dtype(">u8"), np.dtype("<u8"))
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.int16),
        np.dtype(np.float64),
    )
    valid_json_v2 = (">u8", "<u8")
    valid_json_v3 = ("uint64",)
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

    scalar_v2_params = ((UInt64(), 1), (UInt64(), 0))
    scalar_v3_params = ((UInt64(), 1), (UInt64(), 0))
    cast_value_params = (
        (UInt64(), 1, np.uint64(1)),
        (UInt64(), 0, np.uint64(0)),
    )
