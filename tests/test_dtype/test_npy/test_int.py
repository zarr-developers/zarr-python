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

    scalar_v2_params = (("|i1", 1), ("|i1", -1))
    scalar_v3_params = (("int8", 1), ("int8", -1))


class TestInt16(_TestZDType):
    test_cls = Int16
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

    scalar_v2_params = (("<i2", 1), (">i2", -1))
    scalar_v3_params = (("int16", 1), ("int16", -1))


class TestInt32(_TestZDType):
    test_cls = Int32
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

    scalar_v2_params = (("<i4", 1), (">i4", -1))
    scalar_v3_params = (("int32", 1), ("int32", -1))


class TestInt64(_TestZDType):
    test_cls = Int64
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

    scalar_v2_params = (("<i8", 1), (">i8", -1))
    scalar_v3_params = (("int64", 1), ("int64", -1))


class TestUInt8(_TestZDType):
    test_cls = UInt8
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

    scalar_v2_params = (("|u1", 1), ("|u1", 0))
    scalar_v3_params = (("uint8", 1), ("uint8", 0))


class TestUInt16(_TestZDType):
    test_cls = UInt16
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

    scalar_v2_params = (("<u2", 1), (">u2", 0))
    scalar_v3_params = (("uint16", 1), ("uint16", 0))


class TestUInt32(_TestZDType):
    test_cls = UInt32
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

    scalar_v2_params = (("<u4", 1), (">u4", 0))
    scalar_v3_params = (("uint32", 1), ("uint32", 0))


class TestUInt64(_TestZDType):
    test_cls = UInt64
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

    scalar_v2_params = (("<u8", 1), (">u8", 0))
    scalar_v3_params = (("uint64", 1), ("uint64", 0))
