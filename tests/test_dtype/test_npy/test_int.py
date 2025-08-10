from __future__ import annotations

import numpy as np

from tests.test_dtype.test_wrapper import BaseTestZDType
from zarr.core.dtype.npy.int import Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64


class TestInt8(BaseTestZDType):
    test_cls = Int8
    scalar_type = np.int8
    valid_dtype = (np.dtype(np.int8),)
    invalid_dtype = (
        np.dtype(np.int16),
        np.dtype(np.uint16),
        np.dtype(np.float64),
    )
    valid_json_v2 = ({"name": "|i1", "object_codec_id": None},)
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
    invalid_scalar_params = ((Int8(), {"set!"}), (Int8(), ("tuple",)))
    item_size_params = (Int8(),)


class TestInt16(BaseTestZDType):
    test_cls = Int16
    scalar_type = np.int16
    valid_dtype = (np.dtype(">i2"), np.dtype("<i2"))
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.uint16),
        np.dtype(np.float64),
    )
    valid_json_v2 = (
        {"name": ">i2", "object_codec_id": None},
        {"name": "<i2", "object_codec_id": None},
    )
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
    invalid_scalar_params = ((Int16(), {"set!"}), (Int16(), ("tuple",)))
    item_size_params = (Int16(),)


class TestInt32(BaseTestZDType):
    test_cls = Int32
    scalar_type = np.int32
    # The behavior of some tests associated with this class variable are
    # order-dependent -- np.dtype('i') correctly fails certain tests only if it's not
    # in the last position of the tuple. I have no idea how this is possible!
    valid_dtype = (np.dtype("i"), np.dtype(">i4"), np.dtype("<i4"))
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.uint16),
        np.dtype(np.float64),
    )
    valid_json_v2 = (
        {"name": ">i4", "object_codec_id": None},
        {"name": "<i4", "object_codec_id": None},
    )
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
    invalid_scalar_params = ((Int32(), {"set!"}), (Int32(), ("tuple",)))
    item_size_params = (Int32(),)


class TestInt64(BaseTestZDType):
    test_cls = Int64
    scalar_type = np.int64
    valid_dtype = (np.dtype(">i8"), np.dtype("<i8"))
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.uint16),
        np.dtype(np.float64),
    )
    valid_json_v2 = (
        {"name": ">i8", "object_codec_id": None},
        {"name": "<i8", "object_codec_id": None},
    )
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
    invalid_scalar_params = ((Int64(), {"set!"}), (Int64(), ("tuple",)))
    item_size_params = (Int64(),)


class TestUInt8(BaseTestZDType):
    test_cls = UInt8
    scalar_type = np.uint8
    valid_dtype = (np.dtype(np.uint8),)
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.int16),
        np.dtype(np.float64),
    )
    valid_json_v2 = ({"name": "|u1", "object_codec_id": None},)
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
    invalid_scalar_params = ((UInt8(), {"set!"}), (UInt8(), ("tuple",)))
    item_size_params = (UInt8(),)


class TestUInt16(BaseTestZDType):
    test_cls = UInt16
    scalar_type = np.uint16
    valid_dtype = (np.dtype(">u2"), np.dtype("<u2"))
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.int16),
        np.dtype(np.float64),
    )
    valid_json_v2 = (
        {"name": ">u2", "object_codec_id": None},
        {"name": "<u2", "object_codec_id": None},
    )
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
    invalid_scalar_params = ((UInt16(), {"set!"}), (UInt16(), ("tuple",)))
    item_size_params = (UInt16(),)


class TestUInt32(BaseTestZDType):
    test_cls = UInt32
    scalar_type = np.uint32
    valid_dtype = (np.dtype(">u4"), np.dtype("<u4"))
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.int16),
        np.dtype(np.float64),
    )
    valid_json_v2 = (
        {"name": ">u4", "object_codec_id": None},
        {"name": "<u4", "object_codec_id": None},
    )
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
    invalid_scalar_params = ((UInt32(), {"set!"}), (UInt32(), ("tuple",)))
    item_size_params = (UInt32(),)


class TestUInt64(BaseTestZDType):
    test_cls = UInt64
    scalar_type = np.uint64
    valid_dtype = (np.dtype(">u8"), np.dtype("<u8"))
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.int16),
        np.dtype(np.float64),
    )
    valid_json_v2 = (
        {"name": ">u8", "object_codec_id": None},
        {"name": "<u8", "object_codec_id": None},
    )
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
    invalid_scalar_params = ((UInt64(), {"set!"}), (UInt64(), ("tuple",)))
    item_size_params = (UInt64(),)
