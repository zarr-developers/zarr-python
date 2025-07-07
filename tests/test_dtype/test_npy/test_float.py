from __future__ import annotations

import numpy as np

from tests.test_dtype.test_wrapper import BaseTestZDType
from zarr.core.dtype.npy.float import Float16, Float32, Float64


class _BaseTestFloat(BaseTestZDType):
    def scalar_equals(self, scalar1: object, scalar2: object) -> bool:
        if np.isnan(scalar1) and np.isnan(scalar2):  # type: ignore[call-overload]
            return True
        return super().scalar_equals(scalar1, scalar2)

    hex_string_params: tuple[tuple[str, float], ...] = ()

    def test_hex_encoding(self, hex_string_params: tuple[str, float]) -> None:
        """
        Test that hexadecimal strings can be read as NaN values
        """
        hex_string, expected = hex_string_params
        zdtype = self.test_cls()
        observed = zdtype.from_json_scalar(hex_string, zarr_format=3)
        assert self.scalar_equals(observed, expected)


class TestFloat16(_BaseTestFloat):
    test_cls = Float16
    valid_dtype = (np.dtype(">f2"), np.dtype("<f2"))
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.uint16),
        np.dtype(np.float32),
    )
    valid_json_v2 = (
        {"name": ">f2", "object_codec_id": None},
        {"name": "<f2", "object_codec_id": None},
    )
    valid_json_v3 = ("float16",)
    invalid_json_v2 = (
        "|f2",
        "float16",
        "|i1",
    )
    invalid_json_v3 = (
        "|f2",
        "|i1",
        {"name": "float16", "configuration": {"endianness": "little"}},
    )

    scalar_v2_params = (
        (Float16(), 1.0),
        (Float16(), -1.0),
        (Float16(), "NaN"),
        (Float16(), "Infinity"),
    )
    scalar_v3_params = (
        (Float16(), 1.0),
        (Float16(), -1.0),
        (Float16(), "NaN"),
        (Float16(), "Infinity"),
    )
    cast_value_params = (
        (Float16(), 1.0, np.float16(1.0)),
        (Float16(), -1.0, np.float16(-1.0)),
        (Float16(), "NaN", np.float16("NaN")),
    )
    invalid_scalar_params = ((Float16(), {"set!"}),)
    hex_string_params = (("0x7fc0", np.nan), ("0x7fc1", np.nan), ("0x3c00", 1.0))
    item_size_params = (Float16(),)


class TestFloat32(_BaseTestFloat):
    test_cls = Float32
    scalar_type = np.float32
    valid_dtype = (np.dtype(">f4"), np.dtype("<f4"))
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.uint16),
        np.dtype(np.float64),
    )
    valid_json_v2 = (
        {"name": ">f4", "object_codec_id": None},
        {"name": "<f4", "object_codec_id": None},
    )
    valid_json_v3 = ("float32",)
    invalid_json_v2 = (
        "|f4",
        "float32",
        "|i1",
    )
    invalid_json_v3 = (
        "|f4",
        "|i1",
        {"name": "float32", "configuration": {"endianness": "little"}},
    )

    scalar_v2_params = (
        (Float32(), 1.0),
        (Float32(), -1.0),
        (Float32(), "NaN"),
        (Float32(), "Infinity"),
    )
    scalar_v3_params = (
        (Float32(), 1.0),
        (Float32(), -1.0),
        (Float32(), "NaN"),
        (Float32(), "Infinity"),
    )

    cast_value_params = (
        (Float32(), 1.0, np.float32(1.0)),
        (Float32(), -1.0, np.float32(-1.0)),
        (Float32(), "NaN", np.float32("NaN")),
    )
    invalid_scalar_params = ((Float32(), {"set!"}),)
    hex_string_params = (("0x7fc00000", np.nan), ("0x7fc00001", np.nan), ("0x3f800000", 1.0))
    item_size_params = (Float32(),)


class TestFloat64(_BaseTestFloat):
    test_cls = Float64
    valid_dtype = (np.dtype(">f8"), np.dtype("<f8"))
    invalid_dtype = (
        np.dtype(np.int8),
        np.dtype(np.uint16),
        np.dtype(np.float32),
    )
    valid_json_v2 = (
        {"name": ">f8", "object_codec_id": None},
        {"name": "<f8", "object_codec_id": None},
    )
    valid_json_v3 = ("float64",)
    invalid_json_v2 = (
        "|f8",
        "float64",
        "|i1",
    )
    invalid_json_v3 = (
        "|f8",
        "|i1",
        {"name": "float64", "configuration": {"endianness": "little"}},
    )

    scalar_v2_params = (
        (Float64(), 1.0),
        (Float64(), -1.0),
        (Float64(), "NaN"),
        (Float64(), "Infinity"),
    )
    scalar_v3_params = (
        (Float64(), 1.0),
        (Float64(), -1.0),
        (Float64(), "NaN"),
        (Float64(), "Infinity"),
    )

    cast_value_params = (
        (Float64(), 1.0, np.float64(1.0)),
        (Float64(), -1.0, np.float64(-1.0)),
        (Float64(), "NaN", np.float64("NaN")),
    )
    invalid_scalar_params = ((Float64(), {"set!"}),)
    hex_string_params = (
        ("0x7ff8000000000000", np.nan),
        ("0x7ff8000000000001", np.nan),
        ("0x3ff0000000000000", 1.0),
    )
    item_size_params = (Float64(),)
