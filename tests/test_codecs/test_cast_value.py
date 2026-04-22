from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import zarr
from tests.test_codecs.conftest import Expect, ExpectErr
from zarr.codecs.cast_value import CastValue

try:
    import cast_value_rs  # noqa: F401

    _HAS_CAST_VALUE_RS = True
except ModuleNotFoundError:
    _HAS_CAST_VALUE_RS = False

requires_cast_value_rs = pytest.mark.skipif(
    not _HAS_CAST_VALUE_RS, reason="cast-value-rs not installed"
)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            input=CastValue(data_type="uint8"),
            expected={"name": "cast_value", "configuration": {"data_type": "uint8"}},
        ),
        Expect(
            input=CastValue(
                data_type="uint8",
                rounding="towards-zero",
                out_of_range="clamp",
                scalar_map={"encode": [("NaN", 0)]},
            ),
            expected={
                "name": "cast_value",
                "configuration": {
                    "data_type": "uint8",
                    "rounding": "towards-zero",
                    "out_of_range": "clamp",
                    "scalar_map": {"encode": [("NaN", 0)]},
                },
            },
        ),
    ],
    ids=["minimal", "full"],
)
def test_to_dict(case: Expect[CastValue, dict[str, Any]]) -> None:
    """to_dict produces the expected JSON structure."""
    assert case.input.to_dict() == case.expected


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            input={"name": "cast_value", "configuration": {"data_type": "float32"}},
            expected=("float32", "nearest-even", None),
        ),
        Expect(
            input={
                "name": "cast_value",
                "configuration": {
                    "data_type": "int16",
                    "rounding": "towards-zero",
                    "out_of_range": "clamp",
                },
            },
            expected=("int16", "towards-zero", "clamp"),
        ),
    ],
    ids=["defaults", "explicit"],
)
def test_from_dict(case: Expect[dict[str, Any], tuple[str, str, str | None]]) -> None:
    """from_dict deserializes configuration with correct values and defaults."""
    codec = CastValue.from_dict(case.input)
    dtype_name, rounding, out_of_range = case.expected
    assert codec.dtype.to_native_dtype() == np.dtype(dtype_name)
    assert codec.rounding == rounding
    assert codec.out_of_range == out_of_range


@pytest.mark.parametrize(
    "codec",
    [
        CastValue(data_type="int16", rounding="towards-zero", out_of_range="clamp"),
        CastValue(
            data_type="uint8",
            out_of_range="clamp",
            scalar_map={"encode": [("NaN", 0)], "decode": [(0, "NaN")]},
        ),
    ],
    ids=["no-scalar-map", "with-scalar-map"],
)
def test_serialization_roundtrip(codec: CastValue) -> None:
    """to_dict followed by from_dict produces an equal codec."""
    restored = CastValue.from_dict(codec.to_dict())
    assert codec == restored


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_accepts_zdtype_object() -> None:
    """data_type can be a ZDType instance, not just a JSON string."""
    from zarr.core.dtype import UInt8

    codec = CastValue(data_type=UInt8())
    assert codec.dtype.to_native_dtype() == np.dtype("uint8")


def test_construction_rejects_invalid_target_dtype() -> None:
    """Construction rejects target dtypes not in PERMITTED_DATA_TYPE_NAMES."""
    with pytest.raises(ValueError, match="Invalid target data type"):
        CastValue(data_type="complex64")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        ExpectErr(
            input={"dtype": "complex128", "target": "float64"},
            msg="only supports integer and floating-point",
            exception_cls=ValueError,
        ),
        ExpectErr(
            input={"dtype": "float32", "target": "int32", "out_of_range": "wrap"},
            msg="only valid for integer",
            exception_cls=ValueError,
        ),
    ],
    ids=["complex-source", "wrap-float-target"],
)
def test_validation_rejects_invalid(case: ExpectErr[dict[str, Any]]) -> None:
    """Invalid dtype or out_of_range combinations are rejected at array creation."""
    with pytest.raises(case.exception_cls, match=case.msg):
        zarr.create_array(
            store={},
            shape=(10,),
            dtype=case.input["dtype"],
            chunks=(10,),
            filters=[
                CastValue(
                    data_type=case.input["target"],
                    out_of_range=case.input.get("out_of_range"),
                )
            ],
            compressors=None,
            fill_value=0,
        )


def test_zero_itemsize_raises() -> None:
    """Variable-length dtypes (itemsize=0) are rejected by compute_encoded_size."""
    from zarr.core.array_spec import ArrayConfig, ArraySpec
    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.dtype.npy.string import VariableLengthUTF8

    codec = CastValue(data_type="uint8")
    spec = ArraySpec(
        shape=(10,),
        dtype=VariableLengthUTF8(),  # type: ignore[arg-type]
        fill_value="",
        config=ArrayConfig(order="C", write_empty_chunks=True),
        prototype=default_buffer_prototype(),
    )
    with pytest.raises(ValueError, match="fixed-size integer and floating-point data types"):
        codec.compute_encoded_size(100, spec)


# ---------------------------------------------------------------------------
# Encode / decode
# ---------------------------------------------------------------------------


@requires_cast_value_rs
@pytest.mark.parametrize(
    "case",
    [
        Expect(input=("float64", "float32"), expected=np.arange(50, dtype="float64")),
        Expect(input=("float32", "float64"), expected=np.arange(50, dtype="float32")),
        Expect(input=("int32", "int64"), expected=np.arange(50, dtype="int32")),
        Expect(input=("int64", "int16"), expected=np.arange(50, dtype="int64")),
        Expect(input=("float64", "int32"), expected=np.arange(50, dtype="float64")),
        Expect(input=("int32", "float64"), expected=np.arange(50, dtype="int32")),
    ],
    ids=["f64→f32", "f32→f64", "i32→i64", "i64→i16", "f64→i32", "i32→f64"],
)
def test_encode_decode_roundtrip(
    case: Expect[tuple[str, str], np.ndarray[Any, np.dtype[Any]]],
) -> None:
    """Small integer data survives encode → decode for each dtype pair."""
    import zarr

    source_dtype, target_dtype = case.input
    arr = zarr.create_array(
        store={},
        shape=(50,),
        dtype=source_dtype,
        chunks=(50,),
        filters=[CastValue(data_type=target_dtype)],
        compressors=None,
        fill_value=0,
    )
    arr[:] = case.expected
    np.testing.assert_array_equal(arr[:], case.expected)


@requires_cast_value_rs
@pytest.mark.parametrize(
    "case",
    [
        Expect(
            input=np.array([1.7, -1.7, 2.5, -2.5], dtype="float64"),
            expected=np.array([1, -1, 2, -2], dtype="float64"),
        ),
    ],
    ids=["towards-zero"],
)
def test_float_to_int_rounding(
    case: Expect[np.ndarray[Any, np.dtype[Any]], np.ndarray[Any, np.dtype[Any]]],
) -> None:
    """Fractional float values are truncated towards zero when cast to int32."""
    import zarr

    arr = zarr.create_array(
        store={},
        shape=case.input.shape,
        dtype=case.input.dtype,
        chunks=case.input.shape,
        filters=[CastValue(data_type="int32", rounding="towards-zero", out_of_range="clamp")],
        compressors=None,
        fill_value=0,
    )
    arr[:] = case.input
    np.testing.assert_array_equal(arr[:], case.expected)


@requires_cast_value_rs
@pytest.mark.parametrize(
    "case",
    [
        Expect(
            input=np.array([0, 200, -200], dtype="int32"),
            expected=np.array([0, 127, -128], dtype="int32"),
        ),
    ],
    ids=["int32→int8"],
)
def test_out_of_range_clamp(
    case: Expect[np.ndarray[Any, np.dtype[Any]], np.ndarray[Any, np.dtype[Any]]],
) -> None:
    """Values outside the int8 range are clamped to [-128, 127]."""
    import zarr

    arr = zarr.create_array(
        store={},
        shape=case.input.shape,
        dtype=case.input.dtype,
        chunks=case.input.shape,
        filters=[CastValue(data_type="int8", out_of_range="clamp")],
        compressors=None,
        fill_value=0,
    )
    arr[:] = case.input
    np.testing.assert_array_equal(arr[:], case.expected)


def test_compute_encoded_size() -> None:
    """compute_encoded_size correctly scales byte length by itemsize ratio."""
    from zarr.core.array_spec import ArrayConfig, ArraySpec
    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.dtype import get_data_type_from_json

    codec = CastValue(data_type="int16")
    spec = ArraySpec(
        shape=(10,),
        dtype=get_data_type_from_json("float64", zarr_format=3),
        fill_value=0,
        config=ArrayConfig(order="C", write_empty_chunks=True),
        prototype=default_buffer_prototype(),
    )
    # 10 float64 elements = 80 bytes -> 10 int16 elements = 20 bytes
    assert codec.compute_encoded_size(80, spec) == 20


@requires_cast_value_rs
def test_scalar_map_encode_decode_roundtrip() -> None:
    """Scalar map entries are applied during encode and decode."""
    import zarr

    data = np.array([1.0, float("nan"), 3.0], dtype="float64")
    arr = zarr.create_array(
        store={},
        shape=data.shape,
        dtype="float64",
        chunks=data.shape,
        filters=[
            CastValue(
                data_type="int32",
                rounding="nearest-even",
                out_of_range="clamp",
                scalar_map={"encode": [("NaN", -999)], "decode": [(-999, "NaN")]},
            ),
        ],
        compressors=None,
        fill_value=1,
    )
    arr[:] = data
    result = np.asarray(arr[:])
    np.testing.assert_equal(result[0], 1.0)
    np.testing.assert_equal(result[2], 3.0)
    assert np.isnan(result[1])


@pytest.mark.parametrize(
    "case",
    [
        ExpectErr(
            input={
                "dtype": "int32",
                "target": "int8",
                "scalar_map": {"encode": [("NaN", 0)]},
            },
            msg="not representable in dtype int32",
            exception_cls=ValueError,
        ),
        ExpectErr(
            input={
                "dtype": "int32",
                "target": "float64",
                "scalar_map": {"decode": [(0, "NaN")]},
            },
            msg="not representable in dtype int32",
            exception_cls=ValueError,
        ),
        ExpectErr(
            input={
                "dtype": "float64",
                "target": "int8",
                "scalar_map": {"encode": [("NaN", 999)]},
            },
            msg="not representable in dtype int8",
            exception_cls=ValueError,
        ),
        ExpectErr(
            input={
                "dtype": "float64",
                "target": "int8",
                "scalar_map": {"encode": [("NaN", 1.5)]},
            },
            msg="not representable in dtype int8",
            exception_cls=ValueError,
        ),
    ],
    ids=[
        "nan-key-for-int-source",
        "nan-value-for-int-decode-target",
        "encode-value-out-of-range",
        "encode-value-not-integer",
    ],
)
def test_scalar_map_validation_rejects_invalid(case: ExpectErr[dict[str, Any]]) -> None:
    """Invalid scalar_map entries are rejected at array creation."""
    import zarr

    with pytest.raises(case.exception_cls, match=case.msg):
        zarr.create_array(
            store={},
            shape=(10,),
            dtype=case.input["dtype"],
            chunks=(10,),
            filters=[
                CastValue(
                    data_type=case.input["target"],
                    out_of_range="clamp",
                    scalar_map=case.input["scalar_map"],
                )
            ],
            compressors=None,
            fill_value=0,
        )


@requires_cast_value_rs
def test_combined_with_scale_offset() -> None:
    """scale_offset followed by cast_value compresses float64 into int16 and round-trips."""
    import zarr
    from zarr.codecs.scale_offset import ScaleOffset

    arr = zarr.create_array(
        store={},
        shape=(100,),
        dtype="float64",
        chunks=(100,),
        filters=[
            ScaleOffset(offset=0, scale=10),
            CastValue(data_type="int16", rounding="nearest-even", out_of_range="clamp"),
        ],
        compressors=None,
        fill_value=0,
    )
    data = np.arange(100, dtype="float64") * 0.1
    arr[:] = data
    result = arr[:]
    np.testing.assert_array_almost_equal(result, data, decimal=1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "case",
    [
        Expect(
            input={"encode": [("NaN", 0)]},
            expected={"encode": {"NaN": 0}},
        ),
        Expect(
            input={"encode": [("NaN", 0)], "decode": [(0, "NaN")]},
            expected={"encode": {"NaN": 0}, "decode": {0: "NaN"}},
        ),
        Expect(
            input={"encode": {"NaN": 0}},
            expected={"encode": {"NaN": 0}},
        ),
    ],
    ids=["encode-only", "both-directions", "already-normalized"],
)
def test_parse_scalar_map(case: Expect[Any, Any]) -> None:
    from zarr.codecs.cast_value import parse_scalar_map

    assert parse_scalar_map(case.input) == case.expected
