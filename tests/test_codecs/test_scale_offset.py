from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.test_codecs.conftest import Expect, ExpectErr
from zarr.codecs.scale_offset import ScaleOffset

# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        Expect(input=ScaleOffset(), expected={"name": "scale_offset"}),
        Expect(
            input=ScaleOffset(offset=5),
            expected={"name": "scale_offset", "configuration": {"offset": 5}},
        ),
        Expect(
            input=ScaleOffset(scale=0.1),
            expected={"name": "scale_offset", "configuration": {"scale": 0.1}},
        ),
        Expect(
            input=ScaleOffset(offset=5, scale=0.1),
            expected={"name": "scale_offset", "configuration": {"offset": 5, "scale": 0.1}},
        ),
    ],
    ids=["default", "offset-only", "scale-only", "both"],
)
def test_to_dict(case: Expect[ScaleOffset, dict[str, Any]]) -> None:
    """to_dict produces the expected JSON structure."""
    assert case.input.to_dict() == case.expected


@pytest.mark.parametrize(
    "case",
    [
        Expect(input={"name": "scale_offset"}, expected=(0, 1)),
        Expect(
            input={"name": "scale_offset", "configuration": {"offset": 3, "scale": 2}},
            expected=(3, 2),
        ),
    ],
    ids=["no-config", "with-config"],
)
def test_from_dict(case: Expect[dict[str, Any], tuple[int | float, int | float]]) -> None:
    """from_dict deserializes configuration with correct values and defaults."""
    codec = ScaleOffset.from_dict(case.input)
    expected_offset, expected_scale = case.expected
    assert codec.offset == expected_offset
    assert codec.scale == expected_scale


def test_serialization_roundtrip() -> None:
    """to_dict followed by from_dict produces an equal codec."""
    original = ScaleOffset(offset=7, scale=0.5)
    restored = ScaleOffset.from_dict(original.to_dict())
    assert original == restored


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        ExpectErr(
            input={"offset": [1, 2]},
            msg="offset must be a number or string",
            exception_cls=TypeError,
        ),
        ExpectErr(
            input={"scale": [1, 2]}, msg="scale must be a number or string", exception_cls=TypeError
        ),
    ],
    ids=["list-offset", "list-scale"],
)
def test_construction_rejects_non_numeric(case: ExpectErr[dict[str, Any]]) -> None:
    """Non-numeric offset or scale is rejected at construction time."""
    with pytest.raises(case.exception_cls, match=case.msg):
        ScaleOffset(**case.input)


@pytest.mark.parametrize(
    "case",
    [
        Expect(input={"offset": 5, "scale": 2}, expected=(5, 2)),
        Expect(input={"offset": 0.5, "scale": 0.1}, expected=(0.5, 0.1)),
    ],
    ids=["int", "float"],
)
def test_construction_accepts_numeric(
    case: Expect[dict[str, Any], tuple[int | float, int | float]],
) -> None:
    """Integer and float values are accepted for both parameters."""
    codec = ScaleOffset(**case.input)
    assert codec.offset == case.expected[0]
    assert codec.scale == case.expected[1]


# ---------------------------------------------------------------------------
# Encode / decode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dtype", "offset", "scale"),
    [
        ("float64", 10.0, 0.1),
        ("float32", 5.0, 2.0),
        ("int32", 0, 1),
    ],
    ids=["float64", "float32", "int32-identity"],
)
def test_encode_decode_roundtrip(dtype: str, offset: float, scale: float) -> None:
    """Data survives encode → decode."""
    import zarr

    arr = zarr.create_array(
        store={},
        shape=(100,),
        dtype=dtype,
        chunks=(100,),
        filters=[ScaleOffset(offset=offset, scale=scale)],
        compressors=None,
        fill_value=0,
    )
    data = np.arange(100, dtype=dtype)
    arr[:] = data
    np.testing.assert_array_almost_equal(arr[:], data)  # type: ignore[arg-type]


def test_fill_value_transformed() -> None:
    """Fill value is transformed through the encode formula and read back correctly."""
    import zarr

    arr = zarr.create_array(
        store={},
        shape=(10,),
        dtype="float64",
        chunks=(10,),
        filters=[ScaleOffset(offset=5, scale=2)],
        compressors=None,
        fill_value=10.0,
    )
    # fill_value=10.0, encode: (10 - 5) * 2 = 10.0 stored
    # Reading back without writing should return the original fill value
    np.testing.assert_array_equal(arr[:], np.full(10, 10.0))


def test_identity_is_noop() -> None:
    """Default codec (offset=0, scale=1) is a no-op."""
    import zarr

    arr = zarr.create_array(
        store={},
        shape=(50,),
        dtype="float64",
        chunks=(50,),
        filters=[ScaleOffset()],
        compressors=None,
        fill_value=0,
    )
    data = np.arange(50, dtype="float64")
    arr[:] = data
    np.testing.assert_array_equal(arr[:], data)


def test_rejects_complex_dtype() -> None:
    """Complex dtypes are rejected at array creation time."""
    import zarr

    with pytest.raises(ValueError, match="only supports integer and floating-point"):
        zarr.create_array(
            store={},
            shape=(10,),
            dtype="complex128",
            chunks=(10,),
            filters=[ScaleOffset(offset=1, scale=2)],
            compressors=None,
            fill_value=0,
        )


def test_uint64_large_value_roundtrip() -> None:
    """uint64 values above 2**63 must survive encode+decode (spec requires uint64 support)."""
    import zarr

    arr = zarr.create_array(
        store={},
        shape=(3,),
        dtype="uint64",
        chunks=(3,),
        filters=[ScaleOffset(offset=0, scale=1)],
        compressors=None,
        fill_value=0,
    )
    # Value above int64.max (2**63 - 1) — would wrap if we used int64 as wide dtype.
    data = np.array([0, 2**63, 2**64 - 1], dtype="uint64")
    arr[:] = data
    np.testing.assert_array_equal(arr[:], data)


def test_float_nan_inf_preserved() -> None:
    """NaN and Inf are representable in float dtypes per IEEE 754 and must pass through."""
    from zarr.codecs.scale_offset import _decode, _encode

    arr = np.array([1.0, np.nan, np.inf, -np.inf], dtype="float64")
    encoded = _encode(arr, np.float64(0.0), np.float64(2.0))
    np.testing.assert_array_equal(encoded[[0]], np.array([2.0]))
    assert np.isnan(encoded[1])
    assert encoded[2] == np.inf
    assert encoded[3] == -np.inf
    decoded = _decode(encoded, np.float64(0.0), np.float64(2.0), scale_repr=2.0)
    np.testing.assert_array_equal(decoded[[0]], np.array([1.0]))
    assert np.isnan(decoded[1])


def test_uint64_encode_rejects_underflow() -> None:
    """uint64 underflow during encode raises rather than silently wrapping."""
    import zarr

    arr = zarr.create_array(
        store={},
        shape=(3,),
        dtype="uint64",
        chunks=(3,),
        filters=[ScaleOffset(offset=100, scale=1)],
        compressors=None,
        fill_value=100,
    )
    with pytest.raises(ValueError, match="outside the range of dtype uint64"):
        arr[:] = np.array([100, 50, 200], dtype="uint64")


def test_rejects_zero_scale() -> None:
    """scale=0 is rejected (destroys data and breaks decode division)."""
    import zarr

    with pytest.raises(ValueError, match="scale must be non-zero"):
        zarr.create_array(
            store={},
            shape=(10,),
            dtype="int32",
            chunks=(10,),
            filters=[ScaleOffset(offset=0, scale=0)],
            compressors=None,
            fill_value=0,
        )


@pytest.mark.parametrize(
    "case",
    [
        ExpectErr(
            input={"dtype": "int32", "offset": 1.5, "scale": 1},
            msg="offset value 1.5 is not representable",
            exception_cls=ValueError,
        ),
        ExpectErr(
            input={"dtype": "int32", "offset": 0, "scale": 0.5},
            msg="scale value 0.5 is not representable",
            exception_cls=ValueError,
        ),
        ExpectErr(
            input={"dtype": "int16", "offset": "NaN", "scale": 1},
            msg="offset value 'NaN' is not representable",
            exception_cls=ValueError,
        ),
    ],
    ids=["float-offset-for-int", "float-scale-for-int", "nan-offset-for-int"],
)
def test_rejects_unrepresentable_scale_offset(case: ExpectErr[dict[str, Any]]) -> None:
    """Scale/offset values that can't be represented in the array dtype are rejected."""
    import zarr

    with pytest.raises(case.exception_cls, match=case.msg):
        zarr.create_array(
            store={},
            shape=(10,),
            dtype=case.input["dtype"],
            chunks=(10,),
            filters=[ScaleOffset(offset=case.input["offset"], scale=case.input["scale"])],
            compressors=None,
            fill_value=0,
        )


def test_dtype_preservation() -> None:
    """Integer scale/offset arithmetic preserves the array dtype when division is exact."""
    import zarr

    arr = zarr.create_array(
        store={},
        shape=(10,),
        dtype="int8",
        chunks=(10,),
        filters=[ScaleOffset(offset=1, scale=2)],
        compressors=None,
        fill_value=0,
    )
    data = np.arange(10, dtype="int8")
    arr[:] = data
    # encode=(x-1)*2 is always divisible by scale=2, so decode is exact
    np.testing.assert_array_equal(arr[:], data)


def test_integer_decode_rejects_non_exact_division() -> None:
    """Decoding an integer array raises when the stored value isn't divisible by scale."""
    import zarr
    from zarr.storage import MemoryStore

    store = MemoryStore()
    arr = zarr.create_array(
        store=store,
        shape=(3,),
        dtype="int8",
        chunks=(3,),
        filters=[ScaleOffset(offset=0, scale=2)],
        compressors=None,
        fill_value=0,
    )
    # Write raw encoded bytes directly so we can inject a value that isn't divisible by scale.
    # Array layout: int8 [2, 3, 4]; 3 % 2 != 0, so decode must fail.
    import asyncio

    from zarr.core.buffer import default_buffer_prototype

    buf = default_buffer_prototype().buffer.from_bytes(np.array([2, 3, 4], dtype="int8").tobytes())
    asyncio.run(arr.store_path.store.set("c/0", buf))
    with pytest.raises(ValueError, match="non-zero remainder"):
        arr[:]


def test_encode_rejects_signed_integer_overflow() -> None:
    """Encoding raises when (value - offset) * scale exceeds the target integer range."""
    import zarr

    arr = zarr.create_array(
        store={},
        shape=(3,),
        dtype="int8",
        chunks=(3,),
        filters=[ScaleOffset(offset=0, scale=100)],
        compressors=None,
        fill_value=0,
    )
    # 2 * 100 = 200, outside int8 range [-128, 127]
    with pytest.raises(ValueError, match="outside the range of dtype int8"):
        arr[:] = np.array([0, 1, 2], dtype="int8")


def test_encode_rejects_unsigned_integer_underflow() -> None:
    """Encoding raises when value - offset underflows an unsigned dtype."""
    import zarr

    arr = zarr.create_array(
        store={},
        shape=(3,),
        dtype="uint8",
        chunks=(3,),
        filters=[ScaleOffset(offset=10, scale=1)],
        compressors=None,
        fill_value=10,
    )
    # 5 - 10 = -5, outside uint8 range [0, 255]
    with pytest.raises(ValueError, match="outside the range of dtype uint8"):
        arr[:] = np.array([10, 5, 20], dtype="uint8")


def test_float32_dtype_preserved() -> None:
    """float32 arrays survive encode+decode without being promoted to float64."""
    from zarr.codecs.scale_offset import _decode, _encode

    arr = np.arange(100, dtype="float32")
    offset = np.float32(5.0)
    scale = np.float32(0.25)
    encoded = _encode(arr, offset, scale)
    assert encoded.dtype == np.dtype("float32")
    decoded = _decode(encoded, offset, scale, scale_repr=0.25)
    assert decoded.dtype == np.dtype("float32")


def test_float_encode_rejects_wider_scalar() -> None:
    """A float64 scalar passed with a float32 array must not silently widen the result."""
    from zarr.codecs.scale_offset import _encode

    arr = np.arange(10, dtype="float32")
    # A numpy float64 scalar (not a Python float — NEP 50 exempts those) mixed with a
    # float32 ndarray promotes to float64. The codec must reject that.
    with pytest.raises(ValueError, match="changed dtype from float32 to float64"):
        _encode(arr, np.float64(5.0), np.float64(0.25))


def test_float_decode_rejects_wider_scalar() -> None:
    """A float64 scalar passed with a float32 array must not silently widen on decode."""
    from zarr.codecs.scale_offset import _decode

    arr = np.arange(10, dtype="float32")
    with pytest.raises(ValueError, match="changed dtype from float32 to float64"):
        _decode(arr, np.float64(5.0), np.float64(0.25), scale_repr=0.25)


def test_decode_rejects_integer_overflow_on_offset_add() -> None:
    """Decoding raises when quotient + offset overflows the target integer dtype."""
    import asyncio

    import zarr
    from zarr.core.buffer import default_buffer_prototype
    from zarr.storage import MemoryStore

    store = MemoryStore()
    arr = zarr.create_array(
        store=store,
        shape=(3,),
        dtype="int8",
        chunks=(3,),
        filters=[ScaleOffset(offset=100, scale=1)],
        compressors=None,
        fill_value=0,
    )
    # encoded=100 → decoded = 100/1 + 100 = 200, outside int8 range
    buf = default_buffer_prototype().buffer.from_bytes(
        np.array([0, 50, 100], dtype="int8").tobytes()
    )
    asyncio.run(arr.store_path.store.set("c/0", buf))
    with pytest.raises(ValueError, match="outside the range of dtype int8"):
        arr[:]
