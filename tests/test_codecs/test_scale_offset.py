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
    """Integer scale/offset arithmetic preserves the array dtype via floor division."""
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
    # offset=1, scale=2: encode=(x-1)*2, decode=x//2+1
    result = arr[:]
    expected = ((data - 1) * 2) // 2 + 1
    np.testing.assert_array_equal(result, expected.astype("int8"))
