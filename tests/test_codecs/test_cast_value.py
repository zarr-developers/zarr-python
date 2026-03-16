"""Tests for the cast_value codec."""

from __future__ import annotations

import numpy as np
import pytest

import zarr
from zarr.codecs.bytes import BytesCodec
from zarr.codecs.cast_value import CastValue
from zarr.codecs.scale_offset import ScaleOffset
from zarr.core.dtype import get_data_type_from_json
from zarr.storage import MemoryStore


class TestCastValueCodec:
    """Tests for the cast_value codec."""

    def test_float64_to_float32(self) -> None:
        """Cast float64 to float32 and back."""
        store = MemoryStore()
        data = np.array([1.0, 2.0, 3.0], dtype="float64")
        arr = zarr.create(
            store=store,
            shape=data.shape,
            dtype=data.dtype,
            chunks=(3,),
            codecs=[CastValue(data_type="float32"), BytesCodec()],
        )
        arr[:] = data
        result = arr[:]
        np.testing.assert_allclose(result, data)  # type: ignore[arg-type]

    def test_float64_to_int32_towards_zero(self) -> None:
        """Cast float64 to int32 with towards-zero rounding."""
        store = MemoryStore()
        data = np.array([1.7, -1.7, 2.3, -2.3], dtype="float64")
        arr = zarr.create(
            store=store,
            shape=data.shape,
            dtype=data.dtype,
            chunks=(4,),
            codecs=[CastValue(data_type="int32", rounding="towards-zero"), BytesCodec()],
        )
        arr[:] = data
        result = arr[:]
        # After encoding to int32 with towards-zero: [1, -1, 2, -2]
        # After decoding back to float64: [1.0, -1.0, 2.0, -2.0]
        np.testing.assert_array_equal(result, [1.0, -1.0, 2.0, -2.0])

    def test_float64_to_uint8_clamp(self) -> None:
        """Cast float64 to uint8 with clamping out-of-range values."""
        store = MemoryStore()
        data = np.array([0.0, 128.0, 300.0, -10.0], dtype="float64")
        arr = zarr.create(
            store=store,
            shape=data.shape,
            dtype=data.dtype,
            chunks=(4,),
            codecs=[
                CastValue(data_type="uint8", rounding="nearest-even", out_of_range="clamp"),
                BytesCodec(),
            ],
        )
        arr[:] = data
        result = arr[:]
        np.testing.assert_array_equal(result, [0.0, 128.0, 255.0, 0.0])

    def test_float64_to_int8_wrap(self) -> None:
        """Cast float64 to int8 with wrapping for out-of-range values."""
        store = MemoryStore()
        data = np.array([200.0], dtype="float64")
        arr = zarr.create(
            store=store,
            shape=data.shape,
            dtype=data.dtype,
            chunks=(1,),
            codecs=[
                CastValue(data_type="int8", rounding="nearest-even", out_of_range="wrap"),
                BytesCodec(),
            ],
        )
        arr[:] = data
        result = arr[:]
        # 200 wraps in int8 range [-128, 127]: (200 - (-128)) % 256 + (-128) = 328 % 256 - 128 = 72 - 128 = -56
        # Decode: -56 cast back to float64 = -56.0
        np.testing.assert_array_equal(result, [-56.0])

    def test_nan_to_integer_without_scalar_map_errors(self) -> None:
        """NaN cast to integer without scalar_map should raise."""
        store = MemoryStore()
        data = np.array([float("nan")], dtype="float64")
        arr = zarr.create(
            store=store,
            shape=data.shape,
            dtype=data.dtype,
            chunks=(1,),
            codecs=[CastValue(data_type="uint8", out_of_range="clamp"), BytesCodec()],
        )
        with pytest.raises(ValueError, match="Cannot cast NaN"):
            arr[:] = data

    def test_nan_scalar_map(self) -> None:
        """NaN should be mapped via scalar_map when provided."""
        store = MemoryStore()
        data = np.array([1.0, float("nan"), 3.0], dtype="float64")
        arr = zarr.create(
            store=store,
            shape=data.shape,
            dtype=data.dtype,
            chunks=(3,),
            codecs=[
                CastValue(
                    data_type="uint8",
                    out_of_range="clamp",
                    scalar_map={  # type: ignore[arg-type]
                        "encode": [["NaN", 0]],
                        "decode": [[0, "NaN"]],
                    },
                ),
                BytesCodec(),
            ],
        )
        arr[:] = data
        result = arr[:]
        assert result[0] == 1.0  # type: ignore[index]
        assert np.isnan(result[1])  # type: ignore[index]
        assert result[2] == 3.0  # type: ignore[index]

    def test_hex_nan_scalar_map(self) -> None:
        """Hex-encoded NaN values in scalar_map should round-trip correctly.

        The hex string encoding is used for preserving specific NaN payloads
        per the Zarr v3 spec.
        """
        import struct

        # 0x7fc00001 is a NaN with a non-default payload in float32
        hex_nan = "0x7fc00001"
        nan_bytes = bytes.fromhex("7fc00001")
        nan_f32 = np.float32(struct.unpack(">f", nan_bytes)[0])
        assert np.isnan(nan_f32)

        store = MemoryStore()
        data = np.array([1.0, nan_f32, 3.0], dtype="float32")
        arr = zarr.create(
            store=store,
            shape=data.shape,
            dtype="float32",
            chunks=(3,),
            codecs=[
                CastValue(
                    data_type="uint8",
                    out_of_range="clamp",
                    scalar_map={  # type: ignore[arg-type]
                        "encode": [[hex_nan, 255]],
                        "decode": [[255, hex_nan]],
                    },
                ),
                BytesCodec(),
            ],
        )
        arr[:] = data
        result = arr[:]
        assert result[0] == np.float32(1.0)  # type: ignore[index]
        assert np.isnan(result[1])  # type: ignore[index]
        assert result[2] == np.float32(3.0)  # type: ignore[index]

        # Verify the NaN payload is preserved by checking the bit pattern
        result_bytes = struct.pack(">f", result[1])  # type: ignore[index]
        assert result_bytes == nan_bytes

    def test_int64_to_float64_precision_loss_rejected(self) -> None:
        """Casting int64 to float64 is rejected because float64 cannot
        exactly represent all int64 values.

        float64 has a 52-bit mantissa, so it can only exactly represent
        integers up to 2**52. int64.max is 2**63 - 1, far exceeding this.
        """
        store = MemoryStore()
        with pytest.raises(ValueError, match="may silently lose precision"):
            zarr.create(
                store=store,
                shape=(1,),
                dtype="int64",
                chunks=(1,),
                codecs=[
                    CastValue(data_type="float64"),
                    BytesCodec(),
                ],
            )

    def test_int32_to_float64_ok(self) -> None:
        """Casting int32 to float64 is safe because float64 has enough
        mantissa bits (52) to exactly represent all int32 values (up to 2**31 - 1)."""
        store = MemoryStore()
        data = np.array([np.iinfo(np.int32).max, np.iinfo(np.int32).min], dtype="int32")
        arr = zarr.create(
            store=store,
            shape=data.shape,
            dtype="int32",
            chunks=(2,),
            codecs=[
                CastValue(data_type="float64"),
                BytesCodec(),
            ],
        )
        arr[:] = data
        result = arr[:]
        np.testing.assert_array_equal(result, data)

    def test_rounding_nearest_even(self) -> None:
        """nearest-even rounding: 0.5 rounds to 0, 1.5 rounds to 2."""
        store = MemoryStore()
        data = np.array([0.5, 1.5, 2.5, 3.5], dtype="float64")
        arr = zarr.create(
            store=store,
            shape=data.shape,
            dtype=data.dtype,
            chunks=(4,),
            codecs=[
                CastValue(data_type="int32", rounding="nearest-even"),
                BytesCodec(),
            ],
        )
        arr[:] = data
        result = arr[:]
        np.testing.assert_array_equal(result, [0.0, 2.0, 2.0, 4.0])

    def test_rounding_towards_positive(self) -> None:
        """towards-positive rounds up (ceil)."""
        store = MemoryStore()
        data = np.array([1.1, -1.1, 1.9, -1.9], dtype="float64")
        arr = zarr.create(
            store=store,
            shape=data.shape,
            dtype=data.dtype,
            chunks=(4,),
            codecs=[
                CastValue(data_type="int32", rounding="towards-positive"),
                BytesCodec(),
            ],
        )
        arr[:] = data
        result = arr[:]
        np.testing.assert_array_equal(result, [2.0, -1.0, 2.0, -1.0])

    def test_rounding_towards_negative(self) -> None:
        """towards-negative rounds down (floor)."""
        store = MemoryStore()
        data = np.array([1.1, -1.1, 1.9, -1.9], dtype="float64")
        arr = zarr.create(
            store=store,
            shape=data.shape,
            dtype=data.dtype,
            chunks=(4,),
            codecs=[
                CastValue(data_type="int32", rounding="towards-negative"),
                BytesCodec(),
            ],
        )
        arr[:] = data
        result = arr[:]
        np.testing.assert_array_equal(result, [1.0, -2.0, 1.0, -2.0])

    def test_rounding_nearest_away(self) -> None:
        """nearest-away rounds 0.5 away from zero."""
        store = MemoryStore()
        data = np.array([0.5, 1.5, -0.5, -1.5], dtype="float64")
        arr = zarr.create(
            store=store,
            shape=data.shape,
            dtype=data.dtype,
            chunks=(4,),
            codecs=[
                CastValue(data_type="int32", rounding="nearest-away"),
                BytesCodec(),
            ],
        )
        arr[:] = data
        result = arr[:]
        np.testing.assert_array_equal(result, [1.0, 2.0, -1.0, -2.0])

    def test_out_of_range_errors_by_default(self) -> None:
        """Without out_of_range, values outside target range should error."""
        store = MemoryStore()
        data = np.array([300.0], dtype="float64")
        arr = zarr.create(
            store=store,
            shape=data.shape,
            dtype=data.dtype,
            chunks=(1,),
            codecs=[CastValue(data_type="uint8"), BytesCodec()],
        )
        with pytest.raises(ValueError, match="out of range"):
            arr[:] = data

    def test_wrap_only_valid_for_integers(self) -> None:
        """wrap should be rejected for float target types."""
        with pytest.raises(ValueError, match="only valid for integer"):
            zarr.create(
                store=MemoryStore(),
                shape=(5,),
                dtype="float64",
                chunks=(5,),
                codecs=[
                    CastValue(data_type="float32", out_of_range="wrap"),
                    BytesCodec(),
                ],
            )

    def test_validate_rejects_complex_source(self) -> None:
        """Validate should reject complex source dtype."""
        with pytest.raises(ValueError, match="only supports integer and floating-point"):
            zarr.create(
                store=MemoryStore(),
                shape=(5,),
                dtype="complex128",
                chunks=(5,),
                codecs=[CastValue(data_type="float64"), BytesCodec()],
            )

    def test_int32_to_int16_clamp(self) -> None:
        """Integer-to-integer cast with clamping."""
        store = MemoryStore()
        data = np.array([0, 100, 40000, -40000], dtype="int32")
        arr = zarr.create(
            store=store,
            shape=data.shape,
            dtype=data.dtype,
            chunks=(4,),
            codecs=[
                CastValue(data_type="int16", out_of_range="clamp"),
                BytesCodec(),
            ],
        )
        arr[:] = data
        result = arr[:]
        np.testing.assert_array_equal(result, [0, 100, 32767, -32768])

    def test_to_dict(self) -> None:
        """Serialization to dict."""
        codec = CastValue(
            data_type="uint8",
            rounding="towards-zero",
            out_of_range="clamp",
            scalar_map={"encode": [["NaN", 0]], "decode": [[0, "NaN"]]},  # type: ignore[arg-type]
        )
        d = codec.to_dict()
        assert d == {
            "name": "cast_value",
            "configuration": {
                "data_type": "uint8",
                "rounding": "towards-zero",
                "out_of_range": "clamp",
                "scalar_map": {"encode": [["NaN", 0]], "decode": [[0, "NaN"]]},
            },
        }

    def test_to_dict_minimal(self) -> None:
        """Only required fields in dict when defaults are used."""
        codec = CastValue(data_type="float32")
        d = codec.to_dict()
        assert d == {"name": "cast_value", "configuration": {"data_type": "float32"}}

    def test_from_dict(self) -> None:
        """Deserialization from dict."""
        codec = CastValue.from_dict(
            {
                "name": "cast_value",
                "configuration": {
                    "data_type": "uint8",
                    "rounding": "towards-zero",
                    "out_of_range": "clamp",
                },
            }
        )
        assert codec.dtype == get_data_type_from_json("uint8", zarr_format=3)
        assert codec.rounding == "towards-zero"
        assert codec.out_of_range == "clamp"

    def test_roundtrip_json(self) -> None:
        """to_dict -> from_dict should preserve all parameters."""
        original = CastValue(
            data_type="int16",
            rounding="towards-negative",
            out_of_range="clamp",
            scalar_map={"encode": [["NaN", 0]]},  # type: ignore[arg-type]
        )
        restored = CastValue.from_dict(original.to_dict())
        assert restored.dtype == original.dtype
        assert restored.rounding == original.rounding
        assert restored.out_of_range == original.out_of_range
        assert restored.scalar_map == original.scalar_map

    def test_fill_value_cast(self) -> None:
        """Fill value should be cast to the target dtype."""
        store = MemoryStore()
        arr = zarr.create(
            store=store,
            shape=(5,),
            dtype="float64",
            chunks=(5,),
            fill_value=42.0,
            codecs=[CastValue(data_type="int32"), BytesCodec()],
        )
        result = arr[:]
        np.testing.assert_array_equal(result, np.full(5, 42.0))

    def test_computed_encoded_size(self) -> None:
        """Encoded size should reflect the target dtype's item size."""
        codec = CastValue(data_type="uint8")
        from zarr.core.array_spec import ArrayConfig, ArraySpec
        from zarr.core.buffer.cpu import buffer_prototype
        from zarr.core.dtype import parse_dtype

        spec = ArraySpec(
            shape=(10,),
            dtype=parse_dtype("float64", zarr_format=3),
            fill_value=0.0,
            config=ArrayConfig.from_dict({}),
            prototype=buffer_prototype,
        )
        # 10 float64 elements = 80 bytes input, 10 uint8 elements = 10 bytes output
        assert codec.compute_encoded_size(80, spec) == 10


class TestScaleOffsetAndCastValueCombined:
    """Tests for the combined scale_offset + cast_value codec pipeline."""

    def test_float64_to_uint8_roundtrip(self) -> None:
        """Typical usage: float64 -> scale_offset -> cast_value(uint8) -> bytes."""
        store = MemoryStore()
        # Data in range [0, 25.5] maps to [0, 255] with scale=10
        data = np.array([0.0, 1.0, 2.5, 10.0, 25.5], dtype="float64")
        arr = zarr.create(
            store=store,
            shape=data.shape,
            dtype=data.dtype,
            chunks=(5,),
            codecs=[
                ScaleOffset(offset=0, scale=10),
                CastValue(data_type="uint8", out_of_range="clamp"),
                BytesCodec(),
            ],
        )
        arr[:] = data
        result = arr[:]
        np.testing.assert_allclose(result, data, atol=0.1)  # type: ignore[arg-type]

    def test_temperature_storage_pattern(self) -> None:
        """Realistic pattern: store temperature data as uint8.

        Temperature range: -10°C to 45°C
        Encode: (temp - (-10)) * (255/55) = (temp + 10) * 4.636...
        Use offset=-10, scale=255/55
        """
        store = MemoryStore()
        offset = -10.0
        scale = 255.0 / 55.0
        data = np.array([-10.0, 0.0, 20.0, 37.5, 45.0], dtype="float64")
        arr = zarr.create(
            store=store,
            shape=data.shape,
            dtype=data.dtype,
            chunks=(5,),
            codecs=[
                ScaleOffset(offset=offset, scale=scale),
                CastValue(data_type="uint8", out_of_range="clamp"),
                BytesCodec(),
            ],
        )
        arr[:] = data
        result = arr[:]
        # Precision limited by uint8 quantization (~0.22°C step)
        np.testing.assert_allclose(result, data, atol=0.25)  # type: ignore[arg-type]

    def test_nan_handling_pipeline(self) -> None:
        """NaN values should be handled via scalar_map in cast_value."""
        store = MemoryStore()
        data = np.array([1.0, float("nan"), 3.0], dtype="float64")
        arr = zarr.create(
            store=store,
            shape=data.shape,
            dtype=data.dtype,
            chunks=(3,),
            fill_value=float("nan"),
            codecs=[
                ScaleOffset(offset=0, scale=1),
                CastValue(
                    data_type="uint8",
                    out_of_range="clamp",
                    scalar_map={  # type: ignore[arg-type]
                        "encode": [["NaN", 0]],
                        "decode": [[0, "NaN"]],
                    },
                ),
                BytesCodec(),
            ],
        )
        arr[:] = data
        result = arr[:]
        assert result[0] == 1.0  # type: ignore[index]
        assert np.isnan(result[1])  # type: ignore[index]
        assert result[2] == 3.0  # type: ignore[index]

    def test_metadata_persistence(self) -> None:
        """Array metadata should be correctly persisted and reloaded."""
        store = MemoryStore()
        zarr.create(
            store=store,
            shape=(10,),
            dtype="float64",
            chunks=(10,),
            codecs=[
                ScaleOffset(offset=5, scale=0.5),
                CastValue(data_type="int16", out_of_range="clamp"),
                BytesCodec(),
            ],
        )
        # Reopen from same store
        arr2 = zarr.open_array(store, mode="r")
        assert arr2.dtype == np.dtype("float64")
        assert arr2.shape == (10,)
