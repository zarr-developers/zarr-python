"""Tests for the scale_offset codec."""

from __future__ import annotations

import numpy as np
import pytest

import zarr
from zarr.codecs.bytes import BytesCodec
from zarr.codecs.scale_offset import ScaleOffset
from zarr.storage import MemoryStore


class TestScaleOffsetCodec:
    """Tests for the scale_offset codec."""

    def test_identity(self) -> None:
        """Default parameters (offset=0, scale=1) should be a no-op."""
        store = MemoryStore()
        data = np.arange(20, dtype="float64").reshape(4, 5)
        arr = zarr.create(
            store=store,
            shape=data.shape,
            dtype=data.dtype,
            chunks=(4, 5),
            codecs=[ScaleOffset(), BytesCodec()],
        )
        arr[:] = data
        np.testing.assert_array_equal(arr[:], data)

    def test_encode_decode_float64(self) -> None:
        """Encode/decode round-trip with float64 data."""
        store = MemoryStore()
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype="float64")
        arr = zarr.create(
            store=store,
            shape=data.shape,
            dtype=data.dtype,
            chunks=(5,),
            codecs=[ScaleOffset(offset=10, scale=0.1), BytesCodec()],
        )
        arr[:] = data
        result = arr[:]
        np.testing.assert_allclose(result, data, rtol=1e-10)

    def test_encode_decode_float32(self) -> None:
        """Round-trip with float32 data."""
        store = MemoryStore()
        data = np.array([1.0, 2.0, 3.0], dtype="float32")
        arr = zarr.create(
            store=store,
            shape=data.shape,
            dtype=data.dtype,
            chunks=(3,),
            codecs=[ScaleOffset(offset=1, scale=2), BytesCodec()],
        )
        arr[:] = data
        result = arr[:]
        np.testing.assert_allclose(result, data, rtol=1e-6)

    def test_encode_decode_integer(self) -> None:
        """Round-trip with integer data (uses integer arithmetic semantics)."""
        store = MemoryStore()
        data = np.array([10, 20, 30, 40, 50], dtype="int32")
        arr = zarr.create(
            store=store,
            shape=data.shape,
            dtype=data.dtype,
            chunks=(5,),
            codecs=[ScaleOffset(offset=10, scale=1), BytesCodec()],
        )
        arr[:] = data
        result = arr[:]
        np.testing.assert_array_equal(result, data)

    def test_offset_only(self) -> None:
        """Test with only offset (scale=1)."""
        store = MemoryStore()
        data = np.array([100.0, 200.0, 300.0], dtype="float64")
        arr = zarr.create(
            store=store,
            shape=data.shape,
            dtype=data.dtype,
            chunks=(3,),
            codecs=[ScaleOffset(offset=100), BytesCodec()],
        )
        arr[:] = data
        np.testing.assert_allclose(arr[:], data)

    def test_scale_only(self) -> None:
        """Test with only scale (offset=0)."""
        store = MemoryStore()
        data = np.array([1.0, 2.0, 3.0], dtype="float64")
        arr = zarr.create(
            store=store,
            shape=data.shape,
            dtype=data.dtype,
            chunks=(3,),
            codecs=[ScaleOffset(scale=10), BytesCodec()],
        )
        arr[:] = data
        np.testing.assert_allclose(arr[:], data)

    def test_fill_value_transformed(self) -> None:
        """Fill value should be transformed through the codec."""
        store = MemoryStore()
        arr = zarr.create(
            store=store,
            shape=(5,),
            dtype="float64",
            chunks=(5,),
            fill_value=100.0,
            codecs=[ScaleOffset(offset=10, scale=2), BytesCodec()],
        )
        # Without writing, reading should return the fill value
        result = arr[:]
        np.testing.assert_allclose(result, np.full(5, 100.0))

    def test_validate_rejects_complex(self) -> None:
        """Validate should reject complex dtypes."""
        with pytest.raises(ValueError, match="only supports integer and floating-point"):
            zarr.create(
                store=MemoryStore(),
                shape=(5,),
                dtype="complex128",
                chunks=(5,),
                codecs=[ScaleOffset(offset=1, scale=2), BytesCodec()],
            )

    def test_to_dict_no_config(self) -> None:
        """Default codec should serialize without configuration."""
        codec = ScaleOffset()
        assert codec.to_dict() == {"name": "scale_offset"}

    def test_to_dict_with_config(self) -> None:
        """Non-default codec should include configuration."""
        codec = ScaleOffset(offset=5, scale=0.1)
        d = codec.to_dict()
        assert d == {"name": "scale_offset", "configuration": {"offset": 5, "scale": 0.1}}

    def test_to_dict_offset_only(self) -> None:
        """Only offset in config when scale is default."""
        codec = ScaleOffset(offset=5)
        d = codec.to_dict()
        assert d == {"name": "scale_offset", "configuration": {"offset": 5}}

    def test_from_dict_no_config(self) -> None:
        """Parse codec from JSON with no configuration."""
        codec = ScaleOffset.from_dict({"name": "scale_offset"})
        assert codec.offset == 0
        assert codec.scale == 1

    def test_from_dict_with_config(self) -> None:
        """Parse codec from JSON with configuration."""
        codec = ScaleOffset.from_dict(
            {"name": "scale_offset", "configuration": {"offset": 5, "scale": 0.1}}
        )
        assert codec.offset == 5
        assert codec.scale == 0.1

    def test_roundtrip_json(self) -> None:
        """to_dict -> from_dict should preserve parameters."""
        original = ScaleOffset(offset=3.14, scale=2.71)
        restored = ScaleOffset.from_dict(original.to_dict())
        assert restored.offset == original.offset
        assert restored.scale == original.scale
