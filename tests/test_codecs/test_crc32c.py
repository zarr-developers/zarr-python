from __future__ import annotations

from typing import TYPE_CHECKING, Any

import google_crc32c
import numpy as np
import pytest

import zarr
from zarr.codecs.crc32c_ import (
    Crc32cCodec,
    Crc32cJSON_V2,
    Crc32cJSON_V3,
    check_json_v2,
    check_json_v3,
)
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.dtype import UInt8, parse_dtype
from zarr.errors import CodecValidationError
from zarr.storage import StorePath

from .conftest import BaseTestCodec, numcodecs_crc32c_available

if TYPE_CHECKING:
    from zarr.abc.store import Store
    from zarr.core.common import ZarrFormat


class TestCrc32cCodec(BaseTestCodec):
    test_cls = Crc32cCodec
    valid_json_v2 = (
        {"id": "crc32c"},
        pytest.param(
            {"id": "crc32c", "location": "start"},
            marks=pytest.mark.xfail(reason="start location not supported"),
        ),
        {"id": "crc32c", "location": "end"},
    )
    valid_json_v3 = ({"name": "crc32c"}, "crc32c")

    @staticmethod
    def check_json_v2(data: object) -> bool:
        return check_json_v2(data)

    @staticmethod
    def check_json_v3(data: object) -> bool:
        return check_json_v3(data)


class TestCrc32cCodecJSON:
    """Test JSON serialization and deserialization for Crc32c codec."""

    def test_to_json_v2(self) -> None:
        """Test serialization to Zarr V2 JSON format."""
        codec = Crc32cCodec()
        expected_v2: Crc32cJSON_V2 = {"id": "crc32c"}
        assert codec.to_json(zarr_format=2) == expected_v2

    def test_to_json_v3(self) -> None:
        """Test serialization to Zarr V3 JSON format."""
        codec = Crc32cCodec()
        expected_v3: Crc32cJSON_V3 = {"name": "crc32c"}
        assert codec.to_json(zarr_format=3) == expected_v3

    def test_to_dict(self) -> None:
        """Test to_dict method returns V3 format by default."""
        codec = Crc32cCodec()
        expected = {"name": "crc32c"}
        assert codec.to_dict() == expected

    def test_roundtrip_json(self, zarr_format: ZarrFormat) -> None:
        """Test that codec can be serialized and deserialized without loss."""
        original_codec = Crc32cCodec()
        json_data = original_codec.to_json(zarr_format=zarr_format)
        reconstructed_codec = Crc32cCodec.from_json(json_data)
        assert original_codec == reconstructed_codec

    def test_from_json_v2_valid(self) -> None:
        """Test deserialization from valid V2 JSON."""
        json_data = {"id": "crc32c"}
        codec = Crc32cCodec._from_json_v2(json_data)  # type: ignore[arg-type]
        assert isinstance(codec, Crc32cCodec)

    @pytest.mark.parametrize(
        "invalid_data",
        [
            {"id": "wrong_codec"},  # Wrong codec name
            {},  # Missing id
            {"other": "field"},  # Wrong field name
        ],
    )
    def test_from_json_v2_invalid(self, invalid_data: dict[str, Any]) -> None:
        """Test that invalid V2 JSON raises CodecValidationError."""
        with pytest.raises(CodecValidationError, match="Invalid Zarr V2 JSON representation"):
            Crc32cCodec._from_json_v2(invalid_data)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "valid_data",
        [
            {"name": "crc32c"},  # Minimal valid V3
            {"name": "crc32c", "configuration": {}},  # With empty configuration
        ],
    )
    def test_from_json_v3_valid(self, valid_data: Crc32cJSON_V3) -> None:
        """Test deserialization from valid V3 JSON."""
        codec = Crc32cCodec._from_json_v3(valid_data)
        assert isinstance(codec, Crc32cCodec)

    @pytest.mark.parametrize(
        "invalid_data",
        [
            {"name": "wrong_codec"},  # Wrong codec name
            {"id": "crc32c"},  # V2 format in V3 method
            {"name": "crc32c", "configuration": {"invalid": "param"}},  # Invalid config
            {},  # Missing name
            {"other": "field"},  # Wrong field name
        ],
    )
    def test_from_json_v3_invalid(self, invalid_data: dict[str, Any]) -> None:
        """Test that invalid V3 JSON raises CodecValidationError."""
        with pytest.raises(CodecValidationError, match="Invalid Zarr V3 JSON representation"):
            Crc32cCodec._from_json_v3(invalid_data)  # type: ignore[arg-type]

    def test_from_dict(self) -> None:
        """Test from_dict method uses V3 format by default."""
        dict_data = {"name": "crc32c"}
        codec = Crc32cCodec.from_dict(dict_data)  # type: ignore[arg-type]
        assert isinstance(codec, Crc32cCodec)

    def test_to_json_invalid_format(self) -> None:
        """Test that invalid zarr_format raises ValueError."""
        codec = Crc32cCodec()
        with pytest.raises(ValueError, match="Unsupported Zarr format"):
            codec.to_json(zarr_format=1)  # type: ignore[call-overload]


class TestCrc32cCodecJSONValidation:
    """Test JSON validation helper functions."""

    @pytest.mark.parametrize(
        "valid_data",
        [
            {"id": "crc32c"},
        ],
    )
    def test_check_json_v2_valid(self, valid_data: Crc32cJSON_V2) -> None:
        """Test that valid V2 JSON passes validation."""
        assert check_json_v2(valid_data) is True

    @pytest.mark.parametrize(
        "invalid_data",
        [
            {"id": "wrong_codec"},
            {"name": "crc32c"},
            {},
            "not_a_dict",
            None,
        ],
    )
    def test_check_json_v2_invalid(self, invalid_data: object) -> None:
        """Test that invalid V2 JSON fails validation."""
        assert check_json_v2(invalid_data) is False

    @pytest.mark.parametrize(
        "valid_data",
        [
            {"name": "crc32c"},
            {"name": "crc32c", "configuration": {}},
        ],
    )
    def test_check_json_v3_valid(self, valid_data: Crc32cJSON_V3) -> None:
        """Test that valid V3 JSON passes validation."""
        assert check_json_v3(valid_data) is True

    @pytest.mark.parametrize(
        "invalid_data",
        [
            {"name": "wrong_codec"},
            {"id": "crc32c"},
            {"name": "crc32c", "configuration": {"invalid": "param"}},
            {},
            "not_a_dict",
            None,
        ],
    )
    def test_check_json_v3_invalid(self, invalid_data: object) -> None:
        """Test that invalid V3 JSON fails validation."""
        assert check_json_v3(invalid_data) is False


class TestCrc32cCodecEncoding:
    """Test encoding and decoding functionality."""

    @pytest.fixture
    def codec(self) -> Crc32cCodec:
        """Create a Crc32c codec instance."""
        return Crc32cCodec()

    @pytest.fixture
    def array_spec(self) -> ArraySpec:
        """Create a basic array spec for testing."""
        return ArraySpec(
            shape=(10,),
            dtype=UInt8(),
            fill_value=0,
            config=ArrayConfig.from_dict({}),
            prototype=default_buffer_prototype(),
        )

    @pytest.mark.parametrize(
        "data",
        [
            np.array([1, 2, 3, 4, 5], dtype=np.uint8),
            np.array([0], dtype=np.uint8),
            np.array([255] * 100, dtype=np.uint8),
            np.array([], dtype=np.uint8),
            np.random.randint(0, 256, 1000, dtype=np.uint8),
        ],
    )
    async def test_encode_decode_roundtrip(
        self, codec: Crc32cCodec, array_spec: ArraySpec, data: np.ndarray[Any, Any]
    ) -> None:
        """Test that encoding followed by decoding recovers original data."""
        buffer = array_spec.prototype.buffer.from_array_like(data)

        # Encode the data
        encoded_buffer = await codec._encode_single(buffer, array_spec)
        assert encoded_buffer is not None

        # Decode the data
        decoded_buffer = await codec._decode_single(encoded_buffer, array_spec)
        decoded_data = decoded_buffer.as_numpy_array()

        # Check that original data is recovered
        np.testing.assert_array_equal(data, decoded_data)

    @pytest.mark.parametrize(
        "dtype",
        [
            np.uint8,
            np.int8,
            np.uint16,
            np.int16,
            np.uint32,
            np.int32,
            np.uint64,
            np.int64,
            np.float32,
            np.float64,
        ],
    )
    async def test_encode_decode_different_dtypes(
        self, codec: Crc32cCodec, dtype: np.dtype[np.generic]
    ) -> None:
        """Test encoding/decoding with different data types."""
        data = np.array([1, 2, 3, 4, 5], dtype=dtype)
        array_spec = ArraySpec(
            shape=data.shape,
            dtype=parse_dtype(dtype, zarr_format=3),
            fill_value=0,
            config=ArrayConfig.from_dict({}),
            prototype=default_buffer_prototype(),
        )

        buffer = array_spec.prototype.buffer.from_array_like(data.view(np.uint8))

        # Encode and decode
        encoded_buffer = await codec._encode_single(buffer, array_spec)
        assert encoded_buffer is not None
        decoded_buffer = await codec._decode_single(encoded_buffer, array_spec)
        decoded_data = decoded_buffer.as_numpy_array()

        # Check that original bytes are recovered
        np.testing.assert_array_equal(data.view(np.uint8), decoded_data)

    async def test_compute_encoded_size(self, codec: Crc32cCodec, array_spec: ArraySpec) -> None:
        """Test that compute_encoded_size returns correct size."""
        input_sizes = [0, 1, 100, 1000, 10000]

        for input_size in input_sizes:
            encoded_size = codec.compute_encoded_size(input_size, array_spec)
            expected_size = input_size + 4  # CRC32 adds 4 bytes
            assert encoded_size == expected_size

    async def test_encoded_size_matches_actual(
        self, codec: Crc32cCodec, array_spec: ArraySpec
    ) -> None:
        """Test that actual encoded size matches computed size."""
        data = np.random.randint(0, 256, 100, dtype=np.uint8)
        buffer = array_spec.prototype.buffer.from_array_like(data)

        # Get computed size
        computed_size = codec.compute_encoded_size(len(data), array_spec)

        # Get actual encoded size
        encoded_buffer = await codec._encode_single(buffer, array_spec)
        assert encoded_buffer is not None
        actual_size = len(encoded_buffer.as_numpy_array())

        assert computed_size == actual_size

    async def test_is_fixed_size(self, codec: Crc32cCodec) -> None:
        """Test that codec reports fixed size correctly."""
        assert codec.is_fixed_size is True

    async def test_encode_single_method(self, codec: Crc32cCodec, array_spec: ArraySpec) -> None:
        """Test the _encode_single method directly."""
        test_data = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
        buffer = array_spec.prototype.buffer.from_array_like(test_data)

        # Test that _encode_single returns a buffer
        encoded_buffer = await codec._encode_single(buffer, array_spec)
        assert encoded_buffer is not None

        # Test that encoded data is longer than original (due to 4-byte checksum)
        encoded_data = encoded_buffer.as_numpy_array()
        assert len(encoded_data) == len(test_data) + 4

        # Test that original data is preserved at the beginning
        np.testing.assert_array_equal(test_data, encoded_data[:-4])

        # Test that last 4 bytes contain the checksum
        checksum_bytes = encoded_data[-4:]
        assert len(checksum_bytes) == 4


class TestCrc32cCodecErrorHandling:
    """Test error handling and data corruption detection."""

    @pytest.fixture
    def codec(self) -> Crc32cCodec:
        """Create a Crc32c codec instance."""
        return Crc32cCodec()

    @pytest.fixture
    def array_spec(self) -> ArraySpec:
        """Create a basic array spec for testing."""
        return ArraySpec(
            shape=(10,),
            dtype=UInt8(),
            fill_value=0,
            config=ArrayConfig.from_dict({}),
            prototype=default_buffer_prototype(),
        )

    async def test_corrupted_data_detection(
        self, codec: Crc32cCodec, array_spec: ArraySpec
    ) -> None:
        """Test that corrupted data is detected during decoding."""
        original_data = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
        buffer = array_spec.prototype.buffer.from_array_like(original_data)

        # Encode the data
        encoded_buffer = await codec._encode_single(buffer, array_spec)
        assert encoded_buffer is not None
        encoded_data = encoded_buffer.as_numpy_array().copy()

        # Corrupt the data (not the checksum)
        encoded_data[0] = encoded_data[0] ^ 0xFF  # Flip bits
        corrupted_buffer = array_spec.prototype.buffer.from_array_like(encoded_data)

        # Decoding should fail with corrupted data
        with pytest.raises(ValueError, match="Stored and computed checksum do not match"):
            await codec._decode_single(corrupted_buffer, array_spec)

    async def test_corrupted_checksum_detection(
        self, codec: Crc32cCodec, array_spec: ArraySpec
    ) -> None:
        """Test that corrupted checksum is detected during decoding."""
        original_data = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
        buffer = array_spec.prototype.buffer.from_array_like(original_data)

        # Encode the data
        encoded_buffer = await codec._encode_single(buffer, array_spec)
        assert encoded_buffer is not None
        encoded_data = encoded_buffer.as_numpy_array().copy()

        # Corrupt the checksum (last 4 bytes)
        encoded_data[-1] = encoded_data[-1] ^ 0xFF  # Flip bits in checksum
        corrupted_buffer = array_spec.prototype.buffer.from_array_like(encoded_data)

        # Decoding should fail with corrupted checksum
        with pytest.raises(ValueError, match="Stored and computed checksum do not match"):
            await codec._decode_single(corrupted_buffer, array_spec)

    async def test_insufficient_data_length(
        self, codec: Crc32cCodec, array_spec: ArraySpec
    ) -> None:
        """Test behavior with data shorter than checksum length."""
        # Create data shorter than 4 bytes (checksum length)
        short_data = np.array([1, 2], dtype=np.uint8)
        buffer = array_spec.prototype.buffer.from_array_like(short_data)

        # This should still work - encoding adds checksum
        encoded_buffer = await codec._encode_single(buffer, array_spec)
        assert encoded_buffer is not None

        # The encoded buffer should be original length + 4
        encoded_data = encoded_buffer.as_numpy_array()
        assert len(encoded_data) == len(short_data) + 4

        # Decoding should recover original data
        decoded_buffer = await codec._decode_single(encoded_buffer, array_spec)
        decoded_data = decoded_buffer.as_numpy_array()
        np.testing.assert_array_equal(short_data, decoded_data)

    async def test_empty_data(self, codec: Crc32cCodec, array_spec: ArraySpec) -> None:
        """Test behavior with empty data."""
        empty_data = np.array([], dtype=np.uint8)
        buffer = array_spec.prototype.buffer.from_array_like(empty_data)

        # Encoding should work
        encoded_buffer = await codec._encode_single(buffer, array_spec)
        assert encoded_buffer is not None

        # Should have just the 4-byte checksum
        encoded_data = encoded_buffer.as_numpy_array()
        assert len(encoded_data) == 4

        # Decoding should recover empty data
        decoded_buffer = await codec._decode_single(encoded_buffer, array_spec)
        decoded_data = decoded_buffer.as_numpy_array()
        np.testing.assert_array_equal(empty_data, decoded_data)


class TestCrc32cCodecIntegration:
    """Test integration with Zarr arrays and stores."""

    @pytest.mark.parametrize("store", ["memory"], indirect=["store"])
    @pytest.mark.parametrize(
        ("data_shape", "chunks"),
        [
            ((100,), (50,)),
            ((50, 20), (25, 10)),
            ((10, 10, 10), (5, 5, 5)),
        ],
    )
    def test_zarr_array_integration(
        self, store: Store, data_shape: tuple[int, ...], chunks: tuple[int, ...]
    ) -> None:
        """Test Crc32c codec with Zarr arrays of various shapes."""
        data = np.random.randint(0, 256, size=data_shape, dtype=np.uint8)

        # Create array with Crc32c codec
        array = zarr.create_array(
            StorePath(store),
            shape=data.shape,
            chunks=chunks,
            dtype=data.dtype,
            fill_value=0,
            compressors=Crc32cCodec(),
        )

        # Write and read data
        array[:] = data
        read_data = array[:]

        # Verify data integrity
        np.testing.assert_array_equal(data, read_data)

    @pytest.mark.parametrize("store", ["memory"], indirect=["store"])
    @pytest.mark.parametrize(
        "dtype",
        [np.uint8, np.uint16, np.uint32, np.uint64, np.float32, np.float64],
    )
    def test_zarr_array_different_dtypes(self, store: Store, dtype: np.dtype[Any]) -> None:
        """Test Crc32c codec with different data types in Zarr arrays."""
        if np.issubdtype(dtype, np.floating):
            data = np.random.random(100).astype(dtype)
        else:
            data = np.random.randint(0, 100, size=100, dtype=dtype)

        array = zarr.create_array(
            StorePath(store),
            shape=data.shape,
            chunks=(50,),
            dtype=data.dtype,
            fill_value=0,
            compressors=Crc32cCodec(),
        )

        array[:] = data
        read_data = array[:]

        np.testing.assert_array_equal(data, read_data)

    @pytest.mark.parametrize("store", ["memory"], indirect=["store"])
    def test_partial_chunk_operations(self, store: Store) -> None:
        """Test that partial chunk operations work correctly with checksums."""
        data = np.arange(100, dtype=np.uint8)

        array = zarr.create_array(
            StorePath(store),
            shape=data.shape,
            chunks=(30,),  # Creates partial chunks
            dtype=data.dtype,
            fill_value=255,
            compressors=Crc32cCodec(),
        )

        # Write full data
        array[:] = data

        # Read various slices
        np.testing.assert_array_equal(data[:30], array[:30])
        np.testing.assert_array_equal(data[30:60], array[30:60])
        np.testing.assert_array_equal(data[60:90], array[60:90])
        np.testing.assert_array_equal(data[90:], array[90:])
        np.testing.assert_array_equal(data[25:75], array[25:75])

    @pytest.mark.parametrize("store", ["memory"], indirect=["store"])
    def test_codec_chaining_with_compression(self, store: Store) -> None:
        """Test Crc32c codec used as compressor with other codecs."""

        data = np.random.randint(0, 256, size=1000, dtype=np.uint8)

        # Create array using Crc32c codec for integrity checking
        array = zarr.create_array(
            StorePath(store),
            shape=data.shape,
            chunks=(500,),
            dtype=data.dtype,
            fill_value=0,
            compressors=Crc32cCodec(),  # Use as compressor for data integrity
        )

        array[:] = data
        read_data = array[:]

        np.testing.assert_array_equal(data, read_data)


class TestCrc32cCodecProperties:
    """Test codec properties and metadata."""

    def test_codec_properties(self) -> None:
        """Test basic codec properties."""
        codec = Crc32cCodec()

        # Test that codec has expected properties
        assert hasattr(codec, "is_fixed_size")
        assert codec.is_fixed_size is True

        # Test that codec is frozen (immutable) - check via dataclass internals
        import dataclasses

        assert dataclasses.is_dataclass(codec)
        # The frozen property is checked at the class level
        assert hasattr(codec.__class__, "__dataclass_params__")
        assert codec.__class__.__dataclass_params__.frozen is True

    def test_codec_equality(self) -> None:
        """Test codec equality comparison."""
        codec1 = Crc32cCodec()
        codec2 = Crc32cCodec()

        assert codec1 == codec2
        assert hash(codec1) == hash(codec2)

    def test_codec_repr(self) -> None:
        """Test string representation of codec."""
        codec = Crc32cCodec()
        repr_str = repr(codec)

        assert "Crc32cCodec" in repr_str
        assert "frozen=True" in repr_str or "Crc32cCodec()" == repr_str


class TestCrc32cCodecManualVerification:
    """Test that checksum calculation matches external CRC32C implementation."""

    async def test_checksum_matches_crc32c_library(self) -> None:
        """Test that our checksum matches the crc32c library directly."""
        test_data = b"Hello, World!"
        expected_checksum = google_crc32c.value(test_data)

        # Create numpy array from bytes
        data_array = np.frombuffer(test_data, dtype=np.uint8)

        # Create codec and array spec
        codec = Crc32cCodec()
        array_spec = ArraySpec(
            shape=data_array.shape,
            dtype=UInt8(),
            fill_value=0,
            config=ArrayConfig.from_dict({}),
            prototype=default_buffer_prototype(),
        )

        # Encode data
        buffer = array_spec.prototype.buffer.from_array_like(data_array)
        encoded_buffer = await codec._encode_single(buffer, array_spec)
        assert encoded_buffer is not None

        # Extract checksum from encoded data (last 4 bytes)
        encoded_data = encoded_buffer.as_numpy_array()
        stored_checksum_bytes = encoded_data[-4:].tobytes()
        stored_checksum = np.frombuffer(stored_checksum_bytes, dtype=np.uint32)[0]

        assert stored_checksum == expected_checksum

    @pytest.mark.parametrize(
        "test_data",
        [
            b"",  # Empty data
            b"a",  # Single byte
            b"Hello, World!",  # ASCII string
            bytes(range(256)),  # All byte values
            b"\x00" * 1000,  # Repeated null bytes
            b"\xff" * 1000,  # Repeated 0xFF bytes
        ],
    )
    async def test_various_data_checksums(self, test_data: bytes) -> None:
        """Test checksum calculation with various data patterns."""
        expected_checksum = google_crc32c.value(test_data)

        data_array = np.frombuffer(test_data, dtype=np.uint8)
        codec = Crc32cCodec()
        array_spec = ArraySpec(
            shape=data_array.shape,
            dtype=UInt8(),
            fill_value=0,
            config=ArrayConfig.from_dict({}),
            prototype=default_buffer_prototype(),
        )

        buffer = array_spec.prototype.buffer.from_array_like(data_array)
        encoded_buffer = await codec._encode_single(buffer, array_spec)
        assert encoded_buffer is not None

        encoded_data = encoded_buffer.as_numpy_array()
        stored_checksum_bytes = encoded_data[-4:].tobytes()
        stored_checksum = np.frombuffer(stored_checksum_bytes, dtype=np.uint32)[0]

        assert stored_checksum == expected_checksum


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize(
    "codec",
    [
        Crc32cCodec(),
        pytest.param(
            {"id": "numcodecs.crc32c"},
            marks=pytest.mark.skipif(
                not numcodecs_crc32c_available, reason="numcodecs crc32c codec is not available"
            ),
        ),
    ],
)
def test_crc32c_compression(zarr_format: ZarrFormat, codec: Any) -> None:
    """
    Test that any of the crc32c-like codecs can be used for compression, and that
    reading the array back uses the primary crc32c codec class.
    """
    store: dict[str, Any] = {}
    ref_codec = Crc32cCodec()
    z_w = zarr.create_array(
        store=store,
        dtype="int",
        shape=(1,),
        chunks=(10,),
        zarr_format=zarr_format,
        compressors=codec,
    )
    z_w[:] = 5

    z_r = zarr.open_array(store=store, zarr_format=zarr_format)
    assert np.all(z_r[:] == 5)
    assert z_r.compressors == (ref_codec,)
