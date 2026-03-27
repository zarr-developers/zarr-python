from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from zarr.abc.codec import ArrayBytesCodec
from zarr.codecs.bytes import BytesCodec
from zarr.codecs.crc32c_ import Crc32cCodec
from zarr.codecs.gzip import GzipCodec
from zarr.codecs.transpose import TransposeCodec
from zarr.codecs.zstd import ZstdCodec
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import Buffer, NDBuffer, default_buffer_prototype
from zarr.core.codec_pipeline import ChunkTransform
from zarr.core.dtype import get_data_type_from_native_dtype


class AsyncOnlyCodec(ArrayBytesCodec):  # type: ignore[misc]
    """A codec that only supports async, for testing rejection of non-sync codecs."""

    is_fixed_size = True

    async def _decode_single(self, chunk_data: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
        raise NotImplementedError  # pragma: no cover

    async def _encode_single(self, chunk_data: NDBuffer, chunk_spec: ArraySpec) -> Buffer | None:
        raise NotImplementedError  # pragma: no cover

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        return input_byte_length  # pragma: no cover


def _make_array_spec(shape: tuple[int, ...], dtype: np.dtype[np.generic]) -> ArraySpec:
    zdtype = get_data_type_from_native_dtype(dtype)
    return ArraySpec(
        shape=shape,
        dtype=zdtype,
        fill_value=zdtype.cast_scalar(0),
        config=ArrayConfig(order="C", write_empty_chunks=True),
        prototype=default_buffer_prototype(),
    )


def _make_nd_buffer(arr: np.ndarray[Any, np.dtype[Any]]) -> NDBuffer:
    return default_buffer_prototype().nd_buffer.from_numpy_array(arr)


class TestChunkTransform:
    def test_construction_bytes_only(self) -> None:
        # Construction succeeds when all codecs implement SupportsSyncCodec.
        spec = _make_array_spec((100,), np.dtype("float64"))
        ChunkTransform(codecs=(BytesCodec(),), array_spec=spec)

    def test_construction_with_compression(self) -> None:
        # AB + BB codec chain where both implement SupportsSyncCodec.
        spec = _make_array_spec((100,), np.dtype("float64"))
        ChunkTransform(codecs=(BytesCodec(), GzipCodec()), array_spec=spec)

    def test_construction_full_chain(self) -> None:
        # All three codec types (AA + AB + BB), all implementing SupportsSyncCodec.
        spec = _make_array_spec((3, 4), np.dtype("float64"))
        ChunkTransform(
            codecs=(TransposeCodec(order=(1, 0)), BytesCodec(), ZstdCodec()), array_spec=spec
        )

    def test_encode_decode_roundtrip_bytes_only(self) -> None:
        # Minimal round-trip: BytesCodec serializes the array to bytes and back.
        # No compression, no AA transform.
        arr = np.arange(100, dtype="float64")
        spec = _make_array_spec(arr.shape, arr.dtype)
        chain = ChunkTransform(codecs=(BytesCodec(),), array_spec=spec)
        nd_buf = _make_nd_buffer(arr)

        encoded = chain.encode(nd_buf)
        assert encoded is not None
        decoded = chain.decode(encoded)
        np.testing.assert_array_equal(arr, decoded.as_numpy_array())

    def test_encode_decode_roundtrip_with_compression(self) -> None:
        # Round-trip with a BB codec (GzipCodec) to verify that bytes-bytes
        # compression/decompression is wired correctly.
        arr = np.arange(100, dtype="float64")
        spec = _make_array_spec(arr.shape, arr.dtype)
        chain = ChunkTransform(codecs=(BytesCodec(), GzipCodec(level=1)), array_spec=spec)
        nd_buf = _make_nd_buffer(arr)

        encoded = chain.encode(nd_buf)
        assert encoded is not None
        decoded = chain.decode(encoded)
        np.testing.assert_array_equal(arr, decoded.as_numpy_array())

    def test_encode_decode_roundtrip_with_transpose(self) -> None:
        # Full AA + AB + BB chain round-trip. Transpose permutes axes on encode,
        # then BytesCodec serializes, then ZstdCodec compresses. Decode reverses
        # all three stages. Verifies the full pipeline works end to end.
        arr = np.arange(12, dtype="float64").reshape(3, 4)
        spec = _make_array_spec(arr.shape, arr.dtype)
        chain = ChunkTransform(
            codecs=(TransposeCodec(order=(1, 0)), BytesCodec(), ZstdCodec(level=1)),
            array_spec=spec,
        )
        nd_buf = _make_nd_buffer(arr)

        encoded = chain.encode(nd_buf)
        assert encoded is not None
        decoded = chain.decode(encoded)
        np.testing.assert_array_equal(arr, decoded.as_numpy_array())

    def test_rejects_non_sync_codec(self) -> None:
        """Construction must raise TypeError when a codec lacks SupportsSyncCodec."""
        spec = _make_array_spec((100,), np.dtype("float64"))
        with pytest.raises(TypeError, match="AsyncOnlyCodec"):
            ChunkTransform(codecs=(AsyncOnlyCodec(),), array_spec=spec)

    def test_rejects_mixed_sync_and_non_sync(self) -> None:
        """Even if some codecs support sync, a single non-sync codec causes failure."""
        spec = _make_array_spec((3, 4), np.dtype("float64"))
        with pytest.raises(TypeError, match="AsyncOnlyCodec"):
            ChunkTransform(
                codecs=(TransposeCodec(order=(1, 0)), AsyncOnlyCodec()),
                array_spec=spec,
            )

    def test_compute_encoded_size_bytes_only(self) -> None:
        # BytesCodec is size-preserving: encoded size == input size.
        spec = _make_array_spec((100,), np.dtype("float64"))
        chain = ChunkTransform(codecs=(BytesCodec(),), array_spec=spec)
        assert chain.compute_encoded_size(800, spec) == 800

    def test_compute_encoded_size_with_crc32c(self) -> None:
        # Crc32cCodec appends a 4-byte checksum, so encoded size = input + 4.
        spec = _make_array_spec((100,), np.dtype("float64"))
        chain = ChunkTransform(codecs=(BytesCodec(), Crc32cCodec()), array_spec=spec)
        assert chain.compute_encoded_size(800, spec) == 804

    def test_compute_encoded_size_with_transpose(self) -> None:
        # TransposeCodec reorders axes but doesn't change the byte count.
        # Verifies that compute_encoded_size walks through AA codecs correctly.
        spec = _make_array_spec((3, 4), np.dtype("float64"))
        chain = ChunkTransform(codecs=(TransposeCodec(order=(1, 0)), BytesCodec()), array_spec=spec)
        assert chain.compute_encoded_size(96, spec) == 96

    def test_encode_returns_none_propagation(self) -> None:
        # When an AA codec returns None (signaling "this chunk is the fill value,
        # don't store it"), encode must short-circuit and return None
        # instead of passing None into the next codec.

        class NoneReturningAACodec(TransposeCodec):  # type: ignore[misc]
            """An ArrayArrayCodec that always returns None from encode."""

            def _encode_sync(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> NDBuffer | None:
                return None

        spec = _make_array_spec((3, 4), np.dtype("float64"))
        chain = ChunkTransform(
            codecs=(NoneReturningAACodec(order=(1, 0)), BytesCodec()),
            array_spec=spec,
        )
        arr = np.arange(12, dtype="float64").reshape(3, 4)
        nd_buf = _make_nd_buffer(arr)
        assert chain.encode(nd_buf) is None

    def test_encode_decode_roundtrip_with_crc32c(self) -> None:
        # Round-trip through BytesCodec + Crc32cCodec. Crc32c appends a checksum
        # on encode and verifies it on decode, so this tests that the BB codec
        # pipeline runs correctly in both directions.
        arr = np.arange(100, dtype="float64")
        spec = _make_array_spec(arr.shape, arr.dtype)
        chain = ChunkTransform(codecs=(BytesCodec(), Crc32cCodec()), array_spec=spec)
        nd_buf = _make_nd_buffer(arr)

        encoded = chain.encode(nd_buf)
        assert encoded is not None
        decoded = chain.decode(encoded)
        np.testing.assert_array_equal(arr, decoded.as_numpy_array())

    def test_encode_decode_roundtrip_int32(self) -> None:
        # Round-trip with int32 data to verify that the codec chain is not
        # float-specific. Exercises a different dtype path through BytesCodec.
        arr = np.arange(50, dtype="int32")
        spec = _make_array_spec(arr.shape, arr.dtype)
        chain = ChunkTransform(codecs=(BytesCodec(), ZstdCodec(level=1)), array_spec=spec)
        nd_buf = _make_nd_buffer(arr)

        encoded = chain.encode(nd_buf)
        assert encoded is not None
        decoded = chain.decode(encoded)
        np.testing.assert_array_equal(arr, decoded.as_numpy_array())
