from __future__ import annotations

from typing import Any

import numpy as np

from zarr.codecs.bytes import BytesCodec
from zarr.codecs.gzip import GzipCodec
from zarr.codecs.transpose import TransposeCodec
from zarr.codecs.zstd import ZstdCodec
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import NDBuffer, default_buffer_prototype
from zarr.core.codec_pipeline import ChunkTransform
from zarr.core.dtype import get_data_type_from_native_dtype


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
    def test_all_sync(self) -> None:
        spec = _make_array_spec((100,), np.dtype("float64"))
        chain = ChunkTransform(codecs=(BytesCodec(),), array_spec=spec)
        assert chain.all_sync is True

    def test_all_sync_with_compression(self) -> None:
        spec = _make_array_spec((100,), np.dtype("float64"))
        chain = ChunkTransform(codecs=(BytesCodec(), GzipCodec()), array_spec=spec)
        assert chain.all_sync is True

    def test_all_sync_full_chain(self) -> None:
        spec = _make_array_spec((3, 4), np.dtype("float64"))
        chain = ChunkTransform(
            codecs=(TransposeCodec(order=(1, 0)), BytesCodec(), ZstdCodec()), array_spec=spec
        )
        assert chain.all_sync is True

    def test_encode_decode_roundtrip_bytes_only(self) -> None:
        arr = np.arange(100, dtype="float64")
        spec = _make_array_spec(arr.shape, arr.dtype)
        chain = ChunkTransform(codecs=(BytesCodec(),), array_spec=spec)
        nd_buf = _make_nd_buffer(arr)

        encoded = chain.encode_chunk(nd_buf)
        assert encoded is not None
        decoded = chain.decode_chunk(encoded)
        np.testing.assert_array_equal(arr, decoded.as_numpy_array())

    def test_layers_no_aa_codecs(self) -> None:
        spec = _make_array_spec((100,), np.dtype("float64"))
        chunk = ChunkTransform(codecs=(BytesCodec(), GzipCodec()), array_spec=spec)
        assert chunk.layers == ()

    def test_layers_with_transpose(self) -> None:
        spec = _make_array_spec((3, 4), np.dtype("float64"))
        transpose = TransposeCodec(order=(1, 0))
        chunk = ChunkTransform(codecs=(transpose, BytesCodec(), ZstdCodec()), array_spec=spec)
        assert len(chunk.layers) == 1
        assert chunk.layers[0][0] is transpose
        assert chunk.layers[0][1] is spec

    def test_shape_dtype_no_aa_codecs(self) -> None:
        spec = _make_array_spec((100,), np.dtype("float64"))
        chunk = ChunkTransform(codecs=(BytesCodec(),), array_spec=spec)
        assert chunk.shape == (100,)
        assert chunk.dtype == spec.dtype

    def test_shape_dtype_with_transpose(self) -> None:
        spec = _make_array_spec((3, 4), np.dtype("float64"))
        chunk = ChunkTransform(codecs=(TransposeCodec(order=(1, 0)), BytesCodec()), array_spec=spec)
        # After transpose (1,0), shape (3,4) becomes (4,3)
        assert chunk.shape == (4, 3)
        assert chunk.dtype == spec.dtype

    def test_encode_decode_roundtrip_with_compression(self) -> None:
        arr = np.arange(100, dtype="float64")
        spec = _make_array_spec(arr.shape, arr.dtype)
        chain = ChunkTransform(codecs=(BytesCodec(), GzipCodec(level=1)), array_spec=spec)
        nd_buf = _make_nd_buffer(arr)

        encoded = chain.encode_chunk(nd_buf)
        assert encoded is not None
        decoded = chain.decode_chunk(encoded)
        np.testing.assert_array_equal(arr, decoded.as_numpy_array())

    def test_encode_decode_roundtrip_with_transpose(self) -> None:
        arr = np.arange(12, dtype="float64").reshape(3, 4)
        spec = _make_array_spec(arr.shape, arr.dtype)
        chain = ChunkTransform(
            codecs=(TransposeCodec(order=(1, 0)), BytesCodec(), ZstdCodec(level=1)),
            array_spec=spec,
        )
        nd_buf = _make_nd_buffer(arr)

        encoded = chain.encode_chunk(nd_buf)
        assert encoded is not None
        decoded = chain.decode_chunk(encoded)
        np.testing.assert_array_equal(arr, decoded.as_numpy_array())
