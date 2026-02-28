from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from zarr.codecs.bytes import BytesCodec
from zarr.codecs.gzip import GzipCodec
from zarr.codecs.transpose import TransposeCodec
from zarr.codecs.zstd import ZstdCodec
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import NDBuffer, default_buffer_prototype
from zarr.core.codec_pipeline import (
    BatchedCodecPipeline,
    ChunkTransform,
    ReadChunkRequest,
    WriteChunkRequest,
    _choose_workers,
)
from zarr.core.dtype import get_data_type_from_native_dtype
from zarr.storage._common import StorePath
from zarr.storage._memory import MemoryStore


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


# ---------------------------------------------------------------------------
# Helpers for sync pipeline tests
# ---------------------------------------------------------------------------


def _make_pipeline(
    codecs: tuple[Any, ...],
) -> BatchedCodecPipeline:
    return BatchedCodecPipeline.from_codecs(codecs)


def _make_store_path(key: str = "chunk/0") -> StorePath:
    store = MemoryStore()
    return StorePath(store, key)


# ---------------------------------------------------------------------------
# Sync pipeline tests
# ---------------------------------------------------------------------------


class TestSyncPipeline:
    def test_supports_sync_io(self) -> None:
        pipeline = _make_pipeline((BytesCodec(),))
        assert pipeline.supports_sync_io is True

    def test_supports_sync_io_with_compression(self) -> None:
        pipeline = _make_pipeline((BytesCodec(), ZstdCodec()))
        assert pipeline.supports_sync_io is True

    def test_write_sync_read_sync_roundtrip(self) -> None:
        """Write data via write_sync, read it back via read_sync."""
        arr = np.arange(100, dtype="float64")
        spec = _make_array_spec(arr.shape, arr.dtype)
        pipeline = _make_pipeline((BytesCodec(),))
        store_path = _make_store_path()
        transform = ChunkTransform(codecs=tuple(pipeline), array_spec=spec)

        value = _make_nd_buffer(arr)

        # Write
        pipeline.write_sync(
            [
                WriteChunkRequest(
                    byte_setter=store_path,
                    transform=transform,
                    chunk_selection=(slice(None),),
                    out_selection=(slice(None),),
                    is_complete_chunk=True,
                )
            ],
            value,
        )

        # Read
        out = default_buffer_prototype().nd_buffer.create(
            shape=arr.shape,
            dtype=arr.dtype,
            order="C",
            fill_value=0,
        )
        pipeline.read_sync(
            [
                ReadChunkRequest(
                    byte_getter=store_path,
                    transform=transform,
                    chunk_selection=(slice(None),),
                    out_selection=(slice(None),),
                )
            ],
            out,
        )

        np.testing.assert_array_equal(arr, out.as_numpy_array())

    def test_write_sync_read_sync_with_compression(self) -> None:
        """Round-trip with a compression codec."""
        arr = np.arange(200, dtype="float32").reshape(10, 20)
        spec = _make_array_spec(arr.shape, arr.dtype)
        pipeline = _make_pipeline((BytesCodec(), GzipCodec(level=1)))
        store_path = _make_store_path()
        transform = ChunkTransform(codecs=tuple(pipeline), array_spec=spec)

        value = _make_nd_buffer(arr)

        pipeline.write_sync(
            [
                WriteChunkRequest(
                    byte_setter=store_path,
                    transform=transform,
                    chunk_selection=(slice(None), slice(None)),
                    out_selection=(slice(None), slice(None)),
                    is_complete_chunk=True,
                )
            ],
            value,
        )

        out = default_buffer_prototype().nd_buffer.create(
            shape=arr.shape,
            dtype=arr.dtype,
            order="C",
            fill_value=0,
        )
        pipeline.read_sync(
            [
                ReadChunkRequest(
                    byte_getter=store_path,
                    transform=transform,
                    chunk_selection=(slice(None), slice(None)),
                    out_selection=(slice(None), slice(None)),
                )
            ],
            out,
        )

        np.testing.assert_array_equal(arr, out.as_numpy_array())

    def test_write_sync_partial_chunk(self) -> None:
        """Write a partial chunk (is_complete_chunk=False), read back full chunk."""
        shape = (10,)
        spec = _make_array_spec(shape, np.dtype("float64"))
        pipeline = _make_pipeline((BytesCodec(),))
        store_path = _make_store_path()
        transform = ChunkTransform(codecs=tuple(pipeline), array_spec=spec)

        # Write only first 5 elements
        value = _make_nd_buffer(np.arange(5, dtype="float64"))

        pipeline.write_sync(
            [
                WriteChunkRequest(
                    byte_setter=store_path,
                    transform=transform,
                    chunk_selection=(slice(0, 5),),
                    out_selection=(slice(None),),
                    is_complete_chunk=False,
                )
            ],
            value,
        )

        # Read back full chunk
        out = default_buffer_prototype().nd_buffer.create(
            shape=shape,
            dtype=np.dtype("float64"),
            order="C",
            fill_value=-1,
        )
        pipeline.read_sync(
            [
                ReadChunkRequest(
                    byte_getter=store_path,
                    transform=transform,
                    chunk_selection=(slice(None),),
                    out_selection=(slice(None),),
                )
            ],
            out,
        )

        result = out.as_numpy_array()
        np.testing.assert_array_equal(result[:5], np.arange(5, dtype="float64"))
        # Remaining elements should be fill value (0)
        np.testing.assert_array_equal(result[5:], 0)

    def test_read_sync_missing_chunk(self) -> None:
        """Reading a non-existent chunk should fill with fill value."""
        spec = _make_array_spec((10,), np.dtype("float64"))
        pipeline = _make_pipeline((BytesCodec(),))
        store_path = _make_store_path("nonexistent/chunk")
        transform = ChunkTransform(codecs=tuple(pipeline), array_spec=spec)

        out = default_buffer_prototype().nd_buffer.create(
            shape=(10,),
            dtype=np.dtype("float64"),
            order="C",
            fill_value=-1,
        )
        pipeline.read_sync(
            [
                ReadChunkRequest(
                    byte_getter=store_path,
                    transform=transform,
                    chunk_selection=(slice(None),),
                    out_selection=(slice(None),),
                )
            ],
            out,
        )

        # Should be filled with the spec's fill value (0)
        np.testing.assert_array_equal(out.as_numpy_array(), 0)

    def test_write_sync_multiple_chunks(self) -> None:
        """Write and read multiple chunks in one batch."""
        spec = _make_array_spec((10,), np.dtype("float64"))
        pipeline = _make_pipeline((BytesCodec(),))
        store = MemoryStore()
        transform = ChunkTransform(codecs=tuple(pipeline), array_spec=spec)

        value = _make_nd_buffer(np.arange(20, dtype="float64"))

        pipeline.write_sync(
            [
                WriteChunkRequest(
                    byte_setter=StorePath(store, "c/0"),
                    transform=transform,
                    chunk_selection=(slice(None),),
                    out_selection=(slice(0, 10),),
                    is_complete_chunk=True,
                ),
                WriteChunkRequest(
                    byte_setter=StorePath(store, "c/1"),
                    transform=transform,
                    chunk_selection=(slice(None),),
                    out_selection=(slice(10, 20),),
                    is_complete_chunk=True,
                ),
            ],
            value,
        )

        out = default_buffer_prototype().nd_buffer.create(
            shape=(20,),
            dtype=np.dtype("float64"),
            order="C",
            fill_value=0,
        )
        pipeline.read_sync(
            [
                ReadChunkRequest(
                    byte_getter=StorePath(store, "c/0"),
                    transform=transform,
                    chunk_selection=(slice(None),),
                    out_selection=(slice(0, 10),),
                ),
                ReadChunkRequest(
                    byte_getter=StorePath(store, "c/1"),
                    transform=transform,
                    chunk_selection=(slice(None),),
                    out_selection=(slice(10, 20),),
                ),
            ],
            out,
        )

        np.testing.assert_array_equal(out.as_numpy_array(), np.arange(20, dtype="float64"))


class TestChooseWorkers:
    def test_returns_zero_for_single_chunk(self) -> None:
        codecs = (BytesCodec(), ZstdCodec())
        assert _choose_workers(1, 1_000_000, codecs) == 0

    def test_returns_nonzero_for_large_compressed_batch(self) -> None:
        codecs = (BytesCodec(), ZstdCodec())
        n_workers = _choose_workers(10, 1_000_000, codecs)
        assert n_workers > 0

    def test_returns_zero_without_bb_codecs(self) -> None:
        codecs: tuple[Any, ...] = (BytesCodec(),)
        assert _choose_workers(10, 1_000_000, codecs) == 0

    def test_returns_zero_for_small_chunks(self) -> None:
        codecs = (BytesCodec(), ZstdCodec())
        assert _choose_workers(10, 100, codecs) == 0

    @pytest.mark.parametrize("enabled", [True, False])
    def test_config_enabled(self, enabled: bool) -> None:
        from zarr.core.config import config

        with config.set({"threading.codec_workers.enabled": enabled}):
            codecs = (BytesCodec(), ZstdCodec())
            result = _choose_workers(10, 1_000_000, codecs)
            if enabled:
                assert result > 0
            else:
                assert result == 0
