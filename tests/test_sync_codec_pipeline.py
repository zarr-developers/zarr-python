"""Tests for sync codec capabilities in BatchedCodecPipeline."""

from __future__ import annotations

import numpy as np
import pytest

import zarr
from zarr.codecs.bytes import BytesCodec
from zarr.codecs.gzip import GzipCodec
from zarr.codecs.transpose import TransposeCodec
from zarr.codecs.zstd import ZstdCodec
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.codec_pipeline import BatchedCodecPipeline
from zarr.core.dtype import get_data_type_from_native_dtype
from zarr.storage import MemoryStore


def _make_array_spec(
    shape: tuple[int, ...], dtype: np.dtype
) -> ArraySpec:
    zdtype = get_data_type_from_native_dtype(dtype)
    return ArraySpec(
        shape=shape,
        dtype=zdtype,
        fill_value=zdtype.default_scalar(),
        config=ArrayConfig(order="C", write_empty_chunks=True),
        prototype=default_buffer_prototype(),
    )


def _make_nd_buffer(arr: np.ndarray) -> zarr.core.buffer.NDBuffer:
    return default_buffer_prototype().nd_buffer.from_numpy_array(arr)


# ---------------------------------------------------------------------------
# Unit tests: supports_sync property
# ---------------------------------------------------------------------------


class TestSupportsSync:
    def test_gzip_supports_sync(self):
        assert GzipCodec().supports_sync

    def test_zstd_supports_sync(self):
        assert ZstdCodec().supports_sync

    def test_bytes_supports_sync(self):
        assert BytesCodec().supports_sync

    def test_transpose_supports_sync(self):
        assert TransposeCodec(order=(0, 1)).supports_sync

    def test_sharding_supports_sync(self):
        from zarr.codecs.sharding import ShardingCodec

        assert ShardingCodec(chunk_shape=(8,)).supports_sync


# ---------------------------------------------------------------------------
# Unit tests: individual codec sync roundtrips
# ---------------------------------------------------------------------------


class TestGzipCodecSync:
    def test_roundtrip(self):
        codec = GzipCodec(level=1)
        arr = np.arange(100, dtype="float64")
        spec = _make_array_spec(arr.shape, arr.dtype)
        buf = default_buffer_prototype().buffer.from_array_like(arr.view("B"))

        encoded = codec._encode_sync(buf, spec)
        assert encoded is not None
        decoded = codec._decode_sync(encoded, spec)
        result = np.frombuffer(decoded.as_numpy_array(), dtype="float64")
        np.testing.assert_array_equal(arr, result)


class TestZstdCodecSync:
    def test_roundtrip(self):
        codec = ZstdCodec(level=1)
        arr = np.arange(100, dtype="float64")
        spec = _make_array_spec(arr.shape, arr.dtype)
        buf = default_buffer_prototype().buffer.from_array_like(arr.view("B"))

        encoded = codec._encode_sync(buf, spec)
        assert encoded is not None
        decoded = codec._decode_sync(encoded, spec)
        result = np.frombuffer(decoded.as_numpy_array(), dtype="float64")
        np.testing.assert_array_equal(arr, result)


class TestBytesCodecSync:
    def test_roundtrip(self):
        codec = BytesCodec()
        arr = np.arange(100, dtype="float64")
        spec = _make_array_spec(arr.shape, arr.dtype)
        nd_buf = _make_nd_buffer(arr)

        # Evolve from array spec (handles endianness)
        codec = codec.evolve_from_array_spec(spec)

        encoded = codec._encode_sync(nd_buf, spec)
        assert encoded is not None
        decoded = codec._decode_sync(encoded, spec)
        np.testing.assert_array_equal(arr, decoded.as_numpy_array())


class TestTransposeCodecSync:
    def test_roundtrip(self):
        codec = TransposeCodec(order=(1, 0))
        arr = np.arange(12, dtype="float64").reshape(3, 4)
        spec = _make_array_spec(arr.shape, arr.dtype)
        nd_buf = _make_nd_buffer(arr)

        encoded = codec._encode_sync(nd_buf, spec)
        assert encoded is not None
        resolved_spec = codec.resolve_metadata(spec)
        decoded = codec._decode_sync(encoded, resolved_spec)
        np.testing.assert_array_equal(arr, decoded.as_numpy_array())


# ---------------------------------------------------------------------------
# Unit tests: pipeline construction
# ---------------------------------------------------------------------------


class TestPipelineConstruction:
    def test_from_codecs_valid(self):
        pipeline = BatchedCodecPipeline.from_codecs([BytesCodec(), GzipCodec(level=1)])
        assert isinstance(pipeline, BatchedCodecPipeline)
        assert len(pipeline.bytes_bytes_codecs) == 1
        assert isinstance(pipeline.array_bytes_codec, BytesCodec)

    def test_from_codecs_accepts_sharding(self):
        from zarr.codecs.sharding import ShardingCodec

        pipeline = BatchedCodecPipeline.from_codecs([ShardingCodec(chunk_shape=(8,))])
        assert isinstance(pipeline, BatchedCodecPipeline)
        assert pipeline._all_sync

    def test_from_codecs_rejects_missing_array_bytes(self):
        with pytest.raises(ValueError, match="Required ArrayBytesCodec"):
            BatchedCodecPipeline.from_codecs([GzipCodec()])

    def test_from_codecs_with_transpose(self):
        pipeline = BatchedCodecPipeline.from_codecs([
            TransposeCodec(order=(1, 0)),
            BytesCodec(),
            GzipCodec(level=1),
        ])
        assert len(pipeline.array_array_codecs) == 1
        assert isinstance(pipeline.array_array_codecs[0], TransposeCodec)


# ---------------------------------------------------------------------------
# Unit tests: pipeline encode/decode roundtrip
# ---------------------------------------------------------------------------


class TestPipelineRoundtrip:
    @pytest.mark.asyncio
    async def test_encode_decode_single_chunk(self):
        pipeline = BatchedCodecPipeline.from_codecs([BytesCodec(), GzipCodec(level=1)])
        arr = np.random.default_rng(42).standard_normal((32, 32)).astype("float64")
        spec = _make_array_spec(arr.shape, arr.dtype)
        pipeline = pipeline.evolve_from_array_spec(spec)
        nd_buf = _make_nd_buffer(arr)

        encoded = await pipeline.encode([(nd_buf, spec)])
        decoded = await pipeline.decode([(list(encoded)[0], spec)])
        result = list(decoded)[0]
        assert result is not None
        np.testing.assert_array_equal(arr, result.as_numpy_array())

    @pytest.mark.asyncio
    async def test_encode_decode_multiple_chunks(self):
        pipeline = BatchedCodecPipeline.from_codecs([BytesCodec(), GzipCodec(level=1)])
        rng = np.random.default_rng(42)
        spec = _make_array_spec((16, 16), np.dtype("float64"))
        pipeline = pipeline.evolve_from_array_spec(spec)
        chunks = [rng.standard_normal((16, 16)).astype("float64") for _ in range(10)]
        nd_bufs = [_make_nd_buffer(c) for c in chunks]

        encoded = list(await pipeline.encode([(buf, spec) for buf in nd_bufs]))
        decoded = list(await pipeline.decode([(enc, spec) for enc in encoded]))
        for original, dec in zip(chunks, decoded):
            assert dec is not None
            np.testing.assert_array_equal(original, dec.as_numpy_array())

    @pytest.mark.asyncio
    async def test_encode_decode_empty_batch(self):
        pipeline = BatchedCodecPipeline.from_codecs([BytesCodec(), GzipCodec(level=1)])
        encoded = await pipeline.encode([])
        assert list(encoded) == []
        decoded = await pipeline.decode([])
        assert list(decoded) == []

    @pytest.mark.asyncio
    async def test_encode_decode_none_chunk(self):
        pipeline = BatchedCodecPipeline.from_codecs([BytesCodec(), GzipCodec(level=1)])
        spec = _make_array_spec((8,), np.dtype("float64"))
        pipeline = pipeline.evolve_from_array_spec(spec)

        encoded = list(await pipeline.encode([(None, spec)]))
        assert encoded[0] is None

        decoded = list(await pipeline.decode([(None, spec)]))
        assert decoded[0] is None


# ---------------------------------------------------------------------------
# Integration tests: default pipeline has sync capabilities
# ---------------------------------------------------------------------------


class TestDefaultPipelineSync:
    def test_create_array_uses_batched_pipeline(self):
        store = MemoryStore()
        arr = zarr.create_array(
            store,
            shape=(100, 100),
            chunks=(32, 32),
            dtype="float64",
        )
        assert isinstance(arr.async_array.codec_pipeline, BatchedCodecPipeline)

        data = np.random.default_rng(42).standard_normal((100, 100))
        arr[:] = data
        np.testing.assert_array_equal(arr[:], data)

    def test_open_uses_batched_pipeline(self):
        store = MemoryStore()
        arr = zarr.create_array(
            store,
            shape=(50, 50),
            chunks=(25, 25),
            dtype="float64",
        )
        data = np.random.default_rng(42).standard_normal((50, 50))
        arr[:] = data

        arr2 = zarr.open_array(store=store)
        assert isinstance(arr2.async_array.codec_pipeline, BatchedCodecPipeline)
        np.testing.assert_array_equal(arr2[:], data)

    def test_from_array_uses_batched_pipeline(self):
        store1 = MemoryStore()
        arr1 = zarr.create_array(
            store1,
            shape=(20, 20),
            chunks=(10, 10),
            dtype="float64",
        )
        data = np.random.default_rng(42).standard_normal((20, 20))
        arr1[:] = data

        store2 = MemoryStore()
        arr2 = zarr.from_array(store2, data=arr1)
        assert isinstance(arr2.async_array.codec_pipeline, BatchedCodecPipeline)
        np.testing.assert_array_equal(arr2[:], data)

    def test_partial_write(self):
        store = MemoryStore()
        arr = zarr.create_array(
            store,
            shape=(100,),
            chunks=(10,),
            dtype="int32",
            fill_value=0,
        )
        arr[5:15] = np.arange(10, dtype="int32") + 1
        result = arr[:]
        expected = np.zeros(100, dtype="int32")
        expected[5:15] = np.arange(10, dtype="int32") + 1
        np.testing.assert_array_equal(result, expected)

    def test_zstd_codec(self):
        store = MemoryStore()
        arr = zarr.create_array(
            store,
            shape=(50,),
            chunks=(10,),
            dtype="float32",
            compressors=ZstdCodec(level=3),
        )
        data = np.random.default_rng(42).standard_normal(50).astype("float32")
        arr[:] = data
        np.testing.assert_array_equal(arr[:], data)

    def test_supports_sync_io(self):
        """Default pipeline supports sync IO when all codecs are sync."""
        pipeline = BatchedCodecPipeline.from_codecs([BytesCodec(), GzipCodec(level=1)])
        assert pipeline.supports_sync_io

    def test_config_switch_to_sync_pipeline_compat(self):
        """Verify backwards compat: SyncCodecPipeline config path still works."""
        from zarr.experimental.sync_codecs import SyncCodecPipeline

        zarr.config.set(
            {"codec_pipeline.path": "zarr.experimental.sync_codecs.SyncCodecPipeline"}
        )
        try:
            store = MemoryStore()
            arr = zarr.create_array(store, shape=(10,), dtype="float64")
            assert isinstance(arr.async_array.codec_pipeline, SyncCodecPipeline)
            # SyncCodecPipeline is-a BatchedCodecPipeline
            assert isinstance(arr.async_array.codec_pipeline, BatchedCodecPipeline)
        finally:
            zarr.config.set(
                {"codec_pipeline.path": "zarr.core.codec_pipeline.BatchedCodecPipeline"}
            )
