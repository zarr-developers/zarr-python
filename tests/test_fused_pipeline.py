"""Tests for FusedCodecPipeline -- the per-chunk-fused codec pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import zarr
from zarr.codecs.bytes import BytesCodec
from zarr.codecs.gzip import GzipCodec
from zarr.codecs.transpose import TransposeCodec
from zarr.codecs.zstd import ZstdCodec
from zarr.core.codec_pipeline import FusedCodecPipeline
from zarr.core.config import config as zarr_config
from zarr.storage import MemoryStore, StorePath


@pytest.mark.parametrize(
    "codecs",
    [
        (BytesCodec(),),
        (BytesCodec(), GzipCodec(level=1)),
        (BytesCodec(), ZstdCodec(level=1)),
        (TransposeCodec(order=(1, 0)), BytesCodec()),
        (TransposeCodec(order=(1, 0)), BytesCodec(), ZstdCodec(level=1)),
    ],
    ids=["bytes-only", "gzip", "zstd", "transpose", "transpose+zstd"],
)
def test_construction(codecs: tuple[Any, ...]) -> None:
    """FusedCodecPipeline can be constructed from valid codec combinations."""
    pipeline = FusedCodecPipeline.from_codecs(codecs)
    assert pipeline.codecs == codecs


def test_evolve_from_array_spec() -> None:
    """evolve_from_array_spec creates a sync transform."""
    from zarr.core.array_spec import ArrayConfig, ArraySpec
    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.dtype import get_data_type_from_native_dtype

    pipeline = FusedCodecPipeline.from_codecs((BytesCodec(),))
    assert pipeline._sync_transform is None

    zdtype = get_data_type_from_native_dtype(np.dtype("float64"))
    spec = ArraySpec(
        shape=(100,),
        dtype=zdtype,
        fill_value=zdtype.cast_scalar(0),
        config=ArrayConfig(order="C", write_empty_chunks=True),
        prototype=default_buffer_prototype(),
    )
    evolved = pipeline.evolve_from_array_spec(spec)
    assert evolved._sync_transform is not None


# ---------------------------------------------------------------------------
# Sync path tests
#
# These exercise FusedCodecPipeline's synchronous API (write_sync / read_sync /
# _sync_transform), which has no equivalent on BatchedCodecPipeline -- so they
# cannot live in the pipeline-agnostic CodecPipelineTests suite. The async
# roundtrip / fill-value behaviour is covered there (test_scenario) across both
# pipelines and sync/async stores.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dtype", "shape"),
    [
        ("float64", (100,)),
        ("float32", (50,)),
        ("int32", (200,)),
        ("float64", (10, 10)),
    ],
    ids=["f64-1d", "f32-1d", "i32-1d", "f64-2d"],
)
def test_read_write_sync_roundtrip(dtype: str, shape: tuple[int, ...]) -> None:
    """Data written via write_sync can be read back via read_sync."""
    from zarr.core.array_spec import ArrayConfig, ArraySpec
    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.buffer.cpu import NDBuffer as CPUNDBuffer
    from zarr.core.dtype import get_data_type_from_native_dtype

    store = MemoryStore()
    zdtype = get_data_type_from_native_dtype(np.dtype(dtype))
    spec = ArraySpec(
        shape=shape,
        dtype=zdtype,
        fill_value=zdtype.cast_scalar(0),
        config=ArrayConfig(order="C", write_empty_chunks=True),
        prototype=default_buffer_prototype(),
    )

    pipeline = FusedCodecPipeline.from_codecs((BytesCodec(),))
    pipeline = pipeline.evolve_from_array_spec(spec)

    data = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
    value = CPUNDBuffer.from_numpy_array(data)
    chunk_selection = tuple(slice(0, s) for s in shape)
    out_selection = chunk_selection
    store_path = StorePath(store, "c/0")

    # Write sync
    pipeline.write_sync(
        [(store_path, spec, chunk_selection, out_selection, True)],
        value,
    )

    # Read sync
    out = CPUNDBuffer.from_numpy_array(np.zeros(shape, dtype=dtype))
    pipeline.read_sync(
        [(store_path, spec, chunk_selection, out_selection, True)],
        out,
    )

    np.testing.assert_array_equal(data, out.as_numpy_array())


def test_read_sync_missing_chunk_fills() -> None:
    """Sync read of a missing chunk fills with the fill value."""
    from zarr.core.array_spec import ArrayConfig, ArraySpec
    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.buffer.cpu import NDBuffer as CPUNDBuffer
    from zarr.core.dtype import get_data_type_from_native_dtype

    store = MemoryStore()
    zdtype = get_data_type_from_native_dtype(np.dtype("float64"))
    spec = ArraySpec(
        shape=(10,),
        dtype=zdtype,
        fill_value=zdtype.cast_scalar(42.0),
        config=ArrayConfig(order="C", write_empty_chunks=True),
        prototype=default_buffer_prototype(),
    )

    pipeline = FusedCodecPipeline.from_codecs((BytesCodec(),))
    pipeline = pipeline.evolve_from_array_spec(spec)

    out = CPUNDBuffer.from_numpy_array(np.zeros(10, dtype="float64"))
    store_path = StorePath(store, "c/0")
    chunk_sel = (slice(0, 10),)

    pipeline.read_sync(
        [(store_path, spec, chunk_sel, chunk_sel, True)],
        out,
    )

    np.testing.assert_array_equal(out.as_numpy_array(), np.full(10, 42.0))


def test_sync_write_async_read_roundtrip() -> None:
    """Data written via write_sync can be read back via async read."""
    from zarr.core.array_spec import ArrayConfig, ArraySpec
    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.buffer.cpu import NDBuffer as CPUNDBuffer
    from zarr.core.dtype import get_data_type_from_native_dtype
    from zarr.core.sync import sync

    store = MemoryStore()
    zdtype = get_data_type_from_native_dtype(np.dtype("float64"))
    spec = ArraySpec(
        shape=(100,),
        dtype=zdtype,
        fill_value=zdtype.cast_scalar(0),
        config=ArrayConfig(order="C", write_empty_chunks=True),
        prototype=default_buffer_prototype(),
    )

    pipeline = FusedCodecPipeline.from_codecs((BytesCodec(),))
    pipeline = pipeline.evolve_from_array_spec(spec)

    data = np.arange(100, dtype="float64")
    value = CPUNDBuffer.from_numpy_array(data)
    chunk_sel = (slice(0, 100),)
    store_path = StorePath(store, "c/0")

    # Write sync
    pipeline.write_sync(
        [(store_path, spec, chunk_sel, chunk_sel, True)],
        value,
    )

    # Read async
    out = CPUNDBuffer.from_numpy_array(np.zeros(100, dtype="float64"))
    sync(
        pipeline.read(
            [(store_path, spec, chunk_sel, chunk_sel, True)],
            out,
        )
    )


# --- byte-range-write fast-path tests: disabled ---
# The sharding codec's byte-range-write fast path (set_range_sync) was removed
# from this PR pending a decision on the store interface; partial shard writes
# now always take the full-shard-rewrite path. These tests are known-good and
# kept commented out to restore once the store byte-range-write design lands.
# def test_partial_shard_write_uses_set_range() -> None:
#     """Partial shard writes with fixed-size codecs should use set_range_sync.
#
#     Only the FusedCodecPipeline uses byte-range writes for partial shard
#     updates; skipped under other pipelines.
#     """
#     from unittest.mock import patch
#
#     store = zarr.storage.MemoryStore()
#     # write_empty_chunks=True keeps a fixed-size dense layout, which is
#     # required for the byte-range fast path (chunks never transition
#     # present <-> absent).
#     arr = zarr.create_array(
#         store=store,
#         shape=(100,),
#         dtype="float64",
#         chunks=(10,),
#         shards=(100,),
#         compressors=None,
#         fill_value=0.0,
#         config={"write_empty_chunks": True},
#     )
#     if not isinstance(arr._async_array.codec_pipeline, FusedCodecPipeline):
#         pytest.skip("byte-range write optimization is specific to FusedCodecPipeline")
#
#     # Initial full write to create the shard blob
#     arr[:] = np.arange(100, dtype="float64")
#
#     # Partial write — should use set_range_sync, not set_sync
#     with patch.object(type(store), "set_range_sync", wraps=store.set_range_sync) as mock_set_range:
#         arr[5] = 999.0
#
#     # set_range_sync should be called: once for the chunk data, once for the index
#     assert mock_set_range.call_count >= 1, (
#         "Expected set_range_sync to be called for partial shard write"
#     )
#
#     # Verify correctness
#     expected = np.arange(100, dtype="float64")
#     expected[5] = 999.0
#     np.testing.assert_array_equal(arr[:], expected)
#
#
# def test_partial_shard_write_falls_back_for_compressed() -> None:
#     """Partial shard writes with compressed inner codecs should NOT use set_range.
#
#     Only meaningful under FusedCodecPipeline (which can use byte-range writes
#     for fixed-size inner codecs). Other pipelines never use set_range_sync,
#     so the assertion is trivially true and the test is uninformative.
#     """
#     from unittest.mock import patch
#
#     store = zarr.storage.MemoryStore()
#     arr = zarr.create_array(
#         store=store,
#         shape=(100,),
#         dtype="float64",
#         chunks=(10,),
#         shards=(100,),
#         compressors=GzipCodec(),
#         fill_value=0.0,
#     )
#     if not isinstance(arr._async_array.codec_pipeline, FusedCodecPipeline):
#         pytest.skip("byte-range write optimization is specific to FusedCodecPipeline")
#     arr[:] = np.arange(100, dtype="float64")
#
#     with patch.object(type(store), "set_range_sync", wraps=store.set_range_sync) as mock_set_range:
#         arr[5] = 999.0
#
#     # With compression, set_range_sync should NOT be used
#     assert mock_set_range.call_count == 0, (
#         "set_range_sync should not be used with compressed inner codecs"
#     )
#
#     expected = np.arange(100, dtype="float64")
#     expected[5] = 999.0
#     np.testing.assert_array_equal(arr[:], expected)
#
#
# def test_partial_shard_write_skips_set_range_when_write_empty_chunks_false() -> None:
#     """The byte-range fast path must NOT fire under the default write_empty_chunks=False.
#
#     The fast path assumes a fixed, dense shard layout. With empty-chunk skipping
#     (the default) a chunk can transition present<->absent, so an in-place
#     byte-range overwrite would corrupt the layout. The complement of
#     test_partial_shard_write_uses_set_range (which uses write_empty_chunks=True).
#     """
#     from unittest.mock import patch
#
#     store = zarr.storage.MemoryStore()
#     arr = zarr.create_array(
#         store=store,
#         shape=(100,),
#         dtype="float64",
#         chunks=(10,),
#         shards=(100,),
#         compressors=None,
#         fill_value=0.0,
#         # default config: write_empty_chunks=False
#     )
#     if not isinstance(arr._async_array.codec_pipeline, FusedCodecPipeline):
#         pytest.skip("byte-range write optimization is specific to FusedCodecPipeline")
#     arr[:] = np.arange(100, dtype="float64")
#
#     with patch.object(type(store), "set_range_sync", wraps=store.set_range_sync) as mock_set_range:
#         arr[5] = 999.0
#
#     assert mock_set_range.call_count == 0, (
#         "byte-range fast path was taken with write_empty_chunks=False; "
#         "this would produce a dense layout incompatible with empty-chunk skipping"
#     )
#
#     expected = np.arange(100, dtype="float64")
#     expected[5] = 999.0
#     np.testing.assert_array_equal(arr[:], expected)
#
#
# def test_partial_shard_write_handles_readonly_store_buffers(tmp_path: Path) -> None:
#     """The byte-range path decodes the shard index from a store buffer and mutates
#     it; LocalStore returns read-only buffers, so the path must copy before writing.
#
#     Without the copy, the partial write raises
#     ``ValueError: assignment destination is read-only``. Fused-only because only
#     the Fused byte-range path decodes+mutates a shard index in place.
#     """
#     store = zarr.storage.LocalStore(tmp_path / "data.zarr")
#     arr = zarr.create_array(
#         store=store,
#         shape=(16,),
#         chunks=(4,),
#         shards=(8,),
#         dtype="float64",
#         compressors=None,
#         fill_value=0.0,
#         config={"write_empty_chunks": True},
#     )
#     if not isinstance(arr._async_array.codec_pipeline, FusedCodecPipeline):
#         pytest.skip("byte-range write optimization is specific to FusedCodecPipeline")
#     arr[:] = np.arange(16, dtype="float64")
#     arr[2] = 42.0  # triggers the byte-range path against a read-only store buffer
#     assert arr[2] == 42.0


def test_chunk_transform_uses_runtime_prototype() -> None:
    """ChunkTransform must pass each codec the prototype from the runtime chunk_spec,
    not one captured at evolve time. Constructs ChunkTransform directly (a
    Fused-internal data structure with no BatchedCodecPipeline equivalent).
    """
    from zarr.abc.codec import BytesBytesCodec
    from zarr.core.array_spec import ArrayConfig, ArraySpec
    from zarr.core.buffer import Buffer, BufferPrototype, default_buffer_prototype
    from zarr.core.codec_pipeline import ChunkTransform
    from zarr.core.dtype import get_data_type_from_native_dtype

    class _PrototypeRecordingCodec(BytesBytesCodec):  # type: ignore[misc,unused-ignore]
        """A no-op BB codec that records the prototype it was called with."""

        is_fixed_size = True
        seen_prototypes: list[object]

        def __init__(self) -> None:
            object.__setattr__(self, "seen_prototypes", [])

        def to_dict(self) -> dict[str, Any]:
            return {"name": "_prototype_recording", "configuration": {}}

        @classmethod
        def from_dict(cls, data: dict[str, Any]) -> _PrototypeRecordingCodec:
            return cls()

        def compute_encoded_size(self, input_byte_length: int, _spec: ArraySpec) -> int:
            return input_byte_length

        def _encode_sync(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> Buffer | None:
            self.seen_prototypes.append(chunk_spec.prototype)
            return chunk_bytes

        def _decode_sync(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> Buffer:
            self.seen_prototypes.append(chunk_spec.prototype)
            return chunk_bytes

        async def _encode_single(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> Buffer | None:
            return self._encode_sync(chunk_bytes, chunk_spec)

        async def _decode_single(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> Buffer:
            return self._decode_sync(chunk_bytes, chunk_spec)

    recording = _PrototypeRecordingCodec()
    transform = ChunkTransform(codecs=(BytesCodec(), recording))

    zdtype = get_data_type_from_native_dtype(np.dtype("float64"))

    def _spec(prototype: BufferPrototype) -> ArraySpec:
        return ArraySpec(
            shape=(10,),
            dtype=zdtype,
            fill_value=zdtype.cast_scalar(0.0),
            config=ArrayConfig(order="C", write_empty_chunks=False),
            prototype=prototype,
        )

    proto_default = default_buffer_prototype()
    # A distinct BufferPrototype instance with the same buffer/nd_buffer types --
    # fails an identity check but works at runtime.
    proto_other = BufferPrototype(buffer=proto_default.buffer, nd_buffer=proto_default.nd_buffer)
    assert proto_other is not proto_default

    arr = proto_default.nd_buffer.from_numpy_array(np.arange(10, dtype="float64"))
    transform.encode_chunk(arr, _spec(proto_default))
    transform.encode_chunk(arr, _spec(proto_other))

    assert recording.seen_prototypes[0] is proto_default
    assert recording.seen_prototypes[1] is proto_other, (
        "ChunkTransform did not pass the runtime prototype to the codec"
    )


# ---------------------------------------------------------------------------
# Thread-pool (max_workers > 1) tests
#
# The pool dispatch in read_sync/write_sync is Fused-only and off by default
# (codec_pipeline.max_workers defaults to 1 == sequential). These tests opt in
# and exercise the pool path end-to-end, exception propagation from workers,
# and concurrent decode through the shared ChunkTransform.
# ---------------------------------------------------------------------------

_FUSED_POOL_CONFIG = {
    "codec_pipeline.path": "zarr.core.codec_pipeline.FusedCodecPipeline",
    "codec_pipeline.max_workers": 4,
}


def test_read_write_with_thread_pool() -> None:
    """With max_workers > 1, multi-chunk reads and writes dispatch through the
    thread pool (pool.map in read_sync/write_sync) and produce the same results
    as sequential execution."""
    with zarr_config.set(_FUSED_POOL_CONFIG):
        store = MemoryStore()
        arr = zarr.create_array(
            store=store,
            shape=(100,),
            chunks=(10,),
            dtype="float64",
            compressors=None,
            fill_value=0.0,
        )
        assert isinstance(arr._async_array.codec_pipeline, FusedCodecPipeline)
        data = np.arange(100, dtype="float64")
        arr[:] = data  # 10 chunks -> pool dispatch in write_sync
        np.testing.assert_array_equal(arr[:], data)  # pool dispatch in read_sync
        arr[5:25] = 7.0  # partial write: merge path through the pool
        data[5:25] = 7.0
        np.testing.assert_array_equal(arr[:], data)


def test_thread_pool_write_worker_exception_propagates() -> None:
    """A store error raised inside a pool worker during write_sync surfaces to
    the caller (write_sync consumes pool.map, so worker exceptions re-raise)."""
    from unittest.mock import patch

    with zarr_config.set(_FUSED_POOL_CONFIG):
        store = MemoryStore()
        arr = zarr.create_array(
            store=store,
            shape=(100,),
            chunks=(10,),
            dtype="float64",
            compressors=None,
            fill_value=0.0,
        )
        with (
            patch.object(store, "set_sync", side_effect=RuntimeError("simulated store error")),
            pytest.raises(RuntimeError, match="simulated store error"),
        ):
            arr[:] = np.arange(100, dtype="float64")


def test_thread_pool_read_worker_exception_propagates() -> None:
    """A store error raised inside a pool worker during read_sync surfaces to
    the caller (read_sync consumes pool.map into a tuple)."""
    from unittest.mock import patch

    with zarr_config.set(_FUSED_POOL_CONFIG):
        store = MemoryStore()
        arr = zarr.create_array(
            store=store,
            shape=(100,),
            chunks=(10,),
            dtype="float64",
            compressors=None,
            fill_value=0.0,
        )
        arr[:] = np.arange(100, dtype="float64")
        with (
            patch.object(store, "get_sync", side_effect=RuntimeError("simulated store error")),
            pytest.raises(RuntimeError, match="simulated store error"),
        ):
            arr[:]


def test_concurrent_reads_shared_transform_with_pool() -> None:
    """Concurrent decode through the shared ChunkTransform produces correct data.

    The transform's `_resolve_specs` cache is shared mutable state. With no
    array->array codecs the cache is bypassed entirely, so this uses a transpose
    filter to force cache traffic, max_workers=4 so pool workers decode chunks
    concurrently, and an outer thread pool so multiple reads are in flight at
    once. This pins correctness under concurrency (it cannot prove the absence
    of a race, but a torn cache would corrupt results here).
    """
    from concurrent.futures import ThreadPoolExecutor

    with zarr_config.set(_FUSED_POOL_CONFIG):
        store = MemoryStore()
        arr = zarr.create_array(
            store=store,
            shape=(40, 40),
            chunks=(5, 5),
            dtype="int32",
            filters=[TransposeCodec(order=(1, 0))],
            serializer=BytesCodec(),
            compressors=None,
            fill_value=-1,
        )
        data = np.arange(1600, dtype="int32").reshape(40, 40)
        arr[:] = data

        def read_row_block(i: int) -> np.ndarray:
            return np.asarray(arr[i * 4 : (i + 1) * 4, :])

        for _ in range(5):  # several rounds to increase interleaving
            with ThreadPoolExecutor(max_workers=8) as ex:
                futures = {ex.submit(read_row_block, i): i for i in range(10)}
                for fut, i in futures.items():
                    np.testing.assert_array_equal(fut.result(), data[i * 4 : (i + 1) * 4, :])
