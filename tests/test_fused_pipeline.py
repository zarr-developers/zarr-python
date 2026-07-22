"""Tests for FusedCodecPipeline -- the per-chunk-fused codec pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import zarr
from zarr.abc.codec import BytesBytesCodec
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
    assert pipeline.sync_transform is None

    zdtype = get_data_type_from_native_dtype(np.dtype("float64"))
    spec = ArraySpec(
        shape=(100,),
        dtype=zdtype,
        fill_value=zdtype.cast_scalar(0),
        config=ArrayConfig(order="C", write_empty_chunks=True),
        prototype=default_buffer_prototype(),
    )
    evolved = pipeline.evolve_from_array_spec(spec)
    assert evolved.sync_transform is not None


# ---------------------------------------------------------------------------
# Sync path tests
#
# These exercise FusedCodecPipeline's synchronous API (write_sync / read_sync /
# sync_transform), which has no equivalent on BatchedCodecPipeline -- so they
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


def test_chunk_transform_uses_runtime_prototype() -> None:
    """ChunkTransform must pass each codec the prototype from the runtime chunk_spec,
    not one captured at evolve time. Constructs ChunkTransform directly (a
    Fused-internal data structure with no BatchedCodecPipeline equivalent).
    """
    from zarr.abc.codec import BytesBytesCodec
    from zarr.core.array_spec import ArrayConfig, ArraySpec
    from zarr.core.buffer import Buffer, BufferPrototype, default_buffer_prototype
    from zarr.core.chunk_utils import ChunkTransform
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
    as sequential execution.

    The `_get_pool` spy pins that the pool branch actually fires: without it,
    a config-resolution regression (renamed key, `_resolve_max_workers`
    returning 1) would silently degrade all the pool tests into re-testing the
    sequential branch while staying green.
    """
    from unittest.mock import patch

    import zarr.core.codec_pipeline as cp_mod

    with zarr_config.set(_FUSED_POOL_CONFIG):
        assert cp_mod._resolve_max_workers() == 4, (
            "codec_pipeline.max_workers config did not reach _resolve_max_workers"
        )
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
        with patch.object(cp_mod, "_get_pool", wraps=cp_mod._get_pool) as pool_spy:
            arr[:] = data  # 10 chunks -> pool dispatch in write_sync
            assert pool_spy.call_count >= 1, "multi-chunk write did not take the pool branch"
            writes = pool_spy.call_count
            np.testing.assert_array_equal(arr[:], data)  # pool dispatch in read_sync
            assert pool_spy.call_count > writes, "multi-chunk read did not take the pool branch"
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
    once. Each round RE-OPENS the array so the shared transform starts with a
    cold spec cache and the concurrent readers race the non-atomic first fill —
    reading through a single pre-warmed handle would only ever exercise cache
    hits. This pins correctness under concurrency (it cannot prove the absence
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

        for _ in range(5):  # several rounds, each racing a cold cache
            fresh = zarr.open_array(store=store, mode="r")

            def read_row_block(i: int, handle: zarr.Array[Any] = fresh) -> np.ndarray:
                return np.asarray(handle[i * 4 : (i + 1) * 4, :])

            with ThreadPoolExecutor(max_workers=8) as ex:
                futures = {ex.submit(read_row_block, i): i for i in range(10)}
                for fut, i in futures.items():
                    np.testing.assert_array_equal(fut.result(), data[i * 4 : (i + 1) * 4, :])


def test_shared_transform_decode_alternating_specs() -> None:
    """A single ChunkTransform must decode chunks of DIFFERENT specs correctly
    when calls alternate, exercising eviction/refill of its single-entry
    `_resolve_specs` cache.

    The two specs differ in shape, so each call evicts the other's cached entry.
    A transpose filter forces the cache to be used (with no AA codec the cache is
    bypassed). The cache entry is stored as one atomic tuple precisely so a
    concurrent reader can never observe a key paired with another spec's resolved
    chain; this test pins the sequential eviction/refill correctness that
    underpins that guarantee. (The concurrent counterpart is
    `test_concurrent_reads_shared_transform_with_pool`.)
    """
    from zarr.core.array_spec import ArrayConfig, ArraySpec
    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.buffer.cpu import NDBuffer as CPUNDBuffer
    from zarr.core.chunk_utils import ChunkTransform
    from zarr.core.dtype import get_data_type_from_native_dtype

    def _spec(shape: tuple[int, ...]) -> ArraySpec:
        zdtype = get_data_type_from_native_dtype(np.dtype("int32"))
        return ArraySpec(
            shape=shape,
            dtype=zdtype,
            fill_value=zdtype.cast_scalar(0),
            config=ArrayConfig(order="C", write_empty_chunks=True),
            prototype=default_buffer_prototype(),
        )

    transform = ChunkTransform(codecs=(TransposeCodec(order=(1, 0)), BytesCodec()))

    # two distinct specs (different shapes) sharing the one transform + cache slot
    cases = []
    for shape in [(5, 7), (3, 11)]:
        spec = _spec(shape)
        arr = np.arange(int(np.prod(shape)), dtype="int32").reshape(shape)
        encoded = transform.encode_chunk(CPUNDBuffer.from_numpy_array(arr), spec)
        assert encoded is not None
        cases.append((spec, encoded, arr))

    # Alternate specs so every call evicts and refills the single cache slot.
    for i in range(20):
        spec, encoded, expected = cases[i % len(cases)]
        got = transform.decode_chunk(encoded, spec).as_numpy_array()
        np.testing.assert_array_equal(got, expected)


def test_sharded_fallback_inner_chunks_avoid_async_transform() -> None:
    """Inner chunks of a shard on a NON-sync store decode through the sync
    ChunkTransform, not per-chunk AsyncChunkTransform coroutines.

    The sharding byte getters are in-memory dict wrappers; they implement
    SyncByteGetter/SyncByteSetter, and the nested inner pipeline is evolved
    (so its sync transform exists), letting the nested read/write take the
    sync fast path. Without this, every inner chunk pays a coroutine for a
    dict lookup plus an async per-chunk transform — measured at 1.5x (raw) to
    3.6x (gzip) of sharded fallback read time.
    """
    from unittest.mock import patch

    from zarr.core.codec_pipeline import AsyncChunkTransform
    from zarr.testing.store import LatencyStore

    calls = {"decode": 0, "encode": 0}
    orig_decode = AsyncChunkTransform.decode_chunk
    orig_encode = AsyncChunkTransform.encode_chunk

    async def spy_decode(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls["decode"] += 1
        return await orig_decode(self, *args, **kwargs)

    async def spy_encode(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls["encode"] += 1
        return await orig_encode(self, *args, **kwargs)

    # LatencyStore is not sync-capable -> the OUTER pipeline takes the async
    # fallback; the INNER chunks go over the sharding byte getters.
    store = LatencyStore(MemoryStore(), get_latency=0.0, set_latency=0.0)
    arr = zarr.create_array(
        store=store,
        shape=(100,),
        chunks=(10,),
        shards=(50,),
        dtype="uint8",
        compressors=None,
        fill_value=0,
    )
    if not isinstance(arr._async_array.codec_pipeline, FusedCodecPipeline):
        pytest.skip("sync fast path for inner chunks is specific to FusedCodecPipeline")

    data = np.arange(100, dtype="uint8")
    sync_calls = {"read_sync": 0}
    orig_read_sync = FusedCodecPipeline.read_sync

    def spy_read_sync(self: Any, *args: Any, **kwargs: Any) -> Any:
        sync_calls["read_sync"] += 1
        return orig_read_sync(self, *args, **kwargs)

    with (
        patch.object(AsyncChunkTransform, "decode_chunk", spy_decode),
        patch.object(AsyncChunkTransform, "encode_chunk", spy_encode),
        patch.object(FusedCodecPipeline, "read_sync", spy_read_sync),
    ):
        arr[:] = data
        out = np.asarray(arr[:])

    np.testing.assert_array_equal(out, data)
    assert calls == {"decode": 0, "encode": 0}, (
        f"inner chunks went through per-chunk AsyncChunkTransform coroutines: {calls}"
    )
    # The outer store is not sync-capable, so any read_sync calls are the
    # NESTED pipeline taking the sync fast path over the sharding byte getters
    # (the SyncByteGetter gate). Without the gate, inner chunks go through
    # concurrent_map with one coroutine per chunk. (Writes don't appear here:
    # the fallback write encodes whole shards through the outer sync transform
    # -> ShardingCodec._encode_sync, never touching the nested byte setters.)
    assert sync_calls["read_sync"] >= 1, "nested read did not take the sync fast path"


def test_write_over_sync_byte_setter_takes_sync_path() -> None:
    """`FusedCodecPipeline.write` routes a non-StorePath `SyncByteSetter` (the
    sharding codec's `_ShardingByteSetter`) through `write_sync`.

    This is the write-side twin of the SyncByteGetter gate: the read test
    above cannot guard it because fallback whole-array writes encode shards
    via `_encode_sync` and never touch the nested byte setters. The nested
    `write` over `_ShardingByteSetter` is reached from the async shard encode
    paths (`_encode_single`/`_encode_partial_single`), so pin the gate
    directly: without it, this write degrades to the async fallback (one
    coroutine per inner chunk for an in-memory dict store).
    """
    import asyncio
    from unittest.mock import patch

    from zarr.codecs.sharding import _ShardingByteSetter
    from zarr.core.array_spec import ArrayConfig, ArraySpec
    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.dtype import get_data_type_from_native_dtype

    zdtype = get_data_type_from_native_dtype(np.dtype("uint8"))
    spec = ArraySpec(
        shape=(10,),
        dtype=zdtype,
        fill_value=zdtype.cast_scalar(0),
        config=ArrayConfig(order="C", write_empty_chunks=True),
        prototype=default_buffer_prototype(),
    )
    pipeline = FusedCodecPipeline.from_codecs([BytesCodec()]).evolve_from_array_spec(spec)
    assert pipeline.sync_transform is not None

    shard_dict: dict[tuple[int, ...], Any] = {}
    setter = _ShardingByteSetter(shard_dict, (0,))
    value = default_buffer_prototype().nd_buffer.from_numpy_array(np.arange(10, dtype="uint8"))

    sync_calls = {"write_sync": 0}
    orig_write_sync = FusedCodecPipeline.write_sync

    def spy_write_sync(self: Any, *args: Any, **kwargs: Any) -> Any:
        sync_calls["write_sync"] += 1
        return orig_write_sync(self, *args, **kwargs)

    sel = (slice(0, 10),)
    with patch.object(FusedCodecPipeline, "write_sync", spy_write_sync):
        asyncio.run(pipeline.write([(setter, spec, sel, sel, True)], value))

    assert sync_calls["write_sync"] >= 1, (
        "write over a SyncByteSetter did not take the sync fast path"
    )
    written = shard_dict[(0,)]
    np.testing.assert_array_equal(
        np.frombuffer(written.to_bytes(), dtype="uint8"), np.arange(10, dtype="uint8")
    )


# ---------------------------------------------------------------------------
# Async-only codecs inside a shard's inner codec chain
# ---------------------------------------------------------------------------


class _AsyncOnlyNoopCodec(BytesBytesCodec):  # type: ignore[misc,unused-ignore]
    """A no-op BB codec implementing ONLY the async codec interface.

    Deliberately does NOT satisfy `SupportsSyncCodec` (no `_decode_sync` /
    `_encode_sync`), modelling a third-party codec that predates the sync
    protocol. Class-level counters prove the codec actually ran.
    """

    is_fixed_size = True
    encode_calls = 0
    decode_calls = 0

    def to_dict(self) -> dict[str, Any]:
        return {"name": "test-async-only-noop", "configuration": {}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _AsyncOnlyNoopCodec:
        return cls()

    def compute_encoded_size(self, input_byte_length: int, _spec: Any) -> int:
        return input_byte_length

    async def _encode_single(self, chunk_bytes: Any, chunk_spec: Any) -> Any:
        type(self).encode_calls += 1
        return chunk_bytes

    async def _decode_single(self, chunk_bytes: Any, chunk_spec: Any) -> Any:
        type(self).decode_calls += 1
        return chunk_bytes


def test_sharded_roundtrip_with_async_only_inner_codec() -> None:
    """A sharded array whose INNER codec chain contains an async-only codec
    round-trips under FusedCodecPipeline (full write, partial write, full read,
    partial read).

    Regression: the pipeline's top-level guard (evolve_from_array_spec ->
    sync_transform=None) only inspected the top-level chain. ShardingCodec
    structurally satisfies SupportsSyncCodec, so a sync transform was built and
    the sync fast path dove into ShardingCodec's sync shard paths, which raised
    TypeError from the inner ChunkTransform. The pipeline must instead decline
    the sync fast path and fall back to the async inner pipeline, like
    BatchedCodecPipeline.
    """
    _AsyncOnlyNoopCodec.encode_calls = 0
    _AsyncOnlyNoopCodec.decode_calls = 0

    with zarr_config.set({"codec_pipeline.path": "zarr.core.codec_pipeline.FusedCodecPipeline"}):
        store = MemoryStore()
        arr = zarr.create_array(
            store=store,
            shape=(16, 16),
            shards=(8, 8),
            chunks=(4, 4),
            dtype="int32",
            compressors=[_AsyncOnlyNoopCodec()],
            fill_value=-1,
        )
        assert isinstance(arr._async_array.codec_pipeline, FusedCodecPipeline)

        data = np.arange(256, dtype="int32").reshape(16, 16)
        arr[:] = data  # full write
        np.testing.assert_array_equal(arr[:], data)  # full read
        np.testing.assert_array_equal(arr[2:11, 3:14], data[2:11, 3:14])  # partial read

        arr[5:7, 5:13] = 0  # partial write (read-merge-write of existing shards)
        data[5:7, 5:13] = 0
        np.testing.assert_array_equal(arr[:], data)

    assert _AsyncOnlyNoopCodec.encode_calls > 0, "async-only inner codec never encoded"
    assert _AsyncOnlyNoopCodec.decode_calls > 0, "async-only inner codec never decoded"

    # The stored bytes are valid for the default pipeline too: read them back
    # under BatchedCodecPipeline (default codec_pipeline.path). Opening from
    # metadata needs the codec name in the registry.
    from zarr.registry import register_codec

    register_codec("test-async-only-noop", _AsyncOnlyNoopCodec)
    reread = zarr.open_array(store=store, mode="r")
    np.testing.assert_array_equal(reread[:], data)


# ---------------------------------------------------------------------------
# AsyncChunkTransform: the async per-chunk codec chain used on the async
# fallback path. It is the async mirror of ChunkTransform, so it must produce
# identical bytes/arrays. The default (Fused, sync-store) path never uses it;
# these tests drive it directly over multi-codec chains so the aa/bb loops and
# the all-fill drop branch are exercised.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "codecs",
    [
        (BytesCodec(),),
        (BytesCodec(), GzipCodec(level=1)),
        (TransposeCodec(order=(1, 0)), BytesCodec()),
        (TransposeCodec(order=(1, 0)), BytesCodec(), ZstdCodec(level=1)),
    ],
    ids=["bytes-only", "bb", "aa", "aa+ab+bb"],
)
def test_async_chunk_transform_matches_sync(codecs: tuple[Any, ...]) -> None:
    """`AsyncChunkTransform.decode_chunk`/`encode_chunk` must round-trip and
    produce exactly what the synchronous `ChunkTransform` produces, across
    array->array, array->bytes, and bytes->bytes codec combinations.

    This is the async mirror of the codecs the default pipeline runs
    synchronously; a divergence here corrupts data only on the async fallback
    path (remote stores), which no end-to-end test of the default pipeline
    touches.
    """
    import asyncio

    from zarr.core.array_spec import ArrayConfig, ArraySpec
    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.buffer.cpu import NDBuffer as CPUNDBuffer
    from zarr.core.chunk_utils import ChunkTransform, evolve_codecs
    from zarr.core.codec_pipeline import AsyncChunkTransform
    from zarr.core.dtype import get_data_type_from_native_dtype

    shape = (4, 4)
    zdtype = get_data_type_from_native_dtype(np.dtype("int32"))
    spec = ArraySpec(
        shape=shape,
        dtype=zdtype,
        fill_value=zdtype.cast_scalar(0),
        config=ArrayConfig(order="C", write_empty_chunks=True),
        prototype=default_buffer_prototype(),
    )
    evolved = evolve_codecs(codecs, spec)
    sync_t = ChunkTransform(codecs=evolved)
    async_t = AsyncChunkTransform(codecs=evolved)

    data = np.arange(16, dtype="int32").reshape(shape)
    value = CPUNDBuffer.from_numpy_array(data)

    sync_bytes = sync_t.encode_chunk(value, spec)
    async_bytes = asyncio.run(async_t.encode_chunk(value, spec))
    assert sync_bytes is not None
    assert async_bytes is not None
    np.testing.assert_array_equal(async_bytes.to_bytes(), sync_bytes.to_bytes())

    sync_arr = sync_t.decode_chunk(async_bytes, spec)
    async_arr = asyncio.run(async_t.decode_chunk(async_bytes, spec))
    np.testing.assert_array_equal(async_arr.as_numpy_array(), sync_arr.as_numpy_array())
    np.testing.assert_array_equal(async_arr.as_numpy_array(), data)


def test_async_decode_encode_passes_through_none_chunks() -> None:
    """`FusedCodecPipeline.decode`/`encode` (the async batch entry points used
    on the fallback path) map a None chunk to None and leave real chunks
    untouched — pins the None-passthrough branch the default sync path skips."""
    import asyncio

    from zarr.core.array_spec import ArrayConfig, ArraySpec
    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.buffer.cpu import NDBuffer as CPUNDBuffer
    from zarr.core.dtype import get_data_type_from_native_dtype

    zdtype = get_data_type_from_native_dtype(np.dtype("int32"))
    spec = ArraySpec(
        shape=(4,),
        dtype=zdtype,
        fill_value=zdtype.cast_scalar(0),
        config=ArrayConfig(order="C", write_empty_chunks=True),
        prototype=default_buffer_prototype(),
    )
    pipeline = FusedCodecPipeline.from_codecs([BytesCodec()]).evolve_from_array_spec(spec)

    data = np.arange(4, dtype="int32")
    value = CPUNDBuffer.from_numpy_array(data)

    # encode a real chunk and a None chunk together
    encoded = list(asyncio.run(pipeline.encode([(value, spec), (None, spec)])))
    assert encoded[1] is None
    assert encoded[0] is not None

    # decode the real chunk and a None chunk together
    decoded = list(asyncio.run(pipeline.decode([(encoded[0], spec), (None, spec)])))
    assert decoded[1] is None
    assert decoded[0] is not None
    np.testing.assert_array_equal(decoded[0].as_numpy_array(), data)
