"""Benchmark comparing BatchedCodecPipeline vs PhasedCodecPipeline.

Run with: hatch run test.py3.12-minimal:pytest tests/test_pipeline_benchmark.py -v --benchmark-enable
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
import pytest

from zarr.abc.codec import Codec
from zarr.codecs.bytes import BytesCodec
from zarr.codecs.gzip import GzipCodec
from zarr.codecs.sharding import ShardingCodec
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.buffer.cpu import NDBuffer as CPUNDBuffer
from zarr.core.codec_pipeline import BatchedCodecPipeline, PhasedCodecPipeline
from zarr.core.dtype import get_data_type_from_native_dtype
from zarr.core.sync import sync
from zarr.storage import MemoryStore, StorePath


class PipelineKind(Enum):
    batched = "batched"
    phased_async = "phased_async"
    phased_sync = "phased_sync"
    phased_sync_threaded = "phased_sync_threaded"


# 1 MB of float64 = 131072 elements
CHUNK_ELEMENTS = 1024 * 1024 // 8
CHUNK_SHAPE = (CHUNK_ELEMENTS,)


def _make_spec(shape: tuple[int, ...], dtype: str = "float64") -> ArraySpec:
    zdtype = get_data_type_from_native_dtype(np.dtype(dtype))
    return ArraySpec(
        shape=shape,
        dtype=zdtype,
        fill_value=zdtype.cast_scalar(0),
        config=ArrayConfig(order="C", write_empty_chunks=True),
        prototype=default_buffer_prototype(),
    )


def _build_codecs(
    compressor: str,
    serializer: str,
) -> tuple[Codec, ...]:
    """Build a codec tuple from human-readable compressor/serializer names."""
    bb: tuple[Codec, ...] = ()
    if compressor == "gzip":
        bb = (GzipCodec(level=1),)

    if serializer == "sharding":
        # 4 inner chunks per shard
        inner_chunk = (CHUNK_ELEMENTS // 4,)
        inner_codecs: list[Codec] = [BytesCodec()]
        if bb:
            inner_codecs.extend(bb)
        return (ShardingCodec(chunk_shape=inner_chunk, codecs=inner_codecs),)
    else:
        return (BytesCodec(), *bb)


def _make_pipeline(
    kind: PipelineKind,
    codecs: tuple[Codec, ...],
    spec: ArraySpec,
) -> BatchedCodecPipeline | PhasedCodecPipeline:
    if kind == PipelineKind.batched:
        pipeline = BatchedCodecPipeline.from_codecs(codecs)
        # Work around generator-consumption bug in codecs_from_list
        evolved_codecs = tuple(c.evolve_from_array_spec(array_spec=spec) for c in pipeline)
        return BatchedCodecPipeline.from_codecs(evolved_codecs)
    else:  # phased_async, phased_sync, phased_sync_threaded
        pipeline = PhasedCodecPipeline.from_codecs(codecs)
        return pipeline.evolve_from_array_spec(spec)


def _write_and_read(
    pipeline: BatchedCodecPipeline | PhasedCodecPipeline,
    store: MemoryStore,
    spec: ArraySpec,
    data: np.ndarray[Any, np.dtype[Any]],
    kind: PipelineKind,
    n_chunks: int = 1,
) -> None:
    """Write data as n_chunks, then read it all back."""
    chunk_size = data.shape[0] // n_chunks
    chunk_shape = (chunk_size,)
    chunk_spec = _make_spec(chunk_shape, dtype=str(data.dtype))

    # Build batch info for all chunks
    write_batch: list[tuple[Any, ...]] = []
    for i in range(n_chunks):
        store_path = StorePath(store, f"c/{i}")
        chunk_sel = (slice(0, chunk_size),)
        out_sel = (slice(i * chunk_size, (i + 1) * chunk_size),)
        write_batch.append((store_path, chunk_spec, chunk_sel, out_sel, True))

    value = CPUNDBuffer.from_numpy_array(data)

    if kind == PipelineKind.phased_sync:
        assert isinstance(pipeline, PhasedCodecPipeline)
        pipeline.write_sync(write_batch, value)
        out = CPUNDBuffer.from_numpy_array(np.empty_like(data))
        pipeline.read_sync(write_batch, out)
    elif kind == PipelineKind.phased_sync_threaded:
        assert isinstance(pipeline, PhasedCodecPipeline)
        pipeline.write_sync(write_batch, value, n_workers=4)
        out = CPUNDBuffer.from_numpy_array(np.empty_like(data))
        pipeline.read_sync(write_batch, out, n_workers=4)
    else:
        sync(pipeline.write(write_batch, value))
        out = CPUNDBuffer.from_numpy_array(np.empty_like(data))
        sync(pipeline.read(write_batch, out))


@pytest.mark.benchmark(group="pipeline")
@pytest.mark.parametrize(
    "kind",
    [
        PipelineKind.batched,
        PipelineKind.phased_async,
        PipelineKind.phased_sync,
        PipelineKind.phased_sync_threaded,
    ],
    ids=["batched", "phased-async", "phased-sync", "phased-sync-threaded"],
)
@pytest.mark.parametrize("compressor", ["none", "gzip"], ids=["no-compress", "gzip"])
@pytest.mark.parametrize("serializer", ["bytes", "sharding"], ids=["bytes", "sharding"])
@pytest.mark.parametrize("n_chunks", [1, 8], ids=["1chunk", "8chunks"])
def test_pipeline(
    benchmark: Any,
    kind: PipelineKind,
    compressor: str,
    serializer: str,
    n_chunks: int,
) -> None:
    """1 MB per chunk, parametrized over pipeline, compressor, serializer, and chunk count."""
    codecs = _build_codecs(compressor, serializer)

    # Sync paths require SupportsChunkPacking for the BytesCodec-level IO
    # ShardingCodec now has _decode_sync/_encode_sync but not SupportsChunkPacking
    if serializer == "sharding" and kind in (PipelineKind.phased_sync, PipelineKind.phased_sync_threaded):
        pytest.skip("Sync IO path not yet implemented for ShardingCodec")

    # Threading only helps with multiple chunks
    if kind == PipelineKind.phased_sync_threaded and n_chunks == 1:
        pytest.skip("Threading with 1 chunk has no benefit")

    total_elements = CHUNK_ELEMENTS * n_chunks
    spec = _make_spec((total_elements,))
    data = np.random.default_rng(42).random(total_elements)
    store = MemoryStore()
    pipeline = _make_pipeline(kind, codecs, _make_spec(CHUNK_SHAPE))

    benchmark(_write_and_read, pipeline, store, spec, data, kind, n_chunks)
