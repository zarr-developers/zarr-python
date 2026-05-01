"""Tests for FusedCodecPipeline -- the per-chunk-fused codec pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import zarr
from zarr.abc.store import SupportsSetRange
from zarr.codecs.bytes import BytesCodec
from zarr.codecs.gzip import GzipCodec
from zarr.codecs.transpose import TransposeCodec
from zarr.codecs.zstd import ZstdCodec
from zarr.core.buffer import cpu
from zarr.core.codec_pipeline import FusedCodecPipeline
from zarr.storage import MemoryStore, StorePath


def _create_array(
    shape: tuple[int, ...],
    dtype: str = "float64",
    chunks: tuple[int, ...] | None = None,
    codecs: tuple[Any, ...] = (BytesCodec(),),
    fill_value: object = 0,
) -> zarr.Array[Any]:
    """Create a zarr array using FusedCodecPipeline."""
    if chunks is None:
        chunks = shape

    _ = FusedCodecPipeline.from_codecs(codecs)

    return zarr.create_array(
        StorePath(MemoryStore()),
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        filters=[c for c in codecs if not isinstance(c, BytesCodec)],
        serializer=BytesCodec() if any(isinstance(c, BytesCodec) for c in codecs) else "auto",
        compressors=None,
        fill_value=fill_value,
    )


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
def test_read_write_roundtrip(dtype: str, shape: tuple[int, ...]) -> None:
    """Data written through FusedCodecPipeline can be read back correctly via async path."""
    from zarr.core.array_spec import ArrayConfig, ArraySpec
    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.buffer.cpu import NDBuffer as CPUNDBuffer
    from zarr.core.dtype import get_data_type_from_native_dtype
    from zarr.core.sync import sync

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

    # Write
    data = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
    value = CPUNDBuffer.from_numpy_array(data)
    chunk_selection = tuple(slice(0, s) for s in shape)
    out_selection = chunk_selection

    store_path = StorePath(store, "c/0")
    sync(
        pipeline.write(
            [(store_path, spec, chunk_selection, out_selection, True)],
            value,
        )
    )

    # Read
    out = CPUNDBuffer.from_numpy_array(np.zeros(shape, dtype=dtype))
    sync(
        pipeline.read(
            [(store_path, spec, chunk_selection, out_selection, True)],
            out,
        )
    )

    np.testing.assert_array_equal(data, out.as_numpy_array())


def test_read_missing_chunk_fills() -> None:
    """Reading a missing chunk fills with the fill value."""
    from zarr.core.array_spec import ArrayConfig, ArraySpec
    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.buffer.cpu import NDBuffer as CPUNDBuffer
    from zarr.core.dtype import get_data_type_from_native_dtype
    from zarr.core.sync import sync

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

    sync(
        pipeline.read(
            [(store_path, spec, chunk_sel, chunk_sel, True)],
            out,
        )
    )

    np.testing.assert_array_equal(out.as_numpy_array(), np.full(10, 42.0))


# ---------------------------------------------------------------------------
# Sync path tests
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


def test_sync_transform_encode_decode_roundtrip() -> None:
    """Sync transform can encode and decode a chunk."""
    from zarr.core.array_spec import ArrayConfig, ArraySpec
    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.dtype import Float64

    codecs = (BytesCodec(),)
    pipeline = FusedCodecPipeline.from_codecs(codecs)
    zdtype = Float64()
    spec = ArraySpec(
        shape=(100,),
        dtype=zdtype,
        fill_value=zdtype.cast_scalar(0.0),
        prototype=default_buffer_prototype(),
        config=ArrayConfig(order="C", write_empty_chunks=True),
    )
    pipeline = pipeline.evolve_from_array_spec(spec)
    assert pipeline._sync_transform is not None

    # Encode
    proto = default_buffer_prototype()
    data = proto.nd_buffer.from_numpy_array(np.arange(100, dtype="float64"))
    encoded = pipeline._sync_transform.encode_chunk(data, spec)
    assert encoded is not None

    # Decode
    decoded = pipeline._sync_transform.decode_chunk(encoded, spec)
    np.testing.assert_array_equal(decoded.as_numpy_array(), np.arange(100, dtype="float64"))


# ---------------------------------------------------------------------------
# Streaming read tests
# ---------------------------------------------------------------------------


def test_streaming_read_multiple_chunks() -> None:
    """Read with multiple chunks should produce correct results via streaming pipeline."""
    store = zarr.storage.MemoryStore()
    arr = zarr.create_array(
        store=store,
        shape=(100,),
        dtype="float64",
        chunks=(10,),
        shards=None,
        compressors=None,
        fill_value=0.0,
    )
    data = np.arange(100, dtype="float64")
    arr[:] = data
    result = arr[:]
    np.testing.assert_array_equal(result, data)


def test_streaming_read_strided_slice() -> None:
    """Strided slicing should work correctly with streaming read."""
    store = zarr.storage.MemoryStore()
    arr = zarr.create_array(
        store=store,
        shape=(100,),
        dtype="float64",
        chunks=(10,),
        shards=None,
        compressors=None,
        fill_value=0.0,
    )
    data = np.arange(100, dtype="float64")
    arr[:] = data
    result = arr[::3]
    np.testing.assert_array_equal(result, data[::3])


def test_streaming_read_missing_chunks() -> None:
    """Reading chunks that were never written should return fill value."""
    store = zarr.storage.MemoryStore()
    arr = zarr.create_array(
        store=store,
        shape=(100,),
        dtype="float64",
        chunks=(10,),
        shards=None,
        compressors=None,
        fill_value=-1.0,
    )
    result = arr[:]
    np.testing.assert_array_equal(result, np.full(100, -1.0))


# ---------------------------------------------------------------------------
# Streaming write tests
# ---------------------------------------------------------------------------


def test_streaming_write_complete_overwrite() -> None:
    """Complete overwrite should skip fetching existing data."""
    store = zarr.storage.MemoryStore()
    arr = zarr.create_array(
        store=store,
        shape=(100,),
        dtype="float64",
        chunks=(10,),
        shards=None,
        compressors=None,
        fill_value=0.0,
    )
    data = np.arange(100, dtype="float64")
    arr[:] = data
    np.testing.assert_array_equal(arr[:], data)


def test_streaming_write_partial_update() -> None:
    """Partial updates should correctly merge with existing data."""
    store = zarr.storage.MemoryStore()
    arr = zarr.create_array(
        store=store,
        shape=(100,),
        dtype="float64",
        chunks=(10,),
        shards=None,
        compressors=None,
        fill_value=0.0,
    )
    arr[:] = np.ones(100)
    arr[5:15] = np.full(10, 99.0)
    result = arr[:]
    expected = np.ones(100)
    expected[5:15] = 99.0
    np.testing.assert_array_equal(result, expected)


def test_memory_store_supports_byte_range_setter() -> None:
    """MemoryStore should implement SupportsSetRange."""
    store = zarr.storage.MemoryStore()
    assert isinstance(store, SupportsSetRange)


async def test_memory_store_set_range() -> None:
    """MemoryStore.set_range should overwrite bytes at the given offset."""
    store = zarr.storage.MemoryStore()
    await store._ensure_open()
    buf = cpu.Buffer.from_bytes(b"AAAAAAAAAA")  # 10 bytes
    await store.set("test/key", buf)

    patch = cpu.Buffer.from_bytes(b"XX")
    await store.set_range("test/key", patch, start=3)

    result = await store.get("test/key", prototype=cpu.buffer_prototype)
    assert result is not None
    assert result.to_bytes() == b"AAAXXAAAAA"


def test_sharding_codec_inner_codecs_fixed_size_no_compression() -> None:
    """Inner codecs without compression should be fixed-size."""
    from zarr.codecs.sharding import ShardingCodec

    codec = ShardingCodec(chunk_shape=(10,), codecs=[BytesCodec()])
    assert codec._inner_codecs_fixed_size is True


def test_sharding_codec_inner_codecs_fixed_size_with_compression() -> None:
    """Inner codecs with compression should NOT be fixed-size."""
    from zarr.codecs.sharding import ShardingCodec

    codec = ShardingCodec(chunk_shape=(10,), codecs=[BytesCodec(), GzipCodec()])
    assert codec._inner_codecs_fixed_size is False


def test_partial_shard_write_fixed_size() -> None:
    """Writing a single element to a shard with fixed-size codecs should work correctly."""
    store = zarr.storage.MemoryStore()
    arr = zarr.create_array(
        store=store,
        shape=(100,),
        dtype="float64",
        chunks=(10,),
        shards=(100,),
        compressors=None,
        fill_value=0.0,
    )
    arr[:] = np.arange(100, dtype="float64")
    arr[5] = 999.0
    result = arr[:]
    expected = np.arange(100, dtype="float64")
    expected[5] = 999.0
    np.testing.assert_array_equal(result, expected)


def test_partial_shard_write_roundtrip_correctness() -> None:
    """Multiple partial writes to different inner chunks should all be correct."""
    store = zarr.storage.MemoryStore()
    arr = zarr.create_array(
        store=store,
        shape=(100,),
        dtype="float64",
        chunks=(10,),
        shards=(100,),
        compressors=None,
        fill_value=0.0,
    )
    arr[:] = np.zeros(100, dtype="float64")
    arr[0:10] = np.ones(10)
    arr[50:60] = np.full(10, 2.0)
    arr[90:100] = np.full(10, 3.0)
    result = arr[:]
    expected = np.zeros(100)
    expected[0:10] = 1.0
    expected[50:60] = 2.0
    expected[90:100] = 3.0
    np.testing.assert_array_equal(result, expected)


def test_partial_shard_write_uses_set_range() -> None:
    """Partial shard writes with fixed-size codecs should use set_range_sync.

    Only the FusedCodecPipeline uses byte-range writes for partial shard
    updates; skipped under other pipelines.
    """
    from unittest.mock import patch

    store = zarr.storage.MemoryStore()
    # write_empty_chunks=True keeps a fixed-size dense layout, which is
    # required for the byte-range fast path (chunks never transition
    # present <-> absent).
    arr = zarr.create_array(
        store=store,
        shape=(100,),
        dtype="float64",
        chunks=(10,),
        shards=(100,),
        compressors=None,
        fill_value=0.0,
        config={"write_empty_chunks": True},
    )
    if not isinstance(arr._async_array.codec_pipeline, FusedCodecPipeline):
        pytest.skip("byte-range write optimization is specific to FusedCodecPipeline")

    # Initial full write to create the shard blob
    arr[:] = np.arange(100, dtype="float64")

    # Partial write — should use set_range_sync, not set_sync
    with patch.object(type(store), "set_range_sync", wraps=store.set_range_sync) as mock_set_range:
        arr[5] = 999.0

    # set_range_sync should be called: once for the chunk data, once for the index
    assert mock_set_range.call_count >= 1, (
        "Expected set_range_sync to be called for partial shard write"
    )

    # Verify correctness
    expected = np.arange(100, dtype="float64")
    expected[5] = 999.0
    np.testing.assert_array_equal(arr[:], expected)


def test_partial_shard_write_falls_back_for_compressed() -> None:
    """Partial shard writes with compressed inner codecs should NOT use set_range.

    Only meaningful under FusedCodecPipeline (which can use byte-range writes
    for fixed-size inner codecs). Other pipelines never use set_range_sync,
    so the assertion is trivially true and the test is uninformative.
    """
    from unittest.mock import patch

    store = zarr.storage.MemoryStore()
    arr = zarr.create_array(
        store=store,
        shape=(100,),
        dtype="float64",
        chunks=(10,),
        shards=(100,),
        compressors=GzipCodec(),
        fill_value=0.0,
    )
    if not isinstance(arr._async_array.codec_pipeline, FusedCodecPipeline):
        pytest.skip("byte-range write optimization is specific to FusedCodecPipeline")
    arr[:] = np.arange(100, dtype="float64")

    with patch.object(type(store), "set_range_sync", wraps=store.set_range_sync) as mock_set_range:
        arr[5] = 999.0

    # With compression, set_range_sync should NOT be used
    assert mock_set_range.call_count == 0, (
        "set_range_sync should not be used with compressed inner codecs"
    )

    expected = np.arange(100, dtype="float64")
    expected[5] = 999.0
    np.testing.assert_array_equal(arr[:], expected)
