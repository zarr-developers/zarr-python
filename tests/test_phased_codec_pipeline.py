"""Tests for PhasedCodecPipeline — the three-phase prepare/compute/finalize pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import zarr
from zarr.codecs.bytes import BytesCodec
from zarr.codecs.gzip import GzipCodec
from zarr.codecs.transpose import TransposeCodec
from zarr.codecs.zstd import ZstdCodec
from zarr.core.codec_pipeline import PhasedCodecPipeline
from zarr.storage import MemoryStore, StorePath


def _create_array(
    shape: tuple[int, ...],
    dtype: str = "float64",
    chunks: tuple[int, ...] | None = None,
    codecs: tuple[Any, ...] = (BytesCodec(),),
    fill_value: object = 0,
) -> zarr.Array:
    """Create a zarr array using PhasedCodecPipeline."""
    if chunks is None:
        chunks = shape

    pipeline = PhasedCodecPipeline.from_codecs(codecs)

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
    """PhasedCodecPipeline can be constructed from valid codec combinations."""
    pipeline = PhasedCodecPipeline.from_codecs(codecs)
    assert pipeline.codecs == codecs


def test_evolve_from_array_spec() -> None:
    """evolve_from_array_spec creates a ChunkTransform."""
    from zarr.core.array_spec import ArrayConfig, ArraySpec
    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.dtype import get_data_type_from_native_dtype

    pipeline = PhasedCodecPipeline.from_codecs((BytesCodec(),))
    assert pipeline.chunk_transform is None

    zdtype = get_data_type_from_native_dtype(np.dtype("float64"))
    spec = ArraySpec(
        shape=(100,),
        dtype=zdtype,
        fill_value=zdtype.cast_scalar(0),
        config=ArrayConfig(order="C", write_empty_chunks=True),
        prototype=default_buffer_prototype(),
    )
    evolved = pipeline.evolve_from_array_spec(spec)
    assert evolved.chunk_transform is not None


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
async def test_read_write_roundtrip(dtype: str, shape: tuple[int, ...]) -> None:
    """Data written through PhasedCodecPipeline can be read back correctly."""
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

    pipeline = PhasedCodecPipeline.from_codecs((BytesCodec(),))
    pipeline = pipeline.evolve_from_array_spec(spec)

    # Write
    data = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
    value = CPUNDBuffer.from_numpy_array(data)
    chunk_selection = tuple(slice(0, s) for s in shape)
    out_selection = chunk_selection

    store_path = StorePath(store, "c/0")
    await pipeline.write(
        [(store_path, spec, chunk_selection, out_selection, True)],
        value,
    )

    # Read
    out = CPUNDBuffer.from_numpy_array(np.zeros(shape, dtype=dtype))
    await pipeline.read(
        [(store_path, spec, chunk_selection, out_selection, True)],
        out,
    )

    np.testing.assert_array_equal(data, out.as_numpy_array())


async def test_read_missing_chunk_fills() -> None:
    """Reading a missing chunk fills with the fill value."""
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

    pipeline = PhasedCodecPipeline.from_codecs((BytesCodec(),))
    pipeline = pipeline.evolve_from_array_spec(spec)

    out = CPUNDBuffer.from_numpy_array(np.zeros(10, dtype="float64"))
    store_path = StorePath(store, "c/0")
    chunk_sel = (slice(0, 10),)

    await pipeline.read(
        [(store_path, spec, chunk_sel, chunk_sel, True)],
        out,
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

    pipeline = PhasedCodecPipeline.from_codecs((BytesCodec(),))
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

    pipeline = PhasedCodecPipeline.from_codecs((BytesCodec(),))
    pipeline = pipeline.evolve_from_array_spec(spec)

    out = CPUNDBuffer.from_numpy_array(np.zeros(10, dtype="float64"))
    store_path = StorePath(store, "c/0")
    chunk_sel = (slice(0, 10),)

    pipeline.read_sync(
        [(store_path, spec, chunk_sel, chunk_sel, True)],
        out,
    )

    np.testing.assert_array_equal(out.as_numpy_array(), np.full(10, 42.0))


async def test_sync_write_async_read_roundtrip() -> None:
    """Data written via write_sync can be read back via async read."""
    from zarr.core.array_spec import ArrayConfig, ArraySpec
    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.buffer.cpu import NDBuffer as CPUNDBuffer
    from zarr.core.dtype import get_data_type_from_native_dtype

    store = MemoryStore()
    zdtype = get_data_type_from_native_dtype(np.dtype("float64"))
    spec = ArraySpec(
        shape=(100,),
        dtype=zdtype,
        fill_value=zdtype.cast_scalar(0),
        config=ArrayConfig(order="C", write_empty_chunks=True),
        prototype=default_buffer_prototype(),
    )

    pipeline = PhasedCodecPipeline.from_codecs((BytesCodec(),))
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
    await pipeline.read(
        [(store_path, spec, chunk_sel, chunk_sel, True)],
        out,
    )

    np.testing.assert_array_equal(data, out.as_numpy_array())
