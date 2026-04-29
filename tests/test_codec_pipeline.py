from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import zarr
from zarr.core.array import _get_chunk_spec
from zarr.core.buffer.core import default_buffer_prototype
from zarr.core.config import config as zarr_config
from zarr.core.indexing import BasicIndexer
from zarr.errors import ChunkNotFoundError
from zarr.storage import MemoryStore

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def _enable_rectilinear_chunks() -> Generator[None]:
    """Enable rectilinear chunks for all tests in this module."""
    with zarr_config.set({"array.rectilinear_chunks": True}):
        yield


pipeline_paths = [
    "zarr.core.codec_pipeline.BatchedCodecPipeline",
    "zarr.core.codec_pipeline.SyncCodecPipeline",
]


@pytest.fixture(params=pipeline_paths, ids=["batched", "sync"])
def pipeline_class(request: pytest.FixtureRequest) -> Generator[str]:
    """Temporarily set the codec pipeline class for the test."""
    path = request.param
    with zarr_config.set({"codec_pipeline.path": path}):
        yield path


# ---------------------------------------------------------------------------
# GetResult status tests (low-level pipeline API)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("write_slice", "read_slice", "expected_statuses"),
    [
        (slice(None), slice(None), ("present", "present", "present")),
        (slice(0, 2), slice(None), ("present", "missing", "missing")),
        (None, slice(None), ("missing", "missing", "missing")),
    ],
)
async def test_read_returns_get_results(
    pipeline_class: str,
    write_slice: slice | None,
    read_slice: slice,
    expected_statuses: tuple[str, ...],
) -> None:
    """CodecPipeline.read returns GetResult with correct statuses."""
    store = MemoryStore()
    arr = zarr.open_array(store, mode="w", shape=(6,), chunks=(2,), dtype="int64", fill_value=-1)

    if write_slice is not None:
        arr[write_slice] = 0

    async_arr = arr._async_array
    pipeline = async_arr.codec_pipeline
    metadata = async_arr.metadata

    prototype = default_buffer_prototype()
    config = async_arr.config
    indexer = BasicIndexer(
        read_slice,
        shape=metadata.shape,
        chunk_grid=async_arr._chunk_grid,
    )

    out_buffer = prototype.nd_buffer.empty(
        shape=indexer.shape,
        dtype=metadata.dtype.to_native_dtype(),
        order=config.order,
    )

    results = await pipeline.read(
        [
            (
                async_arr.store_path / metadata.encode_chunk_key(chunk_coords),
                _get_chunk_spec(metadata, async_arr._chunk_grid, chunk_coords, config, prototype),
                chunk_selection,
                out_selection,
                is_complete_chunk,
            )
            for chunk_coords, chunk_selection, out_selection, is_complete_chunk in indexer
        ],
        out_buffer,
        drop_axes=indexer.drop_axes,
    )

    assert len(results) == len(expected_statuses)
    for result, expected_status in zip(results, expected_statuses, strict=True):
        assert result["status"] == expected_status


# ---------------------------------------------------------------------------
# End-to-end read/write tests
# ---------------------------------------------------------------------------

array_configs = [
    pytest.param(
        {"shape": (100,), "dtype": "float64", "chunks": (10,), "shards": None, "compressors": None},
        id="1d-unsharded",
    ),
    pytest.param(
        {
            "shape": (100,),
            "dtype": "float64",
            "chunks": (10,),
            "shards": (100,),
            "compressors": None,
        },
        id="1d-sharded",
    ),
    pytest.param(
        {
            "shape": (10, 20),
            "dtype": "int32",
            "chunks": (5, 10),
            "shards": None,
            "compressors": None,
        },
        id="2d-unsharded",
    ),
    pytest.param(
        {
            "shape": (100,),
            "dtype": "float64",
            "chunks": (10,),
            "shards": None,
            "compressors": {"name": "gzip", "configuration": {"level": 1}},
        },
        id="1d-gzip",
    ),
    pytest.param(
        {
            "shape": (60, 100),
            "dtype": "int32",
            "chunks": [[10, 20, 30], [50, 50]],
            "shards": None,
            "compressors": None,
        },
        id="2d-rectilinear",
    ),
]


@pytest.mark.parametrize("arr_kwargs", array_configs)
async def test_roundtrip(pipeline_class: str, arr_kwargs: dict[str, Any]) -> None:
    """Data survives a full write/read roundtrip."""
    store = MemoryStore()
    arr = zarr.create_array(store=store, fill_value=0, **arr_kwargs)
    data = np.arange(int(np.prod(arr.shape)), dtype=arr.dtype).reshape(arr.shape)
    arr[:] = data
    np.testing.assert_array_equal(arr[:], data)


@pytest.mark.parametrize("arr_kwargs", array_configs)
async def test_missing_chunks_fill_value(pipeline_class: str, arr_kwargs: dict[str, Any]) -> None:
    """Reading unwritten chunks returns the fill value."""
    store = MemoryStore()
    fill = -1
    arr = zarr.create_array(store=store, fill_value=fill, **arr_kwargs)
    expected = np.full(arr.shape, fill, dtype=arr.dtype)
    np.testing.assert_array_equal(arr[:], expected)


write_then_read_cases = [
    pytest.param(
        slice(None),
        np.s_[:],
        id="full-write-full-read",
    ),
    pytest.param(
        slice(5, 15),
        np.s_[:],
        id="partial-write-full-read",
    ),
    pytest.param(
        slice(None),
        np.s_[::3],
        id="full-write-strided-read",
    ),
    pytest.param(
        slice(None),
        np.s_[10:20],
        id="full-write-slice-read",
    ),
]


@pytest.mark.parametrize(
    "arr_kwargs",
    [
        pytest.param(
            {
                "shape": (100,),
                "dtype": "float64",
                "chunks": (10,),
                "shards": None,
                "compressors": None,
            },
            id="unsharded",
        ),
        pytest.param(
            {
                "shape": (100,),
                "dtype": "float64",
                "chunks": (10,),
                "shards": (100,),
                "compressors": None,
            },
            id="sharded",
        ),
    ],
)
@pytest.mark.parametrize(("write_sel", "read_sel"), write_then_read_cases)
async def test_write_then_read(
    pipeline_class: str,
    arr_kwargs: dict[str, Any],
    write_sel: slice,
    read_sel: slice,
) -> None:
    """Various write + read selection combinations produce correct results."""
    store = MemoryStore()
    arr = zarr.create_array(store=store, fill_value=0.0, **arr_kwargs)
    full = np.zeros(arr.shape, dtype=arr.dtype)

    write_data = np.arange(len(full[write_sel]), dtype=arr.dtype) + 1
    full[write_sel] = write_data
    arr[write_sel] = write_data

    np.testing.assert_array_equal(arr[read_sel], full[read_sel])


# ---------------------------------------------------------------------------
# write_empty_chunks / read_missing_chunks config tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arr_kwargs",
    [
        pytest.param(
            {
                "shape": (20,),
                "dtype": "float64",
                "chunks": (10,),
                "shards": None,
                "compressors": None,
            },
            id="unsharded",
        ),
        pytest.param(
            {
                "shape": (20,),
                "dtype": "float64",
                "chunks": (10,),
                "shards": (20,),
                "compressors": None,
            },
            id="sharded",
        ),
    ],
)
async def test_write_empty_chunks_false(pipeline_class: str, arr_kwargs: dict[str, Any]) -> None:
    """With write_empty_chunks=False, writing fill_value should not persist the chunk."""
    store = MemoryStore()
    arr = zarr.create_array(
        store=store,
        fill_value=0.0,
        config={"write_empty_chunks": False},
        **arr_kwargs,
    )
    # Write non-fill to first chunk, fill_value to second chunk
    arr[0:10] = np.arange(10, dtype="float64") + 1
    arr[10:20] = np.zeros(10, dtype="float64")  # all fill_value

    # Read back — both chunks should return correct data
    np.testing.assert_array_equal(arr[0:10], np.arange(10, dtype="float64") + 1)
    np.testing.assert_array_equal(arr[10:20], np.zeros(10, dtype="float64"))


async def test_write_empty_chunks_true(pipeline_class: str) -> None:
    """With write_empty_chunks=True, fill_value chunks should still be stored."""
    store: dict[str, Any] = {}
    arr = zarr.create_array(
        store=store,
        shape=(20,),
        dtype="float64",
        chunks=(10,),
        shards=None,
        compressors=None,
        fill_value=0.0,
        config={"write_empty_chunks": True},
    )
    arr[:] = 0.0  # all fill_value

    # With write_empty_chunks=True, chunks should be persisted even though
    # they equal the fill value.
    assert "c/0" in store
    assert "c/1" in store


async def test_write_empty_chunks_false_no_store(pipeline_class: str) -> None:
    """With write_empty_chunks=False, fill_value-only chunks should not be stored."""
    store: dict[str, Any] = {}
    arr = zarr.create_array(
        store=store,
        shape=(20,),
        dtype="float64",
        chunks=(10,),
        shards=None,
        compressors=None,
        fill_value=0.0,
        config={"write_empty_chunks": False},
    )
    arr[:] = 0.0  # all fill_value

    # Chunks should NOT be persisted
    assert "c/0" not in store
    assert "c/1" not in store

    # But reading should still return fill values
    np.testing.assert_array_equal(arr[:], np.zeros(20, dtype="float64"))


async def test_read_missing_chunks_false_raises(pipeline_class: str) -> None:
    """With read_missing_chunks=False, reading a missing chunk should raise."""
    store = MemoryStore()
    arr = zarr.create_array(
        store=store,
        shape=(20,),
        dtype="float64",
        chunks=(10,),
        shards=None,
        compressors=None,
        fill_value=0.0,
        config={"read_missing_chunks": False},
    )
    # Don't write anything — all chunks are missing
    with pytest.raises(ChunkNotFoundError):
        arr[:]


async def test_read_missing_chunks_true_fills(pipeline_class: str) -> None:
    """With read_missing_chunks=True (default), missing chunks return fill_value."""
    store = MemoryStore()
    arr = zarr.create_array(
        store=store,
        shape=(20,),
        dtype="float64",
        chunks=(10,),
        shards=None,
        compressors=None,
        fill_value=-999.0,
    )
    # Don't write anything
    np.testing.assert_array_equal(arr[:], np.full(20, -999.0))


async def test_nested_sharding_roundtrip(pipeline_class: str) -> None:
    """Nested sharding: data survives write/read roundtrip."""
    from zarr.codecs.bytes import BytesCodec
    from zarr.codecs.sharding import ShardingCodec

    inner_sharding = ShardingCodec(chunk_shape=(10,), codecs=[BytesCodec()])
    outer_sharding = ShardingCodec(chunk_shape=(50,), codecs=[inner_sharding])

    store = MemoryStore()
    arr = zarr.create_array(
        store=store,
        shape=(100,),
        dtype="uint8",
        chunks=(100,),
        compressors=None,
        fill_value=0,
        serializer=outer_sharding,
    )
    data = np.arange(100, dtype="uint8")
    arr[:] = data
    np.testing.assert_array_equal(arr[:], data)
    # Partial read
    np.testing.assert_array_equal(arr[40:60], data[40:60])
