from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import zarr
from zarr.codecs import BytesCodec, CastValue
from zarr.core.array import _get_chunk_spec
from zarr.core.buffer.core import default_buffer_prototype
from zarr.core.config import config as zarr_config
from zarr.core.indexing import BasicIndexer
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
    "zarr.core.codec_pipeline.FusedCodecPipeline",
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
# write_empty_chunks / read_missing_chunks config tests
# ---------------------------------------------------------------------------


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


try:
    import cast_value_rs  # noqa: F401

    _HAS_CAST_VALUE_RS = True
except ModuleNotFoundError:
    _HAS_CAST_VALUE_RS = False

requires_cast_value_rs = pytest.mark.skipif(
    not _HAS_CAST_VALUE_RS, reason="cast-value-rs not installed"
)


@requires_cast_value_rs
@pytest.mark.parametrize(
    ("source_dtype", "target_dtype"),
    [
        # Source is single-byte (no endianness); target is multi-byte (has endianness).
        # Without the fix, BytesCodec.evolve_from_array_spec sees the source dtype,
        # strips its `endian` to None, and then chokes when the chunk_spec dtype
        # gets transformed to the multi-byte target before bytes-decoding.
        ("int8", "int16"),
        ("uint8", "int32"),
        ("int8", "float32"),
        # Source is multi-byte; target is single-byte (the reverse direction also
        # exercises the spec-threading logic).
        ("int16", "int8"),
    ],
)
def test_codec_pipeline_threads_dtype_through_evolve(source_dtype: str, target_dtype: str) -> None:
    """Regression for #3937: each codec must be evolved against the spec it
    will see at runtime, not the original array spec. cast_value transforms
    the dtype between AA codecs and the array->bytes serializer."""
    arr = zarr.create_array(
        store={},
        shape=(4,),
        chunks=(4,),
        dtype=source_dtype,
        fill_value=0,
        filters=[CastValue(data_type=target_dtype)],
        serializer=BytesCodec(endian="little"),
        compressors=[],
        zarr_format=3,
        overwrite=True,
    )
    arr[:] = np.asarray([0, 1, 2, 3], dtype=source_dtype)
    np.testing.assert_array_equal(arr[:], np.asarray([0, 1, 2, 3], dtype=source_dtype))
