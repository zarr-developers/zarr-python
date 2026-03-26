from __future__ import annotations

import pytest

import zarr
from zarr.core.buffer.core import default_buffer_prototype
from zarr.core.indexing import BasicIndexer
from zarr.storage import MemoryStore


@pytest.mark.parametrize(
    ("write_slice", "read_slice", "expected_statuses"),
    [
        # Write all chunks, read all — all present
        (slice(None), slice(None), ("present", "present", "present")),
        # Write first chunk only, read all — first present, rest missing
        (slice(0, 2), slice(None), ("present", "missing", "missing")),
        # Write nothing, read all — all missing
        (None, slice(None), ("missing", "missing", "missing")),
    ],
)
async def test_read_returns_get_results(
    write_slice: slice | None,
    read_slice: slice,
    expected_statuses: tuple[str, ...],
) -> None:
    """
    Test that CodecPipeline.read returns a tuple of GetResult with correct statuses.
    """
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
        chunk_grid=metadata.chunk_grid,
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
                metadata.get_chunk_spec(chunk_coords, config, prototype=prototype),
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
