"""Integration tests for RectilinearChunkGrid with array creation."""

from typing import Literal

import numpy as np
import pytest

import zarr
from zarr.core.chunk_grids import RectilinearChunkGrid
from zarr.storage import MemoryStore


@pytest.mark.parametrize("zarr_format", [3])
async def test_create_array_with_nested_chunks(zarr_format: Literal[2, 3]) -> None:
    """
    Test creating an array with nested chunk specification (RectilinearChunkGrid).
    This is an end-to-end test for the feature.
    """
    store = MemoryStore()
    arr = await zarr.api.asynchronous.create_array(
        store=store,
        shape=(60, 100),
        chunks=[[10, 20, 30], [25, 25, 25, 25]],
        dtype="i4",
        zarr_format=zarr_format,
    )

    # Verify metadata has RectilinearChunkGrid
    assert isinstance(arr.metadata.chunk_grid, RectilinearChunkGrid)
    assert arr.metadata.chunk_grid.chunk_shapes == ((10, 20, 30), (25, 25, 25, 25))

    # Verify array is functional - can write and read data
    data = np.arange(60 * 100, dtype="i4").reshape(60, 100)
    await arr.setitem(slice(None), data)
    result = await arr.getitem(slice(None))
    np.testing.assert_array_equal(result, data)


async def test_create_array_nested_chunks_read_write() -> None:
    """
    Test that arrays with RectilinearChunkGrid support standard read/write operations.
    """
    store = MemoryStore()
    arr = await zarr.api.asynchronous.create_array(
        store=store,
        shape=(30, 40),
        chunks=[[10, 10, 10], [10, 10, 10, 10]],
        dtype="f4",
        zarr_format=3,
    )

    # Write data to different chunks
    arr_data = np.random.random((30, 40)).astype("f4")
    await arr.setitem(slice(None), arr_data)

    # Read full array
    result = await arr.getitem(slice(None))
    np.testing.assert_array_almost_equal(np.asarray(result), arr_data)

    # Read partial slices
    partial = await arr.getitem((slice(5, 25), slice(10, 30)))
    np.testing.assert_array_almost_equal(np.asarray(partial), arr_data[5:25, 10:30])


async def test_rectilinear_chunk_grid_roundtrip() -> None:
    """
    Test that RectilinearChunkGrid persists correctly through save/load.
    """
    store = MemoryStore()

    # Create array with nested chunks
    arr1 = await zarr.api.asynchronous.create_array(
        store=store,
        name="test_array",
        shape=(60, 80),
        chunks=[[10, 20, 30], [20, 20, 20, 20]],
        dtype="u1",
        zarr_format=3,
    )

    # Write some data
    data = np.arange(60 * 80, dtype="u1").reshape(60, 80)
    await arr1.setitem(slice(None), data)

    # Re-open the array
    arr2 = await zarr.api.asynchronous.open_array(store=store, path="test_array")

    # Verify chunk_grid is preserved
    assert isinstance(arr2.metadata.chunk_grid, RectilinearChunkGrid)
    assert arr2.metadata.chunk_grid.chunk_shapes == ((10, 20, 30), (20, 20, 20, 20))

    # Verify data is preserved
    result = await arr2.getitem(slice(None))
    np.testing.assert_array_equal(result, data)


async def test_from_array_rejects_nested_chunks() -> None:
    """
    Test that from_array rejects nested chunks (RectilinearChunkGrid) with has_data=True.
    """
    store = MemoryStore()
    data = np.arange(30 * 40, dtype="i4").reshape(30, 40)

    # Should raise error because RectilinearChunkGrid is not compatible with has_data=True
    with pytest.raises(
        ValueError,
        match="Cannot use RectilinearChunkGrid.*when creating array from data",
    ):
        await zarr.api.asynchronous.from_array(
            store=store,
            data=data,
            chunks=[[10, 10, 10], [10, 10, 10, 10]],  # type: ignore[arg-type]
            zarr_format=3,
        )


async def test_nested_chunks_with_different_sizes() -> None:
    """
    Test RectilinearChunkGrid with highly irregular chunk sizes.
    """
    store = MemoryStore()
    arr = await zarr.api.asynchronous.create_array(
        store=store,
        shape=(100, 100),
        chunks=[[5, 10, 15, 20, 50], [100]],  # Very irregular first dim, uniform second
        dtype="i2",
        zarr_format=3,
    )

    assert isinstance(arr.metadata.chunk_grid, RectilinearChunkGrid)
    assert arr.metadata.chunk_grid.chunk_shapes == ((5, 10, 15, 20, 50), (100,))

    # Verify writes work correctly
    data = np.arange(100 * 100, dtype="i2").reshape(100, 100)
    await arr.setitem(slice(None), data)
    result = await arr.getitem(slice(None))
    np.testing.assert_array_equal(result, data)


async def test_rectilinear_chunk_grid_nchunks_not_supported() -> None:
    """
    Test that nchunks property raises NotImplementedError for RectilinearChunkGrid.

    Note: The chunks property (and thus nchunks) is only defined for RegularChunkGrid.
    For RectilinearChunkGrid, use chunk_grid.get_nchunks() instead.
    """
    store = MemoryStore()
    arr = await zarr.api.asynchronous.create_array(
        store=store,
        shape=(60, 100),
        chunks=[[10, 20, 30], [25, 25, 25, 25]],
        dtype="u1",
        zarr_format=3,
    )

    # The chunks property is not defined for RectilinearChunkGrid
    with pytest.raises(
        NotImplementedError, match="only defined for arrays using.*RegularChunkGrid"
    ):
        _ = arr.nchunks

    # But we can get nchunks from the chunk_grid directly
    assert arr.metadata.chunk_grid.get_nchunks((60, 100)) == 12
