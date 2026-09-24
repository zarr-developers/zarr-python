import asyncio
from collections.abc import Generator
from itertools import islice, product
from unittest import mock

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from zarr.core.array import ShardsLike, create_array
from zarr.core.buffer import default_buffer_prototype
from zarr.core.chunk_grids import ChunkGrid
from zarr.core.common import ChunksLike, ZarrFormat
from zarr.core.config import config
from zarr.storage import MemoryStore


@pytest.fixture(autouse=True)
def _enable_rectilinear_chunks() -> Generator[None, None, None]:
    with config.set({"array.rectilinear_chunks": True}):
        yield


@given(st.lists(st.tuples(st.integers(0, 8), st.integers(0, 8)), max_size=4))
def test_resize_chunk_difference_matches_sets(dimensions: list[tuple[int, int]]) -> None:
    from zarr.core.array import _iter_chunk_coords_to_delete

    old_shape = tuple(old for old, _ in dimensions)
    new_shape = tuple(new for _, new in dimensions)
    expected = set(product(*(range(n) for n in old_shape))) - set(
        product(*(range(n) for n in new_shape))
    )
    actual = list(_iter_chunk_coords_to_delete(old_shape, new_shape))
    assert set(actual) == expected
    assert len(actual) == len(expected)


def test_resize_chunk_difference_is_lazy_for_large_axes() -> None:
    from zarr.core.array import _iter_chunk_coords_to_delete

    # Both the per-axis ranges and their product are too large to materialize.
    coords = _iter_chunk_coords_to_delete((2**40, 2**40), (0, 1))
    assert list(islice(coords, 3)) == [(0, 0), (0, 1), (0, 2)]


async def test_resize_shrinking_skips_full_grid(zarr_format: ZarrFormat) -> None:
    store = MemoryStore()
    array = await create_array(
        store, shape=(2**40, 2), chunks=(1, 1), dtype="uint8", zarr_format=zarr_format
    )
    expected = [array.metadata.encode_chunk_key((2**40 - 1, i)) for i in range(2)]
    # Fail safely on the old implementation instead of exhausting the test machine.
    with (
        mock.patch.object(ChunkGrid, "all_chunk_coords", side_effect=AssertionError("full grid")),
        mock.patch.object(store, "delete", wraps=store.delete) as delete,
    ):
        await array.resize((2**40 - 1, 2))
    assert array.shape == (2**40 - 1, 2)
    assert sorted(call.args[0] for call in delete.call_args_list) == expected


@pytest.mark.parametrize("limit", [1, 4, None])
async def test_resize_bounds_pending_deletions(limit: int | None) -> None:
    store = MemoryStore()
    array = await create_array(store, shape=(3000,), chunks=(1,), dtype="uint8")
    initial_tasks = len(asyncio.all_tasks())
    workers = limit if limit is not None else 1000
    peak_tasks = 0
    in_flight = 0
    in_flight_at_start: list[int] = []

    async def delete(key: str) -> None:
        nonlocal peak_tasks, in_flight
        peak_tasks = max(peak_tasks, len(asyncio.all_tasks()) - initial_tasks)
        in_flight_at_start.append(in_flight)
        in_flight += 1
        await asyncio.sleep(0)
        in_flight -= 1

    with config.set({"async.concurrency": limit}), mock.patch.object(store, "delete", delete):
        await array.resize((0,))
    assert len(in_flight_at_start) == 3000
    assert 0 < peak_tasks <= workers
    # Once the pool is full, each deletion replaces a finished one instead of
    # waiting for a whole batch to drain.
    assert set(in_flight_at_start[workers:]) == {workers - 1}


@pytest.mark.parametrize("new_shape", [(4, 5), (10, 5), (4, 12), (7, 8), (0, 9)])
@pytest.mark.parametrize(
    ("zarr_format", "chunks", "shards"),
    [
        (2, (3, 4), None),
        (3, (3, 4), None),
        (3, ((2, 3, 3), (3, 2, 4)), None),
        (3, (1, 1), (3, 4)),
        (3, (1, 1), ((2, 3, 3), (3, 2, 4))),
    ],
    ids=["v2", "v3", "rectilinear", "sharded", "rectilinear-sharded"],
)
@pytest.mark.parametrize("delete_outside_chunks", [True, False])
async def test_resize_keeps_retained_chunk_bytes(
    new_shape: tuple[int, ...],
    zarr_format: ZarrFormat,
    chunks: ChunksLike,
    shards: ShardsLike,
    delete_outside_chunks: bool,
) -> None:
    store = MemoryStore()
    array = await create_array(
        store, shape=(8, 9), chunks=chunks, shards=shards, dtype="int32", zarr_format=zarr_format
    )
    data = np.arange(1, 73, dtype="int32").reshape(8, 9)
    await array.setitem(..., data)
    before = {key: await store.get(key, default_buffer_prototype()) async for key in store.list()}
    old_coords = set(array._chunk_grid.all_chunk_coords())
    new_grid = ChunkGrid.from_metadata(array.metadata.update_shape(new_shape))
    removed = old_coords - set(new_grid.all_chunk_coords()) if delete_outside_chunks else set()
    removed_keys = {array.metadata.encode_chunk_key(coords) for coords in removed}

    with mock.patch.object(store, "delete", wraps=store.delete) as delete:
        await array.resize(new_shape, delete_outside_chunks=delete_outside_chunks)

    assert array.shape == new_shape
    assert {call.args[0] for call in delete.call_args_list} == removed_keys
    assert delete.call_count == len(removed_keys)
    remaining_keys = {key async for key in store.list()}
    assert remaining_keys == before.keys() - removed_keys
    for coords in old_coords - removed:
        key = array.metadata.encode_chunk_key(coords)
        original = before[key]
        current = await store.get(key, default_buffer_prototype())
        assert original is not None
        assert current is not None
        assert current.to_bytes() == original.to_bytes()

    overlap = tuple(slice(0, min(old, new)) for old, new in zip((8, 9), new_shape, strict=True))
    np.testing.assert_array_equal(await array.getitem(overlap), data[overlap])


async def test_resize_delete_failure_preserves_metadata() -> None:
    store = MemoryStore()
    array = await create_array(store, shape=(4,), chunks=(1,), dtype="uint8")
    metadata_before = await store.get("zarr.json", default_buffer_prototype())
    grid_before = array._chunk_grid
    with (
        mock.patch.object(store, "delete", side_effect=OSError("delete failed")),
        pytest.raises(OSError, match="delete failed"),
    ):
        await array.resize((0,))
    metadata_after = await store.get("zarr.json", default_buffer_prototype())
    assert metadata_before is not None
    assert metadata_after is not None
    assert metadata_before.to_bytes() == metadata_after.to_bytes()
    assert array.shape == (4,)
    assert array._chunk_grid is grid_before
