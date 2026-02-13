from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

    from zarr.abc.store import Store

from operator import getitem

import pytest

from zarr import create_array

indexers = (
    (0,) * 3,
    (slice(None),) * 3,
    (slice(0, None, 4),) * 3,
    (slice(10),) * 3,
    (slice(10, -10, 4),) * 3,
    (slice(None), slice(0, 3, 2), slice(0, 10)),
)

shards = (
    None,
    (50,) * 3,
)


@pytest.mark.parametrize("store", ["memory", "memory_get_latency"], indirect=["store"])
@pytest.mark.parametrize("indexer", indexers, ids=str)
@pytest.mark.parametrize("shards", shards, ids=str)
def test_slice_indexing(
    store: Store,
    indexer: tuple[int | slice],
    shards: tuple[int, ...] | None,
    benchmark: BenchmarkFixture,
) -> None:
    data = create_array(
        store=store,
        shape=(105,) * 3,
        dtype="uint8",
        chunks=(10,) * 3,
        shards=shards,
        compressors=None,
        filters=None,
        fill_value=0,
    )

    data[:] = 1
    benchmark(getitem, data, indexer)


# Benchmark for Morton order optimization with power-of-2 shards
# Morton order is used internally by sharding codec for chunk iteration
morton_shards = (
    (16,) * 3,  # With 2x2x2 chunks: 8x8x8 = 512 chunks per shard
    (32,) * 3,  # With 2x2x2 chunks: 16x16x16 = 4096 chunks per shard
)


@pytest.mark.parametrize("store", ["memory"], indirect=["store"])
@pytest.mark.parametrize("shards", morton_shards, ids=str)
def test_sharded_morton_indexing(
    store: Store,
    shards: tuple[int, ...],
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmark sharded array indexing with power-of-2 chunks per shard.

    This benchmark exercises the Morton order iteration path in the sharding
    codec, which benefits from the hypercube and vectorization optimizations.
    The Morton order cache is cleared before each iteration to measure the
    full computation cost.
    """
    from zarr.core.indexing import _morton_order

    # Create array where each shard contains many small chunks
    # e.g., shards=(32,32,32) with chunks=(2,2,2) means 16x16x16 = 4096 chunks per shard
    shape = tuple(s * 2 for s in shards)  # 2 shards per dimension
    chunks = (2,) * 3  # Small chunks to maximize chunks per shard

    data = create_array(
        store=store,
        shape=shape,
        dtype="uint8",
        chunks=chunks,
        shards=shards,
        compressors=None,
        filters=None,
        fill_value=0,
    )

    data[:] = 1
    # Read a sub-shard region to exercise Morton order iteration
    indexer = (slice(shards[0]),) * 3

    def read_with_cache_clear() -> None:
        _morton_order.cache_clear()
        getitem(data, indexer)

    benchmark(read_with_cache_clear)


# Benchmark with larger chunks_per_shard to make Morton order impact more visible
large_morton_shards = (
    (32,) * 3,  # With 1x1x1 chunks: 32x32x32 = 32768 chunks per shard
)


@pytest.mark.parametrize("store", ["memory"], indirect=["store"])
@pytest.mark.parametrize("shards", large_morton_shards, ids=str)
def test_sharded_morton_indexing_large(
    store: Store,
    shards: tuple[int, ...],
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmark sharded array indexing with large chunks_per_shard.

    Uses 1x1x1 chunks to maximize chunks_per_shard (32^3 = 32768), making
    the Morton order computation a more significant portion of total time.
    The Morton order cache is cleared before each iteration.
    """
    from zarr.core.indexing import _morton_order

    # 1x1x1 chunks means chunks_per_shard equals shard shape
    shape = tuple(s * 2 for s in shards)  # 2 shards per dimension
    chunks = (1,) * 3  # 1x1x1 chunks: chunks_per_shard = shards

    data = create_array(
        store=store,
        shape=shape,
        dtype="uint8",
        chunks=chunks,
        shards=shards,
        compressors=None,
        filters=None,
        fill_value=0,
    )

    data[:] = 1
    # Read one full shard
    indexer = (slice(shards[0]),) * 3

    def read_with_cache_clear() -> None:
        _morton_order.cache_clear()
        getitem(data, indexer)

    benchmark(read_with_cache_clear)


@pytest.mark.parametrize("store", ["memory"], indirect=["store"])
@pytest.mark.parametrize("shards", large_morton_shards, ids=str)
def test_sharded_morton_single_chunk(
    store: Store,
    shards: tuple[int, ...],
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmark reading a single chunk from a large shard.

    This isolates the Morton order computation overhead by minimizing I/O.
    Reading one chunk from a shard with 32^3 = 32768 chunks still requires
    computing the full Morton order, making the optimization impact clear.
    The Morton order cache is cleared before each iteration.
    """
    from zarr.core.indexing import _morton_order

    # 1x1x1 chunks means chunks_per_shard equals shard shape
    shape = tuple(s * 2 for s in shards)  # 2 shards per dimension
    chunks = (1,) * 3  # 1x1x1 chunks: chunks_per_shard = shards

    data = create_array(
        store=store,
        shape=shape,
        dtype="uint8",
        chunks=chunks,
        shards=shards,
        compressors=None,
        filters=None,
        fill_value=0,
    )

    data[:] = 1
    # Read only a single chunk (1x1x1) from the shard
    indexer = (slice(1),) * 3

    def read_with_cache_clear() -> None:
        _morton_order.cache_clear()
        getitem(data, indexer)

    benchmark(read_with_cache_clear)


# Benchmark for morton_order_iter directly (no I/O)
morton_iter_shapes = (
    (8, 8, 8),  # 512 elements
    (16, 16, 16),  # 4096 elements
    (32, 32, 32),  # 32768 elements
)


@pytest.mark.parametrize("shape", morton_iter_shapes, ids=str)
def test_morton_order_iter(
    shape: tuple[int, ...],
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmark morton_order_iter directly without I/O.

    This isolates the Morton order computation to measure the
    optimization impact without array read/write overhead.
    The cache is cleared before each iteration.
    """
    from zarr.core.indexing import _morton_order, morton_order_iter

    def compute_morton_order() -> None:
        _morton_order.cache_clear()
        # Consume the iterator to force computation
        list(morton_order_iter(shape))

    benchmark(compute_morton_order)
