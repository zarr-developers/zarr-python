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
    """
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
    benchmark(getitem, data, indexer)
