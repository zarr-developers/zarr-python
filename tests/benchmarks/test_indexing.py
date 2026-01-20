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


@pytest.mark.parametrize("store", ["memory", "local"], indirect=["store"])
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
