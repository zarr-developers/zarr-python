"""
Benchmarks for end-to-end read/write performance of Zarr
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests.benchmarks.common import Layout

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pytest_benchmark.fixture import BenchmarkFixture

    from zarr.abc.store import Store
    from zarr.core.common import NamedConfig
from operator import getitem, setitem
from typing import Any, Literal

import pytest

from zarr import create_array
from zarr.core.config import config as zarr_config

CompressorName = Literal["gzip"] | None

compressors: dict[CompressorName, NamedConfig[Any, Any] | None] = {
    None: None,
    "gzip": {"name": "gzip", "configuration": {"level": 1}},
}


layouts: tuple[Layout, ...] = (
    # No shards, just 1000 chunks
    Layout(shape=(1_000_000,), chunks=(1000,), shards=None),
    # 1:1 chunk:shard shape, should measure overhead of sharding
    Layout(shape=(1_000_000,), chunks=(1000,), shards=(1000,)),
    # One shard with all the chunks, should measure overhead of handling inner shard chunks
    Layout(shape=(1_000_000,), chunks=(100,), shards=(10000 * 100,)),
)

_PIPELINE_PATHS = {
    "batched": "zarr.core.codec_pipeline.BatchedCodecPipeline",
    "fused": "zarr.core.codec_pipeline.FusedCodecPipeline",
}


@pytest.fixture(params=["batched", "fused"])
def pipeline(request: pytest.FixtureRequest) -> Iterator[str]:
    """Set ``codec_pipeline.path`` for the duration of the benchmark.

    Yields the pipeline name so each parametrize cell has a distinct
    benchmark id.
    """
    name = request.param
    with zarr_config.set({"codec_pipeline.path": _PIPELINE_PATHS[name]}):
        yield name


@pytest.mark.parametrize("compression_name", [None, "gzip"])
@pytest.mark.parametrize("layout", layouts, ids=str)
@pytest.mark.parametrize("store", ["memory", "local"], indirect=["store"])
def test_write_array(
    store: Store,
    layout: Layout,
    compression_name: CompressorName,
    pipeline: str,
    benchmark: BenchmarkFixture,
) -> None:
    """
    Test the time required to fill an array with a single value
    """
    arr = create_array(
        store,
        dtype="uint8",
        shape=layout.shape,
        chunks=layout.chunks,
        shards=layout.shards,
        compressors=compressors[compression_name],  # type: ignore[arg-type]
        fill_value=0,
    )

    benchmark(setitem, arr, Ellipsis, 1)


@pytest.mark.parametrize("compression_name", [None, "gzip"])
@pytest.mark.parametrize("layout", layouts, ids=str)
@pytest.mark.parametrize("store", ["memory", "local"], indirect=["store"])
def test_read_array(
    store: Store,
    layout: Layout,
    compression_name: CompressorName,
    pipeline: str,
    benchmark: BenchmarkFixture,
) -> None:
    """
    Test the time required to fill an array with a single value
    """
    arr = create_array(
        store,
        dtype="uint8",
        shape=layout.shape,
        chunks=layout.chunks,
        shards=layout.shards,
        compressors=compressors[compression_name],  # type: ignore[arg-type]
        fill_value=0,
    )
    arr[:] = 1
    benchmark(getitem, arr, Ellipsis)
