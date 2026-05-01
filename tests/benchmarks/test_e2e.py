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
from zarr.testing.store import LatencyStore

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

_LATENCY_VALUES = (0.0, 0.001, 0.05, 0.2)


@pytest.fixture(params=_LATENCY_VALUES, ids=lambda v: f"latency={v}")
def latency(request: pytest.FixtureRequest) -> float:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture
def bench_store(store: Store, latency: float, request: pytest.FixtureRequest) -> Store:
    """Wraps the underlying store in LatencyStore when latency > 0.

    Local-store cases skip nonzero latency — synthetic latency on top of
    a real LocalStore is double-counting; latency simulation only applies
    to the in-process memory store.
    """
    callspec = getattr(request.node, "callspec", None)
    store_kind = callspec.params.get("store", "memory") if callspec is not None else "memory"
    if latency > 0:
        if store_kind == "local":
            pytest.skip("latency injection only applies to in-memory store")
        return LatencyStore(store, get_latency=latency, set_latency=latency)
    return store


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
    bench_store: Store,
    layout: Layout,
    compression_name: CompressorName,
    pipeline: str,
    benchmark: BenchmarkFixture,
) -> None:
    """
    Test the time required to fill an array with a single value
    """
    arr = create_array(
        bench_store,
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
    bench_store: Store,
    layout: Layout,
    compression_name: CompressorName,
    pipeline: str,
    benchmark: BenchmarkFixture,
) -> None:
    """
    Test the time required to fill an array with a single value
    """
    arr = create_array(
        bench_store,
        dtype="uint8",
        shape=layout.shape,
        chunks=layout.chunks,
        shards=layout.shards,
        compressors=compressors[compression_name],  # type: ignore[arg-type]
        fill_value=0,
    )
    arr[:] = 1
    benchmark(getitem, arr, Ellipsis)
