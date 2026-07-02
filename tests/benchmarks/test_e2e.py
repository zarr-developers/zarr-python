"""
Benchmarks for end-to-end read/write performance of Zarr
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
import os
import numpy as np
from typing import TYPE_CHECKING

from tests.benchmarks.common import Layout


import platform
import subprocess

def clear_cache():
    if platform.system() == "Darwin":
        subprocess.call(['sync', '&&', 'sudo', 'purge'])
    elif platform.system() == "Linux":
        subprocess.call(['sudo', 'sh', '-c', "sync; echo 3 > /proc/sys/vm/drop_caches"])
    else:
        raise Exception("Unsupported platform")

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

@lru_cache
def _data(shape: tuple[int]) -> np.ndarray:
    n = shape[0]
    period = 256
    noise_level = 1
    pattern = (np.sin(np.linspace(0, 2 * np.pi, period)) * 50 + 128).round().astype(np.uint8)
    data = np.tile(pattern, int(np.ceil(n / period)))[:n].astype(np.int16)
    data += np.random.randint(-noise_level, noise_level + 1, size=n, dtype=np.int16)
    return np.clip(data, 0, 255).astype(np.uint8)


CompressorName = Literal["zstd"] | None

compressors: dict[CompressorName, NamedConfig[Any, Any] | None] = {
    None: None,
    # Default v3
    "zstd": {"name": "zstd", "configuration": {"level": 0, "checksum": False}}
}


layouts: tuple[Layout, ...] = (
    # No shards, just 1000 chunks
    Layout(shape=(100_000_000,), chunks=(100_000,), shards=None),
    # 1:1 chunk:shard shape, should measure overhead of sharding
    Layout(shape=(100_000_000,), chunks=(100_000,), shards=(100_000,)),
    # One shard with all the chunks, should measure over/under-head of handling inner shard chunks
    Layout(shape=(100_000_000,), chunks=(100_000,), shards=(100_000 * 1_000,)),
    # Mixed layout balancing inner vs. outer concurrency (likely the most real-world case)
    Layout(shape=(1_000_000_000,), chunks=(100_000,), shards=(100_000 * 100,)),
)

_PIPELINE_SETTINGS = {
    "batched": {"codec_pipeline.path": "zarr.core.codec_pipeline.BatchedCodecPipeline" },
    "fused_full_threaded": {"codec_pipeline.path": "zarr.core.codec_pipeline.FusedCodecPipeline","codec_pipeline.max_workers": None},
    "fused_single_threaded": {"codec_pipeline.path": "zarr.core.codec_pipeline.FusedCodecPipeline","codec_pipeline.max_workers": 1},
}

_LATENCY_VALUES = (0, 0.03)


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
        return LatencyStore(
            store, get_latency=(latency, latency * 1.2), set_latency=(latency, latency * 1.2)
        )
    if store_kind == "memory":
        pytest.skip("memory store doesn't offer much over local without latency")
    return store


@pytest.fixture(params=["batched", "fused_full_threaded", "fused_single_threaded"])
def pipeline(request: pytest.FixtureRequest) -> Iterator[str]:
    """Set ``codec_pipeline.path`` for the duration of the benchmark.

    Yields the pipeline name so each parametrize cell has a distinct
    benchmark id.
    """
    name = request.param
    with zarr_config.set(_PIPELINE_SETTINGS[name]):
        yield name

@pytest.mark.parametrize("get_data", [lambda shape: 1, lambda shape: _data(shape)], ids=["repeated", "semi_random"])
@pytest.mark.parametrize("compression_name", ["zstd"])
@pytest.mark.parametrize("layout", layouts, ids=str)
@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
def test_write_array(
    bench_store: Store,
    layout: Layout,
    compression_name: CompressorName,
    pipeline: str,
    benchmark: BenchmarkFixture,
    get_data: Callable[[tuple[int]], np.ndarray | int]
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
    def setup():
        clear_cache()
        return (arr, Ellipsis, get_data(layout.shape)), {}
    benchmark.pedantic(setitem, setup=setup, rounds=3)

@pytest.mark.parametrize("get_data", [lambda shape: 1, lambda shape: _data(shape)], ids=["repeated", "semi_random"])
@pytest.mark.parametrize("compression_name", ["zstd"])
@pytest.mark.parametrize("layout", layouts, ids=str)
@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
def test_read_array(
    bench_store: Store,
    layout: Layout,
    compression_name: CompressorName,
    pipeline: str,
    benchmark: BenchmarkFixture,
    get_data: Callable[[tuple[int]], np.ndarray | int]
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
    arr[:] = get_data(layout.shape)
    def setup():
        clear_cache()
        return (arr, Ellipsis), {}
    benchmark.pedantic(getitem, setup=setup, rounds=3)
