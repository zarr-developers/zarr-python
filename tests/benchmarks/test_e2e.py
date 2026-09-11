"""
Benchmarks for end-to-end read/write performance of Zarr
"""

from __future__ import annotations

import os
import platform
import subprocess
import warnings
from functools import lru_cache
from operator import getitem, setitem
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pytest

import zarr
from tests.benchmarks.common import Layout
from zarr import create_array
from zarr.core.config import config as zarr_config
from zarr.testing.store import LatencyStore


def clear_cache() -> None:
    """Drop the OS page cache between benchmark rounds.

    Requires passwordless sudo, so it is opt-in: set `ZARR_BENCHMARK_CLEAR_CACHE=1`
    to enable it (as the benchmark CI jobs do). By default this is a no-op, so a
    plain `pytest` run never prompts for a sudo password (see issue #4199).
    `sudo -n` guarantees we fail instead of blocking on a password prompt even
    when the variable is set.
    """
    if os.environ.get("ZARR_BENCHMARK_CLEAR_CACHE", "") not in ("1", "true"):
        return
    if platform.system() == "Darwin":
        subprocess.call(["sync"])
        subprocess.call(["sudo", "-n", "purge"])
    elif platform.system() == "Linux":
        subprocess.call(["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"])
    else:
        warnings.warn(
            f"ZARR_BENCHMARK_CLEAR_CACHE is set but cache clearing is not supported on "
            f"{platform.system()}; skipping.",
            stacklevel=2,
        )


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from types import EllipsisType

    from pytest_benchmark.fixture import BenchmarkFixture

    from zarr.abc.store import Store
    from zarr.core.common import NamedConfig


@lru_cache
def _data(shape: tuple[int]) -> np.ndarray:
    n = shape[0]
    period = 256
    noise_level = 1
    pattern = (np.sin(np.linspace(0, 2 * np.pi, period)) * 50 + 128).round().astype(np.uint8)
    data = np.tile(pattern, int(np.ceil(n / period)))[:n].astype(np.int16)
    rng = np.random.default_rng(0)
    data += rng.integers(-noise_level, noise_level + 1, size=n, dtype=np.int16)
    return np.clip(data, 0, 255).astype(np.uint8)


CompressorName = Literal["zstd"] | None

compressors: dict[CompressorName, NamedConfig[Any, Any] | None] = {
    None: None,
    # Default v3
    "zstd": {"name": "zstd", "configuration": {"level": 0, "checksum": False}},
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
    "batched": {"codec_pipeline.path": "zarr.core.codec_pipeline.BatchedCodecPipeline"},
    "fused_full_threaded": {
        "codec_pipeline.path": "zarr.core.codec_pipeline.FusedCodecPipeline",
        "codec_pipeline.max_workers": None,
    },
    "fused_single_threaded": {
        "codec_pipeline.path": "zarr.core.codec_pipeline.FusedCodecPipeline",
        "codec_pipeline.max_workers": 1,
    },
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


@pytest.fixture(params=["batched"])  # , "fused_full_threaded", "fused_single_threaded"])
def pipeline(request: pytest.FixtureRequest) -> Iterator[str]:
    """Set ``codec_pipeline.path`` for the duration of the benchmark.

    Yields the pipeline name so each parametrize cell has a distinct
    benchmark id.
    """
    name = request.param
    with zarr_config.set(_PIPELINE_SETTINGS[name]):
        yield name


@pytest.mark.parametrize(
    "get_data", [lambda shape: 1, lambda shape: _data(shape)], ids=["repeated", "semi_random"]
)
@pytest.mark.parametrize("compression_name", ["zstd", None])
@pytest.mark.parametrize("layout", layouts, ids=str)
@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
def test_write_array(
    bench_store: Store,
    layout: Layout,
    compression_name: CompressorName,
    pipeline: str,
    benchmark: BenchmarkFixture,
    get_data: Callable[[tuple[int]], np.ndarray | int],
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

    def setup() -> tuple[tuple[zarr.Array, EllipsisType, int | np.ndarray], dict]:  # type: ignore[type-arg]
        clear_cache()
        return (arr, Ellipsis, get_data(layout.shape)), {}  # type: ignore[arg-type]

    benchmark.pedantic(setitem, setup=setup, rounds=3)  # type: ignore[no-untyped-call]


@pytest.mark.parametrize(
    "get_data", [lambda shape: 1, lambda shape: _data(shape)], ids=["repeated", "semi_random"]
)
@pytest.mark.parametrize("compression_name", ["zstd", None])
@pytest.mark.parametrize("layout", layouts, ids=str)
@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
def test_read_array(
    bench_store: Store,
    layout: Layout,
    compression_name: CompressorName,
    pipeline: str,
    benchmark: BenchmarkFixture,
    get_data: Callable[[tuple[int]], np.ndarray | int],
) -> None:
    """
    Test the time required to read the entirety of an array
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
    arr[:] = get_data(layout.shape)  # type: ignore[arg-type]

    def setup() -> tuple[tuple[zarr.Array, EllipsisType], dict]:  # type: ignore[type-arg]
        clear_cache()
        return (arr, Ellipsis), {}

    benchmark.pedantic(getitem, setup=setup, rounds=3)  # type: ignore[no-untyped-call]


_CONCURRENT_READ_THREADS = 8
_concurrent_layout = Layout(shape=(64_000_000,), chunks=(4_000_000,), shards=None)


@pytest.mark.parametrize("pipeline", ["batched", "fused_full_threaded"], indirect=True)
@pytest.mark.parametrize("compression_name", ["zstd", None])
@pytest.mark.parametrize("store", ["local"], indirect=["store"])
def test_read_array_concurrent(
    bench_store: Store,
    compression_name: CompressorName,
    pipeline: str,
    benchmark: BenchmarkFixture,
) -> None:
    """Dask-style access: several user threads each reading one chunk per call.

    All sync-API calls are serviced by the one global event loop, so this
    measures how much of each read's IO+compute the pipeline runs while
    holding the loop: anything inline serializes the readers.
    """
    from concurrent.futures import ThreadPoolExecutor

    layout = _concurrent_layout
    arr = create_array(
        bench_store,
        dtype="uint8",
        shape=layout.shape,
        chunks=layout.chunks,
        shards=layout.shards,
        compressors=compressors[compression_name],  # type: ignore[arg-type]
        fill_value=0,
    )
    arr[:] = _data(layout.shape)
    selections = [
        slice(start, start + layout.chunks[0])
        for start in range(0, layout.shape[0], layout.chunks[0])
    ]

    def read_all_chunks_concurrently() -> None:
        with ThreadPoolExecutor(max_workers=_CONCURRENT_READ_THREADS) as executor:
            list(executor.map(lambda sel: arr[sel], selections))

    def setup() -> tuple[tuple[()], dict]:  # type: ignore[type-arg]
        clear_cache()
        return (), {}

    benchmark.pedantic(read_all_chunks_concurrently, setup=setup, rounds=3)  # type: ignore[no-untyped-call]
