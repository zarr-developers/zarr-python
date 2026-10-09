# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "zarr @ git+https://github.com/zarr-developers/zarr-python.git@main",
#   "numpy",
# ]
# ///

"""
Compare the `BatchedCodecPipeline` and the `FusedCodecPipeline`.

The default `BatchedCodecPipeline` schedules storage IO and codec compute
asynchronously -- roughly one coroutine per chunk operation. For a *sharded*
array that means one coroutine per inner chunk inside every shard. That is the
right model for high-latency stores, where there is useful work to do while
waiting for IO. For low-latency stores (in-process memory, the local
filesystem) the IO completes too quickly for the overlap to pay for itself, and
the scheduling becomes pure overhead.

The `FusedCodecPipeline` runs codec compute and synchronous IO synchronously,
removing that overhead. Whether it wins, and whether its thread pool helps,
depends on which resource is actually scarce:

  * Uncompressed IO is dominated by per-chunk *scheduling*, not compute. There
    is nothing for a thread pool to parallelize, so a single worker is already
    fastest and extra workers only add overhead.
  * gzip is genuinely CPU-bound. A single worker compresses every chunk
    sequentially and can be *slower than the default*, while a thread pool
    spreads that compression across cores and reclaims the win. This is when
    `max_workers > 1` earns its keep.

Run it with:

    uv run codec_pipeline_performance.py

Numbers are hardware-, layout-, and codec-dependent. Treat the output as a
measurement of *your* machine, not as a published benchmark.
"""

from __future__ import annotations

import operator
import os
import statistics
import tempfile
import timeit
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import zarr
from zarr.storage import LocalStore, MemoryStore

if TYPE_CHECKING:
    from collections.abc import Callable

    from zarr.abc.store import Store

BATCHED = "zarr.core.codec_pipeline.BatchedCodecPipeline"
FUSED = "zarr.core.codec_pipeline.FusedCodecPipeline"

# gzip is CPU-bound to encode, which is exactly the regime where the thread
# pool matters. Level 6 is gzip's own default.
GZIP = {"name": "gzip", "configuration": {"level": 6}}

# 4096x4096 int32 = 64 MiB, split into 16 shards of 1024x1024, each holding
# 16x16 = 256 inner chunks of 64x64 -> 4096 inner chunks in total. The chunks
# are small enough that per-chunk coroutine scheduling dominates uncompressed
# IO, yet large enough that per-chunk gzip is real work to spread over cores.
SHAPE = (4096, 4096)
SHARDS = (1024, 1024)
CHUNKS = (64, 64)
DTYPE = "int32"

CONFIGS: tuple[tuple[str, dict[str, object]], ...] = (
    ("Batched (default)", {"codec_pipeline.path": BATCHED}),
    ("Fused (1 worker)", {"codec_pipeline.path": FUSED, "codec_pipeline.max_workers": 1}),
    ("Fused (cpu_count)", {"codec_pipeline.path": FUSED, "codec_pipeline.max_workers": None}),
)


def time_call(fn: Callable[[], object], repeat: int = 3) -> float:
    """Median wall-clock seconds for one call to `fn`.

    `timeit.Timer` supplies `perf_counter` and disables the cyclic garbage
    collector during each run, so a collection triggered by earlier work cannot
    land inside a measurement. `number=1` because a single call here already
    moves 64 MiB -- the per-call overhead `timeit` amortizes is irrelevant at
    this scale.
    """
    return statistics.median(timeit.Timer(fn).repeat(repeat=repeat, number=1))


def measure(
    settings: dict[str, object],
    store: Store,
    data: np.ndarray,
    compressors: object,
) -> tuple[float, float]:
    """Time one full write and one full read of `data` under `settings`.

    The whole operation runs inside `zarr.config.set`, not just the array
    construction. The pipeline class is resolved when the array is built, but
    `codec_pipeline.max_workers` is read *per operation*, so a timed call made
    outside the config block would silently use whatever worker count was
    globally in effect -- which makes every configuration look identical.
    """
    everything = slice(None)

    def write_once() -> None:
        with zarr.config.set(settings):
            array = zarr.create_array(
                store=store,
                shape=SHAPE,
                chunks=CHUNKS,
                shards=SHARDS,
                dtype=DTYPE,
                compressors=compressors,
                fill_value=0,
                overwrite=True,
            )
            operator.setitem(array, everything, data)

    write = time_call(write_once)

    # The bytes on disk are identical whichever pipeline wrote them, so reading
    # back what we just wrote isolates read performance on the same data.
    def read_once() -> object:
        with zarr.config.set(settings):
            return zarr.open_array(store=store, mode="r")[everything]

    read = time_call(read_once)

    if not np.array_equal(read_once(), data):
        raise AssertionError("round trip mismatch")
    return write, read


def make_store(kind: str, tmp: Path) -> Store:
    if kind == "memory":
        return MemoryStore()
    return LocalStore(tmp / f"demo_{kind}_{os.getpid()}.zarr")


def main() -> None:
    n_cpu = os.cpu_count() or 1

    # Each regime gets the data that actually exercises it. `arange` is
    # trivially compressible, which is fine when nothing compresses it, but it
    # would make gzip finish almost instantly and hide the CPU-bound behavior
    # this example is about. The noisy array keeps gzip genuinely busy.
    n = int(np.prod(SHAPE))
    plain_data = np.arange(n, dtype=DTYPE).reshape(SHAPE)
    noisy_data = np.random.default_rng(0).integers(0, 2**24, size=SHAPE, dtype=DTYPE)

    n_shards = int(np.prod([s // c for s, c in zip(SHAPE, SHARDS, strict=True)]))
    per_shard = int(np.prod([s // c for s, c in zip(SHARDS, CHUNKS, strict=True)]))
    print(f"zarr {zarr.__version__} | {n_cpu} CPUs")
    print(
        f"array {SHAPE} {DTYPE} = {plain_data.nbytes / 2**20:.0f} MiB | "
        f"{n_shards} shards x {per_shard} inner chunks = {n_shards * per_shard} chunks\n"
    )

    with tempfile.TemporaryDirectory() as tmp:
        for store_kind in ("memory", "local"):
            for codec_label, compressors, data in (
                ("uncompressed", None, plain_data),
                ("gzip-6 (CPU-bound)", GZIP, noisy_data),
            ):
                print(f"=== {store_kind} store / {codec_label} ===")
                print(
                    f"{'pipeline':<22}{'write (s)':>11}{'vs base':>10}"
                    f"{'read (s)':>12}{'vs base':>10}"
                )
                results: dict[str, tuple[float, float]] = {}
                for label, settings in CONFIGS:
                    store = make_store(store_kind, Path(tmp))
                    results[label] = measure(settings, store, data, compressors)

                base_write, base_read = results[CONFIGS[0][0]]
                for label, (write, read) in results.items():
                    print(
                        f"{label:<22}{write:>10.3f}{base_write / write:>9.1f}x"
                        f"{read:>11.3f}{base_read / read:>9.1f}x"
                    )

                # The headline comparison: does the thread pool earn its keep?
                single_write, single_read = results["Fused (1 worker)"]
                pool_write, pool_read = results["Fused (cpu_count)"]
                print(
                    f"  workers (cpu_count vs 1 worker): "
                    f"write {single_write / pool_write:.1f}x   "
                    f"read {single_read / pool_read:.1f}x"
                )
                print()

    print(
        "Reading it:\n"
        "  * Uncompressed IO is scheduling-bound, so Fused (1 worker) is already\n"
        "    fastest -- a thread pool has nothing to parallelize and only adds\n"
        "    overhead.\n"
        "  * gzip is CPU-bound, so Fused (1 worker) can be *slower* than the\n"
        "    default, while Fused (cpu_count) spreads compression across cores\n"
        "    and reclaims the win. That flip is why the fused pipeline is\n"
        "    threaded by default, and why pinning max_workers=1 is worth it for\n"
        "    memory-backed uncompressed data.\n"
        "  * `codec_pipeline.max_workers` is read only by the FusedCodecPipeline;\n"
        "    the default BatchedCodecPipeline ignores it."
    )


if __name__ == "__main__":
    main()
