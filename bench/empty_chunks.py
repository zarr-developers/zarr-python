"""Benchmark for sparse-array reads via the chunk-access primitives.

Compares the stock ``arr[:]`` read against two primitive-based read paths on
sparse arrays (~3% of chunks populated), sweeping chunk count on ``MemoryStore``
and ``LocalStore``:

- ``pack``: :func:`zarr.read_regions` + scatter onto a fill-valued array. This
  reproduces ``arr[:]`` semantics (a single contiguous array) but only touches
  the populated chunks.
- ``stream``: iterate :func:`zarr.read_regions` without packing into one array.
  This is the win for pipelines that operate per chunk and never need the whole
  array materialized.

The stock baseline scales with total chunk count; the primitive-based paths scale
with the populated-chunk count, so the speedup grows with sparsity-at-scale. Each
configuration is skipped if the warmup baseline exceeds ``BASELINE_BUDGET_S`` to
keep total runtime bounded.
"""

from __future__ import annotations

import tempfile
import time
import timeit
from pathlib import Path

import numpy as np

import zarr
from zarr.storage import LocalStore, MemoryStore

CHUNK_SIZE = 1024
DTYPE = "int32"
FILL_VALUE = 0
BASELINE_BUDGET_S = 25.0  # skip configs whose warmup baseline exceeds this

# (n_chunks, n_populated) — ~3% populated, mirrors the zagg HEALPix report.
SWEEP: list[tuple[int, int]] = [
    (1_024, 32),
    (4_096, 128),
    (16_384, 512),
    (49_152, 1_536),
]


def _build_array(store: object, n_chunks: int, n_populated: int) -> zarr.Array:
    arr = zarr.create_array(
        store=store,
        shape=(n_chunks * CHUNK_SIZE,),
        chunks=(CHUNK_SIZE,),
        dtype=DTYPE,
        fill_value=FILL_VALUE,
    )
    rng = np.random.default_rng(seed=0)
    chunk_indices = rng.choice(n_chunks, size=n_populated, replace=False)
    payload = np.arange(CHUNK_SIZE, dtype=DTYPE)
    for ci in chunk_indices:
        start = int(ci) * CHUNK_SIZE
        arr[start : start + CHUNK_SIZE] = payload
    return arr


def _read_baseline(arr: zarr.Array) -> None:
    arr[:]


def _read_pack(arr: zarr.Array) -> np.ndarray:
    out = np.full(arr.shape, arr.fill_value, dtype=arr.dtype)
    for region, data in zarr.read_regions(arr):
        out[region] = np.asarray(data)
    return out


def _read_stream(arr: zarr.Array) -> int:
    # Touch each region without materializing a single contiguous array.
    total = 0
    for _region, data in zarr.read_regions(arr):
        total += int(np.asarray(data).sum())
    return total


def _time(fn: object, repeats: int) -> float:
    return min(timeit.repeat(fn, repeat=repeats, number=1))


def _adaptive_repeats(warmup_s: float) -> int:
    if warmup_s < 0.1:
        return 5
    if warmup_s < 1.0:
        return 3
    return 1


def _run_one(
    store_name: str, store: object, n_chunks: int, n_populated: int
) -> tuple[str, int, int, float, float, float, str]:
    arr = _build_array(store, n_chunks, n_populated)

    t0 = time.perf_counter()
    _read_baseline(arr)
    warmup = time.perf_counter() - t0
    if warmup > BASELINE_BUDGET_S:
        return (
            store_name,
            n_chunks,
            n_populated,
            warmup,
            float("nan"),
            float("nan"),
            f"skipped (>{BASELINE_BUDGET_S:.0f}s budget)",
        )

    # warm both primitive paths once
    _read_pack(arr)
    _read_stream(arr)

    repeats = _adaptive_repeats(warmup)
    t_base = _time(lambda: _read_baseline(arr), repeats)
    t_pack = _time(lambda: _read_pack(arr), repeats)
    t_stream = _time(lambda: _read_stream(arr), repeats)
    return store_name, n_chunks, n_populated, t_base, t_pack, t_stream, f"min of {repeats} runs"


def main() -> None:
    rows = []
    print("Running sweep — this will take a couple of minutes for the largest configs...\n")
    for n_chunks, n_populated in SWEEP:
        rows.append(_run_one("MemoryStore", MemoryStore(), n_chunks, n_populated))
        with tempfile.TemporaryDirectory() as tmpdir:
            rows.append(
                _run_one("LocalStore", LocalStore(str(Path(tmpdir))), n_chunks, n_populated)
            )

    print(
        f"\n{'store':<14}{'n_chunks':>10}{'populated':>11}"
        f"{'arr[:] (s)':>12}{'pack (s)':>11}{'stream (s)':>12}"
        f"{'pack x':>9}{'stream x':>10}  notes"
    )
    print("-" * 100)
    for store_name, n_chunks, n_populated, t_base, t_pack, t_stream, note in rows:
        print(
            f"{store_name:<14}{n_chunks:>10}{n_populated:>11}"
            f"{t_base:>12.4f}{_fmt(t_pack):>11}{_fmt(t_stream):>12}"
            f"{_speedup(t_base, t_pack):>9}{_speedup(t_base, t_stream):>10}  {note}"
        )


def _fmt(t: float) -> str:
    return "—" if np.isnan(t) else f"{t:.4f}"


def _speedup(t_base: float, t: float) -> str:
    return "—" if np.isnan(t) or t <= 0 else f"{t_base / t:.1f}x"


if __name__ == "__main__":
    main()
