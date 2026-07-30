# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "zarr @ git+https://github.com/zarr-developers/zarr-python.git@main",
#   "numpy",
# ]
# ///

"""
Demonstrate byte-range coalescing for partial reads of sharded arrays.

A shard is a single stored object holding many inner chunks, each occupying
its own byte range. Reading N inner chunks could mean issuing N separate
byte-range requests to the store. Because byte ranges are intervals, a reader
can instead merge nearby ranges: `[a, b)` and `[b, c)` together cover
`[a, c)`, so one request can serve both. Merging trades reading some bytes you
did not ask for against issuing fewer requests -- a good trade whenever a
request is expensive, which is the normal case for object storage.

Zarr-Python 3.3.0 does this automatically. Two settings control it:

  * `sharding_coalesce_max_gap_bytes` (default 1 MiB) -- merge two ranges only
    if the gap between them is no larger than this.
  * `sharding_coalesce_max_bytes` (default 16 MiB) -- never let a merged read
    exceed this size.

Setting the gap to 0 disables merging of non-adjacent ranges, which
approximates the pre-3.3.0 behavior. This script compares the two, counting
store requests and measuring wall-clock time against a store with simulated
latency.

Run it with:

    uv run sharding_coalescing.py
"""

from __future__ import annotations

import asyncio
import operator
import statistics
import timeit
from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

import zarr
import zarr.core._coalesce as coalesce_module
from zarr.abc.store import ByteRequest, RangeByteRequest, Store
from zarr.storage import MemoryStore, WrapperStore

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from zarr.core.buffer import Buffer, BufferPrototype

# Emulates the pre-3.3.0 behavior: with a zero gap budget, only ranges that are
# exactly adjacent get merged, so scattered inner chunks are fetched one by one.
NO_COALESCING = {"sharding_coalesce_max_gap_bytes": 0}

# The shipped defaults. Spelled out here so the comparison is explicit rather
# than relying on whatever the global config happens to be.
DEFAULT_COALESCING = {
    "sharding_coalesce_max_gap_bytes": 1 << 20,  # 1 MiB
    "sharding_coalesce_max_bytes": 16 << 20,  # 16 MiB
}

GET_LATENCY_S = 0.005  # 5 ms per request, a modest stand-in for object storage


class PerRequestLatencyStore(WrapperStore[Store]):
    """Wraps a store, charging a fixed latency per byte-range fetch.

    `zarr.testing.store` ships a `LatencyStore`, but importing it pulls in
    `pytest`; defining the wrapper here keeps this example runnable with only
    zarr and numpy installed.

    Two details matter for the measurement:

    * The latency is applied in `get`, which is what an individual fetch costs.
    * `get_ranges` is explicitly *not* overridden to forward to the wrapped
      store. `WrapperStore.get_ranges` does forward, which would skip this
      class's `get` entirely and make every configuration look identical.
      Inheriting the `Store` ABC's implementation instead runs the coalescer
      over `self.get`, so each *merged* fetch pays the latency once -- which is
      exactly the cost coalescing exists to reduce.
    """

    get_ranges = Store.get_ranges

    def __init__(self, store: Store, *, get_latency: float) -> None:
        super().__init__(store)
        self.get_latency = get_latency

    def _with_store(self, store: Store) -> PerRequestLatencyStore:
        # `WrapperStore` rebuilds the wrapper when opening read-only, so the
        # latency setting has to be carried across.
        return type(self)(store, get_latency=self.get_latency)

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        await asyncio.sleep(self.get_latency)
        return await self._store.get(key, prototype, byte_range)


@contextmanager
def counting_requests() -> Iterator[Callable[[], int]]:
    """Count the store fetches issued inside the block.

    Wraps the coalescing planner rather than the store: every merged group it
    returns, plus every range it declined to merge, becomes exactly one fetch.
    Counting here rather than at the store means the number reported is the
    planner's decision, which is precisely what the settings control.
    """
    original = coalesce_module.coalesce_ranges
    total = 0

    def counting_coalesce_ranges(
        byte_ranges: Sequence[ByteRequest | None],
        *,
        max_gap_bytes: int,
        max_coalesced_bytes: int,
    ) -> tuple[
        list[list[tuple[int, RangeByteRequest]]],
        list[tuple[int, ByteRequest | None]],
    ]:
        nonlocal total
        groups, uncoalescable = original(
            byte_ranges,
            max_gap_bytes=max_gap_bytes,
            max_coalesced_bytes=max_coalesced_bytes,
        )
        total += len(groups) + len(uncoalescable)
        return groups, uncoalescable

    coalesce_module.coalesce_ranges = counting_coalesce_ranges
    try:
        yield lambda: total
    finally:
        coalesce_module.coalesce_ranges = original


def measure_read(array: zarr.Array, selection: slice) -> tuple[int, float]:
    """Return (store fetches, median seconds) for reading `selection`."""
    read = partial(operator.getitem, array, selection)

    with counting_requests() as fetches:
        result = read()
        requests = fetches()

    # `timeit.Timer` supplies the loop, `perf_counter`, and GC handling. The
    # median of several runs keeps one unlucky run from dominating.
    elapsed = statistics.median(timeit.Timer(read).repeat(repeat=5, number=1))

    assert result.size > 0  # a read that returned nothing would time as "fast"
    return requests, elapsed


def main() -> None:
    n = 8192
    chunk = 64
    inner_chunks = n // chunk

    base = MemoryStore()
    source = (np.arange(n, dtype="uint64") % 251).astype("uint8")

    # One shard holding every inner chunk, uncompressed so inner-chunk byte
    # offsets stay predictable and the demonstration is easy to reason about.
    writable = zarr.create_array(
        store=base, shape=(n,), chunks=(chunk,), shards=(n,), dtype="uint8", compressors=None
    )
    writable[:] = source

    store = PerRequestLatencyStore(base, get_latency=GET_LATENCY_S)

    print(f"zarr {zarr.__version__}")
    print(f"array: {n} uint8 values, {inner_chunks} inner chunks of {chunk} in a single shard")
    print(f"store: MemoryStore wrapped with {GET_LATENCY_S * 1000:.0f} ms of latency per request\n")

    # A strided selection touches inner chunks with unread chunks in between,
    # so there are real gaps for the coalescer to bridge. A contiguous
    # selection would merge under any setting, since its ranges are adjacent.
    selections = {
        "every 2nd inner chunk": slice(None, None, chunk * 2),
        "every 4th inner chunk": slice(None, None, chunk * 4),
        "contiguous quarter": slice(0, n // 4),
    }

    header = f"{'selection':<24} {'coalescing':<12} {'requests':>9} {'time':>10}"
    print(header)
    print("-" * len(header))

    for label, selection in selections.items():
        results = {}
        for mode, config in (("off", NO_COALESCING), ("default", DEFAULT_COALESCING)):
            array = zarr.open_array(store=store, mode="r").with_config(config)
            requests, elapsed = measure_read(array, selection)
            results[mode] = (requests, elapsed)
            print(f"{label:<24} {mode:<12} {requests:>9} {elapsed * 1000:>9.1f}ms")

        off_requests, off_time = results["off"]
        on_requests, on_time = results["default"]
        if on_requests < off_requests:
            print(
                f"{'':<24} {'->':<12} "
                f"{off_requests // on_requests:>8}x fewer {off_time / on_time:>9.1f}x faster"
            )
        else:
            print(f"{'':<24} {'->':<12} {'no change (ranges already adjacent)':>30}")
        print()

    # Correctness is the point: coalescing must not change what you read back.
    for mode, config in (("off", NO_COALESCING), ("default", DEFAULT_COALESCING)):
        array = zarr.open_array(store=store, mode="r").with_config(config)
        assert np.array_equal(array[::128], source[::128]), mode
    print("Both configurations return identical data; coalescing only changes how it is fetched.")


if __name__ == "__main__":
    main()
