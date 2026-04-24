# tests/test_coalesce.py
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    SuffixByteRequest,
)
from zarr.core._coalesce import (
    DEFAULT_COALESCE_OPTIONS,
    CoalesceOptions,
    coalesced_get,
)
from zarr.core.buffer import Buffer, default_buffer_prototype

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Sequence

pytestmark = pytest.mark.asyncio


def _buf(data: bytes) -> Buffer:
    return default_buffer_prototype().buffer.from_bytes(data)


@dataclass
class FakeFetch:
    """Records every call and serves canned bytes from an in-memory blob."""

    blob: bytes
    key_exists: bool = True
    raise_on: Callable[[ByteRequest | None], bool] | None = None
    calls: list[ByteRequest | None] = field(default_factory=list)

    async def __call__(self, byte_range: ByteRequest | None) -> Buffer | None:
        self.calls.append(byte_range)
        if not self.key_exists:
            return None
        if self.raise_on is not None and self.raise_on(byte_range):
            raise OSError("injected")
        if byte_range is None:
            return _buf(self.blob)
        if isinstance(byte_range, RangeByteRequest):
            return _buf(self.blob[byte_range.start : byte_range.end])
        if isinstance(byte_range, OffsetByteRequest):
            return _buf(self.blob[byte_range.offset :])
        if isinstance(byte_range, SuffixByteRequest):
            return _buf(self.blob[-byte_range.suffix :])
        raise AssertionError(f"unknown byte_range {byte_range!r}")


async def _collect(
    agen: AsyncIterator[Sequence[tuple[int, Buffer | None]]],
) -> list[list[tuple[int, Buffer | None]]]:
    """Drain an async generator of groups into a list of lists of tuples."""
    return [list(group) async for group in agen]


def _contents(groups: list[list[tuple[int, Buffer | None]]]) -> dict[int, bytes]:
    """Flatten to {index: bytes}."""
    result: dict[int, bytes] = {}
    for group in groups:
        for idx, buf in group:
            assert buf is not None
            result[idx] = buf.to_bytes()
    return result


async def test_empty_input() -> None:
    fetch = FakeFetch(b"abc" * 1000)
    groups = await _collect(coalesced_get(fetch, [], options=DEFAULT_COALESCE_OPTIONS))
    assert groups == []
    assert fetch.calls == []


async def test_single_range() -> None:
    fetch = FakeFetch(b"0123456789")
    groups = await _collect(
        coalesced_get(
            fetch,
            [RangeByteRequest(2, 5)],
            options=DEFAULT_COALESCE_OPTIONS,
        )
    )
    assert len(groups) == 1
    assert len(groups[0]) == 1
    assert _contents(groups) == {0: b"234"}
    assert len(fetch.calls) == 1


async def test_fully_disjoint_ranges_each_get_own_group() -> None:
    # Ranges 100 bytes apart with max_gap_bytes < 100 will not merge.
    fetch = FakeFetch(b"x" * 10_000)
    opts: CoalesceOptions = {
        "max_gap_bytes": 50,
        "max_coalesced_bytes": 1 << 20,
        "max_concurrency": 10,
    }
    ranges: list[ByteRequest | None] = [
        RangeByteRequest(0, 10),
        RangeByteRequest(200, 210),
        RangeByteRequest(500, 510),
    ]
    groups = await _collect(coalesced_get(fetch, ranges, options=opts))
    assert len(groups) == 3
    assert all(len(g) == 1 for g in groups)
    # 3 fetches, one per input.
    assert len(fetch.calls) == 3


async def test_adjacent_ranges_merge_into_one_group() -> None:
    # Three ranges within 10 bytes of each other; max_gap_bytes=50 -> one merged fetch.
    fetch = FakeFetch(b"".join(bytes([i % 256]) for i in range(1000)))
    opts: CoalesceOptions = {
        "max_gap_bytes": 50,
        "max_coalesced_bytes": 1 << 20,
        "max_concurrency": 10,
    }
    ranges: list[ByteRequest | None] = [
        RangeByteRequest(0, 5),
        RangeByteRequest(10, 15),
        RangeByteRequest(20, 25),
    ]
    groups = await _collect(coalesced_get(fetch, ranges, options=opts))
    assert len(groups) == 1
    assert len(groups[0]) == 3
    # The single fetch should span the full merged region.
    assert len(fetch.calls) == 1
    call = fetch.calls[0]
    assert isinstance(call, RangeByteRequest)
    assert call.start == 0
    assert call.end == 25
    # Contents correct.
    assert _contents(groups) == {
        0: bytes(range(5)),
        1: bytes(range(10, 15)),
        2: bytes(range(20, 25)),
    }


async def test_offset_and_suffix_and_none_each_get_own_group() -> None:
    fetch = FakeFetch(b"abcdefghij")
    ranges: list[ByteRequest | None] = [
        RangeByteRequest(0, 3),
        OffsetByteRequest(5),
        SuffixByteRequest(2),
        None,
    ]
    groups = await _collect(coalesced_get(fetch, ranges, options=DEFAULT_COALESCE_OPTIONS))
    # 1 group from the RangeByteRequest + 3 one-tuple groups from the rest.
    assert len(groups) == 4
    # Contents.
    flat = _contents(groups)
    assert flat[0] == b"abc"
    assert flat[1] == b"fghij"
    assert flat[2] == b"ij"
    assert flat[3] == b"abcdefghij"


async def test_indices_preserved_under_shuffled_input() -> None:
    fetch = FakeFetch(b"".join(bytes([i % 256]) for i in range(1000)))
    # Construct ranges in a deliberately non-sorted order.
    ranges: list[ByteRequest | None] = [
        RangeByteRequest(500, 510),
        RangeByteRequest(0, 10),
        RangeByteRequest(200, 210),
        RangeByteRequest(300, 310),
    ]
    opts: CoalesceOptions = {
        "max_gap_bytes": 50,
        "max_coalesced_bytes": 1 << 20,
        "max_concurrency": 10,
    }
    groups = await _collect(coalesced_get(fetch, ranges, options=opts))
    flat = _contents(groups)
    # Indices match original positions, not sorted order.
    assert flat[0] == bytes(b % 256 for b in range(500, 510))
    assert flat[1] == bytes(b % 256 for b in range(10))
    assert flat[2] == bytes(b % 256 for b in range(200, 210))
    assert flat[3] == bytes(b % 256 for b in range(300, 310))


async def test_within_group_ordering_is_start_offset() -> None:
    fetch = FakeFetch(b"".join(bytes([i % 256]) for i in range(100)))
    # Two ranges will merge; one has a later start but is listed first in input.
    ranges: list[ByteRequest | None] = [
        RangeByteRequest(20, 25),
        RangeByteRequest(0, 5),
    ]
    opts: CoalesceOptions = {
        "max_gap_bytes": 50,
        "max_coalesced_bytes": 1 << 20,
        "max_concurrency": 10,
    }
    groups = await _collect(coalesced_get(fetch, ranges, options=opts))
    assert len(groups) == 1
    # Within the group, tuples are ordered by start offset.
    # Input index 1 (start=0) comes first, then 0 (start=20).
    assert [idx for idx, _ in groups[0]] == [1, 0]


async def test_mixed_mergeable_and_non_mergeable_counts_correct() -> None:
    fetch = FakeFetch(b"x" * 10_000)
    opts: CoalesceOptions = {
        "max_gap_bytes": 50,
        "max_coalesced_bytes": 1 << 20,
        "max_concurrency": 10,
    }
    # Two clusters + one far-away singleton.
    ranges: list[ByteRequest | None] = [
        RangeByteRequest(0, 10),
        RangeByteRequest(20, 30),
        RangeByteRequest(500, 510),
    ]
    groups = await _collect(coalesced_get(fetch, ranges, options=opts))
    assert len(groups) == 2
    # First group has 2, second has 1.
    sizes = sorted(len(g) for g in groups)
    assert sizes == [1, 2]


async def test_max_concurrency_is_honored() -> None:
    # Build 10 non-mergeable ranges, have the fetch hold a counter of in-flight calls.
    in_flight = 0
    peak = 0
    lock = asyncio.Lock()

    async def fetch(byte_range: ByteRequest | None) -> Buffer | None:
        nonlocal in_flight, peak
        async with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        # give the scheduler a chance to run other tasks
        await asyncio.sleep(0.01)
        async with lock:
            in_flight -= 1
        return _buf(b"x")

    ranges: list[ByteRequest | None] = [RangeByteRequest(i * 1000, i * 1000 + 1) for i in range(10)]
    opts: CoalesceOptions = {
        "max_gap_bytes": 0,  # force no merging
        "max_coalesced_bytes": 1 << 20,
        "max_concurrency": 3,
    }
    async for _group in coalesced_get(fetch, ranges, options=opts):
        pass
    assert peak <= 3
    assert peak >= 2  # must have been some real concurrency


async def test_key_missing_from_first_call_yields_nothing() -> None:
    fetch = FakeFetch(b"x" * 100, key_exists=False)
    ranges: list[ByteRequest | None] = [RangeByteRequest(0, 10), RangeByteRequest(20, 30)]
    groups = await _collect(coalesced_get(fetch, ranges, options=DEFAULT_COALESCE_OPTIONS))
    assert groups == []


async def test_key_missing_mid_stream_yields_earlier_groups_only() -> None:
    # Two non-mergeable ranges; the second fetch returns None.
    call_count = 0

    async def fetch(byte_range: ByteRequest | None) -> Buffer | None:
        nonlocal call_count
        call_count += 1
        # Ensure deterministic ordering: first call serves, second returns None.
        await asyncio.sleep(0.01 if call_count == 1 else 0.02)
        if call_count >= 2:
            return None
        return _buf(b"ok")

    opts: CoalesceOptions = {
        "max_gap_bytes": -1,
        "max_coalesced_bytes": 1 << 20,
        "max_concurrency": 1,  # serialize for determinism
    }
    ranges: list[ByteRequest | None] = [RangeByteRequest(0, 2), RangeByteRequest(100, 102)]
    groups = await _collect(coalesced_get(fetch, ranges, options=opts))
    # Exactly one group (the first) -- the second went missing.
    assert len(groups) == 1
    assert len(groups[0]) == 1


async def test_key_missing_mid_stream_with_concurrency_drains_late_arrivals() -> None:
    # Schedule multiple non-mergeable fetches under max_concurrency=3.
    #   - Fetch #0 completes FIRST (short sleep) -> at least one yield observed.
    #   - Fetch #1 returns None shortly after -> triggers the miss path.
    #   - Fetches #2..#N are gated on an asyncio.Event so they only unblock
    #     AFTER the miss has been observed, producing late arrivals that
    #     exercise the `if stopped: continue` discard branch in the drain loop.
    late_gate = asyncio.Event()
    miss_fired = asyncio.Event()

    async def fetch(byte_range: ByteRequest | None) -> Buffer | None:
        assert isinstance(byte_range, RangeByteRequest)
        start = byte_range.start
        if start == 0:
            # First to complete: small sleep so it arrives before the miss.
            await asyncio.sleep(0.01)
            return _buf(b"ok")
        if start == 1000:
            # Miss: a little later than #0 so #0 yields first.
            await asyncio.sleep(0.03)
            miss_fired.set()
            return None
        # Late arrivals: wait until the miss has been processed, then return
        # a buffer so the drain loop sees them post-stop.
        await asyncio.wait_for(miss_fired.wait(), timeout=5.0)
        await asyncio.wait_for(late_gate.wait(), timeout=5.0)
        return _buf(b"ok")

    opts: CoalesceOptions = {
        "max_gap_bytes": -1,  # force no merging (each range its own fetch)
        "max_coalesced_bytes": 1 << 20,
        "max_concurrency": 3,
    }
    # Stride by 1000 to avoid merging. 7 items fits within max_concurrency=3
    # while producing pending work after the miss.
    ranges: list[ByteRequest | None] = [RangeByteRequest(i * 1000, i * 1000 + 1) for i in range(7)]

    groups: list[list[tuple[int, Buffer | None]]] = []
    agen = coalesced_get(fetch, ranges, options=opts)
    try:
        async for group in agen:
            groups.append(list(group))
            # After the first yield, release the late gate so remaining
            # in-flight tasks can complete and arrive post-stop.
            late_gate.set()
    finally:
        # Guard against a bug preventing any yield: unblock waiters anyway.
        late_gate.set()

    # We observed exactly the one pre-miss yield.
    assert len(groups) == 1
    assert len(groups[0]) == 1
    idx, buf = groups[0][0]
    assert idx == 0
    assert buf is not None
    # The iterator completed cleanly without raising.
    assert miss_fired.is_set()


@pytest.mark.parametrize(
    "byte_range",
    [
        OffsetByteRequest(5),
        SuffixByteRequest(5),
        None,
    ],
    ids=["offset", "suffix", "none"],
)
async def test_key_missing_on_uncoalescable_input_yields_nothing(
    byte_range: ByteRequest | None,
) -> None:
    # Uncoalescable inputs take a distinct code path from the merged-group
    # path; a missing key on that path must still short-circuit cleanly.
    fetch = FakeFetch(b"x" * 100, key_exists=False)
    groups = await _collect(coalesced_get(fetch, [byte_range], options=DEFAULT_COALESCE_OPTIONS))
    assert groups == []


async def test_consumer_break_cancels_pending_fetches() -> None:
    # Kick off many slow ranges with small max_concurrency, break after the
    # first yielded group, and verify the remaining tasks are cancelled rather
    # than allowed to run to completion.
    completed_calls = 0
    cancelled_calls = 0

    async def fetch(byte_range: ByteRequest | None) -> Buffer | None:
        nonlocal completed_calls, cancelled_calls
        assert isinstance(byte_range, RangeByteRequest)
        start = byte_range.start
        try:
            # First fetch returns fast so the async for body runs and can break.
            # Later fetches sleep long enough that cancellation has room to land.
            await asyncio.sleep(0.001 if start == 0 else 2.0)
        except asyncio.CancelledError:
            cancelled_calls += 1
            raise
        completed_calls += 1
        return _buf(b"x")

    opts: CoalesceOptions = {
        "max_gap_bytes": -1,  # no merging
        "max_coalesced_bytes": 1 << 20,
        "max_concurrency": 3,
    }
    ranges: list[ByteRequest | None] = [RangeByteRequest(i * 1000, i * 1000 + 1) for i in range(6)]

    agen = coalesced_get(fetch, ranges, options=opts)
    # Break after receiving the first yield.
    async for _group in agen:
        break
    # Explicitly close the generator so its finally block runs (cancelling
    # in-flight tasks) before we make assertions.
    await agen.aclose()

    # At least one slow fetch was actually running under the semaphore and got
    # cancelled (rather than running to completion).
    assert cancelled_calls >= 1
    # And the first range's fetch completed normally (no spurious cancels there).
    assert completed_calls >= 1


async def test_fetch_raises_propagates() -> None:
    fetch = FakeFetch(
        b"x" * 100,
        raise_on=lambda r: isinstance(r, RangeByteRequest) and r.start >= 100,
    )
    opts: CoalesceOptions = {
        "max_gap_bytes": -1,
        "max_coalesced_bytes": 1 << 20,
        "max_concurrency": 1,
    }
    ranges: list[ByteRequest | None] = [RangeByteRequest(0, 10), RangeByteRequest(200, 210)]
    with pytest.raises(OSError, match="injected"):
        await _collect(coalesced_get(fetch, ranges, options=opts))


async def test_max_coalesced_bytes_prevents_over_cap_merge() -> None:
    fetch = FakeFetch(b"x" * 10_000)
    opts: CoalesceOptions = {
        "max_gap_bytes": 1000,  # allow gap
        "max_coalesced_bytes": 50,  # but cap the merged size
        "max_concurrency": 10,
    }
    # Two ranges 20 bytes apart, each 20 bytes -- merged span would be 20+20+20=60 > 50.
    ranges: list[ByteRequest | None] = [RangeByteRequest(0, 20), RangeByteRequest(40, 60)]
    groups = await _collect(coalesced_get(fetch, ranges, options=opts))
    assert len(groups) == 2
    assert all(len(g) == 1 for g in groups)


async def test_single_range_larger_than_cap_is_passed_through() -> None:
    fetch = FakeFetch(b"x" * 10_000)
    opts: CoalesceOptions = {
        "max_gap_bytes": 1000,
        "max_coalesced_bytes": 50,
        "max_concurrency": 10,
    }
    # A single 200-byte range, larger than the cap. Should still be fetched.
    ranges: list[ByteRequest | None] = [RangeByteRequest(0, 200)]
    groups = await _collect(coalesced_get(fetch, ranges, options=opts))
    assert len(groups) == 1
    assert len(groups[0]) == 1
    idx, buf = groups[0][0]
    assert idx == 0
    assert buf is not None
    assert buf.to_bytes() == b"x" * 200


async def test_coverage_invariant_random_inputs() -> None:
    import random

    rng = random.Random(42)
    blob = bytes(i % 256 for i in range(10_000))
    fetch = FakeFetch(blob)

    # Generate 50 random RangeByteRequests within the blob.
    ranges: list[ByteRequest | None] = []
    for _ in range(50):
        start = rng.randint(0, 9000)
        length = rng.randint(1, 500)
        ranges.append(RangeByteRequest(start, start + length))

    groups = await _collect(coalesced_get(fetch, ranges, options=DEFAULT_COALESCE_OPTIONS))
    seen: list[int] = []
    for group in groups:
        for idx, _buf in group:
            seen.append(idx)
    assert sorted(seen) == list(range(len(ranges)))
    # And the bytes are correct.
    flat = _contents(groups)
    for i, r in enumerate(ranges):
        assert isinstance(r, RangeByteRequest)
        assert flat[i] == blob[r.start : r.end]
