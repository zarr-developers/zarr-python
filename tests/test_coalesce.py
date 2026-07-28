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
    coalesce_ranges,
    coalesced_get,
)
from zarr.core.buffer import Buffer, default_buffer_prototype

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Mapping, Sequence


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


# ---------------------------------------------------------------------------
# Shared coalescing-knob bundles. Each is a complete mapping of all three
# kwargs to splat into `coalesced_get`; `coalesce_ranges` ignores
# `max_concurrency`. The leaf functions in `_coalesce.py` require all knobs
# explicitly — `Store.get_ranges` is the public entry point and owns the
# canonical defaults. Tests pick their own values appropriate to the scenario.
# ---------------------------------------------------------------------------

# Permissive default for tests that don't care about specific thresholds. Mirrors
# `Store.get_ranges`'s public defaults but the test file owns this independently
# of any production constants.
DEFAULT: Mapping[str, int] = {
    "max_concurrency": 10,
    "max_gap_bytes": 1 << 20,
    "max_coalesced_bytes": 16 << 20,
}
"""Permissive defaults; mirrors `Store.get_ranges`'s baseline."""

MERGE_GAP_50: Mapping[str, int] = {
    "max_concurrency": 10,
    "max_gap_bytes": 50,
    "max_coalesced_bytes": 1 << 20,
}
"""Merge ranges within 50 bytes of each other."""

NO_MERGE: Mapping[str, int] = {
    "max_concurrency": 10,
    "max_gap_bytes": -1,
    "max_coalesced_bytes": 1 << 20,
}
"""No merging: any positive gap is > -1, so no pair ever coalesces."""

CAP_50: Mapping[str, int] = {
    "max_concurrency": 10,
    "max_gap_bytes": 1000,
    "max_coalesced_bytes": 50,
}
"""Gap permissive but merged size capped at 50 bytes."""


def _grouping(opts: Mapping[str, int]) -> dict[str, int]:
    """Return only the grouping knobs from a full options bundle.

    `coalesce_ranges` rejects `max_concurrency`; this lets test bundles be
    full kwargs maps (for `coalesced_get`) and still be passed to the pure
    planner via splat.
    """
    return {k: v for k, v in opts.items() if k != "max_concurrency"}


# A deterministic blob used for content-sensitive cases: byte i == (i % 256).
_INDEXED_BLOB = bytes(i % 256 for i in range(10_000))


# ---------------------------------------------------------------------------
# Parametrized structural/content tests (cases without async timing or errors).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StructuralCase:
    """One row of the parametrized structure-and-contents table."""

    id: str
    """pytest id for the case."""
    ranges: list[ByteRequest | None]
    """Input to coalesced_get."""
    options: Mapping[str, int]
    """Coalescing knobs to splat into `coalesced_get`."""
    expected_group_sizes: list[int]
    """Sorted list of group tuple-counts (order-independent)."""
    expected_contents: dict[int, bytes] | None = None
    """{input_index: bytes} to verify bytes, or None to skip the content check."""
    expected_n_fetches: int | None = None
    """Exact number of calls to the fetch callable, or None to skip the check."""


_STRUCTURAL_CASES: list[StructuralCase] = [
    StructuralCase(
        id="empty-input",
        ranges=[],
        options=DEFAULT,
        expected_group_sizes=[],
        expected_n_fetches=0,
    ),
    StructuralCase(
        id="single-range",
        ranges=[RangeByteRequest(2, 5)],
        options=DEFAULT,
        expected_group_sizes=[1],
        expected_contents={0: _INDEXED_BLOB[2:5]},
        expected_n_fetches=1,
    ),
    StructuralCase(
        id="disjoint-3-no-merge",
        ranges=[
            RangeByteRequest(0, 10),
            RangeByteRequest(200, 210),
            RangeByteRequest(500, 510),
        ],
        options=MERGE_GAP_50,
        expected_group_sizes=[1, 1, 1],
        expected_contents={
            0: _INDEXED_BLOB[0:10],
            1: _INDEXED_BLOB[200:210],
            2: _INDEXED_BLOB[500:510],
        },
        expected_n_fetches=3,
    ),
    StructuralCase(
        id="adjacent-3-one-merged-group",
        ranges=[
            RangeByteRequest(0, 5),
            RangeByteRequest(10, 15),
            RangeByteRequest(20, 25),
        ],
        options=MERGE_GAP_50,
        expected_group_sizes=[3],
        expected_contents={
            0: _INDEXED_BLOB[0:5],
            1: _INDEXED_BLOB[10:15],
            2: _INDEXED_BLOB[20:25],
        },
        expected_n_fetches=1,
    ),
    StructuralCase(
        id="two-clusters-one-singleton",
        ranges=[
            RangeByteRequest(0, 10),
            RangeByteRequest(20, 30),
            RangeByteRequest(500, 510),
        ],
        options=MERGE_GAP_50,
        expected_group_sizes=[1, 2],
        expected_contents={
            0: _INDEXED_BLOB[0:10],
            1: _INDEXED_BLOB[20:30],
            2: _INDEXED_BLOB[500:510],
        },
        expected_n_fetches=2,
    ),
    StructuralCase(
        id="uncoalescable-mixed-with-range",
        ranges=[
            RangeByteRequest(0, 3),
            OffsetByteRequest(5),
            SuffixByteRequest(2),
            None,
        ],
        options=DEFAULT,
        expected_group_sizes=[1, 1, 1, 1],
        expected_contents={
            0: _INDEXED_BLOB[0:3],
            1: _INDEXED_BLOB[5:],
            2: _INDEXED_BLOB[-2:],
            3: _INDEXED_BLOB,
        },
        expected_n_fetches=4,
    ),
    StructuralCase(
        id="shuffled-input-indices-preserved",
        ranges=[
            RangeByteRequest(500, 510),
            RangeByteRequest(0, 10),
            RangeByteRequest(200, 210),
            RangeByteRequest(300, 310),
        ],
        options=MERGE_GAP_50,
        expected_group_sizes=[1, 1, 1, 1],
        expected_contents={
            0: _INDEXED_BLOB[500:510],
            1: _INDEXED_BLOB[0:10],
            2: _INDEXED_BLOB[200:210],
            3: _INDEXED_BLOB[300:310],
        },
        expected_n_fetches=4,
    ),
    StructuralCase(
        id="cap-prevents-merge-of-close-ranges",
        # 20 + 20 gap + 20 = 60-byte merged span > cap of 50.
        ranges=[RangeByteRequest(0, 20), RangeByteRequest(40, 60)],
        options=CAP_50,
        expected_group_sizes=[1, 1],
        expected_n_fetches=2,
    ),
    StructuralCase(
        id="single-range-larger-than-cap-passes-through",
        # Cap only applies to MERGE decisions; a lone oversized range still fetches.
        ranges=[RangeByteRequest(0, 200)],
        options=CAP_50,
        expected_group_sizes=[1],
        expected_contents={0: _INDEXED_BLOB[0:200]},
        expected_n_fetches=1,
    ),
]


@pytest.mark.parametrize("case", _STRUCTURAL_CASES, ids=lambda c: c.id)
async def test_coalescing_structure_and_contents(case: StructuralCase) -> None:
    """Group structure, byte contents, and fetch-call count for the deterministic cases."""
    fetch = FakeFetch(_INDEXED_BLOB)
    groups = await _collect(coalesced_get(fetch, case.ranges, **case.options))

    assert sorted(len(g) for g in groups) == sorted(case.expected_group_sizes)

    if case.expected_contents is not None:
        assert _contents(groups) == case.expected_contents

    if case.expected_n_fetches is not None:
        assert len(fetch.calls) == case.expected_n_fetches


# ---------------------------------------------------------------------------
# Focused non-parametrized tests for cases with distinctive assertion shapes.
# ---------------------------------------------------------------------------


async def test_within_group_ordering_is_start_offset() -> None:
    """Within a merged group, tuples are ordered by start offset, not input order."""
    fetch = FakeFetch(_INDEXED_BLOB)
    # Two ranges that merge; one has a later start but is listed first in input.
    ranges: list[ByteRequest | None] = [RangeByteRequest(20, 25), RangeByteRequest(0, 5)]
    groups = await _collect(coalesced_get(fetch, ranges, **MERGE_GAP_50))
    assert len(groups) == 1
    # Input index 1 (start=0) comes first, then 0 (start=20).
    assert [idx for idx, _ in groups[0]] == [1, 0]


async def test_adjacent_ranges_fire_single_fetch_spanning_merged_region() -> None:
    """Verify the merged fetch covers exactly the span from min-start to max-end."""
    fetch = FakeFetch(_INDEXED_BLOB)
    ranges: list[ByteRequest | None] = [
        RangeByteRequest(0, 5),
        RangeByteRequest(10, 15),
        RangeByteRequest(20, 25),
    ]
    await _collect(coalesced_get(fetch, ranges, **MERGE_GAP_50))
    assert len(fetch.calls) == 1
    call = fetch.calls[0]
    assert isinstance(call, RangeByteRequest)
    assert call.start == 0
    assert call.end == 25


# ---------------------------------------------------------------------------
# Concurrency and cancellation.
# ---------------------------------------------------------------------------


async def test_max_concurrency_is_honored() -> None:
    """With 10 non-mergeable ranges and max_concurrency=3, peak in-flight must not exceed 3."""
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
    opts: Mapping[str, int] = {
        "max_gap_bytes": 0,  # force no merging
        "max_coalesced_bytes": 1 << 20,
        "max_concurrency": 3,
    }
    async for _group in coalesced_get(fetch, ranges, **opts):
        pass
    assert peak <= 3
    assert peak >= 2  # must have been some real concurrency


async def test_consumer_break_cancels_pending_fetches() -> None:
    """Breaking out of the async for should cancel pending fetches rather than let them complete."""
    completed_calls = 0
    cancelled_calls = 0

    async def fetch(byte_range: ByteRequest | None) -> Buffer | None:
        nonlocal completed_calls, cancelled_calls
        assert isinstance(byte_range, RangeByteRequest)
        start = byte_range.start
        try:
            # First fetch returns fast so the async-for body runs and can break.
            # Later fetches sleep long enough that cancellation has room to land.
            await asyncio.sleep(0.001 if start == 0 else 2.0)
        except asyncio.CancelledError:
            cancelled_calls += 1
            raise
        completed_calls += 1
        return _buf(b"x")

    opts: Mapping[str, int] = {
        "max_gap_bytes": -1,  # no merging
        "max_coalesced_bytes": 1 << 20,
        "max_concurrency": 3,
    }
    ranges: list[ByteRequest | None] = [RangeByteRequest(i * 1000, i * 1000 + 1) for i in range(6)]

    agen = coalesced_get(fetch, ranges, **opts)
    async for _group in agen:
        break
    # Explicitly close the generator so its finally block runs (cancelling
    # in-flight tasks) before we make assertions.
    await agen.aclose()

    # The fast task completes; the remaining tasks are either cancelled while
    # sleeping (raising CancelledError into the user try block) or cancelled
    # while still waiting on the semaphore (which doesn't enter the try at all).
    # Either way, none of them should be allowed to complete.
    assert completed_calls == 1
    assert cancelled_calls >= 1
    assert completed_calls + cancelled_calls <= len(ranges)


# ---------------------------------------------------------------------------
# Key-missing semantics.
# ---------------------------------------------------------------------------


async def test_key_missing_from_first_call_raises() -> None:
    """If the very first fetch returns None, the iterator raises an ExceptionGroup containing FileNotFoundError."""
    fetch = FakeFetch(b"x" * 100, key_exists=False)
    ranges: list[ByteRequest | None] = [RangeByteRequest(0, 10), RangeByteRequest(20, 30)]
    with pytest.RaisesGroup(pytest.RaisesExc(FileNotFoundError)):
        await _collect(coalesced_get(fetch, ranges, **DEFAULT))


@pytest.mark.parametrize(
    "byte_range",
    [OffsetByteRequest(5), SuffixByteRequest(5), None],
    ids=["offset", "suffix", "none"],
)
async def test_key_missing_on_uncoalescable_input_raises(
    byte_range: ByteRequest | None,
) -> None:
    """Uncoalescable inputs take a distinct path; key-missing must still raise (wrapped in a group)."""
    fetch = FakeFetch(b"x" * 100, key_exists=False)
    with pytest.RaisesGroup(pytest.RaisesExc(FileNotFoundError)):
        await _collect(coalesced_get(fetch, [byte_range], **DEFAULT))


async def test_key_missing_mid_stream_raises_after_earlier_groups() -> None:
    """If a later fetch returns None, earlier-completed groups yield before the raise."""
    call_count = 0

    async def fetch(byte_range: ByteRequest | None) -> Buffer | None:
        nonlocal call_count
        call_count += 1
        # Deterministic: first call serves, second returns None.
        await asyncio.sleep(0.01 if call_count == 1 else 0.02)
        if call_count >= 2:
            return None
        return _buf(b"ok")

    opts: Mapping[str, int] = {
        "max_gap_bytes": -1,
        "max_coalesced_bytes": 1 << 20,
        "max_concurrency": 1,  # serialize for determinism
    }
    ranges: list[ByteRequest | None] = [RangeByteRequest(0, 2), RangeByteRequest(100, 102)]
    agen = coalesced_get(fetch, ranges, **opts)
    first = await anext(agen)
    assert len(first) == 1
    with pytest.RaisesGroup(pytest.RaisesExc(FileNotFoundError)):
        await anext(agen)


async def test_key_missing_mid_stream_with_concurrency_cancels_late_arrivals() -> None:
    """
    Under max_concurrency > 1, a mid-stream miss should raise FileNotFoundError
    and cancel still-in-flight unrelated tasks rather than wait for them.
    """
    late_gate = asyncio.Event()
    miss_fired = asyncio.Event()
    # Driven by the test body after the first successful yield, so the miss
    # task can't race past the start=0 result.
    fire_miss = asyncio.Event()

    async def fetch(byte_range: ByteRequest | None) -> Buffer | None:
        assert isinstance(byte_range, RangeByteRequest)
        start = byte_range.start
        if start == 0:
            return _buf(b"ok")
        if start == 1000:
            # Wait for the test to give the green light before returning None.
            # This makes ordering deterministic regardless of scheduling.
            await asyncio.wait_for(fire_miss.wait(), timeout=5.0)
            miss_fired.set()
            return None
        # Late arrivals would block on this gate; they should be cancelled
        # before they ever return.
        await asyncio.wait_for(late_gate.wait(), timeout=5.0)
        return _buf(b"ok")

    opts: Mapping[str, int] = {
        "max_gap_bytes": -1,
        "max_coalesced_bytes": 1 << 20,
        "max_concurrency": 3,
    }
    ranges: list[ByteRequest | None] = [RangeByteRequest(i * 1000, i * 1000 + 1) for i in range(7)]

    agen = coalesced_get(fetch, ranges, **opts)
    first = await anext(agen)
    assert len(first) == 1
    idx, buf = first[0]
    assert idx == 0
    assert buf is not None
    # Now that #0 has yielded, signal the miss task to return None.
    fire_miss.set()
    with pytest.RaisesGroup(pytest.RaisesExc(FileNotFoundError)):
        await anext(agen)
    assert miss_fired.is_set()
    # Sanity: late_gate was never set, so the cancellation path is what completed the test.
    assert not late_gate.is_set()


# ---------------------------------------------------------------------------
# Error propagation.
# ---------------------------------------------------------------------------


async def test_fetch_raises_propagates() -> None:
    """An exception raised by fetch propagates on the yield that produced the failing group."""
    fetch = FakeFetch(
        _INDEXED_BLOB,
        raise_on=lambda r: isinstance(r, RangeByteRequest) and r.start >= 100,
    )
    opts: Mapping[str, int] = {
        "max_gap_bytes": -1,
        "max_coalesced_bytes": 1 << 20,
        "max_concurrency": 1,
    }
    ranges: list[ByteRequest | None] = [RangeByteRequest(0, 10), RangeByteRequest(200, 210)]
    with pytest.RaisesGroup(pytest.RaisesExc(OSError, match="injected")):
        await _collect(coalesced_get(fetch, ranges, **opts))


# ---------------------------------------------------------------------------
# Property-style coverage invariant.
# ---------------------------------------------------------------------------


async def test_coverage_invariant_random_inputs() -> None:
    """For any random RangeByteRequest input, every input index appears exactly once."""
    import random

    rng = random.Random(42)
    fetch = FakeFetch(_INDEXED_BLOB)

    ranges: list[ByteRequest | None] = []
    for _ in range(50):
        start = rng.randint(0, 9000)
        length = rng.randint(1, 500)
        ranges.append(RangeByteRequest(start, start + length))

    groups = await _collect(coalesced_get(fetch, ranges, **DEFAULT))
    seen: list[int] = [idx for group in groups for idx, _buf in group]
    assert sorted(seen) == list(range(len(ranges)))

    flat = _contents(groups)
    for i, r in enumerate(ranges):
        assert isinstance(r, RangeByteRequest)
        assert flat[i] == _INDEXED_BLOB[r.start : r.end]


# ---------------------------------------------------------------------------
# Pure-function tests for coalesce_ranges (no async, no fetch).
# ---------------------------------------------------------------------------


def test_coalesce_ranges_empty_input() -> None:
    groups, uncoalescable = coalesce_ranges([], max_gap_bytes=1 << 20, max_coalesced_bytes=16 << 20)
    assert groups == []
    assert uncoalescable == []


def test_coalesce_ranges_separates_coalescable_from_uncoalescable() -> None:
    ranges: list[ByteRequest | None] = [
        RangeByteRequest(0, 10),
        OffsetByteRequest(100),
        SuffixByteRequest(5),
        None,
        RangeByteRequest(20, 30),
    ]
    groups, uncoalescable = coalesce_ranges(ranges, **_grouping(MERGE_GAP_50))

    # Both range requests fall within MERGE_GAP_50's gap budget.
    assert len(groups) == 1
    assert [idx for idx, _ in groups[0]] == [0, 4]

    # Non-RangeByteRequest entries preserve their original input indices.
    assert [(idx, type(req).__name__ if req else None) for idx, req in uncoalescable] == [
        (1, "OffsetByteRequest"),
        (2, "SuffixByteRequest"),
        (3, None),
    ]


def test_coalesce_ranges_no_merge_when_gap_exceeds_budget() -> None:
    ranges: list[ByteRequest | None] = [
        RangeByteRequest(0, 10),
        RangeByteRequest(200, 210),
        RangeByteRequest(500, 510),
    ]
    groups, uncoalescable = coalesce_ranges(ranges, **_grouping(MERGE_GAP_50))
    assert uncoalescable == []
    assert [len(g) for g in groups] == [1, 1, 1]
    assert [idx for g in groups for idx, _ in g] == [0, 1, 2]


def test_coalesce_ranges_merges_within_gap_budget() -> None:
    ranges: list[ByteRequest | None] = [
        RangeByteRequest(0, 5),
        RangeByteRequest(10, 15),
        RangeByteRequest(20, 25),
    ]
    groups, _ = coalesce_ranges(ranges, **_grouping(MERGE_GAP_50))
    assert len(groups) == 1
    assert [idx for idx, _ in groups[0]] == [0, 1, 2]


def test_coalesce_ranges_respects_max_coalesced_bytes() -> None:
    # Gap budget is permissive (1000), but the merged span would exceed CAP_50's
    # 50-byte cap, so the second range starts a new group.
    ranges: list[ByteRequest | None] = [
        RangeByteRequest(0, 30),
        RangeByteRequest(40, 80),
    ]
    groups, _ = coalesce_ranges(ranges, **_grouping(CAP_50))
    assert [len(g) for g in groups] == [1, 1]


def test_coalesce_ranges_groups_are_sorted_by_start() -> None:
    """Input order is irrelevant; groups always emerge in start-offset order."""
    ranges: list[ByteRequest | None] = [
        RangeByteRequest(500, 510),
        RangeByteRequest(0, 10),
        RangeByteRequest(20, 30),
        RangeByteRequest(200, 210),
    ]
    groups, _ = coalesce_ranges(ranges, **_grouping(MERGE_GAP_50))
    # First group is the {0-10, 20-30} cluster (from input indices 1, 2).
    # Then the {200-210} singleton, then {500-510}.
    flat = [idx for g in groups for idx, _ in g]
    assert flat == [1, 2, 3, 0]
    # Within each group, members are sorted by start.
    for g in groups:
        starts = [r.start for _, r in g]
        assert starts == sorted(starts)


def test_coalesce_ranges_overlapping_ranges_merge() -> None:
    """Nested/overlapping ranges have a non-positive 'gap' and always merge."""
    ranges: list[ByteRequest | None] = [
        RangeByteRequest(0, 100),
        RangeByteRequest(50, 60),  # nested
        RangeByteRequest(80, 120),  # overlaps
    ]
    groups, _ = coalesce_ranges(ranges, **_grouping(MERGE_GAP_50))
    assert len(groups) == 1
    assert [idx for idx, _ in groups[0]] == [0, 1, 2]


def test_coalesce_ranges_running_end_handles_nesting() -> None:
    """A subsequent range fully inside the running span must not extend group_end backwards."""
    ranges: list[ByteRequest | None] = [
        RangeByteRequest(0, 1000),  # group_end=1000
        RangeByteRequest(100, 200),  # nested; group_end stays at 1000
        RangeByteRequest(990, 1010),  # gap = -10 from running end, still merges
    ]
    groups, _ = coalesce_ranges(ranges, **_grouping(MERGE_GAP_50))
    assert len(groups) == 1
    assert {idx for idx, _ in groups[0]} == {0, 1, 2}


def test_coalesce_ranges_only_uncoalescable_inputs() -> None:
    ranges: list[ByteRequest | None] = [None, OffsetByteRequest(10), SuffixByteRequest(5)]
    groups, uncoalescable = coalesce_ranges(
        ranges, max_gap_bytes=1 << 20, max_coalesced_bytes=16 << 20
    )
    assert groups == []
    assert [idx for idx, _ in uncoalescable] == [0, 1, 2]


def test_coalesce_ranges_total_index_coverage() -> None:
    """Every input index appears exactly once across groups + uncoalescable."""
    ranges: list[ByteRequest | None] = [
        RangeByteRequest(0, 10),
        None,
        RangeByteRequest(15, 25),
        OffsetByteRequest(100),
        RangeByteRequest(30, 40),
    ]
    groups, uncoalescable = coalesce_ranges(ranges, **_grouping(MERGE_GAP_50))
    seen = sorted([idx for g in groups for idx, _ in g] + [idx for idx, _ in uncoalescable])
    assert seen == list(range(len(ranges)))
