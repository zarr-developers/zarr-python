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


# A permissive options value used by most tests (heavy merging allowed).
HEAVY_MERGE: CoalesceOptions = {
    "max_gap_bytes": 1 << 20,
    "max_coalesced_bytes": 1 << 30,
    "max_concurrency": 10,
}

# An options value that forbids all merging (gap threshold 0 and any adjacent
# merges would still be allowed at gap==0, so we also cap size).
NO_MERGE: CoalesceOptions = {
    "max_gap_bytes": -1,  # strictly less-than semantics -- any positive gap breaks
    "max_coalesced_bytes": 1 << 30,
    "max_concurrency": 10,
}


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
