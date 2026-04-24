# tests/test_coalesce.py
from __future__ import annotations

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
