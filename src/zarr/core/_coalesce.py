# src/zarr/core/_coalesce.py
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Final, NamedTuple

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Sequence

    from zarr.abc.store import ByteRequest, RangeByteRequest
    from zarr.core.buffer import Buffer


class _WorkerCtx(NamedTuple):
    """Shared state passed to the per-task worker coroutines.

    Bundling these lets the workers declare their dependencies as one
    parameter instead of capturing them implicitly via closure.
    """

    fetch: Callable[[ByteRequest | None], Awaitable[Buffer | None]]
    semaphore: asyncio.Semaphore


async def _fetch_single(
    ctx: _WorkerCtx, idx: int, req: ByteRequest | None
) -> Sequence[tuple[int, Buffer | None]]:
    """Fetch one byte range. Raises FileNotFoundError if the key is absent."""
    async with ctx.semaphore:
        buf = await ctx.fetch(req)
    if buf is None:
        raise FileNotFoundError
    return ((idx, buf),)


async def _fetch_group(
    ctx: _WorkerCtx, members: list[tuple[int, RangeByteRequest]]
) -> Sequence[tuple[int, Buffer | None]]:
    """Fetch one merged byte range and slice it back into per-input buffers.

    `members` must already be sorted by `start`; callers in this module
    build it from the sorted mergeable list. Raises `FileNotFoundError`
    if the key is absent.
    """
    from zarr.abc.store import RangeByteRequest

    start = members[0][1].start
    end = max(r.end for _, r in members)
    async with ctx.semaphore:
        big = await ctx.fetch(RangeByteRequest(start, end))
    if big is None:
        raise FileNotFoundError
    sliced: list[tuple[int, Buffer | None]] = [
        (idx, big[r.start - start : r.end - start]) for idx, r in members
    ]
    return tuple(sliced)


COALESCE_DEFAULT_MAX_GAP_BYTES: Final = 1 << 20
COALESCE_DEFAULT_MAX_COALESCED_BYTES: Final = 16 << 20
COALESCE_DEFAULT_MAX_CONCURRENCY: Final = 10


def coalesce_ranges(
    byte_ranges: Sequence[ByteRequest | None],
    *,
    max_gap_bytes: int | None = None,
    max_coalesced_bytes: int | None = None,
) -> tuple[
    list[list[tuple[int, RangeByteRequest]]],
    list[tuple[int, ByteRequest | None]],
]:
    """Plan a set of byte-range fetches: which inputs merge, which stand alone.

    Pure (no I/O). The result is the I/O plan a caller would execute: each
    group corresponds to one fetch of a coalesced byte range, and each
    uncoalescable item corresponds to one fetch of the original request.

    Parameters
    ----------
    byte_ranges
        Input ranges. `None` means "the whole value".
    max_gap_bytes
        Two `RangeByteRequest`s separated by at most this many bytes may be
        merged into one fetch. Defaults to `COALESCE_DEFAULT_MAX_GAP_BYTES`.
    max_coalesced_bytes
        Upper bound on the size of a single merged fetch. Defaults to
        `COALESCE_DEFAULT_MAX_COALESCED_BYTES`.

    Returns
    -------
    groups
        List of merged groups. Each group is a list of
        `(input_index, RangeByteRequest)` pairs sorted by `start`. A
        single-element group represents a `RangeByteRequest` that did not
        merge with any neighbor.
    uncoalescable
        List of `(input_index, request)` pairs for inputs that are not
        `RangeByteRequest` (`OffsetByteRequest`, `SuffixByteRequest`,
        `None`). Indices are preserved from the input order.

    Notes
    -----
    Only `RangeByteRequest` inputs participate in coalescing. Two ranges
    merge when both: their gap (next `start` minus current group's running
    `end`) is `<= max_gap_bytes`, and the resulting merged span is
    `<= max_coalesced_bytes`.
    """
    # Local import to avoid cycles at module import time.
    from zarr.abc.store import RangeByteRequest

    if max_gap_bytes is None:
        max_gap_bytes = COALESCE_DEFAULT_MAX_GAP_BYTES
    if max_coalesced_bytes is None:
        max_coalesced_bytes = COALESCE_DEFAULT_MAX_COALESCED_BYTES

    indexed: list[tuple[int, ByteRequest | None]] = list(enumerate(byte_ranges))
    mergeable: list[tuple[int, RangeByteRequest]] = [
        (i, r) for i, r in indexed if isinstance(r, RangeByteRequest)
    ]
    uncoalescable: list[tuple[int, ByteRequest | None]] = [
        (i, r) for i, r in indexed if not isinstance(r, RangeByteRequest)
    ]

    # Sort mergeables by start offset, then merge. Track running start/end of the
    # current group so each merge step is O(1) instead of O(group size).
    mergeable.sort(key=lambda pair: pair[1].start)
    groups: list[list[tuple[int, RangeByteRequest]]] = []
    group_start = 0
    group_end = 0
    for pair in mergeable:
        _i, r = pair
        if groups and r.start - group_end <= max_gap_bytes:
            prospective_end = max(group_end, r.end)
            if prospective_end - group_start <= max_coalesced_bytes:
                groups[-1].append(pair)
                group_end = prospective_end
                continue
        groups.append([pair])
        group_start = r.start
        group_end = r.end

    return groups, uncoalescable


async def coalesced_get(
    fetch: Callable[[ByteRequest | None], Awaitable[Buffer | None]],
    byte_ranges: Sequence[ByteRequest | None],
    *,
    max_concurrency: int | None = None,
    max_gap_bytes: int | None = None,
    max_coalesced_bytes: int | None = None,
) -> AsyncIterator[Sequence[tuple[int, Buffer | None]]]:
    """Read many byte ranges through `fetch` with coalescing and concurrency.

    Nearby ranges are merged into a single underlying I/O, and merged fetches
    are run concurrently. Each yield corresponds to exactly one underlying I/O
    operation: a sequence of `(input_index, result)` tuples for all input
    ranges served by that I/O. Tuples within a yielded sequence are ordered by
    start offset. Yields across groups are in completion order, not input
    order.

    Parameters
    ----------
    fetch
        Callable that reads one byte range and returns a `Buffer` (or `None`
        if the underlying key does not exist). Typically constructed via
        `functools.partial(store.get, key, prototype)`.
    byte_ranges
        Input ranges. `None` means "the whole value".
    max_concurrency
        Maximum number of merged fetches in flight at once. Defaults to
        `COALESCE_DEFAULT_MAX_CONCURRENCY`.
    max_gap_bytes
        Forwarded to `coalesce_ranges`.
    max_coalesced_bytes
        Forwarded to `coalesce_ranges`.

    Yields
    ------
    Sequence[tuple[int, Buffer | None]]
        Per-I/O batch of `(input_index, result)` tuples.

    Notes
    -----
    - Only `RangeByteRequest` inputs are coalesced. `OffsetByteRequest`,
      `SuffixByteRequest`, and `None` are each treated as uncoalescable
      (one fetch, one single-tuple yield per input).
    - If any fetch returns `None`, the iterator raises `FileNotFoundError`
      after cancelling pending fetches. Groups completed before the miss
      remain observable on the yields preceding the raise.
    - If a fetch raises, the exception propagates on the yield that produced
      the failing group; earlier-completed groups remain observable.
    """
    if not byte_ranges:
        return

    if max_concurrency is None:
        max_concurrency = COALESCE_DEFAULT_MAX_CONCURRENCY

    groups, singles = coalesce_ranges(
        byte_ranges,
        max_gap_bytes=max_gap_bytes,
        max_coalesced_bytes=max_coalesced_bytes,
    )

    ctx = _WorkerCtx(fetch=fetch, semaphore=asyncio.Semaphore(max_concurrency))

    # Launch all work as tasks. The semaphore bounds actual I/O concurrency.

    async with asyncio.TaskGroup() as tg:
        tasks = [
            *(tg.create_task(_fetch_group(ctx, group)) for group in groups),
            *(tg.create_task(_fetch_single(ctx, i, single)) for i, single in singles),
        ]

        for fut in asyncio.as_completed(tasks):
            yield await fut
