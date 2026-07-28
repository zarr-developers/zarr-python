# src/zarr/core/_coalesce.py
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, NamedTuple

from zarr.abc.store import RangeByteRequest

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence

    from zarr.abc.store import ByteRequest
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
    if len(members) == 1:
        solo_idx, solo_req = members[0]
        return await _fetch_single(ctx, solo_idx, solo_req)

    start = members[0][1].start
    end = max(r.end for _, r in members)
    async with ctx.semaphore:
        big = await ctx.fetch(RangeByteRequest(start, end))
    if big is None:
        raise FileNotFoundError
    sliced = [(idx, big[r.start - start : r.end - start]) for idx, r in members]
    return tuple(sliced)


def coalesce_ranges(
    byte_ranges: Sequence[ByteRequest | None],
    *,
    max_gap_bytes: int,
    max_coalesced_bytes: int,
) -> tuple[
    list[list[tuple[int, RangeByteRequest]]],
    list[tuple[int, ByteRequest | None]],
]:
    """Plan a set of byte-range fetches: which inputs merge, which stand alone.

    Pure (no I/O). The result is the I/O plan a caller would execute: each
    group corresponds to one fetch of a coalesced byte range, and each
    uncoalescable item corresponds to one fetch of the original request.

    All tuning knobs are required keyword arguments. `Store.get_ranges` is
    the public entry point and owns the canonical default values; this
    function takes them explicitly to avoid duplicating policy.

    Parameters
    ----------
    byte_ranges
        Input ranges. `None` means "the whole value".
    max_gap_bytes
        Two `RangeByteRequest`s separated by at most this many bytes may be
        merged into one fetch.
    max_coalesced_bytes
        Upper bound on the size of a single merged fetch.

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
    indexed = list(enumerate(byte_ranges))
    mergeable = [(i, r) for i, r in indexed if isinstance(r, RangeByteRequest)]
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
    max_concurrency: int,
    max_gap_bytes: int,
    max_coalesced_bytes: int,
) -> AsyncGenerator[Sequence[tuple[int, Buffer | None]]]:
    """Read many byte ranges through `fetch` with coalescing and concurrency.

    Nearby ranges are merged into a single underlying I/O, and merged fetches
    are run concurrently. Each yield corresponds to exactly one underlying I/O
    operation: a sequence of `(input_index, result)` tuples for all input
    ranges served by that I/O. Tuples within a yielded sequence are ordered by
    start offset. Yields across groups are in completion order, not input
    order.

    All tuning knobs are required keyword arguments. `Store.get_ranges` is
    the public entry point and owns the canonical default values; this
    function takes them explicitly to avoid duplicating policy.

    Parameters
    ----------
    fetch
        Callable that reads one byte range and returns a `Buffer` (or `None`
        if the underlying key does not exist). Typically constructed via
        `functools.partial(store.get, key, prototype)`.
    byte_ranges
        Input ranges. `None` means "the whole value".
    max_concurrency
        Maximum number of merged fetches in flight at once.
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
    - Failures from underlying fetches surface as a `BaseExceptionGroup`
      (PEP 654). Inner exceptions include `FileNotFoundError` if a fetch
      returns `None`, plus any exception `fetch` raises. Pending fetches are
      cancelled as soon as one task fails, so the group typically contains a
      single non-`CancelledError` exception even under high concurrency.
    - Groups completed before the failure remain observable on the yields
      preceding the raise.
    - `GeneratorExit` raised by `aclose()` is filtered out so the iterator
      closes cleanly; callers don't see a group containing only it.
    """
    if not byte_ranges:
        return

    groups, singles = coalesce_ranges(
        byte_ranges,
        max_gap_bytes=max_gap_bytes,
        max_coalesced_bytes=max_coalesced_bytes,
    )

    ctx = _WorkerCtx(fetch=fetch, semaphore=asyncio.Semaphore(max_concurrency))

    # Launch all work as tasks. The semaphore bounds actual I/O concurrency.
    # TaskGroup wraps task exceptions in BaseExceptionGroup; we propagate the
    # group unchanged as part of the public contract (callers handle batch
    # failures via `except*` / PEP 654). GeneratorExit (raised when the
    # consumer calls aclose()) is filtered out so close completes cleanly.
    try:
        async with asyncio.TaskGroup() as tg:
            tasks = [
                *(tg.create_task(_fetch_group(ctx, group)) for group in groups),
                *(tg.create_task(_fetch_single(ctx, i, single)) for i, single in singles),
            ]

            for fut in asyncio.as_completed(tasks):
                yield await fut
    except BaseExceptionGroup as eg:
        # Strip GeneratorExits (consumer aclose()) and propagate whatever remains.
        _, other_errors = eg.split(GeneratorExit)

        if other_errors is not None:
            raise other_errors from None
