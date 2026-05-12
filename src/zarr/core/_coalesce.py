# src/zarr/core/_coalesce.py
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Final, Literal, NamedTuple

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence

    from zarr.abc.store import ByteRequest, RangeByteRequest
    from zarr.core.buffer import Buffer

    _CompletionEntry = (
        tuple[Literal["ok"], Sequence[tuple[int, Buffer | None]]]
        | tuple[Literal["missing"], None]
        | tuple[Literal["error"], BaseException]
    )


class _WorkerCtx(NamedTuple):
    """Shared state passed to the per-task worker coroutines.

    Bundling these lets the workers declare their dependencies as one
    parameter instead of capturing them implicitly via closure.
    """

    fetch: Callable[[ByteRequest | None], Awaitable[Buffer | None]]
    semaphore: asyncio.Semaphore
    queue: asyncio.Queue[_CompletionEntry]


async def _fetch_single(ctx: _WorkerCtx, idx: int, req: ByteRequest | None) -> None:
    try:
        async with ctx.semaphore:
            buf = await ctx.fetch(req)
        if buf is None:
            await ctx.queue.put(("missing", None))
            return
        await ctx.queue.put(("ok", ((idx, buf),)))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        await ctx.queue.put(("error", exc))


async def _fetch_group(ctx: _WorkerCtx, members: list[tuple[int, RangeByteRequest]]) -> None:
    """Fetch one merged byte range and slice it back into per-input buffers.

    `members` must already be sorted by `start`; callers in this module
    build it from the sorted mergeable list.
    """
    from zarr.abc.store import RangeByteRequest

    try:
        start = members[0][1].start
        end = max(r.end for _, r in members)
        async with ctx.semaphore:
            big = await ctx.fetch(RangeByteRequest(start, end))
        if big is None:
            await ctx.queue.put(("missing", None))
            return
        sliced: list[tuple[int, Buffer | None]] = [
            (idx, big[r.start - start : r.end - start]) for idx, r in members
        ]
        await ctx.queue.put(("ok", tuple(sliced)))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        await ctx.queue.put(("error", exc))


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
) -> AsyncGenerator[Sequence[tuple[int, Buffer | None]], None]:
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
    - If any fetch returns `None` the iterator stops scheduling further fetches
      and completes without yielding the missing group. Groups completed before
      the miss remain observable.
    - If a fetch raises, the exception propagates on the yield that produced the
      failing group; earlier-completed groups remain observable.
    """
    if max_concurrency is None:
        max_concurrency = COALESCE_DEFAULT_MAX_CONCURRENCY

    groups, uncoalescable = coalesce_ranges(
        byte_ranges,
        max_gap_bytes=max_gap_bytes,
        max_coalesced_bytes=max_coalesced_bytes,
    )
    if not groups and not uncoalescable:
        return

    ctx = _WorkerCtx(
        fetch=fetch,
        semaphore=asyncio.Semaphore(max_concurrency),
        queue=asyncio.Queue(),
    )

    # Launch all work as tasks. The semaphore bounds actual I/O concurrency.
    # A one-member group is a RangeByteRequest that did not merge with a
    # neighbor; route it through _fetch_single so it skips the redundant
    # slice-by-zero in _fetch_group.
    tasks: set[asyncio.Task[None]] = set()
    for group in groups:
        if len(group) == 1:
            solo_idx, solo_req = group[0]
            tasks.add(asyncio.create_task(_fetch_single(ctx, solo_idx, solo_req)))
        else:
            tasks.add(asyncio.create_task(_fetch_group(ctx, group)))
    for idx, single in uncoalescable:
        tasks.add(asyncio.create_task(_fetch_single(ctx, idx, single)))
    total_work = len(tasks)

    try:
        pending_error: BaseException | None = None
        for _ in range(total_work):
            entry = await ctx.queue.get()
            if entry[0] == "ok":
                yield entry[1]
                continue
            # "missing" or "error": stop scheduling and cancel pending work.
            # Late arrivals that raced to enqueue before cancellation took
            # effect sit in the completion queue and are discarded by the
            # finally block (the queue is local and will be garbage-collected).
            for t in tasks:
                if not t.done():
                    t.cancel()
            if entry[0] == "error":
                pending_error = entry[1]
            break
        if pending_error is not None:
            raise pending_error
    finally:
        # Best-effort cancellation for in-flight tasks (covers the consumer
        # break / early-exit case where we did not proactively cancel).
        for t in tasks:
            if not t.done():
                t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
