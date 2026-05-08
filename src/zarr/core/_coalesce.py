# src/zarr/core/_coalesce.py
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Literal, NamedTuple, TypedDict

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable, Sequence

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

    ``members`` must already be sorted by ``start``; callers in this module
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


class CoalesceOptions(TypedDict):
    """Knobs for coalescing contiguous byte ranges into fewer I/O requests.

    All fields required. See DEFAULT_COALESCE_OPTIONS for a sensible default.
    """

    max_gap_bytes: int
    """Two RangeByteRequests separated by at most this many bytes may be merged into one fetch."""
    max_coalesced_bytes: int
    """Upper bound on the size of a single merged fetch (ignored for an already-oversized single request)."""
    max_concurrency: int
    """Maximum number of merged fetches in flight at once."""


DEFAULT_COALESCE_OPTIONS: CoalesceOptions = {
    "max_gap_bytes": 1 << 20,  # 1 MiB
    "max_coalesced_bytes": 16 << 20,  # 16 MiB
    "max_concurrency": 10,
}


async def coalesced_get(
    fetch: Callable[[ByteRequest | None], Awaitable[Buffer | None]],
    byte_ranges: Iterable[ByteRequest | None],
    *,
    options: CoalesceOptions,
) -> AsyncGenerator[Sequence[tuple[int, Buffer | None]], None]:
    """Read many byte ranges through ``fetch`` with coalescing and concurrency.

    Nearby ranges are merged into a single underlying I/O (subject to
    ``options``), and merged fetches are run concurrently. Each yield
    corresponds to exactly one underlying I/O operation: a sequence of
    ``(input_index, result)`` tuples for all input ranges served by that I/O.
    Tuples within a yielded sequence are ordered by start offset. Yields across
    groups are in completion order, not input order.

    Parameters
    ----------
    fetch
        Callable that reads one byte range and returns a ``Buffer`` (or ``None``
        if the underlying key does not exist). Typically constructed via
        ``functools.partial(store.get, key, prototype)``.
    byte_ranges
        Input ranges. ``None`` means "the whole value".
    options
        Coalescing knobs.

    Yields
    ------
    Sequence[tuple[int, Buffer | None]]
        Per-I/O batch of ``(input_index, result)`` tuples.

    Notes
    -----
    - Only ``RangeByteRequest`` inputs are coalesced. ``OffsetByteRequest``,
      ``SuffixByteRequest``, and ``None`` are each treated as uncoalescable
      (one fetch, one single-tuple yield per input).
    - If any fetch returns ``None`` the iterator stops scheduling further fetches
      and completes without yielding the missing group. Groups completed before
      the miss remain observable.
    - If a fetch raises, the exception propagates on the yield that produced the
      failing group; earlier-completed groups remain observable.
    """
    # Local import to avoid cycles at module import time.
    from zarr.abc.store import RangeByteRequest

    indexed: list[tuple[int, ByteRequest | None]] = list(enumerate(byte_ranges))
    if not indexed:
        return

    # Split inputs into coalescable (RangeByteRequest only) and uncoalescable (the rest).
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
        if groups and r.start - group_end <= options["max_gap_bytes"]:
            prospective_end = max(group_end, r.end)
            if prospective_end - group_start <= options["max_coalesced_bytes"]:
                groups[-1].append(pair)
                group_end = prospective_end
                continue
        groups.append([pair])
        group_start = r.start
        group_end = r.end

    ctx = _WorkerCtx(
        fetch=fetch,
        semaphore=asyncio.Semaphore(options["max_concurrency"]),
        queue=asyncio.Queue(),
    )

    # Launch all work as tasks. The semaphore bounds actual I/O concurrency.
    tasks: set[asyncio.Task[None]] = set()
    for group in groups:
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
