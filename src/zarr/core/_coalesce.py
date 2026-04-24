# src/zarr/core/_coalesce.py
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Sequence

    from zarr.abc.store import ByteRequest
    from zarr.core.buffer import Buffer


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
) -> AsyncIterator[Sequence[tuple[int, Buffer | None]]]:
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

    # Sort mergeables by start offset, then merge.
    mergeable.sort(key=lambda pair: pair[1].start)
    groups: list[list[tuple[int, RangeByteRequest]]] = []
    for pair in mergeable:
        _i, r = pair
        if groups:
            last = groups[-1]
            last_end = max(x[1].end for x in last)
            gap = r.start - last_end
            merged_start = min(x[1].start for x in last)
            prospective_end = max(last_end, r.end)
            prospective_size = prospective_end - merged_start
            if (
                gap <= options["max_gap_bytes"]
                and prospective_size <= options["max_coalesced_bytes"]
            ):
                last.append(pair)
                continue
        groups.append([pair])

    # Build a uniform list of work items. Each work item is a list of
    # (input_index, ByteRequest | None) pairs. Merged groups have multiple
    # members (all RangeByteRequest); uncoalescable items have a single member.
    work_items: list[list[tuple[int, ByteRequest | None]]] = [
        [(idx, r) for idx, r in g] for g in groups
    ]
    work_items.extend([(idx, single)] for idx, single in uncoalescable)

    total = len(work_items)
    if total == 0:
        return

    # Completion queue entries are either ("ok", payload), ("missing", None),
    # or ("error", exception). Kept as Any internally to avoid dragging
    # Sequence out of TYPE_CHECKING.
    completion_queue: asyncio.Queue[
        tuple[str, Sequence[tuple[int, Buffer | None]] | BaseException | None]
    ] = asyncio.Queue()
    semaphore = asyncio.Semaphore(options["max_concurrency"])

    async def run_one(members: list[tuple[int, ByteRequest | None]]) -> None:
        try:
            async with semaphore:
                if len(members) == 1 and not isinstance(members[0][1], RangeByteRequest):
                    # Uncoalescable single fetch.
                    idx, single = members[0]
                    buf = await fetch(single)
                    if buf is None:
                        await completion_queue.put(("missing", None))
                        return
                    await completion_queue.put(("ok", ((idx, buf),)))
                    return
                # Merged group path: all members are RangeByteRequest.
                starts = [r.start for _, r in members if isinstance(r, RangeByteRequest)]
                ends = [r.end for _, r in members if isinstance(r, RangeByteRequest)]
                group_start = min(starts)
                group_end = max(ends)
                big = await fetch(RangeByteRequest(group_start, group_end))
                if big is None:
                    await completion_queue.put(("missing", None))
                    return
                ordered = sorted(
                    members,
                    key=lambda pair: pair[1].start if isinstance(pair[1], RangeByteRequest) else 0,
                )
                sliced: list[tuple[int, Buffer | None]] = []
                for idx, r in ordered:
                    assert isinstance(r, RangeByteRequest)
                    sliced.append((idx, big[r.start - group_start : r.end - group_start]))
                await completion_queue.put(("ok", tuple(sliced)))
        except asyncio.CancelledError:
            # Cancellation is expected when we stop scheduling on a missing key.
            raise
        except BaseException as exc:
            await completion_queue.put(("error", exc))

    # Launch all work items as tasks. The semaphore bounds actual concurrency.
    tasks: set[asyncio.Task[None]] = set()
    for item in work_items:
        tasks.add(asyncio.create_task(run_one(item)))

    try:
        drained = 0
        stopped = False
        pending_error: BaseException | None = None
        while drained < total:
            kind, payload = await completion_queue.get()
            drained += 1
            if stopped:
                continue  # Discard remaining results after a miss or error.
            if kind == "ok":
                assert payload is not None
                assert not isinstance(payload, BaseException)
                yield payload
            elif kind == "missing":
                stopped = True
                # Cancel any still-pending tasks to avoid unnecessary I/O.
                for t in tasks:
                    if not t.done():
                        t.cancel()
            else:  # "error"
                assert isinstance(payload, BaseException)
                stopped = True
                pending_error = payload
                for t in tasks:
                    if not t.done():
                        t.cancel()
        if pending_error is not None:
            raise pending_error
    finally:
        # Ensure we wait for any cancelled tasks to finish so no task escapes.
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
