# src/zarr/core/_coalesce.py
from __future__ import annotations

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
    """Read many byte ranges through ``fetch``, coalescing nearby ranges and firing merged requests concurrently.

    Each yield corresponds to exactly one underlying I/O operation: a sequence of
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

    # For now, serve groups sequentially (concurrency added in Task 5).
    for group in groups:
        group_start = min(x[1].start for x in group)
        group_end = max(x[1].end for x in group)
        big = await fetch(RangeByteRequest(group_start, group_end))
        if big is None:
            return  # key missing, stop yielding
        # Slice back into per-input buffers, ordered by start offset.
        group.sort(key=lambda pair: pair[1].start)
        yielded: list[tuple[int, Buffer | None]] = []
        for i, r in group:
            local_start = r.start - group_start
            local_end = r.end - group_start
            yielded.append((i, big[local_start:local_end]))
        yield tuple(yielded)

    # Uncoalescable inputs are fetched one at a time, each as its own one-tuple group.
    for idx, single in uncoalescable:
        buf = await fetch(single)
        if buf is None:
            return  # key missing, stop yielding
        yield ((idx, buf),)
