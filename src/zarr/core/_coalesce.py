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
    # Stub body; real implementation filled in by later tasks.
    raise NotImplementedError
    yield ()  # type: ignore[unreachable]  # pragma: no cover -- keeps this function an async generator
