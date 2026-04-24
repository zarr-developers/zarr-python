# src/zarr/storage/_protocols.py
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable, Sequence

    from zarr.abc.store import ByteRequest
    from zarr.core.buffer import Buffer, BufferPrototype


@runtime_checkable
class SupportsGetRanges(Protocol):
    """Stores that satisfy this protocol can efficiently read many byte ranges
    from a single key in a single call, typically via coalescing and concurrent fetch.

    Private / unstable. Shape may change before being made public.
    """

    def get_ranges(
        self,
        key: str,
        byte_ranges: Iterable[ByteRequest | None],
        *,
        prototype: BufferPrototype,
    ) -> AsyncIterator[Sequence[tuple[int, Buffer | None]]]:
        """Read many byte ranges from ``key``.

        Each yield corresponds to one underlying I/O operation.

        See :func:`zarr.core._coalesce.coalesced_get` for full semantics.
        """
        ...
