from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Self

import numpy as np

from zarr_storage.legacy._abc import ByteRequest, Store
from zarr_storage.legacy._wrapper import WrapperStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable, Sequence

    from zarr.core.buffer import Buffer
    from zarr.core.buffer.core import BufferPrototype


class LatencyStore(WrapperStore[Store]):
    """
    A wrapper class that takes any store class in its constructor and
    adds latency to the `set` and `get` methods. This can be used for
    performance testing.
    """

    _get_latency: float | tuple[float, float]
    _set_latency: float | tuple[float, float]

    def __init__(
        self,
        store: Store,
        *,
        get_latency: float | tuple[float, float] = 0,
        set_latency: float | tuple[float, float] = 0,
    ) -> None:
        super().__init__(store)
        self._get_latency = get_latency if isinstance(get_latency, tuple) else float(get_latency)
        self._set_latency = set_latency if isinstance(set_latency, tuple) else float(set_latency)

    @property
    def get_latency(self) -> float:
        if isinstance(self._get_latency, float):
            return self._get_latency
        return max(0.0, np.random.normal(loc=self._get_latency[0], scale=self._get_latency[1]))

    @property
    def set_latency(self) -> float:
        if isinstance(self._set_latency, float):
            return self._set_latency
        return max(0.0, np.random.normal(loc=self._set_latency[0], scale=self._set_latency[1]))

    def _with_store(self, store: Store) -> Self:
        # Pass the raw latency config, not the sampled `get_latency`/`set_latency`
        # properties — sampling would freeze a `(loc, scale)` distribution into
        # one fixed float on derived stores (e.g. via `with_read_only`).
        return type(self)(store, get_latency=self._get_latency, set_latency=self._set_latency)

    async def set(self, key: str, value: Buffer) -> None:
        """
        Add latency to the ``set`` method.

        Calls ``asyncio.sleep(self.set_latency)`` before invoking the wrapped ``set`` method.

        Parameters
        ----------
        key : str
            The key to set
        value : Buffer
            The value to set

        Returns
        -------
        None
        """
        await asyncio.sleep(self.set_latency)
        await self._store.set(key, value)

    async def get(
        self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        """
        Add latency to the ``get`` method.

        Calls ``asyncio.sleep(self.get_latency)`` before invoking the wrapped ``get`` method.

        Parameters
        ----------
        key : str
            The key to get
        prototype : BufferPrototype
            The BufferPrototype to use.
        byte_range : ByteRequest, optional
            An optional byte range.

        Returns
        -------
        buffer : Buffer or None
        """
        await asyncio.sleep(self.get_latency)
        return await self._store.get(key, prototype=prototype, byte_range=byte_range)

    def get_sync(
        self,
        key: str,
        *,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        """Add latency to `get_sync`.

        Sleeps `self.get_latency` on the calling thread (the sync path runs on
        worker threads, not the event loop) before delegating to the wrapped
        store.
        """
        time.sleep(self.get_latency)
        return super().get_sync(key, prototype=prototype, byte_range=byte_range)

    def set_sync(self, key: str, value: Buffer) -> None:
        """Add latency to `set_sync`.

        Sleeps `self.set_latency` on the calling thread (the sync path runs on
        worker threads, not the event loop) before delegating to the wrapped
        store.
        """
        time.sleep(self.set_latency)
        super().set_sync(key, value)

    async def get_ranges(
        self,
        key: str,
        byte_ranges: Sequence[ByteRequest | None],
        *,
        prototype: BufferPrototype,
        max_concurrency: int | None = None,
        max_gap_bytes: int | None = None,
        max_coalesced_bytes: int | None = None,
    ) -> AsyncIterator[Sequence[tuple[int, Buffer | None]]]:
        """Byte-range reads built on `self.get`, so each fetch pays latency.

        Routes through the coalescing `Store.get_ranges` default instead of the
        `WrapperStore` delegation, which would bypass this wrapper's `get` and
        therefore the synthetic latency. `None` for a coalescing kwarg means
        "use the `Store` default".
        """
        kwargs: dict[str, int] = {}
        if max_concurrency is not None:
            kwargs["max_concurrency"] = max_concurrency
        if max_gap_bytes is not None:
            kwargs["max_gap_bytes"] = max_gap_bytes
        if max_coalesced_bytes is not None:
            kwargs["max_coalesced_bytes"] = max_coalesced_bytes
        async for group in Store.get_ranges(self, key, byte_ranges, prototype=prototype, **kwargs):
            yield group

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        """Partial-value reads built on `self.get`, so each fetch pays latency.

        Issues one `self.get` per `(key, byte_range)` pair instead of the
        `WrapperStore` delegation, which would bypass this wrapper's `get` and
        therefore the synthetic latency.
        """
        return list(
            await asyncio.gather(
                *(
                    self.get(key, prototype=prototype, byte_range=byte_range)
                    for key, byte_range in key_ranges
                )
            )
        )
