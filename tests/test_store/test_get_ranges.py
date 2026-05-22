# tests/test_store/test_get_ranges.py
"""Tests for `Store.get_ranges` — the ABC default implementation and wrapper delegation.

`Store.get_ranges` is defined on the ABC with a default implementation built
on `coalesced_get(self.get, ...)`, so every store inherits a working version.
These tests cover that inherited path and the explicit delegation in
`WrapperStore` (which ensures wrapped stores' optimized overrides are honored).
Store-specific overrides (e.g. `FsspecStore`) have their own test modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from zarr.abc.store import RangeByteRequest
from zarr.core.buffer import default_buffer_prototype
from zarr.storage import MemoryStore
from zarr.storage._wrapper import WrapperStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from zarr.abc.store import ByteRequest
    from zarr.core.buffer import Buffer, BufferPrototype


async def _write(store: MemoryStore, key: str, data: bytes) -> None:
    buf = default_buffer_prototype().buffer.from_bytes(data)
    await store.set(key, buf)


async def test_memory_store_inherits_get_ranges_from_abc() -> None:
    """MemoryStore doesn't override `get_ranges`; the ABC default must work end-to-end."""
    store = MemoryStore()
    blob = bytes(i % 256 for i in range(512))
    await _write(store, "blob", blob)

    ranges = [RangeByteRequest(0, 10), RangeByteRequest(100, 110)]
    proto = default_buffer_prototype()
    flat: dict[int, bytes] = {}
    async for group in store.get_ranges("blob", ranges, prototype=proto):
        for idx, buf in group:
            assert buf is not None
            flat[idx] = buf.to_bytes()

    assert flat[0] == blob[0:10]
    assert flat[1] == blob[100:110]


async def test_memory_store_get_ranges_missing_key_raises() -> None:
    """A missing key on a default-impl store raises BaseExceptionGroup containing FileNotFoundError."""
    store = MemoryStore()
    proto = default_buffer_prototype()
    agen = store.get_ranges("does-not-exist", [RangeByteRequest(0, 10)], prototype=proto)
    with pytest.RaisesGroup(pytest.RaisesExc(FileNotFoundError)):
        await anext(agen)


async def test_wrapper_store_delegates_get_ranges() -> None:
    """WrapperStore.get_ranges must delegate to the wrapped store, not fall back to the default."""

    class CountingMemoryStore(MemoryStore):
        """Tallies get_ranges invocations so we can assert delegation."""

        get_ranges_calls: int = 0

        async def get_ranges(
            self,
            key: str,
            byte_ranges: Sequence[ByteRequest | None],
            *,
            prototype: BufferPrototype,
            max_concurrency: int = 10,
            max_gap_bytes: int = 1 << 20,
            max_coalesced_bytes: int = 16 << 20,
        ) -> AsyncIterator[Sequence[tuple[int, Buffer | None]]]:
            type(self).get_ranges_calls += 1
            async for group in super().get_ranges(
                key,
                byte_ranges,
                prototype=prototype,
                max_concurrency=max_concurrency,
                max_gap_bytes=max_gap_bytes,
                max_coalesced_bytes=max_coalesced_bytes,
            ):
                yield group

    inner = CountingMemoryStore()
    blob = b"x" * 100
    await _write(inner, "k", blob)
    wrapped = WrapperStore(inner)

    proto = default_buffer_prototype()
    groups: list[list[tuple[int, Buffer | None]]] = [
        list(group)
        async for group in wrapped.get_ranges("k", [RangeByteRequest(0, 5)], prototype=proto)
    ]

    assert CountingMemoryStore.get_ranges_calls == 1
    assert len(groups) == 1
    assert groups[0][0][0] == 0


async def test_wrapper_store_forwards_coalescing_kwargs() -> None:
    """Coalescing kwargs flow through WrapperStore to the wrapped store's get_ranges."""

    class SpyMemoryStore(MemoryStore):
        last_max_gap_bytes: int | None = None

        async def get_ranges(
            self,
            key: str,
            byte_ranges: Sequence[ByteRequest | None],
            *,
            prototype: BufferPrototype,
            max_concurrency: int = 10,
            max_gap_bytes: int = 1 << 20,
            max_coalesced_bytes: int = 16 << 20,
        ) -> AsyncIterator[Sequence[tuple[int, Buffer | None]]]:
            type(self).last_max_gap_bytes = max_gap_bytes
            async for group in super().get_ranges(
                key,
                byte_ranges,
                prototype=prototype,
                max_concurrency=max_concurrency,
                max_gap_bytes=max_gap_bytes,
                max_coalesced_bytes=max_coalesced_bytes,
            ):
                yield group

    inner = SpyMemoryStore()
    await _write(inner, "k", b"y" * 100)
    wrapped = WrapperStore(inner)
    proto = default_buffer_prototype()
    async for _ in wrapped.get_ranges(
        "k", [RangeByteRequest(0, 5)], prototype=proto, max_gap_bytes=-1
    ):
        pass

    assert SpyMemoryStore.last_max_gap_bytes == -1
