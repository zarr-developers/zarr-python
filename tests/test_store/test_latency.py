from __future__ import annotations

import time
from unittest.mock import patch

import numpy as np
import pytest

import zarr
from zarr.abc.store import RangeByteRequest
from zarr.core.buffer import default_buffer_prototype
from zarr.core.codec_pipeline import FusedCodecPipeline
from zarr.core.config import config as zarr_config
from zarr.storage import MemoryStore
from zarr.testing.store import LatencyStore


async def test_latency_store_with_read_only_round_trip() -> None:
    """
    Ensure that LatencyStore.with_read_only returns another LatencyStore with
    the requested read_only state, preserves latency configuration, and does
    not change the original wrapper.
    """
    base = await MemoryStore.open()
    # Start from a read-only underlying store
    ro_base = base.with_read_only(read_only=True)
    latency_ro = LatencyStore(ro_base, get_latency=0.01, set_latency=0.02)

    assert latency_ro.read_only
    assert latency_ro.get_latency == pytest.approx(0.01)
    assert latency_ro.set_latency == pytest.approx(0.02)

    buf = default_buffer_prototype().buffer.from_bytes(b"abcd")

    # Cannot write through the read-only wrapper
    with pytest.raises(
        ValueError, match="store was opened in read-only mode and does not support writing"
    ):
        await latency_ro.set("key", buf)

    # Create a writable wrapper from the read-only one
    writer = latency_ro.with_read_only(read_only=False)
    assert isinstance(writer, LatencyStore)
    assert not writer.read_only
    # Latency configuration is preserved
    assert writer.get_latency == latency_ro.get_latency
    assert writer.set_latency == latency_ro.set_latency

    # Writes via the writable wrapper succeed
    await writer.set("key", buf)
    out = await writer.get("key", prototype=default_buffer_prototype())
    assert out is not None
    assert out.to_bytes() == buf.to_bytes()

    # Creating a read-only copy from the writable wrapper works and is enforced
    reader = writer.with_read_only(read_only=True)
    assert isinstance(reader, LatencyStore)
    assert reader.read_only
    with pytest.raises(
        ValueError, match="store was opened in read-only mode and does not support writing"
    ):
        await reader.set("other", buf)

    # The original read-only wrapper remains read-only
    assert latency_ro.read_only


@pytest.mark.parametrize(
    ("get_latency", "set_latency"),
    [
        (0.01, 0.02),
        ((0.1, 0.05), (0.2, 0.01)),
    ],
    ids=["scalar", "distribution"],
)
def test_with_store_preserves_latency_config(
    get_latency: float | tuple[float, float], set_latency: float | tuple[float, float]
) -> None:
    """Derived stores (e.g. via `with_read_only`) keep the raw latency config —
    a `(loc, scale)` distribution must not collapse to one sampled float."""
    store = LatencyStore(MemoryStore(), get_latency=get_latency, set_latency=set_latency)
    derived = store.with_read_only(True)
    assert derived._get_latency == store._get_latency
    assert derived._set_latency == store._set_latency


def test_sync_methods_inject_latency(monkeypatch: pytest.MonkeyPatch) -> None:
    """`get_sync`/`set_sync` sleep the configured latency on the calling thread
    before delegating to the wrapped store."""
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)

    store = LatencyStore(MemoryStore(), get_latency=0.123, set_latency=0.456)
    buf = default_buffer_prototype().buffer.from_bytes(b"abcd")
    store.set_sync("key", buf)
    assert sleeps == [pytest.approx(0.456)]
    out = store.get_sync("key", prototype=default_buffer_prototype())
    assert out is not None
    assert out.to_bytes() == b"abcd"
    assert sleeps == [pytest.approx(0.456), pytest.approx(0.123)]


async def test_get_ranges_pays_latency_per_fetch() -> None:
    """`get_ranges` routes through the coalescing default built on `self.get`,
    so each merged fetch pays the configured latency instead of bypassing it
    via WrapperStore delegation. Two ranges further apart than `max_gap_bytes`
    cannot coalesce -> exactly two `get` calls."""
    proto = default_buffer_prototype()
    inner = MemoryStore()
    await inner.set("blob", proto.buffer.from_bytes(bytes(4 << 20)))
    store = LatencyStore(inner, get_latency=0.0)

    requests = [RangeByteRequest(0, 10), RangeByteRequest(2 << 20, (2 << 20) + 10)]
    results: list[tuple[int, object]] = []
    with patch.object(store, "get", wraps=store.get) as get_spy:
        async for group in store.get_ranges("blob", requests, prototype=proto):
            results.extend(group)
    assert get_spy.await_count == 2
    assert sorted(idx for idx, _ in results) == [0, 1]
    for _, buf in results:
        assert buf is not None
        assert len(buf) == 10  # type: ignore[arg-type]


async def test_get_partial_values_routes_through_get() -> None:
    """`get_partial_values` issues one `self.get` per key-range so each fetch
    pays the configured latency instead of bypassing it via WrapperStore
    delegation."""
    proto = default_buffer_prototype()
    inner = MemoryStore()
    await inner.set("blob", proto.buffer.from_bytes(b"0123456789"))
    store = LatencyStore(inner, get_latency=0.0)

    with patch.object(store, "get", wraps=store.get) as get_spy:
        results = await store.get_partial_values(
            proto, [("blob", RangeByteRequest(0, 4)), ("blob", None)]
        )
    assert get_spy.await_count == 2
    assert results[0] is not None
    assert results[0].to_bytes() == b"0123"
    assert results[1] is not None
    assert results[1].to_bytes() == b"0123456789"


def test_latency_store_engages_fused_sync_path() -> None:
    """A LatencyStore wrapping a sync-capable store must take the fused sync
    fast path: reads go through the inner store's `get_sync`, not the async
    fallback."""
    inner = MemoryStore()
    store = LatencyStore(inner, get_latency=0.0, set_latency=0.0)
    with zarr_config.set({"codec_pipeline.path": "zarr.core.codec_pipeline.FusedCodecPipeline"}):
        arr = zarr.create_array(
            store=store,
            shape=(8,),
            chunks=(4,),
            dtype="uint8",
            compressors=None,
            fill_value=0,
        )
        assert isinstance(arr._async_array.codec_pipeline, FusedCodecPipeline)
        data = np.arange(8, dtype="uint8")
        arr[:] = data
        with patch.object(inner, "get_sync", wraps=inner.get_sync) as get_sync_spy:
            np.testing.assert_array_equal(arr[:], data)
        assert get_sync_spy.call_count == 2  # one per chunk
