from __future__ import annotations

import pytest

from zarr.core.buffer import default_buffer_prototype
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
