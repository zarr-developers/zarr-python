from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from zarr.core.buffer import default_buffer_prototype
from zarr.store.logging import LoggingStore

if TYPE_CHECKING:
    from zarr.abc.store import Store


@pytest.mark.parametrize("store", ("local", "memory", "zip"), indirect=["store"])
async def test_logging_store(store: Store, caplog) -> None:
    wrapped = LoggingStore(store=store, log_level="DEBUG")
    buffer = default_buffer_prototype().buffer

    caplog.clear()
    res = await wrapped.set("foo/bar/c/0", buffer.from_bytes(b"\x01\x02\x03\x04"))
    assert res is None
    assert len(caplog.record_tuples) == 2
    for tup in caplog.record_tuples:
        assert str(store) in tup[0]
    assert f"Calling {type(store).__name__}.set" in caplog.record_tuples[0][2]
    assert f"Finished {type(store).__name__}.set" in caplog.record_tuples[1][2]

    caplog.clear()
    keys = [k async for k in wrapped.list()]
    assert keys == ["foo/bar/c/0"]
    assert len(caplog.record_tuples) == 2
    for tup in caplog.record_tuples:
        assert str(store) in tup[0]
    assert f"Calling {type(store).__name__}.list" in caplog.record_tuples[0][2]
    assert f"Finished {type(store).__name__}.list" in caplog.record_tuples[1][2]
