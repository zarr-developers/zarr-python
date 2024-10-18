from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import zarr
import zarr.storage
from zarr.core.buffer import default_buffer_prototype
from zarr.storage.logging import LoggingStore

if TYPE_CHECKING:
    from zarr.abc.store import Store


@pytest.mark.parametrize("store", ["local", "memory", "zip"], indirect=["store"])
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


@pytest.mark.parametrize("store", ["local", "memory", "zip"], indirect=["store"])
async def test_logging_store_counter(store: Store) -> None:
    wrapped = LoggingStore(store=store, log_level="DEBUG")

    arr = zarr.create(shape=(10,), store=wrapped, overwrite=True)
    arr[:] = 1

    assert wrapped.counter["set"] == 2
    assert wrapped.counter["get"] == 0  # 1 if overwrite=False
    assert wrapped.counter["list"] == 0
    assert wrapped.counter["list_dir"] == 0
    assert wrapped.counter["list_prefix"] == 0


async def test_with_mode():
    wrapped = LoggingStore(store=zarr.storage.MemoryStore(mode="w"), log_level="INFO")
    new = wrapped.with_mode(mode="r")
    assert new.mode.str == "r"
    assert new.log_level == "INFO"
