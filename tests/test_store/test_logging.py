from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

import zarr
from zarr.core.buffer import Buffer, cpu, default_buffer_prototype
from zarr.storage import LocalStore, LoggingStore
from zarr.testing.store import StoreTests

if TYPE_CHECKING:
    from _pytest.compat import LEGACY_PATH

    from zarr.abc.store import Store


class TestLoggingStore(StoreTests[LoggingStore, cpu.Buffer]):
    store_cls = LoggingStore
    buffer_cls = cpu.Buffer

    async def get(self, store: LoggingStore, key: str) -> Buffer:
        return self.buffer_cls.from_bytes((store._store.root / key).read_bytes())

    async def set(self, store: LoggingStore, key: str, value: Buffer) -> None:
        parent = (store._store.root / key).parent
        if not parent.exists():
            parent.mkdir(parents=True)
        (store._store.root / key).write_bytes(value.to_bytes())

    @pytest.fixture
    def store_kwargs(self, tmpdir: LEGACY_PATH) -> dict[str, str]:
        return {"store": LocalStore(str(tmpdir)), "log_level": "DEBUG"}

    @pytest.fixture
    def open_kwargs(self, tmpdir) -> dict[str, str]:
        return {"store_cls": LocalStore, "root": str(tmpdir), "log_level": "DEBUG"}

    @pytest.fixture
    def store(self, store_kwargs: str | dict[str, Buffer] | None) -> LoggingStore:
        return self.store_cls(**store_kwargs)

    def test_store_supports_writes(self, store: LoggingStore) -> None:
        assert store.supports_writes

    def test_store_supports_partial_writes(self, store: LoggingStore) -> None:
        assert store.supports_partial_writes

    def test_store_supports_listing(self, store: LoggingStore) -> None:
        assert store.supports_listing

    def test_store_repr(self, store: LoggingStore) -> None:
        assert f"{store!r}" == f"LoggingStore(LocalStore, 'file://{store._store.root.as_posix()}')"

    def test_store_str(self, store: LoggingStore) -> None:
        assert str(store) == f"logging-file://{store._store.root.as_posix()}"

    async def test_default_handler(self, local_store, capsys) -> None:
        # Store and then remove existing handlers to enter default handler code path
        handlers = logging.getLogger().handlers[:]
        for h in handlers:
            logging.getLogger().removeHandler(h)
        # Test logs are sent to stdout
        wrapped = LoggingStore(store=local_store)
        buffer = default_buffer_prototype().buffer
        res = await wrapped.set("foo/bar/c/0", buffer.from_bytes(b"\x01\x02\x03\x04"))
        assert res is None
        captured = capsys.readouterr()
        assert len(captured) == 2
        assert "Calling LocalStore.set" in captured.out
        assert "Finished LocalStore.set" in captured.out
        # Restore handlers
        for h in handlers:
            logging.getLogger().addHandler(h)

    def test_is_open_setter_raises(self, store: LoggingStore) -> None:
        "Test that a user cannot change `_is_open` without opening the underlying store."
        with pytest.raises(
            NotImplementedError, match="LoggingStore must be opened via the `_open` method"
        ):
            store._is_open = True


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
    assert wrapped.counter["list"] == 0
    assert wrapped.counter["list_dir"] == 0
    assert wrapped.counter["list_prefix"] == 0
    if store.supports_deletes:
        assert wrapped.counter["get"] == 0  # 1 if overwrite=False
        assert wrapped.counter["delete_dir"] == 1
    else:
        assert wrapped.counter["get"] == 1
        assert wrapped.counter["delete_dir"] == 0
