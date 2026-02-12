from __future__ import annotations

import logging
from typing import TYPE_CHECKING, TypedDict

import pytest

import zarr
from zarr.buffer import cpu
from zarr.storage import LocalStore, LoggingStore
from zarr.testing.store import StoreTests

if TYPE_CHECKING:
    from pathlib import Path

    from zarr.abc.store import Store
    from zarr.core.buffer.core import Buffer


class StoreKwargs(TypedDict):
    store: LocalStore
    log_level: str


class TestLoggingStore(StoreTests[LoggingStore[LocalStore], cpu.Buffer]):
    # store_cls is needed to do an isinstance check, so can't be a subscripted generic
    store_cls = LoggingStore  # type: ignore[assignment]
    buffer_cls = cpu.Buffer

    async def get(self, store: LoggingStore[LocalStore], key: str) -> Buffer:
        return self.buffer_cls.from_bytes((store._store.root / key).read_bytes())

    async def set(self, store: LoggingStore[LocalStore], key: str, value: Buffer) -> None:
        parent = (store._store.root / key).parent
        if not parent.exists():
            parent.mkdir(parents=True)
        (store._store.root / key).write_bytes(value.to_bytes())

    @pytest.fixture
    def store_kwargs(self, tmp_path: Path) -> StoreKwargs:
        return {"store": LocalStore(str(tmp_path)), "log_level": "DEBUG"}

    @pytest.fixture
    def open_kwargs(self, tmp_path: Path) -> dict[str, type[LocalStore] | str]:
        return {"store_cls": LocalStore, "root": str(tmp_path), "log_level": "DEBUG"}

    @pytest.fixture
    def store(self, store_kwargs: StoreKwargs) -> LoggingStore[LocalStore]:
        return self.store_cls(**store_kwargs)

    def test_store_supports_writes(self, store: LoggingStore[LocalStore]) -> None:
        assert store.supports_writes

    def test_store_supports_listing(self, store: LoggingStore[LocalStore]) -> None:
        assert store.supports_listing

    def test_store_repr(self, store: LoggingStore[LocalStore]) -> None:
        assert f"{store!r}" == f"LoggingStore(LocalStore, 'file://{store._store.root.as_posix()}')"

    def test_store_str(self, store: LoggingStore[LocalStore]) -> None:
        assert str(store) == f"logging-file://{store._store.root.as_posix()}"

    async def test_default_handler(
        self, local_store: LocalStore, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Store and then remove existing handlers to enter default handler code path
        handlers = logging.getLogger().handlers[:]
        for h in handlers:
            logging.getLogger().removeHandler(h)
        # Test logs are sent to stdout
        wrapped = LoggingStore(store=local_store)
        buffer = cpu.Buffer
        res = await wrapped.set("foo/bar/c/0", buffer.from_bytes(b"\x01\x02\x03\x04"))  # type: ignore[func-returns-value]
        assert res is None
        captured = capsys.readouterr()
        assert len(captured) == 2
        assert "Calling LocalStore.set" in captured.out
        assert "Finished LocalStore.set" in captured.out
        # Restore handlers
        for h in handlers:
            logging.getLogger().addHandler(h)

    def test_is_open_setter_raises(self, store: LoggingStore[LocalStore]) -> None:
        "Test that a user cannot change `_is_open` without opening the underlying store."
        with pytest.raises(
            NotImplementedError, match="LoggingStore must be opened via the `_open` method"
        ):
            store._is_open = True

    async def test_with_read_only_round_trip(self, local_store: LocalStore) -> None:
        """
        Ensure that LoggingStore.with_read_only returns another LoggingStore with
        the requested read_only state, preserves logging configuration, and does
        not change the original store.
        """
        # Start from a read-only underlying store
        ro_store = local_store.with_read_only(read_only=True)
        wrapped_ro = LoggingStore(store=ro_store, log_level="INFO")
        assert wrapped_ro.read_only

        buf = default_buffer_prototype().buffer.from_bytes(b"0123")

        # Cannot write through the read-only wrapper
        with pytest.raises(
            ValueError, match="store was opened in read-only mode and does not support writing"
        ):
            await wrapped_ro.set("foo", buf)

        # Create a writable wrapper
        writer = wrapped_ro.with_read_only(read_only=False)
        assert isinstance(writer, LoggingStore)
        assert not writer.read_only
        # logging configuration is preserved
        assert writer.log_level == wrapped_ro.log_level
        assert writer.log_handler == wrapped_ro.log_handler

        # Writes via the writable wrapper succeed
        await writer.set("foo", buf)
        out = await writer.get("foo", prototype=default_buffer_prototype())
        assert out is not None
        assert out.to_bytes() == buf.to_bytes()

        # The original wrapper remains read-only
        assert wrapped_ro.read_only
        with pytest.raises(
            ValueError, match="store was opened in read-only mode and does not support writing"
        ):
            await wrapped_ro.set("bar", buf)


@pytest.mark.parametrize("store", ["local", "memory", "zip"], indirect=["store"])
async def test_logging_store(store: Store, caplog: pytest.LogCaptureFixture) -> None:
    wrapped = LoggingStore(store=store, log_level="DEBUG")
    buffer = cpu.Buffer

    caplog.clear()
    res = await wrapped.set("foo/bar/c/0", buffer.from_bytes(b"\x01\x02\x03\x04"))  # type: ignore[func-returns-value]
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
