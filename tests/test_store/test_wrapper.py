from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

import pytest

from zarr.abc.store import ByteRequest, Store, _store_supports_sync_io
from zarr.core.buffer import Buffer
from zarr.core.buffer.cpu import Buffer as CPUBuffer
from zarr.core.buffer.cpu import buffer_prototype
from zarr.storage import LocalStore, MemoryStore, WrapperStore, ZipStore
from zarr.testing.store import LatencyStore, StoreTests

if TYPE_CHECKING:
    from pathlib import Path

    from zarr.core.buffer.core import BufferPrototype


class StoreKwargs(TypedDict):
    store: LocalStore


class OpenKwargs(TypedDict):
    store_cls: type[LocalStore]
    root: str


# TODO: fix this warning
@pytest.mark.filterwarnings(
    "ignore:coroutine 'ClientCreatorContext.__aexit__' was never awaited:RuntimeWarning"
)
class TestWrapperStore(StoreTests[WrapperStore[Any], Buffer]):
    store_cls = WrapperStore
    buffer_cls = CPUBuffer

    async def get(self, store: WrapperStore[LocalStore], key: str) -> Buffer:
        return self.buffer_cls.from_bytes((store._store.root / key).read_bytes())

    async def set(self, store: WrapperStore[LocalStore], key: str, value: Buffer) -> None:
        parent = (store._store.root / key).parent
        if not parent.exists():
            parent.mkdir(parents=True)
        (store._store.root / key).write_bytes(value.to_bytes())

    @pytest.fixture
    def store_kwargs(self, tmp_path: Path) -> StoreKwargs:
        return {"store": LocalStore(str(tmp_path))}

    @pytest.fixture
    def open_kwargs(self, tmp_path: Path) -> OpenKwargs:
        return {"store_cls": LocalStore, "root": str(tmp_path)}

    def test_store_supports_writes(self, store: WrapperStore[LocalStore]) -> None:
        assert store.supports_writes

    def test_store_supports_listing(self, store: WrapperStore[LocalStore]) -> None:
        assert store.supports_listing

    def test_store_repr(self, store: WrapperStore[LocalStore]) -> None:
        assert f"{store!r}" == f"WrapperStore(LocalStore, 'file://{store._store.root.as_posix()}')"

    def test_store_str(self, store: WrapperStore[LocalStore]) -> None:
        assert str(store) == f"wrapping-file://{store._store.root.as_posix()}"

    def test_check_writeable(self, store: WrapperStore[LocalStore]) -> None:
        """
        Test _check_writeable() runs without errors.
        """
        store._check_writable()

    def test_close(self, store: WrapperStore[LocalStore]) -> None:
        "Test store can be closed"
        store.close()
        assert not store._is_open

    def test_is_open_setter_raises(self, store: WrapperStore[LocalStore]) -> None:
        """
        Test that a user cannot change `_is_open` without opening the underlying store.
        """
        with pytest.raises(
            NotImplementedError, match="WrapperStore must be opened via the `_open` method"
        ):
            store._is_open = True


# TODO: work out where warning is coming from and fix
@pytest.mark.filterwarnings(
    "ignore:coroutine 'ClientCreatorContext.__aexit__' was never awaited:RuntimeWarning"
)
@pytest.mark.parametrize("store", ["local", "memory", "zip"], indirect=True)
async def test_wrapped_set(store: Store, capsys: pytest.CaptureFixture[str]) -> None:
    # define a class that prints when it sets
    class NoisySetter(WrapperStore[Store]):
        async def set(self, key: str, value: Buffer) -> None:
            print(f"setting {key}")
            await super().set(key, value)

    key = "foo"
    value = CPUBuffer.from_bytes(b"bar")
    store_wrapped = NoisySetter(store)
    await store_wrapped.set(key, value)
    captured = capsys.readouterr()
    assert f"setting {key}" in captured.out
    assert await store_wrapped.get(key, buffer_prototype) == value


@pytest.mark.filterwarnings("ignore:Unclosed client session:ResourceWarning")
@pytest.mark.parametrize("store", ["local", "memory", "zip"], indirect=True)
async def test_wrapped_get(store: Store, capsys: pytest.CaptureFixture[str]) -> None:
    # define a class that prints when it sets
    class NoisyGetter(WrapperStore[Any]):
        async def get(
            self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None = None
        ) -> None:
            print(f"getting {key}")
            await super().get(key, prototype=prototype, byte_range=byte_range)

    key = "foo"
    value = CPUBuffer.from_bytes(b"bar")
    store_wrapped = NoisyGetter(store)
    await store_wrapped.set(key, value)
    await store_wrapped.get(key, buffer_prototype)
    captured = capsys.readouterr()
    assert f"getting {key}" in captured.out


@pytest.mark.parametrize(
    ("store_factory", "expected"),
    [
        (lambda tmp: MemoryStore(), True),
        (lambda tmp: LocalStore(str(tmp)), True),
        (lambda tmp: WrapperStore(MemoryStore()), True),
        (lambda tmp: LatencyStore(MemoryStore()), True),
        (lambda tmp: ZipStore(tmp / "store.zip", mode="w"), False),
        (lambda tmp: WrapperStore(ZipStore(tmp / "store.zip", mode="w")), False),
    ],
    ids=[
        "memory",
        "local",
        "wrapper-of-memory",
        "latency-wrapper-of-memory",
        "zip",
        "wrapper-of-zip",
    ],
)
def test_supports_sync_io(store_factory: Any, expected: bool, tmp_path: Path | Any) -> None:
    """`_store_supports_sync_io` is True only for stores implementing the full
    sync surface (get_sync + set_sync + delete_sync); wrappers forward the
    wrapped store's capability via `_supports_sync_io`."""
    assert _store_supports_sync_io(store_factory(tmp_path)) is expected


def test_wrapper_get_sync_without_inner_sync_raises(tmp_path: Any) -> None:
    store = WrapperStore(ZipStore(tmp_path / "store.zip", mode="w"))
    with pytest.raises(TypeError, match="does not support synchronous get"):
        store.get_sync("key")


def test_wrapper_set_sync_without_inner_sync_raises(tmp_path: Any) -> None:
    store = WrapperStore(ZipStore(tmp_path / "store.zip", mode="w"))
    with pytest.raises(TypeError, match="does not support synchronous set"):
        store.set_sync("key", CPUBuffer.from_bytes(b"data"))


def test_wrapper_delete_sync_without_inner_sync_raises(tmp_path: Any) -> None:
    store = WrapperStore(ZipStore(tmp_path / "store.zip", mode="w"))
    with pytest.raises(TypeError, match="does not support synchronous delete"):
        store.delete_sync("key")
