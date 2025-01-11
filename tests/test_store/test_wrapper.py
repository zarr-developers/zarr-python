from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from zarr.core.buffer.cpu import Buffer, buffer_prototype
from zarr.storage import LocalStore, WrapperStore
from zarr.testing.store import StoreTests

if TYPE_CHECKING:
    from zarr.abc.store import Store
    from zarr.core.buffer.core import BufferPrototype


class TestWrapperStore(StoreTests[WrapperStore, Buffer]):
    store_cls = WrapperStore
    buffer_cls = Buffer

    async def get(self, store: WrapperStore, key: str) -> Buffer:
        return self.buffer_cls.from_bytes((store._store.root / key).read_bytes())

    async def set(self, store: WrapperStore, key: str, value: Buffer) -> None:
        parent = (store._store.root / key).parent
        if not parent.exists():
            parent.mkdir(parents=True)
        (store._store.root / key).write_bytes(value.to_bytes())

    @pytest.fixture
    def store_kwargs(self, local_store) -> dict[str, str]:
        return {"store": local_store}

    @pytest.fixture
    def open_kwargs(self, tmpdir) -> dict[str, str]:
        return {"store_cls": LocalStore, "root": str(tmpdir)}

    @pytest.fixture
    def store(self, store_kwargs: dict[str, str]) -> WrapperStore:
        return self.store_cls(**store_kwargs)

    def test_store_supports_writes(self, store: WrapperStore) -> None:
        assert store.supports_writes

    def test_store_supports_partial_writes(self, store: WrapperStore) -> None:
        assert store.supports_partial_writes

    def test_store_supports_listing(self, store: WrapperStore) -> None:
        assert store.supports_listing

    def test_store_repr(self, store: WrapperStore) -> None:
        assert str(store) == f"wrapping-file://{store._store.root.as_posix()}"


@pytest.mark.parametrize("store", ["local", "memory", "zip"], indirect=True)
async def test_wrapped_set(store: Store, capsys: pytest.CaptureFixture[str]) -> None:
    # define a class that prints when it sets
    class NoisySetter(WrapperStore):
        async def set(self, key: str, value: Buffer) -> None:
            print(f"setting {key}")
            await super().set(key, value)

    key = "foo"
    value = Buffer.from_bytes(b"bar")
    store_wrapped = NoisySetter(store)
    await store_wrapped.set(key, value)
    captured = capsys.readouterr()
    assert f"setting {key}" in captured.out
    assert await store_wrapped.get(key, buffer_prototype) == value


@pytest.mark.parametrize("store", ["local", "memory", "zip"], indirect=True)
async def test_wrapped_get(store: Store, capsys: pytest.CaptureFixture[str]) -> None:
    # define a class that prints when it sets
    class NoisyGetter(WrapperStore):
        def get(self, key: str, prototype: BufferPrototype) -> None:
            print(f"getting {key}")
            return super().get(key, prototype=prototype)

    key = "foo"
    value = Buffer.from_bytes(b"bar")
    store_wrapped = NoisyGetter(store)
    await store_wrapped.set(key, value)
    assert await store_wrapped.get(key, buffer_prototype) == value
    captured = capsys.readouterr()
    assert f"getting {key}" in captured.out
