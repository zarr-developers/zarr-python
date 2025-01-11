from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from zarr.core.buffer.cpu import Buffer, buffer_prototype
from zarr.storage import LocalStore, WrapperStore
from zarr.testing.store import StoreTests

if TYPE_CHECKING:
    from typing import Any

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
    def store(self, store_kwargs: str | dict[str, Buffer] | None) -> WrapperStore:
        return self.store_cls(**store_kwargs)

    def test_store_supports_writes(self, store: WrapperStore) -> None:
        assert store.supports_writes

    def test_store_supports_partial_writes(self, store: WrapperStore) -> None:
        assert store.supports_partial_writes

    def test_store_supports_listing(self, store: WrapperStore) -> None:
        assert store.supports_listing

    @pytest.mark.parametrize("read_only", [True, False])
    async def test_store_open_read_only(
        self, store_kwargs: dict[str, Any], read_only: bool, tmpdir
    ) -> None:
        store_kwargs = {
            **store_kwargs,
            "read_only": read_only,
            "root": str(tmpdir),
            "store_cls": LocalStore,
        }
        store_kwargs.pop("store")
        store = await self.store_cls.open(**store_kwargs)
        assert store._is_open
        assert store.read_only == read_only

    async def test_read_only_store_raises(self, store_kwargs: dict[str, Any], tmpdir) -> None:
        store_kwargs = {
            **store_kwargs,
            "read_only": True,
            "root": str(tmpdir),
            "store_cls": LocalStore,
        }
        store_kwargs.pop("store")
        store = await self.store_cls.open(**store_kwargs)
        assert store.read_only

        # set
        with pytest.raises(ValueError):
            await store.set("foo", self.buffer_cls.from_bytes(b"bar"))

        # delete
        with pytest.raises(ValueError):
            await store.delete("foo")


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
