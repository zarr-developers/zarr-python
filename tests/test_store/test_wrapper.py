from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from zarr.core.buffer.cpu import Buffer, buffer_prototype
from zarr.storage import LocalStore, WrapperStore
from zarr.testing.store import StoreTests

if TYPE_CHECKING:
    from _pytest.compat import LEGACY_PATH

    from zarr.abc.store import Store
    from zarr.core.buffer.core import BufferPrototype


# TODO: fix this warning
@pytest.mark.filterwarnings(
    "ignore:coroutine 'ClientCreatorContext.__aexit__' was never awaited:RuntimeWarning"
)
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
    def store_kwargs(self, tmpdir: LEGACY_PATH) -> dict[str, str]:
        return {"store": LocalStore(str(tmpdir))}

    @pytest.fixture
    def open_kwargs(self, tmpdir) -> dict[str, str]:
        return {"store_cls": LocalStore, "root": str(tmpdir)}

    def test_store_supports_writes(self, store: WrapperStore) -> None:
        assert store.supports_writes

    def test_store_supports_partial_writes(self, store: WrapperStore) -> None:
        assert store.supports_partial_writes

    def test_store_supports_listing(self, store: WrapperStore) -> None:
        assert store.supports_listing

    def test_store_repr(self, store: WrapperStore) -> None:
        assert f"{store!r}" == f"WrapperStore(LocalStore, 'file://{store._store.root.as_posix()}')"

    def test_store_str(self, store: WrapperStore) -> None:
        assert str(store) == f"wrapping-file://{store._store.root.as_posix()}"

    def test_check_writeable(self, store: WrapperStore) -> None:
        """
        Test _check_writeable() runs without errors.
        """
        store._check_writable()

    def test_close(self, store: WrapperStore) -> None:
        "Test store can be closed"
        store.close()
        assert not store._is_open

    def test_is_open_setter_raises(self, store: WrapperStore) -> None:
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


@pytest.mark.filterwarnings("ignore:Unclosed client session:ResourceWarning")
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
