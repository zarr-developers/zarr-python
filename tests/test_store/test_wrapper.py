from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from zarr.core.buffer.cpu import Buffer, buffer_prototype
from zarr.storage.wrapper import WrapperStore

if TYPE_CHECKING:
    from zarr.abc.store import Store
    from zarr.core.buffer.core import BufferPrototype


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
