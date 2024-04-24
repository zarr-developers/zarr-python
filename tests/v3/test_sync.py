from collections.abc import AsyncGenerator
import asyncio
import time
from unittest.mock import patch

from zarr.sync import sync, _get_loop, _get_lock, SyncError, SyncMixin
from zarr.config import SyncConfiguration

import pytest


@pytest.fixture(params=[True, False])
def sync_loop(request) -> asyncio.AbstractEventLoop | None:
    if request.param is True:
        return _get_loop()

    if request.param is False:
        return None


def test_get_loop() -> None:
    # test that calling _get_loop() twice returns the same loop
    loop = _get_loop()
    loop2 = _get_loop()
    assert loop is loop2


def test_get_lock() -> None:
    # test that calling _get_lock() twice returns the same lock
    lock = _get_lock()
    lock2 = _get_lock()
    assert lock is lock2


def test_sync(sync_loop: asyncio.AbstractEventLoop | None) -> None:
    async def foo() -> str:
        return "foo"

    assert sync(foo(), loop=sync_loop) == "foo"


def test_sync_raises(sync_loop: asyncio.AbstractEventLoop | None) -> None:
    async def foo() -> str:
        raise ValueError("foo")

    with pytest.raises(ValueError):
        sync(foo(), loop=sync_loop)


def test_sync_timeout() -> None:
    duration = 0.002

    async def foo() -> None:
        time.sleep(duration)

    with pytest.raises(asyncio.TimeoutError):
        sync(foo(), timeout=duration / 2)


def test_sync_raises_if_no_coroutine(sync_loop: asyncio.AbstractEventLoop | None) -> None:
    def foo() -> str:
        return "foo"

    with pytest.raises(TypeError):
        sync(foo(), loop=sync_loop)


@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited")
def test_sync_raises_if_loop_is_closed() -> None:
    loop = _get_loop()

    async def foo() -> str:
        return "foo"

    with patch.object(loop, "is_closed", return_value=True):
        with pytest.raises(RuntimeError):
            sync(foo(), loop=loop)


@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited")
def test_sync_raises_if_calling_sync_from_within_a_running_loop(
    sync_loop: asyncio.AbstractEventLoop | None,
) -> None:
    async def foo() -> str:
        return "foo"

    async def bar() -> str:
        return sync(foo())

    with pytest.raises(SyncError):
        sync(bar(), loop=sync_loop)


@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited")
def test_sync_raises_if_loop_is_invalid_type() -> None:
    async def foo() -> str:
        return "foo"

    with pytest.raises(TypeError):
        sync(foo(), loop=1)


def test_sync_mixin(sync_loop) -> None:
    class AsyncFoo:
        def __init__(self) -> None:
            pass

        async def foo(self) -> str:
            return "foo"

        async def bar(self) -> AsyncGenerator:
            for i in range(10):
                yield i

    class SyncFoo(SyncMixin):
        def __init__(self, async_foo: AsyncFoo) -> None:
            self._async_foo = async_foo
            self._sync_configuration = SyncConfiguration(asyncio_loop=sync_loop)

        def foo(self) -> str:
            return self._sync(self._async_foo.foo())

        def bar(self) -> list[int]:
            return self._sync_iter(self._async_foo.bar())

    async_foo = AsyncFoo()
    foo = SyncFoo(async_foo)
    assert foo.foo() == "foo"
    assert foo.bar() == list(range(10))
