import asyncio
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

import zarr
from zarr.abc.store import ByteRequest
from zarr.core.buffer import Buffer, BufferPrototype
from zarr.core.sync import (
    SyncError,
    SyncMixin,
    _get_executor,
    _get_lock,
    _get_loop,
    cleanup_resources,
    loop,
    sync,
)
from zarr.storage import MemoryStore, WrapperStore


@pytest.fixture(params=[True, False])
def sync_loop(request: pytest.FixtureRequest) -> asyncio.AbstractEventLoop | None:
    if request.param is True:
        return _get_loop()
    else:
        return None


@pytest.fixture
def clean_state():
    # use this fixture to make sure no existing threads/loops exist in zarr.core.sync
    cleanup_resources()
    yield
    cleanup_resources()


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
    foo = AsyncMock(return_value="foo")
    assert sync(foo(), loop=sync_loop) == "foo"
    foo.assert_awaited_once()


def test_sync_raises(sync_loop: asyncio.AbstractEventLoop | None) -> None:
    foo = AsyncMock(side_effect=ValueError("foo-bar"))
    with pytest.raises(ValueError, match="foo-bar"):
        sync(foo(), loop=sync_loop)
    foo.assert_awaited_once()


def test_sync_timeout() -> None:
    duration = 0.02

    async def foo() -> None:
        await asyncio.sleep(duration)

    with pytest.raises(asyncio.TimeoutError):
        sync(foo(), timeout=duration / 10)


def test_sync_raises_if_no_coroutine(sync_loop: asyncio.AbstractEventLoop | None) -> None:
    def foo() -> str:
        return "foo"

    with pytest.raises(TypeError):
        sync(foo(), loop=sync_loop)  # type: ignore[arg-type]


@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited")
def test_sync_raises_if_loop_is_closed() -> None:
    loop = _get_loop()

    foo = AsyncMock(return_value="foo")
    with patch.object(loop, "is_closed", return_value=True):
        with pytest.raises(RuntimeError):
            sync(foo(), loop=loop)
    foo.assert_not_awaited()


@pytest.mark.filterwarnings("ignore:Unclosed client session:ResourceWarning")
@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited")
def test_sync_raises_if_calling_sync_from_within_a_running_loop(
    sync_loop: asyncio.AbstractEventLoop | None,
) -> None:
    def foo() -> str:
        # technically, this should be an async function but doing that
        # yields a warning because it is never awaited by the inner function
        return "foo"

    async def bar() -> str:
        return sync(foo(), loop=sync_loop)  # type: ignore[arg-type]

    with pytest.raises(SyncError):
        sync(bar(), loop=sync_loop)


@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited")
def test_sync_raises_if_loop_is_invalid_type() -> None:
    foo = AsyncMock(return_value="foo")
    with pytest.raises(TypeError):
        sync(foo(), loop=1)  # type: ignore[arg-type]
    foo.assert_not_awaited()


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

        def foo(self) -> str:
            return self._sync(self._async_foo.foo())

        def bar(self) -> list[int]:
            return self._sync_iter(self._async_foo.bar())

    async_foo = AsyncFoo()
    foo = SyncFoo(async_foo)
    assert foo.foo() == "foo"
    assert foo.bar() == list(range(10))


@pytest.mark.parametrize("workers", [None, 1, 2])
def test_threadpool_executor(clean_state, workers: int | None) -> None:
    with zarr.config.set({"threading.max_workers": workers}):
        _ = zarr.zeros(shape=(1,))  # trigger executor creation
        assert loop != [None]  # confirm loop was created
        if workers is None:
            # confirm no executor was created if no workers were specified
            # (this is the default behavior)
            assert loop[0]._default_executor is None
        else:
            # confirm executor was created and attached to loop as the default executor
            # note: python doesn't have a direct way to get the default executor so we
            # use the private attribute
            assert _get_executor() is loop[0]._default_executor
            assert _get_executor()._max_workers == workers


def test_cleanup_resources_idempotent() -> None:
    _get_executor()  # trigger resource creation (iothread, loop, thread-pool)
    cleanup_resources()
    cleanup_resources()


class LoopBoundStore(WrapperStore[MemoryStore]):
    """A store whose I/O only works on the event loop that first drove it.

    Mimics fsspec/aiohttp-backed stores, whose sessions and connectors are
    lazily bound to whichever event loop first performs I/O.
    """

    _bound_loop: asyncio.AbstractEventLoop | None = None

    def _check_loop(self) -> None:
        running = asyncio.get_running_loop()
        if self._bound_loop is None:
            self._bound_loop = running
        elif running is not self._bound_loop:
            raise RuntimeError("store I/O driven from a different event loop than it is bound to")

    async def get(
        self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        self._check_loop()
        return await super().get(key, prototype, byte_range)

    async def set(self, key: str, value: Buffer) -> None:
        self._check_loop()
        await super().set(key, value)


@pytest.mark.skipif(
    not hasattr(zarr.Array, "getitem_async"),
    reason="Array.*_async methods do not exist on this version",
)
@pytest.mark.xfail(
    reason=(
        "Design-independent limitation: a store with event-loop affinity binds to "
        "zarr's background loop during sync create/open, so awaiting Array.*_async "
        "from a different user-owned loop trips the affinity check. Orthogonal to "
        "keeping vs. removing AsyncArray; tracked separately."
    ),
    strict=True,
)
def test_array_async_methods_with_loop_bound_store() -> None:
    # The supported pattern is: open an array synchronously (its I/O runs on
    # zarr's internal background loop), then await the array's *_async methods
    # from user-owned async code (a different loop). Stores with loop affinity
    # must keep working across that boundary.
    store = LoopBoundStore(MemoryStore())
    z = zarr.create_array(store, shape=(8,), chunks=(4,), dtype="i4")
    z[:] = np.arange(8, dtype="i4")

    async def read() -> object:
        return await z.getitem_async(slice(2, 6))

    # Drive a user-owned loop explicitly and close it in `finally`, so the
    # expected RuntimeError (xfail) doesn't leave an unclosed loop behind for
    # the GC to report as an unraisable exception.
    user_loop = asyncio.new_event_loop()
    try:
        result = user_loop.run_until_complete(read())
    finally:
        user_loop.close()
    np.testing.assert_array_equal(np.arange(2, 6, dtype="i4"), result)
