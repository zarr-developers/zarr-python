import asyncio
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, patch

import pytest

import zarr
from zarr.core.sync import (
    SyncError,
    SyncMixin,
    _get_executor,
    _get_lock,
    _get_loop,
    cleanup_resources,
    loop,
    run,
    sync,
)


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


# --- public `zarr.run` API ---------------------------------------------------
# `zarr.run` is the supported public bridge for running async zarr operations
# from synchronous code. It is a thin wrapper over the internal `sync`; these
# tests pin the public contract independently of the internal function.


def test_run_returns_coroutine_result() -> None:
    """`zarr.run` returns the value the coroutine resolves to."""
    foo = AsyncMock(return_value="foo")
    assert run(foo()) == "foo"
    foo.assert_awaited_once()


def test_run_is_public() -> None:
    """`run` is exported at the top level and is the same object as the internal one."""
    assert zarr.run is run
    assert "run" in zarr.__all__


def test_run_propagates_exception() -> None:
    """An exception raised inside the coroutine propagates to the caller."""
    foo = AsyncMock(side_effect=ValueError("foo-bar"))
    with pytest.raises(ValueError, match="foo-bar"):
        run(foo())
    foo.assert_awaited_once()


def test_run_timeout() -> None:
    """`zarr.run` raises `TimeoutError` if the coroutine exceeds `timeout`."""
    duration = 0.02

    async def foo() -> None:
        await asyncio.sleep(duration)

    with pytest.raises(asyncio.TimeoutError):
        run(foo(), timeout=duration / 10)


@pytest.mark.filterwarnings("ignore:coroutine.*was never awaited")
def test_run_raises_runtimeerror_inside_running_loop() -> None:
    """Calling `zarr.run` from within a running loop raises `RuntimeError`.

    This mirrors `asyncio.run`'s behavior for the same misuse, and hides the
    internal `SyncError` from the public surface.
    """

    def inner() -> str:
        # plain (not async) on purpose: an un-awaited inner coroutine would be
        # garbage-collected during a later test and surface as a spurious
        # "coroutine was never awaited" failure. Mirrors the internal-`sync`
        # test above.
        return "inner"

    async def outer() -> str:
        return run(inner())  # type: ignore[arg-type]

    with pytest.raises(RuntimeError):
        run(outer())


def test_run_inside_running_loop_does_not_leak_syncerror() -> None:
    """The internal `SyncError` is not surfaced to callers of `zarr.run`."""

    def inner() -> str:
        return "inner"

    async def outer() -> str:
        return run(inner())  # type: ignore[arg-type]

    with pytest.raises(RuntimeError) as excinfo:
        run(outer())
    # SyncError is preserved as the cause but is not the raised type.
    assert not isinstance(excinfo.value, SyncError)
    assert isinstance(excinfo.value.__cause__, SyncError)


def test_run_composes_with_gather() -> None:
    """The headline downstream pattern: run several coroutines concurrently.

    The `gather` is constructed inside a coroutine so it binds to zarr's loop
    rather than the calling thread (which has no running loop).
    """

    async def double(x: int) -> int:
        await asyncio.sleep(0)
        return x * 2

    async def run_all() -> list[int]:
        return await asyncio.gather(*(double(i) for i in range(5)))

    results = run(run_all())
    assert results == [0, 2, 4, 6, 8]
