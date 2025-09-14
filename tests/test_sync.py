import asyncio
import sys
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, patch

import pytest

import zarr
from zarr.core.sync import (
    SyncError,
    SyncMixin,
    _create_event_loop,
    _get_executor,
    _get_lock,
    _get_loop,
    cleanup_resources,
    loop,
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
            # Note: uvloop doesn't expose _default_executor attribute, so we skip this check for uvloop
            if hasattr(loop[0], "_default_executor"):
                assert loop[0]._default_executor is None
        else:
            # confirm executor was created and attached to loop as the default executor
            # note: python doesn't have a direct way to get the default executor so we
            # use the private attribute (when available)
            assert _get_executor()._max_workers == workers
            if hasattr(loop[0], "_default_executor"):
                assert _get_executor() is loop[0]._default_executor


def test_cleanup_resources_idempotent() -> None:
    _get_executor()  # trigger resource creation (iothread, loop, thread-pool)
    cleanup_resources()
    cleanup_resources()


def test_create_event_loop_default_config() -> None:
    """Test that _create_event_loop respects the default config."""
    # Reset config to default
    with zarr.config.set({"async.use_uvloop": True}):
        loop = _create_event_loop()
        if sys.platform != "win32":
            try:
                import uvloop

                assert isinstance(loop, uvloop.Loop)
            except ImportError:
                # uvloop not available, should use asyncio
                assert isinstance(loop, asyncio.AbstractEventLoop)
                assert "uvloop" not in str(type(loop))
        else:
            # Windows doesn't support uvloop
            assert isinstance(loop, asyncio.AbstractEventLoop)
            assert "uvloop" not in str(type(loop))

        loop.close()


def test_create_event_loop_uvloop_disabled() -> None:
    """Test that uvloop can be disabled via config."""
    with zarr.config.set({"async.use_uvloop": False}):
        loop = _create_event_loop()
        # Should always use asyncio when disabled
        assert isinstance(loop, asyncio.AbstractEventLoop)
        assert "uvloop" not in str(type(loop))
        loop.close()


@pytest.mark.skipif(sys.platform == "win32", reason="uvloop is not supported on Windows")
def test_create_event_loop_uvloop_enabled_non_windows() -> None:
    """Test uvloop usage on non-Windows platforms when uvloop is installed."""
    uvloop = pytest.importorskip("uvloop")

    with zarr.config.set({"async.use_uvloop": True}):
        loop = _create_event_loop()
        assert isinstance(loop, uvloop.Loop)
        loop.close()


@pytest.mark.skipif(sys.platform != "win32", reason="This test is specific to Windows behavior")
def test_create_event_loop_windows_no_uvloop() -> None:
    """Test that uvloop is never used on Windows."""
    with zarr.config.set({"async.use_uvloop": True}):
        loop = _create_event_loop()
        # Should use asyncio even when uvloop is requested on Windows
        assert isinstance(loop, asyncio.AbstractEventLoop)
        assert "uvloop" not in str(type(loop))
        loop.close()


def test_uvloop_mock_import_error(clean_state) -> None:
    """Test graceful handling when uvloop import fails."""
    with zarr.config.set({"async.use_uvloop": True}):
        # Mock uvloop import failure
        with patch.dict("sys.modules", {"uvloop": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module named 'uvloop'")):
                loop = _create_event_loop()
                # Should fall back to asyncio
                assert isinstance(loop, asyncio.AbstractEventLoop)
                assert "uvloop" not in str(type(loop))
                loop.close()
