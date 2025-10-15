import asyncio
import sys
import threading
import time
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
    set_event_loop,
    sync,
)
from zarr.storage import MemoryStore


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

    with pytest.raises(TimeoutError):
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
    """Test that _create_event_loop uses asyncio by default."""
    # Default config should use asyncio (not uvloop)
    loop = _create_event_loop()
    # Should always use asyncio by default
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


@pytest.mark.skipif(sys.platform == "win32", reason="uvloop is not supported on Windows")
def test_uvloop_mock_import_error(clean_state) -> None:
    """Test graceful handling when uvloop import fails."""
    with zarr.config.set({"async.use_uvloop": True}):
        # Mock uvloop import failure by putting None in sys.modules
        # This simulates the module being unavailable/corrupted
        with patch.dict("sys.modules", {"uvloop": None}):
            # When Python tries to import uvloop, it will get None and treat it as ImportError
            loop = _create_event_loop()
            # Should fall back to asyncio
            assert isinstance(loop, asyncio.AbstractEventLoop)
            assert "uvloop" not in str(type(loop))
            loop.close()


# Tests for set_event_loop


def test_set_event_loop_basic(clean_state) -> None:
    """Test basic functionality of set_event_loop."""
    # Create a custom event loop
    custom_loop = asyncio.new_event_loop()

    # Start it in a background thread
    thread = threading.Thread(target=custom_loop.run_forever, daemon=True)
    thread.start()
    time.sleep(0.1)

    # Set it as Zarr's loop
    set_event_loop(custom_loop)

    # Verify that Zarr operations use this loop
    store = MemoryStore()
    group = zarr.open_group(store=store, mode="w")
    array = group.create_array("test", shape=(10, 10), chunks=(5, 5), dtype="float32")

    # Write and read data
    import numpy as np

    data = np.random.random((10, 10)).astype("float32")
    array[:] = data
    result = array[:]

    assert np.array_equal(data, result)

    # Clean up
    custom_loop.call_soon_threadsafe(custom_loop.stop)
    thread.join(timeout=1.0)
    custom_loop.close()


@pytest.mark.skipif(sys.platform == "win32", reason="uvloop not supported on Windows")
def test_set_event_loop_with_uvloop(clean_state) -> None:
    """Test set_event_loop with uvloop."""
    uvloop = pytest.importorskip("uvloop")

    # Create uvloop instance
    uvloop_instance = uvloop.new_event_loop()

    # Start it in a background thread
    thread = threading.Thread(target=uvloop_instance.run_forever, daemon=True)
    thread.start()
    time.sleep(0.1)

    # Set as Zarr's loop
    set_event_loop(uvloop_instance)

    # Perform Zarr operations
    store = MemoryStore()
    group = zarr.open_group(store=store, mode="w")
    array = group.create_array("test", shape=(20, 20), chunks=(10, 10), dtype="int32")

    # Verify operations work
    import numpy as np

    data = np.arange(400).reshape(20, 20).astype("int32")
    array[:] = data
    result = array[:]

    assert np.array_equal(data, result)

    # Verify we're actually using uvloop
    assert "uvloop" in str(type(uvloop_instance))

    # Clean up
    uvloop_instance.call_soon_threadsafe(uvloop_instance.stop)
    thread.join(timeout=1.0)
    uvloop_instance.close()


def test_set_event_loop_type_validation() -> None:
    """Test that set_event_loop validates the input type."""
    # Should raise TypeError for non-loop objects
    with pytest.raises(TypeError, match="must be an instance of asyncio.AbstractEventLoop"):
        set_event_loop("not a loop")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="must be an instance of asyncio.AbstractEventLoop"):
        set_event_loop(123)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="must be an instance of asyncio.AbstractEventLoop"):
        set_event_loop(None)  # type: ignore[arg-type]


def test_set_event_loop_warns_on_replacement(clean_state, caplog) -> None:
    """Test that replacing an existing loop produces a warning."""
    # First, trigger creation of Zarr's default loop
    store = MemoryStore()
    _ = zarr.open_group(store=store, mode="w")

    # Now try to replace it
    custom_loop = asyncio.new_event_loop()
    thread = threading.Thread(target=custom_loop.run_forever, daemon=True)
    thread.start()
    time.sleep(0.1)

    # Should produce a warning
    with caplog.at_level("WARNING"):
        set_event_loop(custom_loop)

    assert "Replacing existing Zarr event loop" in caplog.text

    # Clean up
    custom_loop.call_soon_threadsafe(custom_loop.stop)
    thread.join(timeout=1.0)
    custom_loop.close()


def test_set_event_loop_concurrent_operations(clean_state) -> None:
    """Test that custom loop handles concurrent Zarr operations."""
    custom_loop = asyncio.new_event_loop()
    thread = threading.Thread(target=custom_loop.run_forever, daemon=True)
    thread.start()
    time.sleep(0.1)

    set_event_loop(custom_loop)

    # Create multiple arrays and perform concurrent operations
    store = MemoryStore()
    group = zarr.open_group(store=store, mode="w")

    arrays = []
    for i in range(5):
        arr = group.create_array(f"array_{i}", shape=(10, 10), chunks=(5, 5), dtype="float32")
        arrays.append(arr)

    import numpy as np

    # Write to all arrays
    data = np.random.random((10, 10)).astype("float32")
    for arr in arrays:
        arr[:] = data

    # Read from all arrays
    for arr in arrays:
        result = arr[:]
        assert np.array_equal(data, result)

    # Clean up
    custom_loop.call_soon_threadsafe(custom_loop.stop)
    thread.join(timeout=1.0)
    custom_loop.close()


def test_set_event_loop_before_first_use(clean_state) -> None:
    """Test setting custom loop before any Zarr operations (recommended usage)."""
    # Create and set custom loop BEFORE any Zarr operations
    custom_loop = asyncio.new_event_loop()
    thread = threading.Thread(target=custom_loop.run_forever, daemon=True)
    thread.start()
    time.sleep(0.1)

    # Set the loop before doing anything with Zarr
    set_event_loop(custom_loop)

    # Now perform Zarr operations
    store = MemoryStore()
    group = zarr.open_group(store=store, mode="w")
    array = group.create_array("test", shape=(10, 10), chunks=(5, 5), dtype="float32")

    import numpy as np

    data = np.random.random((10, 10)).astype("float32")
    array[:] = data
    result = array[:]

    assert np.array_equal(data, result)

    # Clean up
    custom_loop.call_soon_threadsafe(custom_loop.stop)
    thread.join(timeout=1.0)
    custom_loop.close()


def test_set_event_loop_thread_safety(clean_state) -> None:
    """Test that set_event_loop is thread-safe."""
    import concurrent.futures

    custom_loop = asyncio.new_event_loop()
    thread = threading.Thread(target=custom_loop.run_forever, daemon=True)
    thread.start()
    time.sleep(0.1)

    # Try setting the loop from multiple threads simultaneously
    def set_loop():
        set_event_loop(custom_loop)
        return True

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(set_loop) for _ in range(10)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    # All should succeed
    assert all(results)

    # Clean up
    custom_loop.call_soon_threadsafe(custom_loop.stop)
    thread.join(timeout=1.0)
    custom_loop.close()
