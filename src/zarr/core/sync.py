from __future__ import annotations

import asyncio
import atexit
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from typing import TYPE_CHECKING, TypeVar

from typing_extensions import ParamSpec

from zarr.core.config import config

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Coroutine
    from typing import Any

logger = logging.getLogger(__name__)


P = ParamSpec("P")
T = TypeVar("T")

# From https://github.com/fsspec/filesystem_spec/blob/master/fsspec/asyn.py

iothread: list[threading.Thread | None] = [None]  # dedicated IO thread
loop: list[asyncio.AbstractEventLoop | None] = [
    None
]  # global event loop for any non-async instance
_lock: threading.Lock | None = None  # global lock placeholder
_executor: ThreadPoolExecutor | None = None  # global executor placeholder


class SyncError(Exception):
    pass


def _get_lock() -> threading.Lock:
    """Allocate or return a threading lock.

    The lock is allocated on first use to allow setting one lock per forked process.
    """
    global _lock
    if not _lock:
        _lock = threading.Lock()
    return _lock


def _get_executor() -> ThreadPoolExecutor:
    """Return Zarr Thread Pool Executor

    The executor is allocated on first use.
    """
    global _executor
    if not _executor:
        max_workers = config.get("threading.max_workers", None)
        print(max_workers)
        # if max_workers is not None and max_workers > 0:
        #     raise ValueError(max_workers)
        _executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="zarr_pool")
        _get_loop().set_default_executor(_executor)
    return _executor


def cleanup_resources() -> None:
    global _executor
    if _executor:
        _executor.shutdown(wait=True, cancel_futures=True)
    _executor = None

    if loop[0] is not None:
        with _get_lock():
            # Stop the event loop safely
            loop[0].call_soon_threadsafe(loop[0].stop)  # Stop loop from another thread
            if iothread[0] is not None:
                iothread[0].join(timeout=0.2)  # Add a timeout to avoid hanging

                if iothread[0].is_alive():
                    logger.warning(
                        "Thread did not finish cleanly; forcefully closing the event loop."
                    )

            # Forcefully close the event loop to release resources
            loop[0].close()

            # dereference the loop and iothread
            loop[0] = None
            iothread[0] = None


atexit.register(cleanup_resources)


async def _runner(coro: Coroutine[Any, Any, T]) -> T | BaseException:
    """
    Await a coroutine and return the result of running it. If awaiting the coroutine raises an
    exception, the exception will be returned.
    """
    try:
        return await coro
    except Exception as ex:
        return ex


def sync(
    coro: Coroutine[Any, Any, T],
    loop: asyncio.AbstractEventLoop | None = None,
    timeout: float | None = None,
) -> T:
    """
    Make loop run coroutine until it returns. Runs in other thread

    Examples
    --------
    >>> sync(async_function(), existing_loop)
    """
    if loop is None:
        # NB: if the loop is not running *yet*, it is OK to submit work
        # and we will wait for it
        loop = _get_loop()
    if not isinstance(loop, asyncio.AbstractEventLoop):
        raise TypeError(f"loop cannot be of type {type(loop)}")
    if loop.is_closed():
        raise RuntimeError("Loop is not running")
    try:
        loop0 = asyncio.events.get_running_loop()
        if loop0 is loop:
            raise SyncError("Calling sync() from within a running loop")
    except RuntimeError:
        pass

    future = asyncio.run_coroutine_threadsafe(_runner(coro), loop)

    finished, unfinished = wait([future], return_when=asyncio.ALL_COMPLETED, timeout=timeout)
    if len(unfinished) > 0:
        raise TimeoutError(f"Coroutine {coro} failed to finish in within {timeout}s")
    assert len(finished) == 1
    return_result = next(iter(finished)).result()

    if isinstance(return_result, BaseException):
        raise return_result
    else:
        return return_result


def _get_loop() -> asyncio.AbstractEventLoop:
    """Create or return the default fsspec IO loop

    The loop will be running on a separate thread.
    """
    if loop[0] is None:
        with _get_lock():
            # repeat the check just in case the loop got filled between the
            # previous two calls from another thread
            if loop[0] is None:
                new_loop = asyncio.new_event_loop()
                loop[0] = new_loop
                iothread[0] = threading.Thread(target=new_loop.run_forever, name="zarr_io")
                assert iothread[0] is not None
                iothread[0].daemon = True
                iothread[0].start()
    assert loop[0] is not None
    return loop[0]


async def _collect_aiterator(data: AsyncIterator[T]) -> tuple[T, ...]:
    """
    Collect an entire async iterator into a tuple
    """
    result = [x async for x in data]
    return tuple(result)


def collect_aiterator(data: AsyncIterator[T]) -> tuple[T, ...]:
    """
    Synchronously collect an entire async iterator into a tuple.
    """
    return sync(_collect_aiterator(data))


class SyncMixin:
    def _sync(self, coroutine: Coroutine[Any, Any, T]) -> T:
        # TODO: refactor this to to take *args and **kwargs and pass those to the method
        # this should allow us to better type the sync wrapper
        return sync(
            coroutine,
            timeout=config.get("async.timeout"),
        )

    def _sync_iter(self, async_iterator: AsyncIterator[T]) -> list[T]:
        async def iter_to_list() -> list[T]:
            return [item async for item in async_iterator]

        return self._sync(iter_to_list())
