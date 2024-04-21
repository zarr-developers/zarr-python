from __future__ import annotations

import asyncio
from concurrent.futures import wait
import threading
from typing import (
    Any,
    AsyncIterator,
    Coroutine,
    List,
    TypeVar,
)
from typing_extensions import ParamSpec

from zarr.v3.config import SyncConfiguration

P = ParamSpec("P")
T = TypeVar("T")

# From https://github.com/fsspec/filesystem_spec/blob/master/fsspec/asyn.py

iothread: list[threading.Thread | None] = [None]  # dedicated IO thread
loop: list[asyncio.AbstractEventLoop | None] = [
    None
]  # global event loop for any non-async instance
_lock: threading.Lock | None = None  # global lock placeholder
get_running_loop = asyncio.get_running_loop


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


async def _runner(coro: Coroutine[Any, Any, T]) -> T | BaseException:
    """
    Await a coroutine and return the result of running it. If await it raises an exception,
    that will be returned instead.
    """
    try:
        return await coro
    except Exception as ex:
        return ex


def sync(coro: Coroutine[Any, Any, T], loop: asyncio.AbstractEventLoop | None = None) -> T:
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

    # TODO: add timeout
    done, _ = wait([future], return_when=asyncio.ALL_COMPLETED)
    assert len(done) == 1
    return_result = list(done)[0].result()

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
                th = threading.Thread(target=new_loop.run_forever, name="zarrIO")
                th.daemon = True
                th.start()
                iothread[0] = th
    assert loop[0] is not None
    return loop[0]


class SyncMixin:
    _sync_configuration: SyncConfiguration

    def _sync(self, coroutine: Coroutine[Any, Any, T]) -> T:
        # TODO: refactor this to to take *args and **kwargs and pass those to the method
        # this should allow us to better type the sync wrapper
        return sync(coroutine, loop=self._sync_configuration.asyncio_loop)

    def _sync_iter(self, async_iterator: AsyncIterator[T]) -> List[T]:
        async def iter_to_list() -> List[T]:
            return [item async for item in async_iterator]

        return self._sync(iter_to_list())
