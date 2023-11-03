from __future__ import annotations

import asyncio
import threading
from typing import Any, Coroutine, List, Optional

# From https://github.com/fsspec/filesystem_spec/blob/master/fsspec/asyn.py

iothread: List[Optional[threading.Thread]] = [None]  # dedicated IO thread
loop: List[Optional[asyncio.AbstractEventLoop]] = [
    None
]  # global event loop for any non-async instance
_lock: Optional[threading.Lock] = None  # global lock placeholder
get_running_loop = asyncio.get_running_loop


def _get_lock() -> threading.Lock:
    """Allocate or return a threading lock.

    The lock is allocated on first use to allow setting one lock per forked process.
    """
    global _lock
    if not _lock:
        _lock = threading.Lock()
    return _lock


async def _runner(
    event: threading.Event, coro: Coroutine, result_box: List[Optional[Any]]
):
    try:
        result_box[0] = await coro
    except Exception as ex:
        result_box[0] = ex
    finally:
        event.set()


def sync(coro: Coroutine, loop: Optional[asyncio.AbstractEventLoop] = None):
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
    if loop is None or loop.is_closed():
        raise RuntimeError("Loop is not running")
    try:
        loop0 = asyncio.events.get_running_loop()
        if loop0 is loop:
            raise NotImplementedError("Calling sync() from within a running loop")
    except RuntimeError:
        pass
    result_box: List[Optional[Any]] = [None]
    event = threading.Event()
    asyncio.run_coroutine_threadsafe(_runner(event, coro, result_box), loop)
    while True:
        # this loops allows thread to get interrupted
        if event.wait(1):
            break

    return_result = result_box[0]
    if isinstance(return_result, BaseException):
        raise return_result
    else:
        return return_result


def _get_loop():
    """Create or return the default fsspec IO loop

    The loop will be running on a separate thread.
    """
    if loop[0] is None:
        with _get_lock():
            # repeat the check just in case the loop got filled between the
            # previous two calls from another thread
            if loop[0] is None:
                loop[0] = asyncio.new_event_loop()
                th = threading.Thread(target=loop[0].run_forever, name="zarritaIO")
                th.daemon = True
                th.start()
                iothread[0] = th
    return loop[0]
