from abc import abstractmethod
import os
from collections import defaultdict
from threading import Lock
from typing import Protocol, runtime_checkable, ContextManager

import fasteners


@runtime_checkable
class Synchronized(Protocol):
    @abstractmethod
    def _write_context(self, key: str) -> ContextManager:
        ...


@runtime_checkable
class SyncLike(Protocol):
    @abstractmethod
    def __getitem__(self, item: str) -> ContextManager:
        pass


class NoLock(ContextManager):
    """A lock that doesn't lock."""

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


class DummySynchronizer(SyncLike):
    """
    A dummy synchronizer that doesn't provide any synchronization
    """

    def __getitem__(self, item: str) -> NoLock:
        return NoLock()


class ThreadSynchronizer(SyncLike):
    """Provides synchronization using thread locks."""

    def __init__(self):
        self.mutex = Lock()
        self.locks = defaultdict(Lock)

    def __getitem__(self, item: str) -> Lock:
        with self.mutex:
            return self.locks[item]

    def __getstate__(self):
        return True

    def __setstate__(self, *args):
        # reinitialize from scratch
        self.__init__()


class ProcessSynchronizer(SyncLike):
    """Provides synchronization using file locks via the
    `fasteners <https://fasteners.readthedocs.io/en/latest/api/inter_process/>`_
    package.

    Parameters
    ----------
    path : string
        Path to a directory on a file system that is shared by all processes.
        N.B., this should be a *different* path to where you store the array.

    """

    def __init__(self, path):
        self.path = path

    def __getitem__(self, item: str) -> fasteners.InterProcessLock:
        path = os.path.join(self.path, item)
        lock = fasteners.InterProcessLock(path)
        return lock

    # pickling and unpickling should be handled automatically
