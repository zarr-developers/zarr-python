import os
from collections import defaultdict
from threading import Lock
from typing import Protocol


class Synchronizer(Protocol):
    """Base class for synchronizers."""

    def __getitem__(self, item):
        # see subclasses
        ...


class ThreadSynchronizer(Synchronizer):
    """Provides synchronization using thread locks."""

    def __init__(self):
        self.mutex = Lock()
        self.locks = defaultdict(Lock)

    def __getitem__(self, item):
        with self.mutex:
            return self.locks[item]

    def __getstate__(self):
        return True

    def __setstate__(self, *args):
        # reinitialize from scratch
        self.__init__()


class ProcessSynchronizer(Synchronizer):
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

    def __getitem__(self, item):
        import fasteners

        path = os.path.join(self.path, item)
        lock = fasteners.InterProcessLock(path)
        return lock

    # pickling and unpickling should be handled automatically
