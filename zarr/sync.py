import os
from collections import defaultdict
from threading import Lock

import fasteners


class ThreadSynchronizer(object):
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


class ProcessSynchronizer(object):
    """Provides synchronization using file locks via the
    `fasteners <http://fasteners.readthedocs.io/en/latest/api/process_lock.html>`_
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
        path = os.path.join(self.path, item)
        lock = fasteners.InterProcessLock(path)
        return lock

    # pickling and unpickling should be handled automatically
