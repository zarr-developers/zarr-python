# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from threading import Lock
from collections import defaultdict
import os


import fasteners


from zarr.core import Array
from zarr.attrs import Attributes
from zarr.storage import attrs_key


class ThreadSynchronizer(object):
    """Provides synchronization using thread locks."""

    def __init__(self):
        self.mutex = Lock()
        self.locks = defaultdict(Lock)

    def __getitem__(self, item):
        with self.mutex:
            return self.locks[item]

    def __getstate__(self):
        return dict()

    def __setstate__(self, d):
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

    """  # flake8: noqa

    def __init__(self, path):
        self.path = path

    def __getitem__(self, item):
        lock = fasteners.InterProcessLock(
            os.path.join(self.path, '%s.lock' % item)
        )
        return lock

    # pickling and unpickling should be handled automatically
