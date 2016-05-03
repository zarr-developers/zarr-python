# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from threading import Lock
from collections import defaultdict
import os


import fasteners


class ThreadSynchronizer(object):
    """Provides synchronization using thread locks."""

    def __init__(self):
        self.mutex = Lock()
        self.attrs_lock = Lock()
        self.chunk_locks = defaultdict(Lock)

    def chunk_lock(self, ckey):
        with self.mutex:
            lock = self.chunk_locks[ckey]
        return lock

    def __getstate__(self):
        return dict()

    def __setstate__(self, d):
        # reinitialise from scratch
        self.mutex = Lock()
        self.attrs_lock = Lock()
        self.chunk_locks = defaultdict(Lock)


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

    @property
    def attrs_lock(self):
        return fasteners.InterProcessLock(
            os.path.join(self.path, 'attrs.lock')
        )

    def chunk_lock(self, ckey):
        lock = fasteners.InterProcessLock(
            os.path.join(self.path, '%s.lock' % ckey)
        )
        return lock
