# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from contextlib import contextmanager
from threading import Lock
from collections import defaultdict
import os


import fasteners


class ThreadSynchronizer(object):

    def __init__(self):
        self._mutex = Lock()
        self._array_lock = Lock()
        self._attrs_lock = Lock()
        self._chunk_locks = defaultdict(Lock)

    @contextmanager
    def lock_array(self):
        self._array_lock.acquire()
        try:
            yield
        finally:
            self._array_lock.release()

    @contextmanager
    def lock_attrs(self):
        self._attrs_lock.acquire()
        try:
            yield
        finally:
            self._attrs_lock.release()

    @contextmanager
    def lock_chunk(self, key):
        with self._mutex:
            # TODO is the mutex needed? Or is defaultdict already synchronized?
            lock = self._chunk_locks[key]
        lock.acquire()
        try:
            yield
        finally:
            lock.release()


class ProcessSynchronizer(object):

    def __init__(self, path):
        self.path = path

    @contextmanager
    def lock_array(self):
        lock = fasteners.InterProcessLock(
            os.path.join(self.path, 'array.lock')
        )
        lock.acquire()
        try:
            yield
        finally:
            lock.release()

    @contextmanager
    def lock_attrs(self):
        lock = fasteners.InterProcessLock(
            os.path.join(self.path, 'attrs.lock')
        )
        lock.acquire()
        try:
            yield
        finally:
            lock.release()

    @contextmanager
    def lock_chunk(self, key):
        lock = fasteners.InterProcessLock(
            os.path.join(self.path, '%s.lock' % key)
        )
        lock.acquire()
        try:
            yield
        finally:
            lock.release()
