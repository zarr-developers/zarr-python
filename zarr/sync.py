# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from threading import Lock
from collections import defaultdict
import os


import fasteners


class ArraySynchronizer(metaclass=ABCMeta):
    """Abstract class defining the interface to a synchronization manager for a
    single array."""

    @contextmanager
    @abstractmethod
    def lock_array(self):
        """Obtain a lock on the entire array."""
        pass

    @contextmanager
    @abstractmethod
    def lock_attrs(self):
        """Obtain a lock on the user-defined attributes."""
        pass

    @contextmanager
    @abstractmethod
    def lock_chunk(self, key):
        """Obtain a lock on a single chunk.

        Parameters
        ----------
        key : tuple of ints
            Chunk index.
        """
        pass


class ThreadSynchronizer(ArraySynchronizer):

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
            # TODO is the mutex needed? Or is defaultdict already
            # synchronized?
            lock = self._chunk_locks[key]
        lock.acquire()
        try:
            yield
        finally:
            lock.release()


class ProcessSynchronizer(ArraySynchronizer):

    def __init__(self, path):
        self._path = path

    @contextmanager
    def lock_array(self):
        lock = fasteners.InterProcessLock(os.path.join(self._path,
                                                       'array.lock'))
        lock.acquire()
        try:
            yield
        finally:
            lock.release()

    @contextmanager
    def lock_attrs(self):
        lock = fasteners.InterProcessLock(os.path.join(self._path,
                                                       'attrs.lock'))
        lock.acquire()
        try:
            yield
        finally:
            lock.release()

    @contextmanager
    def lock_chunk(self, key):
        lock = fasteners.InterProcessLock(os.path.join(self._path,
                                                       'chunk.%s.lock' % key))
        lock.acquire()
        try:
            yield
        finally:
            lock.release()
