# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


from abc import ABCMeta, abstractmethod
from contextlib import contextmanager


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
        pass

    @contextmanager
    def lock_array(self):
        # TODO
        pass

    @contextmanager
    def lock_attrs(self):
        # TODO
        pass

    @contextmanager
    def lock_chunk(self, key):
        # TODO
        pass


class ProcessSynchronizer(ArraySynchronizer):

    def __init__(self, path):
        self._path = path

    @contextmanager
    def lock_array(self):
        # TODO
        pass

    @contextmanager
    def lock_attrs(self):
        # TODO
        pass

    @contextmanager
    def lock_chunk(self, key):
        # TODO
        pass
