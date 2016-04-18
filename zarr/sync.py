# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


from abc import ABCMeta, abstractmethod
from contextlib import contextmanager


class ArraySynchronizer(metaclass=ABCMeta):

    @contextmanager
    @abstractmethod
    def lock_array(self): pass

    @contextmanager
    @abstractmethod
    def lock_attrs(self): pass

    @contextmanager
    @abstractmethod
    def lock_chunk(self, key): pass


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
