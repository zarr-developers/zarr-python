# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


from abc import ABCMeta, abstractmethod


class ArrayStore(metaclass=ABCMeta):

    @property
    @abstractmethod
    def meta(self): pass

    @property
    @abstractmethod
    def data(self): pass

    @property
    @abstractmethod
    def attrs(self): pass

    @property
    @abstractmethod
    def cbytes(self): pass

    @property
    @abstractmethod
    def initialized(self): pass


class MemoryStore(ArrayStore):

    def __init__(self):
        # TODO
        pass

    @property
    def meta(self):
        pass

    @property
    def data(self):
        pass

    @property
    def attrs(self):
        pass

    @property
    def cbytes(self):
        pass

    @property
    def initialized(self):
        pass


class DirectoryStore(ArrayStore):

    def __init__(self, path):
        self._path = path
        # TODO

    @property
    def meta(self):
        pass

    @property
    def data(self):
        pass

    @property
    def attrs(self):
        pass

    @property
    def cbytes(self):
        pass

    @property
    def initialized(self):
        pass
