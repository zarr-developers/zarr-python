# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


from abc import ABCMeta, abstractmethod


class ArrayStore(metaclass=ABCMeta):
    """Abstract class defining the interface for storage of a single array."""

    @property
    @abstractmethod
    def meta(self):
        """A MutableMapping holding configuration metadata for the array."""
        pass

    @property
    @abstractmethod
    def data(self):
        """A MutableMapping holding compressed data for each chunk of the
        array."""
        pass

    @property
    @abstractmethod
    def attrs(self):
        """A MutableMapping holding user-defined attributes."""
        pass

    @property
    @abstractmethod
    def cbytes(self):
        """The total size in number of bytes of compressed data held for the
        array."""
        pass

    @property
    @abstractmethod
    def initialized(self):
        """The number of chunks that have been initialized."""
        pass

    @abstractmethod
    def resize(self, *args):
        """Resize the array."""
        pass
