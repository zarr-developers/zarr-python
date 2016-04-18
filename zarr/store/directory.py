# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


from zarr.store.base import ArrayStore


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
