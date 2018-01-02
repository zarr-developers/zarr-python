# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import collections


class CountingDict(collections.MutableMapping):

    def __init__(self):
        self.wrapped = dict()
        self.counter = collections.Counter()

    def __len__(self):
        self.counter['__len__'] += 1
        return len(self.wrapped)

    def keys(self):
        self.counter['keys'] += 1
        return self.wrapped.keys()

    def __iter__(self):
        self.counter['__iter__'] += 1
        return iter(self.wrapped)

    def __contains__(self, item):
        self.counter['__contains__', item] += 1
        return item in self.wrapped

    def __getitem__(self, item):
        self.counter['__getitem__', item] += 1
        return self.wrapped[item]

    def __setitem__(self, key, value):
        self.counter['__setitem__', key] += 1
        self.wrapped[key] = value

    def __delitem__(self, key):
        self.counter['__delitem__', key] += 1
        del self.wrapped[key]
