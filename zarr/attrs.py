# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import json
import os
from threading import Lock
import fasteners


class PersistentAttributes(object):

    def __init__(self, path, mode):
        self._path = path
        self._mode = mode

    def __getitem__(self, item):

        if not os.path.exists(self._path):
            raise KeyError(item)

        with open(self._path, mode='r') as f:
            return json.load(f)[item]

    def __setitem__(self, key, value):

        # handle read-only state
        if self._mode == 'r':
            raise ValueError('array is read-only')

        # load existing data
        if not os.path.exists(self._path):
            d = dict()
        else:
            with open(self._path, mode='r') as f:
                d = json.load(f)

        # set key value
        d[key] = value

        # write modified data
        with open(self._path, mode='w') as f:
            json.dump(d, f, indent=4, sort_keys=True)

    def __delitem__(self, key):

        # handle read-only state
        if self._mode == 'r':
            raise ValueError('array is read-only')

        # load existing data
        if not os.path.exists(self._path):
            d = dict()
        else:
            with open(self._path, mode='r') as f:
                d = json.load(f)

        # delete key value
        del d[key]

        # write modified data
        with open(self._path, mode='w') as f:
            json.dump(d, f, indent=4, sort_keys=True)

    def asdict(self):
        if not os.path.exists(self._path):
            d = dict()
        else:
            with open(self._path, mode='r') as f:
                d = json.load(f)
        return d

    def __iter__(self):
        return iter(self.asdict())

    def __len__(self):
        return len(self.asdict())

    def keys(self):
        return self.asdict().keys()

    def values(self):
        return self.asdict().values()

    def items(self):
        return self.asdict().items()


class SynchronizedPersistentAttributes(PersistentAttributes):

    def __init__(self, path, mode):
        super(SynchronizedPersistentAttributes, self).__init__(path, mode)
        lock_path = self._path + '.lock'
        self._thread_lock = Lock()
        self._file_lock = fasteners.InterProcessLock(lock_path)

    def __getitem__(self, item):
        with self._thread_lock:
            with self._file_lock:
                v = super(SynchronizedPersistentAttributes, self).__getitem__(item)
                return v

    def __setitem__(self, key, value):
        with self._thread_lock:
            with self._file_lock:
                super(SynchronizedPersistentAttributes, self).__setitem__(key, value)

    def __delitem__(self, key):
        with self._thread_lock:
            with self._file_lock:
                super(SynchronizedPersistentAttributes, self).__delitem__(key)

    def asdict(self):
        with self._thread_lock:
            with self._file_lock:
                return super(SynchronizedPersistentAttributes, self).asdict()
