# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from threading import Lock
from collections import defaultdict
import os


import fasteners


from zarr.core import Array
from zarr.attrs import Attributes


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


class SynchronizedArray(Array):
    """Instantiate a synchronized array.

    Parameters
    ----------
    store : MutableMapping
        Array store, already initialised.
    synchronizer : object
        Array synchronizer.
    readonly : bool, optional
        True if array should be protected against modification.

    Examples
    --------
    >>> import zarr
    >>> store = dict()
    >>> zarr.init_store(store, shape=1000, chunks=100)
    >>> synchronizer = zarr.ThreadSynchronizer()
    >>> z = zarr.SynchronizedArray(store, synchronizer)
    >>> z
    zarr.sync.SynchronizedArray((1000,), float64, chunks=(100,), order=C)
      compression: blosc; compression_opts: {'clevel': 5, 'cname': 'blosclz', 'shuffle': 1}
      nbytes: 7.8K; nbytes_stored: 289; ratio: 27.7; initialized: 0/10
      store: builtins.dict; synchronizer: zarr.sync.ThreadSynchronizer

    Notes
    -----
    Only writing data to the array via the __setitem__() method and
    modification of user attributes are synchronized. Neither append() nor
    resize() are synchronized.

    Writing to the array is synchronized at the chunk level. I.e.,
    the array supports concurrent write operations via the __setitem__()
    method, but these will only exclude each other if they both require
    modification of the same chunk.

    """  # flake8: noqa

    def __init__(self, store, synchronizer, readonly=False):
        super(SynchronizedArray, self).__init__(store, readonly=readonly)
        self.synchronizer = synchronizer
        self._attrs = SynchronizedAttributes(store, synchronizer,
                                             readonly=readonly)

    def _chunk_setitem(self, cidx, key, value):
        ckey = '.'.join(map(str, cidx))
        with self.synchronizer.chunk_lock(ckey):
            super(SynchronizedArray, self)._chunk_setitem(cidx, key, value)

    def __repr__(self):
        r = super(SynchronizedArray, self).__repr__()
        r += ('; synchronizer: %s.%s' %
              (type(self.synchronizer).__module__,
               type(self.synchronizer).__name__))
        return r

    def __getstate__(self):
        return self._store, self.synchronizer, self._readonly

    def __setstate__(self, state):
        self.__init__(*state)


class SynchronizedAttributes(Attributes):

    def __init__(self, store, synchronizer, key='attrs', readonly=False):
        super(SynchronizedAttributes, self).__init__(store, key=key,
                                                     readonly=readonly)
        self.synchronizer = synchronizer

    def __setitem__(self, key, value):
        with self.synchronizer.attrs_lock:
            super(SynchronizedAttributes, self).__setitem__(key, value)

    def __delitem__(self, key):
        with self.synchronizer.attrs_lock:
            super(SynchronizedAttributes, self).__delitem__(key)

    def update(self, *args, **kwargs):
        with self.synchronizer.attrs_lock:
            super(SynchronizedAttributes, self).update(*args, **kwargs)
