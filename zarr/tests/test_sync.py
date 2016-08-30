# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from tempfile import mkdtemp
import atexit
import json
import shutil


from zarr.tests.test_attrs import TestAttributes
from zarr.tests.test_core import TestArray
from zarr.sync import ThreadSynchronizer, ProcessSynchronizer
from zarr.core import SynchronizedArray
from zarr.attrs import SynchronizedAttributes
from zarr.storage import init_array


class TestThreadSynchronizedAttributes(TestAttributes):

    def init_attributes(self, store, readonly=False):
        key = 'attrs'
        store[key] = json.dumps(dict()).encode('ascii')
        synchronizer = ThreadSynchronizer()
        return SynchronizedAttributes(store, synchronizer, key=key,
                                      readonly=readonly)


class TestProcessSynchronizedAttributes(TestAttributes):

    def init_attributes(self, store, readonly=False):
        key = 'attrs'
        store[key] = json.dumps(dict()).encode('ascii')
        sync_path = mkdtemp()
        atexit.register(shutil.rmtree, sync_path)
        synchronizer = ProcessSynchronizer(sync_path)
        return SynchronizedAttributes(store, synchronizer, key=key,
                                      readonly=readonly)


class TestThreadSynchronizedArray(TestArray):

    def create_array(self, store=None, path=None, readonly=False,
                     chunk_store=None, **kwargs):
        if store is None:
            store = dict()
        init_array(store, path=path, chunk_store=chunk_store, **kwargs)
        return SynchronizedArray(store,
                                 path=path, synchronizer=ThreadSynchronizer(),
                                 readonly=readonly, chunk_store=chunk_store)

    def test_repr(self):
        pass


class TestProcessSynchronizedArray(TestArray):

    def create_array(self, store=None, path=None, readonly=False,
                     chunk_store=None, **kwargs):
        if store is None:
            store = dict()
        init_array(store, path=path, chunk_store=chunk_store, **kwargs)
        sync_path = mkdtemp()
        atexit.register(shutil.rmtree, sync_path)
        synchronizer = ProcessSynchronizer(sync_path)
        return SynchronizedArray(store, path=path, synchronizer=synchronizer,
                                 readonly=readonly, chunk_store=chunk_store)

    def test_repr(self):
        pass
