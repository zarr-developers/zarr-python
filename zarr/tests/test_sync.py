# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from tempfile import mkdtemp
import atexit
import shutil


from zarr.tests.test_attrs import TestAttributes
from zarr.tests.test_core import TestArray
from zarr.sync import ThreadSynchronizer, ProcessSynchronizer, \
    SynchronizedArray, SynchronizedAttributes
from zarr.storage import init_store


class TestThreadSynchronizedAttributes(TestAttributes):

    def init_attributes(self, store, readonly=False):
        synchronizer = ThreadSynchronizer()
        return SynchronizedAttributes(store, synchronizer, readonly=readonly)


class TestProcessSynchronizedAttributes(TestAttributes):

    def init_attributes(self, store, readonly=False):
        sync_path = mkdtemp()
        atexit.register(shutil.rmtree, sync_path)
        synchronizer = ProcessSynchronizer(sync_path)
        return SynchronizedAttributes(store, synchronizer, readonly=readonly)


class TestThreadSynchronizedArray(TestArray):

    def create_array(self, store=None, readonly=False, **kwargs):
        if store is None:
            store = dict()
        init_store(store, **kwargs)
        return SynchronizedArray(store, synchronizer=ThreadSynchronizer(),
                                 readonly=readonly)


class TestProcessSynchronizedArray(TestArray):

    def create_array(self, store=None, readonly=False, **kwargs):
        if store is None:
            store = dict()
        init_store(store, **kwargs)
        sync_path = mkdtemp()
        atexit.register(shutil.rmtree, sync_path)
        synchronizer = ProcessSynchronizer(sync_path)
        return SynchronizedArray(store, synchronizer=synchronizer,
                                 readonly=readonly)
