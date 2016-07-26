# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import os


from zarr.attrs import Attributes
from zarr.core import Array
from zarr.storage import check_array, check_group, init_group, MemoryStore, \
    DirectoryStore
from zarr.creation import array, create


class Group(object):
    """TODO"""

    def __init__(self, store, readonly=False, name=None):

        self._store = store
        self._readonly = readonly
        self._name = name

        # initialise attributes
        self._attrs = Attributes(store, readonly=readonly)

    @property
    def store(self):
        """TODO"""
        return self._store

    @property
    def readonly(self):
        """TODO"""
        return self._readonly

    @property
    def name(self):
        """TODO"""
        return self._name

    @property
    def attrs(self):
        """TODO"""
        return self._attrs

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return sum(1 for _ in self.keys())

    def __repr__(self):
        r = 'Group('
        if self.name:
            r += self.name + ', '
        r += str(len(self))
        r += ')'
        array_keys = list(self.array_keys())
        if array_keys:
            arrays_line = '\n  arrays: %s; %s' % \
                (len(array_keys), ', '.join(array_keys))
            if len(arrays_line) > 80:
                arrays_line = arrays_line[:77] + '...'
            r += arrays_line
        group_keys = list(self.group_keys())
        if group_keys:
            groups_line = '\n  groups: %s; %s' % \
                (len(group_keys), ', '.join(group_keys))
            if len(groups_line) > 80:
                groups_line = groups_line[:77] + '...'
            r += groups_line
        r += '\n  store: %s.%s' % (type(self._store).__module__,
                                   type(self._store).__name__)
        return r

    def __contains__(self, key):
        # TODO
        pass

    def __getitem__(self, key):
        names = [s for s in key.split('/') if s]
        if not names:
            raise ValueError(key)
        store = self.store
        for name in names:
            store = store.get_store(name)
        if check_array(store):
            return Array(store, readonly=self.readonly)
        elif check_group(store):
            return Group(store, readonly=self.readonly, name=names[-1])
        else:
            raise KeyError(key)

    def __setitem__(self, key, value):
        # don't implement this for now
        raise NotImplementedError()

    def keys(self):
        for key, store in self.store.stores():
            if check_array(store) or check_group(store):
                yield key

    def values(self):
        return (v for k, v in self.items())

    def items(self):
        for key, store in self.store.stores():
            if check_array(store):
                # TODO what about synchronizer?
                yield key, Array(store, readonly=self.readonly)
            elif check_group(store):
                yield key, Group(store, readonly=self.readonly)

    def group_keys(self):
        for key, store in self.store.stores():
            if check_group(store):
                yield key

    def groups(self):
        for key, store in self.store.stores():
            if check_group(store):
                yield key, Group(store, readonly=self.readonly)

    def array_keys(self):
        for key, store in self.store.stores():
            if check_array(store):
                yield key

    def arrays(self):
        for key, store in self.store.stores():
            if check_array(store):
                # TODO what about synchronizer?
                yield key, Array(store, readonly=self.readonly)

    def create_group(self, name):

        # handle compound request
        names = [s for s in name.split('/') if s]
        if not names:
            raise ValueError(name)

        # create intermediate groups
        store = self.store
        for name in names[:-1]:
            store = store.require_store(name)
            if check_array(store):
                raise KeyError(name)  # TODO better error?
            elif check_group(store):
                pass
            else:
                init_group(store)

        # create final group (must not exist)
        store = store.require_store(names[-1])
        if check_array(store):
            raise KeyError(name)
        elif check_group(store):
            raise KeyError(name)
        else:
            init_group(store)

        return Group(store, readonly=self.readonly, name=names[-1])

    def require_group(self, name):

        # handle compound request
        names = [s for s in name.split('/') if s]
        if not names:
            raise ValueError(name)

        # create groups
        store = self.store
        for name in names:
            store = store.require_store(name)
            if check_array(store):
                raise KeyError(name)  # TODO better error?
            elif check_group(store):
                pass
            else:
                init_group(store)

        return Group(store, readonly=self.readonly, name=names[-1])

    def create_dataset(self, name, data=None, shape=None, chunks=None,
                       dtype=None, compression='default',
                       compression_opts=None, fill_value=None, order='C',
                       synchronizer=None, **kwargs):

        # handle compound request
        names = [s for s in name.split('/') if s]
        if not names:
            raise ValueError(name)

        # create intermediate groups
        store = self.store
        for name in names[:-1]:
            store = store.require_store(name)
            if check_array(store):
                raise KeyError(name)  # TODO better error?
            elif check_group(store):
                pass
            else:
                init_group(store)

        # create store to hold the new array (must not exist)
        store = store.require_store(names[-1])
        if check_array(store):
            raise KeyError(name)
        elif check_group(store):
            raise KeyError(name)

        # compatibility with h5py
        fill_value = kwargs.get('fillvalue', fill_value)

        if name in self:
            raise KeyError(name)

        if data is not None:
            a = array(data, chunks=chunks, dtype=dtype,
                      compression=compression,
                      compression_opts=compression_opts,
                      fill_value=fill_value, order=order,
                      synchronizer=synchronizer, store=store)

        else:
            a = create(shape=shape, chunks=chunks, dtype=dtype,
                       compression=compression,
                       compression_opts=compression_opts,
                       fill_value=fill_value, order=order,
                       synchronizer=synchronizer, store=store)

        return a

    def require_dataset(self, name, shape=None, dtype=None,
                        exact=False, **kwargs):
        # TODO
        pass


def group(store=None, readonly=False):
    """TODO"""
    if store is None:
        store = MemoryStore()
    init_group(store)
    return Group(store, readonly=readonly)


def open_group(path, mode='a'):
    """TODO"""

    # ensure directory exists
    if not os.path.exists(path):
        if mode in ['w', 'w-', 'x', 'a']:
            os.makedirs(path)
        elif mode in ['r', 'r+']:
            raise ValueError('path does not exist: %r' % path)

    # setup store
    store = DirectoryStore(path)

    # store can either hold array or group, not both
    if check_array(store):
        raise ValueError('path contains array')

    exists = check_group(store)

    # ensure store is initialized
    if mode in ['r', 'r+'] and not exists:
        raise ValueError('group does not exist')
    elif mode in ['w-', 'x'] and exists:
        raise ValueError('group exists')
    elif (mode == 'w' or
          (mode in ['w-', 'x'] and not exists)):
        init_group(store, overwrite=True)
    elif mode == 'a':
        init_group(store, overwrite=False)

    # determine readonly status
    readonly = mode == 'r'

    return Group(store, readonly=readonly)
