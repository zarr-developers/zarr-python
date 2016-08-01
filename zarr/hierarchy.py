# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import os


from zarr.attrs import Attributes
from zarr.core import Array
from zarr.storage import contains_array, contains_group, init_group, \
    DictStore, DirectoryStore, normalize_prefix, listdir, group_attrs_key
from zarr.creation import array, create


class Group(object):
    """Instantiate a group from an initialised store.

    Parameters
    ----------
    store : HierarchicalStore
        Group store, already initialised.
    readonly : bool, optional
        True if group should be protected against modification.
    name : string, optional
        Group name.

    Attributes
    ----------
    store
    readonly
    name
    attrs

    Methods
    -------
    __iter__
    __len__
    __contains__
    __getitem__
    keys
    values
    items
    group_keys
    groups
    array_keys
    arrays
    create_group
    require_group
    create_dataset
    require_dataset
    empty
    zeros
    ones
    full
    array
    empty_like
    zeros_like
    ones_like
    full_like
    copy

    """

    def __init__(self, store, name=None, readonly=False):

        self._store = store
        self._prefix = normalize_prefix(name)
        self._readonly = readonly

        # guard conditions
        if contains_array(store, prefix=self._prefix):
            raise ValueError('store contains an array')

        # setup attributes
        attrs_key = group_attrs_key(self._prefix)
        self._attrs = Attributes(store, key=attrs_key, readonly=readonly)

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
        if self._prefix:
            # follow h5py convention: add leading slash, remove trailing slash
            return '/' + self._prefix[:-1]
        return '/'

    @property
    def attrs(self):
        """TODO"""
        return self._attrs

    def __eq__(self, other):
        return (
            isinstance(other, Group) and
            self.store == other.store and
            self.readonly == other.readonly and
            self.name == other.name
            # N.B., no need to compare attributes, should be covered by
            # store comparison
        )

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return sum(1 for _ in self.keys())

    def __repr__(self):
        r = '%s.%s(' % (type(self).__module__, type(self).__name__)
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
        # TODO recode to use prefix

        names = [s for s in key.split('/') if s]
        if not names:
            raise KeyError(key)

        # recursively get store
        store = self.store
        for name in names:
            store = store.get_store(name)

        # determine absolute name
        if self.name:
            absname = self.name + '/'
        else:
            absname = ''
        absname += '/'.join(names)

        if contains_array(store):
            return Array(store, readonly=self.readonly, name=absname)
        elif contains_group(store):
            # create group
            return Group(store, readonly=self.readonly, name=absname)
        else:
            raise KeyError(key)

    def __setitem__(self, key, value):
        # don't implement this for now
        raise NotImplementedError()

    def keys(self):
        # TODO recode to use prefix
        for key, store in self.store.stores():
            if contains_array(store) or contains_group(store):
                yield key

    def values(self):
        # TODO recode to use prefix
        return (v for k, v in self.items())

    def items(self):
        # TODO recode to use prefix
        for key, store in self.store.stores():
            if contains_array(store):
                # TODO what about synchronizer?
                yield key, Array(store, readonly=self.readonly)
            elif contains_group(store):
                yield key, Group(store, readonly=self.readonly)

    def group_keys(self):
        # TODO recode to use prefix
        for key, store in self.store.stores():
            if contains_group(store):
                yield key

    def groups(self):
        # TODO recode to use prefix
        for key, store in self.store.stores():
            if contains_group(store):
                yield key, Group(store, readonly=self.readonly)

    def array_keys(self):
        # TODO recode to use prefix
        for key, store in self.store.stores():
            if contains_array(store):
                yield key

    def arrays(self):
        # TODO recode to use prefix
        for key, store in self.store.stores():
            if contains_array(store):
                # TODO what about synchronizer?
                yield key, Array(store, readonly=self.readonly)

    def _require_store(self, name):
        # TODO recode to use prefix

        # handle compound request
        names = [s for s in name.split('/') if s]
        if not names:
            raise KeyError(name)

        # create intermediate stores as groups
        store = self.store
        for name in names[:-1]:
            store = store.require_store(name)
            if contains_array(store):
                raise KeyError(name)  # TODO better error?
            elif contains_group(store):
                pass
            else:
                init_group(store)

        # create final store
        store = store.require_store(names[-1])

        # determine absolute name
        if self.name:
            absname = self.name + '/'
        else:
            absname = ''
        absname += '/'.join(names)

        return store, absname

    def create_group(self, name):
        # TODO recode to use prefix

        # obtain store
        store, absname = self._require_store(name)

        # initialise group
        if contains_array(store):
            raise KeyError(name)
        elif contains_group(store):
            raise KeyError(name)
        else:
            init_group(store)

        return Group(store, readonly=self.readonly, name=absname)

    def require_group(self, name):
        # TODO recode to use prefix

        # obtain store
        store, absname = self._require_store(name)

        # initialise group
        if contains_array(store):
            raise KeyError(name)
        elif contains_group(store):
            pass
        else:
            init_group(store)

        return Group(store, readonly=self.readonly, name=absname)

    def create_dataset(self, name, data=None, shape=None, chunks=None,
                       dtype=None, compression='default',
                       compression_opts=None, fill_value=None, order='C',
                       synchronizer=None, **kwargs):
        # TODO recode to use prefix

        # obtain store
        store, absname = self._require_store(name)

        # guard conditions
        if contains_array(store):
            raise KeyError(name)
        elif contains_group(store):
            raise KeyError(name)

        # compatibility with h5py
        fill_value = kwargs.get('fillvalue', fill_value)

        if data is not None:
            a = array(data, chunks=chunks, dtype=dtype,
                      compression=compression,
                      compression_opts=compression_opts,
                      fill_value=fill_value, order=order,
                      synchronizer=synchronizer, store=store, name=absname)

        else:
            a = create(shape=shape, chunks=chunks, dtype=dtype,
                       compression=compression,
                       compression_opts=compression_opts,
                       fill_value=fill_value, order=order,
                       synchronizer=synchronizer, store=store, name=absname)

        return a

    def require_dataset(self, name, shape=None, dtype=None,
                        exact=False, **kwargs):
        # TODO
        pass

    def empty(self, name, **kwargs):
        # TODO
        pass

    def zeros(self, name, **kwargs):
        # TODO
        pass

    def ones(self, name, **kwargs):
        # TODO
        pass

    def full(self, name, **kwargs):
        # TODO
        pass

    def array(self, name, data, **kwargs):
        # TODO
        pass

    def empty_like(self, name, data, **kwargs):
        # TODO
        pass

    def zeros_like(self, name, data, **kwargs):
        # TODO
        pass

    def ones_like(self, name, data, **kwargs):
        # TODO
        pass

    def full_like(self, name, data, **kwargs):
        # TODO
        pass

    def copy(self, source, dest, name, shallow=False):
        # TODO
        pass


def group(store=None, readonly=False):
    """TODO"""
    if store is None:
        store = DictStore()
    init_group(store)
    return Group(store, readonly=readonly)


def open_group(path, mode='a'):
    """TODO"""

    # TODO recode to use prefix

    # ensure directory exists
    if not os.path.exists(path):
        if mode in ['w', 'w-', 'x', 'a']:
            os.makedirs(path)
        elif mode in ['r', 'r+']:
            raise ValueError('path does not exist: %r' % path)

    # setup store
    store = DirectoryStore(path)

    # store can either hold array or group, not both
    if contains_array(store):
        raise ValueError('path contains array')

    exists = contains_group(store)

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
