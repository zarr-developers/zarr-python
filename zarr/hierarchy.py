# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import os


from zarr.attrs import Attributes
from zarr.core import Array
from zarr.storage import contains_array, contains_group, init_group, \
    DictStore, DirectoryStore, group_meta_key, attrs_key, listdir
from zarr.creation import array, create
from zarr.util import normalize_storage_path
from zarr.errors import ReadOnlyError
from zarr.meta import decode_group_metadata


class Group(object):
    """Instantiate a group from an initialised store.

    Parameters
    ----------
    store : HierarchicalStore
        Group store, already initialised.
    readonly : bool, optional
        True if group should be protected against modification.
    path : string, optional
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

    def __init__(self, store, path=None, readonly=False):

        self._store = store
        self._path = normalize_storage_path(path)
        if self._path:
            self._key_prefix = self._path + '/'
        else:
            self._key_prefix = ''
        self._readonly = readonly

        # guard conditions
        if contains_array(store, path=self._path):
            raise ValueError('store contains an array')

        # initialise metadata
        try:
            mkey = self._key_prefix + group_meta_key
            meta_bytes = store[mkey]
        except KeyError:
            raise ValueError('store has no metadata')
        else:
            meta = decode_group_metadata(meta_bytes)
            self._meta = meta

        # setup attributes
        self._attrs = Attributes(store, key=self.attrs_key, readonly=readonly)

    @property
    def store(self):
        """TODO"""
        return self._store

    @property
    def path(self):
        """TODO doc me"""
        return self._path

    @property
    def meta_key(self):
        """TODO doc me"""
        if self.path:
            return self.path + '/' + group_meta_key
        else:
            return group_meta_key

    @property
    def attrs_key(self):
        """TODO doc me"""
        if self.path:
            return self.path + '/' + attrs_key
        else:
            return attrs_key

    @property
    def readonly(self):
        """TODO"""
        return self._readonly

    @property
    def name(self):
        """TODO doc me"""
        if self.path:
            # follow h5py convention: add leading slash
            name = self.path
            if name[0] != '/':
                name = '/' + name
            return name
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
            self.path == other.path
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

    def _item_path(self, item):
        if item and item[0] == '/':
            # absolute path
            path = normalize_storage_path(item)
        else:
            # relative path
            path = normalize_storage_path(item)
            if self.path:
                path = self.path + '/' + path
        return path

    def __contains__(self, item):
        path = self._item_path(item)
        if contains_array(self.store, path):
            return True
        elif contains_group(self.store, path):
            return True
        else:
            return False

    def __getitem__(self, item):
        path = self._item_path(item)
        if contains_array(self.store, path):
            return Array(self.store, readonly=self.readonly, path=path)
        elif contains_group(self.store, path):
            return Group(self.store, readonly=self.readonly, path=path)
        else:
            raise KeyError(item)

    def __setitem__(self, key, value):
        # don't implement this for now
        raise NotImplementedError()

    def keys(self):
        for key in listdir(self.store, self.path):
            path = self.path + '/' + key
            if (contains_array(self.store, path) or
                    contains_group(self.store, path)):
                yield key

    def values(self):
        return (v for _, v in self.items())

    def items(self):
        for key in listdir(self.store, self.path):
            path = self.path + '/' + key
            if contains_array(self.store, path):
                yield key, Array(self.store, path=path, readonly=self.readonly)
            elif contains_group(self.store, path):
                yield key, Group(self.store, path=path, readonly=self.readonly)

    def group_keys(self):
        for key in listdir(self.store, self.path):
            path = self.path + '/' + key
            if contains_group(self.store, path):
                yield key

    def groups(self):
        for key in listdir(self.store, self.path):
            path = self.path + '/' + key
            if contains_group(self.store, path):
                yield key, Group(self.store, path=path, readonly=self.readonly)

    def array_keys(self):
        for key in listdir(self.store, self.path):
            path = self.path + '/' + key
            if contains_array(self.store, path):
                yield key

    def arrays(self):
        for key in listdir(self.store, self.path):
            path = self.path + '/' + key
            if contains_array(self.store, path):
                yield key, Array(self.store, path=path, readonly=self.readonly)

    def create_group(self, name):
        if self.readonly:
            raise ReadOnlyError('group is read-only')

        path = self._item_path(name)

        # require intermediate groups
        segments = path.split('/')
        for i in range(len(segments)):
            p = '/'.join(segments[:i])
            if contains_array(self.store, p):
                raise KeyError(name)
            elif not contains_group(self.store, p):
                init_group(self.store, path=p)

        # create terminal group
        if contains_array(self.store, path):
            raise KeyError(name)
        if contains_group(self.store, path):
            raise KeyError(name)
        else:
            init_group(self.store, path=path)
            return Group(self.store, path=path, readonly=self.readonly)

    def require_group(self, name):
        path = self._item_path(name)

        # require all intermediate groups
        segments = path.split('/')
        for i in range(len(segments) + 1):
            p = '/'.join(segments[:i])
            if contains_array(self.store, p):
                raise KeyError(name)
            elif not contains_group(self.store, p):
                if self.readonly:
                    raise ReadOnlyError('group is read-only')
                init_group(self.store, path=p)

        return Group(self.store, path=path, readonly=self.readonly)

    def create_dataset(self, name, data=None, shape=None, chunks=None,
                       dtype=None, compression='default',
                       compression_opts=None, fill_value=None, order='C',
                       synchronizer=None, **kwargs):
        if self.readonly:
            raise ReadOnlyError('group is read-only')

        path = self._item_path(name)

        # require intermediate groups
        segments = path.split('/')
        for i in range(len(segments)):
            p = '/'.join(segments[:i])
            if contains_array(self.store, p):
                raise KeyError(name)
            elif not contains_group(self.store, p):
                init_group(self.store, path=p)

        # create terminal group
        if contains_array(self.store, path):
            raise KeyError(name)
        if contains_group(self.store, path):
            raise KeyError(name)

        # compatibility with h5py
        fill_value = kwargs.get('fillvalue', fill_value)

        if data is not None:
            a = array(data, chunks=chunks, dtype=dtype,
                      compression=compression,
                      compression_opts=compression_opts,
                      fill_value=fill_value, order=order,
                      synchronizer=synchronizer, store=self.store,
                      path=path)

        else:
            a = create(shape=shape, chunks=chunks, dtype=dtype,
                       compression=compression,
                       compression_opts=compression_opts,
                       fill_value=fill_value, order=order,
                       synchronizer=synchronizer, store=self.store,
                       path=path)

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


def group(store=None, readonly=False, overwrite=False):
    """TODO doc me"""
    if store is None:
        store = DictStore()
    init_group(store, overwrite=overwrite)
    return Group(store, readonly=readonly)


def open_group(path, mode='a'):
    """TODO doc me"""

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
