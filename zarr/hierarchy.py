# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np


from zarr.attrs import Attributes
from zarr.core import Array
from zarr.storage import contains_array, contains_group, init_group, \
    DictStore, DirectoryStore, group_meta_key, attrs_key, listdir
from zarr.creation import array, create, empty, zeros, ones, full, \
    empty_like, zeros_like, ones_like, full_like
from zarr.util import normalize_storage_path, normalize_shape
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
        akey = self._key_prefix + attrs_key
        self._attrs = Attributes(store, key=akey, readonly=readonly)

    @property
    def store(self):
        """TODO"""
        return self._store

    @property
    def path(self):
        """TODO doc me"""
        return self._path

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
        for key in sorted(listdir(self.store, self.path)):
            path = self.path + '/' + key
            if (contains_array(self.store, path) or
                    contains_group(self.store, path)):
                yield key

    def values(self):
        return (v for _, v in self.items())

    def items(self):
        for key in sorted(listdir(self.store, self.path)):
            path = self.path + '/' + key
            if contains_array(self.store, path):
                yield key, Array(self.store, path=path, readonly=self.readonly)
            elif contains_group(self.store, path):
                yield key, Group(self.store, path=path, readonly=self.readonly)

    def group_keys(self):
        for key in sorted(listdir(self.store, self.path)):
            path = self.path + '/' + key
            if contains_group(self.store, path):
                yield key

    def groups(self):
        for key in sorted(listdir(self.store, self.path)):
            path = self.path + '/' + key
            if contains_group(self.store, path):
                yield key, Group(self.store, path=path, readonly=self.readonly)

    def array_keys(self):
        for key in sorted(listdir(self.store, self.path)):
            path = self.path + '/' + key
            if contains_array(self.store, path):
                yield key

    def arrays(self):
        for key in sorted(listdir(self.store, self.path)):
            path = self.path + '/' + key
            if contains_array(self.store, path):
                yield key, Array(self.store, path=path, readonly=self.readonly)

    def create_group(self, name):
        """TODO doc me"""

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

    def create_groups(self, *names):
        """TODO doc me"""
        return tuple(self.create_group(name) for name in names)

    def require_group(self, name):
        """TODO doc me"""

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

    def require_groups(self, *names):
        """TODO doc me"""
        return tuple(self.require_group(name) for name in names)

    def _require_parent_group(self, path):
        segments = path.split('/')
        for i in range(len(segments)):
            p = '/'.join(segments[:i])
            if contains_array(self.store, p):
                raise KeyError(path)
            elif not contains_group(self.store, p):
                init_group(self.store, path=p)

    def create_dataset(self, name, data=None, shape=None, chunks=None,
                       dtype=None, compression='default',
                       compression_opts=None, fill_value=None, order='C',
                       synchronizer=None, **kwargs):
        """TODO doc me"""

        # setup
        if self.readonly:
            raise ReadOnlyError('group is read-only')
        path = self._item_path(name)
        self._require_parent_group(path)

        # guard conditions
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

    def require_dataset(self, name, shape, dtype=None, exact=False, **kwargs):
        """TODO doc me"""

        path = self._item_path(name)

        if contains_array(self.store, path):
            a = Array(self.store, path=path, readonly=self.readonly)
            shape = normalize_shape(shape)
            if shape != a.shape:
                raise TypeError('shapes do not match')
            dtype = np.dtype(dtype)
            if exact:
                if dtype != a.dtype:
                    raise TypeError('dtypes do not match exactly')
            else:
                if not np.can_cast(dtype, a.dtype):
                    raise TypeError('dtypes cannot be safely cast')
            return a

        else:
            return self.create_dataset(name, shape=shape, dtype=dtype,
                                       **kwargs)

    def create(self, name, **kwargs):
        if self.readonly:
            raise ReadOnlyError('group is read-only')
        path = self._item_path(name)
        self._require_parent_group(path)
        return create(store=self.store, path=path, **kwargs)

    def empty(self, name, **kwargs):
        if self.readonly:
            raise ReadOnlyError('group is read-only')
        path = self._item_path(name)
        self._require_parent_group(path)
        return empty(store=self.store, path=path, **kwargs)

    def zeros(self, name, **kwargs):
        if self.readonly:
            raise ReadOnlyError('group is read-only')
        path = self._item_path(name)
        self._require_parent_group(path)
        return zeros(store=self.store, path=path, **kwargs)

    def ones(self, name, **kwargs):
        if self.readonly:
            raise ReadOnlyError('group is read-only')
        path = self._item_path(name)
        self._require_parent_group(path)
        return ones(store=self.store, path=path, **kwargs)

    def full(self, name, **kwargs):
        if self.readonly:
            raise ReadOnlyError('group is read-only')
        path = self._item_path(name)
        self._require_parent_group(path)
        return full(store=self.store, path=path, **kwargs)

    def array(self, name, data, **kwargs):
        if self.readonly:
            raise ReadOnlyError('group is read-only')
        path = self._item_path(name)
        self._require_parent_group(path)
        return array(data, store=self.store, path=path, **kwargs)

    def empty_like(self, name, data, **kwargs):
        if self.readonly:
            raise ReadOnlyError('group is read-only')
        path = self._item_path(name)
        self._require_parent_group(path)
        return empty_like(data, store=self.store, path=path, **kwargs)

    def zeros_like(self, name, data, **kwargs):
        if self.readonly:
            raise ReadOnlyError('group is read-only')
        path = self._item_path(name)
        self._require_parent_group(path)
        return zeros_like(data, store=self.store, path=path, **kwargs)

    def ones_like(self, name, data, **kwargs):
        if self.readonly:
            raise ReadOnlyError('group is read-only')
        path = self._item_path(name)
        self._require_parent_group(path)
        return ones_like(data, store=self.store, path=path, **kwargs)

    def full_like(self, name, data, **kwargs):
        if self.readonly:
            raise ReadOnlyError('group is read-only')
        path = self._item_path(name)
        self._require_parent_group(path)
        return full_like(data, store=self.store, path=path, **kwargs)

    # def copy(self, source, dest, name, shallow=False):
    #     # TODO
    #     pass


def group(store=None, overwrite=False):
    """Create a group.

    Parameters
    ----------
    store : MutableMapping, optional
        Group storage. If not provided, a DictStore will be used, meaning
        that data will be stored in memory.
    overwrite : bool, optional
        If True, delete any pre-existing data in `store` at `path` before
        creating the group.

    Returns
    -------
    g : zarr.hierarchy.Group

    Examples
    --------

    Create a group in memory::

        >>> import zarr
        >>> g = zarr.group()
        >>> g
        zarr.hierarchy.Group(/, 0)
          store: zarr.storage.DictStore

    Create a group with a different store::

        >>> store = zarr.DirectoryStore('example')
        >>> g = zarr.group(store=store, overwrite=True)
        >>> g
        zarr.hierarchy.Group(/, 0)
          store: zarr.storage.DirectoryStore

    """

    # ensure store
    if store is None:
        store = DictStore()

    # require group
    if overwrite:
        init_group(store, overwrite=True)
    elif contains_array(store):
        raise ValueError('store contains an array')
    elif not contains_group(store):
        init_group(store)

    return Group(store, readonly=False)


def open_group(path, mode='a'):
    """Open a group stored in a directory on the file system.

    Parameters
    ----------
    path : string
        Path to directory in file system in which to store the group.
    mode : {'r', 'r+', 'a', 'w', 'w-'}
        Persistence mode: 'r' means readonly (must exist); 'r+' means
        read/write (must exist); 'a' means read/write (create if doesn't
        exist); 'w' means create (overwrite if exists); 'w-' means create
        (fail if exists).

    Returns
    -------
    g : zarr.hierarchy.Group

    Examples
    --------
    >>> import zarr
    >>> root = zarr.open_group('example', mode='w')
    >>> foo = root.create_group('foo')
    >>> bar = root.create_group('bar')
    >>> root
    zarr.hierarchy.Group(/, 2)
      groups: 2; bar, foo
      store: zarr.storage.DirectoryStore
    >>> root2 = zarr.open_group('example', mode='a')
    >>> root2
    zarr.hierarchy.Group(/, 2)
      groups: 2; bar, foo
      store: zarr.storage.DirectoryStore
    >>> root == root2
    True

    """

    # setup store
    store = DirectoryStore(path)

    # ensure store is initialized

    if mode in ['r', 'r+']:
        if contains_array(store):
            raise ValueError('store contains array')
        elif not contains_group(store):
            raise ValueError('group does not exist')

    elif mode == 'w':
        init_group(store, overwrite=True)

    elif mode == 'a':
        if contains_array(store):
            raise ValueError('store contains array')
        elif not contains_group(store):
            init_group(store)

    elif mode in ['w-', 'x']:
        if contains_array(store):
            raise ValueError('store contains array')
        elif contains_group(store):
            raise ValueError('store contains group')
        else:
            init_group(store)

    # determine readonly status
    readonly = mode == 'r'

    return Group(store, readonly=readonly)
