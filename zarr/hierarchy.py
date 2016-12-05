# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from collections import MutableMapping


import numpy as np


from zarr.attrs import Attributes
from zarr.core import Array
from zarr.storage import contains_array, contains_group, init_group, \
    DictStore, DirectoryStore, group_meta_key, attrs_key, listdir, rmdir
from zarr.creation import array, create, empty, zeros, ones, full, \
    empty_like, zeros_like, ones_like, full_like
from zarr.util import normalize_storage_path, normalize_shape
from zarr.errors import PermissionError, err_contains_array, \
    err_contains_group, err_group_not_found, err_read_only
from zarr.meta import decode_group_metadata


class Group(MutableMapping):
    """Instantiate a group from an initialized store.

    Parameters
    ----------
    store : MutableMapping
        Group store, already initialized.
    path : string, optional
        Group path.
    read_only : bool, optional
        True if group should be protected against modification.
    chunk_store : MutableMapping, optional
        Separate storage for chunks. If not provided, `store` will be used 
        for storage of both chunks and metadata.
    synchronizer : object, optional
        Array synchronizer.

    Attributes
    ----------
    store
    path
    name
    read_only
    chunk_store
    synchronizer
    attrs

    Methods
    -------
    __len__
    __iter__
    __contains__
    __getitem__
    group_keys
    groups
    array_keys
    arrays
    create_group
    require_group
    create_groups
    require_groups
    create_dataset
    require_dataset
    create
    empty
    zeros
    ones
    full
    array
    empty_like
    zeros_like
    ones_like
    full_like

    """

    def __init__(self, store, path=None, read_only=False, chunk_store=None,
                 synchronizer=None):

        self._store = store
        self._path = normalize_storage_path(path)
        if self._path:
            self._key_prefix = self._path + '/'
        else:
            self._key_prefix = ''
        self._read_only = read_only
        if chunk_store is None:
            self._chunk_store = store
        else:
            self._chunk_store = chunk_store
        self._synchronizer = synchronizer

        # guard conditions
        if contains_array(store, path=self._path):
            err_contains_array(path)

        # initialize metadata
        try:
            mkey = self._key_prefix + group_meta_key
            meta_bytes = store[mkey]
        except KeyError:
            err_group_not_found(path)
        else:
            meta = decode_group_metadata(meta_bytes)
            self._meta = meta

        # setup attributes
        akey = self._key_prefix + attrs_key
        self._attrs = Attributes(store, key=akey, read_only=read_only,
                                 synchronizer=synchronizer)

    @property
    def store(self):
        """A MutableMapping providing the underlying storage for the group."""
        return self._store

    @property
    def path(self):
        """Storage path."""
        return self._path

    @property
    def name(self):
        """Group name following h5py convention."""
        if self._path:
            # follow h5py convention: add leading slash
            name = self._path
            if name[0] != '/':
                name = '/' + name
            return name
        return '/'

    @property
    def read_only(self):
        """A boolean, True if modification operations are not permitted."""
        return self._read_only

    @property
    def chunk_store(self):
        """A MutableMapping providing the underlying storage for array 
        chunks."""
        return self._chunk_store

    @property
    def synchronizer(self):
        """Object used to synchronize write access to groups and arrays."""
        return self._synchronizer

    @property
    def attrs(self):
        """A MutableMapping containing user-defined attributes. Note that
        attribute values must be JSON serializable."""
        return self._attrs

    def __eq__(self, other):
        return (
            isinstance(other, Group) and
            self._store == other.store and
            self._read_only == other.read_only and
            self._path == other.path
            # N.B., no need to compare attributes, should be covered by
            # store comparison
        )

    def __iter__(self):
        """Return an iterator over group member names.

        Examples
        --------
        >>> import zarr
        >>> g1 = zarr.group()
        >>> g2 = g1.create_group('foo')
        >>> g3 = g1.create_group('bar')
        >>> d1 = g1.create_dataset('baz', shape=100, chunks=10)
        >>> d2 = g1.create_dataset('quux', shape=200, chunks=20)
        >>> for name in g1:
        ...     print(name)
        bar
        baz
        foo
        quux

        """
        for key in sorted(listdir(self._store, self._path)):
            path = self._key_prefix + key
            if (contains_array(self._store, path) or
                    contains_group(self._store, path)):
                yield key

    def __len__(self):
        """Number of members."""
        return sum(1 for _ in self)

    def __repr__(self):

        # main line
        r = '%s(' % type(self).__name__
        r += self.name + ', '
        r += str(len(self))
        r += ')'

        # members
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

        # storage and synchronizer classes
        r += '\n  store: %s' % type(self.store).__name__
        if self.store != self.chunk_store:
            r += '; chunk_store: %s' % type(self.chunk_store).__name__
        if self.synchronizer is not None:
            r += '; synchronizer: %s' % type(self.synchronizer).__name__

        return r

    def __getstate__(self):
        return self._store, self._path, self._read_only, self._chunk_store, \
               self._synchronizer

    def __setstate__(self, state):
        self.__init__(*state)

    def _item_path(self, item):
        if item and item[0] == '/':
            # absolute path
            path = normalize_storage_path(item)
        else:
            # relative path
            path = normalize_storage_path(item)
            if self._path:
                path = self._key_prefix + path
        return path

    def __contains__(self, item):
        """Test for group membership.

        Examples
        --------
        >>> import zarr
        >>> g1 = zarr.group()
        >>> g2 = g1.create_group('foo')
        >>> d1 = g1.create_dataset('bar', shape=100, chunks=10)
        >>> 'foo' in g1
        True
        >>> 'bar' in g1
        True
        >>> 'baz' in g1
        False

        """
        path = self._item_path(item)
        return contains_array(self._store, path) or \
            contains_group(self._store, path)

    def __getitem__(self, item):
        """Obtain a group member.

        Parameters
        ----------
        item : string
            Member name or path.

        Examples
        --------
        >>> import zarr
        >>> g1 = zarr.group()
        >>> d1 = g1.create_dataset('foo/bar/baz', shape=100, chunks=10)
        >>> g1['foo']
        Group(/foo, 1)
          groups: 1; bar
          store: DictStore
        >>> g1['foo/bar']
        Group(/foo/bar, 1)
          arrays: 1; baz
          store: DictStore
        >>> g1['foo/bar/baz']
        Array(/foo/bar/baz, (100,), float64, chunks=(10,), order=C)
          nbytes: 800; nbytes_stored: 290; ratio: 2.8; initialized: 0/10
          compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
          store: DictStore

        """  # flake8: noqa
        path = self._item_path(item)
        if contains_array(self._store, path):
            return Array(self._store, read_only=self._read_only, path=path,
                         chunk_store=self._chunk_store,
                         synchronizer=self._synchronizer)
        elif contains_group(self._store, path):
            return Group(self._store, read_only=self._read_only, path=path,
                         chunk_store=self._chunk_store,
                         synchronizer=self._synchronizer)
        else:
            raise KeyError(item)

    def __setitem__(self, item, value):
        self.array(item, value, overwrite=True)

    def __delitem__(self, item):
        return self._write_op(self._delitem_nosync, item)

    def _delitem_nosync(self, item):
        path = self._item_path(item)
        if contains_array(self._store, path) or \
                contains_group(self._store, path):
            rmdir(self._store, path)
        else:
            raise KeyError(item)

    def __getattr__(self, item):
        # allow access to group members via dot notation
        try:
            return self.__getitem__(item)
        except KeyError:
            raise AttributeError

    def group_keys(self):
        """Return an iterator over member names for groups only.

        Examples
        --------
        >>> import zarr
        >>> g1 = zarr.group()
        >>> g2 = g1.create_group('foo')
        >>> g3 = g1.create_group('bar')
        >>> d1 = g1.create_dataset('baz', shape=100, chunks=10)
        >>> d2 = g1.create_dataset('quux', shape=200, chunks=20)
        >>> sorted(g1.group_keys())
        ['bar', 'foo']

        """
        for key in sorted(listdir(self._store, self._path)):
            path = self._key_prefix + key
            if contains_group(self._store, path):
                yield key

    def groups(self):
        """Return an iterator over (name, value) pairs for groups only.

        Examples
        --------
        >>> import zarr
        >>> g1 = zarr.group()
        >>> g2 = g1.create_group('foo')
        >>> g3 = g1.create_group('bar')
        >>> d1 = g1.create_dataset('baz', shape=100, chunks=10)
        >>> d2 = g1.create_dataset('quux', shape=200, chunks=20)
        >>> for n, v in g1.groups():
        ...     print(n, type(v))
        bar <class 'zarr.hierarchy.Group'>
        foo <class 'zarr.hierarchy.Group'>

        """
        for key in sorted(listdir(self._store, self._path)):
            path = self._key_prefix + key
            if contains_group(self._store, path):
                yield key, Group(self._store, path=path,
                                 read_only=self._read_only,
                                 chunk_store=self._chunk_store,
                                 synchronizer=self._synchronizer)

    def array_keys(self):
        """Return an iterator over member names for arrays only.

        Examples
        --------
        >>> import zarr
        >>> g1 = zarr.group()
        >>> g2 = g1.create_group('foo')
        >>> g3 = g1.create_group('bar')
        >>> d1 = g1.create_dataset('baz', shape=100, chunks=10)
        >>> d2 = g1.create_dataset('quux', shape=200, chunks=20)
        >>> sorted(g1.array_keys())
        ['baz', 'quux']

        """
        for key in sorted(listdir(self._store, self._path)):
            path = self._key_prefix + key
            if contains_array(self._store, path):
                yield key

    def arrays(self):
        """Return an iterator over (name, value) pairs for arrays only.

        Examples
        --------
        >>> import zarr
        >>> g1 = zarr.group()
        >>> g2 = g1.create_group('foo')
        >>> g3 = g1.create_group('bar')
        >>> d1 = g1.create_dataset('baz', shape=100, chunks=10)
        >>> d2 = g1.create_dataset('quux', shape=200, chunks=20)
        >>> for n, v in g1.arrays():
        ...     print(n, type(v))
        baz <class 'zarr.core.Array'>
        quux <class 'zarr.core.Array'>

        """
        for key in sorted(listdir(self._store, self._path)):
            path = self._key_prefix + key
            if contains_array(self._store, path):
                yield key, Array(self._store, path=path,
                                 read_only=self._read_only,
                                 chunk_store=self._chunk_store,
                                 synchronizer=self._synchronizer)

    def _write_op(self, f, *args, **kwargs):

        # guard condition
        if self._read_only:
            err_read_only()

        # synchronization
        if self._synchronizer is None:
            return f(*args, **kwargs)
        else:
            # synchronize on the root group
            with self._synchronizer[group_meta_key]:
                return f(*args, **kwargs)

    def create_group(self, name, overwrite=False):
        """Create a sub-group.

        Parameters
        ----------
        name : string
            Group name.
        overwrite : bool, optional
            If True, overwrite any existing array with the given name.

        Returns
        -------
        g : zarr.hierarchy.Group

        Examples
        --------
        >>> import zarr
        >>> g1 = zarr.group()
        >>> g2 = g1.create_group('foo')
        >>> g3 = g1.create_group('bar')
        >>> g4 = g1.create_group('baz/quux')

        """

        return self._write_op(self._create_group_nosync, name,
                              overwrite=overwrite)

    def _create_group_nosync(self, name, overwrite=False):
        path = self._item_path(name)

        # create terminal group
        init_group(self._store, path=path, chunk_store=self._chunk_store,
                   overwrite=overwrite)

        return Group(self._store, path=path, read_only=self._read_only,
                     chunk_store=self._chunk_store,
                     synchronizer=self._synchronizer)

    def create_groups(self, *names, **kwargs):
        """Convenience method to create multiple groups in a single call."""
        return tuple(self.create_group(name, **kwargs) for name in names)

    def require_group(self, name, overwrite=False):
        """Obtain a sub-group, creating one if it doesn't exist.

        Parameters
        ----------
        name : string
            Group name.
        overwrite : bool, optional
            Overwrite any existing array with given `name` if present.

        Returns
        -------
        g : zarr.hierarchy.Group

        Examples
        --------
        >>> import zarr
        >>> g1 = zarr.group()
        >>> g2 = g1.require_group('foo')
        >>> g3 = g1.require_group('foo')
        >>> g2 == g3
        True

        """

        return self._write_op(self._require_group_nosync, name,
                              overwrite=overwrite)

    def _require_group_nosync(self, name, overwrite=False):
        path = self._item_path(name)

        # create terminal group if necessary
        if not contains_group(self._store, path):
            init_group(store=self._store, path=path,
                       chunk_store=self._chunk_store,
                       overwrite=overwrite)

        return Group(self._store, path=path, read_only=self._read_only,
                     chunk_store=self._chunk_store,
                     synchronizer=self._synchronizer)

    def require_groups(self, *names):
        """Convenience method to require multiple groups in a single call."""
        return tuple(self.require_group(name) for name in names)

    def create_dataset(self, name, **kwargs):
        """Create an array.

        Parameters
        ----------
        name : string
            Array name.
        data : array_like, optional
            Initial data.
        shape : int or tuple of ints
            Array shape.
        chunks : int or tuple of ints, optional
            Chunk shape. If not provided, will be guessed from `shape` and
            `dtype`.
        dtype : string or dtype, optional
            NumPy dtype.
        compressor : Codec, optional
            Primary compressor.
        fill_value : object
            Default value to use for uninitialized portions of the array.
        order : {'C', 'F'}, optional
            Memory layout to be used within each chunk.
        synchronizer : zarr.sync.ArraySynchronizer, optional
            Array synchronizer.
        filters : sequence of Codecs, optional
            Sequence of filters to use to encode chunk data prior to
            compression.
        overwrite : bool, optional
            If True, replace any existing array or group with the given name.
        cache_metadata : bool, optional
            If True, array configuration metadata will be cached for the
            lifetime of the object. If False, array metadata will be reloaded
            prior to all data access and modification operations (may incur
            overhead depending on storage and data access pattern).

        Returns
        -------
        a : zarr.core.Array

        Examples
        --------
        >>> import zarr
        >>> g1 = zarr.group()
        >>> d1 = g1.create_dataset('foo', shape=(10000, 10000),
        ...                        chunks=(1000, 1000))
        >>> d1
        Array(/foo, (10000, 10000), float64, chunks=(1000, 1000), order=C)
          nbytes: 762.9M; nbytes_stored: 323; ratio: 2476780.2; initialized: 0/100
          compressor: Blosc(cname='lz4', clevel=5, shuffle=1)
          store: DictStore

        """  # flake8: noqa

        return self._write_op(self._create_dataset_nosync, name, **kwargs)

    def _create_dataset_nosync(self, name, data=None, **kwargs):

        path = self._item_path(name)

        # determine synchronizer
        kwargs.setdefault('synchronizer', self._synchronizer)

        # create array
        if data is None:
            a = create(store=self._store, path=path,
                       chunk_store=self._chunk_store, **kwargs)

        else:
            a = array(data, store=self._store, path=path,
                      chunk_store=self._chunk_store, **kwargs)

        return a

    def require_dataset(self, name, shape, dtype=None, exact=False, **kwargs):
        """Obtain an array, creating if it doesn't exist. Other `kwargs` are
        as per :func:`zarr.hierarchy.Group.create_dataset`.

        Parameters
        ----------
        name : string
            Array name.
        shape : int or tuple of ints
            Array shape.
        dtype : string or dtype, optional
            NumPy dtype.
        exact : bool, optional
            If True, require `dtype` to match exactly. If false, require
            `dtype` can be cast from array dtype.

        """

        return self._write_op(self._require_dataset_nosync, name, shape=shape,
                              dtype=dtype, exact=exact, **kwargs)

    def _require_dataset_nosync(self, name, shape, dtype=None, exact=False,
                                **kwargs):

        path = self._item_path(name)

        if contains_array(self._store, path):
            synchronizer = kwargs.get('synchronizer', self._synchronizer)
            cache_metadata = kwargs.get('cache_metadata', True)
            a = Array(self._store, path=path, read_only=self._read_only,
                      chunk_store=self._chunk_store,
                      synchronizer=synchronizer, cache_metadata=cache_metadata)
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
            return self._create_dataset_nosync(name, shape=shape, dtype=dtype,
                                               **kwargs)

    def create(self, name, **kwargs):
        """Create an array. Keyword arguments as per
        :func:`zarr.creation.create`."""
        return self._write_op(self._create_nosync, name, **kwargs)

    def _create_nosync(self, name, **kwargs):
        path = self._item_path(name)
        kwargs.setdefault('synchronizer', self._synchronizer)
        return create(store=self._store, path=path,
                      chunk_store=self._chunk_store, **kwargs)

    def empty(self, name, **kwargs):
        """Create an array. Keyword arguments as per
        :func:`zarr.creation.empty`."""
        return self._write_op(self._empty_nosync, name, **kwargs)

    def _empty_nosync(self, name, **kwargs):
        path = self._item_path(name)
        kwargs.setdefault('synchronizer', self._synchronizer)
        return empty(store=self._store, path=path,
                     chunk_store=self._chunk_store, **kwargs)

    def zeros(self, name, **kwargs):
        """Create an array. Keyword arguments as per
        :func:`zarr.creation.zeros`."""
        return self._write_op(self._zeros_nosync, name, **kwargs)

    def _zeros_nosync(self, name, **kwargs):
        path = self._item_path(name)
        kwargs.setdefault('synchronizer', self._synchronizer)
        return zeros(store=self._store, path=path,
                     chunk_store=self._chunk_store, **kwargs)

    def ones(self, name, **kwargs):
        """Create an array. Keyword arguments as per
        :func:`zarr.creation.ones`."""
        return self._write_op(self._ones_nosync, name, **kwargs)

    def _ones_nosync(self, name, **kwargs):
        path = self._item_path(name)
        kwargs.setdefault('synchronizer', self._synchronizer)
        return ones(store=self._store, path=path,
                    chunk_store=self._chunk_store, **kwargs)

    def full(self, name, fill_value, **kwargs):
        """Create an array. Keyword arguments as per
        :func:`zarr.creation.full`."""
        return self._write_op(self._full_nosync, name, fill_value, **kwargs)

    def _full_nosync(self, name, fill_value, **kwargs):
        path = self._item_path(name)
        kwargs.setdefault('synchronizer', self._synchronizer)
        return full(store=self._store, path=path,
                    chunk_store=self._chunk_store,
                    fill_value=fill_value, **kwargs)

    def array(self, name, data, **kwargs):
        """Create an array. Keyword arguments as per
        :func:`zarr.creation.array`."""
        return self._write_op(self._array_nosync, name, data, **kwargs)

    def _array_nosync(self, name, data, **kwargs):
        path = self._item_path(name)
        kwargs.setdefault('synchronizer', self._synchronizer)
        return array(data, store=self._store, path=path,
                     chunk_store=self._chunk_store, **kwargs)

    def empty_like(self, name, data, **kwargs):
        """Create an array. Keyword arguments as per
        :func:`zarr.creation.empty_like`."""
        return self._write_op(self._empty_like_nosync, name, data, **kwargs)

    def _empty_like_nosync(self, name, data, **kwargs):
        path = self._item_path(name)
        kwargs.setdefault('synchronizer', self._synchronizer)
        return empty_like(data, store=self._store, path=path,
                          chunk_store=self._chunk_store, **kwargs)

    def zeros_like(self, name, data, **kwargs):
        """Create an array. Keyword arguments as per
        :func:`zarr.creation.zeros_like`."""
        return self._write_op(self._zeros_like_nosync, name, data, **kwargs)

    def _zeros_like_nosync(self, name, data, **kwargs):
        path = self._item_path(name)
        kwargs.setdefault('synchronizer', self._synchronizer)
        return zeros_like(data, store=self._store, path=path,
                          chunk_store=self._chunk_store, **kwargs)

    def ones_like(self, name, data, **kwargs):
        """Create an array. Keyword arguments as per
        :func:`zarr.creation.ones_like`."""
        return self._write_op(self._ones_like_nosync, name, data, **kwargs)

    def _ones_like_nosync(self, name, data, **kwargs):
        path = self._item_path(name)
        kwargs.setdefault('synchronizer', self._synchronizer)
        return ones_like(data, store=self._store, path=path,
                         chunk_store=self._chunk_store, **kwargs)

    def full_like(self, name, data, **kwargs):
        """Create an array. Keyword arguments as per
        :func:`zarr.creation.full_like`."""
        return self._write_op(self._full_like_nosync, name, data, **kwargs)

    def _full_like_nosync(self, name, data, **kwargs):
        path = self._item_path(name)
        kwargs.setdefault('synchronizer', self._synchronizer)
        return full_like(data, store=self._store, path=path,
                         chunk_store=self._chunk_store, **kwargs)


def _handle_store_arg(store):
    if store is None:
        return DictStore()
    elif isinstance(store, str):
        return DirectoryStore(store)
    else:
        return store


def group(store=None, overwrite=False, chunk_store=None, synchronizer=None,
          path=None):
    """Create a group.

    Parameters
    ----------
    store : MutableMapping or string
        Store or path to directory in file system.
    overwrite : bool, optional
        If True, delete any pre-existing data in `store` at `path` before
        creating the group.
    chunk_store : MutableMapping, optional
        Separate storage for chunks. If not provided, `store` will be used
        for storage of both chunks and metadata.
    synchronizer : object, optional
        Array synchronizer.
    path : string, optional
        Group path.

    Returns
    -------
    g : zarr.hierarchy.Group

    Examples
    --------

    Create a group in memory::

        >>> import zarr
        >>> g = zarr.group()
        >>> g
        Group(/, 0)
          store: DictStore

    Create a group with a different store::

        >>> store = zarr.DirectoryStore('example')
        >>> g = zarr.group(store=store, overwrite=True)
        >>> g
        Group(/, 0)
          store: DirectoryStore

    """

    # handle polymorphic store arg
    store = _handle_store_arg(store)
    path = normalize_storage_path(path)

    # require group
    if overwrite or not contains_group(store):
        init_group(store, overwrite=overwrite, chunk_store=chunk_store,
                   path=path)

    return Group(store, read_only=False, chunk_store=chunk_store,
                 synchronizer=synchronizer, path=path)


def open_group(store=None, mode='a', synchronizer=None, path=None):
    """Open a group using mode-like semantics.

    Parameters
    ----------
    store : MutableMapping or string
        Store or path to directory in file system.
    mode : {'r', 'r+', 'a', 'w', 'w-'}
        Persistence mode: 'r' means read only (must exist); 'r+' means
        read/write (must exist); 'a' means read/write (create if doesn't
        exist); 'w' means create (overwrite if exists); 'w-' means create
        (fail if exists).
    synchronizer : object, optional
        Array synchronizer.
    path : string, optional
        Group path.

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
    Group(/, 2)
      groups: 2; bar, foo
      store: DirectoryStore
    >>> root2 = zarr.open_group('example', mode='a')
    >>> root2
    Group(/, 2)
      groups: 2; bar, foo
      store: DirectoryStore
    >>> root == root2
    True

    """

    # handle polymorphic store arg
    store = _handle_store_arg(store)
    path = normalize_storage_path(path)

    # ensure store is initialized

    if mode in ['r', 'r+']:
        if contains_array(store, path=path):
            err_contains_array(path)
        elif not contains_group(store, path=path):
            err_group_not_found(path)

    elif mode == 'w':
        init_group(store, overwrite=True, path=path)

    elif mode == 'a':
        if contains_array(store, path=path):
            err_contains_array(path)
        if not contains_group(store, path=path):
            init_group(store, path=path)

    elif mode in ['w-', 'x']:
        if contains_array(store, path=path):
            err_contains_array(path)
        elif contains_group(store, path=path):
            err_contains_group(path)
        else:
            init_group(store, path=path)

    # determine read only status
    read_only = mode == 'r'

    return Group(store, read_only=read_only, synchronizer=synchronizer,
                 path=path)
