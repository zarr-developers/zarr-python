from collections.abc import MutableMapping
from itertools import islice

import numpy as np

from zarr._storage.store import _get_metadata_suffix, data_root, meta_root, DEFAULT_ZARR_VERSION
from zarr.attrs import Attributes
from zarr.core import Array
from zarr.creation import (array, create, empty, empty_like, full, full_like,
                           ones, ones_like, zeros, zeros_like)
from zarr.errors import (
    ContainsArrayError,
    ContainsGroupError,
    GroupNotFoundError,
    ReadOnlyError,
)
from zarr.storage import (
    _get_hierarchy_metadata,
    _prefix_to_group_key,
    BaseStore,
    MemoryStore,
    attrs_key,
    contains_array,
    contains_group,
    group_meta_key,
    init_group,
    listdir,
    normalize_store_arg,
    rename,
    rmdir,
)
from zarr.storage_v3 import MemoryStoreV3
from zarr.util import (
    InfoReporter,
    TreeViewer,
    is_valid_python_name,
    nolock,
    normalize_shape,
    normalize_storage_path,
)


class Group(MutableMapping):
    """Instantiate a group from an initialized store.

    Parameters
    ----------
    store : MutableMapping
        Group store, already initialized.
        If the Group is used in a context manager, and the store has a ``close`` method,
        it will be called on exit.
    path : string, optional
        Group path.
    read_only : bool, optional
        True if group should be protected against modification.
    chunk_store : MutableMapping, optional
        Separate storage for chunks. If not provided, `store` will be used
        for storage of both chunks and metadata.
    cache_attrs : bool, optional
        If True (default), user attributes will be cached for attribute read
        operations. If False, user attributes are reloaded from the store prior
        to all attribute read operations.
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
    info

    Methods
    -------
    __len__
    __iter__
    __contains__
    __getitem__
    __enter__
    __exit__
    group_keys
    groups
    array_keys
    arrays
    visit
    visitkeys
    visitvalues
    visititems
    tree
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
    info
    move

    """

    def __init__(self, store, path=None, read_only=False, chunk_store=None,
                 cache_attrs=True, synchronizer=None, zarr_version=None):
        store: BaseStore = _normalize_store_arg(store, zarr_version=zarr_version)
        if zarr_version is None:
            zarr_version = getattr(store, '_store_version', DEFAULT_ZARR_VERSION)
        if chunk_store is not None:
            chunk_store: BaseStore = _normalize_store_arg(chunk_store, zarr_version=zarr_version)
        self._store = store
        self._chunk_store = chunk_store
        self._path = normalize_storage_path(path)
        if self._path:
            self._key_prefix = self._path + '/'
        else:
            self._key_prefix = ''
        self._read_only = read_only
        self._synchronizer = synchronizer
        self._version = zarr_version

        if self._version == 3:
            self._data_key_prefix = data_root + self._key_prefix
            self._data_path = data_root + self._path
            self._hierarchy_metadata = _get_hierarchy_metadata(store=self._store)
            self._metadata_key_suffix = _get_metadata_suffix(store=self._store)

        # guard conditions
        if contains_array(store, path=self._path):
            raise ContainsArrayError(path)

        # initialize metadata
        try:
            mkey = _prefix_to_group_key(self._store, self._key_prefix)
            assert not mkey.endswith("root/.group")
            meta_bytes = store[mkey]
        except KeyError:
            if self._version == 2:
                raise GroupNotFoundError(path)
            else:
                implicit_prefix = meta_root + self._key_prefix
                if self._store.list_prefix(implicit_prefix):
                    # implicit group does not have any metadata
                    self._meta = None
                else:
                    raise GroupNotFoundError(path)
        else:
            self._meta = self._store._metadata_class.decode_group_metadata(meta_bytes)

        # setup attributes
        if self._version == 2:
            akey = self._key_prefix + attrs_key
        else:
            # Note: mkey doesn't actually exist for implicit groups, but the
            # object can still be created.
            akey = mkey
        self._attrs = Attributes(store, key=akey, read_only=read_only,
                                 cache=cache_attrs, synchronizer=synchronizer)

        # setup info
        self._info = InfoReporter(self)

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
    def basename(self):
        """Final component of name."""
        return self.name.split('/')[-1]

    @property
    def read_only(self):
        """A boolean, True if modification operations are not permitted."""
        return self._read_only

    @property
    def chunk_store(self):
        """A MutableMapping providing the underlying storage for array chunks."""
        if self._chunk_store is None:
            return self._store
        else:
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

    @property
    def info(self):
        """Return diagnostic information about the group."""
        return self._info

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
        if getattr(self._store, '_store_version', 2) == 2:
            for key in sorted(listdir(self._store, self._path)):
                path = self._key_prefix + key
                if (contains_array(self._store, path) or
                        contains_group(self._store, path)):
                    yield key
        else:
            # TODO: Should this iterate over data folders and/or metadata
            #       folders and/or metadata files

            dir_path = meta_root + self._key_prefix
            name_start = len(dir_path)
            keys, prefixes = self._store.list_dir(dir_path)

            # yield any groups or arrays
            sfx = self._metadata_key_suffix
            for key in keys:
                len_suffix = len('.group') + len(sfx)  # same for .array
                if key.endswith(('.group' + sfx, '.array' + sfx)):
                    yield key[name_start:-len_suffix]

            # also yield any implicit groups
            for prefix in prefixes:
                prefix = prefix.rstrip('/')
                # only implicit if there is no .group.sfx file
                if not prefix + '.group' + sfx in self._store:
                    yield prefix[name_start:]

            # Note: omit data/root/ to avoid duplicate listings
            #       any group in data/root/ must has an entry in meta/root/

    def __len__(self):
        """Number of members."""
        return sum(1 for _ in self)

    def __repr__(self):
        t = type(self)
        r = '<{}.{}'.format(t.__module__, t.__name__)
        if self.name:
            r += ' %r' % self.name
        if self._read_only:
            r += ' read-only'
        r += '>'
        return r

    def __enter__(self):
        """Return the Group for use as a context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Call the close method of the underlying Store."""
        self.store.close()

    def info_items(self):

        def typestr(o):
            return '{}.{}'.format(type(o).__module__, type(o).__name__)

        items = []

        # basic info
        if self.name is not None:
            items += [('Name', self.name)]
        items += [
            ('Type', typestr(self)),
            ('Read-only', str(self.read_only)),
        ]

        # synchronizer
        if self._synchronizer is not None:
            items += [('Synchronizer type', typestr(self._synchronizer))]

        # storage info
        items += [('Store type', typestr(self._store))]
        if self._chunk_store is not None:
            items += [('Chunk store type', typestr(self._chunk_store))]

        # members
        items += [('No. members', len(self))]
        array_keys = sorted(self.array_keys())
        group_keys = sorted(self.group_keys())
        items += [('No. arrays', len(array_keys))]
        items += [('No. groups', len(group_keys))]
        if array_keys:
            items += [('Arrays', ', '.join(array_keys))]
        if group_keys:
            items += [('Groups', ', '.join(group_keys))]

        return items

    def __getstate__(self):
        return (self._store, self._path, self._read_only, self._chunk_store,
                self.attrs.cache, self._synchronizer)

    def __setstate__(self, state):
        self.__init__(*state)

    def _item_path(self, item):
        absolute = isinstance(item, str) and item and item[0] == '/'
        path = normalize_storage_path(item)
        if not absolute and self._path:
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
            contains_group(self._store, path, explicit_only=False)

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
        <zarr.hierarchy.Group '/foo'>
        >>> g1['foo/bar']
        <zarr.hierarchy.Group '/foo/bar'>
        >>> g1['foo/bar/baz']
        <zarr.core.Array '/foo/bar/baz' (100,) float64>

        """
        path = self._item_path(item)
        if contains_array(self._store, path):
            return Array(self._store, read_only=self._read_only, path=path,
                         chunk_store=self._chunk_store,
                         synchronizer=self._synchronizer, cache_attrs=self.attrs.cache,
                         zarr_version=self._version)
        elif contains_group(self._store, path, explicit_only=True):
            return Group(self._store, read_only=self._read_only, path=path,
                         chunk_store=self._chunk_store, cache_attrs=self.attrs.cache,
                         synchronizer=self._synchronizer, zarr_version=self._version)
        elif self._version == 3:
            implicit_group = meta_root + path + '/'
            # non-empty folder in the metadata path implies an implicit group
            if self._store.list_prefix(implicit_group):
                return Group(self._store, read_only=self._read_only, path=path,
                             chunk_store=self._chunk_store, cache_attrs=self.attrs.cache,
                             synchronizer=self._synchronizer, zarr_version=self._version)
            else:
                raise KeyError(item)
        else:
            raise KeyError(item)

    def __setitem__(self, item, value):
        self.array(item, value, overwrite=True)

    def __delitem__(self, item):
        return self._write_op(self._delitem_nosync, item)

    def _delitem_nosync(self, item):
        path = self._item_path(item)
        if contains_array(self._store, path) or \
                contains_group(self._store, path, explicit_only=False):
            rmdir(self._store, path)
        else:
            raise KeyError(item)

    def __getattr__(self, item):
        # allow access to group members via dot notation
        try:
            return self.__getitem__(item)
        except KeyError:
            raise AttributeError

    def __dir__(self):
        # noinspection PyUnresolvedReferences
        base = super().__dir__()
        keys = sorted(set(base + list(self)))
        keys = [k for k in keys if is_valid_python_name(k)]
        return keys

    def _ipython_key_completions_(self):
        return sorted(self)

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
        if self._version == 2:
            for key in sorted(listdir(self._store, self._path)):
                path = self._key_prefix + key
                if contains_group(self._store, path):
                    yield key
        else:
            dir_name = meta_root + self._path
            group_sfx = '.group' + self._metadata_key_suffix
            for key in sorted(listdir(self._store, dir_name)):
                if key.endswith(group_sfx):
                    key = key[:-len(group_sfx)]
                path = self._key_prefix + key
                if path.endswith(".array" + self._metadata_key_suffix):
                    # skip array keys
                    continue
                if contains_group(self._store, path, explicit_only=False):
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
        if self._version == 2:
            for key in sorted(listdir(self._store, self._path)):
                path = self._key_prefix + key
                if contains_group(self._store, path, explicit_only=False):
                    yield key, Group(
                        self._store,
                        path=path,
                        read_only=self._read_only,
                        chunk_store=self._chunk_store,
                        cache_attrs=self.attrs.cache,
                        synchronizer=self._synchronizer,
                        zarr_version=self._version)

        else:
            dir_name = meta_root + self._path
            group_sfx = '.group' + self._metadata_key_suffix
            for key in sorted(listdir(self._store, dir_name)):
                if key.endswith(group_sfx):
                    key = key[:-len(group_sfx)]
                path = self._key_prefix + key
                if path.endswith(".array" + self._metadata_key_suffix):
                    # skip array keys
                    continue
                if contains_group(self._store, path, explicit_only=False):
                    yield key, Group(
                        self._store,
                        path=path,
                        read_only=self._read_only,
                        chunk_store=self._chunk_store,
                        cache_attrs=self.attrs.cache,
                        synchronizer=self._synchronizer,
                        zarr_version=self._version)

    def array_keys(self, recurse=False):
        """Return an iterator over member names for arrays only.

        Parameters
        ----------
        recurse : recurse, optional
            Option to return member names for all arrays, even from groups
            below the current one. If False, only member names for arrays in
            the current group will be returned. Default value is False.

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
        return self._array_iter(keys_only=True,
                                method='array_keys',
                                recurse=recurse)

    def arrays(self, recurse=False):
        """Return an iterator over (name, value) pairs for arrays only.

        Parameters
        ----------
        recurse : recurse, optional
            Option to return (name, value) pairs for all arrays, even from groups
            below the current one. If False, only (name, value) pairs for arrays in
            the current group will be returned. Default value is False.

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
        return self._array_iter(keys_only=False,
                                method='arrays',
                                recurse=recurse)

    def _array_iter(self, keys_only, method, recurse):
        if self._version == 2:
            for key in sorted(listdir(self._store, self._path)):
                path = self._key_prefix + key
                assert not path.startswith("meta")
                if contains_array(self._store, path):
                    _key = key.rstrip("/")
                    yield _key if keys_only else (_key, self[key])
                elif recurse and contains_group(self._store, path):
                    group = self[key]
                    for i in getattr(group, method)(recurse=recurse):
                        yield i
        else:
            dir_name = meta_root + self._path
            array_sfx = '.array' + self._metadata_key_suffix
            for key in sorted(listdir(self._store, dir_name)):
                if key.endswith(array_sfx):
                    key = key[:-len(array_sfx)]
                path = self._key_prefix + key
                assert not path.startswith("meta")
                if key.endswith('.group' + self._metadata_key_suffix):
                    # skip group metadata keys
                    continue
                if contains_array(self._store, path):
                    _key = key.rstrip("/")
                    yield _key if keys_only else (_key, self[key])
                elif recurse and contains_group(self._store, path):
                    group = self[key]
                    for i in getattr(group, method)(recurse=recurse):
                        yield i

    def visitvalues(self, func):
        """Run ``func`` on each object.

        Note: If ``func`` returns ``None`` (or doesn't return),
              iteration continues. However, if ``func`` returns
              anything else, it ceases and returns that value.

        Examples
        --------
        >>> import zarr
        >>> g1 = zarr.group()
        >>> g2 = g1.create_group('foo')
        >>> g3 = g1.create_group('bar')
        >>> g4 = g3.create_group('baz')
        >>> g5 = g3.create_group('quux')
        >>> def print_visitor(obj):
        ...     print(obj)
        >>> g1.visitvalues(print_visitor)
        <zarr.hierarchy.Group '/bar'>
        <zarr.hierarchy.Group '/bar/baz'>
        <zarr.hierarchy.Group '/bar/quux'>
        <zarr.hierarchy.Group '/foo'>
        >>> g3.visitvalues(print_visitor)
        <zarr.hierarchy.Group '/bar/baz'>
        <zarr.hierarchy.Group '/bar/quux'>

        """

        def _visit(obj):
            yield obj
            keys = sorted(getattr(obj, "keys", lambda: [])())
            for k in keys:
                for v in _visit(obj[k]):
                    yield v

        for each_obj in islice(_visit(self), 1, None):
            value = func(each_obj)
            if value is not None:
                return value

    def visit(self, func):
        """Run ``func`` on each object's path.

        Note: If ``func`` returns ``None`` (or doesn't return),
              iteration continues. However, if ``func`` returns
              anything else, it ceases and returns that value.

        Examples
        --------
        >>> import zarr
        >>> g1 = zarr.group()
        >>> g2 = g1.create_group('foo')
        >>> g3 = g1.create_group('bar')
        >>> g4 = g3.create_group('baz')
        >>> g5 = g3.create_group('quux')
        >>> def print_visitor(name):
        ...     print(name)
        >>> g1.visit(print_visitor)
        bar
        bar/baz
        bar/quux
        foo
        >>> g3.visit(print_visitor)
        baz
        quux

        """

        base_len = len(self.name)
        return self.visitvalues(lambda o: func(o.name[base_len:].lstrip("/")))

    def visitkeys(self, func):
        """An alias for :py:meth:`~Group.visit`.
        """

        return self.visit(func)

    def visititems(self, func):
        """Run ``func`` on each object's path and the object itself.

        Note: If ``func`` returns ``None`` (or doesn't return),
              iteration continues. However, if ``func`` returns
              anything else, it ceases and returns that value.

        Examples
        --------
        >>> import zarr
        >>> g1 = zarr.group()
        >>> g2 = g1.create_group('foo')
        >>> g3 = g1.create_group('bar')
        >>> g4 = g3.create_group('baz')
        >>> g5 = g3.create_group('quux')
        >>> def print_visitor(name, obj):
        ...     print((name, obj))
        >>> g1.visititems(print_visitor)
        ('bar', <zarr.hierarchy.Group '/bar'>)
        ('bar/baz', <zarr.hierarchy.Group '/bar/baz'>)
        ('bar/quux', <zarr.hierarchy.Group '/bar/quux'>)
        ('foo', <zarr.hierarchy.Group '/foo'>)
        >>> g3.visititems(print_visitor)
        ('baz', <zarr.hierarchy.Group '/bar/baz'>)
        ('quux', <zarr.hierarchy.Group '/bar/quux'>)

        """

        base_len = len(self.name)
        return self.visitvalues(lambda o: func(o.name[base_len:].lstrip("/"), o))

    def tree(self, expand=False, level=None):
        """Provide a ``print``-able display of the hierarchy.

        Parameters
        ----------
        expand : bool, optional
            Only relevant for HTML representation. If True, tree will be fully expanded.
        level : int, optional
            Maximum depth to descend into hierarchy.

        Examples
        --------
        >>> import zarr
        >>> g1 = zarr.group()
        >>> g2 = g1.create_group('foo')
        >>> g3 = g1.create_group('bar')
        >>> g4 = g3.create_group('baz')
        >>> g5 = g3.create_group('quux')
        >>> d1 = g5.create_dataset('baz', shape=100, chunks=10)
        >>> g1.tree()
        /
         ├── bar
         │   ├── baz
         │   └── quux
         │       └── baz (100,) float64
         └── foo
        >>> g1.tree(level=2)
        /
         ├── bar
         │   ├── baz
         │   └── quux
         └── foo
        >>> g3.tree()
        bar
         ├── baz
         └── quux
             └── baz (100,) float64

        Notes
        -----
        Please note that this is an experimental feature. The behaviour of this
        function is still evolving and the default output and/or parameters may change
        in future versions.

        """

        return TreeViewer(self, expand=expand, level=level)

    def _write_op(self, f, *args, **kwargs):

        # guard condition
        if self._read_only:
            raise ReadOnlyError()

        if self._synchronizer is None:
            # no synchronization
            lock = nolock
        else:
            # synchronize on the root group
            lock = self._synchronizer[group_meta_key]

        with lock:
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

        return self._write_op(self._create_group_nosync, name, overwrite=overwrite)

    def _create_group_nosync(self, name, overwrite=False):
        path = self._item_path(name)

        # create terminal group
        init_group(self._store, path=path, chunk_store=self._chunk_store,
                   overwrite=overwrite)

        return Group(self._store, path=path, read_only=self._read_only,
                     chunk_store=self._chunk_store, cache_attrs=self.attrs.cache,
                     synchronizer=self._synchronizer, zarr_version=self._version)

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
            init_group(store=self._store, path=path, chunk_store=self._chunk_store,
                       overwrite=overwrite)

        return Group(self._store, path=path, read_only=self._read_only,
                     chunk_store=self._chunk_store, cache_attrs=self.attrs.cache,
                     synchronizer=self._synchronizer, zarr_version=self._version)

    def require_groups(self, *names):
        """Convenience method to require multiple groups in a single call."""
        return tuple(self.require_group(name) for name in names)

    # noinspection PyIncorrectDocstring
    def create_dataset(self, name, **kwargs):
        """Create an array.

        Arrays are known as "datasets" in HDF5 terminology. For compatibility
        with h5py, Zarr groups also implement the require_dataset() method.

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
        dimension_separator : {'.', '/'}, optional
            Separator placed between the dimensions of a chunk.

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
        <zarr.core.Array '/foo' (10000, 10000) float64>
        >>> d2 = g1.create_dataset('bar/baz/qux', shape=(100, 100, 100),
        ...                        chunks=(100, 10, 10))
        >>> d2
        <zarr.core.Array '/bar/baz/qux' (100, 100, 100) float64>

        """
        assert "mode" not in kwargs

        return self._write_op(self._create_dataset_nosync, name, **kwargs)

    def _create_dataset_nosync(self, name, data=None, **kwargs):

        assert "mode" not in kwargs
        path = self._item_path(name)

        # determine synchronizer
        kwargs.setdefault('synchronizer', self._synchronizer)
        kwargs.setdefault('cache_attrs', self.attrs.cache)

        # create array
        if data is None:
            a = create(store=self._store, path=path, chunk_store=self._chunk_store,
                       **kwargs)

        else:
            a = array(data, store=self._store, path=path, chunk_store=self._chunk_store,
                      **kwargs)

        return a

    def require_dataset(self, name, shape, dtype=None, exact=False, **kwargs):
        """Obtain an array, creating if it doesn't exist.

        Arrays are known as "datasets" in HDF5 terminology. For compatibility
        with h5py, Zarr groups also implement the create_dataset() method.

        Other `kwargs` are as per :func:`zarr.hierarchy.Group.create_dataset`.

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

            # array already exists at path, validate that it is the right shape and type

            synchronizer = kwargs.get('synchronizer', self._synchronizer)
            cache_metadata = kwargs.get('cache_metadata', True)
            cache_attrs = kwargs.get('cache_attrs', self.attrs.cache)
            a = Array(self._store, path=path, read_only=self._read_only,
                      chunk_store=self._chunk_store, synchronizer=synchronizer,
                      cache_metadata=cache_metadata, cache_attrs=cache_attrs)
            shape = normalize_shape(shape)
            if shape != a.shape:
                raise TypeError('shape do not match existing array; expected {}, got {}'
                                .format(a.shape, shape))
            dtype = np.dtype(dtype)
            if exact:
                if dtype != a.dtype:
                    raise TypeError('dtypes do not match exactly; expected {}, got {}'
                                    .format(a.dtype, dtype))
            else:
                if not np.can_cast(dtype, a.dtype):
                    raise TypeError('dtypes ({}, {}) cannot be safely cast'
                                    .format(dtype, a.dtype))
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
        kwargs.setdefault('cache_attrs', self.attrs.cache)
        return create(store=self._store, path=path, chunk_store=self._chunk_store,
                      **kwargs)

    def empty(self, name, **kwargs):
        """Create an array. Keyword arguments as per
        :func:`zarr.creation.empty`."""
        return self._write_op(self._empty_nosync, name, **kwargs)

    def _empty_nosync(self, name, **kwargs):
        path = self._item_path(name)
        kwargs.setdefault('synchronizer', self._synchronizer)
        kwargs.setdefault('cache_attrs', self.attrs.cache)
        return empty(store=self._store, path=path, chunk_store=self._chunk_store,
                     **kwargs)

    def zeros(self, name, **kwargs):
        """Create an array. Keyword arguments as per
        :func:`zarr.creation.zeros`."""
        return self._write_op(self._zeros_nosync, name, **kwargs)

    def _zeros_nosync(self, name, **kwargs):
        path = self._item_path(name)
        kwargs.setdefault('synchronizer', self._synchronizer)
        kwargs.setdefault('cache_attrs', self.attrs.cache)
        return zeros(store=self._store, path=path, chunk_store=self._chunk_store,
                     **kwargs)

    def ones(self, name, **kwargs):
        """Create an array. Keyword arguments as per
        :func:`zarr.creation.ones`."""
        return self._write_op(self._ones_nosync, name, **kwargs)

    def _ones_nosync(self, name, **kwargs):
        path = self._item_path(name)
        kwargs.setdefault('synchronizer', self._synchronizer)
        kwargs.setdefault('cache_attrs', self.attrs.cache)
        return ones(store=self._store, path=path, chunk_store=self._chunk_store, **kwargs)

    def full(self, name, fill_value, **kwargs):
        """Create an array. Keyword arguments as per
        :func:`zarr.creation.full`."""
        return self._write_op(self._full_nosync, name, fill_value, **kwargs)

    def _full_nosync(self, name, fill_value, **kwargs):
        path = self._item_path(name)
        kwargs.setdefault('synchronizer', self._synchronizer)
        kwargs.setdefault('cache_attrs', self.attrs.cache)
        return full(store=self._store, path=path, chunk_store=self._chunk_store,
                    fill_value=fill_value, **kwargs)

    def array(self, name, data, **kwargs):
        """Create an array. Keyword arguments as per
        :func:`zarr.creation.array`."""
        return self._write_op(self._array_nosync, name, data, **kwargs)

    def _array_nosync(self, name, data, **kwargs):
        path = self._item_path(name)
        kwargs.setdefault('synchronizer', self._synchronizer)
        kwargs.setdefault('cache_attrs', self.attrs.cache)
        return array(data, store=self._store, path=path, chunk_store=self._chunk_store,
                     **kwargs)

    def empty_like(self, name, data, **kwargs):
        """Create an array. Keyword arguments as per
        :func:`zarr.creation.empty_like`."""
        return self._write_op(self._empty_like_nosync, name, data, **kwargs)

    def _empty_like_nosync(self, name, data, **kwargs):
        path = self._item_path(name)
        kwargs.setdefault('synchronizer', self._synchronizer)
        kwargs.setdefault('cache_attrs', self.attrs.cache)
        return empty_like(data, store=self._store, path=path,
                          chunk_store=self._chunk_store, **kwargs)

    def zeros_like(self, name, data, **kwargs):
        """Create an array. Keyword arguments as per
        :func:`zarr.creation.zeros_like`."""
        return self._write_op(self._zeros_like_nosync, name, data, **kwargs)

    def _zeros_like_nosync(self, name, data, **kwargs):
        path = self._item_path(name)
        kwargs.setdefault('synchronizer', self._synchronizer)
        kwargs.setdefault('cache_attrs', self.attrs.cache)
        return zeros_like(data, store=self._store, path=path,
                          chunk_store=self._chunk_store, **kwargs)

    def ones_like(self, name, data, **kwargs):
        """Create an array. Keyword arguments as per
        :func:`zarr.creation.ones_like`."""
        return self._write_op(self._ones_like_nosync, name, data, **kwargs)

    def _ones_like_nosync(self, name, data, **kwargs):
        path = self._item_path(name)
        kwargs.setdefault('synchronizer', self._synchronizer)
        kwargs.setdefault('cache_attrs', self.attrs.cache)
        return ones_like(data, store=self._store, path=path,
                         chunk_store=self._chunk_store, **kwargs)

    def full_like(self, name, data, **kwargs):
        """Create an array. Keyword arguments as per
        :func:`zarr.creation.full_like`."""
        return self._write_op(self._full_like_nosync, name, data, **kwargs)

    def _full_like_nosync(self, name, data, **kwargs):
        path = self._item_path(name)
        kwargs.setdefault('synchronizer', self._synchronizer)
        kwargs.setdefault('cache_attrs', self.attrs.cache)
        return full_like(data, store=self._store, path=path,
                         chunk_store=self._chunk_store, **kwargs)

    def _move_nosync(self, path, new_path):
        rename(self._store, path, new_path)
        if self._chunk_store is not None:
            rename(self._chunk_store, path, new_path)

    def move(self, source, dest):
        """Move contents from one path to another relative to the Group.

        Parameters
        ----------
        source : string
            Name or path to a Zarr object to move.
        dest : string
            New name or path of the Zarr object.
        """

        source = self._item_path(source)
        dest = self._item_path(dest)

        # Check that source exists.
        if not (contains_array(self._store, source) or
                contains_group(self._store, source, explicit_only=False)):
            raise ValueError('The source, "%s", does not exist.' % source)
        if (contains_array(self._store, dest) or
                contains_group(self._store, dest, explicit_only=False)):
            raise ValueError('The dest, "%s", already exists.' % dest)

        # Ensure groups needed for `dest` exist.
        if "/" in dest:
            self.require_group("/" + dest.rsplit("/", 1)[0])

        self._write_op(self._move_nosync, source, dest)


def _normalize_store_arg(store, *, storage_options=None, mode="r",
                         zarr_version=None):
    if zarr_version is None:
        zarr_version = getattr(store, '_store_version', DEFAULT_ZARR_VERSION)
    if store is None:
        return MemoryStore() if zarr_version == 2 else MemoryStoreV3()
    return normalize_store_arg(store,
                               storage_options=storage_options, mode=mode,
                               zarr_version=zarr_version)


def group(store=None, overwrite=False, chunk_store=None,
          cache_attrs=True, synchronizer=None, path=None, *, zarr_version=None):
    """Create a group.

    Parameters
    ----------
    store : MutableMapping or string, optional
        Store or path to directory in file system.
    overwrite : bool, optional
        If True, delete any pre-existing data in `store` at `path` before
        creating the group.
    chunk_store : MutableMapping, optional
        Separate storage for chunks. If not provided, `store` will be used
        for storage of both chunks and metadata.
    cache_attrs : bool, optional
        If True (default), user attributes will be cached for attribute read
        operations. If False, user attributes are reloaded from the store prior
        to all attribute read operations.
    synchronizer : object, optional
        Array synchronizer.
    path : string, optional
        Group path within store.

    Returns
    -------
    g : zarr.hierarchy.Group

    Examples
    --------
    Create a group in memory::

        >>> import zarr
        >>> g = zarr.group()
        >>> g
        <zarr.hierarchy.Group '/'>

    Create a group with a different store::

        >>> store = zarr.DirectoryStore('data/example.zarr')
        >>> g = zarr.group(store=store, overwrite=True)
        >>> g
        <zarr.hierarchy.Group '/'>

    """

    # handle polymorphic store arg
    store = _normalize_store_arg(store, zarr_version=zarr_version)
    if zarr_version is None:
        zarr_version = getattr(store, '_store_version', DEFAULT_ZARR_VERSION)
    if zarr_version == 3 and path is None:
        raise ValueError(f"path must be provided for a v{zarr_version} group")
    path = normalize_storage_path(path)

    if zarr_version == 2:
        requires_init = overwrite or not contains_group(store)
    elif zarr_version == 3:
        requires_init = overwrite or not contains_group(store, path)

    if requires_init:
        init_group(store, overwrite=overwrite, chunk_store=chunk_store,
                   path=path)

    return Group(store, read_only=False, chunk_store=chunk_store,
                 cache_attrs=cache_attrs, synchronizer=synchronizer, path=path,
                 zarr_version=zarr_version)


def open_group(store=None, mode='a', cache_attrs=True, synchronizer=None, path=None,
               chunk_store=None, storage_options=None, *, zarr_version=None):
    """Open a group using file-mode-like semantics.

    Parameters
    ----------
    store : MutableMapping or string, optional
        Store or path to directory in file system or name of zip file.
    mode : {'r', 'r+', 'a', 'w', 'w-'}, optional
        Persistence mode: 'r' means read only (must exist); 'r+' means
        read/write (must exist); 'a' means read/write (create if doesn't
        exist); 'w' means create (overwrite if exists); 'w-' means create
        (fail if exists).
    cache_attrs : bool, optional
        If True (default), user attributes will be cached for attribute read
        operations. If False, user attributes are reloaded from the store prior
        to all attribute read operations.
    synchronizer : object, optional
        Array synchronizer.
    path : string, optional
        Group path within store.
    chunk_store : MutableMapping or string, optional
        Store or path to directory in file system or name of zip file.
    storage_options : dict
        If using an fsspec URL to create the store, these will be passed to
        the backend implementation. Ignored otherwise.

    Returns
    -------
    g : zarr.hierarchy.Group

    Examples
    --------
    >>> import zarr
    >>> root = zarr.open_group('data/example.zarr', mode='w')
    >>> foo = root.create_group('foo')
    >>> bar = root.create_group('bar')
    >>> root
    <zarr.hierarchy.Group '/'>
    >>> root2 = zarr.open_group('data/example.zarr', mode='a')
    >>> root2
    <zarr.hierarchy.Group '/'>
    >>> root == root2
    True

    """

    # handle polymorphic store arg
    store = _normalize_store_arg(
        store, storage_options=storage_options, mode=mode,
        zarr_version=zarr_version)
    if zarr_version is None:
        zarr_version = getattr(store, '_store_version', DEFAULT_ZARR_VERSION)
    if chunk_store is not None:
        chunk_store = _normalize_store_arg(chunk_store,
                                           storage_options=storage_options,
                                           mode=mode)
        if not getattr(chunk_store, '_store_version', DEFAULT_ZARR_VERSION) == zarr_version:
            raise ValueError(
                "zarr_version of store and chunk_store must match"
            )

    store_version = getattr(store, '_store_version', 2)
    if store_version == 3 and path is None:
        raise ValueError("path must be supplied to initialize a zarr v3 group")

    path = normalize_storage_path(path)

    # ensure store is initialized

    if mode in ['r', 'r+']:
        if not contains_group(store, path=path):
            if contains_array(store, path=path):
                raise ContainsArrayError(path)
            raise GroupNotFoundError(path)

    elif mode == 'w':
        init_group(store, overwrite=True, path=path, chunk_store=chunk_store)

    elif mode == 'a':
        if not contains_group(store, path=path):
            if contains_array(store, path=path):
                raise ContainsArrayError(path)
            init_group(store, path=path, chunk_store=chunk_store)

    elif mode in ['w-', 'x']:
        if contains_array(store, path=path):
            raise ContainsArrayError(path)
        elif contains_group(store, path=path):
            raise ContainsGroupError(path)
        else:
            init_group(store, path=path, chunk_store=chunk_store)

    # determine read only status
    read_only = mode == 'r'

    return Group(store, read_only=read_only, cache_attrs=cache_attrs,
                 synchronizer=synchronizer, path=path, chunk_store=chunk_store,
                 zarr_version=zarr_version)
