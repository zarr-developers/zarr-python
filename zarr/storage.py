# -*- coding: utf-8 -*-
"""This module contains storage classes for use with Zarr arrays and groups.

Note that any object implementing the :class:`MutableMapping` interface from the
:mod:`collections` module in the Python standard library can be used as a Zarr
array store, as long as it accepts string (str) keys and bytes values.

In addition to the :class:`MutableMapping` interface, store classes may also implement
optional methods `listdir` (list members of a "directory") and `rmdir` (remove all
members of a "directory"). These methods should be implemented if the store class is
aware of the hierarchical organisation of resources within the store and can provide
efficient implementations. If these methods are not available, Zarr will fall back to
slower implementations that work via the :class:`MutableMapping` interface. Store
classes may also optionally implement a `rename` method (rename all members under a given
path) and a `getsize` method (return the size in bytes of a given value).

"""
from __future__ import absolute_import, print_function, division
from collections import MutableMapping, OrderedDict
import os
import tempfile
import zipfile
import shutil
import atexit
import re
import sys
import multiprocessing
from threading import Lock, RLock
import glob
import warnings


import numpy as np


from zarr.util import (normalize_shape, normalize_chunks, normalize_order,
                       normalize_storage_path, buffer_size,
                       normalize_fill_value, nolock, normalize_dtype)
from zarr.meta import encode_array_metadata, encode_group_metadata
from zarr.compat import PY2, binary_type, OrderedDict_move_to_end
from numcodecs.registry import codec_registry
from zarr.errors import (err_contains_group, err_contains_array, err_bad_compressor,
                         err_fspath_exists_notdir, err_read_only)


array_meta_key = '.zarray'
group_meta_key = '.zgroup'
attrs_key = '.zattrs'
try:
    # noinspection PyUnresolvedReferences
    from zarr.codecs import Blosc
    default_compressor = Blosc()
except ImportError:  # pragma: no cover
    from zarr.codecs import Zlib
    default_compressor = Zlib()


def _path_to_prefix(path):
    # assume path already normalized
    if path:
        prefix = path + '/'
    else:
        prefix = ''
    return prefix


def contains_array(store, path=None):
    """Return True if the store contains an array at the given logical path."""
    path = normalize_storage_path(path)
    prefix = _path_to_prefix(path)
    key = prefix + array_meta_key
    return key in store


def contains_group(store, path=None):
    """Return True if the store contains a group at the given logical path."""
    path = normalize_storage_path(path)
    prefix = _path_to_prefix(path)
    key = prefix + group_meta_key
    return key in store


def _rmdir_from_keys(store, path=None):
    # assume path already normalized
    prefix = _path_to_prefix(path)
    for key in list(store.keys()):
        if key.startswith(prefix):
            del store[key]


def rmdir(store, path=None):
    """Remove all items under the given path. If `store` provides a `rmdir` method,
    this will be called, otherwise will fall back to implementation via the
    `MutableMapping` interface."""
    path = normalize_storage_path(path)
    if hasattr(store, 'rmdir'):
        # pass through
        store.rmdir(path)
    else:
        # slow version, delete one key at a time
        _rmdir_from_keys(store, path)


def _rename_from_keys(store, src_path, dst_path):
    # assume path already normalized
    src_prefix = _path_to_prefix(src_path)
    dst_prefix = _path_to_prefix(dst_path)
    for key in list(store.keys()):
        if key.startswith(src_prefix):
            new_key = dst_prefix + key.lstrip(src_prefix)
            store[new_key] = store.pop(key)


def rename(store, src_path, dst_path):
    """Rename all items under the given path. If `store` provides a `rename` method,
    this will be called, otherwise will fall back to implementation via the
    `MutableMapping` interface."""
    src_path = normalize_storage_path(src_path)
    dst_path = normalize_storage_path(dst_path)
    if hasattr(store, 'rename'):
        # pass through
        store.rename(src_path, dst_path)
    else:
        # slow version, delete one key at a time
        _rename_from_keys(store, src_path, dst_path)


def _listdir_from_keys(store, path=None):
    # assume path already normalized
    prefix = _path_to_prefix(path)
    children = set()
    for key in list(store.keys()):
        if key.startswith(prefix) and len(key) > len(prefix):
            suffix = key[len(prefix):]
            child = suffix.split('/')[0]
            children.add(child)
    return sorted(children)


def listdir(store, path=None):
    """Obtain a directory listing for the given path. If `store` provides a `listdir`
    method, this will be called, otherwise will fall back to implementation via the
    `MutableMapping` interface."""
    path = normalize_storage_path(path)
    if hasattr(store, 'listdir'):
        # pass through
        return store.listdir(path)
    else:
        # slow version, iterate through all keys
        return _listdir_from_keys(store, path)


def getsize(store, path=None):
    """Compute size of stored items for a given path. If `store` provides a `getsize`
    method, this will be called, otherwise will return -1."""
    path = normalize_storage_path(path)
    if hasattr(store, 'getsize'):
        # pass through
        return store.getsize(path)
    elif isinstance(store, dict):
        # compute from size of values
        if path in store:
            v = store[path]
            size = buffer_size(v)
        else:
            members = listdir(store, path)
            prefix = _path_to_prefix(path)
            size = 0
            for k in members:
                try:
                    v = store[prefix + k]
                except KeyError:
                    pass
                else:
                    try:
                        size += buffer_size(v)
                    except TypeError:
                        return -1
        return size
    else:
        return -1


def _require_parent_group(path, store, chunk_store, overwrite):
    # assume path is normalized
    if path:
        segments = path.split('/')
        for i in range(len(segments)):
            p = '/'.join(segments[:i])
            if contains_array(store, p):
                _init_group_metadata(store, path=p, chunk_store=chunk_store,
                                     overwrite=overwrite)
            elif not contains_group(store, p):
                _init_group_metadata(store, path=p, chunk_store=chunk_store)


def init_array(store, shape, chunks=True, dtype=None, compressor='default',
               fill_value=None, order='C', overwrite=False, path=None,
               chunk_store=None, filters=None, object_codec=None):
    """Initialize an array store with the given configuration. Note that this is a low-level
    function and there should be no need to call this directly from user code.

    Parameters
    ----------
    store : MutableMapping
        A mapping that supports string keys and bytes-like values.
    shape : int or tuple of ints
        Array shape.
    chunks : int or tuple of ints, optional
        Chunk shape. If True, will be guessed from `shape` and `dtype`. If
        False, will be set to `shape`, i.e., single chunk for the whole array.
    dtype : string or dtype, optional
        NumPy dtype.
    compressor : Codec, optional
        Primary compressor.
    fill_value : object
        Default value to use for uninitialized portions of the array.
    order : {'C', 'F'}, optional
        Memory layout to be used within each chunk.
    overwrite : bool, optional
        If True, erase all data in `store` prior to initialisation.
    path : string, optional
        Path under which array is stored.
    chunk_store : MutableMapping, optional
        Separate storage for chunks. If not provided, `store` will be used
        for storage of both chunks and metadata.
    filters : sequence, optional
        Sequence of filters to use to encode chunk data prior to compression.
    object_codec : Codec, optional
        A codec to encode object arrays, only needed if dtype=object.

    Examples
    --------
    Initialize an array store::

        >>> from zarr.storage import init_array
        >>> store = dict()
        >>> init_array(store, shape=(10000, 10000), chunks=(1000, 1000))
        >>> sorted(store.keys())
        ['.zarray']

    Array metadata is stored as JSON::

        >>> print(store['.zarray'].decode())
        {
            "chunks": [
                1000,
                1000
            ],
            "compressor": {
                "blocksize": 0,
                "clevel": 5,
                "cname": "lz4",
                "id": "blosc",
                "shuffle": 1
            },
            "dtype": "<f8",
            "fill_value": null,
            "filters": null,
            "order": "C",
            "shape": [
                10000,
                10000
            ],
            "zarr_format": 2
        }

    Initialize an array using a storage path::

        >>> store = dict()
        >>> init_array(store, shape=100000000, chunks=1000000, dtype='i1', path='foo')
        >>> sorted(store.keys())
        ['.zgroup', 'foo/.zarray']
        >>> print(store['foo/.zarray'].decode())
        {
            "chunks": [
                1000000
            ],
            "compressor": {
                "blocksize": 0,
                "clevel": 5,
                "cname": "lz4",
                "id": "blosc",
                "shuffle": 1
            },
            "dtype": "|i1",
            "fill_value": null,
            "filters": null,
            "order": "C",
            "shape": [
                100000000
            ],
            "zarr_format": 2
        }

    Notes
    -----
    The initialisation process involves normalising all array metadata, encoding
    as JSON and storing under the '.zarray' key.

    """

    # normalize path
    path = normalize_storage_path(path)

    # ensure parent group initialized
    _require_parent_group(path, store=store, chunk_store=chunk_store, overwrite=overwrite)

    _init_array_metadata(store, shape=shape, chunks=chunks, dtype=dtype,
                         compressor=compressor, fill_value=fill_value,
                         order=order, overwrite=overwrite, path=path,
                         chunk_store=chunk_store, filters=filters,
                         object_codec=object_codec)


def _init_array_metadata(store, shape, chunks=None, dtype=None, compressor='default',
                         fill_value=None, order='C', overwrite=False, path=None,
                         chunk_store=None, filters=None, object_codec=None):

    # guard conditions
    if overwrite:
        # attempt to delete any pre-existing items in store
        rmdir(store, path)
        if chunk_store is not None:
            rmdir(chunk_store, path)
    elif contains_array(store, path):
        err_contains_array(path)
    elif contains_group(store, path):
        err_contains_group(path)

    # normalize metadata
    shape = normalize_shape(shape)
    dtype, object_codec = normalize_dtype(dtype, object_codec)
    chunks = normalize_chunks(chunks, shape, dtype.itemsize)
    order = normalize_order(order)
    fill_value = normalize_fill_value(fill_value, dtype)

    # compressor prep
    if shape == ():
        # no point in compressing a 0-dimensional array, only a single value
        compressor = None
    elif compressor == 'none':
        # compatibility
        compressor = None
    elif compressor == 'default':
        compressor = default_compressor

    # obtain compressor config
    compressor_config = None
    if compressor:
        try:
            compressor_config = compressor.get_config()
        except AttributeError:
            err_bad_compressor(compressor)

    # obtain filters config
    if filters:
        filters_config = [f.get_config() for f in filters]
    else:
        filters_config = []

    # deal with object encoding
    if dtype == object:
        if object_codec is None:
            if not filters:
                # there are no filters so we can be sure there is no object codec
                raise ValueError('missing object_codec for object array')
            else:
                # one of the filters may be an object codec, issue a warning rather
                # than raise an error to maintain backwards-compatibility
                warnings.warn('missing object_codec for object array; this will raise a '
                              'ValueError in version 3.0', FutureWarning)
        else:
            filters_config.insert(0, object_codec.get_config())
    elif object_codec is not None:
        warnings.warn('an object_codec is only needed for object arrays')

    # use null to indicate no filters
    if not filters_config:
        filters_config = None

    # initialize metadata
    meta = dict(shape=shape, chunks=chunks, dtype=dtype,
                compressor=compressor_config, fill_value=fill_value,
                order=order, filters=filters_config)
    key = _path_to_prefix(path) + array_meta_key
    store[key] = encode_array_metadata(meta)


# backwards compatibility
init_store = init_array


def init_group(store, overwrite=False, path=None, chunk_store=None):
    """Initialize a group store. Note that this is a low-level function and there should be no
    need to call this directly from user code.

    Parameters
    ----------
    store : MutableMapping
        A mapping that supports string keys and byte sequence values.
    overwrite : bool, optional
        If True, erase all data in `store` prior to initialisation.
    path : string, optional
        Path under which array is stored.
    chunk_store : MutableMapping, optional
        Separate storage for chunks. If not provided, `store` will be used
        for storage of both chunks and metadata.

    """

    # normalize path
    path = normalize_storage_path(path)

    # ensure parent group initialized
    _require_parent_group(path, store=store, chunk_store=chunk_store,
                          overwrite=overwrite)

    # initialise metadata
    _init_group_metadata(store=store, overwrite=overwrite, path=path,
                         chunk_store=chunk_store)


def _init_group_metadata(store, overwrite=False, path=None, chunk_store=None):

    # guard conditions
    if overwrite:
        # attempt to delete any pre-existing items in store
        rmdir(store, path)
        if chunk_store is not None:
            rmdir(chunk_store, path)
    elif contains_array(store, path):
        err_contains_array(path)
    elif contains_group(store, path):
        err_contains_group(path)

    # initialize metadata
    # N.B., currently no metadata properties are needed, however there may
    # be in future
    meta = dict()
    key = _path_to_prefix(path) + group_meta_key
    store[key] = encode_group_metadata(meta)


def ensure_bytes(s):
    if isinstance(s, binary_type):
        return s
    if isinstance(s, np.ndarray):
        if PY2:  # pragma: py3 no cover
            # noinspection PyArgumentList
            return s.tostring(order='A')
        else:  # pragma: py2 no cover
            # noinspection PyArgumentList
            return s.tobytes(order='A')
    if hasattr(s, 'tobytes'):
        return s.tobytes()
    if PY2 and hasattr(s, 'tostring'):  # pragma: py3 no cover
        return s.tostring()
    return memoryview(s).tobytes()


def _dict_store_keys(d, prefix='', cls=dict):
    for k in d.keys():
        v = d[k]
        if isinstance(v, cls):
            for sk in _dict_store_keys(v, prefix + k + '/', cls):
                yield sk
        else:
            yield prefix + k


class DictStore(MutableMapping):
    """Store class that uses a hierarchy of :class:`dict` objects, thus all data
    will be held in main memory.

    Examples
    --------
    This is the default class used when creating a group. E.g.::

        >>> import zarr
        >>> g = zarr.group()
        >>> type(g.store)
        <class 'zarr.storage.DictStore'>

    Note that the default class when creating an array is the built-in
    :class:`dict` class, i.e.::

        >>> z = zarr.zeros(100)
        >>> type(z.store)
        <class 'dict'>

    Notes
    -----
    Safe to write in multiple threads.

    """

    def __init__(self, root=None, cls=dict):
        if root is None:
            self.root = cls()
        else:
            self.root = root
        self.cls = cls
        self.write_mutex = Lock()

    def __getstate__(self):
        return self.root, self.cls

    def __setstate__(self, state):
        root, cls = state
        self.__init__(root=root, cls=cls)

    def _get_parent(self, item):
        parent = self.root
        # split the item
        segments = item.split('/')
        # find the parent container
        for k in segments[:-1]:
            parent = parent[k]
            if not isinstance(parent, self.cls):
                raise KeyError(item)
        return parent, segments[-1]

    def _require_parent(self, item):
        parent = self.root
        # split the item
        segments = item.split('/')
        # require the parent container
        for k in segments[:-1]:
            try:
                parent = parent[k]
            except KeyError:
                parent[k] = self.cls()
                parent = parent[k]
            else:
                if not isinstance(parent, self.cls):
                    raise KeyError(item)
        return parent, segments[-1]

    def __getitem__(self, item):
        parent, key = self._get_parent(item)
        try:
            value = parent[key]
        except KeyError:
            raise KeyError(item)
        else:
            if isinstance(value, self.cls):
                raise KeyError(item)
            else:
                return value

    def __setitem__(self, item, value):
        with self.write_mutex:
            parent, key = self._require_parent(item)
            parent[key] = value

    def __delitem__(self, item):
        with self.write_mutex:
            parent, key = self._get_parent(item)
            try:
                del parent[key]
            except KeyError:
                raise KeyError(item)

    def __contains__(self, item):
        try:
            parent, key = self._get_parent(item)
            value = parent[key]
        except KeyError:
            return False
        else:
            return not isinstance(value, self.cls)

    def __eq__(self, other):
        return (
            isinstance(other, DictStore) and
            self.root == other.root and
            self.cls == other.cls
        )

    def keys(self):
        for k in _dict_store_keys(self.root, cls=self.cls):
            yield k

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return sum(1 for _ in self.keys())

    def listdir(self, path=None):
        path = normalize_storage_path(path)
        if path:
            try:
                parent, key = self._get_parent(path)
                value = parent[key]
            except KeyError:
                return []
        else:
            value = self.root
        if isinstance(value, self.cls):
            return sorted(value.keys())
        else:
            return []

    def rename(self, src_path, dst_path):
        src_path = normalize_storage_path(src_path)
        dst_path = normalize_storage_path(dst_path)

        src_parent, src_key = self._get_parent(src_path)
        dst_parent, dst_key = self._require_parent(dst_path)

        dst_parent[dst_key] = src_parent.pop(src_key)

    def rmdir(self, path=None):
        path = normalize_storage_path(path)
        if path:
            try:
                parent, key = self._get_parent(path)
                value = parent[key]
            except KeyError:
                return
            else:
                if isinstance(value, self.cls):
                    del parent[key]
        else:
            # clear out root
            self.root = self.cls()

    def getsize(self, path=None):
        path = normalize_storage_path(path)

        # obtain value to return size of
        value = None
        if path:
            try:
                parent, key = self._get_parent(path)
                value = parent[key]
            except KeyError:
                pass
        else:
            value = self.root

        # obtain size of value
        if value is None:
            return 0

        elif isinstance(value, self.cls):
            # total size for directory
            size = 0
            for v in value.values():
                if not isinstance(v, self.cls):
                    try:
                        size += buffer_size(v)
                    except TypeError:
                        return -1
            return size

        else:
            try:
                return buffer_size(value)
            except TypeError:
                return -1

    def clear(self):
        with self.write_mutex:
            self.root.clear()


class DirectoryStore(MutableMapping):
    """Storage class using directories and files on a standard file system.

    Parameters
    ----------
    path : string
        Location of directory to use as the root of the storage hierarchy.

    Examples
    --------
    Store a single array::

        >>> import zarr
        >>> store = zarr.DirectoryStore('data/array.zarr')
        >>> z = zarr.zeros((10, 10), chunks=(5, 5), store=store, overwrite=True)
        >>> z[...] = 42

    Each chunk of the array is stored as a separate file on the file system,
    i.e.::

        >>> import os
        >>> sorted(os.listdir('data/array.zarr'))
        ['.zarray', '0.0', '0.1', '1.0', '1.1']

    Store a group::

        >>> store = zarr.DirectoryStore('data/group.zarr')
        >>> root = zarr.group(store=store, overwrite=True)
        >>> foo = root.create_group('foo')
        >>> bar = foo.zeros('bar', shape=(10, 10), chunks=(5, 5))
        >>> bar[...] = 42

    When storing a group, levels in the group hierarchy will correspond to
    directories on the file system, i.e.::

        >>> sorted(os.listdir('data/group.zarr'))
        ['.zgroup', 'foo']
        >>> sorted(os.listdir('data/group.zarr/foo'))
        ['.zgroup', 'bar']
        >>> sorted(os.listdir('data/group.zarr/foo/bar'))
        ['.zarray', '0.0', '0.1', '1.0', '1.1']

    Notes
    -----
    Atomic writes are used, which means that data are first written to a
    temporary file, then moved into place when the write is successfully
    completed. Files are only held open while they are being read or written and are
    closed immediately afterwards, so there is no need to manually close any files.

    Safe to write in multiple threads or processes.

    """

    def __init__(self, path):

        # guard conditions
        path = os.path.abspath(path)
        if os.path.exists(path) and not os.path.isdir(path):
            err_fspath_exists_notdir(path)

        self.path = path

    def __getitem__(self, key):
        filepath = os.path.join(self.path, key)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                return f.read()
        else:
            raise KeyError(key)

    def __setitem__(self, key, value):

        # handle F-contiguous numpy arrays
        if isinstance(value, np.ndarray) and value.flags.f_contiguous:
            value = ensure_bytes(value)

        # destination path for key
        file_path = os.path.join(self.path, key)

        # ensure there is no directory in the way
        if os.path.isdir(file_path):
            shutil.rmtree(file_path)

        # ensure containing directory exists
        dir_path, file_name = os.path.split(file_path)
        if os.path.isfile(dir_path):
            raise KeyError(key)
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path)
            except Exception:
                raise KeyError(key)

        # write to temporary file
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, dir=dir_path,
                                             prefix=file_name + '.',
                                             suffix='.partial') as f:
                temp_path = f.name
                f.write(value)

            # move temporary file into place
            if os.path.exists(file_path):
                os.remove(file_path)
            os.rename(temp_path, file_path)

        finally:
            # clean up if temp file still exists for whatever reason
            if temp_path is not None and os.path.exists(temp_path):
                os.remove(temp_path)

    def __delitem__(self, key):
        path = os.path.join(self.path, key)
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            # include support for deleting directories, even though strictly
            # speaking these do not exist as keys in the store
            shutil.rmtree(path)
        else:
            raise KeyError(key)

    def __contains__(self, key):
        file_path = os.path.join(self.path, key)
        return os.path.isfile(file_path)

    def __eq__(self, other):
        return (
            isinstance(other, DirectoryStore) and
            self.path == other.path
        )

    def keys(self):
        if os.path.exists(self.path):
            directories = [(self.path, '')]
            while directories:
                dir_name, prefix = directories.pop()
                for name in os.listdir(dir_name):
                    path = os.path.join(dir_name, name)
                    if os.path.isfile(path):
                        yield prefix + name
                    elif os.path.isdir(path):
                        directories.append((path, prefix + name + '/'))

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return sum(1 for _ in self.keys())

    def dir_path(self, path=None):
        store_path = normalize_storage_path(path)
        dir_path = self.path
        if store_path:
            dir_path = os.path.join(dir_path, store_path)
        return dir_path

    def listdir(self, path=None):
        dir_path = self.dir_path(path)
        if os.path.isdir(dir_path):
            return sorted(os.listdir(dir_path))
        else:
            return []

    def rename(self, src_path, dst_path):
        store_src_path = normalize_storage_path(src_path)
        store_dst_path = normalize_storage_path(dst_path)

        dir_path = self.path

        src_path = os.path.join(dir_path, store_src_path)
        dst_path = os.path.join(dir_path, store_dst_path)

        dst_dir = os.path.dirname(dst_path)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        os.rename(src_path, dst_path)

    def rmdir(self, path=None):
        store_path = normalize_storage_path(path)
        dir_path = self.path
        if store_path:
            dir_path = os.path.join(dir_path, store_path)
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)

    def getsize(self, path=None):
        store_path = normalize_storage_path(path)
        fs_path = self.path
        if store_path:
            fs_path = os.path.join(fs_path, store_path)
        if os.path.isfile(fs_path):
            return os.path.getsize(fs_path)
        elif os.path.isdir(fs_path):
            children = os.listdir(fs_path)
            size = 0
            for child in children:
                child_fs_path = os.path.join(fs_path, child)
                if os.path.isfile(child_fs_path):
                    size += os.path.getsize(child_fs_path)
            return size
        else:
            return 0

    def clear(self):
        shutil.rmtree(self.path)


def atexit_rmtree(path,
                  isdir=os.path.isdir,
                  rmtree=shutil.rmtree):  # pragma: no cover
    """Ensure directory removal at interpreter exit."""
    if isdir(path):
        rmtree(path)


# noinspection PyShadowingNames
def atexit_rmglob(path,
                  glob=glob.glob,
                  isdir=os.path.isdir,
                  isfile=os.path.isfile,
                  remove=os.remove,
                  rmtree=shutil.rmtree):  # pragma: no cover
    """Ensure removal of multiple files at interpreter exit."""
    for p in glob(path):
        if isfile(p):
            remove(p)
        elif isdir(p):
            rmtree(p)


class TempStore(DirectoryStore):
    """Directory store using a temporary directory for storage.

    Parameters
    ----------
    suffix : string, optional
        Suffix for the temporary directory name.
    prefix : string, optional
        Prefix for the temporary directory name.
    dir : string, optional
        Path to parent directory in which to create temporary directory.

    """

    # noinspection PyShadowingBuiltins
    def __init__(self, suffix='', prefix='zarr', dir=None):
        path = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
        atexit.register(atexit_rmtree, path)
        super(TempStore, self).__init__(path)


_prog_ckey = re.compile(r'^(\d+)(\.\d+)+$')
_prog_number = re.compile(r'^\d+$')


def _nested_map_ckey(key):
    segments = list(key.split('/'))
    if segments:
        last_segment = segments[-1]
        if _prog_ckey.match(last_segment):
            last_segment = last_segment.replace('.', '/')
            segments = segments[:-1] + [last_segment]
            key = '/'.join(segments)
    return key


class NestedDirectoryStore(DirectoryStore):
    """Storage class using directories and files on a standard file system, with
    special handling for chunk keys so that chunk files for multidimensional
    arrays are stored in a nested directory tree.

    Parameters
    ----------
    path : string
        Location of directory to use as the root of the storage hierarchy.

    Examples
    --------
    Store a single array::

        >>> import zarr
        >>> store = zarr.NestedDirectoryStore('data/array.zarr')
        >>> z = zarr.zeros((10, 10), chunks=(5, 5), store=store, overwrite=True)
        >>> z[...] = 42

    Each chunk of the array is stored as a separate file on the file system,
    note the multiple directory levels used for the chunk files::

        >>> import os
        >>> sorted(os.listdir('data/array.zarr'))
        ['.zarray', '0', '1']
        >>> sorted(os.listdir('data/array.zarr/0'))
        ['0', '1']
        >>> sorted(os.listdir('data/array.zarr/1'))
        ['0', '1']

    Store a group::

        >>> store = zarr.NestedDirectoryStore('data/group.zarr')
        >>> root = zarr.group(store=store, overwrite=True)
        >>> foo = root.create_group('foo')
        >>> bar = foo.zeros('bar', shape=(10, 10), chunks=(5, 5))
        >>> bar[...] = 42

    When storing a group, levels in the group hierarchy will correspond to
    directories on the file system, i.e.::

        >>> sorted(os.listdir('data/group.zarr'))
        ['.zgroup', 'foo']
        >>> sorted(os.listdir('data/group.zarr/foo'))
        ['.zgroup', 'bar']
        >>> sorted(os.listdir('data/group.zarr/foo/bar'))
        ['.zarray', '0', '1']
        >>> sorted(os.listdir('data/group.zarr/foo/bar/0'))
        ['0', '1']
        >>> sorted(os.listdir('data/group.zarr/foo/bar/1'))
        ['0', '1']

    Notes
    -----
    The :class:`DirectoryStore` class stores all chunk files for an array
    together in a single directory. On some file systems, the potentially large
    number of files in a single directory can cause performance issues. The
    :class:`NestedDirectoryStore` class provides an alternative where chunk
    files for multidimensional arrays will be organised into a directory
    hierarchy, thus reducing the number of files in any one directory.

    Safe to write in multiple threads or processes.

    """

    def __init__(self, path):
        super(NestedDirectoryStore, self).__init__(path)

    def __getitem__(self, key):
        key = _nested_map_ckey(key)
        return super(NestedDirectoryStore, self).__getitem__(key)

    def __setitem__(self, key, value):
        key = _nested_map_ckey(key)
        super(NestedDirectoryStore, self).__setitem__(key, value)

    def __delitem__(self, key):
        key = _nested_map_ckey(key)
        super(NestedDirectoryStore, self).__delitem__(key)

    def __contains__(self, key):
        key = _nested_map_ckey(key)
        return super(NestedDirectoryStore, self).__contains__(key)

    def __eq__(self, other):
        return (
            isinstance(other, NestedDirectoryStore) and
            self.path == other.path
        )

    def listdir(self, path=None):
        children = super(NestedDirectoryStore, self).listdir(path=path)
        if array_meta_key in children:
            # special handling of directories containing an array to map nested chunk
            # keys back to standard chunk keys
            new_children = []
            root_path = self.dir_path(path)
            for entry in children:
                entry_path = os.path.join(root_path, entry)
                if _prog_number.match(entry) and os.path.isdir(entry_path):
                    for dir_path, _, file_names in os.walk(entry_path):
                        for file_name in file_names:
                            file_path = os.path.join(dir_path, file_name)
                            rel_path = file_path.split(root_path + os.path.sep)[1]
                            new_children.append(rel_path.replace(os.path.sep, '.'))
                else:
                    new_children.append(entry)
            return sorted(new_children)
        else:
            return children


# noinspection PyPep8Naming
class ZipStore(MutableMapping):
    """Storage class using a Zip file.

    Parameters
    ----------
    path : string
        Location of file.
    compression : integer, optional
        Compression method to use when writing to the archive.
    allowZip64 : bool, optional
        If True (the default) will create ZIP files that use the ZIP64
        extensions when the zipfile is larger than 2 GiB. If False
        will raise an exception when the ZIP file would require ZIP64
        extensions.
    mode : string, optional
        One of 'r' to read an existing file, 'w' to truncate and write a new
        file, 'a' to append to an existing file, or 'x' to exclusively create
        and write a new file.

    Examples
    --------
    Store a single array::

        >>> import zarr
        >>> store = zarr.ZipStore('data/array.zip', mode='w')
        >>> z = zarr.zeros((10, 10), chunks=(5, 5), store=store)
        >>> z[...] = 42
        >>> store.close()  # don't forget to call this when you're done

    Store a group::

        >>> store = zarr.ZipStore('data/group.zip', mode='w')
        >>> root = zarr.group(store=store)
        >>> foo = root.create_group('foo')
        >>> bar = foo.zeros('bar', shape=(10, 10), chunks=(5, 5))
        >>> bar[...] = 42
        >>> store.close()  # don't forget to call this when you're done

    After modifying a ZipStore, the ``close()`` method must be called, otherwise
    essential data will not be written to the underlying Zip file. The ZipStore
    class also supports the context manager protocol, which ensures the ``close()``
    method is called on leaving the context, e.g.::

        >>> with zarr.ZipStore('data/array.zip', mode='w') as store:
        ...     z = zarr.zeros((10, 10), chunks=(5, 5), store=store)
        ...     z[...] = 42
        ...     # no need to call store.close()

    Notes
    -----
    Each chunk of an array is stored as a separate entry in the Zip file. Note
    that Zip files do not provide any way to remove or replace existing entries.
    If an attempt is made to replace an entry, then a warning is generated by
    the Python standard library about a duplicate Zip file entry. This can be
    triggered if you attempt to write data to a Zarr array more than once,
    e.g.::

        >>> store = zarr.ZipStore('data/example.zip', mode='w')
        >>> z = zarr.zeros(100, chunks=10, store=store)
        >>> z[...] = 42  # first write OK
        >>> z[...] = 42  # second write generates warnings
        >>> store.close()

    This can also happen in a more subtle situation, where data are written only
    once to a Zarr array, but the write operations are not aligned with chunk
    boundaries, e.g.::

        >>> store = zarr.ZipStore('data/example.zip', mode='w')
        >>> z = zarr.zeros(100, chunks=10, store=store)
        >>> z[5:15] = 42
        >>> z[15:25] = 42  # write overlaps chunk previously written, generates warnings

    To avoid creating duplicate entries, only write data once, and align writes
    with chunk boundaries. This alignment is done automatically if you call
    ``z[...] = ...`` or create an array from existing data via :func:`zarr.array`.

    Alternatively, use a :class:`DirectoryStore` when writing the data, then
    manually Zip the directory and use the Zip file for subsequent reads.

    Safe to write in multiple threads but not in multiple processes.

    """

    def __init__(self, path, compression=zipfile.ZIP_STORED, allowZip64=True, mode='a'):

        # store properties
        path = os.path.abspath(path)
        self.path = path
        self.compression = compression
        self.allowZip64 = allowZip64
        self.mode = mode

        # Current understanding is that zipfile module in stdlib is not thread-safe,
        # and so locking is required for both read and write. However, this has not
        # been investigated in detail, perhaps no lock is needed if mode='r'.
        self.mutex = RLock()

        # open zip file
        self.zf = zipfile.ZipFile(path, mode=mode, compression=compression,
                                  allowZip64=allowZip64)

    def __getstate__(self):
        self.flush()
        return self.path, self.compression, self.allowZip64, self.mode

    def __setstate__(self, state):
        path, compression, allowZip64, mode = state
        # if initially opened with mode 'w' or 'x', re-open in mode 'a' so file doesn't
        # get clobbered
        if mode in 'wx':
            mode = 'a'
        self.__init__(path=path, compression=compression, allowZip64=allowZip64,
                      mode=mode)

    def close(self):
        """Closes the underlying zip file, ensuring all records are written."""
        with self.mutex:
            self.zf.close()

    def flush(self):
        """Closes the underlying zip file, ensuring all records are written,
        then re-opens the file for further modifications."""
        if self.mode != 'r':
            with self.mutex:
                self.zf.close()
                # N.B., re-open with mode 'a' regardless of initial mode so we don't wipe
                # what's been written
                self.zf = zipfile.ZipFile(self.path, mode='a',
                                          compression=self.compression,
                                          allowZip64=self.allowZip64)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __getitem__(self, key):
        with self.mutex:
            with self.zf.open(key) as f:  # will raise KeyError
                return f.read()

    def __setitem__(self, key, value):
        if self.mode == 'r':
            err_read_only()
        value = ensure_bytes(value)
        with self.mutex:
            self.zf.writestr(key, value)

    def __delitem__(self, key):
        raise NotImplementedError

    def __eq__(self, other):
        return (
            isinstance(other, ZipStore) and
            self.path == other.path and
            self.compression == other.compression and
            self.allowZip64 == other.allowZip64
        )

    def keylist(self):
        with self.mutex:
            return sorted(self.zf.namelist())

    def keys(self):
        for key in self.keylist():
            yield key

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return sum(1 for _ in self.keys())

    def __contains__(self, key):
        try:
            with self.mutex:
                self.zf.getinfo(key)
        except KeyError:
            return False
        else:
            return True

    def listdir(self, path=None):
        path = normalize_storage_path(path)
        return _listdir_from_keys(self, path)

    def getsize(self, path=None):
        path = normalize_storage_path(path)
        with self.mutex:
            children = self.listdir(path)
            if children:
                size = 0
                for child in children:
                    if path:
                        name = path + '/' + child
                    else:
                        name = child
                    try:
                        info = self.zf.getinfo(name)
                    except KeyError:
                        pass
                    else:
                        size += info.compress_size
                return size
            elif path:
                try:
                    info = self.zf.getinfo(path)
                    return info.compress_size
                except KeyError:
                    return 0
            else:
                return 0

    def clear(self):
        if self.mode == 'r':
            err_read_only()
        with self.mutex:
            self.close()
            os.remove(self.path)
            self.zf = zipfile.ZipFile(self.path, mode=self.mode,
                                      compression=self.compression,
                                      allowZip64=self.allowZip64)


def migrate_1to2(store):
    """Migrate array metadata in `store` from Zarr format version 1 to
    version 2.

    Parameters
    ----------
    store : MutableMapping
        Store to be migrated.

    Notes
    -----
    Version 1 did not support hierarchies, so this migration function will
    look for a single array in `store` and migrate the array metadata to
    version 2.

    """

    # migrate metadata
    from zarr import meta_v1
    meta = meta_v1.decode_metadata(store['meta'])
    del store['meta']

    # add empty filters
    meta['filters'] = None

    # migration compression metadata
    compression = meta['compression']
    if compression is None or compression == 'none':
        compressor_config = None
    else:
        compression_opts = meta['compression_opts']
        codec_cls = codec_registry[compression]
        if isinstance(compression_opts, dict):
            compressor = codec_cls(**compression_opts)
        else:
            compressor = codec_cls(compression_opts)
        compressor_config = compressor.get_config()
    meta['compressor'] = compressor_config
    del meta['compression']
    del meta['compression_opts']

    # store migrated metadata
    store[array_meta_key] = encode_array_metadata(meta)

    # migrate user attributes
    store[attrs_key] = store['attrs']
    del store['attrs']


def _dbm_encode_key(key):
    if hasattr(key, 'encode'):
        key = key.encode('ascii')
    return key


def _dbm_decode_key(key):
    if hasattr(key, 'decode'):
        key = key.decode('ascii')
    return key


# noinspection PyShadowingBuiltins
class DBMStore(MutableMapping):
    """Storage class using a DBM-style database.

    Parameters
    ----------
    path : string
        Location of database file.
    flag : string, optional
        Flags for opening the database file.
    mode : int
        File mode used if a new file is created.
    open : function, optional
        Function to open the database file. If not provided, :func:`dbm.open` will be
        used on Python 3, and :func:`anydbm.open` will be used on Python 2.
    write_lock: bool, optional
        Use a lock to prevent concurrent writes from multiple threads (True by default).
    **open_kwargs
        Keyword arguments to pass the `open` function.

    Examples
    --------
    Store a single array::

        >>> import zarr
        >>> store = zarr.DBMStore('data/array.db')
        >>> z = zarr.zeros((10, 10), chunks=(5, 5), store=store, overwrite=True)
        >>> z[...] = 42
        >>> store.close()  # don't forget to call this when you're done

    Store a group::

        >>> store = zarr.DBMStore('data/group.db')
        >>> root = zarr.group(store=store, overwrite=True)
        >>> foo = root.create_group('foo')
        >>> bar = foo.zeros('bar', shape=(10, 10), chunks=(5, 5))
        >>> bar[...] = 42
        >>> store.close()  # don't forget to call this when you're done

    After modifying a DBMStore, the ``close()`` method must be called, otherwise
    essential data may not be written to the underlying database file. The
    DBMStore class also supports the context manager protocol, which ensures the
    ``close()`` method is called on leaving the context, e.g.::

        >>> with zarr.DBMStore('data/array.db') as store:
        ...     z = zarr.zeros((10, 10), chunks=(5, 5), store=store, overwrite=True)
        ...     z[...] = 42
        ...     # no need to call store.close()

    A different database library can be used by passing a different function to
    the `open` parameter. For example, if the `bsddb3
    <https://www.jcea.es/programacion/pybsddb.htm>`_ package is installed, a
    Berkeley DB database can be used::

        >>> import bsddb3
        >>> store = zarr.DBMStore('data/array.bdb', open=bsddb3.btopen)
        >>> z = zarr.zeros((10, 10), chunks=(5, 5), store=store, overwrite=True)
        >>> z[...] = 42
        >>> store.close()

    Notes
    -----
    Please note that, by default, this class will use the Python standard
    library `dbm.open` function to open the database file (or `anydbm.open` on
    Python 2). There are up to three different implementations of DBM-style
    databases available in any Python installation, and which one is used may
    vary from one system to another.  Database file formats are not compatible
    between these different implementations.  Also, some implementations are
    more efficient than others. In particular, the "dumb" implementation will be
    the fall-back on many systems, and has very poor performance for some usage
    scenarios. If you want to ensure a specific implementation is used, pass the
    corresponding open function, e.g., `dbm.gnu.open` to use the GNU DBM
    library.

    Safe to write in multiple threads. May be safe to write in multiple processes,
    depending on which DBM implementation is being used, although this has not been
    tested.

    """

    def __init__(self, path, flag='c', mode=0o666, open=None, write_lock=True,
                 **open_kwargs):
        if open is None:
            if PY2:  # pragma: py3 no cover
                import anydbm
                open = anydbm.open
            else:  # pragma: py2 no cover
                import dbm
                open = dbm.open
        path = os.path.abspath(path)
        # noinspection PyArgumentList
        self.db = open(path, flag, mode, **open_kwargs)
        self.path = path
        self.flag = flag
        self.mode = mode
        self.open = open
        self.write_lock = write_lock
        if write_lock:
            # This may not be required as some dbm implementations manage their own
            # locks, but err on the side of caution.
            self.write_mutex = Lock()
        else:
            self.write_mutex = nolock
        self.open_kwargs = open_kwargs

    def __getstate__(self):
        self.flush()  # needed for py2 and ndbm
        return (self.path, self.flag, self.mode, self.open, self.write_lock,
                self.open_kwargs)

    def __setstate__(self, state):
        path, flag, mode, open, write_lock, open_kws = state
        if flag[0] == 'n':
            flag = 'c' + flag[1:]  # don't clobber an existing database
        self.__init__(path=path, flag=flag, mode=mode, open=open,
                      write_lock=write_lock, **open_kws)

    def close(self):
        """Closes the underlying database file."""
        if hasattr(self.db, 'close'):
            with self.write_mutex:
                self.db.close()

    def flush(self):
        """Synchronizes data to the underlying database file."""
        if self.flag[0] != 'r':
            with self.write_mutex:
                if hasattr(self.db, 'sync'):
                        self.db.sync()
                else:
                    # fall-back, close and re-open, needed for ndbm
                    flag = self.flag
                    if flag[0] == 'n':
                        flag = 'c' + flag[1:]  # don't clobber an existing database
                    self.db.close()
                    # noinspection PyArgumentList
                    self.db = self.open(self.path, flag, self.mode, **self.open_kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __getitem__(self, key):
        key = _dbm_encode_key(key)
        return self.db[key]

    def __setitem__(self, key, value):
        key = _dbm_encode_key(key)
        value = ensure_bytes(value)
        with self.write_mutex:
            self.db[key] = value

    def __delitem__(self, key):
        key = _dbm_encode_key(key)
        with self.write_mutex:
            del self.db[key]

    def __eq__(self, other):
        return (
            isinstance(other, DBMStore) and
            self.path == other.path and
            # allow flag and mode to differ
            self.open == other.open and
            self.open_kwargs == other.open_kwargs
        )

    def keys(self):
        return (_dbm_decode_key(k) for k in iter(self.db.keys()))

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return sum(1 for _ in self.keys())

    def __contains__(self, key):
        key = _dbm_encode_key(key)
        return key in self.db


if PY2:  # pragma: py3 no cover

    def _lmdb_decode_key_buffer(key):
        # assume buffers=True
        return str(key)

    def _lmdb_decode_key_bytes(key):
        return key

else:  # pragma: py2 no cover

    def _lmdb_decode_key_buffer(key):
        # assume buffers=True
        return key.tobytes().decode('ascii')

    def _lmdb_decode_key_bytes(key):
        # assume buffers=False
        return key.decode('ascii')


class LMDBStore(MutableMapping):
    """Storage class using LMDB. Requires the `lmdb <http://lmdb.readthedocs.io/>`_
    package to be installed.


    Parameters
    ----------
    path : string
        Location of database file.
    buffers : bool, optional
        If True (default) use support for buffers, which should increase performance by
        reducing memory copies.
    **kwargs
        Keyword arguments passed through to the `lmdb.open` function.

    Examples
    --------
    Store a single array::

        >>> import zarr
        >>> store = zarr.LMDBStore('data/array.mdb')
        >>> z = zarr.zeros((10, 10), chunks=(5, 5), store=store, overwrite=True)
        >>> z[...] = 42
        >>> store.close()  # don't forget to call this when you're done

    Store a group::

        >>> store = zarr.LMDBStore('data/group.mdb')
        >>> root = zarr.group(store=store, overwrite=True)
        >>> foo = root.create_group('foo')
        >>> bar = foo.zeros('bar', shape=(10, 10), chunks=(5, 5))
        >>> bar[...] = 42
        >>> store.close()  # don't forget to call this when you're done

    After modifying a DBMStore, the ``close()`` method must be called, otherwise
    essential data may not be written to the underlying database file. The
    DBMStore class also supports the context manager protocol, which ensures the
    ``close()`` method is called on leaving the context, e.g.::

        >>> with zarr.LMDBStore('data/array.mdb') as store:
        ...     z = zarr.zeros((10, 10), chunks=(5, 5), store=store, overwrite=True)
        ...     z[...] = 42
        ...     # no need to call store.close()

    Notes
    -----
    By default writes are not immediately flushed to disk to increase performance. You
    can ensure data are flushed to disk by calling the ``flush()`` or ``close()`` methods.

    Should be safe to write in multiple threads or processes due to the synchronization
    support within LMDB, although writing from multiple processes has not been tested.

    """

    def __init__(self, path, buffers=True, **kwargs):
        import lmdb

        # set default memory map size to something larger than the lmdb default, which is
        # very likely to be too small for any moderate dataset (logic copied from zict)
        map_size = (2**40 if sys.maxsize >= 2**32 else 2**28)
        kwargs.setdefault('map_size', map_size)

        # don't initialize buffers to zero by default, shouldn't be necessary
        kwargs.setdefault('meminit', False)

        # decide whether to use the writemap option based on the operating system's
        # support for sparse files - writemap requires sparse file support otherwise
        # the whole# `map_size` may be reserved up front on disk (logic copied from zict)
        writemap = sys.platform.startswith('linux')
        kwargs.setdefault('writemap', writemap)

        # decide options for when data are flushed to disk - choose to delay syncing
        # data to filesystem, otherwise pay a large performance penalty (zict also does
        # this)
        kwargs.setdefault('metasync', False)
        kwargs.setdefault('sync', False)
        kwargs.setdefault('map_async', False)

        # set default option for number of cached transactions
        max_spare_txns = multiprocessing.cpu_count()
        kwargs.setdefault('max_spare_txns', max_spare_txns)

        # normalize path
        path = os.path.abspath(path)

        # open database
        self.db = lmdb.open(path, **kwargs)

        # store properties
        if buffers:
            self.decode_key = _lmdb_decode_key_buffer
        else:
            self.decode_key = _lmdb_decode_key_bytes
        self.buffers = buffers
        self.path = path
        self.kwargs = kwargs

    def __getstate__(self):
        self.flush()  # just in case
        return self.path, self.buffers, self.kwargs

    def __setstate__(self, state):
        path, buffers, kwargs = state
        self.__init__(path=path, buffers=buffers, **kwargs)

    def close(self):
        """Closes the underlying database."""
        self.db.close()

    def flush(self):
        """Synchronizes data to the file system."""
        self.db.sync()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __getitem__(self, key):
        key = _dbm_encode_key(key)
        # use the buffers option, should avoid a memory copy
        with self.db.begin(buffers=self.buffers) as txn:
            value = txn.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def __setitem__(self, key, value):
        key = _dbm_encode_key(key)
        with self.db.begin(write=True, buffers=self.buffers) as txn:
            txn.put(key, value)

    def __delitem__(self, key):
        key = _dbm_encode_key(key)
        with self.db.begin(write=True) as txn:
            if not txn.delete(key):
                raise KeyError(key)

    def __contains__(self, key):
        key = _dbm_encode_key(key)
        with self.db.begin(buffers=self.buffers) as txn:
            with txn.cursor() as cursor:
                return cursor.set_key(key)

    def items(self):
        with self.db.begin(buffers=self.buffers) as txn:
            with txn.cursor() as cursor:
                for k, v in cursor.iternext(keys=True, values=True):
                    yield self.decode_key(k), v

    def keys(self):
        with self.db.begin(buffers=self.buffers) as txn:
            with txn.cursor() as cursor:
                for k in cursor.iternext(keys=True, values=False):
                    yield self.decode_key(k)

    def values(self):
        with self.db.begin(buffers=self.buffers) as txn:
            with txn.cursor() as cursor:
                for v in cursor.iternext(keys=False, values=True):
                    yield v

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return self.db.stat()['entries']


class LRUStoreCache(MutableMapping):
    """Storage class that implements a least-recently-used (LRU) cache layer over
    some other store. Intended primarily for use with stores that can be slow to
    access, e.g., remote stores that require network communication to store and
    retrieve data.

    Parameters
    ----------
    store : MutableMapping
        The store containing the actual data to be cached.
    max_size : int
        The maximum size that the cache may grow to, in number of bytes. Provide `None`
        if you would like the cache to have unlimited size.

    Examples
    --------
    The example below wraps an S3 store with an LRU cache::

        >>> import s3fs
        >>> import zarr
        >>> s3 = s3fs.S3FileSystem(anon=True, client_kwargs=dict(region_name='eu-west-2'))
        >>> store = s3fs.S3Map(root='zarr-demo/store', s3=s3, check=False)
        >>> cache = zarr.LRUStoreCache(store, max_size=2**28)
        >>> root = zarr.group(store=cache)
        >>> z = root['foo/bar/baz']
        >>> from timeit import timeit
        >>> # first data access is relatively slow, retrieved from store
        ... timeit('print(z[:].tostring())', number=1, globals=globals())  # doctest: +SKIP
        b'Hello from the cloud!'
        0.1081731989979744
        >>> # second data access is faster, uses cache
        ... timeit('print(z[:].tostring())', number=1, globals=globals())  # doctest: +SKIP
        b'Hello from the cloud!'
        0.0009490990014455747

    """

    def __init__(self, store, max_size):
        self._store = store
        self._max_size = max_size
        self._current_size = 0
        self._keys_cache = None
        self._contains_cache = None
        self._listdir_cache = dict()
        self._values_cache = OrderedDict()
        self._mutex = Lock()
        self.hits = self.misses = 0

    def __getstate__(self):
        return (self._store, self._max_size, self._current_size, self._keys_cache,
                self._contains_cache, self._listdir_cache, self._values_cache, self.hits,
                self.misses)

    def __setstate__(self, state):
        (self._store, self._max_size, self._current_size, self._keys_cache,
         self._contains_cache, self._listdir_cache, self._values_cache, self.hits,
         self.misses) = state
        self._mutex = Lock()

    def __len__(self):
        return len(self._keys())

    def __iter__(self):
        return self.keys()

    def __contains__(self, key):
        with self._mutex:
            if self._contains_cache is None:
                self._contains_cache = set(self._keys())
            return key in self._contains_cache

    def clear(self):
        self._store.clear()
        self.invalidate()

    def keys(self):
        with self._mutex:
            return iter(self._keys())

    def _keys(self):
        if self._keys_cache is None:
            self._keys_cache = list(self._store.keys())
        return self._keys_cache

    def listdir(self, path=None):
        with self._mutex:
            try:
                return self._listdir_cache[path]
            except KeyError:
                listing = listdir(self._store, path)
                self._listdir_cache[path] = listing
                return listing

    def getsize(self, path=None):
        return getsize(self._store, path=path)

    def _pop_value(self):
        # remove the first value from the cache, as this will be the least recently
        # used value
        _, v = self._values_cache.popitem(last=False)
        return v

    def _accommodate_value(self, value_size):
        if self._max_size is None:
            return
        # ensure there is enough space in the cache for a new value
        while self._current_size + value_size > self._max_size:
            v = self._pop_value()
            self._current_size -= buffer_size(v)

    def _cache_value(self, key, value):
        # cache a value
        value_size = buffer_size(value)
        # check size of the value against max size, as if the value itself exceeds max
        # size then we are never going to cache it
        if self._max_size is None or value_size <= self._max_size:
            self._accommodate_value(value_size)
            self._values_cache[key] = value
            self._current_size += value_size

    def invalidate(self):
        """Completely clear the cache."""
        with self._mutex:
            self._values_cache.clear()
            self._invalidate_keys()

    def invalidate_values(self):
        """Clear the values cache."""
        with self._mutex:
            self._values_cache.clear()

    def invalidate_keys(self):
        """Clear the keys cache."""
        with self._mutex:
            self._invalidate_keys()

    def _invalidate_keys(self):
        self._keys_cache = None
        self._contains_cache = None
        self._listdir_cache.clear()

    def _invalidate_value(self, key):
        if key in self._values_cache:
            value = self._values_cache.pop(key)
            self._current_size -= buffer_size(value)

    def __getitem__(self, key):
        try:
            # first try to obtain the value from the cache
            with self._mutex:
                value = self._values_cache[key]
                # cache hit if no KeyError is raised
                self.hits += 1
                # treat the end as most recently used
                OrderedDict_move_to_end(self._values_cache, key)

        except KeyError:
            # cache miss, retrieve value from the store
            value = self._store[key]
            with self._mutex:
                self.misses += 1
                # need to check if key is not in the cache, as it may have been cached
                # while we were retrieving the value from the store
                if key not in self._values_cache:
                    self._cache_value(key, value)

        return value

    def __setitem__(self, key, value):
        self._store[key] = value
        with self._mutex:
            self._invalidate_keys()
            self._invalidate_value(key)
            self._cache_value(key, value)

    def __delitem__(self, key):
        del self._store[key]
        with self._mutex:
            self._invalidate_keys()
            self._invalidate_value(key)
