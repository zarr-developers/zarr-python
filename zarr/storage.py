# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from collections import MutableMapping
import os
import tempfile
import json
import zipfile
import shutil
import operator


import numpy as np


from zarr.util import normalize_shape, normalize_chunks, normalize_order, \
    normalize_storage_path
from zarr.compressors import get_compressor_cls
from zarr.meta import encode_array_metadata, encode_group_metadata
from zarr.compat import PY2, binary_type, reduce


array_meta_key = '.zarray'
group_meta_key = '.zgroup'
attrs_key = '.zattrs'


def _path_to_prefix(path):
    # assume path already normalized
    if path:
        prefix = path + '/'
    else:
        prefix = ''
    return prefix
    

def contains_array(store, path=None):
    """TODO doc me"""
    path = normalize_storage_path(path)
    prefix = _path_to_prefix(path)
    key = prefix + array_meta_key
    return key in store


def contains_group(store, path=None):
    """TODO"""
    path = normalize_storage_path(path)
    prefix = _path_to_prefix(path)
    key = prefix + group_meta_key
    return key in store


def _rmdir_from_keys(store, path=None):
    # TODO review, esp. with None prefix
    # assume path already normalized
    prefix = _path_to_prefix(path)
    for key in set(store.keys()):
        if key.startswith(prefix):
            del store[key]


def rmdir(store, path=None):
    """TODO"""
    # TODO review
    path = normalize_storage_path(path)
    if hasattr(store, 'rmdir'):
        # pass through
        store.rmdir(path)
    else:
        # slow version, delete one key at a time
        _rmdir_from_keys(store, path)


def _listdir_from_keys(store, path=None):
    # TODO review, esp. with None prefix
    # assume path already normalized
    prefix = _path_to_prefix(path)
    children = set()
    for key in store.keys():
        if key.startswith(prefix) and len(key) > len(prefix):
            suffix = key[len(prefix):]
            child = suffix.split('/')[0]
            children.add(child)
    return sorted(children)


def listdir(store, path=None):
    """TODO"""
    # TODO review
    path = normalize_storage_path(path)
    if hasattr(store, 'listdir'):
        # pass through
        return store.listdir(path)
    else:
        # slow version, iterate through all keys
        return _listdir_from_keys(store, path)


def init_array(store, shape, chunks, dtype=None, compression='default',
               compression_opts=None, fill_value=None,
               order='C', overwrite=False, path=None):
    """initialize an array store with the given configuration.

    Parameters
    ----------
    store : MutableMapping
        A mapping that supports string keys and byte sequence values.
    shape : int or tuple of ints
        Array shape.
    chunks : int or tuple of ints
        Chunk shape.
    dtype : string or dtype, optional
        NumPy dtype.
    compression : string, optional
        Name of primary compression library, e.g., 'blosc', 'zlib', 'bz2',
        'lzma'.
    compression_opts : object, optional
        Options to primary compressor. E.g., for blosc, provide a dictionary
        with keys 'cname', 'clevel' and 'shuffle'.
    fill_value : object
        Default value to use for uninitialized portions of the array.
    order : {'C', 'F'}, optional
        Memory layout to be used within each chunk.
    overwrite : bool, optional
        If True, erase all data in `store` prior to initialisation.
    path : string, optional
        Path under which array is stored.

    Examples
    --------
    Initialize an array store::

        >>> from zarr.storage import init_array
        >>> store = dict()
        >>> init_array(store, shape=(10000, 10000), chunks=(1000, 1000))
        >>> sorted(store.keys())
        ['.zattrs', '.zarray']

    Array metadata is stored as JSON::

        >>> print(str(store['.zarray'], 'ascii'))
        {
            "chunks": [
                1000,
                1000
            ],
            "compression": "blosc",
            "compression_opts": {
                "clevel": 5,
                "cname": "lz4",
                "shuffle": 1
            },
            "dtype": "<f8",
            "fill_value": null,
            "order": "C",
            "shape": [
                10000,
                10000
            ],
            "zarr_format": 1
        }

    User-defined attributes are also stored as JSON, initially empty::

        >>> print(str(store['.zattrs'], 'ascii'))
        {}

    initialize an array using a storage path::

        >>> init_array(store, shape=100000000, chunks=1000000, dtype='i1',
        ...            path='foo/bar')
        >>> sorted(store.keys())
        TODO
        >>> print(str(store['foo/bar/.zarray'], 'ascii'))
        TODO

    Notes
    -----
    The initialisation process involves normalising all array metadata,
    encoding as JSON and storing under the '.zarray' key. User attributes are 
    also initialized and stored as JSON under the '.zattrs' key.

    """

    # normalize path
    path = normalize_storage_path(path)
    
    # guard conditions
    if overwrite:
        # attempt to delete any pre-existing items in store
        rmdir(store, path)
    elif contains_array(store, path):
        raise ValueError('store contains an array')
    elif contains_group(store, path):
        raise ValueError('store contains a group')

    # normalize metadata
    shape = normalize_shape(shape)
    chunks = normalize_chunks(chunks, shape)
    dtype = np.dtype(dtype)
    compressor_cls = get_compressor_cls(compression)
    compression = compressor_cls.canonical_name
    compression_opts = compressor_cls.normalize_opts(
        compression_opts
    )
    order = normalize_order(order)

    # initialize metadata
    meta = dict(shape=shape, chunks=chunks, dtype=dtype,
                compression=compression, compression_opts=compression_opts,
                fill_value=fill_value, order=order)
    key = _path_to_prefix(path) + array_meta_key
    store[key] = encode_array_metadata(meta)

    # initialize attributes
    key = _path_to_prefix(path) + attrs_key
    store[key] = json.dumps(dict()).encode('ascii')


# backwards compatibility
init_store = init_array


def init_group(store, overwrite=False, path=None):
    """initialize a group store.

    Parameters
    ----------
    store : MutableMapping
        A mapping that supports string keys and byte sequence values.
    overwrite : bool, optional
        If True, erase all data in `store` prior to initialisation.
    path : string, optional
        Path under which array is stored.

    """

    # normalize path
    path = normalize_storage_path(path)
    
    # guard conditions
    if overwrite:
        # attempt to delete any pre-existing items in store
        rmdir(store, path)
    elif contains_array(store, path):
        raise ValueError('store contains an array')
    elif contains_group(store, path):
        raise ValueError('store contains a group')

    # initialize metadata
    # N.B., currently no metadata properties are needed, however there may
    # be in future
    meta = dict()
    key = _path_to_prefix(path) + group_meta_key
    store[key] = encode_group_metadata(meta)

    # initialize attributes
    key = _path_to_prefix(path) + attrs_key
    store[key] = json.dumps(dict()).encode('ascii')


def ensure_bytes(s):
    if isinstance(s, binary_type):
        return s
    if hasattr(s, 'encode'):
        return s.encode()
    if hasattr(s, 'tobytes'):
        return s.tobytes()
    if PY2 and hasattr(s, 'tostring'):
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


def _getbuffersize(v):
    from array import array as _stdlib_array
    if PY2 and isinstance(v, _stdlib_array):
        # special case array.array because does not support buffer
        # interface in PY2
        return v.buffer_info()[1] * v.itemsize
    else:
        v = memoryview(v)
        return reduce(operator.mul, v.shape) * v.itemsize


class DictStore(MutableMapping):
    """Extended mutable mapping interface to a hierarchy of dicts.

    Examples
    --------
    >>> import zarr
    >>> store = zarr.DictStore('example')
    >>> store['foo'] = b'bar'
    >>> store['foo']
    b'bar'
    >>> store['a/b/c'] = b'xxx'
    >>> store['a/b/c']
    b'xxx'
    >>> sorted(store.keys())
    ['foo', 'a/b/c']
    >>> store.listdir()
    ['a', 'foo']
    >>> store.listdir('a/b')
    ['c']
    >>> store.rmdir('a')
    >>> sorted(store.keys())
    ['foo']

    """  # flake8: noqa

    def __init__(self, cls=dict):
        self.root = cls()
        self.cls = cls

    def __getitem__(self, key):
        c = self.root
        for k in key.split('/'):
            c = c[k]
        if isinstance(c, self.cls):
            raise KeyError(key)
        return c

    def __setitem__(self, key, value):
        c = self.root
        keys = key.split('/')

        # ensure intermediate containers
        for k in keys[:-1]:
            try:
                c = c[k]
                if not isinstance(c, self.cls):
                    raise KeyError(key)
            except KeyError:
                c[k] = self.cls()
                c = c[k]

        # set final value
        c[keys[-1]] = value

    def __delitem__(self, key):
        c = self.root
        keys = key.split('/')

        # obtain final container
        for k in keys[:-1]:
            c = c[k]

        # delete item
        del c[keys[-1]]

    def __contains__(self, key):
        keys = key.split('/')
        c = self.root
        for k in keys:
            try:
                c = c[k]
            except KeyError:
                return False
        return not isinstance(c, self.cls)

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
        c = self.root
        if path:
            # split path and find container
            for k in path.split('/'):
                c = c[k]
        return sorted(c.keys())

    def rmdir(self, path=None):
        path = normalize_storage_path(path)
        c = self.root
        if path:
            # split path and find container
            segments = path.split('/')
            for k in segments[:-1]:
                c = c[k]
            # remove final key
            del c[segments[-1]]
        else:
            # clear out root
            self.root = self.cls()

    def getsize(self, path=None):
        path = normalize_storage_path(path)
        c = self.root
        if path:
            # split path and find value
            segments = path.split('/')
            try:
                for k in segments:
                    c = c[k]
            except KeyError:
                raise ValueError('path not found: %r' % path)
        if isinstance(c, self.cls):
            # total size for directory
            size = 0
            for v in c.values():
                if not isinstance(v, self.cls):
                    try:
                        size += _getbuffersize(v)
                    except TypeError:
                        return -1
            return size
        else:
            try:
                return _getbuffersize(c)
            except TypeError:
                return -1


class DirectoryStore(MutableMapping):
    """Mutable Mapping interface to a directory. Keys must be strings,
    values must be bytes-like objects.

    Parameters
    ----------
    path : string
        Location of directory.

    Examples
    --------
    >>> import zarr
    >>> store = zarr.DirectoryStore('example')
    >>> store['foo'] = b'bar'
    >>> store['foo']
    b'bar'
    >>> open('example/foo').read()
    b'bar'
    >>> store['a/b/c'] = b'xxx'
    >>> store['a/b/c']
    b'xxx'
    >>> open('example/a/b/c').read()
    b'xxx'
    >>> sorted(store.keys())
    ['foo', 'a/b/c']
    >>> store.listdir()
    ['a', 'foo']
    >>> store.listdir('a/b')
    ['c']
    >>> store.rmdir('a')
    >>> sorted(store.keys())
    ['foo']
    >>> import os
    >>> os.path.exists('example/a')
    False

    """  # flake8: noqa

    def __init__(self, path):

        # guard conditions
        path = os.path.abspath(path)
        if os.path.exists(path) and not os.path.isdir(path):
            raise ValueError('path exists but is not a directory')

        self.path = path

    def __getitem__(self, key):
        filepath = os.path.join(self.path, key)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                return f.read()
        else:
            raise KeyError(key)

    def __setitem__(self, key, value):

        # destination path for key
        file_path = os.path.join(self.path, key)

        # guard conditions
        if os.path.exists(file_path) and not os.path.isfile(file_path):
            raise KeyError(key)

        # ensure containing directory exists
        dir_path, file_name = os.path.split(file_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # write to temporary file
        with tempfile.NamedTemporaryFile(mode='wb', delete=False,
                                         dir=dir_path,
                                         prefix=file_name + '.',
                                         suffix='.partial') as f:
            f.write(value)
            temp_path = f.name

        # move temporary file into place
        if os.path.exists(file_path):
            os.remove(file_path)
        os.rename(temp_path, file_path)

    def __delitem__(self, key):
        file_path = os.path.join(self.path, key)
        if os.path.isfile(file_path):
            os.remove(file_path)
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
        todo = [(self.path, '')]
        while todo:
            dir_name, prefix = todo.pop()
            for name in os.listdir(dir_name):
                path = os.path.join(dir_name, name)
                if os.path.isfile(path):
                    yield prefix + name
                elif os.path.isdir(path):
                    todo.append((path, prefix + name + '/'))

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return sum(1 for _ in self.keys())

    def listdir(self, path=None):
        store_path = normalize_storage_path(path)
        dir_path = self.path
        if store_path:
            dir_path = os.path.join(dir_path, store_path)
        if os.path.isdir(dir_path):
            return sorted(os.listdir(dir_path))
        else:
            return []

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
            raise ValueError('path not found: %r' % path)


# noinspection PyPep8Naming
class ZipStore(MutableMapping):
    """TODO"""

    def __init__(self, path, compression=zipfile.ZIP_STORED,
                 allowZip64=True, mode='a'):

        # ensure zip file exists
        path = os.path.abspath(path)
        with zipfile.ZipFile(path, mode=mode):
            pass

        self.path = path
        self.compression = compression
        self.allowZip64 = allowZip64

    def __getitem__(self, key):
        with zipfile.ZipFile(self.path) as zf:
            with zf.open(key) as f:  # will raise KeyError
                return f.read()

    def __setitem__(self, key, value):
        value = ensure_bytes(value)
        with zipfile.ZipFile(self.path, mode='a',
                             compression=self.compression,
                             allowZip64=self.allowZip64) as zf:
            zf.writestr(key, value)

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
        with zipfile.ZipFile(self.path) as zf:
            keylist = sorted(zf.namelist())
        return keylist

    def keys(self):
        for key in self.keylist():
            yield key

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return sum(1 for _ in self.keys())

    def __contains__(self, key):
        with zipfile.ZipFile(self.path) as zf:
            try:
                zf.getinfo(key)
            except KeyError:
                return False
            else:
                return True

    def listdir(self, path=None):
        path = normalize_storage_path(path)
        return _listdir_from_keys(self, path)

    def getsize(self, path=None):
        path = normalize_storage_path(path)
        children = self.listdir(path)
        with zipfile.ZipFile(self.path) as zf:
            if children:
                size = 0
                with zipfile.ZipFile(self.path) as zf:
                    for child in children:
                        if path:
                            name = path + '/' + child
                        else:
                            name = child
                        try:
                            info = zf.getinfo(name)
                        except KeyError:
                            pass
                        else:
                            size += info.compress_size
                return size
            elif path:
                try:
                    info = zf.getinfo(path)
                    return info.compress_size
                except KeyError:
                    raise ValueError('path not found: %r' % path)
            else:
                return 0
