# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from collections import MutableMapping
import os
import tempfile
import json
import io
import zipfile
import shutil


import numpy as np


from zarr.util import normalize_shape, normalize_chunks, normalize_order
from zarr.compressors import get_compressor_cls
from zarr.meta import encode_metadata
from zarr.compat import PY2, binary_type


def normalize_key(key):

    # key must be something
    if not key:
        raise KeyError(key)

    # convert back slash to forward slash
    key = key.replace('\\', '/')

    # remove leading slashes
    while key[0] == '/':
        key = key[1:]

    # remove trailing slashes
    while key[-1] == '/':
        key = key[:-1]

    # collapse any repeated slashes
    previous_char = None
    normed = ''
    for char in key:
        if char != '/':
            normed += char
        elif previous_char != '/':
            normed += char
        previous_char = char

    # don't allow path segments with just '.' or '..'
    segments = normed.split('/')
    if any(s in {'.', '..'} for s in segments):
        raise ValueError("key containing '.' or '..' as path segment not "
                         "allowed")
    key = '/'.join(segments)

    # check there's something left
    if not key:
        raise KeyError(key)

    return key


def normalize_prefix(prefix):
    """TODO"""
    
    # normalise None or empty prefix
    if not prefix:
        return ''

    # normalise slashes etc.
    prefix = normalize_key(prefix)

    # add one trailing slash
    prefix += '/'

    return prefix


def array_meta_key(prefix=None):
    prefix = normalize_prefix(prefix)
    return prefix + 'meta'


def array_attrs_key(prefix=None):
    prefix = normalize_prefix(prefix)
    return prefix + 'attrs'


def group_attrs_key(prefix=None):
    prefix = normalize_prefix(prefix)
    return prefix + '.grpattrs'


def contains_array(store, prefix=None):
    """TODO"""

    # TODO review this, how to tell if a store contains an array?
    # currently we use the presence of an array metadata key as an indicator
    # that the store contains an array

    return array_meta_key(prefix) in store


def contains_group(store, prefix=None):
    """TODO"""

    # TODO review this, how to tell if a store contains a group?
    # currently we use the presence of a group attributes key as an indicator
    # that the store contains a group

    return group_attrs_key(prefix) in store


def _rmdir_from_keys(store, prefix=None):
    for key in set(store.keys()):
        if key.startswith(prefix):
            del store[key]


def rmdir(store, prefix=None):
    """TODO"""
    prefix = normalize_prefix(prefix)
    if hasattr(store, 'rmdir'):
        # pass through
        store.rmdir(prefix)
    else:
        # slow version, delete one key at a time
        _rmdir_from_keys(store, prefix)


def _listdir_from_keys(store, prefix=None):
    children = set()
    for key in store.keys():
        if key.startswith(prefix) and len(key) > len(prefix):
            suffix = key[len(prefix):]
            child = suffix.split('/')[0]
            children.add(child)
    return sorted(children)


def listdir(store, prefix=None):
    """TODO"""
    prefix = normalize_prefix(prefix)
    if hasattr(store, 'listdir'):
        # pass through
        return store.listdir(prefix)
    else:
        # slow version, iterate through all keys
        return _listdir_from_keys(store, prefix)


def init_array(store, shape, chunks, dtype=None, compression='default',
               compression_opts=None, fill_value=None,
               order='C', overwrite=False, prefix=None):
    """Initialise an array store with the given configuration.

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
        Default value to use for uninitialised portions of the array.
    order : {'C', 'F'}, optional
        Memory layout to be used within each chunk.
    overwrite : bool, optional
        If True, erase all data in `store` prior to initialisation.
    prefix : string, optional
        Prefix under which array is stored.

    Examples
    --------
    >>> from zarr.storage import init_array
    >>> store = dict()
    >>> init_array(store, shape=(10000, 10000), chunks=(1000, 1000))
    >>> sorted(store.keys())
    ['attrs', 'meta']
    >>> print(str(store['meta'], 'ascii'))
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
    >>> print(str(store['attrs'], 'ascii'))
    {}
    >>> init_array(store, shape=100000000, chunks=1000000, dtype='i1', 
    ...            prefix='foo/bar')
    >>> sorted(store.keys())
    >>> print(str(store['foo/bar/meta'], 'ascii'))

    Notes
    -----
    The initialisation process involves normalising all array metadata,
    encoding as JSON and storing under the 'meta' key. User attributes are also
    initialised and stored as JSON under the 'attrs' key.

    """

    # guard conditions
    if overwrite:
        # attempt to delete any pre-existing items in store
        rmdir(store, prefix)
    elif contains_array(store, prefix):
        raise ValueError('store contains an array')
    elif contains_group(store, prefix):
        raise ValueError('store contains a group')

    # normalise metadata
    shape = normalize_shape(shape)
    chunks = normalize_chunks(chunks, shape)
    dtype = np.dtype(dtype)
    compressor_cls = get_compressor_cls(compression)
    compression = compressor_cls.canonical_name
    compression_opts = compressor_cls.normalize_opts(
        compression_opts
    )
    order = normalize_order(order)

    # initialise metadata
    meta = dict(shape=shape, chunks=chunks, dtype=dtype,
                compression=compression, compression_opts=compression_opts,
                fill_value=fill_value, order=order)
    meta_key = array_meta_key(prefix)
    store[meta_key] = encode_metadata(meta)

    # initialise attributes
    attrs_key = array_attrs_key(prefix)
    store[attrs_key] = json.dumps(dict()).encode('ascii')


# backwards compatibility
init_store = init_array


def init_group(store, overwrite=False, prefix=None):
    """Initialise a group store.

    Parameters
    ----------
    store : MutableMapping
        A mapping that supports string keys and byte sequence values.
    overwrite : bool, optional
        If True, erase all data in `store` prior to initialisation.
    prefix : string, optional
        Prefix under which the group is stored.

    """

    # guard conditions
    if overwrite:
        # attempt to delete any pre-existing items in store
        rmdir(store, prefix)
    elif contains_array(store, prefix):
        raise ValueError('store contains an array')
    elif contains_group(store, prefix):
        raise ValueError('store contains a group')

    # initialise attributes
    attrs_key = group_attrs_key(prefix)
    store[attrs_key] = json.dumps(dict()).encode('ascii')


def ensure_bytes(s):
    if isinstance(s, binary_type):
        return s
    if hasattr(s, 'encode'):
        return s.encode()
    if hasattr(s, 'tobytes'):
        return s.tobytes()
    if PY2 and hasattr(s, 'tostring'):
        return s.tostring()
    return io.BytesIO(s).getvalue()


def _dict_store_keys(d, prefix='', cls=dict):
    for k in d.keys():
        v = d[k]
        if isinstance(v, cls):
            for sk in _dict_store_keys(v, prefix + k + '/', cls):
                yield sk
        else:
            yield prefix + k


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
        key = normalize_key(key)
        c = self.root
        for k in key.split('/'):
            c = c[k]
        if isinstance(c, self.cls):
            raise KeyError(key)
        return c

    def __setitem__(self, key, value):
        key = normalize_key(key)
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
        key = normalize_key(key)
        c = self.root
        keys = key.split('/')

        # obtain final container
        for k in keys[:-1]:
            c = c[k]

        # delete item
        del c[keys[-1]]

    def __contains__(self, key):
        key = normalize_key(key)
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

    def listdir(self, prefix=None):
        prefix = normalize_prefix(prefix)
        c = self.root
        if prefix:
            # remove trailing slash
            prefix = prefix[:-1]
            # split prefix and find container
            for k in prefix.split('/'):
                c = c[k]
        return sorted(c.keys())

    def rmdir(self, prefix=None):
        prefix = normalize_prefix(prefix)
        c = self.root
        if prefix:
            # remove trailing slash
            prefix = prefix[:-1]
            # split prefix and find container
            keys = prefix.split('/')
            for k in keys[:-1]:
                c = c[k]
            # remove final key
            del c[keys[-1]]

    def getsize(self, prefix=None):
        prefix = normalize_prefix(prefix)
        c = self.root
        if prefix:
            # remove trailing slash
            prefix = prefix[:-1]
            # split prefix and find container
            for k in prefix.split('/'):
                c = c[k]
        size = 0
        for k, v in c.items():
            if not isinstance(v, self.cls):
                try:
                    size += len(v)
                except TypeError:
                    return -1
        return size


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
        if os.path.exists and not os.path.isdir(path):
            raise ValueError('path exists but is not a directory')

        self.path = path

    def __getitem__(self, key):
        key = normalize_key(key)
        path = os.path.join(self.path, key)
        if os.path.isfile(path):
            with open(path, 'rb') as f:
                return f.read()
        else:
            raise KeyError(key)

    def __setitem__(self, key, value):

        # setup
        key = normalize_key(key)

        # destination path for key
        path = os.path.join(self.path, key)

        # guard conditions
        if os.path.exists(path) and not os.path.isfile(path):
            raise KeyError(key)

        # ensure containing directory exists
        dirname, filename = os.path.split(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        # write to temporary file
        with tempfile.NamedTemporaryFile(mode='wb', delete=False,
                                         dir=dirname,
                                         prefix=filename + '.',
                                         suffix='.partial') as f:
            f.write(value)
            temp_path = f.name

        # move temporary file into place
        if os.path.exists(path):
            os.remove(path)
        os.rename(temp_path, path)

    def __delitem__(self, key):
        key = normalize_key(key)
        path = os.path.join(self.path, key)
        if os.path.isfile(path):
            os.remove(path)
        else:
            raise KeyError(key)

    def __contains__(self, key):
        key = normalize_key(key)
        path = os.path.join(self.path, key)
        return os.path.isfile(path)

    def __eq__(self, other):
        return (
            isinstance(other, DirectoryStore) and
            self.path == other.path
        )

    def keys(self):
        dirnames = [self.path]
        while dirnames:
            dirname = dirnames.pop()
            for name in os.listdir(dirname):
                path = os.path.join(dirname, name)
                if os.path.isfile(path):
                    yield path
                elif os.path.isdir(path):
                    dirnames.append(path)

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return sum(1 for _ in self.keys())

    def listdir(self, prefix=None):
        path = self.path
        prefix = normalize_prefix(prefix)
        if prefix:
            path = os.path.join(path, prefix)
        return sorted(os.listdir(path))

    def rmdir(self, prefix=None):
        path = self.path
        prefix = normalize_prefix(prefix)
        if prefix:
            path = os.path.join(path, prefix)
        if os.path.isdir(path):
            shutil.rmtree(path)

    def getsize(self, prefix=None):
        prefix = normalize_prefix(prefix)
        children = self.listdir(prefix)
        size = 0
        for child in children:
            path = os.path.join(self.path, prefix, child)
            if os.path.isfile(path):
                size += os.path.getsize(path)
        return size


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
        key = normalize_key(key)
        with zipfile.ZipFile(self.path) as zf:
            with zf.open(key) as f:  # will raise KeyError
                return f.read()

    def __setitem__(self, key, value):
        key = normalize_key(key)
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
        key = normalize_key(key)
        with zipfile.ZipFile(self.path) as zf:
            try:
                zf.getinfo(key)
            except KeyError:
                return False
            else:
                return True

    def listdir(self, prefix=None):
        prefix = normalize_prefix(prefix)
        return _listdir_from_keys(self, prefix)

    def rmdir(self, prefix=None):
        raise NotImplementedError

    def getsize(self, prefix=None):
        prefix = normalize_prefix(prefix)
        children = self.listdir(prefix)
        size = 0
        with zipfile.ZipFile(self.path) as zf:
            for child in children:
                name = prefix + child
                try:
                    info = zf.getinfo(name)
                except KeyError:
                    pass
                else:
                    size += info.compress_size
        return size
