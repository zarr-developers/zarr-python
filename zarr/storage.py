# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from collections import MutableMapping
import os
import tempfile
import json
import io
import zipfile


import numpy as np


from zarr.util import normalize_shape, normalize_chunks, normalize_order
from zarr.compressors import get_compressor_cls
from zarr.meta import encode_metadata
from zarr.compat import PY2, binary_type


def normalize_prefix(prefix):
    """TODO"""
    
    # map None to empty string
    if prefix is None:
        return ''

    # convert back slash to forward slash
    prefix = prefix.replace('\\', '/')

    # remove leading slashes
    while prefix[0] == '/':
        prefix = prefix[1:]

    # remove trailing slashes
    while prefix[-1] == '/':
        prefix = prefix[:-1]

    # collapse any repeated slashes
    previous_char = None
    normed = ''
    for char in prefix:
        if previous_char != '/':
            normed += char
        previous_char = char

    # don't allow path segments with just '.' or '..'
    segments = normed.split('/')
    if any(s in {'.', '..'} for s in segments):
        raise ValueError("prefix containing '.' or '..' not allowed")
    prefix = '/'.join(segments)

    # check there's something left
    if prefix:
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


def rm(store, prefix=None):
    """TODO"""
    prefix = normalize_prefix(prefix)
    if hasattr(store, 'rm'):
        store.rm(prefix)
    else:
        for key in set(store.keys()):
            if key.startswith(prefix):
                del store[key]


def ls(store, prefix=None):
    """TODO"""
    prefix = normalize_prefix(prefix)
    if hasattr(store, 'ls'):
        return store.ls(prefix)
    else:
        children = set()
        for key in store.keys():
            if key.startswith(prefix) and len(key) > len(prefix):
                suffix = key[len(prefix):]
                child = suffix.split('/')[0]
                children.add(child)
        return children


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
        rm(store, prefix)
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
        rm(store, prefix)
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
    >>> store = zarr.DirectoryStore('example.zarr')
    >>> zarr.init_array(store, shape=(10000, 10000), chunks=(1000, 1000),
    ...                 fill_value=0, overwrite=True)
    >>> import os
    >>> sorted(os.listdir('example.zarr'))
    ['attrs', 'meta']
    >>> print(open('example.zarr/meta').read())
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
        "fill_value": 0,
        "order": "C",
        "shape": [
            10000,
            10000
        ],
        "zarr_format": 1
    }
    >>> print(open('example.zarr/attrs').read())
    {}
    >>> z = zarr.Array(store)
    >>> z
    zarr.core.Array((10000, 10000), float64, chunks=(1000, 1000), order=C)
      compression: blosc; compression_opts: {'clevel': 5, 'cname': 'lz4', 'shuffle': 1}
      nbytes: 762.9M; nbytes_stored: 313; ratio: 2555910.5; initialized: 0/100
      store: zarr.storage.DirectoryStore
    >>> z[:] = 1
    >>> len(os.listdir('example.zarr'))
    102
    >>> sorted(os.listdir('example.zarr'))[0:5]
    ['0.0', '0.1', '0.2', '0.3', '0.4']
    >>> print(open('example.zarr/0.0', 'rb').read(10))
    b'\\x02\\x01!\\x08\\x00\\x12z\\x00\\x00\\x80'

    See Also
    --------
    zarr.creation.open

    """  # flake8: noqa

    def __init__(self, path):

        # guard conditions
        path = os.path.abspath(path)
        if os.path.exists and not os.path.isdir(path):
            raise ValueError('path exists but is not a directory')

        self.path = path

    def abspath(self, name):
        if any(sep in name for sep in '/\\'):
            raise ValueError('invalid name: %s' % name)
        return os.path.join(self.path, name)

    def __getitem__(self, key):
        path = os.path.join(self.path, key)
        if os.path.isfile(path):
            with open(path, 'rb') as f:
                return f.read()
        else:
            raise KeyError(key)

    def __setitem__(self, key, value):
        # accept any value that can be written to a file

        # destination path for key
        path = os.path.join(self.path, key)

        # guard conditions
        if os.path.isdir(path):
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
        path = os.path.join(self.path, key)
        if os.path.isfile(path):
            os.remove(path)
        else:
            raise KeyError(key)

    def __contains__(self, key):
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

    @property
    def nbytes_stored(self):
        """Total size of all values in number of bytes."""
        paths = (os.path.join(self.path, key) for key in self.keys())
        return sum(os.path.getsize(path)
                   for path in paths
                   if os.path.isfile(path))


# noinspection PyPep8Naming
class ZipStore(MutableMapping, HierarchicalStore):
    """TODO"""

    def __init__(self, path, arcpath=None, compression=zipfile.ZIP_STORED,
                 allowZip64=True, mode='a'):

        # guard conditions
        path = os.path.abspath(path)
        # ensure zip file exists
        with zipfile.ZipFile(path, mode=mode):
            pass
        self.path = path

        # sanitize/normalize arcpath, maybe not needed?
        if arcpath:
            arcpath = os.path.normpath(os.path.splitdrive(arcpath)[1])
            while arcpath[0] in (os.sep, os.altsep):
                arcpath= arcpath[1:]
            while arcpath[-1] in (os.sep, os.altsep):
                arcpath= arcpath[:-1]
            if os.sep != "/" and os.sep in arcpath:
                arcpath = arcpath.replace(os.sep, "/")
        self.arcpath = arcpath

        self.compression = compression
        self.allowZip64 = allowZip64

    def arcname(self, key):
        if any(sep in key for sep in '/\\'):
            raise ValueError('invalid key: %s' % key)
        if self.arcpath:
            return '/'.join([self.arcpath, key])
        else:
            return key

    def __getitem__(self, key):
        arcname = self.arcname(key)
        with zipfile.ZipFile(self.path) as zf:
            with zf.open(arcname) as f:  # will raise KeyError
                return f.read()

    def __setitem__(self, key, value):
        # accept any value that can be written to a zip file

        # destination path for key
        arcname = self.arcname(key)

        # ensure bytes
        value = ensure_bytes(value)

        # write to archive
        with zipfile.ZipFile(self.path, mode='a',
                             compression=self.compression,
                             allowZip64=self.allowZip64) as zf:
            zf.writestr(arcname, value)

    def __delitem__(self, key):
        raise NotImplementedError

    def __eq__(self, other):
        return (
            isinstance(other, ZipStore) and
            self.path == other.path and
            self.arcpath == other.arcpath and
            self.compression == other.compression and
            self.allowZip64 == other.allowZip64
        )

    def keyset(self):
        if self.arcpath:
            prefix = self.arcpath + '/'
        else:
            prefix = ''
        keyset = set()
        with zipfile.ZipFile(self.path) as zf:
            for name in zf.namelist():
                if name.startswith(prefix) and len(name) > len(prefix):
                    suffix = name[len(prefix):]
                    key = suffix.split('/')[0]
                    keyset.add(key)
        return keyset

    def keys(self):
        for k in self.keyset():
            yield k

    def store_keyset(self):
        if self.arcpath:
            prefix = self.arcpath + '/'
        else:
            prefix = ''
        store_keyset = set()
        with zipfile.ZipFile(self.path) as zf:
            for name in zf.namelist():
                if name.startswith(prefix) and len(name) > len(prefix):
                    suffix = name[len(prefix):]
                    if '/' in suffix:
                        key = suffix.split('/')[0]
                        store_keyset.add(key)
        return store_keyset

    def stores(self):
        if self.arcpath:
            prefix = self.arcpath + '/'
        else:
            prefix = ''
        for k in self.store_keyset():
            yield k, ZipStore(self.path,
                              arcpath=prefix+k,
                              compression=self.compression,
                              allowZip64=self.allowZip64)

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return sum(1 for _ in self.keys())

    def __contains__(self, key):
        return key in self.keyset()

    def get_store(self, key):
        arcname = self.arcname(key)

        # guard condition
        with zipfile.ZipFile(self.path) as zf:
            try:
                zf.getinfo(arcname)
            except KeyError:
                pass
            else:
                # key refers to file in archive
                raise KeyError(key)

        return ZipStore(self.path, arcpath=arcname,
                        compression=self.compression,
                        allowZip64=self.allowZip64)

    def require_store(self, key):
        # can't really create directories in a zip file
        return self.get_store(key)

    @property
    def nbytes_stored(self):
        """Total size of all values in number of bytes."""
        n = 0
        if self.arcpath:
            prefix = self.arcpath + '/'
        else:
            prefix = ''
        with zipfile.ZipFile(self.path) as zf:
            for name in zf.namelist():
                if name.startswith(prefix) and len(name) > len(prefix):
                    suffix = name[len(prefix):]
                    if '/' not in suffix:
                        info = zf.getinfo(name)
                        n += info.compress_size
        return n
