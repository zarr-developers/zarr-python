# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from collections import MutableMapping
import os
import tempfile
import json
import zipfile
import shutil
import atexit


import numpy as np


from zarr.util import normalize_shape, normalize_chunks, normalize_order, \
    normalize_storage_path, buffer_size
from zarr.meta import encode_array_metadata, encode_group_metadata
from zarr.compat import PY2, binary_type
from zarr.codecs import codec_registry
from zarr.errors import err_contains_group, err_contains_array, \
    err_path_not_found, err_bad_compressor, err_fspath_exists_notdir, \
    err_read_only


array_meta_key = '.zarray'
group_meta_key = '.zgroup'
attrs_key = '.zattrs'
try:
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
    """Remove all items under the given path."""
    path = normalize_storage_path(path)
    if hasattr(store, 'rmdir'):
        # pass through
        store.rmdir(path)
    else:
        # slow version, delete one key at a time
        _rmdir_from_keys(store, path)


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
    """Obtain a directory listing for the given path."""
    path = normalize_storage_path(path)
    if hasattr(store, 'listdir'):
        # pass through
        return store.listdir(path)
    else:
        # slow version, iterate through all keys
        return _listdir_from_keys(store, path)
    
    
def getsize(store, path=None):
    """Compute size of stored items for a given path."""
    path = normalize_storage_path(path)
    if hasattr(store, 'getsize'):
        # pass through
        return store.getsize(path)
    elif isinstance(store, dict):
        # compute from size of values
        prefix = _path_to_prefix(path)
        size = 0
        for k in listdir(store, path):
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


def init_array(store, shape, chunks=None, dtype=None, compressor='default',
               fill_value=None, order='C', overwrite=False, path=None,
               chunk_store=None, filters=None):
    """initialize an array store with the given configuration.

    Parameters
    ----------
    store : MutableMapping
        A mapping that supports string keys and bytes-like values.
    shape : int or tuple of ints
        Array shape.
    chunks : int or tuple of ints, optional
        Chunk shape. If not provided, will be guessed from `shape` and `dtype`.
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

    Examples
    --------
    Initialize an array store::

        >>> from zarr.storage import init_array
        >>> store = dict()
        >>> init_array(store, shape=(10000, 10000), chunks=(1000, 1000))
        >>> sorted(store.keys())
        ['.zarray', '.zattrs']

    Array metadata is stored as JSON::

        >>> print(str(store['.zarray'], 'ascii'))
        {
            "chunks": [
                1000,
                1000
            ],
            "compressor": {
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

    User-defined attributes are also stored as JSON, initially empty::

        >>> print(str(store['.zattrs'], 'ascii'))
        {}

    Initialize an array using a storage path::

        >>> store = dict()
        >>> init_array(store, shape=100000000, chunks=1000000, dtype='i1',
        ...            path='foo')
        >>> sorted(store.keys())
        ['.zattrs', '.zgroup', 'foo/.zarray', 'foo/.zattrs']
        >>> print(str(store['foo/.zarray'], 'ascii'))
        {
            "chunks": [
                1000000
            ],
            "compressor": {
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
    The initialisation process involves normalising all array metadata,
    encoding as JSON and storing under the '.zarray' key. User attributes are 
    also initialized and stored as JSON under the '.zattrs' key.

    """

    # normalize path
    path = normalize_storage_path(path)
    
    # ensure parent group initialized
    _require_parent_group(path, store=store, chunk_store=chunk_store,
                          overwrite=overwrite)

    _init_array_metadata(store, shape=shape, chunks=chunks, dtype=dtype,
                         compressor=compressor, fill_value=fill_value,
                         order=order, overwrite=overwrite, path=path,
                         chunk_store=chunk_store, filters=filters)


def _init_array_metadata(store, shape, chunks=None, dtype=None,
                         compressor='default',
                         fill_value=None, order='C', overwrite=False,
                         path=None, chunk_store=None, filters=None):

    # guard conditions
    if overwrite:
        # attempt to delete any pre-existing items in store
        rmdir(store, path)
        if chunk_store is not None and chunk_store != store:
            rmdir(chunk_store, path)
    elif contains_array(store, path):
        err_contains_array(path)
    elif contains_group(store, path):
        err_contains_group(path)

    # normalize metadata
    shape = normalize_shape(shape)
    dtype = np.dtype(dtype)
    chunks = normalize_chunks(chunks, shape, dtype.itemsize)
    order = normalize_order(order)

    # obtain compressor config
    if compressor == 'none':
        # compatibility
        compressor = None
    elif compressor == 'default':
        compressor = default_compressor
    if compressor:
        try:
            compressor_config = compressor.get_config()
        except AttributeError:
            err_bad_compressor(compressor)
    else:
        compressor_config = None

    # obtain filters config
    if filters:
        filters_config = [f.get_config() for f in filters]
    else:
        filters_config = None

    # initialize metadata
    meta = dict(shape=shape, chunks=chunks, dtype=dtype,
                compressor=compressor_config, fill_value=fill_value,
                order=order, filters=filters_config)
    key = _path_to_prefix(path) + array_meta_key
    store[key] = encode_array_metadata(meta)

    # initialize attributes
    key = _path_to_prefix(path) + attrs_key
    store[key] = json.dumps(dict()).encode('ascii')


# backwards compatibility
init_store = init_array


def init_group(store, overwrite=False, path=None, chunk_store=None):
    """initialize a group store.

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
        if chunk_store is not None and chunk_store != store:
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

    # initialize attributes
    key = _path_to_prefix(path) + attrs_key
    store[key] = json.dumps(dict()).encode('ascii')


def ensure_bytes(s):
    if isinstance(s, binary_type):
        return s
    if hasattr(s, 'tobytes'):
        return s.tobytes()
    if PY2 and hasattr(s, 'tostring'):  # pragma: no cover
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
    """Extended mutable mapping interface to a hierarchy of dicts.

    Examples
    --------
    >>> import zarr
    >>> store = zarr.DictStore()
    >>> store['foo'] = b'bar'
    >>> store['foo']
    b'bar'
    >>> store['a/b/c'] = b'xxx'
    >>> store['a/b/c']
    b'xxx'
    >>> sorted(store.keys())
    ['a/b/c', 'foo']
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
        parent, key = self._require_parent(item)
        parent[key] = value

    def __delitem__(self, item):
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
        if path:
            try:
                parent, key = self._get_parent(path)
                value = parent[key]
            except KeyError:
                err_path_not_found(path)
        else:
            value = self.root

        # obtain size of value
        if isinstance(value, self.cls):
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
    >>> store = zarr.DirectoryStore('example_store')
    >>> store['foo'] = b'bar'
    >>> store['foo']
    b'bar'
    >>> open('example_store/foo', 'rb').read()
    b'bar'
    >>> store['a/b/c'] = b'xxx'
    >>> store['a/b/c']
    b'xxx'
    >>> open('example_store/a/b/c', 'rb').read()
    b'xxx'
    >>> sorted(store.keys())
    ['a/b/c', 'foo']
    >>> store.listdir()
    ['a', 'foo']
    >>> store.listdir('a/b')
    ['c']
    >>> store.rmdir('a')
    >>> sorted(store.keys())
    ['foo']
    >>> import os
    >>> os.path.exists('example_store/a')
    False

    """  # flake8: noqa

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
            err_path_not_found(path)


def atexit_rmtree(path,
                  isdir=os.path.isdir,
                  rmtree=shutil.rmtree):  # pragma: no cover
    """Ensure directory removal at interpreter exit."""
    if isdir(path):
        rmtree(path)


class TempStore(DirectoryStore):
    """Directory store using a temporary directory for storage."""

    def __init__(self, suffix='', prefix='zarr', dir=None):
        path = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
        atexit.register(atexit_rmtree, path)
        super(TempStore, self).__init__(path)


# noinspection PyPep8Naming
class ZipStore(MutableMapping):
    """Mutable Mapping interface to a Zip file. Keys must be strings,
    values must be bytes-like objects.

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
    >>> import zarr
    >>> store = zarr.ZipStore('example.zip', mode='w')
    >>> store['foo'] = b'bar'
    >>> store['foo']
    b'bar'
    >>> store['a/b/c'] = b'xxx'
    >>> store['a/b/c']
    b'xxx'
    >>> sorted(store.keys())
    ['a/b/c', 'foo']
    >>> store.close()
    >>> import zipfile
    >>> zf = zipfile.ZipFile('example.zip', mode='r')
    >>> sorted(zf.namelist())
    ['a/b/c', 'foo']

    Notes
    -----
    When modifying a ZipStore the close() method must be called otherwise
    essential data will not be written to the underlying zip file. The
    ZipStore class also supports the context manager protocol, which ensures
    the close() method is called on leaving the with statement.

    """

    def __init__(self, path, compression=zipfile.ZIP_STORED,
                 allowZip64=True, mode='a'):

        # store properties
        path = os.path.abspath(path)
        self.path = path
        self.compression = compression
        self.allowZip64 = allowZip64
        self.mode = mode

        # open zip file
        self.zf = zipfile.ZipFile(path, mode=mode,
                                  compression=compression,
                                  allowZip64=allowZip64)

    def __getstate__(self):
        return self.path, self.compression, self.allowZip64, self.mode

    def __setstate__(self, state):
        self.__init__(*state)

    def close(self):
        """Closes the underlying zip file, ensuring all records are written."""
        self.zf.close()

    def flush(self):
        """Closes the underlying zip file, ensuring all records are written,
        then re-opens the file for further modifications."""
        self.zf.close()
        self.zf = zipfile.ZipFile(self.path, mode=self.mode,
                                  compression=self.compression,
                                  allowZip64=self.allowZip64)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __getitem__(self, key):
        with self.zf.open(key) as f:  # will raise KeyError
            return f.read()

    def __setitem__(self, key, value):
        if self.mode == 'r':
            err_read_only()
        value = ensure_bytes(value)
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
                err_path_not_found(path)
        else:
            return 0


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
