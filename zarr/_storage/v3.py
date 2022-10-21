import os
import shutil
from collections import OrderedDict
from collections.abc import MutableMapping
from threading import Lock
from typing import Union, Dict, Any

from zarr.errors import (
    MetadataError,
    ReadOnlyError,
)
from zarr.util import (buffer_size, json_loads, normalize_storage_path)

from zarr._storage.absstore import ABSStoreV3  # noqa: F401
from zarr._storage.store import (_get_hierarchy_metadata,  # noqa: F401
                                 _get_metadata_suffix,
                                 _listdir_from_keys,
                                 _rename_from_keys,
                                 _rename_metadata_v3,
                                 _rmdir_from_keys,
                                 _rmdir_from_keys_v3,
                                 _path_to_prefix,
                                 _prefix_to_array_key,
                                 _prefix_to_group_key,
                                 array_meta_key,
                                 attrs_key,
                                 data_root,
                                 group_meta_key,
                                 meta_root,
                                 BaseStore,
                                 Store,
                                 StoreV3)
from zarr.storage import (DBMStore, ConsolidatedMetadataStore, DirectoryStore, FSStore, KVStore,
                          LMDBStore, LRUStoreCache, MemoryStore, MongoDBStore, RedisStore,
                          SQLiteStore, ZipStore, _getsize)

__doctest_requires__ = {
    ('RedisStore', 'RedisStore.*'): ['redis'],
    ('MongoDBStore', 'MongoDBStore.*'): ['pymongo'],
    ('LRUStoreCache', 'LRUStoreCache.*'): ['s3fs'],
}


try:
    # noinspection PyUnresolvedReferences
    from zarr.codecs import Blosc
    default_compressor = Blosc()
except ImportError:  # pragma: no cover
    from zarr.codecs import Zlib
    default_compressor = Zlib()


Path = Union[str, bytes, None]
# allow MutableMapping for backwards compatibility
StoreLike = Union[BaseStore, MutableMapping]


class RmdirV3():
    """Mixin class that can be used to ensure override of any existing v2 rmdir class."""

    def rmdir(self, path: str = "") -> None:
        path = normalize_storage_path(path)
        _rmdir_from_keys_v3(self, path)  # type: ignore


class KVStoreV3(RmdirV3, KVStore, StoreV3):

    def list(self):
        return list(self._mutable_mapping.keys())

    def __setitem__(self, key, value):
        self._validate_key(key)
        super().__setitem__(key, value)

    def __eq__(self, other):
        return (
            isinstance(other, KVStoreV3) and
            self._mutable_mapping == other._mutable_mapping
        )


KVStoreV3.__doc__ = KVStore.__doc__


def _get_files_and_dirs_from_path(store, path):
    path = normalize_storage_path(path)

    files = []
    # add array metadata file if present
    array_key = _prefix_to_array_key(store, path)
    if array_key in store:
        files.append(os.path.join(store.path, array_key))

    # add group metadata file if present
    group_key = _prefix_to_group_key(store, path)
    if group_key in store:
        files.append(os.path.join(store.path, group_key))

    dirs = []
    # add array and group folders if present
    for d in [data_root + path, meta_root + path]:
        dir_path = os.path.join(store.path, d)
        if os.path.exists(dir_path):
            dirs.append(dir_path)
    return files, dirs


class FSStoreV3(FSStore, StoreV3):

    # FSStoreV3 doesn't use this (FSStore uses it within _normalize_key)
    _META_KEYS = ()

    def __setitem__(self, key, value):
        self._validate_key(key)
        super().__setitem__(key, value)

    def _default_key_separator(self):
        if self.key_separator is None:
            self.key_separator = "/"

    def list(self):
        return list(self.keys())

    def _normalize_key(self, key):
        key = normalize_storage_path(key).lstrip('/')
        return key.lower() if self.normalize_keys else key

    def getsize(self, path=None):
        size = 0
        if path is None or path == '':
            # size of both the data and meta subdirs
            dirs = []
            for d in ['data/root', 'meta/root']:
                dir_path = os.path.join(self.path, d)
                if os.path.exists(dir_path):
                    dirs.append(dir_path)
        elif path in self:
            # access individual element by full path
            return buffer_size(self[path])
        else:
            files, dirs = _get_files_and_dirs_from_path(self, path)
            for file in files:
                size += os.path.getsize(file)
        for d in dirs:
            size += self.fs.du(d, total=True, maxdepth=None)
        return size

    def setitems(self, values):
        if self.mode == 'r':
            raise ReadOnlyError()
        values = {self._normalize_key(key): val for key, val in values.items()}

        # initialize the /data/root/... folder corresponding to the array!
        # Note: zarr.tests.test_core_v3.TestArrayWithFSStoreV3PartialRead fails
        # without this explicit creation of directories
        subdirectories = set(os.path.dirname(v) for v in values.keys())
        for subdirectory in subdirectories:
            data_dir = os.path.join(self.path, subdirectory)
            if not self.fs.exists(data_dir):
                self.fs.mkdir(data_dir)

        self.map.setitems(values)

    def rmdir(self, path=None):
        if self.mode == 'r':
            raise ReadOnlyError()
        if path:
            for base in [meta_root, data_root]:
                store_path = self.dir_path(base + path)
                if self.fs.isdir(store_path):
                    self.fs.rm(store_path, recursive=True)

            # remove any associated metadata files
            sfx = _get_metadata_suffix(self)
            meta_dir = (meta_root + path).rstrip('/')
            array_meta_file = meta_dir + '.array' + sfx
            self.pop(array_meta_file, None)
            group_meta_file = meta_dir + '.group' + sfx
            self.pop(group_meta_file, None)
        else:
            store_path = self.dir_path(path)
            if self.fs.isdir(store_path):
                self.fs.rm(store_path, recursive=True)


class MemoryStoreV3(MemoryStore, StoreV3):

    def __init__(self, root=None, cls=dict, dimension_separator=None):
        if root is None:
            self.root = cls()
        else:
            self.root = root
        self.cls = cls
        self.write_mutex = Lock()
        self._dimension_separator = dimension_separator  # TODO: modify for v3?

    def __eq__(self, other):
        return (
            isinstance(other, MemoryStoreV3) and
            self.root == other.root and
            self.cls == other.cls
        )

    def __setitem__(self, key, value):
        self._validate_key(key)
        super().__setitem__(key, value)

    def list(self):
        return list(self.keys())

    def getsize(self, path: Path = None):
        return _getsize(self, path)

    def rename(self, src_path: Path, dst_path: Path):
        src_path = normalize_storage_path(src_path)
        dst_path = normalize_storage_path(dst_path)

        any_renamed = False
        for base in [meta_root, data_root]:
            if self.list_prefix(base + src_path):
                src_parent, src_key = self._get_parent(base + src_path)
                dst_parent, dst_key = self._require_parent(base + dst_path)

                if src_key in src_parent:
                    dst_parent[dst_key] = src_parent.pop(src_key)

                if base == meta_root:
                    # check for and move corresponding metadata
                    sfx = _get_metadata_suffix(self)
                    src_meta = src_key + '.array' + sfx
                    if src_meta in src_parent:
                        dst_meta = dst_key + '.array' + sfx
                        dst_parent[dst_meta] = src_parent.pop(src_meta)
                    src_meta = src_key + '.group' + sfx
                    if src_meta in src_parent:
                        dst_meta = dst_key + '.group' + sfx
                        dst_parent[dst_meta] = src_parent.pop(src_meta)
                any_renamed = True
        any_renamed = _rename_metadata_v3(self, src_path, dst_path) or any_renamed
        if not any_renamed:
            raise ValueError(f"no item {src_path} found to rename")

    def rmdir(self, path: Path = None):
        path = normalize_storage_path(path)
        if path:
            for base in [meta_root, data_root]:
                try:
                    parent, key = self._get_parent(base + path)
                    value = parent[key]
                except KeyError:
                    continue
                else:
                    if isinstance(value, self.cls):
                        del parent[key]

            # remove any associated metadata files
            sfx = _get_metadata_suffix(self)
            meta_dir = (meta_root + path).rstrip('/')
            array_meta_file = meta_dir + '.array' + sfx
            self.pop(array_meta_file, None)
            group_meta_file = meta_dir + '.group' + sfx
            self.pop(group_meta_file, None)
        else:
            # clear out root
            self.root = self.cls()


MemoryStoreV3.__doc__ = MemoryStore.__doc__


class DirectoryStoreV3(DirectoryStore, StoreV3):

    def list(self):
        return list(self.keys())

    def __eq__(self, other):
        return (
            isinstance(other, DirectoryStoreV3) and
            self.path == other.path
        )

    def __setitem__(self, key, value):
        self._validate_key(key)
        super().__setitem__(key, value)

    def getsize(self, path: Path = None):
        return _getsize(self, path)

    def rename(self, src_path, dst_path, metadata_key_suffix='.json'):
        store_src_path = normalize_storage_path(src_path)
        store_dst_path = normalize_storage_path(dst_path)

        dir_path = self.path
        any_existed = False
        for root_prefix in ['meta', 'data']:
            src_path = os.path.join(dir_path, root_prefix, 'root', store_src_path)
            if os.path.exists(src_path):
                any_existed = True
                dst_path = os.path.join(dir_path, root_prefix, 'root', store_dst_path)
                os.renames(src_path, dst_path)

        for suffix in ['.array' + metadata_key_suffix,
                       '.group' + metadata_key_suffix]:
            src_meta = os.path.join(dir_path, 'meta', 'root', store_src_path + suffix)
            if os.path.exists(src_meta):
                any_existed = True
                dst_meta = os.path.join(dir_path, 'meta', 'root', store_dst_path + suffix)
                dst_dir = os.path.dirname(dst_meta)
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                os.rename(src_meta, dst_meta)
        if not any_existed:
            raise FileNotFoundError("nothing found at src_path")

    def rmdir(self, path=None):
        store_path = normalize_storage_path(path)
        dir_path = self.path
        if store_path:
            for base in [meta_root, data_root]:
                dir_path = os.path.join(dir_path, base + store_path)
                if os.path.isdir(dir_path):
                    shutil.rmtree(dir_path)

            # remove any associated metadata files
            sfx = _get_metadata_suffix(self)
            meta_dir = (meta_root + path).rstrip('/')
            array_meta_file = meta_dir + '.array' + sfx
            self.pop(array_meta_file, None)
            group_meta_file = meta_dir + '.group' + sfx
            self.pop(group_meta_file, None)

        elif os.path.isdir(dir_path):
            shutil.rmtree(dir_path)


DirectoryStoreV3.__doc__ = DirectoryStore.__doc__


class ZipStoreV3(ZipStore, StoreV3):

    def list(self):
        return list(self.keys())

    def __eq__(self, other):
        return (
            isinstance(other, ZipStore) and
            self.path == other.path and
            self.compression == other.compression and
            self.allowZip64 == other.allowZip64
        )

    def __setitem__(self, key, value):
        self._validate_key(key)
        super().__setitem__(key, value)

    def getsize(self, path=None):
        path = normalize_storage_path(path)
        with self.mutex:
            children = self.list_prefix(data_root + path)
            children += self.list_prefix(meta_root + path)
            print(f"path={path}, children={children}")
            if children:
                size = 0
                for name in children:
                    info = self.zf.getinfo(name)
                    size += info.compress_size
                return size
            elif path in self:
                info = self.zf.getinfo(path)
                return info.compress_size
            else:
                return 0


ZipStoreV3.__doc__ = ZipStore.__doc__


class RedisStoreV3(RmdirV3, RedisStore, StoreV3):

    def list(self):
        return list(self.keys())

    def __setitem__(self, key, value):
        self._validate_key(key)
        super().__setitem__(key, value)


RedisStoreV3.__doc__ = RedisStore.__doc__


class MongoDBStoreV3(RmdirV3, MongoDBStore, StoreV3):

    def list(self):
        return list(self.keys())

    def __setitem__(self, key, value):
        self._validate_key(key)
        super().__setitem__(key, value)


MongoDBStoreV3.__doc__ = MongoDBStore.__doc__


class DBMStoreV3(RmdirV3, DBMStore, StoreV3):

    def list(self):
        return list(self.keys())

    def __setitem__(self, key, value):
        self._validate_key(key)
        super().__setitem__(key, value)


DBMStoreV3.__doc__ = DBMStore.__doc__


class LMDBStoreV3(RmdirV3, LMDBStore, StoreV3):

    def list(self):
        return list(self.keys())

    def __setitem__(self, key, value):
        self._validate_key(key)
        super().__setitem__(key, value)


LMDBStoreV3.__doc__ = LMDBStore.__doc__


class SQLiteStoreV3(SQLiteStore, StoreV3):

    def list(self):
        return list(self.keys())

    def getsize(self, path=None):
        # TODO: why does the query below not work in this case?
        #       For now fall back to the default _getsize implementation
        # size = 0
        # for _path in [data_root + path, meta_root + path]:
        #     c = self.cursor.execute(
        #         '''
        #         SELECT COALESCE(SUM(LENGTH(v)), 0) FROM zarr
        #         WHERE k LIKE (? || "%") AND
        #               0 == INSTR(LTRIM(SUBSTR(k, LENGTH(?) + 1), "/"), "/")
        #         ''',
        #         (_path, _path)
        #     )
        #     for item_size, in c:
        #         size += item_size
        # return size

        # fallback to default implementation for now
        return _getsize(self, path)

    def __setitem__(self, key, value):
        self._validate_key(key)
        super().__setitem__(key, value)

    def rmdir(self, path=None):
        path = normalize_storage_path(path)
        if path:
            for base in [meta_root, data_root]:
                with self.lock:
                    self.cursor.execute(
                        'DELETE FROM zarr WHERE k LIKE (? || "/%")', (base + path,)
                    )
            # remove any associated metadata files
            sfx = _get_metadata_suffix(self)
            meta_dir = (meta_root + path).rstrip('/')
            array_meta_file = meta_dir + '.array' + sfx
            self.pop(array_meta_file, None)
            group_meta_file = meta_dir + '.group' + sfx
            self.pop(group_meta_file, None)
        else:
            self.clear()


SQLiteStoreV3.__doc__ = SQLiteStore.__doc__


class LRUStoreCacheV3(RmdirV3, LRUStoreCache, StoreV3):

    def __init__(self, store, max_size: int):
        self._store = StoreV3._ensure_store(store)
        self._max_size = max_size
        self._current_size = 0
        self._keys_cache = None
        self._contains_cache = None
        self._listdir_cache: Dict[Path, Any] = dict()
        self._values_cache: Dict[Path, Any] = OrderedDict()
        self._mutex = Lock()
        self.hits = self.misses = 0

    def list(self):
        return list(self.keys())

    def __setitem__(self, key, value):
        self._validate_key(key)
        super().__setitem__(key, value)


LRUStoreCacheV3.__doc__ = LRUStoreCache.__doc__


class ConsolidatedMetadataStoreV3(ConsolidatedMetadataStore, StoreV3):
    """A layer over other storage, where the metadata has been consolidated into
    a single key.

    The purpose of this class, is to be able to get all of the metadata for
    a given array in a single read operation from the underlying storage.
    See :func:`zarr.convenience.consolidate_metadata` for how to create this
    single metadata key.

    This class loads from the one key, and stores the data in a dict, so that
    accessing the keys no longer requires operations on the backend store.

    This class is read-only, and attempts to change the array metadata will
    fail, but changing the data is possible. If the backend storage is changed
    directly, then the metadata stored here could become obsolete, and
    :func:`zarr.convenience.consolidate_metadata` should be called again and the class
    re-invoked. The use case is for write once, read many times.

    .. note:: This is an experimental feature.

    Parameters
    ----------
    store: Store
        Containing the zarr array.
    metadata_key: str
        The target in the store where all of the metadata are stored. We
        assume JSON encoding.

    See Also
    --------
    zarr.convenience.consolidate_metadata, zarr.convenience.open_consolidated

    """

    def __init__(self, store: StoreLike, metadata_key=meta_root + "consolidated/.zmetadata"):
        self.store = StoreV3._ensure_store(store)

        # retrieve consolidated metadata
        meta = json_loads(self.store[metadata_key])

        # check format of consolidated metadata
        consolidated_format = meta.get('zarr_consolidated_format', None)
        if consolidated_format != 1:
            raise MetadataError('unsupported zarr consolidated metadata format: %s' %
                                consolidated_format)

        # decode metadata
        self.meta_store: Store = KVStoreV3(meta["metadata"])

    def rmdir(self, key):
        raise ReadOnlyError()


def _normalize_store_arg_v3(store: Any, storage_options=None, mode="r") -> BaseStore:
    # default to v2 store for backward compatibility
    zarr_version = getattr(store, '_store_version', 3)
    if zarr_version != 3:
        raise ValueError("store must be a version 3 store")
    if store is None:
        store = KVStoreV3(dict())
        # add default zarr.json metadata
        store['zarr.json'] = store._metadata_class.encode_hierarchy_metadata(None)
        return store
    if isinstance(store, os.PathLike):
        store = os.fspath(store)
    if isinstance(store, str):
        if "://" in store or "::" in store:
            store = FSStoreV3(store, mode=mode, **(storage_options or {}))
        elif storage_options:
            raise ValueError("storage_options passed with non-fsspec path")
        elif store.endswith('.zip'):
            store = ZipStoreV3(store, mode=mode)
        elif store.endswith('.n5'):
            raise NotImplementedError("N5Store not yet implemented for V3")
            # return N5StoreV3(store)
        else:
            store = DirectoryStoreV3(store)
        # add default zarr.json metadata
        store['zarr.json'] = store._metadata_class.encode_hierarchy_metadata(None)
        return store
    else:
        store = StoreV3._ensure_store(store)
        if 'zarr.json' not in store:
            # add default zarr.json metadata
            store['zarr.json'] = store._metadata_class.encode_hierarchy_metadata(None)
    return store
