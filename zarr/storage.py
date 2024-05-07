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

import atexit
import errno
import glob
import multiprocessing
import operator
import os
import re
import shutil
import sys
import tempfile
import warnings
import zipfile
from collections import OrderedDict
from collections.abc import MutableMapping
from functools import lru_cache
from os import scandir
from pickle import PicklingError
from threading import Lock, RLock
from typing import Sequence, Mapping, Optional, Union, List, Tuple, Dict, Any
import uuid
import time

from numcodecs.abc import Codec
from numcodecs.compat import ensure_bytes, ensure_text, ensure_contiguous_ndarray_like
from numcodecs.registry import codec_registry
from zarr.context import Context
from zarr.types import PathLike as Path, DIMENSION_SEPARATOR
from zarr.util import NoLock

from zarr.errors import (
    MetadataError,
    BadCompressorError,
    ContainsArrayError,
    ContainsGroupError,
    FSPathExistNotDir,
    ReadOnlyError,
)
from zarr.meta import encode_array_metadata, encode_group_metadata
from zarr.util import (
    buffer_size,
    json_loads,
    nolock,
    normalize_chunks,
    normalize_dimension_separator,
    normalize_dtype,
    normalize_fill_value,
    normalize_order,
    normalize_shape,
    normalize_storage_path,
    retry_call,
    ensure_contiguous_ndarray_or_bytes,
)

from zarr._storage.absstore import ABSStore  # noqa: F401
from zarr._storage.store import (  # noqa: F401
    _get_hierarchy_metadata,
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
    DEFAULT_ZARR_VERSION,
    BaseStore,
    Store,
    V3_DEPRECATION_MESSAGE,
)

__doctest_requires__ = {
    ("RedisStore", "RedisStore.*"): ["redis"],
    ("MongoDBStore", "MongoDBStore.*"): ["pymongo"],
    ("LRUStoreCache", "LRUStoreCache.*"): ["s3fs"],
}


try:
    # noinspection PyUnresolvedReferences
    from zarr.codecs import Blosc

    default_compressor = Blosc()
except ImportError:  # pragma: no cover
    from zarr.codecs import Zlib

    default_compressor = Zlib()


# allow MutableMapping for backwards compatibility
StoreLike = Union[BaseStore, MutableMapping]


def contains_array(store: StoreLike, path: Path = None) -> bool:
    """Return True if the store contains an array at the given logical path."""
    path = normalize_storage_path(path)
    prefix = _path_to_prefix(path)
    key = _prefix_to_array_key(store, prefix)
    return key in store


def contains_group(store: StoreLike, path: Path = None, explicit_only=True) -> bool:
    """Return True if the store contains a group at the given logical path."""
    path = normalize_storage_path(path)
    prefix = _path_to_prefix(path)
    key = _prefix_to_group_key(store, prefix)
    store_version = getattr(store, "_store_version", 2)
    if store_version == 2 or explicit_only:
        return key in store
    else:
        if key in store:
            return True
        # for v3, need to also handle implicit groups

        sfx = _get_metadata_suffix(store)  # type: ignore
        implicit_prefix = key.replace(".group" + sfx, "")
        if not implicit_prefix.endswith("/"):
            implicit_prefix += "/"
        if store.list_prefix(implicit_prefix):  # type: ignore
            return True
        return False


def _normalize_store_arg_v2(store: Any, storage_options=None, mode="r") -> BaseStore:
    # default to v2 store for backward compatibility
    zarr_version = getattr(store, "_store_version", 2)
    if zarr_version != 2:
        raise ValueError("store must be a version 2 store")
    if store is None:
        store = KVStore(dict())
        return store
    if isinstance(store, os.PathLike):
        store = os.fspath(store)
    if FSStore._fsspec_installed():
        import fsspec

        if isinstance(store, fsspec.FSMap):
            return FSStore(
                store.root,
                fs=store.fs,
                mode=mode,
                check=store.check,
                create=store.create,
                missing_exceptions=store.missing_exceptions,
                **(storage_options or {}),
            )
    if isinstance(store, str):
        if "://" in store or "::" in store:
            return FSStore(store, mode=mode, **(storage_options or {}))
        elif storage_options:
            raise ValueError("storage_options passed with non-fsspec path")
        if store.endswith(".zip"):
            return ZipStore(store, mode=mode)
        elif store.endswith(".n5"):
            from zarr.n5 import N5Store

            return N5Store(store)
        else:
            return DirectoryStore(store)
    else:
        store = Store._ensure_store(store)
    return store


def normalize_store_arg(
    store: Any, storage_options=None, mode="r", *, zarr_version=None
) -> BaseStore:
    if zarr_version is None:
        # default to v2 store for backward compatibility
        zarr_version = getattr(store, "_store_version", DEFAULT_ZARR_VERSION)
    if zarr_version == 2:
        normalize_store = _normalize_store_arg_v2
    elif zarr_version == 3:
        from zarr._storage.v3 import _normalize_store_arg_v3

        normalize_store = _normalize_store_arg_v3
    else:
        raise ValueError("zarr_version must be either 2 or 3")
    return normalize_store(store, storage_options, mode)


def rmdir(store: StoreLike, path: Path = None):
    """Remove all items under the given path. If `store` provides a `rmdir` method,
    this will be called, otherwise will fall back to implementation via the
    `Store` interface."""
    path = normalize_storage_path(path)
    store_version = getattr(store, "_store_version", 2)
    if hasattr(store, "rmdir") and store.is_erasable():  # type: ignore
        # pass through
        store.rmdir(path)
    else:
        # slow version, delete one key at a time
        if store_version == 2:
            _rmdir_from_keys(store, path)
        else:
            _rmdir_from_keys_v3(store, path)  # type: ignore


def rename(store: Store, src_path: Path, dst_path: Path):
    """Rename all items under the given path. If `store` provides a `rename` method,
    this will be called, otherwise will fall back to implementation via the
    `Store` interface."""
    src_path = normalize_storage_path(src_path)
    dst_path = normalize_storage_path(dst_path)
    if hasattr(store, "rename"):
        # pass through
        store.rename(src_path, dst_path)
    else:
        # slow version, delete one key at a time
        _rename_from_keys(store, src_path, dst_path)


def listdir(store: BaseStore, path: Path = None):
    """Obtain a directory listing for the given path. If `store` provides a `listdir`
    method, this will be called, otherwise will fall back to implementation via the
    `MutableMapping` interface."""
    path = normalize_storage_path(path)
    if hasattr(store, "listdir"):
        # pass through
        return store.listdir(path)
    else:
        # slow version, iterate through all keys
        warnings.warn(
            f"Store {store} has no `listdir` method. From zarr 2.9 onwards "
            "may want to inherit from `Store`.",
            stacklevel=2,
        )
        return _listdir_from_keys(store, path)


def _getsize(store: BaseStore, path: Path = None) -> int:
    # compute from size of values
    if path and path in store:
        v = store[path]
        size = buffer_size(v)
    else:
        path = "" if path is None else normalize_storage_path(path)
        size = 0
        store_version = getattr(store, "_store_version", 2)
        if store_version == 3:
            if path == "":
                # have to list the root folders without trailing / in this case
                members = store.list_prefix(data_root.rstrip("/"))  # type: ignore
                members += store.list_prefix(meta_root.rstrip("/"))  # type: ignore
            else:
                members = store.list_prefix(data_root + path)  # type: ignore
                members += store.list_prefix(meta_root + path)  # type: ignore
            # also include zarr.json?
            # members += ['zarr.json']
        else:
            members = listdir(store, path)
            prefix = _path_to_prefix(path)
            members = [prefix + k for k in members]
        for k in members:
            try:
                v = store[k]
            except KeyError:
                pass
            else:
                try:
                    size += buffer_size(v)
                except TypeError:
                    return -1
    return size


def getsize(store: BaseStore, path: Path = None) -> int:
    """Compute size of stored items for a given path. If `store` provides a `getsize`
    method, this will be called, otherwise will return -1."""
    if hasattr(store, "getsize"):
        # pass through
        path = normalize_storage_path(path)
        return store.getsize(path)
    elif isinstance(store, MutableMapping):
        return _getsize(store, path)
    else:
        return -1


def _require_parent_group(
    path: Optional[str],
    store: StoreLike,
    chunk_store: Optional[StoreLike],
    overwrite: bool,
):
    # assume path is normalized
    if path:
        segments = path.split("/")
        for i in range(len(segments)):
            p = "/".join(segments[:i])
            if contains_array(store, p):
                _init_group_metadata(store, path=p, chunk_store=chunk_store, overwrite=overwrite)
            elif not contains_group(store, p):
                _init_group_metadata(store, path=p, chunk_store=chunk_store)


def init_array(
    store: StoreLike,
    shape: Union[int, Tuple[int, ...]],
    chunks: Union[bool, int, Tuple[int, ...]] = True,
    dtype=None,
    compressor="default",
    fill_value=None,
    order: str = "C",
    overwrite: bool = False,
    path: Optional[Path] = None,
    chunk_store: Optional[StoreLike] = None,
    filters=None,
    object_codec=None,
    dimension_separator: Optional[DIMENSION_SEPARATOR] = None,
    storage_transformers=(),
):
    """Initialize an array store with the given configuration. Note that this is a low-level
    function and there should be no need to call this directly from user code.

    Parameters
    ----------
    store : Store
        A mapping that supports string keys and bytes-like values.
    shape : int or tuple of ints
        Array shape.
    chunks : bool, int or tuple of ints, optional
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
    path : string, bytes, optional
        Path under which array is stored.
    chunk_store : Store, optional
        Separate storage for chunks. If not provided, `store` will be used
        for storage of both chunks and metadata.
    filters : sequence, optional
        Sequence of filters to use to encode chunk data prior to compression.
    object_codec : Codec, optional
        A codec to encode object arrays, only needed if dtype=object.
    dimension_separator : {'.', '/'}, optional
        Separator placed between the dimensions of a chunk.

    Examples
    --------
    Initialize an array store::

        >>> from zarr.storage import init_array, KVStore
        >>> store = KVStore(dict())
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

        >>> store = KVStore(dict())
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
    store_version = getattr(store, "_store_version", 2)
    if store_version < 3:
        _require_parent_group(path, store=store, chunk_store=chunk_store, overwrite=overwrite)

    if store_version == 3 and "zarr.json" not in store:
        # initialize with default zarr.json entry level metadata
        store["zarr.json"] = store._metadata_class.encode_hierarchy_metadata(None)  # type: ignore

    if not compressor:
        # compatibility with legacy tests using compressor=[]
        compressor = None
    _init_array_metadata(
        store,
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        compressor=compressor,
        fill_value=fill_value,
        order=order,
        overwrite=overwrite,
        path=path,
        chunk_store=chunk_store,
        filters=filters,
        object_codec=object_codec,
        dimension_separator=dimension_separator,
        storage_transformers=storage_transformers,
    )


def _init_array_metadata(
    store: StoreLike,
    shape,
    chunks=None,
    dtype=None,
    compressor="default",
    fill_value=None,
    order="C",
    overwrite=False,
    path: Optional[str] = None,
    chunk_store: Optional[StoreLike] = None,
    filters=None,
    object_codec=None,
    dimension_separator: Optional[DIMENSION_SEPARATOR] = None,
    storage_transformers=(),
):
    store_version = getattr(store, "_store_version", 2)

    path = normalize_storage_path(path)

    # guard conditions
    if overwrite:
        if store_version == 2:
            # attempt to delete any pre-existing array in store
            rmdir(store, path)
            if chunk_store is not None:
                rmdir(chunk_store, path)
        else:
            group_meta_key = _prefix_to_group_key(store, _path_to_prefix(path))
            array_meta_key = _prefix_to_array_key(store, _path_to_prefix(path))
            data_prefix = data_root + _path_to_prefix(path)

            # attempt to delete any pre-existing array in store
            if array_meta_key in store:
                store.erase(array_meta_key)  # type: ignore
            if group_meta_key in store:
                store.erase(group_meta_key)  # type: ignore
            store.erase_prefix(data_prefix)  # type: ignore
            if chunk_store is not None:
                chunk_store.erase_prefix(data_prefix)  # type: ignore

            if "/" in path:
                # path is a subfolder of an existing array, remove that array
                parent_path = "/".join(path.split("/")[:-1])
                sfx = _get_metadata_suffix(store)  # type: ignore
                array_key = meta_root + parent_path + ".array" + sfx
                if array_key in store:
                    store.erase(array_key)  # type: ignore

    if not overwrite:
        if contains_array(store, path):
            raise ContainsArrayError(path)
        elif contains_group(store, path, explicit_only=False):
            raise ContainsGroupError(path)
        elif store_version == 3:
            if "/" in path:
                # cannot create an array within an existing array path
                parent_path = "/".join(path.split("/")[:-1])
                if contains_array(store, parent_path):
                    raise ContainsArrayError(path)

    # normalize metadata
    dtype, object_codec = normalize_dtype(dtype, object_codec)
    shape = normalize_shape(shape) + dtype.shape
    dtype = dtype.base
    chunks = normalize_chunks(chunks, shape, dtype.itemsize)
    order = normalize_order(order)
    fill_value = normalize_fill_value(fill_value, dtype)

    # optional array metadata
    if dimension_separator is None and store_version == 2:
        dimension_separator = getattr(store, "_dimension_separator", None)
    dimension_separator = normalize_dimension_separator(dimension_separator)

    # compressor prep
    if shape == ():
        # no point in compressing a 0-dimensional array, only a single value
        compressor = None
    elif compressor == "none":
        # compatibility
        compressor = None
    elif compressor == "default":
        compressor = default_compressor

    # obtain compressor config
    compressor_config = None
    if compressor:
        if store_version == 2:
            try:
                compressor_config = compressor.get_config()
            except AttributeError as e:
                raise BadCompressorError(compressor) from e
        elif not isinstance(compressor, Codec):
            raise ValueError("expected a numcodecs Codec for compressor")
            # TODO: alternatively, could autoconvert str to a Codec
            #       e.g. 'zlib' -> numcodec.Zlib object
            # compressor = numcodecs.get_codec({'id': compressor})

    # obtain filters config
    if filters:
        # TODO: filters was removed from the metadata in v3
        #       raise error here if store_version > 2?
        filters_config = [f.get_config() for f in filters]
    else:
        filters_config = []

    # deal with object encoding
    if dtype.hasobject:
        if object_codec is None:
            if not filters:
                # there are no filters so we can be sure there is no object codec
                raise ValueError("missing object_codec for object array")
            else:
                # one of the filters may be an object codec, issue a warning rather
                # than raise an error to maintain backwards-compatibility
                warnings.warn(
                    "missing object_codec for object array; this will raise a "
                    "ValueError in version 3.0",
                    FutureWarning,
                    stacklevel=2,
                )
        else:
            filters_config.insert(0, object_codec.get_config())
    elif object_codec is not None:
        warnings.warn(
            "an object_codec is only needed for object arrays",
            stacklevel=2,
        )

    # use null to indicate no filters
    if not filters_config:
        filters_config = None  # type: ignore

    # initialize metadata
    # TODO: don't store redundant dimension_separator for v3?
    _compressor = compressor_config if store_version == 2 else compressor
    meta = dict(
        shape=shape,
        compressor=_compressor,
        fill_value=fill_value,
        dimension_separator=dimension_separator,
    )
    if store_version < 3:
        meta.update(dict(chunks=chunks, dtype=dtype, order=order, filters=filters_config))
        assert not storage_transformers
    else:
        if dimension_separator is None:
            dimension_separator = "/"
        if filters_config:
            attributes = {"filters": filters_config}
        else:
            attributes = {}
        meta.update(
            dict(
                chunk_grid=dict(type="regular", chunk_shape=chunks, separator=dimension_separator),
                chunk_memory_layout=order,
                data_type=dtype,
                attributes=attributes,
                storage_transformers=storage_transformers,
            )
        )

    key = _prefix_to_array_key(store, _path_to_prefix(path))
    if hasattr(store, "_metadata_class"):
        store[key] = store._metadata_class.encode_array_metadata(meta)
    else:
        store[key] = encode_array_metadata(meta)


# backwards compatibility
init_store = init_array


def init_group(
    store: StoreLike,
    overwrite: bool = False,
    path: Path = None,
    chunk_store: Optional[StoreLike] = None,
):
    """Initialize a group store. Note that this is a low-level function and there should be no
    need to call this directly from user code.

    Parameters
    ----------
    store : Store
        A mapping that supports string keys and byte sequence values.
    overwrite : bool, optional
        If True, erase all data in `store` prior to initialisation.
    path : string, optional
        Path under which array is stored.
    chunk_store : Store, optional
        Separate storage for chunks. If not provided, `store` will be used
        for storage of both chunks and metadata.

    """

    # normalize path
    path = normalize_storage_path(path)

    store_version = getattr(store, "_store_version", 2)
    if store_version < 3:
        # ensure parent group initialized
        _require_parent_group(path, store=store, chunk_store=chunk_store, overwrite=overwrite)

    if store_version == 3 and "zarr.json" not in store:
        # initialize with default zarr.json entry level metadata
        store["zarr.json"] = store._metadata_class.encode_hierarchy_metadata(None)  # type: ignore

    # initialise metadata
    _init_group_metadata(store=store, overwrite=overwrite, path=path, chunk_store=chunk_store)

    if store_version == 3:
        # TODO: Should initializing a v3 group also create a corresponding
        #       empty folder under data/root/? I think probably not until there
        #       is actual data written there.
        pass


def _init_group_metadata(
    store: StoreLike,
    overwrite: Optional[bool] = False,
    path: Optional[str] = None,
    chunk_store: Optional[StoreLike] = None,
):
    store_version = getattr(store, "_store_version", 2)
    path = normalize_storage_path(path)

    # guard conditions
    if overwrite:
        if store_version == 2:
            # attempt to delete any pre-existing items in store
            rmdir(store, path)
            if chunk_store is not None:
                rmdir(chunk_store, path)
        else:
            group_meta_key = _prefix_to_group_key(store, _path_to_prefix(path))
            array_meta_key = _prefix_to_array_key(store, _path_to_prefix(path))
            data_prefix = data_root + _path_to_prefix(path)
            meta_prefix = meta_root + _path_to_prefix(path)

            # attempt to delete any pre-existing array in store
            if array_meta_key in store:
                store.erase(array_meta_key)  # type: ignore
            if group_meta_key in store:
                store.erase(group_meta_key)  # type: ignore
            store.erase_prefix(data_prefix)  # type: ignore
            store.erase_prefix(meta_prefix)  # type: ignore
            if chunk_store is not None:
                chunk_store.erase_prefix(data_prefix)  # type: ignore

    if not overwrite:
        if contains_array(store, path):
            raise ContainsArrayError(path)
        elif contains_group(store, path):
            raise ContainsGroupError(path)
        elif store_version == 3 and "/" in path:
            # cannot create a group overlapping with an existing array name
            parent_path = "/".join(path.split("/")[:-1])
            if contains_array(store, parent_path):
                raise ContainsArrayError(path)

    # initialize metadata
    # N.B., currently no metadata properties are needed, however there may
    # be in future
    if store_version == 3:
        meta = {"attributes": {}}  # type: ignore
    else:
        meta = {}
    key = _prefix_to_group_key(store, _path_to_prefix(path))
    if hasattr(store, "_metadata_class"):
        store[key] = store._metadata_class.encode_group_metadata(meta)
    else:
        store[key] = encode_group_metadata(meta)


def _dict_store_keys(d: Dict, prefix="", cls=dict):
    for k in d.keys():
        v = d[k]
        if isinstance(v, cls):
            yield from _dict_store_keys(v, prefix + k + "/", cls)
        else:
            yield prefix + k


class KVStore(Store):
    """
    This provides a default implementation of a store interface around
    a mutable mapping, to avoid having to test stores for presence of methods.

    This, for most methods should just be a pass-through to the underlying KV
    store which is likely to expose a MuttableMapping interface,
    """

    def __init__(self, mutablemapping):
        self._mutable_mapping = mutablemapping

    def __getitem__(self, key):
        return self._mutable_mapping[key]

    def __setitem__(self, key, value):
        self._mutable_mapping[key] = value

    def __delitem__(self, key):
        del self._mutable_mapping[key]

    def __contains__(self, key):
        return key in self._mutable_mapping

    def get(self, key, default=None):
        return self._mutable_mapping.get(key, default)

    def values(self):
        return self._mutable_mapping.values()

    def __iter__(self):
        return iter(self._mutable_mapping)

    def __len__(self):
        return len(self._mutable_mapping)

    def __repr__(self):
        return f"<{self.__class__.__name__}: \n{self._mutable_mapping!r}\n at {id(self):#x}>"

    def __eq__(self, other):
        if isinstance(other, KVStore):
            return self._mutable_mapping == other._mutable_mapping
        else:
            return NotImplemented


class MemoryStore(Store):
    """Store class that uses a hierarchy of :class:`KVStore` objects, thus all data
    will be held in main memory.

    Examples
    --------
    This is the default class used when creating a group. E.g.::

        >>> import zarr
        >>> g = zarr.group()
        >>> type(g.store)
        <class 'zarr.storage.MemoryStore'>

    Note that the default class when creating an array is the built-in
    :class:`KVStore` class, i.e.::

        >>> z = zarr.zeros(100)
        >>> type(z.store)
        <class 'zarr.storage.KVStore'>

    Notes
    -----
    Safe to write in multiple threads.

    """

    def __init__(self, root=None, cls=dict, dimension_separator=None):
        if root is None:
            self.root = cls()
        else:
            self.root = root
        self.cls = cls
        self.write_mutex = Lock()
        self._dimension_separator = dimension_separator

    def __getstate__(self):
        return self.root, self.cls

    def __setstate__(self, state):
        root, cls = state
        self.__init__(root=root, cls=cls)

    def _get_parent(self, item: str):
        parent = self.root
        # split the item
        segments = item.split("/")
        # find the parent container
        for k in segments[:-1]:
            parent = parent[k]
            if not isinstance(parent, self.cls):
                raise KeyError(item)
        return parent, segments[-1]

    def _require_parent(self, item):
        parent = self.root
        # split the item
        segments = item.split("/")
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

    def __getitem__(self, item: str):
        parent, key = self._get_parent(item)
        try:
            value = parent[key]
        except KeyError as e:
            raise KeyError(item) from e
        else:
            if isinstance(value, self.cls):
                raise KeyError(item)
            else:
                return value

    def __setitem__(self, item: str, value):
        with self.write_mutex:
            parent, key = self._require_parent(item)
            value = ensure_bytes(value)
            parent[key] = value

    def __delitem__(self, item: str):
        with self.write_mutex:
            parent, key = self._get_parent(item)
            try:
                del parent[key]
            except KeyError as e:
                raise KeyError(item) from e

    def __contains__(self, item: str):  # type: ignore[override]
        try:
            parent, key = self._get_parent(item)
            value = parent[key]
        except KeyError:
            return False
        else:
            return not isinstance(value, self.cls)

    def __eq__(self, other):
        return isinstance(other, MemoryStore) and self.root == other.root and self.cls == other.cls

    def keys(self):
        yield from _dict_store_keys(self.root, cls=self.cls)

    def __iter__(self):
        return self.keys()

    def __len__(self) -> int:
        return sum(1 for _ in self.keys())

    def listdir(self, path: Path = None) -> List[str]:
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

    def rename(self, src_path: Path, dst_path: Path):
        src_path = normalize_storage_path(src_path)
        dst_path = normalize_storage_path(dst_path)

        src_parent, src_key = self._get_parent(src_path)
        dst_parent, dst_key = self._require_parent(dst_path)

        dst_parent[dst_key] = src_parent.pop(src_key)

    def rmdir(self, path: Path = None):
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

    def getsize(self, path: Path = None):
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
                    size += buffer_size(v)
            return size

        else:
            return buffer_size(value)

    def clear(self):
        with self.write_mutex:
            self.root.clear()


class DictStore(MemoryStore):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "DictStore has been renamed to MemoryStore in 2.4.0 and "
            "will be removed in the future. Please use MemoryStore.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class DirectoryStore(Store):
    """Storage class using directories and files on a standard file system.

    Parameters
    ----------
    path : string
        Location of directory to use as the root of the storage hierarchy.
    normalize_keys : bool, optional
        If True, all store keys will be normalized to use lower case characters
        (e.g. 'foo' and 'FOO' will be treated as equivalent). This can be
        useful to avoid potential discrepancies between case-sensitive and
        case-insensitive file system. Default value is False.
    dimension_separator : {'.', '/'}, optional
        Separator placed between the dimensions of a chunk.

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

    def __init__(
        self, path, normalize_keys=False, dimension_separator: Optional[DIMENSION_SEPARATOR] = None
    ):
        # guard conditions
        path = os.path.abspath(path)
        if os.path.exists(path) and not os.path.isdir(path):
            raise FSPathExistNotDir(path)

        self.path = path
        self.normalize_keys = normalize_keys
        self._dimension_separator = dimension_separator

    def _normalize_key(self, key):
        return key.lower() if self.normalize_keys else key

    @staticmethod
    def _fromfile(fn):
        """Read data from a file

        Parameters
        ----------
        fn : str
            Filepath to open and read from.

        Notes
        -----
        Subclasses should overload this method to specify any custom
        file reading logic.
        """
        with open(fn, "rb") as f:
            return f.read()

    @staticmethod
    def _tofile(a, fn):
        """Write data to a file

        Parameters
        ----------
        a : array-like
            Data to write into the file.
        fn : str
            Filepath to open and write to.

        Notes
        -----
        Subclasses should overload this method to specify any custom
        file writing logic.
        """
        with open(fn, mode="wb") as f:
            f.write(a)

    def __getitem__(self, key):
        key = self._normalize_key(key)
        filepath = os.path.join(self.path, key)
        if os.path.isfile(filepath):
            return self._fromfile(filepath)
        else:
            raise KeyError(key)

    def __setitem__(self, key, value):
        key = self._normalize_key(key)

        # coerce to flat, contiguous array (ideally without copying)
        value = ensure_contiguous_ndarray_like(value)

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
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise KeyError(key) from e

        # write to temporary file
        # note we're not using tempfile.NamedTemporaryFile to avoid restrictive file permissions
        temp_name = file_name + "." + uuid.uuid4().hex + ".partial"
        temp_path = os.path.join(dir_path, temp_name)
        try:
            self._tofile(value, temp_path)

            # move temporary file into place;
            # make several attempts at writing the temporary file to get past
            # potential antivirus file locking issues
            retry_call(os.replace, (temp_path, file_path), exceptions=(PermissionError,))

        finally:
            # clean up if temp file still exists for whatever reason
            if os.path.exists(temp_path):  # pragma: no cover
                os.remove(temp_path)

    def __delitem__(self, key):
        key = self._normalize_key(key)
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
        key = self._normalize_key(key)
        file_path = os.path.join(self.path, key)
        return os.path.isfile(file_path)

    def __eq__(self, other):
        return isinstance(other, DirectoryStore) and self.path == other.path

    def keys(self):
        if os.path.exists(self.path):
            yield from self._keys_fast(self.path)

    @staticmethod
    def _keys_fast(path, walker=os.walk):
        for dirpath, _, filenames in walker(path):
            dirpath = os.path.relpath(dirpath, path)
            if dirpath == os.curdir:
                for f in filenames:
                    yield f
            else:
                dirpath = dirpath.replace("\\", "/")
                for f in filenames:
                    yield "/".join((dirpath, f))

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
        return (
            self._nested_listdir(path)
            if self._dimension_separator == "/"
            else self._flat_listdir(path)
        )

    def _flat_listdir(self, path=None):
        dir_path = self.dir_path(path)
        if os.path.isdir(dir_path):
            return sorted(os.listdir(dir_path))
        else:
            return []

    def _nested_listdir(self, path=None):
        children = self._flat_listdir(path=path)
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
                            new_children.append(
                                rel_path.replace(os.path.sep, self._dimension_separator or ".")
                            )
                else:
                    new_children.append(entry)
            return sorted(new_children)
        else:
            return children

    def rename(self, src_path, dst_path):
        store_src_path = normalize_storage_path(src_path)
        store_dst_path = normalize_storage_path(dst_path)

        dir_path = self.path

        src_path = os.path.join(dir_path, store_src_path)
        dst_path = os.path.join(dir_path, store_dst_path)

        os.renames(src_path, dst_path)

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
            size = 0
            for child in scandir(fs_path):
                if child.is_file():
                    size += child.stat().st_size
            return size
        else:
            return 0

    def clear(self):
        shutil.rmtree(self.path)


def atexit_rmtree(path, isdir=os.path.isdir, rmtree=shutil.rmtree):  # pragma: no cover
    """Ensure directory removal at interpreter exit."""
    if isdir(path):
        rmtree(path)


# noinspection PyShadowingNames
def atexit_rmglob(
    path,
    glob=glob.glob,
    isdir=os.path.isdir,
    isfile=os.path.isfile,
    remove=os.remove,
    rmtree=shutil.rmtree,
):  # pragma: no cover
    """Ensure removal of multiple files at interpreter exit."""
    for p in glob(path):
        if isfile(p):
            remove(p)
        elif isdir(p):
            rmtree(p)


class FSStore(Store):
    """Wraps an fsspec.FSMap to give access to arbitrary filesystems

    Requires that ``fsspec`` is installed, as well as any additional
    requirements for the protocol chosen.

    Parameters
    ----------
    url : str
        The destination to map. If no fs is provided, should include protocol
        and path, like "s3://bucket/root". If an fs is provided, can be a path
        within that filesystem, like "bucket/root"
    normalize_keys : bool
    key_separator : str
        public API for accessing dimension_separator. Never `None`
        See dimension_separator for more information.
    mode : str
        "w" for writable, "r" for read-only
    exceptions : list of Exception subclasses
        When accessing data, any of these exceptions will be treated
        as a missing key
    dimension_separator : {'.', '/'}, optional
        Separator placed between the dimensions of a chunk.
    fs : fsspec.spec.AbstractFileSystem, optional
        An existing filesystem to use for the store.
    check : bool, optional
        If True, performs a touch at the root location, to check for write access.
        Passed to `fsspec.mapping.FSMap` constructor.
    create : bool, optional
        If True, performs a mkdir at the rool location.
        Passed to `fsspec.mapping.FSMap` constructor.
    missing_exceptions : sequence of Exceptions, optional
        Exceptions classes to associate with missing files.
        Passed to `fsspec.mapping.FSMap` constructor.
    storage_options : passed to the fsspec implementation. Cannot be used
        together with fs.
    """

    _array_meta_key = array_meta_key
    _group_meta_key = group_meta_key
    _attrs_key = attrs_key

    def __init__(
        self,
        url,
        normalize_keys=False,
        key_separator=None,
        mode="w",
        exceptions=(KeyError, PermissionError, IOError),
        dimension_separator: Optional[DIMENSION_SEPARATOR] = None,
        fs=None,
        check=False,
        create=False,
        missing_exceptions=None,
        **storage_options,
    ):
        if not self._fsspec_installed():  # pragma: no cover
            raise ImportError("`fsspec` is required to use zarr's FSStore")
        import fsspec

        mapper_options = {"check": check, "create": create}
        # https://github.com/zarr-developers/zarr-python/pull/911#discussion_r841926292
        # Some fsspec implementations don't accept missing_exceptions.
        # This is a workaround to avoid passing it in the most common scenarios.
        # Remove this and add missing_exceptions to mapper_options when fsspec is released.
        if missing_exceptions is not None:
            mapper_options["missing_exceptions"] = missing_exceptions  # pragma: no cover

        if fs is None:
            protocol, _ = fsspec.core.split_protocol(url)
            # set auto_mkdir to True for local file system
            if protocol in (None, "file") and not storage_options.get("auto_mkdir"):
                storage_options["auto_mkdir"] = True
            self.map = fsspec.get_mapper(url, **{**mapper_options, **storage_options})
            self.fs = self.map.fs  # for direct operations
            self.path = self.fs._strip_protocol(url)
        else:
            if storage_options:
                raise ValueError("Cannot specify both fs and storage_options")
            self.fs = fs
            self.path = self.fs._strip_protocol(url)
            self.map = self.fs.get_mapper(self.path, **mapper_options)

        self.normalize_keys = normalize_keys
        self.mode = mode
        self.exceptions = exceptions
        # For backwards compatibility. Guaranteed to be non-None
        if key_separator is not None:
            dimension_separator = key_separator

        self.key_separator = dimension_separator
        self._default_key_separator()

        # Pass attributes to array creation
        self._dimension_separator = dimension_separator

    def _default_key_separator(self):
        if self.key_separator is None:
            self.key_separator = "."

    def _normalize_key(self, key):
        key = normalize_storage_path(key).lstrip("/")
        if key:
            *bits, end = key.split("/")

            if end not in (self._array_meta_key, self._group_meta_key, self._attrs_key):
                end = end.replace(".", self.key_separator)
                key = "/".join(bits + [end])

        return key.lower() if self.normalize_keys else key

    def getitems(
        self, keys: Sequence[str], *, contexts: Mapping[str, Context]
    ) -> Mapping[str, Any]:
        keys_transformed = {self._normalize_key(key): key for key in keys}
        results_transformed = self.map.getitems(list(keys_transformed), on_error="return")
        results = {}
        for k, v in results_transformed.items():
            if isinstance(v, self.exceptions):
                # Cause recognized exceptions to prompt a KeyError in the
                # function calling this method
                continue
            elif isinstance(v, Exception):
                # Raise any other exception
                raise v
            else:
                # The function calling this method may not recognize the transformed
                # keys, so we send the values returned by self.map.getitems back into
                # the original key space.
                results[keys_transformed[k]] = v
        return results

    def __getitem__(self, key):
        key = self._normalize_key(key)
        try:
            return self.map[key]
        except self.exceptions as e:
            raise KeyError(key) from e

    def setitems(self, values):
        if self.mode == "r":
            raise ReadOnlyError()

        # Normalize keys and make sure the values are bytes
        values = {
            self._normalize_key(key): ensure_contiguous_ndarray_or_bytes(val)
            for key, val in values.items()
        }
        self.map.setitems(values)

    def __setitem__(self, key, value):
        if self.mode == "r":
            raise ReadOnlyError()
        key = self._normalize_key(key)
        value = ensure_contiguous_ndarray_or_bytes(value)
        path = self.dir_path(key)
        try:
            if self.fs.isdir(path):
                self.fs.rm(path, recursive=True)
            self.map[key] = value
            self.fs.invalidate_cache(self.fs._parent(path))
        except self.exceptions as e:
            raise KeyError(key) from e

    def __delitem__(self, key):
        if self.mode == "r":
            raise ReadOnlyError()
        key = self._normalize_key(key)
        path = self.dir_path(key)
        if self.fs.isdir(path):
            self.fs.rm(path, recursive=True)
        else:
            del self.map[key]

    def delitems(self, keys):
        if self.mode == "r":
            raise ReadOnlyError()
        # only remove the keys that exist in the store
        nkeys = [self._normalize_key(key) for key in keys if key in self]
        # rm errors if you pass an empty collection
        if len(nkeys) > 0:
            self.map.delitems(nkeys)

    def __contains__(self, key):
        key = self._normalize_key(key)
        return key in self.map

    def __eq__(self, other):
        return type(self) is type(other) and self.map == other.map and self.mode == other.mode

    def keys(self):
        return iter(self.map)

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return len(list(self.keys()))

    def dir_path(self, path=None):
        store_path = normalize_storage_path(path)
        return self.map._key_to_str(store_path)

    def listdir(self, path=None):
        dir_path = self.dir_path(path)
        try:
            children = sorted(
                p.rstrip("/").rsplit("/", 1)[-1] for p in self.fs.ls(dir_path, detail=False)
            )
            if self.key_separator != "/":
                return children
            else:
                if self._array_meta_key in children:
                    # special handling of directories containing an array to map nested chunk
                    # keys back to standard chunk keys
                    new_children = []
                    root_path = self.dir_path(path)
                    for entry in children:
                        entry_path = os.path.join(root_path, entry)
                        if _prog_number.match(entry) and self.fs.isdir(entry_path):
                            for file_name in self.fs.find(entry_path):
                                file_path = os.path.join(dir_path, file_name)
                                rel_path = file_path.split(root_path)[1]
                                rel_path = rel_path.lstrip("/")
                                new_children.append(rel_path.replace("/", "."))
                        else:
                            new_children.append(entry)
                    return sorted(new_children)
                else:
                    return children
        except OSError:
            return []

    def rmdir(self, path=None):
        if self.mode == "r":
            raise ReadOnlyError()
        store_path = self.dir_path(path)
        if self.fs.isdir(store_path):
            self.fs.rm(store_path, recursive=True)

    def getsize(self, path=None):
        store_path = self.dir_path(path)
        return self.fs.du(store_path, True, True)

    def clear(self):
        if self.mode == "r":
            raise ReadOnlyError()
        self.map.clear()

    @classmethod
    @lru_cache(maxsize=None)
    def _fsspec_installed(cls):
        """Returns true if fsspec is installed"""
        import importlib.util

        return importlib.util.find_spec("fsspec") is not None


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
    normalize_keys : bool, optional
        If True, all store keys will be normalized to use lower case characters
        (e.g. 'foo' and 'FOO' will be treated as equivalent). This can be
        useful to avoid potential discrepancies between case-sensitive and
        case-insensitive file system. Default value is False.
    dimension_separator : {'.', '/'}, optional
        Separator placed between the dimensions of a chunk.
    """

    # noinspection PyShadowingBuiltins
    def __init__(
        self,
        suffix="",
        prefix="zarr",
        dir=None,
        normalize_keys=False,
        dimension_separator: Optional[DIMENSION_SEPARATOR] = None,
    ):
        path = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
        atexit.register(atexit_rmtree, path)
        super().__init__(path, normalize_keys=normalize_keys)


_prog_ckey = re.compile(r"^(\d+)(\.\d+)+$")
_prog_number = re.compile(r"^\d+$")


class NestedDirectoryStore(DirectoryStore):
    """Storage class using directories and files on a standard file system, with
    special handling for chunk keys so that chunk files for multidimensional
    arrays are stored in a nested directory tree.

    .. deprecated:: 2.18.0
            NestedDirectoryStore will be removed in Zarr-Python 3.0 where controlling
            the chunk key encoding will be supported as part of the array metadata. See
            `GH1274 <https://github.com/zarr-developers/zarr-python/issues/1274>`_
            for more information.

    Parameters
    ----------
    path : string
        Location of directory to use as the root of the storage hierarchy.
    normalize_keys : bool, optional
        If True, all store keys will be normalized to use lower case characters
        (e.g. 'foo' and 'FOO' will be treated as equivalent). This can be
        useful to avoid potential discrepancies between case-sensitive and
        case-insensitive file system. Default value is False.
    dimension_separator : {'/'}, optional
        Separator placed between the dimensions of a chunk.
        Only supports "/" unlike other implementations.

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

    def __init__(
        self, path, normalize_keys=False, dimension_separator: Optional[DIMENSION_SEPARATOR] = "/"
    ):

        warnings.warn(
            V3_DEPRECATION_MESSAGE.format(store=self.__class__.__name__),
            FutureWarning,
            stacklevel=2,
        )

        super().__init__(path, normalize_keys=normalize_keys)
        if dimension_separator is None:
            dimension_separator = "/"
        elif dimension_separator != "/":
            raise ValueError("NestedDirectoryStore only supports '/' as dimension_separator")
        self._dimension_separator = dimension_separator

    def __eq__(self, other):
        return isinstance(other, NestedDirectoryStore) and self.path == other.path


# noinspection PyPep8Naming
class ZipStore(Store):
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
    dimension_separator : {'.', '/'}, optional
        Separator placed between the dimensions of a chunk.

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
        >>> # first write OK
        ... z[...] = 42
        >>> # second write generates warnings
        ... z[...] = 42  # doctest: +SKIP
        >>> store.close()

    This can also happen in a more subtle situation, where data are written only
    once to a Zarr array, but the write operations are not aligned with chunk
    boundaries, e.g.::

        >>> store = zarr.ZipStore('data/example.zip', mode='w')
        >>> z = zarr.zeros(100, chunks=10, store=store)
        >>> z[5:15] = 42
        >>> # write overlaps chunk previously written, generates warnings
        ... z[15:25] = 42  # doctest: +SKIP

    To avoid creating duplicate entries, only write data once, and align writes
    with chunk boundaries. This alignment is done automatically if you call
    ``z[...] = ...`` or create an array from existing data via :func:`zarr.array`.

    Alternatively, use a :class:`DirectoryStore` when writing the data, then
    manually Zip the directory and use the Zip file for subsequent reads.
    Take note that the files in the Zip file must be relative to the root of the
    Zarr archive. You may find it easier to create such a Zip file with ``7z``, e.g.::

        7z a -tzip archive.zarr.zip archive.zarr/.

    Safe to write in multiple threads but not in multiple processes.

    """

    _erasable = False

    def __init__(
        self,
        path,
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
        mode="a",
        dimension_separator: Optional[DIMENSION_SEPARATOR] = None,
    ):
        # store properties
        path = os.path.abspath(path)
        self.path = path
        self.compression = compression
        self.allowZip64 = allowZip64
        self.mode = mode
        self._dimension_separator = dimension_separator

        # Current understanding is that zipfile module in stdlib is not thread-safe,
        # and so locking is required for both read and write. However, this has not
        # been investigated in detail, perhaps no lock is needed if mode='r'.
        self.mutex = RLock()

        # open zip file
        self.zf = zipfile.ZipFile(path, mode=mode, compression=compression, allowZip64=allowZip64)

    def __getstate__(self):
        self.flush()
        return self.path, self.compression, self.allowZip64, self.mode

    def __setstate__(self, state):
        path, compression, allowZip64, mode = state
        # if initially opened with mode 'w' or 'x', re-open in mode 'a' so file doesn't
        # get clobbered
        if mode in "wx":
            mode = "a"
        self.__init__(path=path, compression=compression, allowZip64=allowZip64, mode=mode)

    def close(self):
        """Closes the underlying zip file, ensuring all records are written."""
        with self.mutex:
            self.zf.close()

    def flush(self):
        """Closes the underlying zip file, ensuring all records are written,
        then re-opens the file for further modifications."""
        if self.mode != "r":
            with self.mutex:
                self.zf.close()
                # N.B., re-open with mode 'a' regardless of initial mode so we don't wipe
                # what's been written
                self.zf = zipfile.ZipFile(
                    self.path, mode="a", compression=self.compression, allowZip64=self.allowZip64
                )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __getitem__(self, key):
        with self.mutex:
            with self.zf.open(key) as f:  # will raise KeyError
                return f.read()

    def __setitem__(self, key, value):
        if self.mode == "r":
            raise ReadOnlyError()
        value = ensure_contiguous_ndarray_like(value).view("u1")
        with self.mutex:
            # writestr(key, value) writes with default permissions from
            # zipfile (600) that are too restrictive, build ZipInfo for
            # the key to work around limitation
            keyinfo = zipfile.ZipInfo(filename=key, date_time=time.localtime(time.time())[:6])
            keyinfo.compress_type = self.compression
            if keyinfo.filename[-1] == os.sep:
                keyinfo.external_attr = 0o40775 << 16  # drwxrwxr-x
                keyinfo.external_attr |= 0x10  # MS-DOS directory flag
            else:
                keyinfo.external_attr = 0o644 << 16  # ?rw-r--r--

            self.zf.writestr(keyinfo, value)

    def __delitem__(self, key):
        raise NotImplementedError

    def __eq__(self, other):
        return (
            isinstance(other, ZipStore)
            and self.path == other.path
            and self.compression == other.compression
            and self.allowZip64 == other.allowZip64
        )

    def keylist(self):
        with self.mutex:
            return sorted(self.zf.namelist())

    def keys(self):
        yield from self.keylist()

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
                        name = path + "/" + child
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
        if self.mode == "r":
            raise ReadOnlyError()
        with self.mutex:
            self.close()
            os.remove(self.path)
            self.zf = zipfile.ZipFile(
                self.path, mode=self.mode, compression=self.compression, allowZip64=self.allowZip64
            )


def migrate_1to2(store):
    """Migrate array metadata in `store` from Zarr format version 1 to
    version 2.

    Parameters
    ----------
    store : Store
        Store to be migrated.

    Notes
    -----
    Version 1 did not support hierarchies, so this migration function will
    look for a single array in `store` and migrate the array metadata to
    version 2.

    """

    # migrate metadata
    from zarr import meta_v1

    meta = meta_v1.decode_metadata(store["meta"])
    del store["meta"]

    # add empty filters
    meta["filters"] = None

    # migration compression metadata
    compression = meta["compression"]
    if compression is None or compression == "none":
        compressor_config = None
    else:
        compression_opts = meta["compression_opts"]
        codec_cls = codec_registry[compression]
        if isinstance(compression_opts, dict):
            compressor = codec_cls(**compression_opts)
        else:
            compressor = codec_cls(compression_opts)
        compressor_config = compressor.get_config()
    meta["compressor"] = compressor_config
    del meta["compression"]
    del meta["compression_opts"]

    # store migrated metadata
    if hasattr(store, "_metadata_class"):
        store[array_meta_key] = store._metadata_class.encode_array_metadata(meta)
    else:
        store[array_meta_key] = encode_array_metadata(meta)

    # migrate user attributes
    store[attrs_key] = store["attrs"]
    del store["attrs"]


# noinspection PyShadowingBuiltins
class DBMStore(Store):
    """Storage class using a DBM-style database.

    .. deprecated:: 2.18.0
            DBMStore will be removed in Zarr-Python 3.0. See
            `GH1274 <https://github.com/zarr-developers/zarr-python/issues/1274>`_
            for more information.

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
    dimension_separator : {'.', '/'}, optional
        Separator placed between the dimensions of a chunk.e
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

    def __init__(
        self,
        path,
        flag="c",
        mode=0o666,
        open=None,
        write_lock=True,
        dimension_separator: Optional[DIMENSION_SEPARATOR] = None,
        **open_kwargs,
    ):
        warnings.warn(
            V3_DEPRECATION_MESSAGE.format(store=self.__class__.__name__),
            FutureWarning,
            stacklevel=2,
        )

        if open is None:
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
        self.write_mutex: Union[Lock, NoLock]
        if write_lock:
            # This may not be required as some dbm implementations manage their own
            # locks, but err on the side of caution.
            self.write_mutex = Lock()
        else:
            self.write_mutex = nolock
        self.open_kwargs = open_kwargs
        self._dimension_separator = dimension_separator

    def __getstate__(self):
        try:
            self.flush()  # needed for ndbm
        except Exception:
            # flush may fail if db has already been closed
            pass
        return (self.path, self.flag, self.mode, self.open, self.write_lock, self.open_kwargs)

    def __setstate__(self, state):
        path, flag, mode, open, write_lock, open_kws = state
        if flag[0] == "n":
            flag = "c" + flag[1:]  # don't clobber an existing database
        self.__init__(path=path, flag=flag, mode=mode, open=open, write_lock=write_lock, **open_kws)

    def close(self):
        """Closes the underlying database file."""
        if hasattr(self.db, "close"):
            with self.write_mutex:
                self.db.close()

    def flush(self):
        """Synchronizes data to the underlying database file."""
        if self.flag[0] != "r":
            with self.write_mutex:
                if hasattr(self.db, "sync"):
                    self.db.sync()
                else:  # pragma: no cover
                    # we don't cover this branch anymore as ndbm (oracle) is not packaged
                    # by conda-forge on non-mac OS:
                    # https://github.com/conda-forge/staged-recipes/issues/4476
                    # fall-back, close and re-open, needed for ndbm
                    flag = self.flag
                    if flag[0] == "n":
                        flag = "c" + flag[1:]  # don't clobber an existing database
                    self.db.close()
                    # noinspection PyArgumentList
                    self.db = self.open(self.path, flag, self.mode, **self.open_kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __getitem__(self, key):
        if isinstance(key, str):
            key = key.encode("ascii")
        return self.db[key]

    def __setitem__(self, key, value):
        if isinstance(key, str):
            key = key.encode("ascii")
        value = ensure_bytes(value)
        with self.write_mutex:
            self.db[key] = value

    def __delitem__(self, key):
        if isinstance(key, str):
            key = key.encode("ascii")
        with self.write_mutex:
            del self.db[key]

    def __eq__(self, other):
        return (
            isinstance(other, DBMStore)
            and self.path == other.path
            and
            # allow flag and mode to differ
            self.open == other.open
            and self.open_kwargs == other.open_kwargs
        )

    def keys(self):
        return (ensure_text(k, "ascii") for k in iter(self.db.keys()))

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return sum(1 for _ in self.keys())

    def __contains__(self, key):
        if isinstance(key, str):
            key = key.encode("ascii")
        return key in self.db

    def rmdir(self, path: str = "") -> None:
        path = normalize_storage_path(path)
        _rmdir_from_keys(self, path)


class LMDBStore(Store):
    """Storage class using LMDB. Requires the `lmdb <https://lmdb.readthedocs.io/>`_
    package to be installed.

    .. deprecated:: 2.18.0
        LMDBStore will be removed in Zarr-Python 3.0. See
        `GH1274 <https://github.com/zarr-developers/zarr-python/issues/1274>`_
        for more information.

    Parameters
    ----------
    path : string
        Location of database file.
    buffers : bool, optional
        If True (default) use support for buffers, which should increase performance by
        reducing memory copies.
    dimension_separator : {'.', '/'}, optional
        Separator placed between the dimensions of a chunk.
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

    def __init__(
        self,
        path,
        buffers=True,
        dimension_separator: Optional[DIMENSION_SEPARATOR] = None,
        **kwargs,
    ):
        import lmdb

        warnings.warn(
            V3_DEPRECATION_MESSAGE.format(store=self.__class__.__name__),
            FutureWarning,
            stacklevel=2,
        )

        # set default memory map size to something larger than the lmdb default, which is
        # very likely to be too small for any moderate array (logic copied from zict)
        map_size = 2**40 if sys.maxsize >= 2**32 else 2**28
        kwargs.setdefault("map_size", map_size)

        # don't initialize buffers to zero by default, shouldn't be necessary
        kwargs.setdefault("meminit", False)

        # decide whether to use the writemap option based on the operating system's
        # support for sparse files - writemap requires sparse file support otherwise
        # the whole# `map_size` may be reserved up front on disk (logic copied from zict)
        writemap = sys.platform.startswith("linux")
        kwargs.setdefault("writemap", writemap)

        # decide options for when data are flushed to disk - choose to delay syncing
        # data to filesystem, otherwise pay a large performance penalty (zict also does
        # this)
        kwargs.setdefault("metasync", False)
        kwargs.setdefault("sync", False)
        kwargs.setdefault("map_async", False)

        # set default option for number of cached transactions
        max_spare_txns = multiprocessing.cpu_count()
        kwargs.setdefault("max_spare_txns", max_spare_txns)

        # normalize path
        path = os.path.abspath(path)

        # open database
        self.db = lmdb.open(path, **kwargs)

        # store properties
        self.buffers = buffers
        self.path = path
        self.kwargs = kwargs
        self._dimension_separator = dimension_separator

    def __getstate__(self):
        try:
            self.flush()  # just in case
        except Exception:
            # flush may fail if db has already been closed
            pass
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
        if isinstance(key, str):
            key = key.encode("ascii")
        # use the buffers option, should avoid a memory copy
        with self.db.begin(buffers=self.buffers) as txn:
            value = txn.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def __setitem__(self, key, value):
        if isinstance(key, str):
            key = key.encode("ascii")
        with self.db.begin(write=True, buffers=self.buffers) as txn:
            txn.put(key, value)

    def __delitem__(self, key):
        if isinstance(key, str):
            key = key.encode("ascii")
        with self.db.begin(write=True) as txn:
            if not txn.delete(key):
                raise KeyError(key)

    def __contains__(self, key):
        if isinstance(key, str):
            key = key.encode("ascii")
        with self.db.begin(buffers=self.buffers) as txn:
            with txn.cursor() as cursor:
                return cursor.set_key(key)

    def items(self):
        with self.db.begin(buffers=self.buffers) as txn:
            with txn.cursor() as cursor:
                for k, v in cursor.iternext(keys=True, values=True):
                    yield ensure_text(k, "ascii"), v

    def keys(self):
        with self.db.begin(buffers=self.buffers) as txn:
            with txn.cursor() as cursor:
                for k in cursor.iternext(keys=True, values=False):
                    yield ensure_text(k, "ascii")

    def values(self):
        with self.db.begin(buffers=self.buffers) as txn:
            with txn.cursor() as cursor:
                yield from cursor.iternext(keys=False, values=True)

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return self.db.stat()["entries"]


class LRUStoreCache(Store):
    """Storage class that implements a least-recently-used (LRU) cache layer over
    some other store. Intended primarily for use with stores that can be slow to
    access, e.g., remote stores that require network communication to store and
    retrieve data.

    Parameters
    ----------
    store : Store
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
        >>> root = zarr.group(store=cache)  # doctest: +REMOTE_DATA
        >>> z = root['foo/bar/baz']  # doctest: +REMOTE_DATA
        >>> from timeit import timeit
        >>> # first data access is relatively slow, retrieved from store
        ... timeit('print(z[:].tobytes())', number=1, globals=globals())  # doctest: +SKIP
        b'Hello from the cloud!'
        0.1081731989979744
        >>> # second data access is faster, uses cache
        ... timeit('print(z[:].tobytes())', number=1, globals=globals())  # doctest: +SKIP
        b'Hello from the cloud!'
        0.0009490990014455747

    """

    def __init__(self, store: StoreLike, max_size: int):
        self._store: BaseStore = BaseStore._ensure_store(store)
        self._max_size = max_size
        self._current_size = 0
        self._keys_cache = None
        self._contains_cache: Dict[Any, Any] = {}
        self._listdir_cache: Dict[Path, Any] = dict()
        self._values_cache: Dict[Path, Any] = OrderedDict()
        self._mutex = Lock()
        self.hits = self.misses = 0

    def __getstate__(self):
        return (
            self._store,
            self._max_size,
            self._current_size,
            self._keys_cache,
            self._contains_cache,
            self._listdir_cache,
            self._values_cache,
            self.hits,
            self.misses,
        )

    def __setstate__(self, state):
        (
            self._store,
            self._max_size,
            self._current_size,
            self._keys_cache,
            self._contains_cache,
            self._listdir_cache,
            self._values_cache,
            self.hits,
            self.misses,
        ) = state
        self._mutex = Lock()

    def __len__(self):
        return len(self._keys())

    def __iter__(self):
        return self.keys()

    def __contains__(self, key):
        with self._mutex:
            if key not in self._contains_cache:
                self._contains_cache[key] = key in self._store
            return self._contains_cache[key]

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

    def listdir(self, path: Path = None):
        with self._mutex:
            try:
                return self._listdir_cache[path]
            except KeyError:
                listing = listdir(self._store, path)
                self._listdir_cache[path] = listing
                return listing

    def getsize(self, path=None) -> int:
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

    def _cache_value(self, key: Path, value):
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
            self._current_size = 0

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
        self._contains_cache.clear()
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
                self._values_cache.move_to_end(key)

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


class SQLiteStore(Store):
    """Storage class using SQLite.

    .. deprecated:: 2.18.0
            SQLiteStore will be removed in Zarr-Python 3.0. See
            `GH1274 <https://github.com/zarr-developers/zarr-python/issues/1274>`_
            for more information.

    Parameters
    ----------
    path : string
        Location of database file.
    dimension_separator : {'.', '/'}, optional
        Separator placed between the dimensions of a chunk.
    **kwargs
        Keyword arguments passed through to the `sqlite3.connect` function.

    Examples
    --------
    Store a single array::

        >>> import zarr
        >>> store = zarr.SQLiteStore('data/array.sqldb')
        >>> z = zarr.zeros((10, 10), chunks=(5, 5), store=store, overwrite=True)
        >>> z[...] = 42
        >>> store.close()  # don't forget to call this when you're done

    Store a group::

        >>> store = zarr.SQLiteStore('data/group.sqldb')
        >>> root = zarr.group(store=store, overwrite=True)
        >>> foo = root.create_group('foo')
        >>> bar = foo.zeros('bar', shape=(10, 10), chunks=(5, 5))
        >>> bar[...] = 42
        >>> store.close()  # don't forget to call this when you're done
    """

    def __init__(self, path, dimension_separator: Optional[DIMENSION_SEPARATOR] = None, **kwargs):
        import sqlite3

        warnings.warn(
            V3_DEPRECATION_MESSAGE.format(store=self.__class__.__name__),
            FutureWarning,
            stacklevel=2,
        )

        self._dimension_separator = dimension_separator

        # normalize path
        if path != ":memory:":
            path = os.path.abspath(path)

        # store properties
        self.path = path
        self.kwargs = kwargs

        # allow threading if SQLite connections are thread-safe
        #
        # ref: https://www.sqlite.org/releaselog/3_3_1.html
        # ref: https://github.com/python/cpython/issues/71377
        check_same_thread = True
        if sqlite3.sqlite_version_info >= (3, 3, 1):
            check_same_thread = False

        # keep a lock for serializing mutable operations
        self.lock = Lock()

        # open database
        self.db = sqlite3.connect(
            self.path,
            detect_types=0,
            isolation_level=None,
            check_same_thread=check_same_thread,
            **self.kwargs,
        )

        # handle keys as `str`s
        self.db.text_factory = str

        # get a cursor to read/write to the database
        self.cursor = self.db.cursor()

        # initialize database with our table if missing
        with self.lock:
            self.cursor.execute("CREATE TABLE IF NOT EXISTS zarr(k TEXT PRIMARY KEY, v BLOB)")

    def __getstate__(self):
        if self.path == ":memory:":
            raise PicklingError("Cannot pickle in-memory SQLite databases")
        return self.path, self.kwargs

    def __setstate__(self, state):
        path, kwargs = state
        self.__init__(path=path, **kwargs)

    def close(self):
        """Closes the underlying database."""

        # close cursor and db objects
        self.cursor.close()
        self.db.close()

    def __getitem__(self, key):
        value = self.cursor.execute("SELECT v FROM zarr WHERE (k = ?)", (key,))
        for (v,) in value:
            return v
        raise KeyError(key)

    def __setitem__(self, key, value):
        self.update({key: value})

    def __delitem__(self, key):
        with self.lock:
            self.cursor.execute("DELETE FROM zarr WHERE (k = ?)", (key,))
            if self.cursor.rowcount < 1:
                raise KeyError(key)

    def __contains__(self, key):
        cs = self.cursor.execute("SELECT COUNT(*) FROM zarr WHERE (k = ?)", (key,))
        for (has,) in cs:
            has = bool(has)
            return has

    def items(self):
        kvs = self.cursor.execute("SELECT k, v FROM zarr")
        yield from kvs

    def keys(self):
        ks = self.cursor.execute("SELECT k FROM zarr")
        for (k,) in ks:
            yield k

    def values(self):
        vs = self.cursor.execute("SELECT v FROM zarr")
        for (v,) in vs:
            yield v

    def __iter__(self):
        return self.keys()

    def __len__(self):
        cs = self.cursor.execute("SELECT COUNT(*) FROM zarr")
        for (c,) in cs:
            return c

    def update(self, *args, **kwargs):
        args += (kwargs,)

        kv_list = []
        for dct in args:
            for k, v in dct.items():
                v = ensure_contiguous_ndarray_like(v)

                # Accumulate key-value pairs for storage
                kv_list.append((k, v))

        with self.lock:
            self.cursor.executemany("REPLACE INTO zarr VALUES (?, ?)", kv_list)

    def listdir(self, path=None):
        path = normalize_storage_path(path)
        sep = "_" if path == "" else "/"
        keys = self.cursor.execute(
            f"""
            SELECT DISTINCT SUBSTR(m, 0, INSTR(m, "/")) AS l FROM (
                SELECT LTRIM(SUBSTR(k, LENGTH(?) + 1), "/") || "/" AS m
                FROM zarr WHERE k LIKE (? || "{sep}%")
            ) ORDER BY l ASC
            """,
            (path, path),
        )
        keys = list(map(operator.itemgetter(0), keys))
        return keys

    def getsize(self, path=None):
        path = normalize_storage_path(path)
        size = self.cursor.execute(
            """
            SELECT COALESCE(SUM(LENGTH(v)), 0) FROM zarr
            WHERE k LIKE (? || "%") AND
                  0 == INSTR(LTRIM(SUBSTR(k, LENGTH(?) + 1), "/"), "/")
            """,
            (path, path),
        )
        for (s,) in size:
            return s

    def rmdir(self, path=None):
        path = normalize_storage_path(path)
        if path:
            with self.lock:
                self.cursor.execute('DELETE FROM zarr WHERE k LIKE (? || "/%")', (path,))
        else:
            self.clear()

    def clear(self):
        with self.lock:
            self.cursor.executescript(
                """
                BEGIN TRANSACTION;
                    DROP TABLE zarr;
                    CREATE TABLE zarr(k TEXT PRIMARY KEY, v BLOB);
                COMMIT TRANSACTION;
                """
            )


class MongoDBStore(Store):
    """Storage class using MongoDB.

    .. note:: This is an experimental feature.

    .. deprecated:: 2.18.0
            MongoDBStore will be removed in Zarr-Python 3.0. See
            `GH1274 <https://github.com/zarr-developers/zarr-python/issues/1274>`_
            for more information.

    Requires the `pymongo <https://pymongo.readthedocs.io/en/stable/>`_
    package to be installed.

    Parameters
    ----------
    database : string
        Name of database
    collection : string
        Name of collection
    dimension_separator : {'.', '/'}, optional
        Separator placed between the dimensions of a chunk.
    **kwargs
        Keyword arguments passed through to the `pymongo.MongoClient` function.

    Notes
    -----
    The maximum chunksize in MongoDB documents is 16 MB.

    """

    _key = "key"
    _value = "value"

    def __init__(
        self,
        database="mongodb_zarr",
        collection="zarr_collection",
        dimension_separator: Optional[DIMENSION_SEPARATOR] = None,
        **kwargs,
    ):
        import pymongo

        warnings.warn(
            V3_DEPRECATION_MESSAGE.format(store=self.__class__.__name__),
            FutureWarning,
            stacklevel=2,
        )

        self._database = database
        self._collection = collection
        self._dimension_separator = dimension_separator
        self._kwargs = kwargs

        self.client = pymongo.MongoClient(**self._kwargs)
        self.db = self.client.get_database(self._database)
        self.collection = self.db.get_collection(self._collection)

    def __getitem__(self, key):
        doc = self.collection.find_one({self._key: key})

        if doc is None:
            raise KeyError(key)
        else:
            return doc[self._value]

    def __setitem__(self, key, value):
        value = ensure_bytes(value)
        self.collection.replace_one(
            {self._key: key}, {self._key: key, self._value: value}, upsert=True
        )

    def __delitem__(self, key):
        result = self.collection.delete_many({self._key: key})
        if not result.deleted_count == 1:
            raise KeyError(key)

    def __iter__(self):
        for f in self.collection.find({}):
            yield f[self._key]

    def __len__(self):
        return self.collection.count_documents({})

    def __getstate__(self):
        return self._database, self._collection, self._kwargs

    def __setstate__(self, state):
        database, collection, kwargs = state
        self.__init__(database=database, collection=collection, **kwargs)

    def close(self):
        """Cleanup client resources and disconnect from MongoDB."""
        self.client.close()

    def clear(self):
        """Remove all items from store."""
        self.collection.delete_many({})


class RedisStore(Store):
    """Storage class using Redis.

    .. note:: This is an experimental feature.

    .. deprecated:: 2.18.0
            RedisStore will be removed in Zarr-Python 3.0. See
            `GH1274 <https://github.com/zarr-developers/zarr-python/issues/1274>`_
            for more information.

    Requires the `redis <https://redis-py.readthedocs.io/>`_
    package to be installed.

    Parameters
    ----------
    prefix : string
        Name of prefix for Redis keys
    dimension_separator : {'.', '/'}, optional
        Separator placed between the dimensions of a chunk.
    **kwargs
        Keyword arguments passed through to the `redis.Redis` function.

    """

    def __init__(
        self, prefix="zarr", dimension_separator: Optional[DIMENSION_SEPARATOR] = None, **kwargs
    ):
        import redis

        warnings.warn(
            V3_DEPRECATION_MESSAGE.format(store=self.__class__.__name__),
            FutureWarning,
            stacklevel=2,
        )

        self._prefix = prefix
        self._kwargs = kwargs
        self._dimension_separator = dimension_separator

        self.client = redis.Redis(**kwargs)

    def _key(self, key):
        return f"{self._prefix}:{key}"

    def __getitem__(self, key):
        return self.client[self._key(key)]

    def __setitem__(self, key, value):
        value = ensure_bytes(value)
        self.client[self._key(key)] = value

    def __delitem__(self, key):
        count = self.client.delete(self._key(key))
        if not count:
            raise KeyError(key)

    def keylist(self):
        offset = len(self._key(""))  # length of prefix
        return [key[offset:].decode("utf-8") for key in self.client.keys(self._key("*"))]

    def keys(self):
        yield from self.keylist()

    def __iter__(self):
        yield from self.keys()

    def __len__(self):
        return len(self.keylist())

    def __getstate__(self):
        return self._prefix, self._kwargs

    def __setstate__(self, state):
        prefix, kwargs = state
        self.__init__(prefix=prefix, **kwargs)

    def clear(self):
        for key in self.keys():
            del self[key]


class ConsolidatedMetadataStore(Store):
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

    .. versionadded:: 2.3

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

    def __init__(self, store: StoreLike, metadata_key=".zmetadata"):
        self.store = Store._ensure_store(store)

        # retrieve consolidated metadata
        meta = json_loads(self.store[metadata_key])

        # check format of consolidated metadata
        consolidated_format = meta.get("zarr_consolidated_format", None)
        if consolidated_format != 1:
            raise MetadataError(
                f"unsupported zarr consolidated metadata format: {consolidated_format}"
            )

        # decode metadata
        self.meta_store: Store = KVStore(meta["metadata"])

    def __getitem__(self, key):
        return self.meta_store[key]

    def __contains__(self, item):
        return item in self.meta_store

    def __iter__(self):
        return iter(self.meta_store)

    def __len__(self):
        return len(self.meta_store)

    def __delitem__(self, key):
        raise ReadOnlyError()

    def __setitem__(self, key, value):
        raise ReadOnlyError()

    def getsize(self, path):
        return getsize(self.meta_store, path)

    def listdir(self, path):
        return listdir(self.meta_store, path)
