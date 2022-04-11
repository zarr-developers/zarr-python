import abc
import os
from collections.abc import MutableMapping
from string import ascii_letters, digits
from typing import Any, List, Mapping, Optional, Union

from zarr.meta import Metadata2, Metadata3
from zarr.util import normalize_storage_path

# v2 store keys
array_meta_key = '.zarray'
group_meta_key = '.zgroup'
attrs_key = '.zattrs'

# v3 paths
meta_root = 'meta/root/'
data_root = 'data/root/'

DEFAULT_ZARR_VERSION = 2

v3_api_available = os.environ.get('ZARR_V3_API_AVAILABLE', '0').lower() not in ['0', 'false']


class BaseStore(MutableMapping):
    """Abstract base class for store implementations.

    This is a thin wrapper over MutableMapping that provides methods to check
    whether a store is readable, writeable, eraseable and or listable.

    Stores cannot be mutable mapping as they do have a couple of other
    requirements that would break Liskov substitution principle (stores only
    allow strings as keys, mutable mapping are more generic).

    Having no-op base method also helps simplifying store usage and do not need
    to check the presence of attributes and methods, like `close()`.

    Stores can be used as context manager to make sure they close on exit.

    .. added: 2.11.0

    """

    _readable = True
    _writeable = True
    _erasable = True
    _listable = True
    _store_version = 2
    _metadata_class = Metadata2

    def is_readable(self):
        return self._readable

    def is_writeable(self):
        return self._writeable

    def is_listable(self):
        return self._listable

    def is_erasable(self):
        return self._erasable

    def __enter__(self):
        if not hasattr(self, "_open_count"):
            self._open_count = 0
        self._open_count += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._open_count -= 1
        if self._open_count == 0:
            self.close()

    def close(self) -> None:
        """Do nothing by default"""
        pass

    def rename(self, src_path: str, dst_path: str) -> None:
        if not self.is_erasable():
            raise NotImplementedError(
                f'{type(self)} is not erasable, cannot call "rename"'
            )  # pragma: no cover
        _rename_from_keys(self, src_path, dst_path)

    @staticmethod
    def _ensure_store(store: Any):
        """
        We want to make sure internally that zarr stores are always a class
        with a specific interface derived from ``BaseStore``, which is slightly
        different than ``MutableMapping``.

        We'll do this conversion in a few places automatically
        """
        from zarr.storage import KVStore  # avoid circular import

        if store is None:
            return None
        elif isinstance(store, BaseStore):
            if not store._store_version == 2:
                raise ValueError(
                    f"cannot initialize a v2 store with a v{store._store_version} store"
                )
            return store
        elif isinstance(store, MutableMapping):
            return KVStore(store)
        else:
            for attr in [
                "keys",
                "values",
                "get",
                "__setitem__",
                "__getitem__",
                "__delitem__",
                "__contains__",
            ]:
                if not hasattr(store, attr):
                    break
            else:
                return KVStore(store)

        raise ValueError(
            "Starting with Zarr 2.11.0, stores must be subclasses of "
            "BaseStore, if your store exposes the MutableMapping interface "
            f"wrap it in Zarr.storage.KVStore. Got {store}"
        )


class Store(BaseStore):
    """Abstract store class used by implementations following the Zarr v2 spec.

    Adds public `listdir`, `rename`, and `rmdir` methods on top of BaseStore.

    .. added: 2.11.0

    """

    def listdir(self, path: str = "") -> List[str]:
        path = normalize_storage_path(path)
        return _listdir_from_keys(self, path)

    def rmdir(self, path: str = "") -> None:
        if not self.is_erasable():
            raise NotImplementedError(
                f'{type(self)} is not erasable, cannot call "rmdir"'
            )  # pragma: no cover
        path = normalize_storage_path(path)
        _rmdir_from_keys(self, path)


class StoreV3(BaseStore):
    _store_version = 3
    _metadata_class = Metadata3
    _valid_key_characters = set(ascii_letters + digits + "/.-_")

    def _valid_key(self, key: str) -> bool:
        """
        Verify that a key conforms to the specification.

        A key is any string containing only character in the range a-z, A-Z,
        0-9, or in the set /.-_ it will return True if that's the case, False
        otherwise.
        """
        if not isinstance(key, str) or not key.isascii():
            return False
        if set(key) - self._valid_key_characters:
            return False
        return True

    def _validate_key(self, key: str):
        """
        Verify that a key conforms to the v3 specification.

        A key is any string containing only character in the range a-z, A-Z,
        0-9, or in the set /.-_ it will return True if that's the case, False
        otherwise.

        In spec v3, keys can only start with the prefix meta/, data/ or be
        exactly zarr.json and should not end with /. This should not be exposed
        to the user, and is a store implementation detail, so this method will
        raise a ValueError in that case.
        """
        if not self._valid_key(key):
            raise ValueError(
                f"Keys must be ascii strings and may only contain the "
                f"characters {''.join(sorted(self._valid_key_characters))}"
            )

        if (
            not key.startswith("data/")
            and (not key.startswith("meta/"))
            and (not key == "zarr.json")
            # TODO: Possibly allow key == ".zmetadata" too if we write a
            #       consolidated metadata spec corresponding to this?
        ):
            raise ValueError("keys starts with unexpected value: `{}`".format(key))

        if key.endswith('/'):
            raise ValueError("keys may not end in /")

    def list_prefix(self, prefix):
        if prefix.startswith('/'):
            raise ValueError("prefix must not begin with /")
        # TODO: force prefix to end with /?
        return [k for k in self.list() if k.startswith(prefix)]

    def erase(self, key):
        self.__delitem__(key)

    def erase_prefix(self, prefix):
        assert prefix.endswith("/")

        if prefix == "/":
            all_keys = self.list()
        else:
            all_keys = self.list_prefix(prefix)
        for key in all_keys:
            self.erase(key)

    def list_dir(self, prefix):
        """
        TODO: carefully test this with trailing/leading slashes
        """
        if prefix:  # allow prefix = "" ?
            assert prefix.endswith("/")

        all_keys = self.list_prefix(prefix)
        len_prefix = len(prefix)
        keys = []
        prefixes = []
        for k in all_keys:
            trail = k[len_prefix:]
            if "/" not in trail:
                keys.append(prefix + trail)
            else:
                prefixes.append(prefix + trail.split("/", maxsplit=1)[0] + "/")
        return keys, list(set(prefixes))

    def list(self):
        return list(self.keys())

    def __contains__(self, key):
        return key in self.list()

    @abc.abstractmethod
    def __setitem__(self, key, value):
        """Set a value."""

    @abc.abstractmethod
    def __getitem__(self, key):
        """Get a value."""

    def clear(self):
        """Remove all items from store."""
        self.erase_prefix("/")

    def __eq__(self, other):
        return NotImplemented

    @staticmethod
    def _ensure_store(store):
        """
        We want to make sure internally that zarr stores are always a class
        with a specific interface derived from ``Store``, which is slightly
        different than ``MutableMapping``.

        We'll do this conversion in a few places automatically
        """
        from zarr.storage_v3 import KVStoreV3  # avoid circular import
        if store is None:
            return None
        elif isinstance(store, StoreV3):
            return store
        elif isinstance(store, Store):
            raise ValueError(
                f"cannot initialize a v3 store with a v{store._store_version} store"
            )
        elif isinstance(store, MutableMapping):
            return KVStoreV3(store)
        else:
            for attr in [
                "keys",
                "values",
                "get",
                "__setitem__",
                "__getitem__",
                "__delitem__",
                "__contains__",
            ]:
                if not hasattr(store, attr):
                    break
            else:
                return KVStoreV3(store)

        raise ValueError(
            "v3 stores must be subclasses of StoreV3, "
            "if your store exposes the MutableMapping interface wrap it in "
            f"Zarr.storage.KVStoreV3. Got {store}"
        )


# allow MutableMapping for backwards compatibility
StoreLike = Union[BaseStore, MutableMapping]


def _path_to_prefix(path: Optional[str]) -> str:
    # assume path already normalized
    if path:
        prefix = path + '/'
    else:
        prefix = ''
    return prefix


def _get_hierarchy_metadata(store: StoreV3) -> Mapping[str, Any]:
    version = getattr(store, '_store_version', 2)
    if version < 3:
        raise ValueError("zarr.json hierarchy metadata not stored for "
                         f"zarr v{version} stores")
    if 'zarr.json' not in store:
        raise ValueError("zarr.json metadata not found in store")
    return store._metadata_class.decode_hierarchy_metadata(store['zarr.json'])


def _get_metadata_suffix(store: StoreV3) -> str:
    if 'zarr.json' in store:
        return _get_hierarchy_metadata(store)['metadata_key_suffix']
    return '.json'


def _rename_metadata_v3(store: StoreV3, src_path: str, dst_path: str) -> bool:
    """Rename source or group metadata file associated with src_path."""
    any_renamed = False
    sfx = _get_metadata_suffix(store)
    src_path = src_path.rstrip('/')
    dst_path = dst_path.rstrip('/')
    _src_array_json = meta_root + src_path + '.array' + sfx
    if _src_array_json in store:
        new_key = meta_root + dst_path + '.array' + sfx
        store[new_key] = store.pop(_src_array_json)
        any_renamed = True
    _src_group_json = meta_root + src_path + '.group' + sfx
    if _src_group_json in store:
        new_key = meta_root + dst_path + '.group' + sfx
        store[new_key] = store.pop(_src_group_json)
        any_renamed = True
    return any_renamed


def _rename_from_keys(store: BaseStore, src_path: str, dst_path: str) -> None:
    # assume path already normalized
    src_prefix = _path_to_prefix(src_path)
    dst_prefix = _path_to_prefix(dst_path)
    version = getattr(store, '_store_version', 2)
    if version == 2:
        for key in list(store.keys()):
            if key.startswith(src_prefix):
                new_key = dst_prefix + key.lstrip(src_prefix)
                store[new_key] = store.pop(key)
    else:
        any_renamed = False
        for root_prefix in [meta_root, data_root]:
            _src_prefix = root_prefix + src_prefix
            _dst_prefix = root_prefix + dst_prefix
            for key in store.list_prefix(_src_prefix):  # type: ignore
                new_key = _dst_prefix + key[len(_src_prefix):]
                store[new_key] = store.pop(key)
                any_renamed = True
        any_meta_renamed = _rename_metadata_v3(store, src_path, dst_path)  # type: ignore
        any_renamed = any_meta_renamed or any_renamed

        if not any_renamed:
            raise ValueError(f"no item {src_path} found to rename")


def _rmdir_from_keys(store: StoreLike, path: Optional[str] = None) -> None:
    # assume path already normalized
    prefix = _path_to_prefix(path)
    for key in list(store.keys()):
        if key.startswith(prefix):
            del store[key]


def _rmdir_from_keys_v3(store: StoreV3, path: str = "") -> None:

    meta_dir = meta_root + path
    meta_dir = meta_dir.rstrip('/')
    _rmdir_from_keys(store, meta_dir)

    # remove data folder
    data_dir = data_root + path
    data_dir = data_dir.rstrip('/')
    _rmdir_from_keys(store, data_dir)

    # remove metadata files
    sfx = _get_metadata_suffix(store)
    array_meta_file = meta_dir + '.array' + sfx
    if array_meta_file in store:
        store.erase(array_meta_file)  # type: ignore
    group_meta_file = meta_dir + '.group' + sfx
    if group_meta_file in store:
        store.erase(group_meta_file)  # type: ignore


def _listdir_from_keys(store: BaseStore, path: Optional[str] = None) -> List[str]:
    # assume path already normalized
    prefix = _path_to_prefix(path)
    children = set()
    for key in list(store.keys()):
        if key.startswith(prefix) and len(key) > len(prefix):
            suffix = key[len(prefix):]
            child = suffix.split('/')[0]
            children.add(child)
    return sorted(children)


def _prefix_to_array_key(store: StoreLike, prefix: str) -> str:
    if getattr(store, "_store_version", 2) == 3:
        if prefix:
            sfx = _get_metadata_suffix(store)  # type: ignore
            key = meta_root + prefix.rstrip("/") + ".array" + sfx
        else:
            raise ValueError("prefix must be supplied to get a v3 array key")
    else:
        key = prefix + array_meta_key
    return key


def _prefix_to_group_key(store: StoreLike, prefix: str) -> str:
    if getattr(store, "_store_version", 2) == 3:
        if prefix:
            sfx = _get_metadata_suffix(store)  # type: ignore
            key = meta_root + prefix.rstrip('/') + ".group" + sfx
        else:
            raise ValueError("prefix must be supplied to get a v3 group key")
    else:
        key = prefix + group_meta_key
    return key


def _prefix_to_attrs_key(store: StoreLike, prefix: str) -> str:
    if getattr(store, "_store_version", 2) == 3:
        # for v3, attributes are stored in the array metadata
        sfx = _get_metadata_suffix(store)  # type: ignore
        if prefix:
            key = meta_root + prefix.rstrip('/') + ".array" + sfx
        else:
            raise ValueError("prefix must be supplied to get a v3 array key")
    else:
        key = prefix + attrs_key
    return key
