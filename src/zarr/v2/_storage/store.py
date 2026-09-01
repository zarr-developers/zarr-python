import abc
import os
import warnings
from collections import defaultdict
from collections.abc import MutableMapping
from copy import copy
from string import ascii_letters, digits
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from zarr.meta import Metadata2, Metadata3
from zarr.util import normalize_storage_path
from zarr.context import Context
from zarr.types import ZARR_VERSION

# v2 store keys
array_meta_key = ".zarray"
group_meta_key = ".zgroup"
attrs_key = ".zattrs"

# v3 paths
meta_root = "meta/root/"
data_root = "data/root/"

DEFAULT_ZARR_VERSION: ZARR_VERSION = 2

v3_api_available = os.environ.get("ZARR_V3_EXPERIMENTAL_API", "0").lower() not in ["0", "false"]
_has_warned_about_v3 = False  # to avoid printing the warning multiple times

V3_DEPRECATION_MESSAGE = (
    "The {store} is deprecated and will be removed in a Zarr-Python version 3, see "
    "https://github.com/zarr-developers/zarr-python/issues/1274 for more information."
)


def assert_zarr_v3_api_available():
    # we issue a warning about the experimental v3 implementation when it is first used
    global _has_warned_about_v3
    if v3_api_available and not _has_warned_about_v3:
        warnings.warn(
            "The experimental Zarr V3 implementation in this version of Zarr-Python is not "
            "in alignment with the final V3 specification. This version will be removed in "
            "Zarr-Python 3 in favor of a spec compliant version.",
            FutureWarning,
            stacklevel=1,
        )
        _has_warned_about_v3 = True
    if not v3_api_available:
        raise NotImplementedError(
            "# V3 reading and writing is experimental! To enable support, set:\n"
            "ZARR_V3_EXPERIMENTAL_API=1"
        )  # pragma: no cover


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

        if isinstance(store, BaseStore):
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

    def getitems(
        self, keys: Sequence[str], *, contexts: Mapping[str, Context]
    ) -> Mapping[str, Any]:
        """Retrieve data from multiple keys.

        Parameters
        ----------
        keys : Iterable[str]
            The keys to retrieve
        contexts: Mapping[str, Context]
            A mapping of keys to their context. Each context is a mapping of store
            specific information. E.g. a context could be a dict telling the store
            the preferred output array type: `{"meta_array": cupy.empty(())}`

        Returns
        -------
        Mapping
            A collection mapping the input keys to their results.

        Notes
        -----
        This default implementation uses __getitem__() to read each key sequentially and
        ignores contexts. Overwrite this method to implement concurrent reads of multiple
        keys and/or to utilize the contexts.
        """
        return {k: self[k] for k in keys if k in self}


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
            not key.startswith(("data/", "meta/"))
            and key != "zarr.json"
            # TODO: Possibly allow key == ".zmetadata" too if we write a
            #       consolidated metadata spec corresponding to this?
        ):
            raise ValueError(f"key starts with unexpected value: `{key}`")

        if key.endswith("/"):
            raise ValueError("keys may not end in /")

    def list_prefix(self, prefix):
        if prefix.startswith("/"):
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

    @abc.abstractmethod
    def rmdir(self, path=None):
        """Remove a data path and all its subkeys and related metadata.
        Expects a path without the data or meta root prefix."""

    @property
    def supports_efficient_get_partial_values(self):
        return False

    def get_partial_values(
        self, key_ranges: Sequence[Tuple[str, Tuple[int, Optional[int]]]]
    ) -> List[Union[bytes, memoryview, bytearray]]:
        """Get multiple partial values.
        key_ranges can be an iterable of key, range pairs,
        where a range specifies two integers range_start and range_length
        as a tuple, (range_start, range_length).
        range_length may be None to indicate to read until the end.
        range_start may be negative to start reading range_start bytes
        from the end of the file.
        A key may occur multiple times with different ranges.
        Inserts None for missing keys into the returned list."""
        results: List[Union[bytes, memoryview, bytearray]] = [None] * len(key_ranges)  # type: ignore[list-item] # noqa: E501
        indexed_ranges_by_key: Dict[str, List[Tuple[int, Tuple[int, Optional[int]]]]] = defaultdict(
            list
        )
        for i, (key, range_) in enumerate(key_ranges):
            indexed_ranges_by_key[key].append((i, range_))
        for key, indexed_ranges in indexed_ranges_by_key.items():
            try:
                value = self[key]
            except KeyError:  # pragma: no cover
                continue
            for i, (range_from, range_length) in indexed_ranges:
                if range_length is None:
                    results[i] = value[range_from:]
                else:
                    results[i] = value[range_from : range_from + range_length]
        return results

    def supports_efficient_set_partial_values(self):
        return False

    def set_partial_values(self, key_start_values):
        """Set multiple partial values.
        key_start_values can be an iterable of key, start and value triplets
        as tuples, (key, start, value), where start defines the offset in bytes.
        A key may occur multiple times with different starts and non-overlapping values.
        Also, start may only be beyond the current value if other values fill the gap.
        start may be negative to start writing start bytes from the current
        end of the file, ending the file with the new value."""
        unique_keys = set(next(zip(*key_start_values)))
        values = {}
        for key in unique_keys:
            old_value = self.get(key)
            values[key] = None if old_value is None else bytearray(old_value)
        for key, start, value in key_start_values:
            if values[key] is None:
                assert start == 0
                values[key] = value
            else:
                if start > len(values[key]):  # pragma: no cover
                    raise ValueError(
                        f"Cannot set value at start {start}, "
                        + f"since it is beyond the data at key {key}, "
                        + f"having length {len(values[key])}."
                    )
                if start < 0:
                    values[key][start:] = value
                else:
                    values[key][start : start + len(value)] = value
        for key, value in values.items():
            self[key] = value

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
        from zarr._storage.v3 import KVStoreV3  # avoid circular import

        if store is None:
            return None
        elif isinstance(store, StoreV3):
            return store
        elif isinstance(store, Store):
            raise ValueError(f"cannot initialize a v3 store with a v{store._store_version} store")
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


class StorageTransformer(MutableMapping, abc.ABC):
    """Base class for storage transformers. The methods simply pass on the data as-is
    and should be overwritten by sub-classes."""

    _store_version = 3
    _metadata_class = Metadata3

    def __init__(self, _type) -> None:
        if _type not in self.valid_types:  # pragma: no cover
            raise ValueError(
                f"Storage transformer cannot be initialized with type {_type}, "
                + f"must be one of {list(self.valid_types)}."
            )
        self.type = _type
        self._inner_store = None

    def _copy_for_array(self, array, inner_store):
        transformer_copy = copy(self)
        transformer_copy._inner_store = inner_store
        return transformer_copy

    @abc.abstractproperty
    def extension_uri(self):
        pass  # pragma: no cover

    @abc.abstractproperty
    def valid_types(self):
        pass  # pragma: no cover

    def get_config(self):
        """Return a dictionary holding configuration parameters for this
        storage transformer. All values must be compatible with JSON encoding."""
        # Override in sub-class if need special encoding of config values.
        # By default, assume all non-private members are configuration
        # parameters except for type .
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_") and k != "type"}

    @classmethod
    def from_config(cls, _type, config):
        """Instantiate storage transformer from a configuration object."""
        # override in sub-class if need special decoding of config values

        # by default, assume constructor accepts configuration parameters as
        # keyword arguments without any special decoding
        return cls(_type, **config)

    @property
    def inner_store(self) -> Union["StorageTransformer", StoreV3]:
        assert (
            self._inner_store is not None
        ), "inner_store is not initialized, first get a copy via _copy_for_array."
        return self._inner_store

    # The following implementations are usually fine to keep as-is:

    def __eq__(self, other):
        return (
            type(self) is type(other)
            and self._inner_store == other._inner_store
            and self.get_config() == other.get_config()
        )

    def erase(self, key):
        self.__delitem__(key)

    def list(self):
        return list(self.keys())

    def list_dir(self, prefix):
        return StoreV3.list_dir(self, prefix)

    def is_readable(self):
        return self.inner_store.is_readable()

    def is_writeable(self):
        return self.inner_store.is_writeable()

    def is_listable(self):
        return self.inner_store.is_listable()

    def is_erasable(self):
        return self.inner_store.is_erasable()

    def clear(self):
        return self.inner_store.clear()

    def __enter__(self):
        return self.inner_store.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        return self.inner_store.__exit__(exc_type, exc_value, traceback)

    def close(self) -> None:
        return self.inner_store.close()

    # The following implementations might need to be re-implemented
    # by subclasses implementing storage transformers:

    def rename(self, src_path: str, dst_path: str) -> None:
        return self.inner_store.rename(src_path, dst_path)

    def list_prefix(self, prefix):
        return self.inner_store.list_prefix(prefix)

    def erase_prefix(self, prefix):
        return self.inner_store.erase_prefix(prefix)

    def rmdir(self, path=None):
        return self.inner_store.rmdir(path)

    def __contains__(self, key):
        return self.inner_store.__contains__(key)

    def __setitem__(self, key, value):
        return self.inner_store.__setitem__(key, value)

    def __getitem__(self, key):
        return self.inner_store.__getitem__(key)

    def __delitem__(self, key):
        return self.inner_store.__delitem__(key)

    def __iter__(self):
        return self.inner_store.__iter__()

    def __len__(self):
        return self.inner_store.__len__()

    @property
    def supports_efficient_get_partial_values(self):
        return self.inner_store.supports_efficient_get_partial_values

    def get_partial_values(self, key_ranges):
        return self.inner_store.get_partial_values(key_ranges)

    def supports_efficient_set_partial_values(self):
        return self.inner_store.supports_efficient_set_partial_values()

    def set_partial_values(self, key_start_values):
        return self.inner_store.set_partial_values(key_start_values)


# allow MutableMapping for backwards compatibility
StoreLike = Union[BaseStore, MutableMapping]


def _path_to_prefix(path: Optional[str]) -> str:
    # assume path already normalized
    if path:
        prefix = path + "/"
    else:
        prefix = ""
    return prefix


def _get_hierarchy_metadata(store: StoreV3) -> Mapping[str, Any]:
    version = getattr(store, "_store_version", 2)
    if version < 3:
        raise ValueError("zarr.json hierarchy metadata not stored for " f"zarr v{version} stores")
    if "zarr.json" not in store:
        raise ValueError("zarr.json metadata not found in store")
    return store._metadata_class.decode_hierarchy_metadata(store["zarr.json"])


def _get_metadata_suffix(store: StoreV3) -> str:
    if "zarr.json" in store:
        return _get_hierarchy_metadata(store)["metadata_key_suffix"]
    return ".json"


def _rename_metadata_v3(store: StoreV3, src_path: str, dst_path: str) -> bool:
    """Rename source or group metadata file associated with src_path."""
    any_renamed = False
    sfx = _get_metadata_suffix(store)
    src_path = src_path.rstrip("/")
    dst_path = dst_path.rstrip("/")
    _src_array_json = meta_root + src_path + ".array" + sfx
    if _src_array_json in store:
        new_key = meta_root + dst_path + ".array" + sfx
        store[new_key] = store.pop(_src_array_json)
        any_renamed = True
    _src_group_json = meta_root + src_path + ".group" + sfx
    if _src_group_json in store:
        new_key = meta_root + dst_path + ".group" + sfx
        store[new_key] = store.pop(_src_group_json)
        any_renamed = True
    return any_renamed


def _rename_from_keys(store: BaseStore, src_path: str, dst_path: str) -> None:
    # assume path already normalized
    src_prefix = _path_to_prefix(src_path)
    dst_prefix = _path_to_prefix(dst_path)
    version = getattr(store, "_store_version", 2)
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
                new_key = _dst_prefix + key[len(_src_prefix) :]
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
    meta_dir = meta_dir.rstrip("/")
    _rmdir_from_keys(store, meta_dir)

    # remove data folder
    data_dir = data_root + path
    data_dir = data_dir.rstrip("/")
    _rmdir_from_keys(store, data_dir)

    # remove metadata files
    sfx = _get_metadata_suffix(store)
    array_meta_file = meta_dir + ".array" + sfx
    if array_meta_file in store:
        store.erase(array_meta_file)
    group_meta_file = meta_dir + ".group" + sfx
    if group_meta_file in store:
        store.erase(group_meta_file)


def _listdir_from_keys(store: BaseStore, path: Optional[str] = None) -> List[str]:
    # assume path already normalized
    prefix = _path_to_prefix(path)
    children = set()
    for key in list(store.keys()):
        if key.startswith(prefix) and len(key) > len(prefix):
            suffix = key[len(prefix) :]
            child = suffix.split("/")[0]
            children.add(child)
    return sorted(children)


def _prefix_to_array_key(store: StoreLike, prefix: str) -> str:
    if getattr(store, "_store_version", 2) == 3:
        sfx = _get_metadata_suffix(store)  # type: ignore
        if prefix:
            key = meta_root + prefix.rstrip("/") + ".array" + sfx
        else:
            key = meta_root[:-1] + ".array" + sfx
    else:
        key = prefix + array_meta_key
    return key


def _prefix_to_group_key(store: StoreLike, prefix: str) -> str:
    if getattr(store, "_store_version", 2) == 3:
        sfx = _get_metadata_suffix(store)  # type: ignore
        if prefix:
            key = meta_root + prefix.rstrip("/") + ".group" + sfx
        else:
            key = meta_root[:-1] + ".group" + sfx
    else:
        key = prefix + group_meta_key
    return key


def _prefix_to_attrs_key(store: StoreLike, prefix: str) -> str:
    if getattr(store, "_store_version", 2) == 3:
        # for v3, attributes are stored in the array metadata
        sfx = _get_metadata_suffix(store)  # type: ignore
        if prefix:
            key = meta_root + prefix.rstrip("/") + ".array" + sfx
        else:
            key = meta_root[:-1] + ".array" + sfx
    else:
        key = prefix + attrs_key
    return key
