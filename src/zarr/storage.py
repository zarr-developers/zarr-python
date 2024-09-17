import importlib
import warnings
from typing import Any

_names = [
    "DictStore",
    "KVStore",
    "DirectoryStore",
    "ZipStore",
    "FSStore",
]
_mappings = {
    "DictStore": ("memory", "MemoryStore"),
    "KVStore": ("memory", "MemoryStore"),
    "DirectoryStore": ("local", "LocalStore"),
    "ZipStore": ("zip", "ZipStore"),
    "FSStore": ("remote", "RemoteStore"),
}


def _deprecated_import(old: str) -> Any:
    try:
        submodule, new = _mappings[old]
        mod = importlib.import_module(f"zarr.store.{submodule}")
        store = getattr(mod, new)
    except KeyError as e:
        raise ImportError(f"cannot import name {old}") from e
    else:
        warnings.warn(
            "'zarr.storage' is deprecated. Import from 'zarr.store' instead.",
            FutureWarning,
            stacklevel=3,
        )
        return store


def __getattr__(name: str) -> Any:
    if name in _names:
        return _deprecated_import(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
