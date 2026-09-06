from __future__ import annotations

import importlib
import inspect
import subprocess
import sys

import pytest

from zarr_storage import legacy


@pytest.mark.parametrize(
    "name",
    [
        "MemoryStore",
        "ManagedMemoryStore",
        "GpuMemoryStore",
        "LocalStore",
        "ZipStore",
        "FsspecStore",
        "ObjectStore",
        "WrapperStore",
        "LoggingStore",
        "LatencyStore",
    ],
)
def test_stores_belong_to_extracted_hierarchy(name: str) -> None:
    store_class = getattr(legacy, name)
    assert issubclass(store_class, legacy.Store)
    assert store_class.__module__.startswith("zarr_storage.")


def test_cache_store_belongs_to_extracted_hierarchy() -> None:
    module = importlib.import_module("zarr_storage.legacy.experimental.cache_store")
    assert issubclass(module.CacheStore, legacy.WrapperStore)


def test_conformance_suite_is_distributed() -> None:
    module = importlib.import_module("zarr_storage.testing")
    assert module.StoreTests.__module__ == "zarr_storage.testing.store"


@pytest.mark.parametrize(
    ("old_module", "new_module", "name"),
    [
        ("zarr.storage", "zarr_storage.legacy", name)
        for name in (
            "MemoryStore",
            "ManagedMemoryStore",
            "GpuMemoryStore",
            "LocalStore",
            "ZipStore",
            "FsspecStore",
            "ObjectStore",
            "WrapperStore",
            "LoggingStore",
            "StorePath",
        )
    ]
    + [
        ("zarr.testing.store", "zarr_storage.legacy", "LatencyStore"),
        (
            "zarr.experimental.cache_store",
            "zarr_storage.legacy.experimental.cache_store",
            "CacheStore",
        ),
    ],
)
def test_store_implementation_signatures(old_module: str, new_module: str, name: str) -> None:
    old = getattr(importlib.import_module(old_module), name)
    new = getattr(importlib.import_module(new_module), name)
    assert old is not new
    assert str(inspect.signature(old)) == str(inspect.signature(new))
    for method_name, method in vars(old).items():
        if isinstance(method, (classmethod, staticmethod)):
            method = method.__func__
            replacement = vars(new)[method_name].__func__
        elif inspect.isfunction(method):
            replacement = vars(new)[method_name]
        elif isinstance(method, property):
            method = method.fget
            replacement = vars(new)[method_name].fget
        else:
            continue
        assert str(inspect.signature(method)) == str(inspect.signature(replacement))
        assert inspect.iscoroutinefunction(method) == inspect.iscoroutinefunction(replacement)


def test_import_keeps_zarr_bindings_and_does_not_import_pytest() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import zarr.storage
from zarr.abc.store import Store

original_memory_store = zarr.storage.MemoryStore
from zarr_storage.legacy import LatencyStore, MemoryStore

store = LatencyStore(MemoryStore())
assert type(store._store) is MemoryStore
assert zarr.storage.MemoryStore is original_memory_store
assert not isinstance(store._store, Store)
assert 'pytest' not in sys.modules
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
