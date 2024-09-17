import importlib

import pytest

import zarr.store.local
import zarr.store.memory
import zarr.store.remote
import zarr.store.zip


@pytest.mark.parametrize(
    ["name", "expected"],
    [
        ("DictStore", zarr.store.memory.MemoryStore),
        ("KVStore", zarr.store.memory.MemoryStore),
        ("DirectoryStore", zarr.store.local.LocalStore),
        ("FSStore", zarr.store.remote.RemoteStore),
        ("ZipStore", zarr.store.zip.ZipStore),
    ],
)
def test_storage_deprecated(name: str, expected: type) -> None:
    with pytest.warns(FutureWarning, match="zarr.store"):
        result = getattr(importlib.import_module("zarr.storage"), name)

    assert result == expected
