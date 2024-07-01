from pathlib import Path

import pytest

from zarr.store.core import make_store_path
from zarr.store.local import LocalStore
from zarr.store.memory import MemoryStore


def test_make_store_path(tmpdir) -> None:
    # None
    store_path = make_store_path(None)
    assert isinstance(store_path.store, MemoryStore)

    # str
    store_path = make_store_path(str(tmpdir))
    assert isinstance(store_path.store, LocalStore)
    assert Path(store_path.store.root) == Path(tmpdir)

    # Path
    store_path = make_store_path(Path(tmpdir))
    assert isinstance(store_path.store, LocalStore)
    assert Path(store_path.store.root) == Path(tmpdir)

    # Store
    store_path = make_store_path(store_path.store)
    assert isinstance(store_path.store, LocalStore)
    assert Path(store_path.store.root) == Path(tmpdir)

    # StorePath
    store_path = make_store_path(store_path)
    assert isinstance(store_path.store, LocalStore)
    assert Path(store_path.store.root) == Path(tmpdir)

    with pytest.raises(TypeError):
        make_store_path(1)
