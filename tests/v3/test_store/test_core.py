import tempfile
from pathlib import Path

import pytest

from zarr.storage.common import StoreLike, StorePath, make_store_path
from zarr.storage.local import LocalStore
from zarr.storage.memory import MemoryStore
from zarr.storage.remote import RemoteStore


async def test_make_store_path(tmpdir: str) -> None:
    # None
    store_path = await make_store_path(None)
    assert isinstance(store_path.store, MemoryStore)

    # str
    store_path = await make_store_path(str(tmpdir))
    assert isinstance(store_path.store, LocalStore)
    assert Path(store_path.store.root) == Path(tmpdir)

    # Path
    store_path = await make_store_path(Path(tmpdir))
    assert isinstance(store_path.store, LocalStore)
    assert Path(store_path.store.root) == Path(tmpdir)

    # Store
    store_path = await make_store_path(store_path.store)
    assert isinstance(store_path.store, LocalStore)
    assert Path(store_path.store.root) == Path(tmpdir)

    # StorePath
    store_path = await make_store_path(store_path)
    assert isinstance(store_path.store, LocalStore)
    assert Path(store_path.store.root) == Path(tmpdir)

    with pytest.raises(TypeError):
        await make_store_path(1)  # type: ignore[arg-type]


async def test_make_store_path_fsspec(monkeypatch) -> None:
    import fsspec.implementations.memory

    monkeypatch.setattr(fsspec.implementations.memory.MemoryFileSystem, "async_impl", True)
    store_path = await make_store_path("memory://")
    assert isinstance(store_path.store, RemoteStore)


@pytest.mark.parametrize(
    "store_like",
    [
        None,
        str(tempfile.TemporaryDirectory()),
        Path(tempfile.TemporaryDirectory().name),
        StorePath(store=MemoryStore(store_dict={}, mode="w"), path="/"),
        MemoryStore(store_dict={}, mode="w"),
        {},
    ],
)
async def test_make_store_path_storage_options_raises(store_like: StoreLike) -> None:
    with pytest.raises(TypeError, match="storage_options"):
        await make_store_path(store_like, storage_options={"foo": "bar"})


async def test_unsupported() -> None:
    with pytest.raises(TypeError, match="Unsupported type for store_like: 'int'"):
        await make_store_path(1)  # type: ignore[arg-type]
