import tempfile
from pathlib import Path
from typing import Literal

import pytest
from _pytest.compat import LEGACY_PATH
from upath import UPath

from zarr.storage._utils import normalize_path
from zarr.storage.common import StoreLike, StorePath, make_store_path
from zarr.storage.local import LocalStore
from zarr.storage.memory import MemoryStore
from zarr.storage.remote import RemoteStore


@pytest.mark.parametrize("path", [None, "", "bar"])
async def test_make_store_path_none(path: str) -> None:
    """
    Test that creating a store_path with None creates a memorystore
    """
    store_path = await make_store_path(None, path=path)
    assert isinstance(store_path.store, MemoryStore)
    assert store_path.path == normalize_path(path)


@pytest.mark.parametrize("path", [None, "", "bar"])
@pytest.mark.parametrize("store_type", [str, Path, LocalStore])
@pytest.mark.parametrize("mode", ["r", "w", "a"])
async def test_make_store_path_local(
    tmpdir: LEGACY_PATH,
    store_type: type[str] | type[Path] | type[LocalStore],
    path: str,
    mode: Literal["r", "w", "a"],
) -> None:
    """
    Test the various ways of invoking make_store_path that create a LocalStore
    """
    store_like = store_type(str(tmpdir))
    store_path = await make_store_path(store_like, path=path, mode=mode)
    assert isinstance(store_path.store, LocalStore)
    assert Path(store_path.store.root) == Path(tmpdir)
    assert store_path.path == normalize_path(path)
    assert store_path.store.mode.str == mode


@pytest.mark.parametrize("path", [None, "", "bar"])
@pytest.mark.parametrize("mode", ["r", "w", "a"])
async def test_make_store_path_store_path(
    tmpdir: LEGACY_PATH, path: str, mode: Literal["r", "w", "a"]
) -> None:
    """
    Test invoking make_store_path when the input is another store_path. In particular we want to ensure
    that a new path is handled correctly.
    """
    store_like = StorePath(LocalStore(str(tmpdir)), path="root")
    store_path = await make_store_path(store_like, path=path, mode=mode)
    assert isinstance(store_path.store, LocalStore)
    assert Path(store_path.store.root) == Path(tmpdir)
    path_normalized = normalize_path(path)
    assert store_path.path == (store_like / path_normalized).path

    assert store_path.store.mode.str == mode


async def test_make_store_path_invalid() -> None:
    """
    Test that invalid types raise TypeError
    """
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


@pytest.mark.parametrize(
    "path",
    [
        "/foo/bar",
        "//foo/bar",
        "foo///bar",
        "foo/bar///",
        Path("foo/bar"),
        b"foo/bar",
        UPath("foo/bar"),
    ],
)
def test_normalize_path_valid(path: str | bytes | Path | UPath) -> None:
    assert normalize_path(path) == "foo/bar"


def test_normalize_path_none():
    assert normalize_path(None) == ""


@pytest.mark.parametrize("path", [".", ".."])
def test_normalize_path_invalid(path: str):
    with pytest.raises(ValueError):
        normalize_path(path)
