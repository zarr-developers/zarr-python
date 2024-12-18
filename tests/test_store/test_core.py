import tempfile
from pathlib import Path

import pytest
from _pytest.compat import LEGACY_PATH

from zarr.core.common import AccessModeLiteral
from zarr.storage._utils import normalize_path
from zarr.storage.common import StoreLike, StorePath, make_store_path
from zarr.storage.fsspec import FsspecStore
from zarr.storage.local import LocalStore
from zarr.storage.memory import MemoryStore


@pytest.mark.parametrize("path", [None, "", "bar"])
async def test_make_store_path_none(path: str) -> None:
    """
    Test that creating a store_path with None creates a memorystore
    """
    store_path = await make_store_path(None, path=path)
    assert isinstance(store_path.store, MemoryStore)
    assert store_path.path == normalize_path(path)


@pytest.mark.parametrize("path", [None, "", "bar"])
@pytest.mark.parametrize("store_type", [str, Path])
@pytest.mark.parametrize("mode", ["r", "w"])
async def test_make_store_path_local(
    tmpdir: LEGACY_PATH,
    store_type: type[str] | type[Path] | type[LocalStore],
    path: str,
    mode: AccessModeLiteral,
) -> None:
    """
    Test the various ways of invoking make_store_path that create a LocalStore
    """
    store_like = store_type(str(tmpdir))
    store_path = await make_store_path(store_like, path=path, mode=mode)
    assert isinstance(store_path.store, LocalStore)
    assert Path(store_path.store.root) == Path(tmpdir)
    assert store_path.path == normalize_path(path)
    assert store_path.read_only == (mode == "r")


@pytest.mark.parametrize("path", [None, "", "bar"])
@pytest.mark.parametrize("mode", ["r", "w"])
async def test_make_store_path_store_path(
    tmpdir: LEGACY_PATH, path: str, mode: AccessModeLiteral
) -> None:
    """
    Test invoking make_store_path when the input is another store_path. In particular we want to ensure
    that a new path is handled correctly.
    """
    ro = mode == "r"
    store_like = await StorePath.open(LocalStore(str(tmpdir), read_only=ro), path="root", mode=mode)
    store_path = await make_store_path(store_like, path=path, mode=mode)
    assert isinstance(store_path.store, LocalStore)
    assert Path(store_path.store.root) == Path(tmpdir)
    path_normalized = normalize_path(path)
    assert store_path.path == (store_like / path_normalized).path

    assert store_path.read_only == ro


async def test_make_store_path_invalid() -> None:
    """
    Test that invalid types raise TypeError
    """
    with pytest.raises(TypeError):
        await make_store_path(1)  # type: ignore[arg-type]


async def test_make_store_path_fsspec(monkeypatch) -> None:
    pytest.importorskip("fsspec")
    store_path = await make_store_path("http://foo.com/bar")
    assert isinstance(store_path.store, FsspecStore)


@pytest.mark.parametrize(
    "store_like",
    [
        None,
        tempfile.TemporaryDirectory().name,
        Path(tempfile.TemporaryDirectory().name),
        StorePath(store=MemoryStore(store_dict={}), path="/"),
        MemoryStore(store_dict={}),
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
    ],
)
def test_normalize_path_valid(path: str | bytes | Path) -> None:
    assert normalize_path(path) == "foo/bar"


def test_normalize_path_upath() -> None:
    upath = pytest.importorskip("upath")
    assert normalize_path(upath.UPath("foo/bar")) == "foo/bar"


def test_normalize_path_none():
    assert normalize_path(None) == ""


@pytest.mark.parametrize("path", [".", ".."])
def test_normalize_path_invalid(path: str):
    with pytest.raises(ValueError):
        normalize_path(path)
