import tempfile
from pathlib import Path

import pytest
from _pytest.compat import LEGACY_PATH

import zarr
from zarr import Group
from zarr.core.common import AccessModeLiteral, ZarrFormat
from zarr.storage import FsspecStore, LocalStore, MemoryStore, StoreLike, StorePath, ZipStore
from zarr.storage._common import contains_array, contains_group, make_store_path
from zarr.storage._utils import (
    _join_paths,
    _normalize_path_keys,
    _normalize_paths,
    _relativize_path,
    normalize_path,
)


@pytest.fixture(
    params=["none", "temp_dir_str", "temp_dir_path", "store_path", "memory_store", "dict"]
)
def store_like(request):
    if request.param == "none":
        yield None
    elif request.param == "temp_dir_str":
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    elif request.param == "temp_dir_path":
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    elif request.param == "store_path":
        yield StorePath(store=MemoryStore(store_dict={}), path="/")
    elif request.param == "memory_store":
        yield MemoryStore(store_dict={})
    elif request.param == "dict":
        yield {}


@pytest.mark.parametrize("path", ["foo", "foo/bar"])
@pytest.mark.parametrize("write_group", [True, False])
@pytest.mark.parametrize("zarr_format", [2, 3])
async def test_contains_group(
    local_store, path: str, write_group: bool, zarr_format: ZarrFormat
) -> None:
    """
    Test that the contains_group method correctly reports the existence of a group.
    """
    root = Group.from_store(store=local_store, zarr_format=zarr_format)
    if write_group:
        root.create_group(path)
    store_path = StorePath(local_store, path=path)
    assert await contains_group(store_path, zarr_format=zarr_format) == write_group


@pytest.mark.parametrize("path", ["foo", "foo/bar"])
@pytest.mark.parametrize("write_array", [True, False])
@pytest.mark.parametrize("zarr_format", [2, 3])
async def test_contains_array(
    local_store, path: str, write_array: bool, zarr_format: ZarrFormat
) -> None:
    """
    Test that the contains array method correctly reports the existence of an array.
    """
    root = Group.from_store(store=local_store, zarr_format=zarr_format)
    if write_array:
        root.create_array(path, shape=(100,), chunks=(10,), dtype="i4")
    store_path = StorePath(local_store, path=path)
    assert await contains_array(store_path, zarr_format=zarr_format) == write_array


@pytest.mark.parametrize("func", [contains_array, contains_group])
async def test_contains_invalid_format_raises(local_store, func: callable) -> None:
    """
    Test contains_group and contains_array raise errors for invalid zarr_formats
    """
    store_path = StorePath(local_store)
    with pytest.raises(ValueError):
        assert await func(store_path, zarr_format="3.0")


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


@pytest.mark.parametrize("modes", [(True, "w"), (False, "x")])
async def test_store_path_invalid_mode_raises(tmpdir: LEGACY_PATH, modes: tuple) -> None:
    """
    Test that ValueErrors are raise for invalid mode.
    """
    with pytest.raises(ValueError):
        await StorePath.open(LocalStore(str(tmpdir), read_only=modes[0]), path=None, mode=modes[1])


async def test_make_store_path_invalid() -> None:
    """
    Test that invalid types raise TypeError
    """
    with pytest.raises(TypeError):
        await make_store_path(1)  # type: ignore[arg-type]


async def test_make_store_path_fsspec(monkeypatch) -> None:
    pytest.importorskip("fsspec")
    pytest.importorskip("requests")
    pytest.importorskip("aiohttp")
    store_path = await make_store_path("http://foo.com/bar")
    assert isinstance(store_path.store, FsspecStore)


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


@pytest.mark.parametrize("paths", [("", "foo"), ("foo", "bar")])
def test_join_paths(paths: tuple[str, str]) -> None:
    """
    Test that _join_paths joins paths in a way that is robust to an empty string
    """
    observed = _join_paths(paths)
    if paths[0] == "":
        assert observed == paths[1]
    else:
        assert observed == "/".join(paths)


class TestNormalizePaths:
    @staticmethod
    def test_valid() -> None:
        """
        Test that path normalization works as expected
        """
        paths = ["a", "b", "c", "d", "", "//a///b//"]
        assert _normalize_paths(paths) == tuple(normalize_path(p) for p in paths)

    @staticmethod
    @pytest.mark.parametrize("paths", [("", "/"), ("///a", "a")])
    def test_invalid(paths: tuple[str, str]) -> None:
        """
        Test that name collisions after normalization raise a ``ValueError``
        """
        msg = (
            f"After normalization, the value '{paths[1]}' collides with '{paths[0]}'. "
            f"Both '{paths[1]}' and '{paths[0]}' normalize to the same value: '{normalize_path(paths[0])}'. "
            f"You should use either '{paths[1]}' or '{paths[0]}', but not both."
        )
        with pytest.raises(ValueError, match=msg):
            _normalize_paths(paths)


def test_normalize_path_keys():
    """
    Test that ``_normalize_path_keys`` just applies the normalize_path function to each key of its
    input
    """
    data = {"a": 10, "//b": 10}
    assert _normalize_path_keys(data) == {normalize_path(k): v for k, v in data.items()}


@pytest.mark.parametrize(
    ("path", "prefix", "expected"),
    [
        ("a", "", "a"),
        ("a/b/c", "a/b", "c"),
        ("a/b/c", "a", "b/c"),
    ],
)
def test_relativize_path_valid(path: str, prefix: str, expected: str) -> None:
    """
    Test the normal behavior of the _relativize_path function. Prefixes should be removed from the
    path argument.
    """
    assert _relativize_path(path=path, prefix=prefix) == expected


def test_relativize_path_invalid() -> None:
    path = "a/b/c"
    prefix = "b"
    msg = f"The first component of {path} does not start with {prefix}."
    with pytest.raises(ValueError, match=msg):
        _relativize_path(path="a/b/c", prefix="b")


def test_different_open_mode(tmp_path: LEGACY_PATH) -> None:
    # Test with a store that implements .with_read_only()
    store = MemoryStore()
    zarr.create((100,), store=store, zarr_format=2, path="a")
    arr = zarr.open_array(store=store, path="a", zarr_format=2, mode="r")
    assert arr.store.read_only

    # Test with a store that doesn't implement .with_read_only()
    zarr_path = tmp_path / "foo.zarr"
    store = ZipStore(zarr_path, mode="w")
    zarr.create((100,), store=store, zarr_format=2, path="a")
    with pytest.raises(
        ValueError,
        match="Store is not read-only but mode is 'r'. Unable to create a read-only copy of the store. Please use a read-only store or a storage class that implements .with_read_only().",
    ):
        zarr.open_array(store=store, path="a", zarr_format=2, mode="r")
