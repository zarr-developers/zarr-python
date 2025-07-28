from __future__ import annotations

import inspect
import re
from typing import TYPE_CHECKING

import zarr.codecs
import zarr.storage
from zarr.core.array import init_array
from zarr.storage._common import StorePath

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from zarr.abc.store import Store
    from zarr.core.common import JSON, MemoryOrder, ZarrFormat

import contextlib
from typing import Literal

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import zarr
import zarr.api.asynchronous
import zarr.api.synchronous
import zarr.core.group
from zarr import Array, Group
from zarr.api.synchronous import (
    create,
    create_array,
    create_group,
    from_array,
    group,
    load,
    open_group,
    save,
    save_array,
    save_group,
)
from zarr.core.buffer import NDArrayLike
from zarr.errors import MetadataValidationError
from zarr.storage import LocalStore, MemoryStore, ZipStore
from zarr.storage._utils import normalize_path
from zarr.testing.utils import gpu_test


def test_create(memory_store: Store) -> None:
    store = memory_store

    # create array
    z = create(shape=100, store=store)
    assert isinstance(z, Array)
    assert z.shape == (100,)

    # create array, overwrite, specify chunk shape
    z = create(shape=200, chunk_shape=20, store=store, overwrite=True)
    assert isinstance(z, Array)
    assert z.shape == (200,)
    assert z.chunks == (20,)

    # create array, overwrite, specify chunk shape via chunks param
    z = create(shape=400, chunks=40, store=store, overwrite=True)
    assert isinstance(z, Array)
    assert z.shape == (400,)
    assert z.chunks == (40,)

    # create array with float shape
    with pytest.raises(TypeError):
        z = create(shape=(400.5, 100), store=store, overwrite=True)  # type: ignore [arg-type]

    # create array with float chunk shape
    with pytest.raises(TypeError):
        z = create(shape=(400, 100), chunks=(16, 16.5), store=store, overwrite=True)  # type: ignore [arg-type]


# TODO: parametrize over everything this function takes
@pytest.mark.parametrize("store", ["memory"], indirect=True)
def test_create_array(store: Store, zarr_format: ZarrFormat) -> None:
    attrs: dict[str, JSON] = {"foo": 100}  # explicit type annotation to avoid mypy error
    shape = (10, 10)
    path = "foo"
    data_val = 1
    array_w = create_array(
        store,
        name=path,
        shape=shape,
        attributes=attrs,
        chunks=shape,
        dtype="uint8",
        zarr_format=zarr_format,
    )
    array_w[:] = data_val
    assert array_w.shape == shape
    assert array_w.attrs == attrs
    assert np.array_equal(array_w[:], np.zeros(shape, dtype=array_w.dtype) + data_val)


@pytest.mark.parametrize("write_empty_chunks", [True, False])
def test_write_empty_chunks_warns(write_empty_chunks: bool, zarr_format: ZarrFormat) -> None:
    """
    Test that using the `write_empty_chunks` kwarg on array access will raise a warning.
    """
    match = "The `write_empty_chunks` keyword argument .*"
    with pytest.warns(RuntimeWarning, match=match):
        _ = zarr.array(
            data=np.arange(10),
            shape=(10,),
            dtype="uint8",
            write_empty_chunks=write_empty_chunks,
            zarr_format=zarr_format,
        )

    with pytest.warns(RuntimeWarning, match=match):
        _ = zarr.create(
            shape=(10,),
            dtype="uint8",
            write_empty_chunks=write_empty_chunks,
            zarr_format=zarr_format,
        )


@pytest.mark.parametrize("path", ["foo", "/", "/foo", "///foo/bar"])
@pytest.mark.parametrize("node_type", ["array", "group"])
def test_open_normalized_path(
    memory_store: MemoryStore, path: str, node_type: Literal["array", "group"]
) -> None:
    node: Group | Array
    if node_type == "group":
        node = group(store=memory_store, path=path)
    elif node_type == "array":
        node = create(store=memory_store, path=path, shape=(2,))

    assert node.path == normalize_path(path)


async def test_open_array(memory_store: MemoryStore, zarr_format: ZarrFormat) -> None:
    store = memory_store

    # open array, create if doesn't exist
    z = zarr.api.synchronous.open(store=store, shape=100, zarr_format=zarr_format)
    assert isinstance(z, Array)
    assert z.shape == (100,)

    # open array, overwrite
    # store._store_dict = {}
    store = MemoryStore()
    z = zarr.api.synchronous.open(store=store, shape=200, zarr_format=zarr_format)
    assert isinstance(z, Array)
    assert z.shape == (200,)

    # open array, read-only
    store_cls = type(store)
    ro_store = await store_cls.open(store_dict=store._store_dict, read_only=True)
    z = zarr.api.synchronous.open(store=ro_store, mode="r")
    assert isinstance(z, Array)
    assert z.shape == (200,)
    assert z.read_only

    # path not found
    with pytest.raises(FileNotFoundError):
        zarr.api.synchronous.open(store="doesnotexist", mode="r", zarr_format=zarr_format)


@pytest.mark.parametrize("store", ["memory", "local", "zip"], indirect=True)
def test_v2_and_v3_exist_at_same_path(store: Store) -> None:
    zarr.create_array(store, shape=(10,), dtype="uint8", zarr_format=3)
    zarr.create_array(store, shape=(10,), dtype="uint8", zarr_format=2)
    msg = f"Both zarr.json (Zarr format 3) and .zarray (Zarr format 2) metadata objects exist at {store}. Zarr v3 will be used."
    with pytest.warns(UserWarning, match=re.escape(msg)):
        zarr.open(store=store)


@pytest.mark.parametrize("store", ["memory"], indirect=True)
async def test_create_group(store: Store, zarr_format: ZarrFormat) -> None:
    attrs = {"foo": 100}
    path = "node"
    node = create_group(store, path=path, attributes=attrs, zarr_format=zarr_format)
    assert isinstance(node, Group)
    assert node.attrs == attrs
    assert node.metadata.zarr_format == zarr_format


async def test_open_group(memory_store: MemoryStore) -> None:
    store = memory_store

    # open group, create if doesn't exist
    g = open_group(store=store)
    g.create_group("foo")
    assert isinstance(g, Group)
    assert "foo" in g

    # open group, overwrite
    g = open_group(store=store, mode="w")
    assert isinstance(g, Group)
    assert "foo" not in g

    # open group, read-only
    store_cls = type(store)
    ro_store = await store_cls.open(store_dict=store._store_dict, read_only=True)
    g = open_group(store=ro_store, mode="r")
    assert isinstance(g, Group)
    assert g.read_only


@pytest.mark.parametrize("zarr_format", [None, 2, 3])
async def test_open_group_unspecified_version(tmpdir: Path, zarr_format: ZarrFormat) -> None:
    """Regression test for https://github.com/zarr-developers/zarr-python/issues/2175"""

    # create a group with specified zarr format (could be 2, 3, or None)
    _ = await zarr.api.asynchronous.open_group(
        store=str(tmpdir), mode="w", zarr_format=zarr_format, attributes={"foo": "bar"}
    )

    # now open that group without specifying the format
    g2 = await zarr.api.asynchronous.open_group(store=str(tmpdir), mode="r")

    assert g2.attrs == {"foo": "bar"}

    if zarr_format is not None:
        assert g2.metadata.zarr_format == zarr_format


@pytest.mark.parametrize("store", ["local", "memory", "zip"], indirect=["store"])
@pytest.mark.parametrize("n_args", [10, 1, 0])
@pytest.mark.parametrize("n_kwargs", [10, 1, 0])
@pytest.mark.parametrize("path", [None, "some_path"])
def test_save(store: Store, n_args: int, n_kwargs: int, path: None | str) -> None:
    data = np.arange(10)
    args = [np.arange(10) for _ in range(n_args)]
    kwargs = {f"arg_{i}": data for i in range(n_kwargs)}

    if n_kwargs == 0 and n_args == 0:
        with pytest.raises(ValueError):
            save(store, path=path)
    elif n_args == 1 and n_kwargs == 0:
        save(store, *args, path=path)
        array = zarr.api.synchronous.open(store, path=path)
        assert isinstance(array, Array)
        assert_array_equal(array[:], data)
    else:
        save(store, *args, path=path, **kwargs)  # type: ignore [arg-type]
        group = zarr.api.synchronous.open(store, path=path)
        assert isinstance(group, Group)
        for array in group.array_values():
            assert_array_equal(array[:], data)
        for k in kwargs:
            assert k in group
        assert group.nmembers() == n_args + n_kwargs


def test_save_errors() -> None:
    with pytest.raises(ValueError):
        # no arrays provided
        save_group("data/group.zarr")
    with pytest.raises(TypeError):
        # no array provided
        save_array("data/group.zarr")  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        # no arrays provided
        save("data/group.zarr")
    a = np.arange(10)
    with pytest.raises(TypeError):
        # mode is no valid argument and would get handled as an array
        zarr.save("data/example.zarr", a, mode="w")


def test_open_with_mode_r(tmp_path: Path) -> None:
    # 'r' means read only (must exist)
    with pytest.raises(FileNotFoundError):
        zarr.open(store=tmp_path, mode="r")
    z1 = zarr.ones(store=tmp_path, shape=(3, 3))
    assert z1.fill_value == 1
    z2 = zarr.open(store=tmp_path, mode="r")
    assert isinstance(z2, Array)
    assert z2.fill_value == 1
    result = z2[:]
    assert isinstance(result, NDArrayLike)
    assert (result == 1).all()
    with pytest.raises(ValueError):
        z2[:] = 3


def test_open_with_mode_r_plus(tmp_path: Path) -> None:
    # 'r+' means read/write (must exist)
    with pytest.raises(FileNotFoundError):
        zarr.open(store=tmp_path, mode="r+")
    zarr.ones(store=tmp_path, shape=(3, 3))
    z2 = zarr.open(store=tmp_path, mode="r+")
    assert isinstance(z2, Array)
    result = z2[:]
    assert isinstance(result, NDArrayLike)
    assert (result == 1).all()
    z2[:] = 3


async def test_open_with_mode_a(tmp_path: Path) -> None:
    # Open without shape argument should default to group
    g = zarr.open(store=tmp_path, mode="a")
    assert isinstance(g, Group)
    await g.store_path.delete()

    # 'a' means read/write (create if doesn't exist)
    arr = zarr.open(store=tmp_path, mode="a", shape=(3, 3))
    assert isinstance(arr, Array)
    arr[...] = 1
    z2 = zarr.open(store=tmp_path, mode="a")
    assert isinstance(z2, Array)
    result = z2[:]
    assert isinstance(result, NDArrayLike)
    assert (result == 1).all()
    z2[:] = 3


def test_open_with_mode_w(tmp_path: Path) -> None:
    # 'w' means create (overwrite if exists);
    arr = zarr.open(store=tmp_path, mode="w", shape=(3, 3))
    assert isinstance(arr, Array)

    arr[...] = 3
    z2 = zarr.open(store=tmp_path, mode="w", shape=(3, 3))
    assert isinstance(z2, Array)
    result = z2[:]
    assert isinstance(result, NDArrayLike)
    assert not (result == 3).all()
    z2[:] = 3


def test_open_with_mode_w_minus(tmp_path: Path) -> None:
    # 'w-' means create  (fail if exists)
    arr = zarr.open(store=tmp_path, mode="w-", shape=(3, 3))
    assert isinstance(arr, Array)
    arr[...] = 1
    with pytest.raises(FileExistsError):
        zarr.open(store=tmp_path, mode="w-")


@pytest.mark.parametrize("order", ["C", "F", None])
@pytest.mark.parametrize("config", [{"order": "C"}, {"order": "F"}, {}], ids=["C", "F", "None"])
def test_array_order(
    order: MemoryOrder | None, config: dict[str, MemoryOrder | None], zarr_format: ZarrFormat
) -> None:
    """
    Check that:
    - For v2, memory order is taken from the `order` keyword argument.
    - For v3, memory order is taken from `config`, and when order is passed a warning is raised
    - The numpy array returned has the expected order
    - For v2, the order metadata is set correctly
    """
    default_order = zarr.config.get("array.order")
    ctx: contextlib.AbstractContextManager  # type: ignore[type-arg]

    if zarr_format == 3:
        if order is None:
            ctx = contextlib.nullcontext()
        else:
            ctx = pytest.warns(
                RuntimeWarning,
                match="The `order` keyword argument has no effect for Zarr format 3 arrays",
            )

        expected_order = config.get("order", default_order)

    if zarr_format == 2:
        ctx = contextlib.nullcontext()
        expected_order = order or config.get("order", default_order)

    with ctx:
        arr = zarr.ones(shape=(2, 2), order=order, zarr_format=zarr_format, config=config)

    assert arr.order == expected_order
    vals = np.asarray(arr)
    if expected_order == "C":
        assert vals.flags.c_contiguous
    elif expected_order == "F":
        assert vals.flags.f_contiguous
    else:
        raise AssertionError

    if zarr_format == 2:
        assert arr.metadata.zarr_format == 2
        assert arr.metadata.order == expected_order


async def test_init_order_warns() -> None:
    with pytest.warns(
        RuntimeWarning, match="The `order` keyword argument has no effect for Zarr format 3 arrays"
    ):
        await init_array(
            store_path=StorePath(store=MemoryStore()),
            shape=(1,),
            dtype="uint8",
            config=None,
            zarr_format=3,
            order="F",
        )


# def test_lazy_loader():
#     foo = np.arange(100)
#     bar = np.arange(100, 0, -1)
#     store = "data/group.zarr"
#     save(store, foo=foo, bar=bar)
#     loader = load(store)
#     assert "foo" in loader
#     assert "bar" in loader
#     assert "baz" not in loader
#     assert len(loader) == 2
#     assert sorted(loader) == ["bar", "foo"]
#     assert_array_equal(foo, loader["foo"])
#     assert_array_equal(bar, loader["bar"])
#     assert "LazyLoader: " in repr(loader)


def test_load_array(sync_store: Store) -> None:
    store = sync_store
    foo = np.arange(100)
    bar = np.arange(100, 0, -1)
    save(store, foo=foo, bar=bar)

    # can also load arrays directly into a numpy array
    for array_name in ["foo", "bar"]:
        array = load(store, path=array_name)
        assert isinstance(array, np.ndarray)
        if array_name == "foo":
            assert_array_equal(foo, array)
        else:
            assert_array_equal(bar, array)


@pytest.mark.parametrize("path", ["data", None])
@pytest.mark.parametrize("load_read_only", [True, False, None])
def test_load_zip(tmp_path: Path, path: str | None, load_read_only: bool | None) -> None:
    file = tmp_path / "test.zip"
    data = np.arange(100).reshape(10, 10)

    with ZipStore(file, mode="w", read_only=False) as zs:
        save(zs, data, path=path)
    with ZipStore(file, mode="r", read_only=load_read_only) as zs:
        result = zarr.load(store=zs, path=path)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, data)
    with ZipStore(file, read_only=load_read_only) as zs:
        result = zarr.load(store=zs, path=path)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, data)


@pytest.mark.parametrize("path", ["data", None])
@pytest.mark.parametrize("load_read_only", [True, False])
def test_load_local(tmp_path: Path, path: str | None, load_read_only: bool) -> None:
    file = tmp_path / "test.zip"
    data = np.arange(100).reshape(10, 10)

    with LocalStore(file, read_only=False) as zs:
        save(zs, data, path=path)
    with LocalStore(file, read_only=load_read_only) as zs:
        result = zarr.load(store=zs, path=path)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, data)


def test_tree() -> None:
    pytest.importorskip("rich")
    g1 = zarr.group()
    g1.create_group("foo")
    g3 = g1.create_group("bar")
    g3.create_group("baz")
    g5 = g3.create_group("qux")
    g5.create_array("baz", shape=(100,), chunks=(10,), dtype="float64")
    with pytest.warns(DeprecationWarning, match=r"Group\.tree instead\."):  # noqa: PT031
        assert repr(zarr.tree(g1)) == repr(g1.tree())
        assert str(zarr.tree(g1)) == str(g1.tree())


# @pytest.mark.parametrize("stores_from_path", [False, True])
# @pytest.mark.parametrize(
#     "with_chunk_store,listable",
#     [(False, True), (True, True), (False, False)],
#     ids=["default-listable", "with_chunk_store-listable", "default-unlistable"],
# )
# def test_consolidate_metadata(with_chunk_store, listable, monkeypatch, stores_from_path):
#     # setup initial data
#     if stores_from_path:
#         store = tempfile.mkdtemp()
#         atexit.register(atexit_rmtree, store)
#         if with_chunk_store:
#             chunk_store = tempfile.mkdtemp()
#             atexit.register(atexit_rmtree, chunk_store)
#         else:
#             chunk_store = None
#     else:
#         store = MemoryStore()
#         chunk_store = MemoryStore() if with_chunk_store else None
#     path = None
#     z = group(store, chunk_store=chunk_store, path=path)

#     # Reload the actual store implementation in case str
#     store_to_copy = z.store

#     z.create_group("g1")
#     g2 = z.create_group("g2")
#     g2.attrs["hello"] = "world"
#     arr = g2.create_array("arr", shape=(20, 20), chunks=(5, 5), dtype="f8")
#     assert 16 == arr.nchunks
#     assert 0 == arr.nchunks_initialized
#     arr.attrs["data"] = 1
#     arr[:] = 1.0
#     assert 16 == arr.nchunks_initialized

#     if stores_from_path:
#         # get the actual store class for use with consolidate_metadata
#         store_class = z._store
#     else:
#         store_class = store

#     # perform consolidation
#     out = consolidate_metadata(store_class, path=path)
#     assert isinstance(out, Group)
#     assert ["g1", "g2"] == list(out)
#     if not stores_from_path:
#         assert isinstance(out._store, ConsolidatedMetadataStore)
#         assert ".zmetadata" in store
#         meta_keys = [
#             ".zgroup",
#             "g1/.zgroup",
#             "g2/.zgroup",
#             "g2/.zattrs",
#             "g2/arr/.zarray",
#             "g2/arr/.zattrs",
#         ]

#         for key in meta_keys:
#             del store[key]

#     # https://github.com/zarr-developers/zarr-python/issues/993
#     # Make sure we can still open consolidated on an unlistable store:
#     if not listable:
#         fs_memory = pytest.importorskip("fsspec.implementations.memory")
#         monkeypatch.setattr(fs_memory.MemoryFileSystem, "isdir", lambda x, y: False)
#         monkeypatch.delattr(fs_memory.MemoryFileSystem, "ls")
#         fs = fs_memory.MemoryFileSystem()
#         store_to_open = FSStore("", fs=fs)
#         # copy original store to new unlistable store
#         store_to_open.update(store_to_copy)

#     else:
#         store_to_open = store

#     # open consolidated
#     z2 = open_consolidated(store_to_open, chunk_store=chunk_store, path=path)
#     assert ["g1", "g2"] == list(z2)
#     assert "world" == z2.g2.attrs["hello"]
#     assert 1 == z2.g2.arr.attrs["data"]
#     assert (z2.g2.arr[:] == 1.0).all()
#     assert 16 == z2.g2.arr.nchunks
#     if listable:
#         assert 16 == z2.g2.arr.nchunks_initialized
#     else:
#         with pytest.raises(NotImplementedError):
#             _ = z2.g2.arr.nchunks_initialized

#     if stores_from_path:
#         # path string is note a BaseStore subclass so cannot be used to
#         # initialize a ConsolidatedMetadataStore.

#         with pytest.raises(ValueError):
#             cmd = ConsolidatedMetadataStore(store)
#     else:
#         # tests del/write on the store

#         cmd = ConsolidatedMetadataStore(store)
#         with pytest.raises(PermissionError):
#             del cmd[".zgroup"]
#         with pytest.raises(PermissionError):
#             cmd[".zgroup"] = None

#         # test getsize on the store
#         assert isinstance(getsize(cmd), Integral)

#     # test new metadata are not writeable
#     with pytest.raises(PermissionError):
#         z2.create_group("g3")
#     with pytest.raises(PermissionError):
#         z2.create_dataset("spam", shape=42, chunks=7, dtype="i4")
#     with pytest.raises(PermissionError):
#         del z2["g2"]

#     # test consolidated metadata are not writeable
#     with pytest.raises(PermissionError):
#         z2.g2.attrs["hello"] = "universe"
#     with pytest.raises(PermissionError):
#         z2.g2.arr.attrs["foo"] = "bar"

#     # test the data are writeable
#     z2.g2.arr[:] = 2
#     assert (z2.g2.arr[:] == 2).all()

#     # test invalid modes
#     with pytest.raises(ValueError):
#         open_consolidated(store, chunk_store=chunk_store, mode="a", path=path)
#     with pytest.raises(ValueError):
#         open_consolidated(store, chunk_store=chunk_store, mode="w", path=path)
#     with pytest.raises(ValueError):
#         open_consolidated(store, chunk_store=chunk_store, mode="w-", path=path)

#     # make sure keyword arguments are passed through without error
#     open_consolidated(
#         store,
#         chunk_store=chunk_store,
#         path=path,
#         cache_attrs=True,
#         synchronizer=None,
#     )


# @pytest.mark.parametrize(
#     "options",
#     (
#         {"dimension_separator": "/"},
#         {"dimension_separator": "."},
#         {"dimension_separator": None},
#     ),
# )
# def test_save_array_separator(tmpdir, options):
#     data = np.arange(6).reshape((3, 2))
#     url = tmpdir.join("test.zarr")
#     save_array(url, data, **options)


# class TestCopyStore(unittest.TestCase):
#     _version = 2

#     def setUp(self):
#         source = dict()
#         source["foo"] = b"xxx"
#         source["bar/baz"] = b"yyy"
#         source["bar/qux"] = b"zzz"
#         self.source = source

#     def _get_dest_store(self):
#         return dict()

#     def test_no_paths(self):
#         source = self.source
#         dest = self._get_dest_store()
#         copy_store(source, dest)
#         assert len(source) == len(dest)
#         for key in source:
#             assert source[key] == dest[key]

#     def test_source_path(self):
#         source = self.source
#         # paths should be normalized
#         for source_path in "bar", "bar/", "/bar", "/bar/":
#             dest = self._get_dest_store()
#             copy_store(source, dest, source_path=source_path)
#             assert 2 == len(dest)
#             for key in source:
#                 if key.startswith("bar/"):
#                     dest_key = key.split("bar/")[1]
#                     assert source[key] == dest[dest_key]
#                 else:
#                     assert key not in dest

#     def test_dest_path(self):
#         source = self.source
#         # paths should be normalized
#         for dest_path in "new", "new/", "/new", "/new/":
#             dest = self._get_dest_store()
#             copy_store(source, dest, dest_path=dest_path)
#             assert len(source) == len(dest)
#             for key in source:
#                 if self._version == 3:
#                     dest_key = key[:10] + "new/" + key[10:]
#                 else:
#                     dest_key = "new/" + key
#                 assert source[key] == dest[dest_key]

#     def test_source_dest_path(self):
#         source = self.source
#         # paths should be normalized
#         for source_path in "bar", "bar/", "/bar", "/bar/":
#             for dest_path in "new", "new/", "/new", "/new/":
#                 dest = self._get_dest_store()
#                 copy_store(source, dest, source_path=source_path, dest_path=dest_path)
#                 assert 2 == len(dest)
#                 for key in source:
#                     if key.startswith("bar/"):
#                         dest_key = "new/" + key.split("bar/")[1]
#                         assert source[key] == dest[dest_key]
#                     else:
#                         assert key not in dest
#                         assert ("new/" + key) not in dest

#     def test_excludes_includes(self):
#         source = self.source

#         # single excludes
#         dest = self._get_dest_store()
#         excludes = "f.*"
#         copy_store(source, dest, excludes=excludes)
#         assert len(dest) == 2

#         root = ""
#         assert root + "foo" not in dest

#         # multiple excludes
#         dest = self._get_dest_store()
#         excludes = "b.z", ".*x"
#         copy_store(source, dest, excludes=excludes)
#         assert len(dest) == 1
#         assert root + "foo" in dest
#         assert root + "bar/baz" not in dest
#         assert root + "bar/qux" not in dest

#         # excludes and includes
#         dest = self._get_dest_store()
#         excludes = "b.*"
#         includes = ".*x"
#         copy_store(source, dest, excludes=excludes, includes=includes)
#         assert len(dest) == 2
#         assert root + "foo" in dest
#         assert root + "bar/baz" not in dest
#         assert root + "bar/qux" in dest

#     def test_dry_run(self):
#         source = self.source
#         dest = self._get_dest_store()
#         copy_store(source, dest, dry_run=True)
#         assert 0 == len(dest)

#     def test_if_exists(self):
#         source = self.source
#         dest = self._get_dest_store()
#         root = ""
#         dest[root + "bar/baz"] = b"mmm"

#         # default ('raise')
#         with pytest.raises(CopyError):
#             copy_store(source, dest)

#         # explicit 'raise'
#         with pytest.raises(CopyError):
#             copy_store(source, dest, if_exists="raise")

#         # skip
#         copy_store(source, dest, if_exists="skip")
#         assert 3 == len(dest)
#         assert dest[root + "foo"] == b"xxx"
#         assert dest[root + "bar/baz"] == b"mmm"
#         assert dest[root + "bar/qux"] == b"zzz"

#         # replace
#         copy_store(source, dest, if_exists="replace")
#         assert 3 == len(dest)
#         assert dest[root + "foo"] == b"xxx"
#         assert dest[root + "bar/baz"] == b"yyy"
#         assert dest[root + "bar/qux"] == b"zzz"

#         # invalid option
#         with pytest.raises(ValueError):
#             copy_store(source, dest, if_exists="foobar")


# def check_copied_array(original, copied, without_attrs=False, expect_props=None):
#     # setup
#     source_h5py = original.__module__.startswith("h5py.")
#     dest_h5py = copied.__module__.startswith("h5py.")
#     zarr_to_zarr = not (source_h5py or dest_h5py)
#     h5py_to_h5py = source_h5py and dest_h5py
#     zarr_to_h5py = not source_h5py and dest_h5py
#     h5py_to_zarr = source_h5py and not dest_h5py
#     if expect_props is None:
#         expect_props = dict()
#     else:
#         expect_props = expect_props.copy()

#     # common properties in zarr and h5py
#     for p in "dtype", "shape", "chunks":
#         expect_props.setdefault(p, getattr(original, p))

#     # zarr-specific properties
#     if zarr_to_zarr:
#         for p in "compressor", "filters", "order", "fill_value":
#             expect_props.setdefault(p, getattr(original, p))

#     # h5py-specific properties
#     if h5py_to_h5py:
#         for p in (
#             "maxshape",
#             "compression",
#             "compression_opts",
#             "shuffle",
#             "scaleoffset",
#             "fletcher32",
#             "fillvalue",
#         ):
#             expect_props.setdefault(p, getattr(original, p))

#     # common properties with some name differences
#     if h5py_to_zarr:
#         expect_props.setdefault("fill_value", original.fillvalue)
#     if zarr_to_h5py:
#         expect_props.setdefault("fillvalue", original.fill_value)

#     # compare properties
#     for k, v in expect_props.items():
#         assert v == getattr(copied, k)

#     # compare data
#     assert_array_equal(original[:], copied[:])

#     # compare attrs
#     if without_attrs:
#         for k in original.attrs.keys():
#             assert k not in copied.attrs
#     else:
#         if dest_h5py and "filters" in original.attrs:
#             # special case in v3 (storing filters metadata under attributes)
#             # we explicitly do not copy this info over to HDF5
#             original_attrs = original.attrs.asdict().copy()
#             original_attrs.pop("filters")
#         else:
#             original_attrs = original.attrs
#         assert sorted(original_attrs.items()) == sorted(copied.attrs.items())


# def check_copied_group(original, copied, without_attrs=False, expect_props=None, shallow=False):
#     # setup
#     if expect_props is None:
#         expect_props = dict()
#     else:
#         expect_props = expect_props.copy()

#     # compare children
#     for k, v in original.items():
#         if hasattr(v, "shape"):
#             assert k in copied
#             check_copied_array(v, copied[k], without_attrs=without_attrs, expect_props=expect_props)
#         elif shallow:
#             assert k not in copied
#         else:
#             assert k in copied
#             check_copied_group(
#                 v,
#                 copied[k],
#                 without_attrs=without_attrs,
#                 shallow=shallow,
#                 expect_props=expect_props,
#             )

#     # compare attrs
#     if without_attrs:
#         for k in original.attrs.keys():
#             assert k not in copied.attrs
#     else:
#         assert sorted(original.attrs.items()) == sorted(copied.attrs.items())


# def test_copy_all():
#     """
#     https://github.com/zarr-developers/zarr-python/issues/269

#     copy_all used to not copy attributes as `.keys()` does not return hidden `.zattrs`.

#     """
#     original_group = zarr.group(store=MemoryStore(), overwrite=True)
#     original_group.attrs["info"] = "group attrs"
#     original_subgroup = original_group.create_group("subgroup")
#     original_subgroup.attrs["info"] = "sub attrs"

#     destination_group = zarr.group(store=MemoryStore(), overwrite=True)

#     # copy from memory to directory store
#     copy_all(
#         original_group,
#         destination_group,
#         dry_run=False,
#     )

#     assert "subgroup" in destination_group
#     assert destination_group.attrs["info"] == "group attrs"
#     assert destination_group.subgroup.attrs["info"] == "sub attrs"


# class TestCopy:
#     @pytest.fixture(params=[False, True], ids=["zarr", "hdf5"])
#     def source(self, request, tmpdir):
#         def prep_source(source):
#             foo = source.create_group("foo")
#             foo.attrs["experiment"] = "weird science"
#             baz = foo.create_dataset("bar/baz", data=np.arange(100), chunks=(50,))
#             baz.attrs["units"] = "metres"
#             if request.param:
#                 extra_kws = dict(
#                     compression="gzip",
#                     compression_opts=3,
#                     fillvalue=84,
#                     shuffle=True,
#                     fletcher32=True,
#                 )
#             else:
#                 extra_kws = dict(compressor=Zlib(3), order="F", fill_value=42, filters=[Adler32()])
#             source.create_dataset(
#                 "spam",
#                 data=np.arange(100, 200).reshape(20, 5),
#                 chunks=(10, 2),
#                 dtype="i2",
#                 **extra_kws,
#             )
#             return source

#         if request.param:
#             h5py = pytest.importorskip("h5py")
#             fn = tmpdir.join("source.h5")
#             with h5py.File(str(fn), mode="w") as h5f:
#                 yield prep_source(h5f)
#         else:
#             yield prep_source(group())

#     @pytest.fixture(params=[False, True], ids=["zarr", "hdf5"])
#     def dest(self, request, tmpdir):
#         if request.param:
#             h5py = pytest.importorskip("h5py")
#             fn = tmpdir.join("dest.h5")
#             with h5py.File(str(fn), mode="w") as h5f:
#                 yield h5f
#         else:
#             yield group()

#     def test_copy_array(self, source, dest):
#         # copy array with default options
#         copy(source["foo/bar/baz"], dest)
#         check_copied_array(source["foo/bar/baz"], dest["baz"])
#         copy(source["spam"], dest)
#         check_copied_array(source["spam"], dest["spam"])

#     def test_copy_bad_dest(self, source, dest):
#         # try to copy to an array, dest must be a group
#         dest = dest.create_dataset("eggs", shape=(100,))
#         with pytest.raises(ValueError):
#             copy(source["foo/bar/baz"], dest)

#     def test_copy_array_name(self, source, dest):
#         # copy array with name
#         copy(source["foo/bar/baz"], dest, name="qux")
#         assert "baz" not in dest
#         check_copied_array(source["foo/bar/baz"], dest["qux"])

#     def test_copy_array_create_options(self, source, dest):
#         dest_h5py = dest.__module__.startswith("h5py.")

#         # copy array, provide creation options
#         compressor = Zlib(9)
#         create_kws = dict(chunks=(10,))
#         if dest_h5py:
#             create_kws.update(
#                 compression="gzip", compression_opts=9, shuffle=True, fletcher32=True, fillvalue=42
#             )
#         else:
#             create_kws.update(compressor=compressor, fill_value=42, order="F", filters=[Adler32()])
#         copy(source["foo/bar/baz"], dest, without_attrs=True, **create_kws)
#         check_copied_array(
#             source["foo/bar/baz"], dest["baz"], without_attrs=True, expect_props=create_kws
#         )

#     def test_copy_array_exists_array(self, source, dest):
#         # copy array, dest array in the way
#         dest.create_dataset("baz", shape=(10,))

#         # raise
#         with pytest.raises(CopyError):
#             # should raise by default
#             copy(source["foo/bar/baz"], dest)
#         assert (10,) == dest["baz"].shape
#         with pytest.raises(CopyError):
#             copy(source["foo/bar/baz"], dest, if_exists="raise")
#         assert (10,) == dest["baz"].shape

#         # skip
#         copy(source["foo/bar/baz"], dest, if_exists="skip")
#         assert (10,) == dest["baz"].shape

#         # replace
#         copy(source["foo/bar/baz"], dest, if_exists="replace")
#         check_copied_array(source["foo/bar/baz"], dest["baz"])

#         # invalid option
#         with pytest.raises(ValueError):
#             copy(source["foo/bar/baz"], dest, if_exists="foobar")

#     def test_copy_array_exists_group(self, source, dest):
#         # copy array, dest group in the way
#         dest.create_group("baz")

#         # raise
#         with pytest.raises(CopyError):
#             copy(source["foo/bar/baz"], dest)
#         assert not hasattr(dest["baz"], "shape")
#         with pytest.raises(CopyError):
#             copy(source["foo/bar/baz"], dest, if_exists="raise")
#         assert not hasattr(dest["baz"], "shape")

#         # skip
#         copy(source["foo/bar/baz"], dest, if_exists="skip")
#         assert not hasattr(dest["baz"], "shape")

#         # replace
#         copy(source["foo/bar/baz"], dest, if_exists="replace")
#         check_copied_array(source["foo/bar/baz"], dest["baz"])

#     def test_copy_array_skip_initialized(self, source, dest):
#         dest_h5py = dest.__module__.startswith("h5py.")

#         dest.create_dataset("baz", shape=(100,), chunks=(10,), dtype="i8")
#         assert not np.all(source["foo/bar/baz"][:] == dest["baz"][:])

#         if dest_h5py:
#             with pytest.raises(ValueError):
#                 # not available with copy to h5py
#                 copy(source["foo/bar/baz"], dest, if_exists="skip_initialized")

#         else:
#             # copy array, dest array exists but not yet initialized
#             copy(source["foo/bar/baz"], dest, if_exists="skip_initialized")
#             check_copied_array(source["foo/bar/baz"], dest["baz"])

#             # copy array, dest array exists and initialized, will be skipped
#             dest["baz"][:] = np.arange(100, 200)
#             copy(source["foo/bar/baz"], dest, if_exists="skip_initialized")
#             assert_array_equal(np.arange(100, 200), dest["baz"][:])
#             assert not np.all(source["foo/bar/baz"][:] == dest["baz"][:])

#     def test_copy_group(self, source, dest):
#         # copy group, default options
#         copy(source["foo"], dest)
#         check_copied_group(source["foo"], dest["foo"])

#     def test_copy_group_no_name(self, source, dest):
#         with pytest.raises(TypeError):
#             # need a name if copy root
#             copy(source, dest)

#         copy(source, dest, name="root")
#         check_copied_group(source, dest["root"])

#     def test_copy_group_options(self, source, dest):
#         # copy group, non-default options
#         copy(source["foo"], dest, name="qux", without_attrs=True)
#         assert "foo" not in dest
#         check_copied_group(source["foo"], dest["qux"], without_attrs=True)

#     def test_copy_group_shallow(self, source, dest):
#         # copy group, shallow
#         copy(source, dest, name="eggs", shallow=True)
#         check_copied_group(source, dest["eggs"], shallow=True)

#     def test_copy_group_exists_group(self, source, dest):
#         # copy group, dest groups exist
#         dest.create_group("foo/bar")
#         copy(source["foo"], dest)
#         check_copied_group(source["foo"], dest["foo"])

#     def test_copy_group_exists_array(self, source, dest):
#         # copy group, dest array in the way
#         dest.create_dataset("foo/bar", shape=(10,))

#         # raise
#         with pytest.raises(CopyError):
#             copy(source["foo"], dest)
#         assert dest["foo/bar"].shape == (10,)
#         with pytest.raises(CopyError):
#             copy(source["foo"], dest, if_exists="raise")
#         assert dest["foo/bar"].shape == (10,)

#         # skip
#         copy(source["foo"], dest, if_exists="skip")
#         assert dest["foo/bar"].shape == (10,)

#         # replace
#         copy(source["foo"], dest, if_exists="replace")
#         check_copied_group(source["foo"], dest["foo"])

#     def test_copy_group_dry_run(self, source, dest):
#         # dry run, empty destination
#         n_copied, n_skipped, n_bytes_copied = copy(
#             source["foo"], dest, dry_run=True, return_stats=True
#         )
#         assert 0 == len(dest)
#         assert 3 == n_copied
#         assert 0 == n_skipped
#         assert 0 == n_bytes_copied

#         # dry run, array exists in destination
#         baz = np.arange(100, 200)
#         dest.create_dataset("foo/bar/baz", data=baz)
#         assert not np.all(source["foo/bar/baz"][:] == dest["foo/bar/baz"][:])
#         assert 1 == len(dest)

#         # raise
#         with pytest.raises(CopyError):
#             copy(source["foo"], dest, dry_run=True)
#         assert 1 == len(dest)

#         # skip
#         n_copied, n_skipped, n_bytes_copied = copy(
#             source["foo"], dest, dry_run=True, if_exists="skip", return_stats=True
#         )
#         assert 1 == len(dest)
#         assert 2 == n_copied
#         assert 1 == n_skipped
#         assert 0 == n_bytes_copied
#         assert_array_equal(baz, dest["foo/bar/baz"])

#         # replace
#         n_copied, n_skipped, n_bytes_copied = copy(
#             source["foo"], dest, dry_run=True, if_exists="replace", return_stats=True
#         )
#         assert 1 == len(dest)
#         assert 3 == n_copied
#         assert 0 == n_skipped
#         assert 0 == n_bytes_copied
#         assert_array_equal(baz, dest["foo/bar/baz"])

#     def test_logging(self, source, dest, tmpdir):
#         # callable log
#         copy(source["foo"], dest, dry_run=True, log=print)

#         # file name
#         fn = str(tmpdir.join("log_name"))
#         copy(source["foo"], dest, dry_run=True, log=fn)

#         # file
#         with tmpdir.join("log_file").open(mode="w") as f:
#             copy(source["foo"], dest, dry_run=True, log=f)

#         # bad option
#         with pytest.raises(TypeError):
#             copy(source["foo"], dest, dry_run=True, log=True)


def test_open_falls_back_to_open_group() -> None:
    # https://github.com/zarr-developers/zarr-python/issues/2309
    store = MemoryStore()
    zarr.open_group(store, attributes={"key": "value"})

    group = zarr.open(store)
    assert isinstance(group, Group)
    assert group.attrs == {"key": "value"}


async def test_open_falls_back_to_open_group_async(zarr_format: ZarrFormat) -> None:
    # https://github.com/zarr-developers/zarr-python/issues/2309
    store = MemoryStore()
    await zarr.api.asynchronous.open_group(
        store, attributes={"key": "value"}, zarr_format=zarr_format
    )

    group = await zarr.api.asynchronous.open(store=store)
    assert isinstance(group, zarr.core.group.AsyncGroup)
    assert group.metadata.zarr_format == zarr_format
    assert group.attrs == {"key": "value"}


@pytest.mark.parametrize("mode", ["r", "r+", "w", "a"])
def test_open_modes_creates_group(tmp_path: Path, mode: str) -> None:
    # https://github.com/zarr-developers/zarr-python/issues/2490
    zarr_dir = tmp_path / f"mode-{mode}-test.zarr"
    if mode in ["r", "r+"]:
        # Expect FileNotFoundError to be raised if 'r' or 'r+' mode
        with pytest.raises(FileNotFoundError):
            zarr.open(store=zarr_dir, mode=mode)  # type: ignore[arg-type]
    else:
        group = zarr.open(store=zarr_dir, mode=mode)  # type: ignore[arg-type]
        assert isinstance(group, Group)


async def test_metadata_validation_error() -> None:
    with pytest.raises(
        MetadataValidationError,
        match="Invalid value for 'zarr_format'. Expected '2, 3, or None'. Got '3.0'.",
    ):
        await zarr.api.asynchronous.open_group(zarr_format="3.0")  # type: ignore [arg-type]

    with pytest.raises(
        MetadataValidationError,
        match="Invalid value for 'zarr_format'. Expected '2, 3, or None'. Got '3.0'.",
    ):
        await zarr.api.asynchronous.open_array(shape=(1,), zarr_format="3.0")  # type: ignore [arg-type]


@pytest.mark.parametrize(
    "store",
    ["local", "memory", "zip"],
    indirect=True,
)
def test_open_array_with_mode_r_plus(store: Store, zarr_format: ZarrFormat) -> None:
    # 'r+' means read/write (must exist)
    with pytest.raises(FileNotFoundError):
        zarr.open_array(store=store, mode="r+", zarr_format=zarr_format)
    zarr.ones(store=store, shape=(3, 3), zarr_format=zarr_format)
    z2 = zarr.open_array(store=store, mode="r+")
    assert isinstance(z2, Array)
    assert z2.metadata.zarr_format == zarr_format
    result = z2[:]
    assert isinstance(result, NDArrayLike)
    assert (result == 1).all()
    z2[:] = 3


@pytest.mark.parametrize(
    ("a_func", "b_func"),
    [
        (zarr.api.asynchronous.create_array, zarr.api.synchronous.create_array),
        (zarr.api.asynchronous.save, zarr.api.synchronous.save),
        (zarr.api.asynchronous.save_array, zarr.api.synchronous.save_array),
        (zarr.api.asynchronous.save_group, zarr.api.synchronous.save_group),
        (zarr.api.asynchronous.open_group, zarr.api.synchronous.open_group),
        (zarr.api.asynchronous.create, zarr.api.synchronous.create),
    ],
)
def test_consistent_signatures(
    a_func: Callable[[object], object], b_func: Callable[[object], object]
) -> None:
    """
    Ensure that pairs of functions have the same signature
    """
    base_sig = inspect.signature(a_func)
    test_sig = inspect.signature(b_func)
    wrong: dict[str, list[object]] = {
        "missing_from_test": [],
        "missing_from_base": [],
        "wrong_type": [],
    }
    for key, value in base_sig.parameters.items():
        if key not in test_sig.parameters:
            wrong["missing_from_test"].append((key, value))
    for key, value in test_sig.parameters.items():
        if key not in base_sig.parameters:
            wrong["missing_from_base"].append((key, value))
        if base_sig.parameters[key] != value:
            wrong["wrong_type"].append({key: {"test": value, "base": base_sig.parameters[key]}})
    assert wrong["missing_from_base"] == []
    assert wrong["missing_from_test"] == []
    assert wrong["wrong_type"] == []


def test_api_exports() -> None:
    """
    Test that the sync API and the async API export the same objects
    """
    assert zarr.api.asynchronous.__all__ == zarr.api.synchronous.__all__


@gpu_test
@pytest.mark.parametrize(
    "store",
    ["local", "memory", "zip"],
    indirect=True,
)
@pytest.mark.parametrize("zarr_format", [None, 2, 3])
def test_gpu_basic(store: Store, zarr_format: ZarrFormat | None) -> None:
    import cupy as cp

    if zarr_format == 2:
        # Without this, the zstd codec attempts to convert the cupy
        # array to bytes.
        compressors = None
    else:
        compressors = "auto"

    with zarr.config.enable_gpu():
        src = cp.random.uniform(size=(100, 100))  # allocate on the device
        z = zarr.create_array(
            store,
            name="a",
            shape=src.shape,
            chunks=(10, 10),
            dtype=src.dtype,
            overwrite=True,
            zarr_format=zarr_format,
            compressors=compressors,
        )
        z[:10, :10] = src[:10, :10]

        result = z[:10, :10]
        # assert_array_equal doesn't check the type
        assert isinstance(result, type(src))
        cp.testing.assert_array_equal(result, src[:10, :10])


def test_v2_without_compressor() -> None:
    # Make sure it's possible to set no compressor for v2 arrays
    arr = zarr.create(store={}, shape=(1), dtype="uint8", zarr_format=2, compressor=None)
    assert arr.compressors == ()


def test_v2_with_v3_compressor() -> None:
    # Check trying to create a v2 array with a v3 compressor fails
    with pytest.raises(
        ValueError,
        match="Cannot use a BytesBytesCodec as a compressor for zarr v2 arrays. Use a numcodecs codec directly instead.",
    ):
        zarr.create(
            store={}, shape=(1), dtype="uint8", zarr_format=2, compressor=zarr.codecs.BloscCodec()
        )


def add_empty_file(path: Path) -> Path:
    fpath = path / "a.txt"
    fpath.touch()
    return fpath


@pytest.mark.parametrize("create_function", [create_array, from_array])
@pytest.mark.parametrize("overwrite", [True, False])
def test_no_overwrite_array(tmp_path: Path, create_function: Callable, overwrite: bool) -> None:  # type:ignore[type-arg]
    store = zarr.storage.LocalStore(tmp_path)
    existing_fpath = add_empty_file(tmp_path)

    assert existing_fpath.exists()
    create_function(store=store, data=np.ones(shape=(1,)), overwrite=overwrite)
    if overwrite:
        assert not existing_fpath.exists()
    else:
        assert existing_fpath.exists()


@pytest.mark.parametrize("create_function", [create_group, group])
@pytest.mark.parametrize("overwrite", [True, False])
def test_no_overwrite_group(tmp_path: Path, create_function: Callable, overwrite: bool) -> None:  # type:ignore[type-arg]
    store = zarr.storage.LocalStore(tmp_path)
    existing_fpath = add_empty_file(tmp_path)

    assert existing_fpath.exists()
    create_function(store=store, overwrite=overwrite)
    if overwrite:
        assert not existing_fpath.exists()
    else:
        assert existing_fpath.exists()


@pytest.mark.parametrize("open_func", [zarr.open, open_group])
@pytest.mark.parametrize("mode", ["r", "r+", "a", "w", "w-"])
def test_no_overwrite_open(tmp_path: Path, open_func: Callable, mode: str) -> None:  # type:ignore[type-arg]
    store = zarr.storage.LocalStore(tmp_path)
    existing_fpath = add_empty_file(tmp_path)

    assert existing_fpath.exists()
    with contextlib.suppress(FileExistsError, FileNotFoundError, UserWarning):
        open_func(store=store, mode=mode)
    if mode == "w":
        assert not existing_fpath.exists()
    else:
        assert existing_fpath.exists()


def test_no_overwrite_load(tmp_path: Path) -> None:
    store = zarr.storage.LocalStore(tmp_path)
    existing_fpath = add_empty_file(tmp_path)

    assert existing_fpath.exists()
    with contextlib.suppress(NotImplementedError):
        zarr.load(store)
    assert existing_fpath.exists()


@pytest.mark.parametrize(
    "f",
    [
        zarr.array,
        zarr.create,
        zarr.create_array,
        zarr.ones,
        zarr.ones_like,
        zarr.empty,
        zarr.empty_like,
        zarr.full,
        zarr.full_like,
        zarr.zeros,
        zarr.zeros_like,
    ],
)
def test_auto_chunks(f: Callable[..., Array]) -> None:
    # Make sure chunks are set automatically across the public API
    # TODO: test shards with this test too
    shape = (1000, 1000)
    dtype = np.uint8
    kwargs = {"shape": shape, "dtype": dtype}
    array = np.zeros(shape, dtype=dtype)
    store = zarr.storage.MemoryStore()

    if f in [zarr.full, zarr.full_like]:
        kwargs["fill_value"] = 0
    if f in [zarr.array]:
        kwargs["data"] = array
    if f in [zarr.empty_like, zarr.full_like, zarr.empty_like, zarr.ones_like, zarr.zeros_like]:
        kwargs["a"] = array
    if f in [zarr.create_array]:
        kwargs["store"] = store

    a = f(**kwargs)
    assert a.chunks == (500, 500)


@pytest.mark.parametrize("kwarg_name", ["synchronizer", "chunk_store", "cache_attrs", "meta_array"])
def test_unimplemented_kwarg_warnings(kwarg_name: str) -> None:
    kwargs = {kwarg_name: 1}
    with pytest.warns(RuntimeWarning, match=".* is not yet implemented"):
        zarr.create(shape=(1,), **kwargs)  # type: ignore[arg-type]
