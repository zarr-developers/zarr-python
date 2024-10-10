import pathlib
import warnings

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import zarr
import zarr.api.asynchronous
import zarr.core.group
from zarr import Array, Group
from zarr.abc.store import Store
from zarr.api.synchronous import create, group, load, open, open_group, save, save_array, save_group
from zarr.core.common import ZarrFormat
from zarr.errors import MetadataValidationError
from zarr.storage.memory import MemoryStore


def test_create_array(memory_store: Store) -> None:
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


async def test_open_array(memory_store: MemoryStore) -> None:
    store = memory_store

    # open array, create if doesn't exist
    z = open(store=store, shape=100)
    assert isinstance(z, Array)
    assert z.shape == (100,)

    # open array, overwrite
    # store._store_dict = {}
    store = MemoryStore(mode="w")
    z = open(store=store, shape=200)
    assert isinstance(z, Array)
    assert z.shape == (200,)

    # open array, read-only
    store_cls = type(store)
    ro_store = await store_cls.open(store_dict=store._store_dict, mode="r")
    z = open(store=ro_store)
    assert isinstance(z, Array)
    assert z.shape == (200,)
    assert z.read_only

    # path not found
    with pytest.raises(FileNotFoundError):
        open(store="doesnotexist", mode="r")


async def test_open_group(memory_store: MemoryStore) -> None:
    store = memory_store

    # open group, create if doesn't exist
    g = open_group(store=store)
    g.create_group("foo")
    assert isinstance(g, Group)
    assert "foo" in g

    # open group, overwrite
    # g = open_group(store=store)
    # assert isinstance(g, Group)
    # assert "foo" not in g

    # open group, read-only
    store_cls = type(store)
    ro_store = await store_cls.open(store_dict=store._store_dict, mode="r")
    g = open_group(store=ro_store)
    assert isinstance(g, Group)
    # assert g.read_only


@pytest.mark.parametrize("zarr_format", [None, 2, 3])
async def test_open_group_unspecified_version(
    tmpdir: pathlib.Path, zarr_format: ZarrFormat
) -> None:
    """regression test for https://github.com/zarr-developers/zarr-python/issues/2175"""

    # create a group with specified zarr format (could be 2, 3, or None)
    _ = await zarr.api.asynchronous.open_group(
        store=str(tmpdir), mode="w", zarr_format=zarr_format, attributes={"foo": "bar"}
    )

    # now open that group without specifying the format
    g2 = await zarr.api.asynchronous.open_group(store=str(tmpdir), mode="r")

    assert g2.attrs == {"foo": "bar"}

    if zarr_format is not None:
        assert g2.metadata.zarr_format == zarr_format


def test_save_errors() -> None:
    with pytest.raises(ValueError):
        # no arrays provided
        save_group("data/group.zarr")
    with pytest.raises(TypeError):
        # no array provided
        save_array("data/group.zarr")
    with pytest.raises(ValueError):
        # no arrays provided
        save("data/group.zarr")


def test_open_with_mode_r(tmp_path: pathlib.Path) -> None:
    # 'r' means read only (must exist)
    with pytest.raises(FileNotFoundError):
        zarr.open(store=tmp_path, mode="r")
    z1 = zarr.ones(store=tmp_path, shape=(3, 3))
    assert z1.fill_value == 1
    z2 = zarr.open(store=tmp_path, mode="r")
    assert isinstance(z2, Array)
    assert z2.fill_value == 1
    assert (z2[:] == 1).all()
    with pytest.raises(ValueError):
        z2[:] = 3


def test_open_with_mode_r_plus(tmp_path: pathlib.Path) -> None:
    # 'r+' means read/write (must exist)
    with pytest.raises(FileNotFoundError):
        zarr.open(store=tmp_path, mode="r+")
    zarr.ones(store=tmp_path, shape=(3, 3))
    z2 = zarr.open(store=tmp_path, mode="r+")
    assert isinstance(z2, Array)
    assert (z2[:] == 1).all()
    z2[:] = 3


async def test_open_with_mode_a(tmp_path: pathlib.Path) -> None:
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
    assert (z2[:] == 1).all()
    z2[:] = 3


def test_open_with_mode_w(tmp_path: pathlib.Path) -> None:
    # 'w' means create (overwrite if exists);
    arr = zarr.open(store=tmp_path, mode="w", shape=(3, 3))
    assert isinstance(arr, Array)

    arr[...] = 3
    z2 = zarr.open(store=tmp_path, mode="w", shape=(3, 3))
    assert isinstance(z2, Array)
    assert not (z2[:] == 3).all()
    z2[:] = 3


def test_open_with_mode_w_minus(tmp_path: pathlib.Path) -> None:
    # 'w-' means create  (fail if exists)
    arr = zarr.open(store=tmp_path, mode="w-", shape=(3, 3))
    assert isinstance(arr, Array)
    arr[...] = 1
    with pytest.raises(FileExistsError):
        zarr.open(store=tmp_path, mode="w-")


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


def test_load_array(memory_store: Store) -> None:
    store = memory_store
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


def test_tree() -> None:
    g1 = zarr.group()
    g1.create_group("foo")
    g3 = g1.create_group("bar")
    g3.create_group("baz")
    g5 = g3.create_group("qux")
    g5.create_array("baz", shape=100, chunks=10)
    # TODO: complete after tree has been reimplemented
    # assert repr(zarr.tree(g1)) == repr(g1.tree())
    # assert str(zarr.tree(g1)) == str(g1.tree())


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


def test_open_positional_args_deprecated() -> None:
    store = MemoryStore({}, mode="w")
    with pytest.warns(FutureWarning, match="pass"):
        open(store, "w", shape=(1,))


def test_save_array_positional_args_deprecated() -> None:
    store = MemoryStore({}, mode="w")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="zarr_version is deprecated", category=DeprecationWarning
        )
        with pytest.warns(FutureWarning, match="pass"):
            save_array(
                store,
                np.ones(
                    1,
                ),
                3,
            )


def test_group_positional_args_deprecated() -> None:
    store = MemoryStore({}, mode="w")
    with pytest.warns(FutureWarning, match="pass"):
        group(store, True)


def test_open_group_positional_args_deprecated() -> None:
    store = MemoryStore({}, mode="w")
    with pytest.warns(FutureWarning, match="pass"):
        open_group(store, "w")


def test_open_falls_back_to_open_group() -> None:
    # https://github.com/zarr-developers/zarr-python/issues/2309
    store = MemoryStore(mode="w")
    zarr.open_group(store, attributes={"key": "value"})

    group = zarr.open(store)
    assert isinstance(group, Group)
    assert group.attrs == {"key": "value"}


async def test_open_falls_back_to_open_group_async() -> None:
    # https://github.com/zarr-developers/zarr-python/issues/2309
    store = MemoryStore(mode="w")
    await zarr.api.asynchronous.open_group(store, attributes={"key": "value"})

    group = await zarr.api.asynchronous.open(store=store)
    assert isinstance(group, zarr.core.group.AsyncGroup)
    assert group.attrs == {"key": "value"}


async def test_metadata_validation_error() -> None:
    with pytest.raises(
        MetadataValidationError,
        match="Invalid value for 'zarr_format'. Expected '2, 3, or None'. Got '3.0'.",
    ):
        await zarr.api.asynchronous.open_group(zarr_format="3.0")  # type: ignore[arg-type]

    with pytest.raises(
        MetadataValidationError,
        match="Invalid value for 'zarr_format'. Expected '2, 3, or None'. Got '3.0'.",
    ):
        await zarr.api.asynchronous.open_array(shape=(1,), zarr_format="3.0")  # type: ignore[arg-type]
