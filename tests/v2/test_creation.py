import atexit
import os.path
import shutil
import warnings

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from zarr.v2.codecs import Zlib
from zarr.v2.core import Array
from zarr.v2.creation import (
    array,
    create,
    empty,
    empty_like,
    full,
    full_like,
    ones,
    ones_like,
    open_array,
    open_like,
    zeros,
    zeros_like,
)
from zarr.v2.hierarchy import open_group
from zarr.v2.n5 import N5Store
from zarr.v2.storage import DirectoryStore, KVStore
from zarr.v2.sync import ThreadSynchronizer
from .util import mktemp, have_fsspec


_VERSIONS = (None, 2)
_VERSIONS2 = (2,)


# something bcolz-like
class MockBcolzArray:
    def __init__(self, data, chunklen):
        self.data = data
        self.chunklen = chunklen

    def __getattr__(self, item):
        return getattr(self.data, item)

    def __getitem__(self, item):
        return self.data[item]


# something h5py-like
class MockH5pyDataset:
    def __init__(self, data, chunks):
        self.data = data
        self.chunks = chunks

    def __getattr__(self, item):
        return getattr(self.data, item)

    def __getitem__(self, item):
        return self.data[item]


def _init_creation_kwargs(at_root=True):
    kwargs = {}
    if not at_root:
        kwargs["path"] = "array"
    return kwargs


@pytest.mark.parametrize("at_root", [False, True])
def test_array(at_root):
    kwargs = _init_creation_kwargs(at_root)

    # with numpy array
    a = np.arange(100)
    z = array(a, chunks=10, **kwargs)
    assert a.shape == z.shape
    assert a.dtype == z.dtype
    assert_array_equal(a, z[:])

    # with array-like
    a = list(range(100))
    z = array(a, chunks=10, **kwargs)
    assert (100,) == z.shape
    assert np.asarray(a).dtype == z.dtype
    assert_array_equal(np.asarray(a), z[:])

    # with another zarr array
    z2 = array(z, **kwargs)
    assert z.shape == z2.shape
    assert z.chunks == z2.chunks
    assert z.dtype == z2.dtype
    assert_array_equal(z[:], z2[:])

    # with chunky array-likes

    b = np.arange(1000).reshape(100, 10)
    c = MockBcolzArray(b, 10)
    z3 = array(c, **kwargs)
    assert c.shape == z3.shape
    assert (10, 10) == z3.chunks

    b = np.arange(1000).reshape(100, 10)
    c = MockH5pyDataset(b, chunks=(10, 2))
    z4 = array(c, **kwargs)
    assert c.shape == z4.shape
    assert (10, 2) == z4.chunks

    c = MockH5pyDataset(b, chunks=None)
    z5 = array(c, **kwargs)
    assert c.shape == z5.shape
    assert isinstance(z5.chunks, tuple)

    # with dtype=None
    a = np.arange(100, dtype="i4")
    z = array(a, dtype=None, **kwargs)
    assert_array_equal(a[:], z[:])
    assert a.dtype == z.dtype

    # with dtype=something else
    a = np.arange(100, dtype="i4")
    z = array(a, dtype="i8", **kwargs)
    assert_array_equal(a[:], z[:])
    assert np.dtype("i8") == z.dtype


@pytest.mark.parametrize("at_root", [False, True])
def test_empty(at_root):
    kwargs = _init_creation_kwargs(at_root)
    z = empty(100, chunks=10, **kwargs)
    assert (100,) == z.shape
    assert (10,) == z.chunks


@pytest.mark.parametrize("at_root", [False, True])
def test_zeros(at_root):
    kwargs = _init_creation_kwargs(at_root)
    z = zeros(100, chunks=10, **kwargs)
    assert (100,) == z.shape
    assert (10,) == z.chunks
    assert_array_equal(np.zeros(100), z[:])


@pytest.mark.parametrize("at_root", [False, True])
def test_ones(at_root):
    kwargs = _init_creation_kwargs(at_root)
    z = ones(100, chunks=10, **kwargs)
    assert (100,) == z.shape
    assert (10,) == z.chunks
    assert_array_equal(np.ones(100), z[:])


@pytest.mark.parametrize("at_root", [False, True])
def test_full(at_root):
    kwargs = _init_creation_kwargs(at_root)
    z = full(100, chunks=10, fill_value=42, dtype="i4", **kwargs)
    assert (100,) == z.shape
    assert (10,) == z.chunks
    assert_array_equal(np.full(100, fill_value=42, dtype="i4"), z[:])

    # nan
    z = full(100, chunks=10, fill_value=np.nan, dtype="f8", **kwargs)
    assert np.all(np.isnan(z[:]))


def test_full_additional_dtypes():
    """Test additional types that aren't part of the base v3 spec."""
    kwargs = _init_creation_kwargs()
    # NaT
    z = full(100, chunks=10, fill_value="NaT", dtype="M8[s]", **kwargs)
    assert np.all(np.isnat(z[:]))
    z = full(100, chunks=10, fill_value="NaT", dtype="m8[s]", **kwargs)
    assert np.all(np.isnat(z[:]))

    # byte string dtype
    v = b"xxx"
    z = full(100, chunks=10, fill_value=v, dtype="S3", **kwargs)
    assert v == z[0]
    a = z[...]
    assert z.dtype == a.dtype
    assert v == a[0]
    assert np.all(a == v)

    # unicode string dtype
    v = "xxx"
    z = full(100, chunks=10, fill_value=v, dtype="U3", **kwargs)
    assert v == z[0]
    a = z[...]
    assert z.dtype == a.dtype
    assert v == a[0]
    assert np.all(a == v)

    # bytes fill value / unicode dtype
    v = b"xxx"
    with pytest.raises(ValueError):
        full(100, chunks=10, fill_value=v, dtype="U3")


@pytest.mark.parametrize("dimension_separator", [".", "/", None])
@pytest.mark.parametrize("at_root", [False, True])
def test_open_array(at_root, dimension_separator):
    store = "data/array.zarr"
    kwargs = _init_creation_kwargs(at_root)

    # mode == 'w'
    z = open_array(
        store, mode="w", shape=100, chunks=10, dimension_separator=dimension_separator, **kwargs
    )
    z[:] = 42
    assert isinstance(z, Array)

    assert isinstance(z.store, DirectoryStore)
    assert (100,) == z.shape
    assert (10,) == z.chunks
    assert_array_equal(np.full(100, fill_value=42), z[:])

    if dimension_separator is None:
        assert z._dimension_separator == "."
    else:
        assert z._dimension_separator == dimension_separator

    # mode in 'r', 'r+'
    group_kwargs = kwargs.copy()
    open_group("data/group.zarr", mode="w", **group_kwargs)
    for mode in "r", "r+":
        with pytest.raises(ValueError):
            open_array("doesnotexist", mode=mode)
        with pytest.raises(ValueError):
            open_array("data/group.zarr", mode=mode)
    z = open_array(store, mode="r", **kwargs)
    assert isinstance(z, Array)
    assert isinstance(z.store, DirectoryStore)
    assert (100,) == z.shape
    assert (10,) == z.chunks
    assert_array_equal(np.full(100, fill_value=42), z[:])
    with pytest.raises(PermissionError):
        z[:] = 43
    z = open_array(store, mode="r+", **kwargs)
    assert isinstance(z, Array)
    assert isinstance(z.store, DirectoryStore)
    assert (100,) == z.shape
    assert (10,) == z.chunks
    assert_array_equal(np.full(100, fill_value=42), z[:])
    z[:] = 43
    assert_array_equal(np.full(100, fill_value=43), z[:])

    # mode == 'a'
    shutil.rmtree(store)
    z = open_array(store, mode="a", shape=100, chunks=10, **kwargs)
    z[:] = 42
    assert isinstance(z, Array)
    assert isinstance(z.store, DirectoryStore)
    assert (100,) == z.shape
    assert (10,) == z.chunks
    assert_array_equal(np.full(100, fill_value=42), z[:])

    with pytest.raises(ValueError):
        open_array("data/group.zarr", mode="a", **kwargs)

    # mode in 'w-', 'x'
    for mode in "w-", "x":
        shutil.rmtree(store)
        z = open_array(store, mode=mode, shape=100, chunks=10, **kwargs)
        z[:] = 42
        assert isinstance(z, Array)
        assert isinstance(z.store, DirectoryStore)
        assert (100,) == z.shape
        assert (10,) == z.chunks
        assert_array_equal(np.full(100, fill_value=42), z[:])
        with pytest.raises(ValueError):
            open_array(store, mode=mode, **kwargs)

        with pytest.raises(ValueError):
            open_array("data/group.zarr", mode=mode, **kwargs)

    # with synchronizer
    z = open_array(store, synchronizer=ThreadSynchronizer(), **kwargs)
    assert isinstance(z, Array)

    # with path
    kwargs_no_path = kwargs.copy()
    kwargs_no_path.pop("path", None)
    z = open_array(store, shape=100, path="foo/bar", mode="w", **kwargs_no_path)
    assert isinstance(z, Array)
    assert "foo/bar" == z.path

    # with chunk store
    meta_store = "data/meta.zarr"
    chunk_store = "data/chunks.zarr"
    z = open_array(store=meta_store, chunk_store=chunk_store, shape=11, mode="w", **kwargs)
    z[:] = 42
    assert os.path.abspath(meta_store) == z.store.path
    assert os.path.abspath(chunk_store) == z.chunk_store.path


def test_open_array_none():
    # open with store = None
    z = open_array(mode="w", shape=100, chunks=10)
    assert isinstance(z, Array)


@pytest.mark.parametrize("dimension_separator", [".", "/", None])
def test_open_array_infer_separator_from_store(dimension_separator):
    StoreClass = DirectoryStore
    path = None
    store = StoreClass("data/array.zarr", dimension_separator=dimension_separator)

    # Note: no dimension_separator kwarg to open_array
    #       we are testing here that it gets inferred from store
    z = open_array(store, path=path, mode="w", shape=100, chunks=10)
    z[:] = 42
    assert isinstance(z, Array)
    assert isinstance(z.store, DirectoryStore)
    assert (100,) == z.shape
    assert (10,) == z.chunks
    assert_array_equal(np.full(100, fill_value=42), z[:])

    if dimension_separator is None:
        assert z._dimension_separator == "."
    else:
        assert z._dimension_separator == dimension_separator


def test_open_array_n5():
    store = "data/array.zarr"
    kwargs = _init_creation_kwargs()

    # for N5 store
    store = "data/array.n5"
    z = open_array(store, mode="w", shape=100, chunks=10, **kwargs)
    z[:] = 42
    assert isinstance(z, Array)
    assert isinstance(z.store, N5Store)
    assert (100,) == z.shape
    assert (10,) == z.chunks
    assert_array_equal(np.full(100, fill_value=42), z[:])

    store = "data/group.n5"
    group_kwargs = kwargs.copy()
    z = open_group(store, mode="w", **group_kwargs)
    i = z.create_group("inner")
    a = i.zeros("array", shape=100, chunks=10)
    a[:] = 42

    # Edit inner/attributes.json to not include "n5"
    with open("data/group.n5/inner/attributes.json", "w") as o:
        o.write("{}")

    # Re-open
    a = open_group(store, **group_kwargs)["inner"]["array"]
    assert isinstance(a, Array)
    assert isinstance(z.store, N5Store)
    assert (100,) == a.shape
    assert (10,) == a.chunks
    assert_array_equal(np.full(100, fill_value=42), a[:])


@pytest.mark.parametrize("at_root", [False, True])
def test_open_array_dict_store(at_root):
    # dict will become a KVStore
    store = dict()
    kwargs = _init_creation_kwargs(at_root)
    expected_store_type = KVStore

    # mode == 'w'
    z = open_array(store, mode="w", shape=100, chunks=10, **kwargs)
    z[:] = 42
    assert isinstance(z, Array)
    assert isinstance(z.store, expected_store_type)
    assert (100,) == z.shape
    assert (10,) == z.chunks
    assert_array_equal(np.full(100, fill_value=42), z[:])


@pytest.mark.parametrize("at_root", [False, True])
def test_create_in_dict(at_root):
    kwargs = _init_creation_kwargs(at_root)
    expected_store_type = KVStore

    for func in [empty, zeros, ones]:
        a = func(100, store=dict(), **kwargs)
        assert isinstance(a.store, expected_store_type)

    a = full(100, 5, store=dict(), **kwargs)
    assert isinstance(a.store, expected_store_type)


@pytest.mark.skipif(have_fsspec is False, reason="needs fsspec")
@pytest.mark.parametrize("at_root", [False, True])
def test_create_writeable_mode(at_root, tmp_path):
    # Regression test for https://github.com/zarr-developers/zarr-python/issues/1306
    import fsspec

    kwargs = _init_creation_kwargs(at_root)
    store = fsspec.get_mapper(str(tmp_path))
    z = create(100, store=store, **kwargs)
    assert z.store.map == store


@pytest.mark.parametrize("at_root", [False, True])
def test_empty_like(at_root):
    kwargs = _init_creation_kwargs(at_root)

    # zarr array
    z = empty(100, chunks=10, dtype="f4", compressor=Zlib(5), order="F", **kwargs)
    z2 = empty_like(z, path=kwargs.get("path"))
    assert z.shape == z2.shape
    assert z.chunks == z2.chunks
    assert z.dtype == z2.dtype
    assert z.compressor.get_config() == z2.compressor.get_config()
    assert z.fill_value == z2.fill_value
    assert z.order == z2.order

    # numpy array
    a = np.empty(100, dtype="f4")
    z3 = empty_like(a, **kwargs)
    assert a.shape == z3.shape
    assert (100,) == z3.chunks
    assert a.dtype == z3.dtype
    assert z3.fill_value is None

    # something slightly silly
    a = [0] * 100
    z3 = empty_like(a, shape=200, **kwargs)
    assert (200,) == z3.shape

    # other array-likes
    b = np.arange(1000).reshape(100, 10)
    c = MockBcolzArray(b, 10)
    z = empty_like(c, **kwargs)
    assert b.shape == z.shape
    assert (10, 10) == z.chunks
    c = MockH5pyDataset(b, chunks=(10, 2))
    z = empty_like(c, **kwargs)
    assert b.shape == z.shape
    assert (10, 2) == z.chunks
    c = MockH5pyDataset(b, chunks=None)
    z = empty_like(c, **kwargs)
    assert b.shape == z.shape
    assert isinstance(z.chunks, tuple)


@pytest.mark.parametrize("at_root", [False, True])
def test_zeros_like(at_root):
    kwargs = _init_creation_kwargs(at_root)

    # zarr array
    z = zeros(100, chunks=10, dtype="f4", compressor=Zlib(5), order="F", **kwargs)
    z2 = zeros_like(z, path=kwargs.get("path"))
    assert z.shape == z2.shape
    assert z.chunks == z2.chunks
    assert z.dtype == z2.dtype
    assert z.compressor.get_config() == z2.compressor.get_config()
    assert z.fill_value == z2.fill_value
    assert z.order == z2.order

    # numpy array
    a = np.empty(100, dtype="f4")
    z3 = zeros_like(a, chunks=10, **kwargs)
    assert a.shape == z3.shape
    assert (10,) == z3.chunks
    assert a.dtype == z3.dtype
    assert 0 == z3.fill_value


@pytest.mark.parametrize("at_root", [False, True])
def test_ones_like(at_root):
    kwargs = _init_creation_kwargs(at_root)

    # zarr array
    z = ones(100, chunks=10, dtype="f4", compressor=Zlib(5), order="F", **kwargs)
    z2 = ones_like(z, path=kwargs.get("path"))
    assert z.shape == z2.shape
    assert z.chunks == z2.chunks
    assert z.dtype == z2.dtype
    assert z.compressor.get_config() == z2.compressor.get_config()
    assert z.fill_value == z2.fill_value
    assert z.order == z2.order

    # numpy array
    a = np.empty(100, dtype="f4")
    z3 = ones_like(a, chunks=10, **kwargs)
    assert a.shape == z3.shape
    assert (10,) == z3.chunks
    assert a.dtype == z3.dtype
    assert 1 == z3.fill_value


@pytest.mark.parametrize("at_root", [False, True])
def test_full_like(at_root):
    kwargs = _init_creation_kwargs(at_root)

    z = full(100, chunks=10, dtype="f4", compressor=Zlib(5), fill_value=42, order="F", **kwargs)
    z2 = full_like(z, path=kwargs.get("path"))
    assert z.shape == z2.shape
    assert z.chunks == z2.chunks
    assert z.dtype == z2.dtype
    assert z.compressor.get_config() == z2.compressor.get_config()
    assert z.fill_value == z2.fill_value
    assert z.order == z2.order

    # numpy array
    a = np.empty(100, dtype="f4")
    z3 = full_like(a, chunks=10, fill_value=42, **kwargs)
    assert a.shape == z3.shape
    assert (10,) == z3.chunks
    assert a.dtype == z3.dtype
    assert 42 == z3.fill_value

    with pytest.raises(TypeError):
        # fill_value missing
        full_like(a, chunks=10, **kwargs)


@pytest.mark.parametrize("at_root", [False, True])
def test_open_like(at_root):
    kwargs = _init_creation_kwargs(at_root)

    # zarr array
    path = mktemp()
    atexit.register(shutil.rmtree, path)
    z = full(100, chunks=10, dtype="f4", compressor=Zlib(5), fill_value=42, order="F", **kwargs)
    z2 = open_like(z, path)
    assert z.shape == z2.shape
    assert z.chunks == z2.chunks
    assert z.dtype == z2.dtype
    assert z.compressor.get_config() == z2.compressor.get_config()
    assert z.fill_value == z2.fill_value
    assert z.order == z2.order

    # numpy array
    path = mktemp()
    atexit.register(shutil.rmtree, path)
    a = np.empty(100, dtype="f4")
    z3 = open_like(a, path, chunks=10)
    assert a.shape == z3.shape
    assert (10,) == z3.chunks
    assert a.dtype == z3.dtype
    assert 0 == z3.fill_value


@pytest.mark.parametrize("at_root", [False, True])
def test_create(at_root):
    kwargs = _init_creation_kwargs(at_root)

    # defaults
    z = create(100, **kwargs)
    assert isinstance(z, Array)
    assert (100,) == z.shape
    assert (100,) == z.chunks  # auto-chunks
    assert np.dtype(None) == z.dtype
    assert "blosc" == z.compressor.codec_id
    assert 0 == z.fill_value

    # all specified
    z = create(100, chunks=10, dtype="i4", compressor=Zlib(1), fill_value=42, order="F", **kwargs)
    assert isinstance(z, Array)
    assert (100,) == z.shape
    assert (10,) == z.chunks
    assert np.dtype("i4") == z.dtype
    assert "zlib" == z.compressor.codec_id
    assert 1 == z.compressor.level
    assert 42 == z.fill_value
    assert "F" == z.order

    # with synchronizer
    synchronizer = ThreadSynchronizer()
    z = create(100, chunks=10, synchronizer=synchronizer, **kwargs)
    assert isinstance(z, Array)
    assert (100,) == z.shape
    assert (10,) == z.chunks
    assert synchronizer is z.synchronizer

    # don't allow string as compressor arg
    with pytest.raises(ValueError):
        create(100, chunks=10, compressor="zlib", **kwargs)

    # h5py compatibility

    z = create(100, compression="zlib", compression_opts=9, **kwargs)
    assert "zlib" == z.compressor.codec_id
    assert 9 == z.compressor.level

    z = create(100, compression="default", **kwargs)
    assert "blosc" == z.compressor.codec_id

    # errors
    with pytest.raises(ValueError):
        # bad compression argument
        create(100, compression=1, **kwargs)
    with pytest.raises(ValueError):
        # bad fill value
        create(100, dtype="i4", fill_value="foo", **kwargs)

    # auto chunks
    z = create(1000000000, chunks=True, **kwargs)
    assert z.chunks[0] < z.shape[0]
    z = create(1000000000, chunks=None, **kwargs)  # backwards-compatibility
    assert z.chunks[0] < z.shape[0]
    # no chunks
    z = create(1000000000, chunks=False, **kwargs)
    assert z.chunks == z.shape


def test_compression_args():
    kwargs = _init_creation_kwargs()

    with warnings.catch_warnings():
        warnings.simplefilter("default")
        z = create(100, compression="zlib", compression_opts=9, **kwargs)
        assert isinstance(z, Array)
        assert "zlib" == z.compressor.codec_id
        assert 9 == z.compressor.level

        # 'compressor' overrides 'compression'
        with pytest.warns(UserWarning):
            z = create(100, compressor=Zlib(9), compression="bz2", compression_opts=1, **kwargs)
        assert isinstance(z, Array)
        assert "zlib" == z.compressor.codec_id
        assert 9 == z.compressor.level

        # 'compressor' ignores 'compression_opts'
        with pytest.warns(UserWarning):
            z = create(100, compressor=Zlib(9), compression_opts=1, **kwargs)
        assert isinstance(z, Array)
        assert "zlib" == z.compressor.codec_id
        assert 9 == z.compressor.level

        with pytest.warns(UserWarning):
            # 'compressor' overrides 'compression'
            create(100, compressor=Zlib(9), compression="bz2", compression_opts=1, **kwargs)
        with pytest.warns(UserWarning):
            # 'compressor' ignores 'compression_opts'
            create(100, compressor=Zlib(9), compression_opts=1, **kwargs)


@pytest.mark.parametrize("at_root", [False, True])
def test_create_read_only(at_root):
    # https://github.com/alimanfoo/zarr/issues/151

    kwargs = _init_creation_kwargs(at_root)

    # create an array initially read-only, then enable writing
    z = create(100, read_only=True, **kwargs)
    assert z.read_only
    with pytest.raises(PermissionError):
        z[:] = 42
    z.read_only = False
    z[:] = 42
    assert np.all(z[...] == 42)
    z.read_only = True
    with pytest.raises(PermissionError):
        z[:] = 0

    # this is subtly different, but here we want to create an array with data, and then
    # have it be read-only
    a = np.arange(100)
    z = array(a, read_only=True, **kwargs)
    assert_array_equal(a, z[...])
    assert z.read_only
    with pytest.raises(PermissionError):
        z[:] = 42


def test_json_dumps_chunks_numpy_dtype():
    z = zeros((10,), chunks=(np.int64(2),))
    assert np.all(z[...] == 0)


@pytest.mark.parametrize(
    ("init_shape", "init_chunks", "shape", "chunks"),
    (
        ((1,), (1,), (1,), (1,)),
        ((1.0,), (1.0,), (1,), (1,)),
        ((1.0,), False, (1,), (1,)),
        ((1.0,), True, (1,), (1,)),
        ((1.0,), None, (1,), (1,)),
    ),
)
def test_shape_chunk_ints(init_shape, init_chunks, shape, chunks):
    g = open_group()
    array = g.create_dataset("ds", shape=init_shape, chunks=init_chunks, dtype=np.uint8)

    assert all(
        isinstance(s, int) for s in array.shape
    ), f"Expected shape to be all ints but found {array.shape=}."
    assert all(
        isinstance(c, int) for c in array.chunks
    ), f"Expected chunks to be all ints but found {array.chunks=}."
    assert array.shape == shape, f"Expected {shape=} but found {array.shape=}."
    assert array.chunks == chunks, f"Expected {chunks=} but found {array.chunks=}."
