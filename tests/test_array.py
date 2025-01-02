import dataclasses
import json
import math
import pickle
import re
from itertools import accumulate
from typing import TYPE_CHECKING, Any, Literal

import numcodecs
import numpy as np
import pytest

import zarr.api.asynchronous
from zarr import Array, AsyncArray, Group
from zarr.codecs import (
    BytesCodec,
    GzipCodec,
    TransposeCodec,
    VLenBytesCodec,
    VLenUTF8Codec,
    ZstdCodec,
)
from zarr.core._info import ArrayInfo
from zarr.core.array import (
    CompressorsLike,
    FiltersLike,
    _get_default_chunk_encoding_v2,
    _get_default_chunk_encoding_v3,
    _parse_chunk_encoding_v2,
    _parse_chunk_encoding_v3,
    chunks_initialized,
    create_array,
)
from zarr.core.buffer import default_buffer_prototype
from zarr.core.buffer.cpu import NDBuffer
from zarr.core.chunk_grids import _auto_partition
from zarr.core.common import JSON, MemoryOrder, ZarrFormat
from zarr.core.group import AsyncGroup
from zarr.core.indexing import ceildiv
from zarr.core.metadata.v3 import DataType
from zarr.core.sync import sync
from zarr.errors import ContainsArrayError, ContainsGroupError
from zarr.storage import LocalStore, MemoryStore
from zarr.storage.common import StorePath

if TYPE_CHECKING:
    from zarr.core.array_spec import ArrayConfigLike


@pytest.mark.parametrize("store", ["local", "memory", "zip"], indirect=["store"])
@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("overwrite", [True, False])
@pytest.mark.parametrize("extant_node", ["array", "group"])
def test_array_creation_existing_node(
    store: LocalStore | MemoryStore,
    zarr_format: ZarrFormat,
    overwrite: bool,
    extant_node: Literal["array", "group"],
) -> None:
    """
    Check that an existing array or group is handled as expected during array creation.
    """
    spath = StorePath(store)
    group = Group.from_store(spath, zarr_format=zarr_format)
    expected_exception: type[ContainsArrayError | ContainsGroupError]
    if extant_node == "array":
        expected_exception = ContainsArrayError
        _ = group.create_array("extant", shape=(10,), dtype="uint8")
    elif extant_node == "group":
        expected_exception = ContainsGroupError
        _ = group.create_group("extant")
    else:
        raise AssertionError

    new_shape = (2, 2)
    new_dtype = "float32"

    if overwrite:
        if not store.supports_deletes:
            pytest.skip("store does not support deletes")
        arr_new = zarr.create_array(
            spath / "extant",
            shape=new_shape,
            dtype=new_dtype,
            overwrite=overwrite,
            zarr_format=zarr_format,
        )
        assert arr_new.shape == new_shape
        assert arr_new.dtype == new_dtype
    else:
        with pytest.raises(expected_exception):
            arr_new = zarr.create_array(
                spath / "extant",
                shape=new_shape,
                dtype=new_dtype,
                overwrite=overwrite,
                zarr_format=zarr_format,
            )


@pytest.mark.parametrize("store", ["local", "memory", "zip"], indirect=["store"])
@pytest.mark.parametrize("zarr_format", [2, 3])
async def test_create_creates_parents(
    store: LocalStore | MemoryStore, zarr_format: ZarrFormat
) -> None:
    # prepare a root node, with some data set
    await zarr.api.asynchronous.open_group(
        store=store, path="a", zarr_format=zarr_format, attributes={"key": "value"}
    )

    # create a child node with a couple intermediates
    await zarr.api.asynchronous.create(
        shape=(2, 2), store=store, path="a/b/c/d", zarr_format=zarr_format
    )
    parts = ["a", "a/b", "a/b/c"]

    if zarr_format == 2:
        files = [".zattrs", ".zgroup"]
    else:
        files = ["zarr.json"]

    expected = [f"{part}/{file}" for file in files for part in parts]

    if zarr_format == 2:
        expected.extend([".zattrs", ".zgroup", "a/b/c/d/.zarray", "a/b/c/d/.zattrs"])
    else:
        expected.extend(["zarr.json", "a/b/c/d/zarr.json"])

    expected = sorted(expected)

    result = sorted([x async for x in store.list_prefix("")])

    assert result == expected

    paths = ["a", "a/b", "a/b/c"]
    for path in paths:
        g = await zarr.api.asynchronous.open_group(store=store, path=path)
        assert isinstance(g, AsyncGroup)


@pytest.mark.parametrize("store", ["local", "memory", "zip"], indirect=["store"])
@pytest.mark.parametrize("zarr_format", [2, 3])
def test_array_name_properties_no_group(
    store: LocalStore | MemoryStore, zarr_format: ZarrFormat
) -> None:
    arr = zarr.create_array(
        store=store, shape=(100,), chunks=(10,), zarr_format=zarr_format, dtype="i4"
    )
    assert arr.path == ""
    assert arr.name == "/"
    assert arr.basename == ""


@pytest.mark.parametrize("store", ["local", "memory", "zip"], indirect=["store"])
@pytest.mark.parametrize("zarr_format", [2, 3])
def test_array_name_properties_with_group(
    store: LocalStore | MemoryStore, zarr_format: ZarrFormat
) -> None:
    root = Group.from_store(store=store, zarr_format=zarr_format)
    foo = root.create_array("foo", shape=(100,), chunks=(10,), dtype="i4")
    assert foo.path == "foo"
    assert foo.name == "/foo"
    assert foo.basename == "foo"

    bar = root.create_group("bar")
    spam = bar.create_array("spam", shape=(100,), chunks=(10,), dtype="i4")

    assert spam.path == "bar/spam"
    assert spam.name == "/bar/spam"
    assert spam.basename == "spam"


@pytest.mark.parametrize("store", ["memory"], indirect=True)
@pytest.mark.parametrize("specifiy_fill_value", [True, False])
@pytest.mark.parametrize("dtype_str", ["bool", "uint8", "complex64"])
def test_array_v3_fill_value_default(
    store: MemoryStore, specifiy_fill_value: bool, dtype_str: str
) -> None:
    """
    Test that creating an array with the fill_value parameter set to None, or unspecified,
    results in the expected fill_value attribute of the array, i.e. 0 cast to the array's dtype.
    """
    shape = (10,)
    default_fill_value = 0
    if specifiy_fill_value:
        arr = zarr.create_array(
            store=store,
            shape=shape,
            dtype=dtype_str,
            zarr_format=3,
            chunks=shape,
            fill_value=None,
        )
    else:
        arr = zarr.create_array(
            store=store, shape=shape, dtype=dtype_str, zarr_format=3, chunks=shape
        )

    assert arr.fill_value == np.dtype(dtype_str).type(default_fill_value)
    assert arr.fill_value.dtype == arr.dtype


@pytest.mark.parametrize("store", ["memory"], indirect=True)
@pytest.mark.parametrize(
    ("dtype_str", "fill_value"),
    [("bool", True), ("uint8", 99), ("float32", -99.9), ("complex64", 3 + 4j)],
)
def test_array_v3_fill_value(store: MemoryStore, fill_value: int, dtype_str: str) -> None:
    shape = (10,)
    arr = zarr.create_array(
        store=store,
        shape=shape,
        dtype=dtype_str,
        zarr_format=3,
        chunks=shape,
        fill_value=fill_value,
    )

    assert arr.fill_value == np.dtype(dtype_str).type(fill_value)
    assert arr.fill_value.dtype == arr.dtype


def test_create_positional_args_deprecated() -> None:
    store = MemoryStore()
    with pytest.warns(FutureWarning, match="Pass"):
        zarr.Array.create(store, (2, 2), dtype="f8")


def test_selection_positional_args_deprecated() -> None:
    store = MemoryStore()
    arr = zarr.create_array(store, shape=(2, 2), dtype="f8")

    with pytest.warns(FutureWarning, match="Pass out"):
        arr.get_basic_selection(..., NDBuffer(array=np.empty((2, 2))))

    with pytest.warns(FutureWarning, match="Pass fields"):
        arr.set_basic_selection(..., 1, None)

    with pytest.warns(FutureWarning, match="Pass out"):
        arr.get_orthogonal_selection(..., NDBuffer(array=np.empty((2, 2))))

    with pytest.warns(FutureWarning, match="Pass"):
        arr.set_orthogonal_selection(..., 1, None)

    with pytest.warns(FutureWarning, match="Pass"):
        arr.get_mask_selection(np.zeros((2, 2), dtype=bool), NDBuffer(array=np.empty((0,))))

    with pytest.warns(FutureWarning, match="Pass"):
        arr.set_mask_selection(np.zeros((2, 2), dtype=bool), 1, None)

    with pytest.warns(FutureWarning, match="Pass"):
        arr.get_coordinate_selection(([0, 1], [0, 1]), NDBuffer(array=np.empty((2,))))

    with pytest.warns(FutureWarning, match="Pass"):
        arr.set_coordinate_selection(([0, 1], [0, 1]), 1, None)

    with pytest.warns(FutureWarning, match="Pass"):
        arr.get_block_selection((0, slice(None)), NDBuffer(array=np.empty((2, 2))))

    with pytest.warns(FutureWarning, match="Pass"):
        arr.set_block_selection((0, slice(None)), 1, None)


@pytest.mark.parametrize("store", ["memory"], indirect=True)
async def test_array_v3_nan_fill_value(store: MemoryStore) -> None:
    shape = (10,)
    arr = zarr.create_array(
        store=store,
        shape=shape,
        dtype=np.float64,
        zarr_format=3,
        chunks=shape,
        fill_value=np.nan,
    )
    arr[:] = np.nan

    assert np.isnan(arr.fill_value)
    assert arr.fill_value.dtype == arr.dtype
    # all fill value chunk is an empty chunk, and should not be written
    assert len([a async for a in store.list_prefix("/")]) == 0


@pytest.mark.parametrize("store", ["local"], indirect=["store"])
@pytest.mark.parametrize("zarr_format", [2, 3])
async def test_serializable_async_array(
    store: LocalStore | MemoryStore, zarr_format: ZarrFormat
) -> None:
    expected = await zarr.api.asynchronous.create_array(
        store=store, shape=(100,), chunks=(10,), zarr_format=zarr_format, dtype="i4"
    )
    # await expected.setitems(list(range(100)))

    p = pickle.dumps(expected)
    actual = pickle.loads(p)

    assert actual == expected
    # np.testing.assert_array_equal(await actual.getitem(slice(None)), await expected.getitem(slice(None)))
    # TODO: uncomment the parts of this test that will be impacted by the config/prototype changes in flight


@pytest.mark.parametrize("store", ["local"], indirect=["store"])
@pytest.mark.parametrize("zarr_format", [2, 3])
def test_serializable_sync_array(store: LocalStore, zarr_format: ZarrFormat) -> None:
    expected = zarr.create_array(
        store=store, shape=(100,), chunks=(10,), zarr_format=zarr_format, dtype="i4"
    )
    expected[:] = list(range(100))

    p = pickle.dumps(expected)
    actual = pickle.loads(p)

    assert actual == expected
    np.testing.assert_array_equal(actual[:], expected[:])


@pytest.mark.parametrize("store", ["memory"], indirect=True)
def test_storage_transformers(store: MemoryStore) -> None:
    """
    Test that providing an actual storage transformer produces a warning and otherwise passes through
    """
    metadata_dict: dict[str, JSON] = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": (10,),
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (1,)}},
        "data_type": "uint8",
        "chunk_key_encoding": {"name": "v2", "configuration": {"separator": "/"}},
        "codecs": (BytesCodec().to_dict(),),
        "fill_value": 0,
        "storage_transformers": ({"test": "should_raise"}),
    }
    match = "Arrays with storage transformers are not supported in zarr-python at this time."
    with pytest.raises(ValueError, match=match):
        Array.from_dict(StorePath(store), data=metadata_dict)


@pytest.mark.parametrize("test_cls", [Array, AsyncArray[Any]])
@pytest.mark.parametrize("nchunks", [2, 5, 10])
def test_nchunks(test_cls: type[Array] | type[AsyncArray[Any]], nchunks: int) -> None:
    """
    Test that nchunks returns the number of chunks defined for the array.
    """
    store = MemoryStore()
    shape = 100
    arr = zarr.create_array(store, shape=(shape,), chunks=(ceildiv(shape, nchunks),), dtype="i4")
    expected = nchunks
    if test_cls == Array:
        observed = arr.nchunks
    else:
        observed = arr._async_array.nchunks
    assert observed == expected


@pytest.mark.parametrize("test_cls", [Array, AsyncArray[Any]])
async def test_nchunks_initialized(test_cls: type[Array] | type[AsyncArray[Any]]) -> None:
    """
    Test that nchunks_initialized accurately returns the number of stored chunks.
    """
    store = MemoryStore()
    arr = zarr.create_array(store, shape=(100,), chunks=(10,), dtype="i4")

    # write chunks one at a time
    for idx, region in enumerate(arr._iter_chunk_regions()):
        arr[region] = 1
        expected = idx + 1
        if test_cls == Array:
            observed = arr.nchunks_initialized
        else:
            observed = await arr._async_array.nchunks_initialized()
        assert observed == expected

    # delete chunks
    for idx, key in enumerate(arr._iter_chunk_keys()):
        sync(arr.store_path.store.delete(key))
        if test_cls == Array:
            observed = arr.nchunks_initialized
        else:
            observed = await arr._async_array.nchunks_initialized()
        expected = arr.nchunks - idx - 1
        assert observed == expected


async def test_chunks_initialized() -> None:
    """
    Test that chunks_initialized accurately returns the keys of stored chunks.
    """
    store = MemoryStore()
    arr = zarr.create_array(store, shape=(100,), chunks=(10,), dtype="i4")

    chunks_accumulated = tuple(
        accumulate(tuple(tuple(v.split(" ")) for v in arr._iter_chunk_keys()))
    )
    for keys, region in zip(chunks_accumulated, arr._iter_chunk_regions(), strict=False):
        arr[region] = 1
        observed = sorted(await chunks_initialized(arr._async_array))
        expected = sorted(keys)
        assert observed == expected


def test_nbytes_stored() -> None:
    arr = zarr.create(shape=(100,), chunks=(10,), dtype="i4", codecs=[BytesCodec()])
    result = arr.nbytes_stored()
    assert result == 366  # the size of the metadata document. This is a fragile test.
    arr[:50] = 1
    result = arr.nbytes_stored()
    assert result == 566  # the size with 5 chunks filled.
    arr[50:] = 2
    result = arr.nbytes_stored()
    assert result == 766  # the size with all chunks filled.


async def test_nbytes_stored_async() -> None:
    arr = await zarr.api.asynchronous.create(
        shape=(100,), chunks=(10,), dtype="i4", codecs=[BytesCodec()]
    )
    result = await arr.nbytes_stored()
    assert result == 366  # the size of the metadata document. This is a fragile test.
    await arr.setitem(slice(50), 1)
    result = await arr.nbytes_stored()
    assert result == 566  # the size with 5 chunks filled.
    await arr.setitem(slice(50, 100), 2)
    result = await arr.nbytes_stored()
    assert result == 766  # the size with all chunks filled.


def test_default_fill_values() -> None:
    a = zarr.Array.create(MemoryStore(), shape=5, chunk_shape=5, dtype="<U4")
    assert a.fill_value == ""

    b = zarr.Array.create(MemoryStore(), shape=5, chunk_shape=5, dtype="<S4")
    assert b.fill_value == b""

    c = zarr.Array.create(MemoryStore(), shape=5, chunk_shape=5, dtype="i")
    assert c.fill_value == 0

    d = zarr.Array.create(MemoryStore(), shape=5, chunk_shape=5, dtype="f")
    assert d.fill_value == 0.0


def test_vlen_errors() -> None:
    with pytest.raises(ValueError, match="At least one ArrayBytesCodec is required."):
        Array.create(MemoryStore(), shape=5, chunks=5, dtype="<U4", codecs=[])

    with pytest.raises(
        ValueError,
        match="For string dtype, ArrayBytesCodec must be `VLenUTF8Codec`, got `BytesCodec`.",
    ):
        Array.create(MemoryStore(), shape=5, chunks=5, dtype="<U4", codecs=[BytesCodec()])

    with pytest.raises(ValueError, match="Only one ArrayBytesCodec is allowed."):
        Array.create(
            MemoryStore(),
            shape=5,
            chunks=5,
            dtype="<U4",
            codecs=[BytesCodec(), VLenBytesCodec()],
        )

    with pytest.raises(
        ValueError,
        match="For string dtype, ArrayBytesCodec must be `VLenUTF8Codec`, got `BytesCodec`.",
    ):
        zarr.create_array(
            MemoryStore(), shape=(5,), chunks=(5,), dtype="<U4", serializer=BytesCodec()
        )


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_update_attrs(zarr_format: ZarrFormat) -> None:
    # regression test for https://github.com/zarr-developers/zarr-python/issues/2328
    store = MemoryStore()
    arr = zarr.create_array(
        store=store, shape=(5,), chunks=(5,), dtype="f8", zarr_format=zarr_format
    )
    arr.attrs["foo"] = "bar"
    assert arr.attrs["foo"] == "bar"

    arr2 = zarr.open_array(store=store, zarr_format=zarr_format)
    assert arr2.attrs["foo"] == "bar"


class TestInfo:
    def test_info_v2(self) -> None:
        arr = zarr.create(shape=(4, 4), chunks=(2, 2), zarr_format=2)
        result = arr.info
        expected = ArrayInfo(
            _zarr_format=2,
            _data_type=np.dtype("float64"),
            _shape=(4, 4),
            _chunk_shape=(2, 2),
            _order="C",
            _read_only=False,
            _store_type="MemoryStore",
            _count_bytes=128,
            _compressor=numcodecs.Zstd(),
        )
        assert result == expected

    def test_info_v3(self) -> None:
        arr = zarr.create(shape=(4, 4), chunks=(2, 2), zarr_format=3)
        result = arr.info
        expected = ArrayInfo(
            _zarr_format=3,
            _data_type=DataType.parse("float64"),
            _shape=(4, 4),
            _chunk_shape=(2, 2),
            _order="C",
            _read_only=False,
            _store_type="MemoryStore",
            _codecs=[BytesCodec(), ZstdCodec()],
            _count_bytes=128,
        )
        assert result == expected

    def test_info_complete(self) -> None:
        arr = zarr.create(shape=(4, 4), chunks=(2, 2), zarr_format=3, codecs=[BytesCodec()])
        result = arr.info_complete()
        expected = ArrayInfo(
            _zarr_format=3,
            _data_type=DataType.parse("float64"),
            _shape=(4, 4),
            _chunk_shape=(2, 2),
            _order="C",
            _read_only=False,
            _store_type="MemoryStore",
            _codecs=[BytesCodec()],
            _count_bytes=128,
            _count_chunks_initialized=0,
            _count_bytes_stored=373,  # the metadata?
        )
        assert result == expected

        arr[:2, :2] = 10
        result = arr.info_complete()
        expected = dataclasses.replace(
            expected, _count_chunks_initialized=1, _count_bytes_stored=405
        )
        assert result == expected

    async def test_info_v2_async(self) -> None:
        arr = await zarr.api.asynchronous.create(shape=(4, 4), chunks=(2, 2), zarr_format=2)
        result = arr.info
        expected = ArrayInfo(
            _zarr_format=2,
            _data_type=np.dtype("float64"),
            _shape=(4, 4),
            _chunk_shape=(2, 2),
            _order="C",
            _read_only=False,
            _store_type="MemoryStore",
            _count_bytes=128,
            _compressor=numcodecs.Zstd(),
        )
        assert result == expected

    async def test_info_v3_async(self) -> None:
        arr = await zarr.api.asynchronous.create(shape=(4, 4), chunks=(2, 2), zarr_format=3)
        result = arr.info
        expected = ArrayInfo(
            _zarr_format=3,
            _data_type=DataType.parse("float64"),
            _shape=(4, 4),
            _chunk_shape=(2, 2),
            _order="C",
            _read_only=False,
            _store_type="MemoryStore",
            _codecs=[BytesCodec(), ZstdCodec()],
            _count_bytes=128,
        )
        assert result == expected

    async def test_info_complete_async(self) -> None:
        arr = await zarr.api.asynchronous.create(
            shape=(4, 4), chunks=(2, 2), zarr_format=3, codecs=[BytesCodec()]
        )
        result = await arr.info_complete()
        expected = ArrayInfo(
            _zarr_format=3,
            _data_type=DataType.parse("float64"),
            _shape=(4, 4),
            _chunk_shape=(2, 2),
            _order="C",
            _read_only=False,
            _store_type="MemoryStore",
            _codecs=[BytesCodec()],
            _count_bytes=128,
            _count_chunks_initialized=0,
            _count_bytes_stored=373,  # the metadata?
        )
        assert result == expected

        await arr.setitem((slice(2), slice(2)), 10)
        result = await arr.info_complete()
        expected = dataclasses.replace(
            expected, _count_chunks_initialized=1, _count_bytes_stored=405
        )
        assert result == expected


@pytest.mark.parametrize("store", ["memory"], indirect=True)
def test_resize_1d(store: MemoryStore, zarr_format: ZarrFormat) -> None:
    z = zarr.create(
        shape=105, chunks=10, dtype="i4", fill_value=0, store=store, zarr_format=zarr_format
    )
    a = np.arange(105, dtype="i4")
    z[:] = a
    assert (105,) == z.shape
    assert (105,) == z[:].shape
    assert np.dtype("i4") == z.dtype
    assert np.dtype("i4") == z[:].dtype
    assert (10,) == z.chunks
    np.testing.assert_array_equal(a, z[:])

    z.resize(205)
    assert (205,) == z.shape
    assert (205,) == z[:].shape
    assert np.dtype("i4") == z.dtype
    assert np.dtype("i4") == z[:].dtype
    assert (10,) == z.chunks
    np.testing.assert_array_equal(a, z[:105])
    np.testing.assert_array_equal(np.zeros(100, dtype="i4"), z[105:])

    z.resize(55)
    assert (55,) == z.shape
    assert (55,) == z[:].shape
    assert np.dtype("i4") == z.dtype
    assert np.dtype("i4") == z[:].dtype
    assert (10,) == z.chunks
    np.testing.assert_array_equal(a[:55], z[:])

    # via shape setter
    new_shape = (105,)
    z.shape = new_shape
    assert new_shape == z.shape
    assert new_shape == z[:].shape


@pytest.mark.parametrize("store", ["memory"], indirect=True)
def test_resize_2d(store: MemoryStore, zarr_format: ZarrFormat) -> None:
    z = zarr.create(
        shape=(105, 105),
        chunks=(10, 10),
        dtype="i4",
        fill_value=0,
        store=store,
        zarr_format=zarr_format,
    )
    a = np.arange(105 * 105, dtype="i4").reshape((105, 105))
    z[:] = a
    assert (105, 105) == z.shape
    assert (105, 105) == z[:].shape
    assert np.dtype("i4") == z.dtype
    assert np.dtype("i4") == z[:].dtype
    assert (10, 10) == z.chunks
    np.testing.assert_array_equal(a, z[:])

    z.resize((205, 205))
    assert (205, 205) == z.shape
    assert (205, 205) == z[:].shape
    assert np.dtype("i4") == z.dtype
    assert np.dtype("i4") == z[:].dtype
    assert (10, 10) == z.chunks
    np.testing.assert_array_equal(a, z[:105, :105])
    np.testing.assert_array_equal(np.zeros((100, 205), dtype="i4"), z[105:, :])
    np.testing.assert_array_equal(np.zeros((205, 100), dtype="i4"), z[:, 105:])

    z.resize((55, 55))
    assert (55, 55) == z.shape
    assert (55, 55) == z[:].shape
    assert np.dtype("i4") == z.dtype
    assert np.dtype("i4") == z[:].dtype
    assert (10, 10) == z.chunks
    np.testing.assert_array_equal(a[:55, :55], z[:])

    z.resize((55, 1))
    assert (55, 1) == z.shape
    assert (55, 1) == z[:].shape
    assert np.dtype("i4") == z.dtype
    assert np.dtype("i4") == z[:].dtype
    assert (10, 10) == z.chunks
    np.testing.assert_array_equal(a[:55, :1], z[:])

    z.resize((1, 55))
    assert (1, 55) == z.shape
    assert (1, 55) == z[:].shape
    assert np.dtype("i4") == z.dtype
    assert np.dtype("i4") == z[:].dtype
    assert (10, 10) == z.chunks
    np.testing.assert_array_equal(a[:1, :10], z[:, :10])
    np.testing.assert_array_equal(np.zeros((1, 55 - 10), dtype="i4"), z[:, 10:55])

    # via shape setter
    new_shape = (105, 105)
    z.shape = new_shape
    assert new_shape == z.shape
    assert new_shape == z[:].shape


@pytest.mark.parametrize("store", ["memory"], indirect=True)
def test_append_1d(store: MemoryStore, zarr_format: ZarrFormat) -> None:
    a = np.arange(105)
    z = zarr.create(shape=a.shape, chunks=10, dtype=a.dtype, store=store, zarr_format=zarr_format)
    z[:] = a
    assert a.shape == z.shape
    assert a.dtype == z.dtype
    assert (10,) == z.chunks
    np.testing.assert_array_equal(a, z[:])

    b = np.arange(105, 205)
    e = np.append(a, b)
    assert z.shape == (105,)
    z.append(b)
    assert e.shape == z.shape
    assert e.dtype == z.dtype
    assert (10,) == z.chunks
    np.testing.assert_array_equal(e, z[:])

    # check append handles array-like
    c = [1, 2, 3]
    f = np.append(e, c)
    z.append(c)
    assert f.shape == z.shape
    assert f.dtype == z.dtype
    assert (10,) == z.chunks
    np.testing.assert_array_equal(f, z[:])


@pytest.mark.parametrize("store", ["memory"], indirect=True)
def test_append_2d(store: MemoryStore, zarr_format: ZarrFormat) -> None:
    a = np.arange(105 * 105, dtype="i4").reshape((105, 105))
    z = zarr.create(
        shape=a.shape, chunks=(10, 10), dtype=a.dtype, store=store, zarr_format=zarr_format
    )
    z[:] = a
    assert a.shape == z.shape
    assert a.dtype == z.dtype
    assert (10, 10) == z.chunks
    actual = z[:]
    np.testing.assert_array_equal(a, actual)

    b = np.arange(105 * 105, 2 * 105 * 105, dtype="i4").reshape((105, 105))
    e = np.append(a, b, axis=0)
    z.append(b)
    assert e.shape == z.shape
    assert e.dtype == z.dtype
    assert (10, 10) == z.chunks
    actual = z[:]
    np.testing.assert_array_equal(e, actual)


@pytest.mark.parametrize("store", ["memory"], indirect=True)
def test_append_2d_axis(store: MemoryStore, zarr_format: ZarrFormat) -> None:
    a = np.arange(105 * 105, dtype="i4").reshape((105, 105))
    z = zarr.create(
        shape=a.shape, chunks=(10, 10), dtype=a.dtype, store=store, zarr_format=zarr_format
    )
    z[:] = a
    assert a.shape == z.shape
    assert a.dtype == z.dtype
    assert (10, 10) == z.chunks
    np.testing.assert_array_equal(a, z[:])

    b = np.arange(105 * 105, 2 * 105 * 105, dtype="i4").reshape((105, 105))
    e = np.append(a, b, axis=1)
    z.append(b, axis=1)
    assert e.shape == z.shape
    assert e.dtype == z.dtype
    assert (10, 10) == z.chunks
    np.testing.assert_array_equal(e, z[:])


@pytest.mark.parametrize("store", ["memory"], indirect=True)
def test_append_bad_shape(store: MemoryStore, zarr_format: ZarrFormat) -> None:
    a = np.arange(100)
    z = zarr.create(shape=a.shape, chunks=10, dtype=a.dtype, store=store, zarr_format=zarr_format)
    z[:] = a
    b = a.reshape(10, 10)
    with pytest.raises(ValueError):
        z.append(b)


@pytest.mark.parametrize("order", ["C", "F", None])
@pytest.mark.parametrize("store", ["memory"], indirect=True)
def test_array_create_metadata_order_v2(
    order: MemoryOrder | None, zarr_format: int, store: MemoryStore
) -> None:
    """
    Test that the ``order`` attribute in zarr v2 array metadata is set correctly via the ``order``
    keyword argument to ``Array.create``. When ``order`` is ``None``, the value of the
    ``array.order`` config is used.
    """
    arr = zarr.create_array(store=store, shape=(2, 2), order=order, zarr_format=2, dtype="i4")

    expected = order or zarr.config.get("array.order")
    assert arr.metadata.order == expected  # type: ignore[union-attr]


@pytest.mark.parametrize("order_config", ["C", "F", None])
@pytest.mark.parametrize("store", ["memory"], indirect=True)
def test_array_create_order(
    order_config: MemoryOrder | None,
    zarr_format: ZarrFormat,
    store: MemoryStore,
) -> None:
    """
    Test that the arrays generated by array indexing have a memory order defined by the config order
    value
    """
    config: ArrayConfigLike = {}
    if order_config is None:
        config = {}
        expected = zarr.config.get("array.order")
    else:
        config = {"order": order_config}
        expected = order_config

    arr = zarr.create_array(
        store=store, shape=(2, 2), zarr_format=zarr_format, dtype="i4", config=config
    )

    vals = np.asarray(arr)
    if expected == "C":
        assert vals.flags.c_contiguous
    elif expected == "F":
        assert vals.flags.f_contiguous
    else:
        raise AssertionError


@pytest.mark.parametrize("write_empty_chunks", [True, False])
def test_write_empty_chunks_config(write_empty_chunks: bool) -> None:
    """
    Test that the value of write_empty_chunks is sensitive to the global config when not set
    explicitly
    """
    with zarr.config.set({"array.write_empty_chunks": write_empty_chunks}):
        arr = zarr.create_array({}, shape=(2, 2), dtype="i4")
        assert arr._async_array._config.write_empty_chunks == write_empty_chunks


@pytest.mark.parametrize("store", ["memory"], indirect=True)
@pytest.mark.parametrize("write_empty_chunks", [True, False])
@pytest.mark.parametrize("fill_value", [0, 5])
def test_write_empty_chunks_behavior(
    zarr_format: ZarrFormat, store: MemoryStore, write_empty_chunks: bool, fill_value: int
) -> None:
    """
    Check that the write_empty_chunks value of the config is applied correctly. We expect that
    when write_empty_chunks is True, writing chunks equal to the fill value will result in
    those chunks appearing in the store.

    When write_empty_chunks is False, writing chunks that are equal to the fill value will result in
    those chunks not being present in the store. In particular, they should be deleted if they were
    already present.
    """

    arr = zarr.create_array(
        store=store,
        shape=(2,),
        zarr_format=zarr_format,
        dtype="i4",
        fill_value=fill_value,
        chunks=(1,),
        config={"write_empty_chunks": write_empty_chunks},
    )

    assert arr._async_array._config.write_empty_chunks == write_empty_chunks

    # initialize the store with some non-fill value chunks
    arr[:] = fill_value + 1
    assert arr.nchunks_initialized == arr.nchunks

    arr[:] = fill_value

    if not write_empty_chunks:
        assert arr.nchunks_initialized == 0
    else:
        assert arr.nchunks_initialized == arr.nchunks


@pytest.mark.parametrize(
    ("fill_value", "expected"),
    [
        (np.nan * 1j, ["NaN", "NaN"]),
        (np.nan, ["NaN", 0.0]),
        (np.inf, ["Infinity", 0.0]),
        (np.inf * 1j, ["NaN", "Infinity"]),
        (-np.inf, ["-Infinity", 0.0]),
        (math.inf, ["Infinity", 0.0]),
    ],
)
async def test_special_complex_fill_values_roundtrip(fill_value: Any, expected: list[Any]) -> None:
    store = MemoryStore()
    zarr.create_array(store=store, shape=(1,), dtype=np.complex64, fill_value=fill_value)
    content = await store.get("zarr.json", prototype=default_buffer_prototype())
    assert content is not None
    actual = json.loads(content.to_bytes())
    assert actual["fill_value"] == expected


@pytest.mark.parametrize("shape", [(1,), (2, 3), (4, 5, 6)])
@pytest.mark.parametrize("dtype", ["uint8", "float32"])
@pytest.mark.parametrize("array_type", ["async", "sync"])
async def test_nbytes(
    shape: tuple[int, ...], dtype: str, array_type: Literal["async", "sync"]
) -> None:
    """
    Test that the ``nbytes`` attribute of an Array or AsyncArray correctly reports the capacity of
    the chunks of that array.
    """
    store = MemoryStore()
    arr = zarr.create_array(store=store, shape=shape, dtype=dtype, fill_value=0)
    if array_type == "async":
        assert arr._async_array.nbytes == np.prod(arr.shape) * arr.dtype.itemsize
    else:
        assert arr.nbytes == np.prod(arr.shape) * arr.dtype.itemsize


@pytest.mark.parametrize(
    ("array_shape", "chunk_shape"),
    [((256,), (2,))],
)
def test_auto_partition_auto_shards(
    array_shape: tuple[int, ...], chunk_shape: tuple[int, ...]
) -> None:
    """
    Test that automatically picking a shard size returns a tuple of 2 * the chunk shape for any axis
    where there are 8 or more chunks.
    """
    dtype = np.dtype("uint8")
    expected_shards: tuple[int, ...] = ()
    for cs, a_len in zip(chunk_shape, array_shape, strict=False):
        if a_len // cs >= 8:
            expected_shards += (2 * cs,)
        else:
            expected_shards += (cs,)

    auto_shards, _ = _auto_partition(
        array_shape=array_shape, chunk_shape=chunk_shape, shard_shape="auto", dtype=dtype
    )
    assert auto_shards == expected_shards


def test_chunks_and_shards() -> None:
    store = StorePath(MemoryStore())
    shape = (100, 100)
    chunks = (5, 5)
    shards = (10, 10)

    arr_v3 = zarr.create_array(store=store / "v3", shape=shape, chunks=chunks, dtype="i4")
    assert arr_v3.chunks == chunks
    assert arr_v3.shards is None

    arr_v3_sharding = zarr.create_array(
        store=store / "v3_sharding",
        shape=shape,
        chunks=chunks,
        shards=shards,
        dtype="i4",
    )
    assert arr_v3_sharding.chunks == chunks
    assert arr_v3_sharding.shards == shards

    arr_v2 = zarr.create_array(
        store=store / "v2", shape=shape, chunks=chunks, zarr_format=2, dtype="i4"
    )
    assert arr_v2.chunks == chunks
    assert arr_v2.shards is None


def test_create_array_default_fill_values() -> None:
    a = zarr.create_array(MemoryStore(), shape=(5,), chunks=(5,), dtype="<U4")
    assert a.fill_value == ""

    b = zarr.create_array(MemoryStore(), shape=(5,), chunks=(5,), dtype="<S4")
    assert b.fill_value == b""

    c = zarr.create_array(MemoryStore(), shape=(5,), chunks=(5,), dtype="i")
    assert c.fill_value == 0

    d = zarr.create_array(MemoryStore(), shape=(5,), chunks=(5,), dtype="f")
    assert d.fill_value == 0.0


@pytest.mark.parametrize("store", ["memory"], indirect=True)
@pytest.mark.parametrize("dtype", ["uint8", "float32", "str"])
@pytest.mark.parametrize("empty_value", [None, ()])
async def test_create_array_no_filters_compressors(
    store: MemoryStore, dtype: str, empty_value: Any
) -> None:
    """
    Test that the default ``filters`` and ``compressors`` are removed when ``create_array`` is invoked.
    """

    # v2
    arr = await create_array(
        store=store,
        dtype=dtype,
        shape=(10,),
        zarr_format=2,
        compressors=empty_value,
        filters=empty_value,
    )
    # The v2 metadata stores None and () separately
    assert arr.metadata.filters == empty_value  # type: ignore[union-attr]
    # The v2 metadata does not allow tuple for compressor, therefore it is turned into None
    assert arr.metadata.compressor is None  # type: ignore[union-attr]

    # v3
    arr = await create_array(
        store=store,
        dtype=dtype,
        shape=(10,),
        compressors=empty_value,
        filters=empty_value,
    )
    if dtype == "str":
        assert arr.metadata.codecs == [VLenUTF8Codec()]  # type: ignore[union-attr]
    else:
        assert arr.metadata.codecs == [BytesCodec()]  # type: ignore[union-attr]


@pytest.mark.parametrize("store", ["memory"], indirect=True)
@pytest.mark.parametrize("dtype", ["uint8", "float32", "str"])
@pytest.mark.parametrize(
    "compressors",
    [
        "auto",
        None,
        (),
        (ZstdCodec(level=3),),
        (ZstdCodec(level=3), GzipCodec(level=0)),
        ZstdCodec(level=3),
        {"name": "zstd", "configuration": {"level": 3}},
        ({"name": "zstd", "configuration": {"level": 3}},),
    ],
)
@pytest.mark.parametrize(
    "filters",
    [
        "auto",
        None,
        (),
        (
            TransposeCodec(
                order=[
                    0,
                ]
            ),
        ),
        (
            TransposeCodec(
                order=[
                    0,
                ]
            ),
            TransposeCodec(
                order=[
                    0,
                ]
            ),
        ),
        TransposeCodec(
            order=[
                0,
            ]
        ),
        {"name": "transpose", "configuration": {"order": [0]}},
        ({"name": "transpose", "configuration": {"order": [0]}},),
    ],
)
async def test_create_array_v3_chunk_encoding(
    store: MemoryStore, compressors: CompressorsLike, filters: FiltersLike, dtype: str
) -> None:
    """
    Test various possibilities for the compressors and filters parameter to create_array
    """
    arr = await create_array(
        store=store,
        dtype=dtype,
        shape=(10,),
        zarr_format=3,
        filters=filters,
        compressors=compressors,
    )
    aa_codecs_expected, _, bb_codecs_expected = _parse_chunk_encoding_v3(
        filters=filters, compressors=compressors, serializer="auto", dtype=np.dtype(dtype)
    )
    # TODO: find a better way to get the filters / compressors from the array.
    assert arr.codec_pipeline.array_array_codecs == aa_codecs_expected  # type: ignore[attr-defined]
    assert arr.codec_pipeline.bytes_bytes_codecs == bb_codecs_expected  # type: ignore[attr-defined]


@pytest.mark.parametrize("store", ["memory"], indirect=True)
@pytest.mark.parametrize("dtype", ["uint8", "float32", "str"])
@pytest.mark.parametrize(
    "compressors",
    [
        "auto",
        None,
        numcodecs.Zstd(level=3),
        (),
        (numcodecs.Zstd(level=3),),
    ],
)
@pytest.mark.parametrize(
    "filters", ["auto", None, numcodecs.GZip(level=1), (numcodecs.GZip(level=1),)]
)
async def test_create_array_v2_chunk_encoding(
    store: MemoryStore, compressors: CompressorsLike, filters: FiltersLike, dtype: str
) -> None:
    arr = await create_array(
        store=store,
        dtype=dtype,
        shape=(10,),
        zarr_format=2,
        compressors=compressors,
        filters=filters,
    )
    filters_expected, compressor_expected = _parse_chunk_encoding_v2(
        filters=filters, compressor=compressors, dtype=np.dtype(dtype)
    )
    # TODO: find a better way to get the filters/compressor from the array.
    assert arr.metadata.compressor == compressor_expected  # type: ignore[union-attr]
    assert arr.metadata.filters == filters_expected  # type: ignore[union-attr]


@pytest.mark.parametrize("store", ["memory"], indirect=True)
@pytest.mark.parametrize("dtype", ["uint8", "float32", "str"])
async def test_create_array_v3_default_filters_compressors(store: MemoryStore, dtype: str) -> None:
    """
    Test that the default ``filters`` and ``compressors`` are used when ``create_array`` is invoked with
    ``zarr_format`` = 3 and ``filters`` and ``compressors`` are not specified.
    """
    arr = await create_array(
        store=store,
        dtype=dtype,
        shape=(10,),
        zarr_format=3,
    )
    expected_aa, expected_ab, expected_bb = _get_default_chunk_encoding_v3(np_dtype=np.dtype(dtype))
    # TODO: define the codec pipeline class such that these fields are required, which will obviate the
    # type ignore statements
    assert arr.codec_pipeline.array_array_codecs == expected_aa  # type: ignore[attr-defined]
    assert arr.codec_pipeline.bytes_bytes_codecs == expected_bb  # type: ignore[attr-defined]
    assert arr.codec_pipeline.array_bytes_codec == expected_ab  # type: ignore[attr-defined]


@pytest.mark.parametrize("store", ["memory"], indirect=True)
@pytest.mark.parametrize("dtype", ["uint8", "float32", "str"])
async def test_create_array_v2_default_filters_compressors(store: MemoryStore, dtype: str) -> None:
    """
    Test that the default ``filters`` and ``compressors`` are used when ``create_array`` is invoked with
    ``zarr_format`` = 2 and ``filters`` and ``compressors`` are not specified.
    """
    arr = await create_array(
        store=store,
        dtype=dtype,
        shape=(10,),
        zarr_format=2,
    )
    expected_filters, expected_compressors = _get_default_chunk_encoding_v2(
        np_dtype=np.dtype(dtype)
    )
    assert arr.metadata.filters == expected_filters  # type: ignore[union-attr]
    assert arr.metadata.compressor == expected_compressors  # type: ignore[union-attr]


@pytest.mark.parametrize("store", ["memory"], indirect=True)
async def test_create_array_v2_no_shards(store: MemoryStore) -> None:
    """
    Test that creating a Zarr v2 array with ``shard_shape`` set to a non-None value raises an error.
    """
    msg = re.escape(
        "Zarr v2 arrays can only be created with `shard_shape` set to `None`. Got `shard_shape=(5,)` instead."
    )
    with pytest.raises(ValueError, match=msg):
        _ = await create_array(
            store=store,
            dtype="uint8",
            shape=(10,),
            shards=(5,),
            zarr_format=2,
        )


async def test_scalar_array() -> None:
    arr = zarr.array(1.5)
    assert arr[...] == 1.5
    assert arr[()] == 1.5
    assert arr.shape == ()
