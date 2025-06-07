import dataclasses
import json
import math
import multiprocessing as mp
import pickle
import re
import sys
from itertools import accumulate
from typing import TYPE_CHECKING, Any, Literal
from unittest import mock

import numcodecs
import numpy as np
import numpy.typing as npt
import pytest
from packaging.version import Version

import zarr.api.asynchronous
import zarr.api.synchronous as sync_api
from zarr import Array, AsyncArray, Group
from zarr.abc.store import Store
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
from zarr.core.buffer import NDArrayLike, NDArrayLikeOrScalar, default_buffer_prototype
from zarr.core.buffer.cpu import NDBuffer
from zarr.core.chunk_grids import _auto_partition
from zarr.core.chunk_key_encodings import ChunkKeyEncodingParams
from zarr.core.common import JSON, MemoryOrder, ZarrFormat
from zarr.core.group import AsyncGroup
from zarr.core.indexing import BasicIndexer, ceildiv
from zarr.core.metadata.v3 import ArrayV3Metadata, DataType
from zarr.core.sync import sync
from zarr.errors import ContainsArrayError, ContainsGroupError
from zarr.storage import LocalStore, MemoryStore, StorePath

if TYPE_CHECKING:
    from zarr.core.array_spec import ArrayConfigLike
from zarr.core.metadata.v2 import ArrayV2Metadata


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


async def test_create_deprecated() -> None:
    with pytest.warns(DeprecationWarning):
        with pytest.warns(FutureWarning, match=re.escape("Pass shape=(2, 2) as keyword args")):
            await zarr.AsyncArray.create(MemoryStore(), (2, 2), dtype="f8")  # type: ignore[call-overload]
    with pytest.warns(DeprecationWarning):
        with pytest.warns(FutureWarning, match=re.escape("Pass shape=(2, 2) as keyword args")):
            zarr.Array.create(MemoryStore(), (2, 2), dtype="f8")


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
@pytest.mark.parametrize("zarr_format", [2, 3, "invalid"])
def test_storage_transformers(store: MemoryStore, zarr_format: ZarrFormat | str) -> None:
    """
    Test that providing an actual storage transformer produces a warning and otherwise passes through
    """
    metadata_dict: dict[str, JSON]
    if zarr_format == 3:
        metadata_dict = {
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
    else:
        metadata_dict = {
            "zarr_format": zarr_format,
            "shape": (10,),
            "chunks": (1,),
            "dtype": "uint8",
            "dimension_separator": ".",
            "codecs": (BytesCodec().to_dict(),),
            "fill_value": 0,
            "order": "C",
            "storage_transformers": ({"test": "should_raise"}),
        }
    if zarr_format == 3:
        match = "Arrays with storage transformers are not supported in zarr-python at this time."
        with pytest.raises(ValueError, match=match):
            Array.from_dict(StorePath(store), data=metadata_dict)
    elif zarr_format == 2:
        # no warning
        Array.from_dict(StorePath(store), data=metadata_dict)
    else:
        match = f"Invalid zarr_format: {zarr_format}. Expected 2 or 3"
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


@pytest.mark.parametrize("path", ["", "foo"])
async def test_chunks_initialized(path: str) -> None:
    """
    Test that chunks_initialized accurately returns the keys of stored chunks.
    """
    store = MemoryStore()
    arr = zarr.create_array(store, name=path, shape=(100,), chunks=(10,), dtype="i4")

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
    assert result == 502  # the size of the metadata document. This is a fragile test.
    arr[:50] = 1
    result = arr.nbytes_stored()
    assert result == 702  # the size with 5 chunks filled.
    arr[50:] = 2
    result = arr.nbytes_stored()
    assert result == 902  # the size with all chunks filled.


async def test_nbytes_stored_async() -> None:
    arr = await zarr.api.asynchronous.create(
        shape=(100,), chunks=(10,), dtype="i4", codecs=[BytesCodec()]
    )
    result = await arr.nbytes_stored()
    assert result == 502  # the size of the metadata document. This is a fragile test.
    await arr.setitem(slice(50), 1)
    result = await arr.nbytes_stored()
    assert result == 702  # the size with 5 chunks filled.
    await arr.setitem(slice(50, 100), 2)
    result = await arr.nbytes_stored()
    assert result == 902  # the size with all chunks filled.


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


@pytest.mark.parametrize(("chunks", "shards"), [((2, 2), None), ((2, 2), (4, 4))])
class TestInfo:
    def test_info_v2(self, chunks: tuple[int, int], shards: tuple[int, int] | None) -> None:
        arr = zarr.create_array(store={}, shape=(8, 8), dtype="f8", chunks=chunks, zarr_format=2)
        result = arr.info
        expected = ArrayInfo(
            _zarr_format=2,
            _data_type=np.dtype("float64"),
            _fill_value=arr.fill_value,
            _shape=(8, 8),
            _chunk_shape=chunks,
            _shard_shape=None,
            _order="C",
            _read_only=False,
            _store_type="MemoryStore",
            _count_bytes=512,
            _compressors=(numcodecs.Zstd(),),
        )
        assert result == expected

    def test_info_v3(self, chunks: tuple[int, int], shards: tuple[int, int] | None) -> None:
        arr = zarr.create_array(store={}, shape=(8, 8), dtype="f8", chunks=chunks, shards=shards)
        result = arr.info
        expected = ArrayInfo(
            _zarr_format=3,
            _data_type=DataType.parse("float64"),
            _fill_value=arr.fill_value,
            _shape=(8, 8),
            _chunk_shape=chunks,
            _shard_shape=shards,
            _order="C",
            _read_only=False,
            _store_type="MemoryStore",
            _compressors=(ZstdCodec(),),
            _serializer=BytesCodec(),
            _count_bytes=512,
        )
        assert result == expected

    def test_info_complete(self, chunks: tuple[int, int], shards: tuple[int, int] | None) -> None:
        arr = zarr.create_array(
            store={},
            shape=(8, 8),
            dtype="f8",
            chunks=chunks,
            shards=shards,
            compressors=(),
        )
        result = arr.info_complete()
        expected = ArrayInfo(
            _zarr_format=3,
            _data_type=DataType.parse("float64"),
            _fill_value=arr.fill_value,
            _shape=(8, 8),
            _chunk_shape=chunks,
            _shard_shape=shards,
            _order="C",
            _read_only=False,
            _store_type="MemoryStore",
            _serializer=BytesCodec(),
            _count_bytes=512,
            _count_chunks_initialized=0,
            _count_bytes_stored=521 if shards is None else 982,  # the metadata?
        )
        assert result == expected

        arr[:4, :4] = 10
        result = arr.info_complete()
        if shards is None:
            expected = dataclasses.replace(
                expected, _count_chunks_initialized=4, _count_bytes_stored=649
            )
        else:
            expected = dataclasses.replace(
                expected, _count_chunks_initialized=1, _count_bytes_stored=1178
            )
        assert result == expected

    async def test_info_v2_async(
        self, chunks: tuple[int, int], shards: tuple[int, int] | None
    ) -> None:
        arr = await zarr.api.asynchronous.create_array(
            store={}, shape=(8, 8), dtype="f8", chunks=chunks, zarr_format=2
        )
        result = arr.info
        expected = ArrayInfo(
            _zarr_format=2,
            _data_type=np.dtype("float64"),
            _fill_value=arr.metadata.fill_value,
            _shape=(8, 8),
            _chunk_shape=(2, 2),
            _shard_shape=None,
            _order="C",
            _read_only=False,
            _store_type="MemoryStore",
            _count_bytes=512,
            _compressors=(numcodecs.Zstd(),),
        )
        assert result == expected

    async def test_info_v3_async(
        self, chunks: tuple[int, int], shards: tuple[int, int] | None
    ) -> None:
        arr = await zarr.api.asynchronous.create_array(
            store={},
            shape=(8, 8),
            dtype="f8",
            chunks=chunks,
            shards=shards,
        )
        result = arr.info
        expected = ArrayInfo(
            _zarr_format=3,
            _data_type=DataType.parse("float64"),
            _fill_value=arr.metadata.fill_value,
            _shape=(8, 8),
            _chunk_shape=chunks,
            _shard_shape=shards,
            _order="C",
            _read_only=False,
            _store_type="MemoryStore",
            _compressors=(ZstdCodec(),),
            _serializer=BytesCodec(),
            _count_bytes=512,
        )
        assert result == expected

    async def test_info_complete_async(
        self, chunks: tuple[int, int], shards: tuple[int, int] | None
    ) -> None:
        arr = await zarr.api.asynchronous.create_array(
            store={},
            dtype="f8",
            shape=(8, 8),
            chunks=chunks,
            shards=shards,
            compressors=None,
        )
        result = await arr.info_complete()
        expected = ArrayInfo(
            _zarr_format=3,
            _data_type=DataType.parse("float64"),
            _fill_value=arr.metadata.fill_value,
            _shape=(8, 8),
            _chunk_shape=chunks,
            _shard_shape=shards,
            _order="C",
            _read_only=False,
            _store_type="MemoryStore",
            _serializer=BytesCodec(),
            _count_bytes=512,
            _count_chunks_initialized=0,
            _count_bytes_stored=521 if shards is None else 982,  # the metadata?
        )
        assert result == expected

        await arr.setitem((slice(4), slice(4)), 10)
        result = await arr.info_complete()
        if shards is None:
            expected = dataclasses.replace(
                expected, _count_chunks_initialized=4, _count_bytes_stored=553
            )
        else:
            expected = dataclasses.replace(
                expected, _count_chunks_initialized=1, _count_bytes_stored=1178
            )


@pytest.mark.parametrize("store", ["memory"], indirect=True)
def test_resize_1d(store: MemoryStore, zarr_format: ZarrFormat) -> None:
    z = zarr.create(
        shape=105, chunks=10, dtype="i4", fill_value=0, store=store, zarr_format=zarr_format
    )
    a = np.arange(105, dtype="i4")
    z[:] = a
    result = z[:]
    assert isinstance(result, NDArrayLike)
    assert (105,) == z.shape
    assert (105,) == result.shape
    assert np.dtype("i4") == z.dtype
    assert np.dtype("i4") == result.dtype
    assert (10,) == z.chunks
    np.testing.assert_array_equal(a, result)

    z.resize(205)
    result = z[:]
    assert isinstance(result, NDArrayLike)
    assert (205,) == z.shape
    assert (205,) == result.shape
    assert np.dtype("i4") == z.dtype
    assert np.dtype("i4") == result.dtype
    assert (10,) == z.chunks
    np.testing.assert_array_equal(a, z[:105])
    np.testing.assert_array_equal(np.zeros(100, dtype="i4"), z[105:])

    z.resize(55)
    result = z[:]
    assert isinstance(result, NDArrayLike)
    assert (55,) == z.shape
    assert (55,) == result.shape
    assert np.dtype("i4") == z.dtype
    assert np.dtype("i4") == result.dtype
    assert (10,) == z.chunks
    np.testing.assert_array_equal(a[:55], result)

    # via shape setter
    new_shape = (105,)
    z.shape = new_shape
    result = z[:]
    assert isinstance(result, NDArrayLike)
    assert new_shape == z.shape
    assert new_shape == result.shape


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
    result = z[:]
    assert isinstance(result, NDArrayLike)
    assert (105, 105) == z.shape
    assert (105, 105) == result.shape
    assert np.dtype("i4") == z.dtype
    assert np.dtype("i4") == result.dtype
    assert (10, 10) == z.chunks
    np.testing.assert_array_equal(a, result)

    z.resize((205, 205))
    result = z[:]
    assert isinstance(result, NDArrayLike)
    assert (205, 205) == z.shape
    assert (205, 205) == result.shape
    assert np.dtype("i4") == z.dtype
    assert np.dtype("i4") == result.dtype
    assert (10, 10) == z.chunks
    np.testing.assert_array_equal(a, z[:105, :105])
    np.testing.assert_array_equal(np.zeros((100, 205), dtype="i4"), z[105:, :])
    np.testing.assert_array_equal(np.zeros((205, 100), dtype="i4"), z[:, 105:])

    z.resize((55, 55))
    result = z[:]
    assert isinstance(result, NDArrayLike)
    assert (55, 55) == z.shape
    assert (55, 55) == result.shape
    assert np.dtype("i4") == z.dtype
    assert np.dtype("i4") == result.dtype
    assert (10, 10) == z.chunks
    np.testing.assert_array_equal(a[:55, :55], result)

    z.resize((55, 1))
    result = z[:]
    assert isinstance(result, NDArrayLike)
    assert (55, 1) == z.shape
    assert (55, 1) == result.shape
    assert np.dtype("i4") == z.dtype
    assert np.dtype("i4") == result.dtype
    assert (10, 10) == z.chunks
    np.testing.assert_array_equal(a[:55, :1], result)

    z.resize((1, 55))
    result = z[:]
    assert isinstance(result, NDArrayLike)
    assert (1, 55) == z.shape
    assert (1, 55) == result.shape
    assert np.dtype("i4") == z.dtype
    assert np.dtype("i4") == result.dtype
    assert (10, 10) == z.chunks
    np.testing.assert_array_equal(a[:1, :10], z[:, :10])
    np.testing.assert_array_equal(np.zeros((1, 55 - 10), dtype="i4"), z[:, 10:55])

    # via shape setter
    new_shape = (105, 105)
    z.shape = new_shape
    result = z[:]
    assert isinstance(result, NDArrayLike)
    assert new_shape == z.shape
    assert new_shape == result.shape


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


@pytest.mark.parametrize("store", ["memory"], indirect=True)
class TestCreateArray:
    @staticmethod
    def test_chunks_and_shards(store: Store) -> None:
        spath = StorePath(store)
        shape = (100, 100)
        chunks = (5, 5)
        shards = (10, 10)

        arr_v3 = zarr.create_array(store=spath / "v3", shape=shape, chunks=chunks, dtype="i4")
        assert arr_v3.chunks == chunks
        assert arr_v3.shards is None

        arr_v3_sharding = zarr.create_array(
            store=spath / "v3_sharding",
            shape=shape,
            chunks=chunks,
            shards=shards,
            dtype="i4",
        )
        assert arr_v3_sharding.chunks == chunks
        assert arr_v3_sharding.shards == shards

        arr_v2 = zarr.create_array(
            store=spath / "v2", shape=shape, chunks=chunks, zarr_format=2, dtype="i4"
        )
        assert arr_v2.chunks == chunks
        assert arr_v2.shards is None

    @staticmethod
    @pytest.mark.parametrize(
        ("dtype", "fill_value_expected"), [("<U4", ""), ("<S4", b""), ("i", 0), ("f", 0.0)]
    )
    def test_default_fill_value(dtype: str, fill_value_expected: object, store: Store) -> None:
        a = zarr.create_array(store, shape=(5,), chunks=(5,), dtype=dtype)
        assert a.fill_value == fill_value_expected

    @staticmethod
    @pytest.mark.parametrize("dtype", ["uint8", "float32", "str"])
    @pytest.mark.parametrize("empty_value", [None, ()])
    async def test_no_filters_compressors(
        store: MemoryStore, dtype: str, empty_value: object, zarr_format: ZarrFormat
    ) -> None:
        """
        Test that the default ``filters`` and ``compressors`` are removed when ``create_array`` is invoked.
        """

        arr = await create_array(
            store=store,
            dtype=dtype,
            shape=(10,),
            zarr_format=zarr_format,
            compressors=empty_value,
            filters=empty_value,
        )
        # Test metadata explicitly
        if zarr_format == 2:
            assert arr.metadata.zarr_format == 2  # guard for mypy
            # v2 spec requires that filters be either a collection with at least one filter, or None
            assert arr.metadata.filters is None
            # Compressor is a single element in v2 metadata; the absence of a compressor is encoded
            # as None
            assert arr.metadata.compressor is None

            assert arr.filters == ()
            assert arr.compressors == ()
        else:
            assert arr.metadata.zarr_format == 3  # guard for mypy
            if dtype == "str":
                assert arr.metadata.codecs == (VLenUTF8Codec(),)
                assert arr.serializer == VLenUTF8Codec()
            else:
                assert arr.metadata.codecs == (BytesCodec(),)
                assert arr.serializer == BytesCodec()

    @staticmethod
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
    @pytest.mark.parametrize(("chunks", "shards"), [((6,), None), ((3,), (6,))])
    async def test_v3_chunk_encoding(
        store: MemoryStore,
        compressors: CompressorsLike,
        filters: FiltersLike,
        dtype: str,
        chunks: tuple[int, ...],
        shards: tuple[int, ...] | None,
    ) -> None:
        """
        Test various possibilities for the compressors and filters parameter to create_array
        """
        arr = await create_array(
            store=store,
            dtype=dtype,
            shape=(12,),
            chunks=chunks,
            shards=shards,
            zarr_format=3,
            filters=filters,
            compressors=compressors,
        )
        filters_expected, _, compressors_expected = _parse_chunk_encoding_v3(
            filters=filters, compressors=compressors, serializer="auto", dtype=np.dtype(dtype)
        )
        assert arr.filters == filters_expected
        assert arr.compressors == compressors_expected

    @staticmethod
    @pytest.mark.parametrize("name", ["v2", "default", "invalid"])
    @pytest.mark.parametrize("separator", [".", "/"])
    async def test_chunk_key_encoding(
        name: str, separator: Literal[".", "/"], zarr_format: ZarrFormat, store: MemoryStore
    ) -> None:
        chunk_key_encoding = ChunkKeyEncodingParams(name=name, separator=separator)  # type: ignore[typeddict-item]
        error_msg = ""
        if name == "invalid":
            error_msg = "Unknown chunk key encoding."
        if zarr_format == 2 and name == "default":
            error_msg = "Invalid chunk key encoding. For Zarr format 2 arrays, the `name` field of the chunk key encoding must be 'v2'."
        if error_msg:
            with pytest.raises(ValueError, match=re.escape(error_msg)):
                arr = await create_array(
                    store=store,
                    dtype="uint8",
                    shape=(10,),
                    chunks=(1,),
                    zarr_format=zarr_format,
                    chunk_key_encoding=chunk_key_encoding,
                )
        else:
            arr = await create_array(
                store=store,
                dtype="uint8",
                shape=(10,),
                chunks=(1,),
                zarr_format=zarr_format,
                chunk_key_encoding=chunk_key_encoding,
            )
            if isinstance(arr.metadata, ArrayV2Metadata):
                assert arr.metadata.dimension_separator == separator

    @staticmethod
    @pytest.mark.parametrize(
        ("kwargs", "error_msg"),
        [
            ({"serializer": "bytes"}, "Zarr format 2 arrays do not support `serializer`."),
            ({"dimension_names": ["test"]}, "Zarr format 2 arrays do not support dimension names."),
        ],
    )
    async def test_create_array_invalid_v2_arguments(
        kwargs: dict[str, Any], error_msg: str, store: MemoryStore
    ) -> None:
        with pytest.raises(ValueError, match=re.escape(error_msg)):
            await zarr.api.asynchronous.create_array(
                store=store, dtype="uint8", shape=(10,), chunks=(1,), zarr_format=2, **kwargs
            )

    @staticmethod
    @pytest.mark.parametrize(
        ("kwargs", "error_msg"),
        [
            (
                {"dimension_names": ["test"]},
                "dimension_names cannot be used for arrays with zarr_format 2.",
            ),
            (
                {"chunk_key_encoding": {"name": "default", "separator": "/"}},
                "chunk_key_encoding cannot be used for arrays with zarr_format 2. Use dimension_separator instead.",
            ),
            (
                {"codecs": "bytes"},
                "codecs cannot be used for arrays with zarr_format 2. Use filters and compressor instead.",
            ),
        ],
    )
    async def test_create_invalid_v2_arguments(
        kwargs: dict[str, Any], error_msg: str, store: MemoryStore
    ) -> None:
        with pytest.raises(ValueError, match=re.escape(error_msg)):
            await zarr.api.asynchronous.create(
                store=store, dtype="uint8", shape=(10,), chunks=(1,), zarr_format=2, **kwargs
            )

    @staticmethod
    @pytest.mark.parametrize(
        ("kwargs", "error_msg"),
        [
            (
                {"chunk_shape": (1,), "chunks": (2,)},
                "Only one of chunk_shape or chunks can be provided.",
            ),
            (
                {"dimension_separator": "/"},
                "dimension_separator cannot be used for arrays with zarr_format 3. Use chunk_key_encoding instead.",
            ),
            (
                {"filters": []},
                "filters cannot be used for arrays with zarr_format 3. Use array-to-array codecs instead",
            ),
            (
                {"compressor": "blosc"},
                "compressor cannot be used for arrays with zarr_format 3. Use bytes-to-bytes codecs instead",
            ),
        ],
    )
    async def test_invalid_v3_arguments(
        kwargs: dict[str, Any], error_msg: str, store: MemoryStore
    ) -> None:
        kwargs.setdefault("chunks", (1,))
        with pytest.raises(ValueError, match=re.escape(error_msg)):
            zarr.create(store=store, dtype="uint8", shape=(10,), zarr_format=3, **kwargs)

    @staticmethod
    @pytest.mark.parametrize("dtype", ["uint8", "float32", "str", "U10", "S10", ">M8[10s]"])
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
    async def test_v2_chunk_encoding(
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
        assert arr.metadata.zarr_format == 2  # guard for mypy
        assert arr.metadata.compressor == compressor_expected
        assert arr.metadata.filters == filters_expected

        # Normalize for property getters
        compressor_expected = () if compressor_expected is None else (compressor_expected,)
        filters_expected = () if filters_expected is None else filters_expected

        assert arr.compressors == compressor_expected
        assert arr.filters == filters_expected

    @staticmethod
    @pytest.mark.parametrize("dtype", ["uint8", "float32", "str"])
    async def test_default_filters_compressors(
        store: MemoryStore, dtype: str, zarr_format: ZarrFormat
    ) -> None:
        """
        Test that the default ``filters`` and ``compressors`` are used when ``create_array`` is invoked with ``filters`` and ``compressors`` unspecified.
        """
        arr = await create_array(
            store=store,
            dtype=dtype,
            shape=(10,),
            zarr_format=zarr_format,
        )
        if zarr_format == 3:
            expected_filters, expected_serializer, expected_compressors = (
                _get_default_chunk_encoding_v3(np_dtype=np.dtype(dtype))
            )

        elif zarr_format == 2:
            default_filters, default_compressors = _get_default_chunk_encoding_v2(
                np_dtype=np.dtype(dtype)
            )
            if default_filters is None:
                expected_filters = ()
            else:
                expected_filters = default_filters
            if default_compressors is None:
                expected_compressors = ()
            else:
                expected_compressors = (default_compressors,)
            expected_serializer = None
        else:
            raise ValueError(f"Invalid zarr_format: {zarr_format}")

        assert arr.filters == expected_filters
        assert arr.serializer == expected_serializer
        assert arr.compressors == expected_compressors

    @staticmethod
    async def test_v2_no_shards(store: Store) -> None:
        """
        Test that creating a Zarr v2 array with ``shard_shape`` set to a non-None value raises an error.
        """
        msg = re.escape(
            "Zarr format 2 arrays can only be created with `shard_shape` set to `None`. Got `shard_shape=(5,)` instead."
        )
        with pytest.raises(ValueError, match=msg):
            _ = await create_array(
                store=store,
                dtype="uint8",
                shape=(10,),
                shards=(5,),
                zarr_format=2,
            )

    @staticmethod
    @pytest.mark.parametrize("impl", ["sync", "async"])
    async def test_with_data(impl: Literal["sync", "async"], store: Store) -> None:
        """
        Test that we can invoke ``create_array`` with a ``data`` parameter.
        """
        data = np.arange(10)
        name = "foo"
        arr: AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata] | Array
        if impl == "sync":
            arr = sync_api.create_array(store, name=name, data=data)
            stored = arr[:]
        elif impl == "async":
            arr = await create_array(store, name=name, data=data, zarr_format=3)
            stored = await arr._get_selection(
                BasicIndexer(..., shape=arr.shape, chunk_grid=arr.metadata.chunk_grid),
                prototype=default_buffer_prototype(),
            )
        else:
            raise ValueError(f"Invalid impl: {impl}")

        assert np.array_equal(stored, data)

    @staticmethod
    async def test_with_data_invalid_params(store: Store) -> None:
        """
        Test that failing to specify data AND shape / dtype results in a ValueError
        """
        with pytest.raises(ValueError, match="shape was not specified"):
            await create_array(store, data=None, shape=None, dtype=None)

        # we catch shape=None first, so specifying a dtype should raise the same exception as before
        with pytest.raises(ValueError, match="shape was not specified"):
            await create_array(store, data=None, shape=None, dtype="uint8")

        with pytest.raises(ValueError, match="dtype was not specified"):
            await create_array(store, data=None, shape=(10, 10))

    @staticmethod
    async def test_data_ignored_params(store: Store) -> None:
        """
        Test that specifying data AND shape AND dtype results in a ValueError
        """
        data = np.arange(10)
        with pytest.raises(
            ValueError, match="The data parameter was used, but the shape parameter was also used."
        ):
            await create_array(store, data=data, shape=data.shape, dtype=None, overwrite=True)

        # we catch shape first, so specifying a dtype should raise the same warning as before
        with pytest.raises(
            ValueError, match="The data parameter was used, but the shape parameter was also used."
        ):
            await create_array(store, data=data, shape=data.shape, dtype=data.dtype, overwrite=True)

        with pytest.raises(
            ValueError, match="The data parameter was used, but the dtype parameter was also used."
        ):
            await create_array(store, data=data, shape=None, dtype=data.dtype, overwrite=True)

    @staticmethod
    @pytest.mark.parametrize("order", ["C", "F", None])
    @pytest.mark.parametrize("with_config", [True, False])
    def test_order(
        order: MemoryOrder | None,
        with_config: bool,
        zarr_format: ZarrFormat,
        store: MemoryStore,
    ) -> None:
        """
        Test that the arrays generated by array indexing have a memory order defined by the config order
        value, and that for zarr v2 arrays, the ``order`` field in the array metadata is set correctly.
        """
        config: ArrayConfigLike | None = {}
        if order is None:
            config = {}
            expected = zarr.config.get("array.order")
        else:
            config = {"order": order}
            expected = order

        if not with_config:
            # Test without passing config parameter
            config = None

        arr = zarr.create_array(
            store=store,
            shape=(2, 2),
            zarr_format=zarr_format,
            dtype="i4",
            order=order,
            config=config,
        )
        assert arr.order == expected
        if zarr_format == 2:
            assert arr.metadata.zarr_format == 2
            assert arr.metadata.order == expected

        vals = np.asarray(arr)
        if expected == "C":
            assert vals.flags.c_contiguous
        elif expected == "F":
            assert vals.flags.f_contiguous
        else:
            raise AssertionError

    @staticmethod
    @pytest.mark.parametrize("write_empty_chunks", [True, False])
    async def test_write_empty_chunks_config(write_empty_chunks: bool, store: Store) -> None:
        """
        Test that the value of write_empty_chunks is sensitive to the global config when not set
        explicitly
        """
        with zarr.config.set({"array.write_empty_chunks": write_empty_chunks}):
            arr = await create_array(store, shape=(2, 2), dtype="i4")
            assert arr._config.write_empty_chunks == write_empty_chunks

    @staticmethod
    @pytest.mark.parametrize("path", [None, "", "/", "/foo", "foo", "foo/bar"])
    async def test_name(store: Store, zarr_format: ZarrFormat, path: str | None) -> None:
        arr = await create_array(
            store, shape=(2, 2), dtype="i4", name=path, zarr_format=zarr_format
        )
        if path is None:
            expected_path = ""
        elif path.startswith("/"):
            expected_path = path.lstrip("/")
        else:
            expected_path = path
        assert arr.path == expected_path
        assert arr.name == "/" + expected_path

        # test that implicit groups were created
        path_parts = expected_path.split("/")
        if len(path_parts) > 1:
            *parents, _ = ["", *accumulate(path_parts, lambda x, y: "/".join([x, y]))]  # noqa: FLY002
            for parent_path in parents:
                # this will raise if these groups were not created
                _ = await zarr.api.asynchronous.open_group(
                    store=store, path=parent_path, zarr_format=zarr_format
                )


@pytest.mark.parametrize("value", [1, 1.4, "a", b"a", np.array(1)])
@pytest.mark.parametrize("zarr_format", [2, 3])
def test_scalar_array(value: Any, zarr_format: ZarrFormat) -> None:
    arr = zarr.array(value, zarr_format=zarr_format)
    assert arr[...] == value
    assert arr.shape == ()
    assert arr.ndim == 0
    assert isinstance(arr[()], NDArrayLikeOrScalar)


@pytest.mark.parametrize("store", ["local"], indirect=True)
@pytest.mark.parametrize("store2", ["local"], indirect=["store2"])
@pytest.mark.parametrize("src_format", [2, 3])
@pytest.mark.parametrize("new_format", [2, 3, None])
async def test_creation_from_other_zarr_format(
    store: Store,
    store2: Store,
    src_format: ZarrFormat,
    new_format: ZarrFormat | None,
) -> None:
    if src_format == 2:
        src = zarr.create(
            (50, 50), chunks=(10, 10), store=store, zarr_format=src_format, dimension_separator="/"
        )
    else:
        src = zarr.create(
            (50, 50),
            chunks=(10, 10),
            store=store,
            zarr_format=src_format,
            chunk_key_encoding=("default", "."),
        )

    src[:] = np.arange(50 * 50).reshape((50, 50))
    result = zarr.from_array(
        store=store2,
        data=src,
        zarr_format=new_format,
    )
    np.testing.assert_array_equal(result[:], src[:])
    assert result.fill_value == src.fill_value
    assert result.dtype == src.dtype
    assert result.chunks == src.chunks
    expected_format = src_format if new_format is None else new_format
    assert result.metadata.zarr_format == expected_format
    if src_format == new_format:
        assert result.metadata == src.metadata

    result2 = zarr.array(
        data=src,
        store=store2,
        overwrite=True,
        zarr_format=new_format,
    )
    np.testing.assert_array_equal(result2[:], src[:])


@pytest.mark.parametrize("store", ["local", "memory", "zip"], indirect=True)
@pytest.mark.parametrize("store2", ["local", "memory", "zip"], indirect=["store2"])
@pytest.mark.parametrize("src_chunks", [(40, 10), (11, 50)])
@pytest.mark.parametrize("new_chunks", [(40, 10), (11, 50)])
async def test_from_array(
    store: Store,
    store2: Store,
    src_chunks: tuple[int, int],
    new_chunks: tuple[int, int],
    zarr_format: ZarrFormat,
) -> None:
    src_fill_value = 2
    src_dtype = np.dtype("uint8")
    src_attributes = None

    src = zarr.create(
        (100, 10),
        chunks=src_chunks,
        dtype=src_dtype,
        store=store,
        fill_value=src_fill_value,
        attributes=src_attributes,
    )
    src[:] = np.arange(1000).reshape((100, 10))

    new_fill_value = 3
    new_attributes: dict[str, JSON] = {"foo": "bar"}

    result = zarr.from_array(
        data=src,
        store=store2,
        chunks=new_chunks,
        fill_value=new_fill_value,
        attributes=new_attributes,
    )

    np.testing.assert_array_equal(result[:], src[:])
    assert result.fill_value == new_fill_value
    assert result.dtype == src_dtype
    assert result.attrs == new_attributes
    assert result.chunks == new_chunks


@pytest.mark.parametrize("store", ["local"], indirect=True)
@pytest.mark.parametrize("chunks", ["keep", "auto"])
@pytest.mark.parametrize("write_data", [True, False])
@pytest.mark.parametrize(
    "src",
    [
        np.arange(1000).reshape(10, 10, 10),
        zarr.ones((10, 10, 10)),
        5,
        [1, 2, 3],
        [[1, 2, 3], [4, 5, 6]],
    ],
)  # add other npt.ArrayLike?
async def test_from_array_arraylike(
    store: Store,
    chunks: Literal["auto", "keep"] | tuple[int, int],
    write_data: bool,
    src: Array | npt.ArrayLike,
) -> None:
    fill_value = 42
    result = zarr.from_array(
        store, data=src, chunks=chunks, write_data=write_data, fill_value=fill_value
    )
    if write_data:
        np.testing.assert_array_equal(result[...], np.array(src))
    else:
        np.testing.assert_array_equal(result[...], np.full_like(src, fill_value))


async def test_orthogonal_set_total_slice() -> None:
    """Ensure that a whole chunk overwrite does not read chunks"""
    store = MemoryStore()
    array = zarr.create_array(store, shape=(20, 20), chunks=(1, 2), dtype=int, fill_value=-1)
    with mock.patch("zarr.storage.MemoryStore.get", side_effect=RuntimeError):
        array[0, slice(4, 10)] = np.arange(6)

    array = zarr.create_array(
        store, shape=(20, 21), chunks=(1, 2), dtype=int, fill_value=-1, overwrite=True
    )
    with mock.patch("zarr.storage.MemoryStore.get", side_effect=RuntimeError):
        array[0, :] = np.arange(21)

    with mock.patch("zarr.storage.MemoryStore.get", side_effect=RuntimeError):
        array[:] = 1


@pytest.mark.skipif(
    Version(numcodecs.__version__) < Version("0.15.1"),
    reason="codec configuration is overwritten on older versions. GH2800",
)
def test_roundtrip_numcodecs() -> None:
    store = MemoryStore()

    compressors = [
        {"name": "numcodecs.shuffle", "configuration": {"elementsize": 2}},
        {"name": "numcodecs.zlib", "configuration": {"level": 4}},
    ]
    filters = [
        {
            "name": "numcodecs.fixedscaleoffset",
            "configuration": {
                "scale": 100.0,
                "offset": 0.0,
                "dtype": "<f8",
                "astype": "<i2",
            },
        },
    ]

    # Create the array with the correct codecs
    root = zarr.group(store)
    root.create_array(
        "test",
        shape=(720, 1440),
        chunks=(720, 1440),
        dtype="float64",
        compressors=compressors,
        filters=filters,
        fill_value=-9.99,
        dimension_names=["lat", "lon"],
    )

    BYTES_CODEC = {"name": "bytes", "configuration": {"endian": "little"}}
    # Read in the array again and check compressor config
    root = zarr.open_group(store)
    metadata = root["test"].metadata.to_dict()
    expected = (*filters, BYTES_CODEC, *compressors)
    assert metadata["codecs"] == expected


def _index_array(arr: Array, index: Any) -> Any:
    return arr[index]


@pytest.mark.parametrize(
    "method",
    [
        pytest.param(
            "fork",
            marks=pytest.mark.skipif(
                sys.platform in ("win32", "darwin"), reason="fork not supported on Windows or OSX"
            ),
        ),
        "spawn",
        pytest.param(
            "forkserver",
            marks=pytest.mark.skipif(
                sys.platform == "win32", reason="forkserver not supported on Windows"
            ),
        ),
    ],
)
@pytest.mark.parametrize("store", ["local"], indirect=True)
def test_multiprocessing(store: Store, method: Literal["fork", "spawn", "forkserver"]) -> None:
    """
    Test that arrays can be pickled and indexed in child processes
    """
    data = np.arange(100)
    arr = zarr.create_array(store=store, data=data)
    ctx = mp.get_context(method)
    with ctx.Pool() as pool:
        results = pool.starmap(_index_array, [(arr, slice(len(data)))])
    assert all(np.array_equal(r, data) for r in results)


async def test_sharding_coordinate_selection() -> None:
    store = MemoryStore()
    g = zarr.open_group(store, mode="w")
    arr = g.create_array(
        name="a",
        shape=(2, 3, 4),
        chunks=(1, 2, 2),
        overwrite=True,
        dtype=np.float32,
        shards=(2, 4, 4),
    )
    arr[:] = np.arange(2 * 3 * 4).reshape((2, 3, 4))
    result = arr[1, [0, 1]]  # type: ignore[index]
    assert isinstance(result, NDArrayLike)
    assert (result == np.array([[12, 13, 14, 15], [16, 17, 18, 19]])).all()


@pytest.mark.parametrize("store", ["local", "memory", "zip"], indirect=["store"])
def test_array_repr(store: Store) -> None:
    shape = (2, 3, 4)
    dtype = "uint8"
    arr = zarr.create_array(store, shape=shape, dtype=dtype)
    assert str(arr) == f"<Array {store} shape={shape} dtype={dtype}>"
