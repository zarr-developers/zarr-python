import json
from collections.abc import Iterator
from typing import Any, Literal

import numcodecs.vlen
import numpy as np
import pytest
from numcodecs import Delta
from numcodecs.blosc import Blosc

import zarr
import zarr.core.buffer
import zarr.storage
from zarr import config
from zarr.storage import MemoryStore, StorePath


@pytest.fixture
async def store() -> Iterator[StorePath]:
    return StorePath(await MemoryStore.open())


def test_simple(store: StorePath) -> None:
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    a = zarr.create_array(
        store / "simple_v2",
        zarr_format=2,
        shape=data.shape,
        chunks=(16, 16),
        dtype=data.dtype,
        fill_value=0,
    )

    a[:, :] = data
    assert np.array_equal(data, a[:, :])


@pytest.mark.parametrize("store", ["memory"], indirect=True)
@pytest.mark.parametrize(
    ("dtype", "fill_value"),
    [
        ("bool", False),
        ("int64", 0),
        ("float64", 0.0),
        ("|S1", b""),
        ("|U1", ""),
        ("object", ""),
        (str, ""),
    ],
)
def test_implicit_fill_value(store: MemoryStore, dtype: str, fill_value: Any) -> None:
    arr = zarr.create(store=store, shape=(4,), fill_value=None, zarr_format=2, dtype=dtype)
    assert arr.metadata.fill_value is None
    assert arr.metadata.to_dict()["fill_value"] is None
    result = arr[:]
    if dtype is str:
        # special case
        numpy_dtype = np.dtype(object)
    else:
        numpy_dtype = np.dtype(dtype)
    expected = np.full(arr.shape, fill_value, dtype=numpy_dtype)
    np.testing.assert_array_equal(result, expected)


def test_codec_pipeline() -> None:
    # https://github.com/zarr-developers/zarr-python/issues/2243
    store = MemoryStore()
    array = zarr.create(
        store=store,
        shape=(1,),
        dtype="i4",
        zarr_format=2,
        filters=[Delta(dtype="i4").get_config()],
        compressor=Blosc().get_config(),
    )
    array[:] = 1
    result = array[:]
    expected = np.ones(1)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("dtype", ["|S", "|V"])
async def test_v2_encode_decode(dtype):
    with config.set(
        {
            "array.v2_default_filters.bytes": [{"id": "vlen-bytes"}],
            "array.v2_default_compressor.bytes": None,
        }
    ):
        store = zarr.storage.MemoryStore()
        g = zarr.group(store=store, zarr_format=2)
        g.create_array(
            name="foo",
            shape=(3,),
            chunks=(3,),
            dtype=dtype,
            fill_value=b"X",
        )

        result = await store.get("foo/.zarray", zarr.core.buffer.default_buffer_prototype())
        assert result is not None

        serialized = json.loads(result.to_bytes())
        expected = {
            "chunks": [3],
            "compressor": None,
            "dtype": f"{dtype}0",
            "fill_value": "WA==",
            "filters": [{"id": "vlen-bytes"}],
            "order": "C",
            "shape": [3],
            "zarr_format": 2,
            "dimension_separator": ".",
        }
        assert serialized == expected

        data = zarr.open_array(store=store, path="foo")[:]
        expected = np.full((3,), b"X", dtype=dtype)
        np.testing.assert_equal(data, expected)


@pytest.mark.parametrize("dtype_value", [["|S", b"Y"], ["|U", "Y"], ["O", b"Y"]])
def test_v2_encode_decode_with_data(dtype_value):
    dtype, value = dtype_value
    with config.set(
        {
            "array.v2_default_filters": {
                "string": [{"id": "vlen-utf8"}],
                "bytes": [{"id": "vlen-bytes"}],
            },
        }
    ):
        expected = np.full((3,), value, dtype=dtype)
        a = zarr.create(
            shape=(3,),
            zarr_format=2,
            dtype=dtype,
        )
        a[:] = expected
        data = a[:]
        np.testing.assert_equal(data, expected)


@pytest.mark.parametrize("dtype", [str, "str"])
async def test_create_dtype_str(dtype: Any) -> None:
    arr = zarr.create(shape=3, dtype=dtype, zarr_format=2)
    assert arr.dtype.kind == "O"
    assert arr.metadata.to_dict()["dtype"] == "|O"
    assert arr.metadata.filters == (numcodecs.vlen.VLenBytes(),)
    arr[:] = [b"a", b"bb", b"ccc"]
    result = arr[:]
    np.testing.assert_array_equal(result, np.array([b"a", b"bb", b"ccc"], dtype="object"))


@pytest.mark.parametrize("filters", [[], [numcodecs.Delta(dtype="<i4")], [numcodecs.Zlib(level=2)]])
@pytest.mark.parametrize("order", ["C", "F"])
def test_v2_filters_codecs(filters: Any, order: Literal["C", "F"]) -> None:
    array_fixture = [42]
    with config.set({"array.order": order}):
        arr = zarr.create(shape=1, dtype="<i4", zarr_format=2, filters=filters)
    arr[:] = array_fixture
    result = arr[:]
    np.testing.assert_array_equal(result, array_fixture)


@pytest.mark.parametrize("array_order", ["C", "F"])
@pytest.mark.parametrize("data_order", ["C", "F"])
def test_v2_non_contiguous(array_order: Literal["C", "F"], data_order: Literal["C", "F"]) -> None:
    arr = zarr.create_array(
        MemoryStore({}),
        shape=(10, 8),
        chunks=(3, 3),
        fill_value=np.nan,
        dtype="float64",
        zarr_format=2,
        overwrite=True,
        order=array_order,
    )

    # Non-contiguous write
    a = np.arange(arr.shape[0] * arr.shape[1]).reshape(arr.shape, order=data_order)
    arr[slice(6, 9, None), slice(3, 6, None)] = a[
        slice(6, 9, None), slice(3, 6, None)
    ]  # The slice on the RHS is important
    np.testing.assert_array_equal(
        arr[slice(6, 9, None), slice(3, 6, None)], a[slice(6, 9, None), slice(3, 6, None)]
    )

    arr = zarr.create_array(
        MemoryStore({}),
        shape=(10, 8),
        chunks=(3, 3),
        fill_value=np.nan,
        dtype="float64",
        zarr_format=2,
        overwrite=True,
        order=array_order,
    )

    # Contiguous write
    a = np.arange(9).reshape((3, 3), order=data_order)
    if data_order == "F":
        assert a.flags.f_contiguous
    else:
        assert a.flags.c_contiguous
    arr[slice(6, 9, None), slice(3, 6, None)] = a
    np.testing.assert_array_equal(arr[slice(6, 9, None), slice(3, 6, None)], a)


def test_default_compressor_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="default_compressor is deprecated"):
        zarr.storage.default_compressor = "zarr.codecs.zstd.ZstdCodec()"


@pytest.mark.parametrize(
    "dtype_expected",
    [
        ["b", "zstd", None],
        ["i", "zstd", None],
        ["f", "zstd", None],
        ["|S1", "zstd", "vlen-bytes"],
        ["|U1", "zstd", "vlen-utf8"],
    ],
)
def test_default_filters_and_compressor(dtype_expected: Any) -> None:
    with config.set(
        {
            "array.v2_default_compressor": {
                "numeric": {"id": "zstd", "level": "0"},
                "string": {"id": "zstd", "level": "0"},
                "bytes": {"id": "zstd", "level": "0"},
            },
            "array.v2_default_filters": {
                "numeric": [],
                "string": [{"id": "vlen-utf8"}],
                "bytes": [{"id": "vlen-bytes"}],
            },
        }
    ):
        dtype, expected_compressor, expected_filter = dtype_expected
        arr = zarr.create(shape=(3,), path="foo", store={}, zarr_format=2, dtype=dtype)
        assert arr.metadata.compressor.codec_id == expected_compressor
        if expected_filter is not None:
            assert arr.metadata.filters[0].codec_id == expected_filter
