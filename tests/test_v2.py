import json
from collections.abc import Iterator
from typing import Any, Literal

import numcodecs.vlen
import numpy as np
import pytest
from numcodecs import Delta
from numcodecs.blosc import Blosc
from numcodecs.zstd import Zstd

import zarr
import zarr.core.buffer
import zarr.storage
from zarr import config
from zarr.abc.store import Store
from zarr.core.buffer.core import default_buffer_prototype
from zarr.core.metadata.v2 import _parse_structured_fill_value
from zarr.core.sync import sync
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


@pytest.mark.parametrize(
    ("dtype", "expected_dtype", "fill_value", "fill_value_encoding"),
    [
        ("|S", "|S0", b"X", "WA=="),
        ("|V", "|V0", b"X", "WA=="),
        ("|V10", "|V10", b"X", "WAAAAAAAAAAAAA=="),
    ],
)
async def test_v2_encode_decode(dtype, expected_dtype, fill_value, fill_value_encoding) -> None:
    with config.set(
        {
            "array.v2_default_filters.bytes": [{"id": "vlen-bytes"}],
            "array.v2_default_compressor.bytes": None,
        }
    ):
        store = zarr.storage.MemoryStore()
        g = zarr.group(store=store, zarr_format=2)
        g.create_array(
            name="foo", shape=(3,), chunks=(3,), dtype=dtype, fill_value=fill_value, compressor=None
        )

        result = await store.get("foo/.zarray", zarr.core.buffer.default_buffer_prototype())
        assert result is not None

        serialized = json.loads(result.to_bytes())
        expected = {
            "chunks": [3],
            "compressor": None,
            "dtype": expected_dtype,
            "fill_value": fill_value_encoding,
            "filters": [{"id": "vlen-bytes"}] if dtype == "|S" else None,
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


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("store", ["memory"], indirect=True)
def test_create_array_defaults(store: Store):
    """
    Test that passing compressor=None results in no compressor. Also test that the default value of the compressor
    parameter does produce a compressor.
    """
    g = zarr.open(store, mode="w", zarr_format=2)
    arr = g.create_array("one", dtype="i8", shape=(1,), chunks=(1,), compressor=None)
    assert arr._async_array.compressor is None
    assert not (arr.filters)
    arr = g.create_array("two", dtype="i8", shape=(1,), chunks=(1,))
    assert arr._async_array.compressor is not None
    assert not (arr.filters)
    arr = g.create_array("three", dtype="i8", shape=(1,), chunks=(1,), compressor=Zstd())
    assert arr._async_array.compressor is not None
    assert not (arr.filters)
    with pytest.raises(ValueError):
        g.create_array(
            "four", dtype="i8", shape=(1,), chunks=(1,), compressor=None, compressors=None
        )


@pytest.mark.parametrize("numpy_order", ["C", "F"])
@pytest.mark.parametrize("zarr_order", ["C", "F"])
def test_v2_non_contiguous(numpy_order: Literal["C", "F"], zarr_order: Literal["C", "F"]) -> None:
    """
    Make sure zarr v2 arrays save data using the memory order given to the zarr array,
    not the memory order of the original numpy array.
    """
    store = MemoryStore()
    arr = zarr.create_array(
        store,
        shape=(10, 8),
        chunks=(3, 3),
        fill_value=np.nan,
        dtype="float64",
        zarr_format=2,
        filters=None,
        compressors=None,
        overwrite=True,
        order=zarr_order,
    )

    # Non-contiguous write, using numpy memory order
    a = np.arange(arr.shape[0] * arr.shape[1]).reshape(arr.shape, order=numpy_order)
    arr[6:9, 3:6] = a[6:9, 3:6]  # The slice on the RHS is important
    np.testing.assert_array_equal(arr[6:9, 3:6], a[6:9, 3:6])

    np.testing.assert_array_equal(
        a[6:9, 3:6],
        np.frombuffer(
            sync(store.get("2.1", default_buffer_prototype())).to_bytes(), dtype="float64"
        ).reshape((3, 3), order=zarr_order),
    )
    # After writing and reading from zarr array, order should be same as zarr order
    if zarr_order == "F":
        assert (arr[6:9, 3:6]).flags.f_contiguous
    else:
        assert (arr[6:9, 3:6]).flags.c_contiguous

    # Contiguous write
    store = MemoryStore()
    arr = zarr.create_array(
        store,
        shape=(10, 8),
        chunks=(3, 3),
        fill_value=np.nan,
        dtype="float64",
        zarr_format=2,
        compressors=None,
        filters=None,
        overwrite=True,
        order=zarr_order,
    )

    a = np.arange(9).reshape((3, 3), order=numpy_order)
    arr[6:9, 3:6] = a
    np.testing.assert_array_equal(arr[6:9, 3:6], a)
    # After writing and reading from zarr array, order should be same as zarr order
    if zarr_order == "F":
        assert (arr[6:9, 3:6]).flags.f_contiguous
    else:
        assert (arr[6:9, 3:6]).flags.c_contiguous


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


@pytest.mark.parametrize("fill_value", [None, (b"", 0, 0.0)], ids=["no_fill", "fill"])
def test_structured_dtype_roundtrip(fill_value, tmp_path) -> None:
    a = np.array(
        [(b"aaa", 1, 4.2), (b"bbb", 2, 8.4), (b"ccc", 3, 12.6)],
        dtype=[("foo", "S3"), ("bar", "i4"), ("baz", "f8")],
    )
    array_path = tmp_path / "data.zarr"
    za = zarr.create(
        shape=(3,),
        store=array_path,
        chunks=(2,),
        fill_value=fill_value,
        zarr_format=2,
        dtype=a.dtype,
    )
    if fill_value is not None:
        assert (np.array([fill_value] * a.shape[0], dtype=a.dtype) == za[:]).all()
    za[...] = a
    za = zarr.open_array(store=array_path)
    assert (a == za[:]).all()


@pytest.mark.parametrize(
    (
        "fill_value",
        "dtype",
        "expected_result",
    ),
    [
        (
            ("Alice", 30),
            np.dtype([("name", "U10"), ("age", "i4")]),
            np.array([("Alice", 30)], dtype=[("name", "U10"), ("age", "i4")])[0],
        ),
        (
            ["Bob", 25],
            np.dtype([("name", "U10"), ("age", "i4")]),
            np.array([("Bob", 25)], dtype=[("name", "U10"), ("age", "i4")])[0],
        ),
        (
            b"\x01\x00\x00\x00\x02\x00\x00\x00",
            np.dtype([("x", "i4"), ("y", "i4")]),
            np.array([(1, 2)], dtype=[("x", "i4"), ("y", "i4")])[0],
        ),
        (
            "BQAAAA==",
            np.dtype([("val", "i4")]),
            np.array([(5,)], dtype=[("val", "i4")])[0],
        ),
        (
            {"x": 1, "y": 2},
            np.dtype([("location", "O")]),
            np.array([({"x": 1, "y": 2},)], dtype=[("location", "O")])[0],
        ),
        (
            {"x": 1, "y": 2, "z": 3},
            np.dtype([("location", "O")]),
            np.array([({"x": 1, "y": 2, "z": 3},)], dtype=[("location", "O")])[0],
        ),
    ],
    ids=[
        "tuple_input",
        "list_input",
        "bytes_input",
        "string_input",
        "dictionary_input",
        "dictionary_input_extra_fields",
    ],
)
def test_parse_structured_fill_value_valid(
    fill_value: Any, dtype: np.dtype[Any], expected_result: Any
) -> None:
    result = _parse_structured_fill_value(fill_value, dtype)
    assert result.dtype == expected_result.dtype
    assert result == expected_result
    if isinstance(expected_result, np.void):
        for name in expected_result.dtype.names or []:
            assert result[name] == expected_result[name]


@pytest.mark.parametrize(
    (
        "fill_value",
        "dtype",
    ),
    [
        (("Alice", 30), np.dtype([("name", "U10"), ("age", "i4"), ("city", "U20")])),
        (b"\x01\x00\x00\x00", np.dtype([("x", "i4"), ("y", "i4")])),
        ("this_is_not_base64", np.dtype([("val", "i4")])),
        ("hello", np.dtype([("age", "i4")])),
        ({"x": 1, "y": 2}, np.dtype([("location", "i4")])),
    ],
    ids=[
        "tuple_list_wrong_length",
        "bytes_wrong_length",
        "invalid_base64",
        "wrong_data_type",
        "wrong_dictionary",
    ],
)
def test_parse_structured_fill_value_invalid(fill_value: Any, dtype: np.dtype[Any]) -> None:
    with pytest.raises(ValueError):
        _parse_structured_fill_value(fill_value, dtype)


@pytest.mark.parametrize("fill_value", [None, b"x"], ids=["no_fill", "fill"])
def test_other_dtype_roundtrip(fill_value, tmp_path) -> None:
    a = np.array([b"a\0\0", b"bb", b"ccc"], dtype="V7")
    array_path = tmp_path / "data.zarr"
    za = zarr.create(
        shape=(3,),
        store=array_path,
        chunks=(2,),
        fill_value=fill_value,
        zarr_format=2,
        dtype=a.dtype,
    )
    if fill_value is not None:
        assert (np.array([fill_value] * a.shape[0], dtype=a.dtype) == za[:]).all()
    za[...] = a
    za = zarr.open_array(store=array_path)
    assert (a == za[:]).all()
