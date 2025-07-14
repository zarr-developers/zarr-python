import json
from pathlib import Path
from typing import Any, Literal

import numcodecs.abc
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
from zarr.core.dtype import FixedLengthUTF32, Structured, VariableLengthUTF8
from zarr.core.dtype.npy.bytes import NullTerminatedBytes
from zarr.core.dtype.wrapper import ZDType
from zarr.core.group import Group
from zarr.core.sync import sync
from zarr.storage import MemoryStore, StorePath


@pytest.fixture
async def store() -> StorePath:
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
    ("dtype", "expected_dtype", "fill_value", "fill_value_json"),
    [
        ("|S1", "|S1", b"X", "WA=="),
        ("|V1", "|V1", b"X", "WA=="),
        ("|V10", "|V10", b"X", "WAAAAAAAAAAAAA=="),
    ],
)
async def test_v2_encode_decode(
    dtype: str, expected_dtype: str, fill_value: bytes, fill_value_json: str
) -> None:
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
        "fill_value": fill_value_json,
        "filters": None,
        "order": "C",
        "shape": [3],
        "zarr_format": 2,
        "dimension_separator": ".",
    }
    assert serialized == expected

    data = zarr.open_array(store=store, path="foo")[:]
    np.testing.assert_equal(data, np.full((3,), b"X", dtype=dtype))

    data = zarr.open_array(store=store, path="foo")[:]
    np.testing.assert_equal(data, np.full((3,), b"X", dtype=dtype))


@pytest.mark.parametrize(
    ("dtype", "value"),
    [
        (NullTerminatedBytes(length=1), b"Y"),
        (FixedLengthUTF32(length=1), "Y"),
        (VariableLengthUTF8(), "Y"),
    ],
)
def test_v2_encode_decode_with_data(dtype: ZDType[Any, Any], value: str) -> None:
    expected = np.full((3,), value, dtype=dtype.to_native_dtype())
    a = zarr.create(
        shape=(3,),
        zarr_format=2,
        dtype=dtype,
    )
    a[:] = expected
    data = a[:]
    np.testing.assert_equal(data, expected)


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
def test_create_array_defaults(store: Store) -> None:
    """
    Test that passing compressor=None results in no compressor. Also test that the default value of the compressor
    parameter does produce a compressor.
    """
    g = zarr.open(store, mode="w", zarr_format=2)
    assert isinstance(g, Group)
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

    buf = sync(store.get("2.1", default_buffer_prototype()))
    assert buf is not None
    np.testing.assert_array_equal(
        a[6:9, 3:6],
        np.frombuffer(buf.to_bytes(), dtype="float64").reshape((3, 3), order=zarr_order),
    )
    # After writing and reading from zarr array, order should be same as zarr order
    sub_arr = arr[6:9, 3:6]
    assert isinstance(sub_arr, np.ndarray)
    if zarr_order == "F":
        assert (sub_arr).flags.f_contiguous
    else:
        assert (sub_arr).flags.c_contiguous

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
    sub_arr = arr[6:9, 3:6]
    assert isinstance(sub_arr, np.ndarray)
    if zarr_order == "F":
        assert (sub_arr).flags.f_contiguous
    else:
        assert (sub_arr).flags.c_contiguous


def test_default_compressor_deprecation_warning() -> None:
    with pytest.warns(DeprecationWarning, match="default_compressor is deprecated"):
        zarr.storage.default_compressor = "zarr.codecs.zstd.ZstdCodec()"  # type: ignore[attr-defined]


@pytest.mark.parametrize("fill_value", [None, (b"", 0, 0.0)], ids=["no_fill", "fill"])
def test_structured_dtype_roundtrip(fill_value: float | bytes, tmp_path: Path) -> None:
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
    ],
    ids=[
        "tuple_input",
        "list_input",
        "bytes_input",
    ],
)
def test_parse_structured_fill_value_valid(
    fill_value: Any, dtype: np.dtype[Any], expected_result: Any
) -> None:
    zdtype = Structured.from_native_dtype(dtype)
    result = zdtype.cast_scalar(fill_value)
    assert result.dtype == expected_result.dtype
    assert result == expected_result
    if isinstance(expected_result, np.void):
        for name in expected_result.dtype.names or []:
            assert result[name] == expected_result[name]


@pytest.mark.parametrize("fill_value", [None, b"x"], ids=["no_fill", "fill"])
def test_other_dtype_roundtrip(fill_value: None | bytes, tmp_path: Path) -> None:
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
