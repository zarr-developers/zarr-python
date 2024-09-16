import pickle
from typing import Literal

import numpy as np
import pytest

from zarr import Array, AsyncArray, Group
from zarr.core.buffer.cpu import NDBuffer
from zarr.core.common import ZarrFormat
from zarr.errors import ContainsArrayError, ContainsGroupError
from zarr.store import LocalStore, MemoryStore
from zarr.store.common import StorePath


@pytest.mark.parametrize("store", ("local", "memory", "zip"), indirect=["store"])
@pytest.mark.parametrize("zarr_format", (2, 3))
@pytest.mark.parametrize("exists_ok", [True, False])
@pytest.mark.parametrize("extant_node", ["array", "group"])
def test_array_creation_existing_node(
    store: LocalStore | MemoryStore,
    zarr_format: ZarrFormat,
    exists_ok: bool,
    extant_node: Literal["array", "group"],
) -> None:
    """
    Check that an existing array or group is handled as expected during array creation.
    """
    spath = StorePath(store)
    group = Group.from_store(spath, zarr_format=zarr_format)
    expected_exception: type[ContainsArrayError] | type[ContainsGroupError]
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

    if exists_ok:
        arr_new = Array.create(
            spath / "extant",
            shape=new_shape,
            dtype=new_dtype,
            exists_ok=exists_ok,
            zarr_format=zarr_format,
        )
        assert arr_new.shape == new_shape
        assert arr_new.dtype == new_dtype
    else:
        with pytest.raises(expected_exception):
            arr_new = Array.create(
                spath / "extant",
                shape=new_shape,
                dtype=new_dtype,
                exists_ok=exists_ok,
                zarr_format=zarr_format,
            )


@pytest.mark.parametrize("store", ("local", "memory", "zip"), indirect=["store"])
@pytest.mark.parametrize("zarr_format", (2, 3))
def test_array_name_properties_no_group(
    store: LocalStore | MemoryStore, zarr_format: ZarrFormat
) -> None:
    arr = Array.create(store=store, shape=(100,), chunks=(10,), zarr_format=zarr_format, dtype="i4")
    assert arr.path == ""
    assert arr.name is None
    assert arr.basename is None


@pytest.mark.parametrize("store", ("local", "memory", "zip"), indirect=["store"])
@pytest.mark.parametrize("zarr_format", (2, 3))
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
        arr = Array.create(
            store=store,
            shape=shape,
            dtype=dtype_str,
            zarr_format=3,
            chunk_shape=shape,
            fill_value=None,
        )
    else:
        arr = Array.create(
            store=store, shape=shape, dtype=dtype_str, zarr_format=3, chunk_shape=shape
        )

    assert arr.fill_value == np.dtype(dtype_str).type(default_fill_value)
    assert arr.fill_value.dtype == arr.dtype


@pytest.mark.parametrize("store", ["memory"], indirect=True)
@pytest.mark.parametrize("fill_value", [False, 0.0, 1, 2.3])
@pytest.mark.parametrize("dtype_str", ["bool", "uint8", "float32", "complex64"])
def test_array_v3_fill_value(store: MemoryStore, fill_value: int, dtype_str: str) -> None:
    shape = (10,)
    arr = Array.create(
        store=store,
        shape=shape,
        dtype=dtype_str,
        zarr_format=3,
        chunk_shape=shape,
        fill_value=fill_value,
    )

    assert arr.fill_value == np.dtype(dtype_str).type(fill_value)
    assert arr.fill_value.dtype == arr.dtype


def test_create_positional_args_deprecated() -> None:
    store = MemoryStore({}, mode="w")
    with pytest.warns(FutureWarning, match="Pass"):
        Array.create(store, (2, 2), dtype="f8")


def test_selection_positional_args_deprecated() -> None:
    store = MemoryStore({}, mode="w")
    arr = Array.create(store, shape=(2, 2), dtype="f8")

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


@pytest.mark.parametrize("store", ("local",), indirect=["store"])
@pytest.mark.parametrize("zarr_format", (2, 3))
async def test_serializable_async_array(
    store: LocalStore | MemoryStore, zarr_format: ZarrFormat
) -> None:
    expected = await AsyncArray.create(
        store=store, shape=(100,), chunks=(10,), zarr_format=zarr_format, dtype="i4"
    )
    # await expected.setitems(list(range(100)))

    p = pickle.dumps(expected)
    actual = pickle.loads(p)

    assert actual == expected
    # np.testing.assert_array_equal(await actual.getitem(slice(None)), await expected.getitem(slice(None)))
    # TODO: uncomment the parts of this test that will be impacted by the config/prototype changes in flight


@pytest.mark.parametrize("store", ("local",), indirect=["store"])
@pytest.mark.parametrize("zarr_format", (2, 3))
def test_serializable_sync_array(store: LocalStore, zarr_format: ZarrFormat) -> None:
    expected = Array.create(
        store=store, shape=(100,), chunks=(10,), zarr_format=zarr_format, dtype="i4"
    )
    expected[:] = list(range(100))

    p = pickle.dumps(expected)
    actual = pickle.loads(p)

    assert actual == expected
    np.testing.assert_array_equal(actual[:], expected[:])
