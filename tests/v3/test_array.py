from typing import Literal

import numpy as np
import pytest

from zarr.array import Array, nchunks_initialized
from zarr.common import ZarrFormat
from zarr.errors import ContainsArrayError, ContainsGroupError
from zarr.group import Group
from zarr.store import LocalStore, MemoryStore
from zarr.store.core import StorePath
from zarr.sync import sync


@pytest.mark.parametrize("store", ("local", "memory"), indirect=True)
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
    group = Group.create(spath, zarr_format=zarr_format)
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


@pytest.mark.parametrize("store", ("local", "memory"), indirect=True)
def test_array_name_properties_no_group(
    store: LocalStore | MemoryStore, zarr_format: ZarrFormat
) -> None:
    arr = Array.create(store=store, shape=(100,), chunks=(10,), zarr_format=zarr_format, dtype="i4")
    assert arr.path == ""
    assert arr.name is None
    assert arr.basename is None


@pytest.mark.parametrize("store", ("local", "memory"), indirect=True)
def test_array_name_properties_with_group(
    store: LocalStore | MemoryStore, zarr_format: ZarrFormat
) -> None:
    root = Group.create(store=store, zarr_format=zarr_format)
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


@pytest.mark.parametrize("store", ("local", "memory"), indirect=True)
def test_nchunks_initialized(store: LocalStore | MemoryStore, zarr_format: ZarrFormat) -> None:
    """
    Test that the nchunks_initialized function accurately reports the number of initialized chunks
    in storage
    """
    num_chunks = 10
    array = Array.create(
        store=store, shape=(num_chunks,), chunks=(1,), dtype="uint8", zarr_format=zarr_format
    )
    assert array.nchunks == num_chunks
    assert nchunks_initialized(array) == 0

    for idx, region in enumerate(array._iter_chunk_regions):
        array[region] = 1
        assert nchunks_initialized(array) == idx + 1

    for idx, key in enumerate(array._iter_chunk_keys):
        sync((array.store_path / key).delete())
        assert nchunks_initialized(array) == array.nchunks - (idx + 1)
