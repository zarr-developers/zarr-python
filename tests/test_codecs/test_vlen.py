from typing import Any

import numpy as np
import pytest

from zarr import Array
from zarr.abc.codec import Codec
from zarr.abc.store import Store
from zarr.codecs import VLenBytesCodec, VLenUTF8Codec, ZstdCodec
from zarr.core.metadata.v3 import ArrayV3Metadata, DataType
from zarr.core.strings import _NUMPY_SUPPORTS_VLEN_STRING
from zarr.storage.common import StorePath

numpy_str_dtypes: list[type | str | None] = [None, str, "str", np.dtypes.StrDType]
expected_zarr_string_dtype: np.dtype[Any]
if _NUMPY_SUPPORTS_VLEN_STRING:
    numpy_str_dtypes.append(np.dtypes.StringDType)
    expected_zarr_string_dtype = np.dtypes.StringDType()
else:
    expected_zarr_string_dtype = np.dtype("O")


@pytest.mark.parametrize("store", ["memory", "local"], indirect=["store"])
@pytest.mark.parametrize("dtype", numpy_str_dtypes)
@pytest.mark.parametrize("as_object_array", [False, True])
@pytest.mark.parametrize("codecs", [None, [VLenUTF8Codec()], [VLenUTF8Codec(), ZstdCodec()]])
def test_vlen_string(
    store: Store, dtype: None | np.dtype[Any], as_object_array: bool, codecs: None | list[Codec]
) -> None:
    strings = ["hello", "world", "this", "is", "a", "test"]
    data = np.array(strings, dtype=dtype).reshape((2, 3))

    sp = StorePath(store, path="string")
    a = Array.create(
        sp,
        shape=data.shape,
        chunk_shape=data.shape,
        dtype=data.dtype,
        fill_value="",
        codecs=codecs,
    )
    assert isinstance(a.metadata, ArrayV3Metadata)  # needed for mypy

    # should also work if input array is an object array, provided we explicitly specified
    # a stringlike dtype when creating the Array
    if as_object_array:
        data = data.astype("O")

    a[:, :] = data
    assert np.array_equal(data, a[:, :])
    assert a.metadata.data_type == DataType.string
    assert a.dtype == expected_zarr_string_dtype

    # test round trip
    b = Array.open(sp)
    assert isinstance(b.metadata, ArrayV3Metadata)  # needed for mypy
    assert np.array_equal(data, b[:, :])
    assert b.metadata.data_type == DataType.string
    assert a.dtype == expected_zarr_string_dtype


@pytest.mark.parametrize("store", ["memory", "local"], indirect=["store"])
@pytest.mark.parametrize("as_object_array", [False, True])
@pytest.mark.parametrize("codecs", [None, [VLenBytesCodec()], [VLenBytesCodec(), ZstdCodec()]])
def test_vlen_bytes(store: Store, as_object_array: bool, codecs: None | list[Codec]) -> None:
    bstrings = [b"hello", b"world", b"this", b"is", b"a", b"test"]
    data = np.array(bstrings).reshape((2, 3))
    assert data.dtype == "|S5"

    sp = StorePath(store, path="string")
    a = Array.create(
        sp,
        shape=data.shape,
        chunk_shape=data.shape,
        dtype=data.dtype,
        fill_value=b"",
        codecs=codecs,
    )
    assert isinstance(a.metadata, ArrayV3Metadata)  # needed for mypy

    # should also work if input array is an object array, provided we explicitly specified
    # a bytesting-like dtype when creating the Array
    if as_object_array:
        data = data.astype("O")
    a[:, :] = data
    assert np.array_equal(data, a[:, :])
    assert a.metadata.data_type == DataType.bytes
    assert a.dtype == "O"

    # test round trip
    b = Array.open(sp)
    assert isinstance(b.metadata, ArrayV3Metadata)  # needed for mypy
    assert np.array_equal(data, b[:, :])
    assert b.metadata.data_type == DataType.bytes
    assert a.dtype == "O"
