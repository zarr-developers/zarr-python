from typing import Any

import numpy as np
import pytest

from zarr import Array
from zarr.abc.store import Store
from zarr.codecs import VLenUTF8Codec
from zarr.core.metadata.v3 import ArrayV3Metadata, DataType
from zarr.storage.common import StorePath
from zarr.strings import NUMPY_SUPPORTS_VLEN_STRING

numpy_str_dtypes: list[type | None] = [None, str, np.dtypes.StrDType]
expected_zarr_string_dtype: np.dtype[Any]
if NUMPY_SUPPORTS_VLEN_STRING:
    numpy_str_dtypes.append(np.dtypes.StringDType)
    expected_zarr_string_dtype = np.dtypes.StringDType()
else:
    expected_zarr_string_dtype = np.dtype("O")


@pytest.mark.parametrize("store", ["memory", "local"], indirect=["store"])
@pytest.mark.parametrize("dtype", numpy_str_dtypes)
async def test_vlen_string(store: Store, dtype: None | np.dtype[Any]) -> None:
    strings = ["hello", "world", "this", "is", "a", "test"]
    data = np.array(strings).reshape((2, 3))
    if dtype is not None:
        data = data.astype(dtype)

    sp = StorePath(store, path="string")
    a = Array.create(
        sp,
        shape=data.shape,
        chunk_shape=data.shape,
        dtype=data.dtype,
        fill_value="",
        codecs=[VLenUTF8Codec()],
    )
    assert isinstance(a.metadata, ArrayV3Metadata)  # needed for mypy

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
