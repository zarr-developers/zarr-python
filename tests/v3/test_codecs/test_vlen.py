import numpy as np
import pytest

from zarr import Array
from zarr.abc.store import Store
from zarr.codecs import VLenUTF8Codec
from zarr.core.metadata.v3 import DataType
from zarr.store.common import StorePath


@pytest.mark.parametrize("store", ["memory", "local"], indirect=["store"])
@pytest.mark.parametrize("dtype", [None, np.dtypes.StrDType])
async def test_vlen_string(store: Store, dtype) -> None:
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

    a[:, :] = data
    assert np.array_equal(data, a[:, :])
    assert a.metadata.data_type == DataType.string
    assert a.dtype == np.dtypes.StringDType()

    # test round trip
    b = Array.open(sp)
    assert np.array_equal(data, b[:, :])
    assert b.metadata.data_type == DataType.string
    assert b.dtype == np.dtypes.StringDType()
