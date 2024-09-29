import numpy as np
import pytest

from zarr import Array
from zarr.abc.store import Store
from zarr.codecs import VLenUTF8Codec
from zarr.store.common import StorePath


@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
def test_vlen_string(store: Store) -> None:
    strings = ["hello", "world", "this", "is", "a", "test"]
    data = np.array(strings).reshape((2, 3))

    a = Array.create(
        StorePath(store, path="string"),
        shape=data.shape,
        chunk_shape=data.shape,
        dtype=data.dtype,
        fill_value="",
        codecs=[VLenUTF8Codec()],
    )

    a[:, :] = data
    assert np.array_equal(data, a[:, :])
