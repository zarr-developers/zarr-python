import numpy as np
import pytest

from zarr.abc.store import Store
from zarr.array import Array
from zarr.codecs import VLenUTF8Codec
from zarr.store.core import StorePath


@pytest.mark.parametrize("store", ("local", "memory"), indirect=["store"])
def test_arrow_vlen_string(store: Store) -> None:
    strings = ["hello", "world", "this", "is", "a", "test"]
    data = np.array(strings).reshape((2, 3))

    a = Array.create(
        StorePath(store, path="arrow"),
        shape=data.shape,
        chunk_shape=data.shape,
        dtype=data.dtype,
        fill_value=0,
        codecs=[VLenUTF8Codec()],
    )

    a[:, :] = data
    print(a)
    print(a[:])
    assert np.array_equal(data, a[:, :])
