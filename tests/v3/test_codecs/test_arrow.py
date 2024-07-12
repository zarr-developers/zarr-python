import numpy as np
import pytest

from zarr.abc.store import Store
from zarr.array import Array
from zarr.codecs import ArrowRecordBatchCodec
from zarr.store.core import StorePath


@pytest.mark.parametrize("store", ("local", "memory"), indirect=["store"])
@pytest.mark.parametrize(
    "dtype",
    [
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "int8",
        "int16",
        "int32",
        "int64",
        "float32",
        "float64",
    ],
)
def test_arrow_standard_dtypes(store: Store, dtype) -> None:
    data = np.arange(0, 256, dtype=dtype).reshape((16, 16))

    a = Array.create(
        StorePath(store, path="arrow"),
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype=data.dtype,
        fill_value=0,
        codecs=[ArrowRecordBatchCodec()],
    )

    a[:, :] = data
    assert np.array_equal(data, a[:, :])


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
        codecs=[ArrowRecordBatchCodec()],
    )

    a[:, :] = data
    assert np.array_equal(data, a[:, :])
