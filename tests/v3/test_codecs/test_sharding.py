import pickle
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from zarr import Array, AsyncArray
from zarr.abc.store import Store
from zarr.codecs import (
    BloscCodec,
    BytesCodec,
    ShardingCodec,
    ShardingCodecIndexLocation,
    TransposeCodec,
)
from zarr.core.buffer import default_buffer_prototype
from zarr.storage.common import StorePath

from ..conftest import ArrayRequest
from .test_codecs import _AsyncArrayProxy, order_from_dim


@pytest.mark.parametrize("store", ["local", "memory", "zip"], indirect=["store"])
@pytest.mark.parametrize("index_location", ["start", "end"])
@pytest.mark.parametrize(
    "array_fixture",
    [
        ArrayRequest(shape=(128,) * 1, dtype="uint8", order="C"),
        ArrayRequest(shape=(128,) * 2, dtype="uint8", order="C"),
        ArrayRequest(shape=(128,) * 3, dtype="uint16", order="F"),
    ],
    indirect=["array_fixture"],
)
@pytest.mark.parametrize("offset", [0, 10])
def test_sharding(
    store: Store,
    array_fixture: npt.NDArray[Any],
    index_location: ShardingCodecIndexLocation,
    offset: int,
) -> None:
    """
    Test that we can create an array with a sharding codec, write data to that array, and get
    the same data out via indexing.
    """
    data = array_fixture
    spath = StorePath(store)
    arr = Array.create(
        spath,
        shape=tuple(s + offset for s in data.shape),
        chunk_shape=(64,) * data.ndim,
        dtype=data.dtype,
        fill_value=6,
        codecs=[
            ShardingCodec(
                chunk_shape=(32,) * data.ndim,
                codecs=[
                    TransposeCodec(order=order_from_dim("F", data.ndim)),
                    BytesCodec(),
                    BloscCodec(cname="lz4"),
                ],
                index_location=index_location,
            )
        ],
    )
    write_region = tuple(slice(offset, None) for dim in range(data.ndim))
    arr[write_region] = data

    if offset > 0:
        empty_region = tuple(slice(0, offset) for dim in range(data.ndim))
        assert np.all(arr[empty_region] == arr.metadata.fill_value)

    read_data = arr[write_region]
    assert data.shape == read_data.shape
    assert np.array_equal(data, read_data)


@pytest.mark.parametrize("index_location", ["start", "end"])
@pytest.mark.parametrize("store", ["local", "memory", "zip"], indirect=["store"])
@pytest.mark.parametrize(
    "array_fixture",
    [
        ArrayRequest(shape=(128,) * 3, dtype="uint16", order="F"),
    ],
    indirect=["array_fixture"],
)
def test_sharding_partial(
    store: Store, array_fixture: npt.NDArray[Any], index_location: ShardingCodecIndexLocation
) -> None:
    data = array_fixture
    spath = StorePath(store)
    a = Array.create(
        spath,
        shape=tuple(a + 10 for a in data.shape),
        chunk_shape=(64, 64, 64),
        dtype=data.dtype,
        fill_value=0,
        codecs=[
            ShardingCodec(
                chunk_shape=(32, 32, 32),
                codecs=[
                    TransposeCodec(order=order_from_dim("F", data.ndim)),
                    BytesCodec(),
                    BloscCodec(cname="lz4"),
                ],
                index_location=index_location,
            )
        ],
    )

    a[10:, 10:, 10:] = data

    read_data = a[0:10, 0:10, 0:10]
    assert np.all(read_data == 0)

    read_data = a[10:, 10:, 10:]
    assert data.shape == read_data.shape
    assert np.array_equal(data, read_data)


@pytest.mark.parametrize(
    "array_fixture",
    [
        ArrayRequest(shape=(128,) * 3, dtype="uint16", order="F"),
    ],
    indirect=["array_fixture"],
)
@pytest.mark.parametrize("index_location", ["start", "end"])
@pytest.mark.parametrize("store", ["local", "memory", "zip"], indirect=["store"])
def test_sharding_partial_read(
    store: Store, array_fixture: npt.NDArray[Any], index_location: ShardingCodecIndexLocation
) -> None:
    data = array_fixture
    spath = StorePath(store)
    a = Array.create(
        spath,
        shape=tuple(a + 10 for a in data.shape),
        chunk_shape=(64, 64, 64),
        dtype=data.dtype,
        fill_value=1,
        codecs=[
            ShardingCodec(
                chunk_shape=(32, 32, 32),
                codecs=[
                    TransposeCodec(order=order_from_dim("F", data.ndim)),
                    BytesCodec(),
                    BloscCodec(cname="lz4"),
                ],
                index_location=index_location,
            )
        ],
    )

    read_data = a[0:10, 0:10, 0:10]
    assert np.all(read_data == 1)


@pytest.mark.parametrize(
    "array_fixture",
    [
        ArrayRequest(shape=(128,) * 3, dtype="uint16", order="F"),
    ],
    indirect=["array_fixture"],
)
@pytest.mark.parametrize("index_location", ["start", "end"])
@pytest.mark.parametrize("store", ["local", "memory", "zip"], indirect=["store"])
def test_sharding_partial_overwrite(
    store: Store, array_fixture: npt.NDArray[Any], index_location: ShardingCodecIndexLocation
) -> None:
    data = array_fixture[:10, :10, :10]
    spath = StorePath(store)
    a = Array.create(
        spath,
        shape=tuple(a + 10 for a in data.shape),
        chunk_shape=(64, 64, 64),
        dtype=data.dtype,
        fill_value=1,
        codecs=[
            ShardingCodec(
                chunk_shape=(32, 32, 32),
                codecs=[
                    TransposeCodec(order=order_from_dim("F", data.ndim)),
                    BytesCodec(),
                    BloscCodec(cname="lz4"),
                ],
                index_location=index_location,
            )
        ],
    )

    a[:10, :10, :10] = data

    read_data = a[0:10, 0:10, 0:10]
    assert np.array_equal(data, read_data)

    data = data + 10
    a[:10, :10, :10] = data
    read_data = a[0:10, 0:10, 0:10]
    assert np.array_equal(data, read_data)


@pytest.mark.parametrize(
    "array_fixture",
    [
        ArrayRequest(shape=(128,) * 3, dtype="uint16", order="F"),
    ],
    indirect=["array_fixture"],
)
@pytest.mark.parametrize(
    "outer_index_location",
    ["start", "end"],
)
@pytest.mark.parametrize(
    "inner_index_location",
    ["start", "end"],
)
@pytest.mark.parametrize("store", ["local", "memory", "zip"], indirect=["store"])
def test_nested_sharding(
    store: Store,
    array_fixture: npt.NDArray[Any],
    outer_index_location: ShardingCodecIndexLocation,
    inner_index_location: ShardingCodecIndexLocation,
) -> None:
    data = array_fixture
    spath = StorePath(store)
    a = Array.create(
        spath,
        shape=data.shape,
        chunk_shape=(64, 64, 64),
        dtype=data.dtype,
        fill_value=0,
        codecs=[
            ShardingCodec(
                chunk_shape=(32, 32, 32),
                codecs=[
                    ShardingCodec(chunk_shape=(16, 16, 16), index_location=inner_index_location)
                ],
                index_location=outer_index_location,
            )
        ],
    )

    a[:, :, :] = data

    read_data = a[0 : data.shape[0], 0 : data.shape[1], 0 : data.shape[2]]
    assert data.shape == read_data.shape
    assert np.array_equal(data, read_data)


@pytest.mark.parametrize("store", ["local", "memory", "zip"], indirect=["store"])
def test_open_sharding(store: Store) -> None:
    path = "open_sharding"
    spath = StorePath(store, path)
    a = Array.create(
        spath,
        shape=(16, 16),
        chunk_shape=(16, 16),
        dtype="int32",
        fill_value=0,
        codecs=[
            ShardingCodec(
                chunk_shape=(8, 8),
                codecs=[
                    TransposeCodec(order=order_from_dim("F", 2)),
                    BytesCodec(),
                    BloscCodec(),
                ],
            )
        ],
    )
    b = Array.open(spath)
    assert a.metadata == b.metadata


@pytest.mark.parametrize("store", ["local", "memory", "zip"], indirect=["store"])
def test_write_partial_sharded_chunks(store: Store) -> None:
    data = np.arange(0, 16 * 16, dtype="uint16").reshape((16, 16))
    spath = StorePath(store)
    a = Array.create(
        spath,
        shape=(40, 40),
        chunk_shape=(20, 20),
        dtype=data.dtype,
        fill_value=1,
        codecs=[
            ShardingCodec(
                chunk_shape=(10, 10),
                codecs=[
                    BytesCodec(),
                    BloscCodec(),
                ],
            )
        ],
    )
    a[0:16, 0:16] = data
    assert np.array_equal(a[0:16, 0:16], data)


@pytest.mark.parametrize("store", ["local", "memory", "zip"], indirect=["store"])
async def test_delete_empty_shards(store: Store) -> None:
    if not store.supports_deletes:
        pytest.skip("store does not support deletes")
    path = "delete_empty_shards"
    spath = StorePath(store, path)
    a = await AsyncArray.create(
        spath,
        shape=(16, 16),
        chunk_shape=(8, 16),
        dtype="uint16",
        fill_value=1,
        codecs=[ShardingCodec(chunk_shape=(8, 8))],
    )
    await _AsyncArrayProxy(a)[:, :].set(np.zeros((16, 16)))
    await _AsyncArrayProxy(a)[8:, :].set(np.ones((8, 16)))
    await _AsyncArrayProxy(a)[:, 8:].set(np.ones((16, 8)))
    # chunk (0, 0) is full
    # chunks (0, 1), (1, 0), (1, 1) are empty
    # shard (0, 0) is half-full
    # shard (1, 0) is empty

    data = np.ones((16, 16), dtype="uint16")
    data[:8, :8] = 0
    assert np.array_equal(data, await _AsyncArrayProxy(a)[:, :].get())
    assert await store.get(f"{path}/c/1/0", prototype=default_buffer_prototype()) is None
    chunk_bytes = await store.get(f"{path}/c/0/0", prototype=default_buffer_prototype())
    assert chunk_bytes is not None
    assert len(chunk_bytes) == 16 * 2 + 8 * 8 * 2 + 4


def test_pickle() -> None:
    codec = ShardingCodec(chunk_shape=(8, 8))
    assert pickle.loads(pickle.dumps(codec)) == codec
