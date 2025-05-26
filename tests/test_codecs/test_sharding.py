import pickle
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import zarr
import zarr.api
import zarr.api.asynchronous
from zarr import Array
from zarr.abc.store import Store
from zarr.codecs import (
    BloscCodec,
    ShardingCodec,
    ShardingCodecIndexLocation,
    TransposeCodec,
)
from zarr.core.buffer import NDArrayLike, default_buffer_prototype
from zarr.storage import StorePath

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

    arr = zarr.create_array(
        spath,
        shape=tuple(s + offset for s in data.shape),
        chunks=(32,) * data.ndim,
        shards={"shape": (64,) * data.ndim, "index_location": index_location},
        dtype=data.dtype,
        fill_value=6,
        filters=[TransposeCodec(order=order_from_dim("F", data.ndim))],
        compressors=BloscCodec(cname="lz4"),
    )
    write_region = tuple(slice(offset, None) for dim in range(data.ndim))
    arr[write_region] = data

    if offset > 0:
        empty_region = tuple(slice(0, offset) for dim in range(data.ndim))
        assert np.all(arr[empty_region] == arr.metadata.fill_value)

    read_data = arr[write_region]
    assert isinstance(read_data, NDArrayLike)
    assert data.shape == read_data.shape
    assert np.array_equal(data, read_data)


@pytest.mark.parametrize("store", ["local", "memory", "zip"], indirect=["store"])
@pytest.mark.parametrize("index_location", ["start", "end"])
@pytest.mark.parametrize("offset", [0, 10])
def test_sharding_scalar(
    store: Store,
    index_location: ShardingCodecIndexLocation,
    offset: int,
) -> None:
    """
    Test that we can create an array with a sharding codec, write data to that array, and get
    the same data out via indexing.
    """
    spath = StorePath(store)

    arr = zarr.create_array(
        spath,
        shape=(128, 128),
        chunks=(32, 32),
        shards={"shape": (64, 64), "index_location": index_location},
        dtype="uint8",
        fill_value=6,
        filters=[TransposeCodec(order=order_from_dim("F", 2))],
        compressors=BloscCodec(cname="lz4"),
    )
    arr[:16, :16] = 10  # intentionally write partial chunks
    read_data = arr[:16, :16]
    np.testing.assert_array_equal(read_data, 10)


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
    a = zarr.create_array(
        spath,
        shape=tuple(a + 10 for a in data.shape),
        chunks=(32, 32, 32),
        shards={"shape": (64, 64, 64), "index_location": index_location},
        compressors=BloscCodec(cname="lz4"),
        filters=[TransposeCodec(order=order_from_dim("F", data.ndim))],
        dtype=data.dtype,
        fill_value=0,
    )

    a[10:, 10:, 10:] = data

    read_data = a[0:10, 0:10, 0:10]
    assert np.all(read_data == 0)

    read_data = a[10:, 10:, 10:]
    assert isinstance(read_data, NDArrayLike)
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
def test_sharding_partial_readwrite(
    store: Store, array_fixture: npt.NDArray[Any], index_location: ShardingCodecIndexLocation
) -> None:
    data = array_fixture
    spath = StorePath(store)
    a = zarr.create_array(
        spath,
        shape=data.shape,
        chunks=(1, data.shape[1], data.shape[2]),
        shards={"shape": data.shape, "index_location": index_location},
        dtype=data.dtype,
        fill_value=0,
        filters=None,
        compressors=None,
    )

    a[:] = data

    for x in range(data.shape[0]):
        read_data = a[x, :, :]
        assert np.array_equal(data[x], read_data)


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
    a = zarr.create_array(
        spath,
        shape=tuple(a + 10 for a in data.shape),
        chunks=(32, 32, 32),
        shards={"shape": (64, 64, 64), "index_location": index_location},
        compressors=BloscCodec(cname="lz4"),
        filters=[TransposeCodec(order=order_from_dim("F", data.ndim))],
        dtype=data.dtype,
        fill_value=1,
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
    a = zarr.create_array(
        spath,
        shape=tuple(a + 10 for a in data.shape),
        chunks=(32, 32, 32),
        shards={"shape": (64, 64, 64), "index_location": index_location},
        compressors=BloscCodec(cname="lz4"),
        filters=[TransposeCodec(order=order_from_dim("F", data.ndim))],
        dtype=data.dtype,
        fill_value=1,
    )

    a[:10, :10, :10] = data

    read_data = a[0:10, 0:10, 0:10]
    assert np.array_equal(data, read_data)

    data += 10
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
    assert isinstance(read_data, NDArrayLike)
    assert data.shape == read_data.shape
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
def test_nested_sharding_create_array(
    store: Store,
    array_fixture: npt.NDArray[Any],
    outer_index_location: ShardingCodecIndexLocation,
    inner_index_location: ShardingCodecIndexLocation,
) -> None:
    data = array_fixture
    spath = StorePath(store)
    a = zarr.create_array(
        spath,
        shape=data.shape,
        chunks=(32, 32, 32),
        dtype=data.dtype,
        fill_value=0,
        serializer=ShardingCodec(
            chunk_shape=(32, 32, 32),
            codecs=[ShardingCodec(chunk_shape=(16, 16, 16), index_location=inner_index_location)],
            index_location=outer_index_location,
        ),
        filters=None,
        compressors=None,
    )
    print(a.metadata.to_dict())

    a[:, :, :] = data

    read_data = a[0 : data.shape[0], 0 : data.shape[1], 0 : data.shape[2]]
    assert isinstance(read_data, NDArrayLike)
    assert data.shape == read_data.shape
    assert np.array_equal(data, read_data)


@pytest.mark.parametrize("store", ["local", "memory", "zip"], indirect=["store"])
def test_open_sharding(store: Store) -> None:
    path = "open_sharding"
    spath = StorePath(store, path)
    a = zarr.create_array(
        spath,
        shape=(16, 16),
        chunks=(8, 8),
        shards=(16, 16),
        filters=[TransposeCodec(order=order_from_dim("F", 2))],
        compressors=BloscCodec(),
        dtype="int32",
        fill_value=0,
    )
    b = Array.open(spath)
    assert a.metadata == b.metadata


@pytest.mark.parametrize("store", ["local", "memory", "zip"], indirect=["store"])
def test_write_partial_sharded_chunks(store: Store) -> None:
    data = np.arange(0, 16 * 16, dtype="uint16").reshape((16, 16))
    spath = StorePath(store)
    a = zarr.create_array(
        spath,
        shape=(40, 40),
        chunks=(10, 10),
        shards=(20, 20),
        dtype=data.dtype,
        compressors=BloscCodec(),
        fill_value=1,
    )
    a[0:16, 0:16] = data
    assert np.array_equal(a[0:16, 0:16], data)


@pytest.mark.parametrize("store", ["local", "memory", "zip"], indirect=["store"])
async def test_delete_empty_shards(store: Store) -> None:
    if not store.supports_deletes:
        pytest.skip("store does not support deletes")
    path = "delete_empty_shards"
    spath = StorePath(store, path)
    a = await zarr.api.asynchronous.create_array(
        spath,
        shape=(16, 16),
        chunks=(8, 8),
        shards=(8, 16),
        dtype="uint16",
        compressors=None,
        fill_value=1,
    )
    print(a.metadata.to_dict())
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


@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
@pytest.mark.parametrize(
    "index_location", [ShardingCodecIndexLocation.start, ShardingCodecIndexLocation.end]
)
async def test_sharding_with_empty_inner_chunk(
    store: Store, index_location: ShardingCodecIndexLocation
) -> None:
    data = np.arange(0, 16 * 16, dtype="uint32").reshape((16, 16))
    fill_value = 1

    path = f"sharding_with_empty_inner_chunk_{index_location}"
    spath = StorePath(store, path)
    a = await zarr.api.asynchronous.create_array(
        spath,
        shape=(16, 16),
        chunks=(4, 4),
        shards={"shape": (8, 8), "index_location": index_location},
        dtype="uint32",
        fill_value=fill_value,
    )
    data[:4, :4] = fill_value
    await a.setitem(..., data)
    print("read data")
    data_read = await a.getitem(...)
    assert np.array_equal(data_read, data)


@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
@pytest.mark.parametrize(
    "index_location",
    [ShardingCodecIndexLocation.start, ShardingCodecIndexLocation.end],
)
@pytest.mark.parametrize("chunks_per_shard", [(5, 2), (2, 5), (5, 5)])
async def test_sharding_with_chunks_per_shard(
    store: Store, index_location: ShardingCodecIndexLocation, chunks_per_shard: tuple[int]
) -> None:
    chunk_shape = (2, 1)
    shape = tuple(x * y for x, y in zip(chunks_per_shard, chunk_shape, strict=False))
    data = np.ones(np.prod(shape), dtype="int32").reshape(shape)
    fill_value = 42

    path = f"test_sharding_with_chunks_per_shard_{index_location}"
    spath = StorePath(store, path)
    a = zarr.create_array(
        spath,
        shape=shape,
        chunks=chunk_shape,
        shards={"shape": shape, "index_location": index_location},
        dtype="int32",
        fill_value=fill_value,
    )
    a[...] = data
    data_read = a[...]
    assert np.array_equal(data_read, data)


@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
def test_invalid_metadata(store: Store) -> None:
    spath1 = StorePath(store, "invalid_inner_chunk_shape")
    with pytest.raises(ValueError):
        zarr.create_array(
            spath1,
            shape=(16, 16),
            shards=(16, 16),
            chunks=(8,),
            dtype=np.dtype("uint8"),
            fill_value=0,
        )
    spath2 = StorePath(store, "invalid_inner_chunk_shape")
    with pytest.raises(ValueError):
        zarr.create_array(
            spath2,
            shape=(16, 16),
            shards=(16, 16),
            chunks=(8, 7),
            dtype=np.dtype("uint8"),
            fill_value=0,
        )
