from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pytest

import zarr.v2
from zarr.abc.codec import Codec
from zarr.abc.store import Store
from zarr.array import Array, AsyncArray
from zarr.codecs import (
    BloscCodec,
    BytesCodec,
    GzipCodec,
    ShardingCodec,
    ShardingCodecIndexLocation,
    TransposeCodec,
    ZstdCodec,
)
from zarr.common import Selection
from zarr.config import config
from zarr.indexing import morton_order_iter
from zarr.store import MemoryStore, StorePath
from zarr.testing.utils import assert_bytes_equal


@dataclass(frozen=True)
class _AsyncArrayProxy:
    array: AsyncArray

    def __getitem__(self, selection: Selection) -> _AsyncArraySelectionProxy:
        return _AsyncArraySelectionProxy(self.array, selection)


@dataclass(frozen=True)
class _AsyncArraySelectionProxy:
    array: AsyncArray
    selection: Selection

    async def get(self) -> np.ndarray:
        return await self.array.getitem(self.selection)

    async def set(self, value: np.ndarray):
        return await self.array.setitem(self.selection, value)


@pytest.fixture
def store() -> Iterator[Store]:
    yield StorePath(MemoryStore(mode="w"))


@pytest.fixture
def sample_data() -> np.ndarray:
    return np.arange(0, 128 * 128 * 128, dtype="uint16").reshape((128, 128, 128), order="F")


def order_from_dim(order: Literal["F", "C"], ndim: int) -> tuple[int, ...]:
    if order == "F":
        return tuple(ndim - x - 1 for x in range(ndim))
    else:
        return tuple(range(ndim))


@pytest.mark.parametrize("index_location", ["start", "end"])
def test_sharding(
    store: Store, sample_data: np.ndarray, index_location: ShardingCodecIndexLocation
):
    a = Array.create(
        store / "sample",
        shape=sample_data.shape,
        chunk_shape=(64, 64, 64),
        dtype=sample_data.dtype,
        fill_value=0,
        codecs=[
            ShardingCodec(
                chunk_shape=(32, 32, 32),
                codecs=[
                    TransposeCodec(order=order_from_dim("F", sample_data.ndim)),
                    BytesCodec(),
                    BloscCodec(cname="lz4"),
                ],
                index_location=index_location,
            )
        ],
    )

    a[:, :, :] = sample_data

    read_data = a[0 : sample_data.shape[0], 0 : sample_data.shape[1], 0 : sample_data.shape[2]]
    assert sample_data.shape == read_data.shape
    assert np.array_equal(sample_data, read_data)


@pytest.mark.parametrize("index_location", ["start", "end"])
def test_sharding_partial(
    store: Store, sample_data: np.ndarray, index_location: ShardingCodecIndexLocation
):
    a = Array.create(
        store / "sample",
        shape=tuple(a + 10 for a in sample_data.shape),
        chunk_shape=(64, 64, 64),
        dtype=sample_data.dtype,
        fill_value=0,
        codecs=[
            ShardingCodec(
                chunk_shape=(32, 32, 32),
                codecs=[
                    TransposeCodec(order=order_from_dim("F", sample_data.ndim)),
                    BytesCodec(),
                    BloscCodec(cname="lz4"),
                ],
                index_location=index_location,
            )
        ],
    )

    a[10:, 10:, 10:] = sample_data

    read_data = a[0:10, 0:10, 0:10]
    assert np.all(read_data == 0)

    read_data = a[10:, 10:, 10:]
    assert sample_data.shape == read_data.shape
    assert np.array_equal(sample_data, read_data)


@pytest.mark.parametrize("index_location", ["start", "end"])
def test_sharding_partial_read(
    store: Store, sample_data: np.ndarray, index_location: ShardingCodecIndexLocation
):
    a = Array.create(
        store / "sample",
        shape=tuple(a + 10 for a in sample_data.shape),
        chunk_shape=(64, 64, 64),
        dtype=sample_data.dtype,
        fill_value=1,
        codecs=[
            ShardingCodec(
                chunk_shape=(32, 32, 32),
                codecs=[
                    TransposeCodec(order=order_from_dim("F", sample_data.ndim)),
                    BytesCodec(),
                    BloscCodec(cname="lz4"),
                ],
                index_location=index_location,
            )
        ],
    )

    read_data = a[0:10, 0:10, 0:10]
    assert np.all(read_data == 1)


@pytest.mark.parametrize("index_location", ["start", "end"])
def test_sharding_partial_overwrite(
    store: Store, sample_data: np.ndarray, index_location: ShardingCodecIndexLocation
):
    data = sample_data[:10, :10, :10]

    a = Array.create(
        store / "sample",
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
    "outer_index_location",
    ["start", "end"],
)
@pytest.mark.parametrize(
    "inner_index_location",
    ["start", "end"],
)
def test_nested_sharding(
    store: Store,
    sample_data: np.ndarray,
    outer_index_location: ShardingCodecIndexLocation,
    inner_index_location: ShardingCodecIndexLocation,
):
    a = Array.create(
        store / "l4_sample" / "color" / "1",
        shape=sample_data.shape,
        chunk_shape=(64, 64, 64),
        dtype=sample_data.dtype,
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

    a[:, :, :] = sample_data

    read_data = a[0 : sample_data.shape[0], 0 : sample_data.shape[1], 0 : sample_data.shape[2]]
    assert sample_data.shape == read_data.shape
    assert np.array_equal(sample_data, read_data)


@pytest.mark.parametrize("input_order", ["F", "C"])
@pytest.mark.parametrize("store_order", ["F", "C"])
@pytest.mark.parametrize("runtime_write_order", ["F", "C"])
@pytest.mark.parametrize("runtime_read_order", ["F", "C"])
@pytest.mark.parametrize("with_sharding", [True, False])
async def test_order(
    store: Store,
    input_order: Literal["F", "C"],
    store_order: Literal["F", "C"],
    runtime_write_order: Literal["F", "C"],
    runtime_read_order: Literal["F", "C"],
    with_sharding: bool,
):
    data = np.arange(0, 256, dtype="uint16").reshape((32, 8), order=input_order)

    codecs_: list[Codec] = (
        [
            ShardingCodec(
                chunk_shape=(16, 8),
                codecs=[TransposeCodec(order=order_from_dim(store_order, data.ndim)), BytesCodec()],
            )
        ]
        if with_sharding
        else [TransposeCodec(order=order_from_dim(store_order, data.ndim)), BytesCodec()]
    )

    with config.set({"array.order": runtime_write_order}):
        a = await AsyncArray.create(
            store / "order",
            shape=data.shape,
            chunk_shape=(32, 8),
            dtype=data.dtype,
            fill_value=0,
            chunk_key_encoding=("v2", "."),
            codecs=codecs_,
        )

    await _AsyncArrayProxy(a)[:, :].set(data)
    read_data = await _AsyncArrayProxy(a)[:, :].get()
    assert np.array_equal(data, read_data)

    with config.set({"array.order": runtime_read_order}):
        a = await AsyncArray.open(
            store / "order",
        )
    read_data = await _AsyncArrayProxy(a)[:, :].get()
    assert np.array_equal(data, read_data)

    if runtime_read_order == "F":
        assert read_data.flags["F_CONTIGUOUS"]
        assert not read_data.flags["C_CONTIGUOUS"]
    else:
        assert not read_data.flags["F_CONTIGUOUS"]
        assert read_data.flags["C_CONTIGUOUS"]

    if not with_sharding:
        # Compare with zarr-python
        z = zarr.v2.create(
            shape=data.shape,
            chunks=(32, 8),
            dtype="<u2",
            order=store_order,
            compressor=None,
            fill_value=1,
        )
        z[:, :] = data
        assert_bytes_equal(await (store / "order/0.0").get(), z._store["0.0"])


@pytest.mark.parametrize("input_order", ["F", "C"])
@pytest.mark.parametrize("runtime_write_order", ["F", "C"])
@pytest.mark.parametrize("runtime_read_order", ["F", "C"])
@pytest.mark.parametrize("with_sharding", [True, False])
def test_order_implicit(
    store: Store,
    input_order: Literal["F", "C"],
    runtime_write_order: Literal["F", "C"],
    runtime_read_order: Literal["F", "C"],
    with_sharding: bool,
):
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16), order=input_order)

    codecs_: list[Codec] | None = [ShardingCodec(chunk_shape=(8, 8))] if with_sharding else None

    with config.set({"array.order": runtime_write_order}):
        a = Array.create(
            store / "order_implicit",
            shape=data.shape,
            chunk_shape=(16, 16),
            dtype=data.dtype,
            fill_value=0,
            codecs=codecs_,
        )

    a[:, :] = data

    with config.set({"array.order": runtime_read_order}):
        a = Array.open(
            store / "order_implicit",
        )
    read_data = a[:, :]
    assert np.array_equal(data, read_data)

    if runtime_read_order == "F":
        assert read_data.flags["F_CONTIGUOUS"]
        assert not read_data.flags["C_CONTIGUOUS"]
    else:
        assert not read_data.flags["F_CONTIGUOUS"]
        assert read_data.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("input_order", ["F", "C"])
@pytest.mark.parametrize("runtime_write_order", ["F", "C"])
@pytest.mark.parametrize("runtime_read_order", ["F", "C"])
@pytest.mark.parametrize("with_sharding", [True, False])
async def test_transpose(
    store: Store,
    input_order: Literal["F", "C"],
    runtime_write_order: Literal["F", "C"],
    runtime_read_order: Literal["F", "C"],
    with_sharding: bool,
):
    data = np.arange(0, 256, dtype="uint16").reshape((1, 32, 8), order=input_order)

    codecs_: list[Codec] = (
        [
            ShardingCodec(
                chunk_shape=(1, 16, 8),
                codecs=[TransposeCodec(order=(2, 1, 0)), BytesCodec()],
            )
        ]
        if with_sharding
        else [TransposeCodec(order=(2, 1, 0)), BytesCodec()]
    )
    with config.set({"array.order": runtime_write_order}):
        a = await AsyncArray.create(
            store / "transpose",
            shape=data.shape,
            chunk_shape=(1, 32, 8),
            dtype=data.dtype,
            fill_value=0,
            chunk_key_encoding=("v2", "."),
            codecs=codecs_,
        )

    await _AsyncArrayProxy(a)[:, :].set(data)
    read_data = await _AsyncArrayProxy(a)[:, :].get()
    assert np.array_equal(data, read_data)

    with config.set({"array.order": runtime_read_order}):
        a = await AsyncArray.open(
            store / "transpose",
        )
    read_data = await _AsyncArrayProxy(a)[:, :].get()
    assert np.array_equal(data, read_data)

    if runtime_read_order == "F":
        assert read_data.flags["F_CONTIGUOUS"]
        assert not read_data.flags["C_CONTIGUOUS"]
    else:
        assert not read_data.flags["F_CONTIGUOUS"]
        assert read_data.flags["C_CONTIGUOUS"]

    if not with_sharding:
        # Compare with zarr-python
        z = zarr.v2.create(
            shape=data.shape,
            chunks=(1, 32, 8),
            dtype="<u2",
            order="F",
            compressor=None,
            fill_value=1,
        )
        z[:, :] = data
        assert await (store / "transpose/0.0").get() == await (store / "transpose_zarr/0.0").get()


@pytest.mark.parametrize("order", [[1, 2, 0], [1, 2, 3, 0], [3, 2, 4, 0, 1]])
def test_transpose_non_self_inverse(store: Store, order):
    shape = [i + 3 for i in range(len(order))]
    data = np.arange(0, np.prod(shape), dtype="uint16").reshape(shape)
    a = Array.create(
        store / "transpose_non_self_inverse",
        shape=data.shape,
        chunk_shape=data.shape,
        dtype=data.dtype,
        fill_value=0,
        codecs=[TransposeCodec(order=order), BytesCodec()],
    )
    a[:, :] = data
    read_data = a[:, :]
    assert np.array_equal(data, read_data)


def test_transpose_invalid(
    store: Store,
):
    data = np.arange(0, 256, dtype="uint16").reshape((1, 32, 8))

    for order in [(1, 0), (3, 2, 1), (3, 3, 1)]:
        with pytest.raises(ValueError):
            Array.create(
                store / "transpose_invalid",
                shape=data.shape,
                chunk_shape=(1, 32, 8),
                dtype=data.dtype,
                fill_value=0,
                chunk_key_encoding=("v2", "."),
                codecs=[TransposeCodec(order=order), BytesCodec()],
            )


def test_open(store: Store):
    a = Array.create(
        store / "open",
        shape=(16, 16),
        chunk_shape=(16, 16),
        dtype="int32",
        fill_value=0,
    )
    b = Array.open(store / "open")
    assert a.metadata == b.metadata


def test_open_sharding(store: Store):
    a = Array.create(
        store / "open_sharding",
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
    b = Array.open(store / "open_sharding")
    assert a.metadata == b.metadata


def test_simple(store: Store):
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    a = Array.create(
        store / "simple",
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype=data.dtype,
        fill_value=0,
    )

    a[:, :] = data
    assert np.array_equal(data, a[:, :])


def test_fill_value(store: Store):
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    a = Array.create(
        store / "fill_value1",
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype=data.dtype,
    )

    assert a.metadata.fill_value == 0
    assert np.array_equiv(0, a[:, :])

    b = Array.create(
        store / "fill_value2",
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype=np.dtype("bool"),
    )

    assert b.metadata.fill_value is False
    assert np.array_equiv(False, b[:, :])

    c = Array.create(
        store / "fill_value3",
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype=data.dtype,
        fill_value=4,
    )

    assert c.metadata.fill_value == 4
    assert np.array_equiv(4, c[:, :])


def test_morton(store: Store):
    assert list(morton_order_iter((2, 2))) == [(0, 0), (1, 0), (0, 1), (1, 1)]
    assert list(morton_order_iter((2, 2, 2))) == [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    ]
    assert list(morton_order_iter((2, 2, 2, 2))) == [
        (0, 0, 0, 0),
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (1, 1, 0, 0),
        (0, 0, 1, 0),
        (1, 0, 1, 0),
        (0, 1, 1, 0),
        (1, 1, 1, 0),
        (0, 0, 0, 1),
        (1, 0, 0, 1),
        (0, 1, 0, 1),
        (1, 1, 0, 1),
        (0, 0, 1, 1),
        (1, 0, 1, 1),
        (0, 1, 1, 1),
        (1, 1, 1, 1),
    ]


def test_write_partial_chunks(store: Store):
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    a = Array.create(
        store / "write_partial_chunks",
        shape=data.shape,
        chunk_shape=(20, 20),
        dtype=data.dtype,
        fill_value=1,
    )
    a[0:16, 0:16] = data
    assert np.array_equal(a[0:16, 0:16], data)


def test_write_full_chunks(store: Store):
    data = np.arange(0, 16 * 16, dtype="uint16").reshape((16, 16))

    a = Array.create(
        store / "write_full_chunks",
        shape=(16, 16),
        chunk_shape=(20, 20),
        dtype=data.dtype,
        fill_value=1,
    )
    a[0:16, 0:16] = data
    assert np.array_equal(a[0:16, 0:16], data)

    a = Array.create(
        store / "write_full_chunks2",
        shape=(20, 20),
        chunk_shape=(20, 20),
        dtype=data.dtype,
        fill_value=1,
    )
    assert np.all(a[16:20, 16:20] == 1)


def test_write_partial_sharded_chunks(store: Store):
    data = np.arange(0, 16 * 16, dtype="uint16").reshape((16, 16))

    a = Array.create(
        store / "write_partial_sharded_chunks",
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


async def test_delete_empty_chunks(store: Store):
    data = np.ones((16, 16))

    a = await AsyncArray.create(
        store / "delete_empty_chunks",
        shape=data.shape,
        chunk_shape=(32, 32),
        dtype=data.dtype,
        fill_value=1,
    )
    await _AsyncArrayProxy(a)[:16, :16].set(np.zeros((16, 16)))
    await _AsyncArrayProxy(a)[:16, :16].set(data)
    assert np.array_equal(await _AsyncArrayProxy(a)[:16, :16].get(), data)
    assert await (store / "delete_empty_chunks/c0/0").get() is None


async def test_delete_empty_shards(store: Store):
    a = await AsyncArray.create(
        store / "delete_empty_shards",
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
    assert await (store / "delete_empty_shards/c/1/0").get() is None
    chunk_bytes = await (store / "delete_empty_shards/c/0/0").get()
    assert chunk_bytes is not None and len(chunk_bytes) == 16 * 2 + 8 * 8 * 2 + 4


async def test_zarr_compat(store: Store):
    data = np.zeros((16, 18), dtype="uint16")

    a = await AsyncArray.create(
        store / "zarr_compat3",
        shape=data.shape,
        chunk_shape=(10, 10),
        dtype=data.dtype,
        chunk_key_encoding=("v2", "."),
        fill_value=1,
    )

    z2 = zarr.v2.create(
        shape=data.shape,
        chunks=(10, 10),
        dtype=data.dtype,
        compressor=None,
        fill_value=1,
    )

    await _AsyncArrayProxy(a)[:16, :18].set(data)
    z2[:16, :18] = data
    assert np.array_equal(data, await _AsyncArrayProxy(a)[:16, :18].get())
    assert np.array_equal(data, z2[:16, :18])

    assert_bytes_equal(z2._store["0.0"], await (store / "zarr_compat3/0.0").get())
    assert_bytes_equal(z2._store["0.1"], await (store / "zarr_compat3/0.1").get())
    assert_bytes_equal(z2._store["1.0"], await (store / "zarr_compat3/1.0").get())
    assert_bytes_equal(z2._store["1.1"], await (store / "zarr_compat3/1.1").get())


async def test_zarr_compat_F(store: Store):
    data = np.zeros((16, 18), dtype="uint16", order="F")

    a = await AsyncArray.create(
        store / "zarr_compatF3",
        shape=data.shape,
        chunk_shape=(10, 10),
        dtype=data.dtype,
        chunk_key_encoding=("v2", "."),
        fill_value=1,
        codecs=[TransposeCodec(order=order_from_dim("F", data.ndim)), BytesCodec()],
    )

    z2 = zarr.v2.create(
        shape=data.shape,
        chunks=(10, 10),
        dtype=data.dtype,
        compressor=None,
        order="F",
        fill_value=1,
    )

    await _AsyncArrayProxy(a)[:16, :18].set(data)
    z2[:16, :18] = data
    assert np.array_equal(data, await _AsyncArrayProxy(a)[:16, :18].get())
    assert np.array_equal(data, z2[:16, :18])

    assert_bytes_equal(z2._store["0.0"], await (store / "zarr_compatF3/0.0").get())
    assert_bytes_equal(z2._store["0.1"], await (store / "zarr_compatF3/0.1").get())
    assert_bytes_equal(z2._store["1.0"], await (store / "zarr_compatF3/1.0").get())
    assert_bytes_equal(z2._store["1.1"], await (store / "zarr_compatF3/1.1").get())


async def test_dimension_names(store: Store):
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    await AsyncArray.create(
        store / "dimension_names",
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype=data.dtype,
        fill_value=0,
        dimension_names=("x", "y"),
    )

    assert (await AsyncArray.open(store / "dimension_names")).metadata.dimension_names == (
        "x",
        "y",
    )

    await AsyncArray.create(
        store / "dimension_names2",
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype=data.dtype,
        fill_value=0,
    )

    assert (await AsyncArray.open(store / "dimension_names2")).metadata.dimension_names is None
    zarr_json_buffer = await (store / "dimension_names2" / "zarr.json").get()
    assert zarr_json_buffer is not None
    assert "dimension_names" not in json.loads(zarr_json_buffer.to_bytes())


def test_gzip(store: Store):
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    a = Array.create(
        store / "gzip",
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype=data.dtype,
        fill_value=0,
        codecs=[BytesCodec(), GzipCodec()],
    )

    a[:, :] = data
    assert np.array_equal(data, a[:, :])


@pytest.mark.parametrize("checksum", [True, False])
def test_zstd(store: Store, checksum: bool):
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    a = Array.create(
        store / "zstd",
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype=data.dtype,
        fill_value=0,
        codecs=[BytesCodec(), ZstdCodec(level=0, checksum=checksum)],
    )

    a[:, :] = data
    assert np.array_equal(data, a[:, :])


@pytest.mark.parametrize("endian", ["big", "little"])
async def test_endian(store: Store, endian: Literal["big", "little"]):
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    a = await AsyncArray.create(
        store / "endian",
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype=data.dtype,
        fill_value=0,
        chunk_key_encoding=("v2", "."),
        codecs=[BytesCodec(endian=endian)],
    )

    await _AsyncArrayProxy(a)[:, :].set(data)
    readback_data = await _AsyncArrayProxy(a)[:, :].get()
    assert np.array_equal(data, readback_data)

    # Compare with zarr-python
    z = zarr.v2.create(
        shape=data.shape,
        chunks=(16, 16),
        dtype=">u2" if endian == "big" else "<u2",
        compressor=None,
        fill_value=1,
    )
    z[:, :] = data
    assert_bytes_equal(await (store / "endian/0.0").get(), z._store["0.0"])


@pytest.mark.parametrize("dtype_input_endian", [">u2", "<u2"])
@pytest.mark.parametrize("dtype_store_endian", ["big", "little"])
async def test_endian_write(
    store: Store,
    dtype_input_endian: Literal[">u2", "<u2"],
    dtype_store_endian: Literal["big", "little"],
):
    data = np.arange(0, 256, dtype=dtype_input_endian).reshape((16, 16))

    a = await AsyncArray.create(
        store / "endian",
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype="uint16",
        fill_value=0,
        chunk_key_encoding=("v2", "."),
        codecs=[BytesCodec(endian=dtype_store_endian)],
    )

    await _AsyncArrayProxy(a)[:, :].set(data)
    readback_data = await _AsyncArrayProxy(a)[:, :].get()
    assert np.array_equal(data, readback_data)

    # Compare with zarr-python
    z = zarr.v2.create(
        shape=data.shape,
        chunks=(16, 16),
        dtype=">u2" if dtype_store_endian == "big" else "<u2",
        compressor=None,
        fill_value=1,
    )
    z[:, :] = data
    assert_bytes_equal(await (store / "endian/0.0").get(), z._store["0.0"])


def test_invalid_metadata(store: Store):
    with pytest.raises(ValueError):
        Array.create(
            store / "invalid_chunk_shape",
            shape=(16, 16, 16),
            chunk_shape=(16, 16),
            dtype=np.dtype("uint8"),
            fill_value=0,
        )

    with pytest.raises(ValueError):
        Array.create(
            store / "invalid_endian",
            shape=(16, 16),
            chunk_shape=(16, 16),
            dtype=np.dtype("uint8"),
            fill_value=0,
            codecs=[
                BytesCodec(endian="big"),
                TransposeCodec(order=order_from_dim("F", 2)),
            ],
        )

    with pytest.raises(TypeError):
        Array.create(
            store / "invalid_order",
            shape=(16, 16),
            chunk_shape=(16, 16),
            dtype=np.dtype("uint8"),
            fill_value=0,
            codecs=[
                BytesCodec(),
                TransposeCodec(order="F"),
            ],
        )

    with pytest.raises(ValueError):
        Array.create(
            store / "invalid_missing_bytes_codec",
            shape=(16, 16),
            chunk_shape=(16, 16),
            dtype=np.dtype("uint8"),
            fill_value=0,
            codecs=[
                TransposeCodec(order=order_from_dim("F", 2)),
            ],
        )

    with pytest.raises(ValueError):
        Array.create(
            store / "invalid_inner_chunk_shape",
            shape=(16, 16),
            chunk_shape=(16, 16),
            dtype=np.dtype("uint8"),
            fill_value=0,
            codecs=[
                ShardingCodec(chunk_shape=(8,)),
            ],
        )
    with pytest.raises(ValueError):
        Array.create(
            store / "invalid_inner_chunk_shape",
            shape=(16, 16),
            chunk_shape=(16, 16),
            dtype=np.dtype("uint8"),
            fill_value=0,
            codecs=[
                ShardingCodec(chunk_shape=(8, 7)),
            ],
        )

    with pytest.warns(UserWarning):
        Array.create(
            store / "warning_inefficient_codecs",
            shape=(16, 16),
            chunk_shape=(16, 16),
            dtype=np.dtype("uint8"),
            fill_value=0,
            codecs=[
                ShardingCodec(chunk_shape=(8, 8)),
                GzipCodec(),
            ],
        )


async def test_resize(store: Store):
    data = np.zeros((16, 18), dtype="uint16")

    a = await AsyncArray.create(
        store / "resize",
        shape=data.shape,
        chunk_shape=(10, 10),
        dtype=data.dtype,
        chunk_key_encoding=("v2", "."),
        fill_value=1,
    )

    await _AsyncArrayProxy(a)[:16, :18].set(data)
    assert await (store / "resize" / "0.0").get() is not None
    assert await (store / "resize" / "0.1").get() is not None
    assert await (store / "resize" / "1.0").get() is not None
    assert await (store / "resize" / "1.1").get() is not None

    a = await a.resize((10, 12))
    assert a.metadata.shape == (10, 12)
    assert await (store / "resize" / "0.0").get() is not None
    assert await (store / "resize" / "0.1").get() is not None
    assert await (store / "resize" / "1.0").get() is None
    assert await (store / "resize" / "1.1").get() is None


async def test_blosc_evolve(store: Store):
    await AsyncArray.create(
        store / "blosc_evolve_u1",
        shape=(16, 16),
        chunk_shape=(16, 16),
        dtype="uint8",
        fill_value=0,
        codecs=[BytesCodec(), BloscCodec()],
    )

    zarr_json = json.loads((await (store / "blosc_evolve_u1" / "zarr.json").get()).to_bytes())
    blosc_configuration_json = zarr_json["codecs"][1]["configuration"]
    assert blosc_configuration_json["typesize"] == 1
    assert blosc_configuration_json["shuffle"] == "bitshuffle"

    await AsyncArray.create(
        store / "blosc_evolve_u2",
        shape=(16, 16),
        chunk_shape=(16, 16),
        dtype="uint16",
        fill_value=0,
        codecs=[BytesCodec(), BloscCodec()],
    )

    zarr_json = json.loads((await (store / "blosc_evolve_u2" / "zarr.json").get()).to_bytes())
    blosc_configuration_json = zarr_json["codecs"][1]["configuration"]
    assert blosc_configuration_json["typesize"] == 2
    assert blosc_configuration_json["shuffle"] == "shuffle"

    await AsyncArray.create(
        store / "sharding_blosc_evolve",
        shape=(16, 16),
        chunk_shape=(16, 16),
        dtype="uint16",
        fill_value=0,
        codecs=[ShardingCodec(chunk_shape=(16, 16), codecs=[BytesCodec(), BloscCodec()])],
    )

    zarr_json = json.loads((await (store / "sharding_blosc_evolve" / "zarr.json").get()).to_bytes())
    blosc_configuration_json = zarr_json["codecs"][0]["configuration"]["codecs"][1]["configuration"]
    assert blosc_configuration_json["typesize"] == 2
    assert blosc_configuration_json["shuffle"] == "shuffle"


def test_exists_ok(store: Store):
    Array.create(
        store / "exists_ok",
        shape=(16, 16),
        chunk_shape=(16, 16),
        dtype=np.dtype("uint8"),
    )
    with pytest.raises(AssertionError):
        Array.create(
            store / "exists_ok",
            shape=(16, 16),
            chunk_shape=(16, 16),
            dtype=np.dtype("uint8"),
        )
    Array.create(
        store / "exists_ok",
        shape=(16, 16),
        chunk_shape=(16, 16),
        dtype=np.dtype("uint8"),
        exists_ok=True,
    )


def test_update_attributes_array(store: Store):
    data = np.zeros((16, 18), dtype="uint16")

    a = Array.create(
        store / "update_attributes",
        shape=data.shape,
        chunk_shape=(10, 10),
        dtype=data.dtype,
        fill_value=1,
        attributes={"hello": "world"},
    )

    a = Array.open(store / "update_attributes")
    assert a.attrs["hello"] == "world"

    a.update_attributes({"hello": "zarrita"})

    a = Array.open(store / "update_attributes")
    assert a.attrs["hello"] == "zarrita"
