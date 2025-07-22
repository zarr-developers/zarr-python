from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import zarr
import zarr.api
import zarr.api.asynchronous
from zarr import Array, AsyncArray, config
from zarr.codecs import (
    BytesCodec,
    GzipCodec,
    ShardingCodec,
    TransposeCodec,
)
from zarr.core.buffer import default_buffer_prototype
from zarr.core.indexing import BasicSelection, morton_order_iter
from zarr.core.metadata.v3 import ArrayV3Metadata
from zarr.storage import StorePath

if TYPE_CHECKING:
    from zarr.abc.store import Store
    from zarr.core.buffer.core import NDArrayLikeOrScalar
    from zarr.core.common import ChunkCoords, MemoryOrder


@dataclass(frozen=True)
class _AsyncArrayProxy:
    array: AsyncArray[Any]

    def __getitem__(self, selection: BasicSelection) -> _AsyncArraySelectionProxy:
        return _AsyncArraySelectionProxy(self.array, selection)


@dataclass(frozen=True)
class _AsyncArraySelectionProxy:
    array: AsyncArray[Any]
    selection: BasicSelection

    async def get(self) -> NDArrayLikeOrScalar:
        return await self.array.getitem(self.selection)

    async def set(self, value: np.ndarray[Any, Any]) -> None:
        return await self.array.setitem(self.selection, value)


def order_from_dim(order: MemoryOrder, ndim: int) -> tuple[int, ...]:
    if order == "F":
        return tuple(ndim - x - 1 for x in range(ndim))
    else:
        return tuple(range(ndim))


def test_sharding_pickle() -> None:
    """
    Test that sharding codecs can be pickled
    """


@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
@pytest.mark.parametrize("input_order", ["F", "C"])
@pytest.mark.parametrize("store_order", ["F", "C"])
@pytest.mark.parametrize("runtime_write_order", ["F", "C"])
@pytest.mark.parametrize("runtime_read_order", ["F", "C"])
@pytest.mark.parametrize("with_sharding", [True, False])
async def test_order(
    store: Store,
    input_order: MemoryOrder,
    store_order: MemoryOrder,
    runtime_write_order: MemoryOrder,
    runtime_read_order: MemoryOrder,
    with_sharding: bool,
) -> None:
    data = np.arange(0, 256, dtype="uint16").reshape((32, 8), order=input_order)
    path = "order"
    spath = StorePath(store, path=path)

    a = await zarr.api.asynchronous.create_array(
        spath,
        shape=data.shape,
        chunks=(16, 8) if with_sharding else (32, 8),
        shards=(32, 8) if with_sharding else None,
        dtype=data.dtype,
        fill_value=0,
        chunk_key_encoding={"name": "v2", "separator": "."},
        filters=[TransposeCodec(order=order_from_dim(store_order, data.ndim))],
        config={"order": runtime_write_order},
    )

    await _AsyncArrayProxy(a)[:, :].set(data)
    read_data = await _AsyncArrayProxy(a)[:, :].get()
    assert np.array_equal(data, read_data)

    with config.set({"array.order": runtime_read_order}):
        a = await AsyncArray.open(
            spath,
        )
    read_data = await _AsyncArrayProxy(a)[:, :].get()
    assert np.array_equal(data, read_data)

    assert isinstance(read_data, np.ndarray)
    if runtime_read_order == "F":
        assert read_data.flags["F_CONTIGUOUS"]
        assert not read_data.flags["C_CONTIGUOUS"]
    else:
        assert not read_data.flags["F_CONTIGUOUS"]
        assert read_data.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
@pytest.mark.parametrize("input_order", ["F", "C"])
@pytest.mark.parametrize("runtime_write_order", ["F", "C"])
@pytest.mark.parametrize("runtime_read_order", ["F", "C"])
@pytest.mark.parametrize("with_sharding", [True, False])
def test_order_implicit(
    store: Store,
    input_order: MemoryOrder,
    runtime_write_order: MemoryOrder,
    runtime_read_order: MemoryOrder,
    with_sharding: bool,
) -> None:
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16), order=input_order)
    path = "order_implicit"
    spath = StorePath(store, path)

    with config.set({"array.order": runtime_write_order}):
        a = zarr.create_array(
            spath,
            shape=data.shape,
            chunks=(8, 8) if with_sharding else (16, 16),
            shards=(16, 16) if with_sharding else None,
            dtype=data.dtype,
            fill_value=0,
        )

    a[:, :] = data

    with config.set({"array.order": runtime_read_order}):
        a = Array.open(spath)
    read_data = a[:, :]
    assert np.array_equal(data, read_data)

    assert isinstance(read_data, np.ndarray)
    if runtime_read_order == "F":
        assert read_data.flags["F_CONTIGUOUS"]
        assert not read_data.flags["C_CONTIGUOUS"]
    else:
        assert not read_data.flags["F_CONTIGUOUS"]
        assert read_data.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
def test_open(store: Store) -> None:
    spath = StorePath(store)
    a = zarr.create_array(
        spath,
        shape=(16, 16),
        chunks=(16, 16),
        dtype="int32",
        fill_value=0,
    )
    b = Array.open(spath)
    assert a.metadata == b.metadata


def test_morton() -> None:
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


@pytest.mark.parametrize(
    "shape",
    [
        [2, 2, 2],
        [5, 2],
        [2, 5],
        [2, 9, 2],
        [3, 2, 12],
        [2, 5, 1],
        [4, 3, 6, 2, 7],
        [3, 2, 1, 6, 4, 5, 2],
    ],
)
def test_morton2(shape: ChunkCoords) -> None:
    order = list(morton_order_iter(shape))
    for i, x in enumerate(order):
        assert x not in order[:i]  # no duplicates
        assert all(x[j] < shape[j] for j in range(len(shape)))  # all indices are within bounds


@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
def test_write_partial_chunks(store: Store) -> None:
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))
    spath = StorePath(store)
    a = zarr.create_array(
        spath,
        shape=data.shape,
        chunks=(20, 20),
        dtype=data.dtype,
        fill_value=1,
    )
    a[0:16, 0:16] = data
    assert np.array_equal(a[0:16, 0:16], data)


@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
async def test_delete_empty_chunks(store: Store) -> None:
    data = np.ones((16, 16))
    path = "delete_empty_chunks"
    spath = StorePath(store, path)
    a = await zarr.api.asynchronous.create_array(
        spath,
        shape=data.shape,
        chunks=(32, 32),
        dtype=data.dtype,
        fill_value=1,
    )
    await _AsyncArrayProxy(a)[:16, :16].set(np.zeros((16, 16)))
    await _AsyncArrayProxy(a)[:16, :16].set(data)
    assert np.array_equal(await _AsyncArrayProxy(a)[:16, :16].get(), data)
    assert await store.get(f"{path}/c0/0", prototype=default_buffer_prototype()) is None


@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
async def test_dimension_names(store: Store) -> None:
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))
    path = "dimension_names"
    spath = StorePath(store, path)
    await zarr.api.asynchronous.create_array(
        spath,
        shape=data.shape,
        chunks=(16, 16),
        dtype=data.dtype,
        fill_value=0,
        dimension_names=("x", "y"),
    )

    assert isinstance(
        meta := (await zarr.api.asynchronous.open_array(store=spath)).metadata, ArrayV3Metadata
    )
    assert meta.dimension_names == (
        "x",
        "y",
    )
    path2 = "dimension_names2"
    spath2 = StorePath(store, path2)
    await zarr.api.asynchronous.create_array(
        spath2,
        shape=data.shape,
        chunks=(16, 16),
        dtype=data.dtype,
        fill_value=0,
    )

    assert isinstance(meta := (await AsyncArray.open(spath2)).metadata, ArrayV3Metadata)
    assert meta.dimension_names is None
    zarr_json_buffer = await store.get(f"{path2}/zarr.json", prototype=default_buffer_prototype())
    assert zarr_json_buffer is not None
    assert "dimension_names" not in json.loads(zarr_json_buffer.to_bytes())


@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
def test_invalid_metadata(store: Store) -> None:
    spath2 = StorePath(store, "invalid_codec_order")
    with pytest.raises(TypeError):
        Array.create(
            spath2,
            shape=(16, 16),
            chunk_shape=(16, 16),
            dtype=np.dtype("uint8"),
            fill_value=0,
            codecs=[
                BytesCodec(),
                TransposeCodec(order=order_from_dim("F", 2)),
            ],
        )
    spath3 = StorePath(store, "invalid_order")
    with pytest.raises(TypeError):
        Array.create(
            spath3,
            shape=(16, 16),
            chunk_shape=(16, 16),
            dtype=np.dtype("uint8"),
            fill_value=0,
            codecs=[
                TransposeCodec(order="F"),  # type: ignore[arg-type]
                BytesCodec(),
            ],
        )
    spath4 = StorePath(store, "invalid_missing_bytes_codec")
    with pytest.raises(ValueError):
        Array.create(
            spath4,
            shape=(16, 16),
            chunk_shape=(16, 16),
            dtype=np.dtype("uint8"),
            fill_value=0,
            codecs=[
                TransposeCodec(order=order_from_dim("F", 2)),
            ],
        )
    spath5 = StorePath(store, "invalid_inner_chunk_shape")
    with pytest.raises(ValueError):
        Array.create(
            spath5,
            shape=(16, 16),
            chunk_shape=(16, 16),
            dtype=np.dtype("uint8"),
            fill_value=0,
            codecs=[
                ShardingCodec(chunk_shape=(8,)),
            ],
        )
    spath6 = StorePath(store, "invalid_inner_chunk_shape")
    with pytest.raises(ValueError):
        Array.create(
            spath6,
            shape=(16, 16),
            chunk_shape=(16, 16),
            dtype=np.dtype("uint8"),
            fill_value=0,
            codecs=[
                ShardingCodec(chunk_shape=(8, 7)),
            ],
        )
    spath7 = StorePath(store, "warning_inefficient_codecs")
    with pytest.warns(
        UserWarning,
        match="Combining a `sharding_indexed` codec disables partial reads and writes, which may lead to inefficient performance",
    ):
        Array.create(
            spath7,
            shape=(16, 16),
            chunk_shape=(16, 16),
            dtype=np.dtype("uint8"),
            fill_value=0,
            codecs=[
                ShardingCodec(chunk_shape=(8, 8)),
                GzipCodec(),
            ],
        )


@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
def test_invalid_metadata_create_array(store: Store) -> None:
    spath = StorePath(store, "warning_inefficient_codecs")
    with pytest.warns(
        UserWarning,
        match="codec disables partial reads and writes, which may lead to inefficient performance",
    ):
        zarr.create_array(
            spath,
            shape=(16, 16),
            chunks=(16, 16),
            dtype=np.dtype("uint8"),
            fill_value=0,
            serializer=ShardingCodec(chunk_shape=(8, 8)),
            compressors=[
                GzipCodec(),
            ],
        )


@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
async def test_resize(store: Store) -> None:
    data = np.zeros((16, 18), dtype="uint16")
    path = "resize"
    spath = StorePath(store, path)
    a = await zarr.api.asynchronous.create_array(
        spath,
        shape=data.shape,
        chunks=(10, 10),
        dtype=data.dtype,
        chunk_key_encoding={"name": "v2", "separator": "."},
        fill_value=1,
    )

    await _AsyncArrayProxy(a)[:16, :18].set(data)
    assert await store.get(f"{path}/1.1", prototype=default_buffer_prototype()) is not None
    assert await store.get(f"{path}/0.0", prototype=default_buffer_prototype()) is not None
    assert await store.get(f"{path}/0.1", prototype=default_buffer_prototype()) is not None
    assert await store.get(f"{path}/1.0", prototype=default_buffer_prototype()) is not None

    await a.resize((10, 12))
    assert a.metadata.shape == (10, 12)
    assert a.shape == (10, 12)
    assert await store.get(f"{path}/0.0", prototype=default_buffer_prototype()) is not None
    assert await store.get(f"{path}/0.1", prototype=default_buffer_prototype()) is not None
    assert await store.get(f"{path}/1.0", prototype=default_buffer_prototype()) is None
    assert await store.get(f"{path}/1.1", prototype=default_buffer_prototype()) is None
