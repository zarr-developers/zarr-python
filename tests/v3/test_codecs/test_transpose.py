from typing import TYPE_CHECKING

import numpy as np
import pytest

from zarr import Array, AsyncArray, config
from zarr.abc.store import Store
from zarr.codecs import BytesCodec, ShardingCodec, TransposeCodec
from zarr.core.common import MemoryOrder
from zarr.storage.common import StorePath

from .test_codecs import _AsyncArrayProxy

if TYPE_CHECKING:
    from zarr.abc.codec import Codec


@pytest.mark.parametrize("input_order", ["F", "C"])
@pytest.mark.parametrize("runtime_write_order", ["F", "C"])
@pytest.mark.parametrize("runtime_read_order", ["F", "C"])
@pytest.mark.parametrize("with_sharding", [True, False])
@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
async def test_transpose(
    store: Store,
    input_order: MemoryOrder,
    runtime_write_order: MemoryOrder,
    runtime_read_order: MemoryOrder,
    with_sharding: bool,
) -> None:
    data = np.arange(0, 256, dtype="uint16").reshape((1, 32, 8), order=input_order)
    spath = StorePath(store, path="transpose")
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
            spath,
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
            spath,
        )
    read_data = await _AsyncArrayProxy(a)[:, :].get()
    assert np.array_equal(data, read_data)

    if runtime_read_order == "F":
        assert read_data.flags["F_CONTIGUOUS"]
        assert not read_data.flags["C_CONTIGUOUS"]
    else:
        assert not read_data.flags["F_CONTIGUOUS"]
        assert read_data.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
@pytest.mark.parametrize("order", [[1, 2, 0], [1, 2, 3, 0], [3, 2, 4, 0, 1]])
def test_transpose_non_self_inverse(store: Store, order: list[int]) -> None:
    shape = [i + 3 for i in range(len(order))]
    data = np.arange(0, np.prod(shape), dtype="uint16").reshape(shape)
    spath = StorePath(store, "transpose_non_self_inverse")
    a = Array.create(
        spath,
        shape=data.shape,
        chunk_shape=data.shape,
        dtype=data.dtype,
        fill_value=0,
        codecs=[TransposeCodec(order=order), BytesCodec()],
    )
    a[:, :] = data
    read_data = a[:, :]
    assert np.array_equal(data, read_data)


@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
def test_transpose_invalid(
    store: Store,
) -> None:
    data = np.arange(0, 256, dtype="uint16").reshape((1, 32, 8))
    spath = StorePath(store, "transpose_invalid")
    for order in [(1, 0), (3, 2, 1), (3, 3, 1)]:
        with pytest.raises(ValueError):
            Array.create(
                spath,
                shape=data.shape,
                chunk_shape=(1, 32, 8),
                dtype=data.dtype,
                fill_value=0,
                chunk_key_encoding=("v2", "."),
                codecs=[TransposeCodec(order=order), BytesCodec()],
            )
