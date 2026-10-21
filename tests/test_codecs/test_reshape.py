import json

import numpy as np
import pytest

from zarr.codecs.reshape import ReshapeCodec
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.dtype import UInt8


def test_deserialize() -> None:
    s = '{"name": "reshape", "configuration": {"shape": [5, [1, 2], -1]}}'
    d = json.loads(s)
    ReshapeCodec.from_dict(d)


@pytest.mark.asyncio
async def test_encode() -> None:
    arr_flat = np.arange(60, dtype="uint8")
    arr_3x2 = np.reshape(arr_flat, (3, 4, 5))
    dt = UInt8()
    codec = ReshapeCodec(shape=((0, 1), 5))
    proto = default_buffer_prototype()
    (encoded, *_) = await codec.encode(
        [
            (
                proto.nd_buffer.from_numpy_array(arr_3x2),
                ArraySpec(
                    arr_3x2.shape,
                    dt,
                    fill_value=0,
                    config=ArrayConfig("C", False),
                    prototype=proto,
                ),
            )
        ]
    )
    assert encoded is not None
    assert encoded.shape == (12, 5)
    assert np.all(np.equal(np.ravel(encoded.as_numpy_array()), arr_flat))


@pytest.mark.asyncio
async def test_decode() -> None:
    arr_flat = np.arange(60, dtype="uint8")
    arr_12x5 = np.reshape(arr_flat, (12, 5))
    dt = UInt8()
    codec = ReshapeCodec(shape=((0, 1), 5))
    proto = default_buffer_prototype()
    (encoded, *_) = await codec.decode(
        [
            (
                proto.nd_buffer.from_numpy_array(arr_12x5),
                ArraySpec(
                    (3, 4, 5),
                    dt,
                    fill_value=0,
                    config=ArrayConfig("C", False),
                    prototype=proto,
                ),
            )
        ]
    )
    assert encoded is not None
    assert encoded.shape == (3, 4, 5)
    assert np.all(np.equal(np.ravel(encoded.as_numpy_array()), arr_flat))


@pytest.mark.asyncio
async def test_example() -> None:
    import json

    import zarr
    from zarr.storage import ManagedMemoryStore

    # Create an array with the reshape codec
    # This reshapes chunks from (100, 50, 64, 3) to (5000, 64, 3)
    store = ManagedMemoryStore()
    _ = zarr.create_array(
        store,
        shape=(1000, 500, 64, 3),
        chunks=(100, 50, 64, 3),
        dtype="float32",
        filters=[ReshapeCodec(shape=((0, 1), (2,), 3))],
    )
    b = await store.get("zarr.json")
    assert b is not None
    s = b.to_bytes().decode("utf-8")
    d = json.loads(s)
    assert d["codecs"][0] == {
        "name": "reshape",
        "configuration": {"shape": [[0, 1], [2], 3]},
    }
