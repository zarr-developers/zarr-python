import json

import numpy as np
import pytest

from zarr import AsyncArray
from zarr.abc.store import Store
from zarr.codecs import BloscCodec, BytesCodec, ShardingCodec
from zarr.core.buffer import default_buffer_prototype
from zarr.storage.common import StorePath


@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
async def test_blosc_evolve(store: Store, dtype: str) -> None:
    typesize = np.dtype(dtype).itemsize
    path = "blosc_evolve"
    spath = StorePath(store, path)
    await AsyncArray.create(
        spath,
        shape=(16, 16),
        chunk_shape=(16, 16),
        dtype=dtype,
        fill_value=0,
        codecs=[BytesCodec(), BloscCodec()],
    )
    buf = await store.get(f"{path}/zarr.json", prototype=default_buffer_prototype())
    assert buf is not None
    zarr_json = json.loads(buf.to_bytes())
    blosc_configuration_json = zarr_json["codecs"][1]["configuration"]
    assert blosc_configuration_json["typesize"] == typesize
    if typesize == 1:
        assert blosc_configuration_json["shuffle"] == "bitshuffle"
    else:
        assert blosc_configuration_json["shuffle"] == "shuffle"

    path2 = "blosc_evolve_sharding"
    spath2 = StorePath(store, path2)
    await AsyncArray.create(
        spath2,
        shape=(16, 16),
        chunk_shape=(16, 16),
        dtype=dtype,
        fill_value=0,
        codecs=[ShardingCodec(chunk_shape=(16, 16), codecs=[BytesCodec(), BloscCodec()])],
    )
    buf = await store.get(f"{path2}/zarr.json", prototype=default_buffer_prototype())
    assert buf is not None
    zarr_json = json.loads(buf.to_bytes())
    blosc_configuration_json = zarr_json["codecs"][0]["configuration"]["codecs"][1]["configuration"]
    assert blosc_configuration_json["typesize"] == typesize
    if typesize == 1:
        assert blosc_configuration_json["shuffle"] == "bitshuffle"
    else:
        assert blosc_configuration_json["shuffle"] == "shuffle"
