import json

import numcodecs
import numpy as np
import pytest
from packaging.version import Version

import zarr
from zarr.abc.store import Store
from zarr.codecs import BloscCodec
from zarr.core.buffer import default_buffer_prototype
from zarr.storage import StorePath


@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
async def test_blosc_evolve(store: Store, dtype: str) -> None:
    typesize = np.dtype(dtype).itemsize
    path = "blosc_evolve"
    spath = StorePath(store, path)
    await zarr.api.asynchronous.create_array(
        spath,
        shape=(16, 16),
        chunks=(16, 16),
        dtype=dtype,
        fill_value=0,
        compressors=BloscCodec(),
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
    await zarr.api.asynchronous.create_array(
        spath2,
        shape=(16, 16),
        chunks=(16, 16),
        shards=(16, 16),
        dtype=dtype,
        fill_value=0,
        compressors=BloscCodec(),
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


async def test_typesize() -> None:
    a = np.arange(1000000)
    codecs = [zarr.codecs.BytesCodec(), zarr.codecs.BloscCodec()]
    z3 = zarr.array(a, chunks=(10000), codecs=codecs)
    v3_size = len(await z3.store.get("c/0", prototype=default_buffer_prototype()))
    assert v3_size == 402 if Version(numcodecs.__version__) >= Version("0.16.0") else 10216
