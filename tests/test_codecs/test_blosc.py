import json

import numcodecs
import numpy as np
import pytest
from packaging.version import Version

import zarr
from zarr.codecs import BloscCodec
from zarr.codecs.blosc import BloscShuffle, Shuffle
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.dtype import UInt16
from zarr.storage import MemoryStore, StorePath


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
async def test_blosc_evolve(dtype: str) -> None:
    typesize = np.dtype(dtype).itemsize
    path = "blosc_evolve"
    store = MemoryStore()
    spath = StorePath(store, path)
    zarr.create_array(
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
    zarr.create_array(
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


@pytest.mark.parametrize("shuffle", [None, "bitshuffle", BloscShuffle.shuffle])
@pytest.mark.parametrize("typesize", [None, 1, 2])
def test_tunable_attrs_param(shuffle: None | Shuffle | BloscShuffle, typesize: None | int) -> None:
    """
    Test that the tunable_attrs parameter is set as expected when creating a BloscCodec,
    """
    codec = BloscCodec(typesize=typesize, shuffle=shuffle)

    if shuffle is None:
        assert codec.shuffle == BloscShuffle.bitshuffle  # default shuffle
        assert "shuffle" in codec._tunable_attrs
    if typesize is None:
        assert codec.typesize == 1  # default typesize
        assert "typesize" in codec._tunable_attrs

    new_dtype = UInt16()
    array_spec = ArraySpec(
        shape=(1,),
        dtype=new_dtype,
        fill_value=1,
        prototype=default_buffer_prototype(),
        config={},  # type: ignore[arg-type]
    )

    evolved_codec = codec.evolve_from_array_spec(array_spec=array_spec)
    if typesize is None:
        assert evolved_codec.typesize == new_dtype.item_size
    else:
        assert evolved_codec.typesize == codec.typesize
    if shuffle is None:
        assert evolved_codec.shuffle == BloscShuffle.shuffle
    else:
        assert evolved_codec.shuffle == codec.shuffle


async def test_typesize() -> None:
    a = np.arange(1000000, dtype=np.uint64)
    codecs = [zarr.codecs.BytesCodec(), zarr.codecs.BloscCodec()]
    z = zarr.array(a, chunks=(10000), codecs=codecs)
    data = await z.store.get("c/0", prototype=default_buffer_prototype())
    assert data is not None
    bytes = data.to_bytes()
    size = len(bytes)
    msg = f"Blosc size mismatch.  First 10 bytes: {bytes[:20]!r} and last 10 bytes: {bytes[-20:]!r}"
    if Version(numcodecs.__version__) >= Version("0.16.0"):
        expected_size = 402
        assert size == expected_size, msg
    else:
        expected_size = 10216
    assert size == expected_size, msg
