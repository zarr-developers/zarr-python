import json
from typing import Any, Literal

import numcodecs
import numpy as np
import pytest
from packaging.version import Version

import zarr
import zarr.codecs.blosc as blosc
import zarr.codecs.numcodecs
from tests.test_codecs.conftest import BaseTestCodec
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.common import ZarrFormat
from zarr.core.dtype import UInt16
from zarr.core.dtype.npy.int import Int64
from zarr.errors import ZarrDeprecationWarning
from zarr.storage import StorePath
from zarr.storage._memory import MemoryStore


class TestBloscCodec(BaseTestCodec):
    test_cls = blosc.BloscCodec
    valid_json_v2 = (
        {
            "id": "blosc",
            "cname": "lz4",
            "clevel": 5,
            "shuffle": 1,
            "blocksize": 0,
        },
    )
    valid_json_v3 = (
        {
            "name": "blosc",
            "configuration": {
                "cname": "lz4",
                "clevel": 5,
                "shuffle": "shuffle",
                "blocksize": 0,
                "typesize": 1,
            },
        },
        {
            "name": "blosc",
            "configuration": {
                "cname": "lz4",
                "clevel": 5,
                "shuffle": 1,
                "blocksize": 0,
                "typesize": 1,
            },
        },
    )

    @staticmethod
    def check_json_v2(data: object) -> bool:
        return blosc.check_json_v2(data)

    @staticmethod
    def check_json_v3(data: object) -> bool:
        return blosc.check_json_v3(data)


@pytest.mark.parametrize("shuffle", blosc.SHUFFLE)
@pytest.mark.parametrize("cname", blosc.CNAME)
@pytest.mark.parametrize("clevel", [1, 2])
@pytest.mark.parametrize("blocksize", [1, 2])
@pytest.mark.parametrize("typesize", [1, 2])
def test_to_json_v2(
    cname: blosc.CName, shuffle: blosc.Shuffle, clevel: int, blocksize: int, typesize: int
) -> None:
    codec = blosc.BloscCodec(
        shuffle=shuffle, cname=cname, clevel=clevel, blocksize=blocksize, typesize=typesize
    )
    expected_v2: blosc.BloscJSON_V2 = {
        "id": "blosc",
        "cname": cname,
        "clevel": clevel,
        "shuffle": blosc.SHUFFLE.index(shuffle),
        "blocksize": blocksize,
    }
    expected_v3: blosc.BloscJSON_V3 = {
        "name": "blosc",
        "configuration": {
            "cname": cname,
            "clevel": clevel,
            "shuffle": shuffle,
            "blocksize": blocksize,
            "typesize": typesize,
        },
    }
    assert codec.to_json(zarr_format=2) == expected_v2
    assert codec.to_json(zarr_format=3) == expected_v3


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize(
    "codec_type",
    [
        "legacy_zarr3",
        "numcodecs",
    ],
)
def test_blosc_compression(
    zarr_format: ZarrFormat, codec_type: Literal["legacy_zarr3", "numcodecs"]
) -> None:
    """
    Test that any of the blosc-like codecs can be used for compression, and that
    reading the array back uses the primary blosc codec class.
    """
    ref_codec = blosc.BloscCodec(cname="lz4", clevel=5, shuffle="shuffle", blocksize=0, typesize=8)
    if codec_type == "legacy_zarr3":
        with pytest.warns(ZarrDeprecationWarning):
            codec: zarr.codecs.numcodecs.Blosc = zarr.codecs.numcodecs.Blosc(
                cname="lz4", clevel=5, shuffle=1, blocksize=0
            )
    elif codec_type == "numcodecs":
        codec = numcodecs.Blosc(cname="lz4", clevel=5, shuffle=1, blocksize=0, typesize=8)
    else:
        raise ValueError(f"Unknown codec_type: {codec_type}")
    store: dict[str, Any] = {}
    z_w = zarr.create_array(
        store=store,
        dtype=Int64(),
        shape=(1,),
        chunks=(10,),
        zarr_format=zarr_format,
        compressors=codec,
    )
    z_w[:] = 5

    z_r = zarr.open_array(store=store, zarr_format=zarr_format)
    assert np.all(z_r[:] == 5)
    assert z_r.compressors == (ref_codec,)


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
        compressors=blosc.BloscCodec(),
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
        compressors=blosc.BloscCodec(),
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


@pytest.mark.parametrize("shuffle", [None, "bitshuffle", blosc.BloscShuffle.shuffle])
@pytest.mark.parametrize("typesize", [None, 1, 2])
def test_tunable_attrs_param(
    shuffle: None | blosc.Shuffle | blosc.BloscShuffle, typesize: None | int
) -> None:
    """
    Test that the tunable_attrs parameter is set as expected when creating a BloscCodec,
    """
    codec = blosc.BloscCodec(typesize=typesize, shuffle=shuffle)

    if shuffle is None:
        assert codec.shuffle == blosc.BloscShuffle.bitshuffle  # default shuffle
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
        assert evolved_codec.shuffle == blosc.BloscShuffle.shuffle
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
