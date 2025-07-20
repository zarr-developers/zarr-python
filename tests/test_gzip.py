import json

import pytest

import zarr
from zarr.codecs import GzipCodec
from zarr.core.common import ZarrFormat


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_gzip_compression(zarr_format):
    store = {}
    arr_in = zarr.create_array(
        store=store,
        dtype="int",
        shape=(1,),
        chunks=(10,),
        zarr_format=zarr_format,
        compressors=GzipCodec(),
    )

    if zarr_format == 2:
        print(json.dumps(json.loads(store[".zarray"].to_bytes()), indent=2))
    else:
        print(json.dumps(json.loads(store["zarr.json"].to_bytes()), indent=2))

    arr_out = zarr.open_array(store=store, zarr_format=zarr_format)
