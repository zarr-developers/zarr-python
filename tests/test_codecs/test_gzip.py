import numpy as np
import pytest

import zarr
from zarr.abc.store import Store
from zarr.codecs import GzipCodec
from zarr.codecs.gzip import GZipJSON_V2, GZipJSON_V3
from zarr.storage import StorePath

@pytest.mark.parametrize("level", [1, 5, 9])
def test_json(level: int) -> None:
    codec = GzipCodec(level=level)
    expected_v2: GZipJSON_V2 = {
        "id": "gzip",
        "level": level,
    }
    expected_v3: GZipJSON_V3 = {
        "name": "gzip",
        "configuration": {
            "level": level,
        },
    }
    assert codec.to_json(zarr_format=2) == expected_v2
    assert codec.to_json(zarr_format=3) == expected_v3


@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
def test_gzip(store: Store) -> None:
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    a = zarr.create_array(
        StorePath(store),
        shape=data.shape,
        chunks=(16, 16),
        dtype=data.dtype,
        fill_value=0,
        compressors=GzipCodec(),
    )

    a[:, :] = data
    assert np.array_equal(data, a[:, :])
