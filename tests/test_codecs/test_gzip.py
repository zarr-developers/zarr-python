import numpy as np
import pytest

import zarr
from tests.test_codecs.conftest import BaseTestCodec
from zarr.abc.store import Store
from zarr.codecs import GzipCodec
from zarr.storage import StorePath


class TestGZipCodec(BaseTestCodec):
    test_cls = GzipCodec
    valid_json_v2 = (
        {
            "id": "gzip",
            "level": 1,
        },
    )
    valid_json_v3 = (
        {
            "name": "gzip",
            "configuration": {
                "level": 1,
            },
        },
    )


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
