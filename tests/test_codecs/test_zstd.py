import numpy as np
import pytest

from zarr import Array
from zarr.abc.store import Store
from zarr.codecs import BytesCodec, ZstdCodec
from zarr.storage.common import StorePath


@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
@pytest.mark.parametrize("checksum", [True, False])
def test_zstd(store: Store, checksum: bool) -> None:
    data = np.arange(0, 256, dtype="uint16").reshape((16, 16))

    a = Array.create(
        StorePath(store, path="zstd"),
        shape=data.shape,
        chunk_shape=(16, 16),
        dtype=data.dtype,
        fill_value=0,
        codecs=[BytesCodec(), ZstdCodec(level=0, checksum=checksum)],
    )

    a[:, :] = data
    assert np.array_equal(data, a[:, :])
