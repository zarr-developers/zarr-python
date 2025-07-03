import pytest

import zarr
from zarr.abc.store import Store
from zarr.codecs import NvcompZstdCodec
from zarr.storage import StorePath
from zarr.testing.utils import gpu_test


@gpu_test
@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
@pytest.mark.parametrize(
    "checksum",
    [
        False,
    ],
)
def test_nvcomp_zstd(store: Store, checksum: bool) -> None:
    import cupy as cp

    with zarr.config.enable_gpu():
        data = cp.arange(0, 256, dtype="uint16").reshape((16, 16))

        a = zarr.create_array(
            StorePath(store, path="nvcomp_zstd"),
            shape=data.shape,
            chunks=(16, 16),
            dtype=data.dtype,
            fill_value=0,
            compressors=NvcompZstdCodec(level=0, checksum=checksum),
        )

        a[:, :] = data
        cp.testing.assert_array_equal(data, a[:, :])
