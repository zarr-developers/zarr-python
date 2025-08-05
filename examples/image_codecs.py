# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "zarr @ git+https://github.com/d-v-b/zarr-python.git@a2bc6555",
#   "imagecodecs==2025.3.30",
#   "pytest"
# ]
# ///

#   "zarr @ git+https://github.com/zarr-developers/zarr-python.git@main",
from typing import Literal

import numcodecs
import numpy as np
import pytest
from imagecodecs.numcodecs import Jpeg

import zarr

numcodecs.register_codec(Jpeg)
jpg_codec = Jpeg()


@pytest.mark.parametrize("zarr_format", [2, 3])
def test(zarr_format: Literal[2, 3]) -> None:
    store = {}
    if zarr_format == 2:
        z_w = zarr.create_array(
            store=store,
            data=np.zeros((100, 100, 3), dtype=np.uint8),
            compressors=jpg_codec,
            zarr_format=zarr_format,
        )
    else:
        z_w = zarr.create_array(
            store=store,
            data=np.zeros((100, 100, 3), dtype=np.uint8),
            serializer=jpg_codec,
            zarr_format=zarr_format,
        )
    z_w[:] = 2
    z_r = zarr.open_array(store=store, zarr_format=zarr_format)
    assert np.all(z_r[:] == 2)
    if zarr_format == 2:
        print(z_r.metadata.to_dict()["compressor"])
    else:
        print(z_r.metadata.to_dict()["codecs"])


if __name__ == "__main__":
    pytest.main([__file__, f"-c {__file__}", "-s"])
