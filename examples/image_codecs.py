# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "zarr @ file:///home/bennettd/dev/zarr-python/",
#   "imagecodecs==2025.3.30"
# ]
# ///

#   "zarr @ git+https://github.com/zarr-developers/zarr-python.git@main",

import numcodecs
import numpy as np
from imagecodecs.numcodecs import Jpeg

import zarr

numcodecs.register_codec(Jpeg)
jpg_codec = Jpeg()
store = {}

z_w = zarr.create_array(
    store=store, data=np.zeros((100, 100, 3), dtype=np.uint8), serializer=jpg_codec, zarr_format=3
)

# breakpoint()

z_r = zarr.open_array(store=store, zarr_format=3)

print(z_r.metadata.to_dict()["codecs"])
