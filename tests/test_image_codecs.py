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

z_r = zarr.open_array(store=store, zarr_format=3)

print(z_r.metadata.to_dict()["codecs"])
