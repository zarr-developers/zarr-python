"""Composing views keeps indexing lazy until the final result call."""

import numpy as np

from zarr_indexing import LazyArray


# --8<-- [start:lazy-composition]
image = np.arange(12).reshape(3, 4)
view = LazyArray(image).with_parts((2, 2)).lazy[1, 0:4]
composed = view.lazy[::-1].lazy[1:]

assert composed.result().tolist() == image[1, 0:4][::-1][1:].tolist()
# --8<-- [end:lazy-composition]
