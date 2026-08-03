"""Composing views keeps indexing lazy until the final result call."""

import numpy as np

from zarr_indexing import LazyArray


# --8<-- [start:lazy-composition]
source = np.array([10, 11, 12, 13, 14, 15])
view = LazyArray(source).with_parts((2,)).lazy[2:5]
composed = view.lazy[::-1].lazy[1:]

assert composed.result().tolist() == source[2:5][::-1][1:].tolist()
# --8<-- [end:lazy-composition]
