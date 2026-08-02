"""Composing views keeps indexing lazy until the final result call."""

import numpy as np
from typing import Any, cast

from zarr_indexing import LazyArray


# --8<-- [start:lazy-composition]
image = np.arange(48).reshape(6, 8)
view = LazyArray(cast(Any, image)).with_parts((3, 4)).lazy[1:5, 2]
composed = view.lazy[::-1].lazy[1:]

assert composed.result().tolist() == image[1:5, 2][::-1][1:].tolist()
# --8<-- [end:lazy-composition]
