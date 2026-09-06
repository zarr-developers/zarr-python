"""Indexing defines result axes as well as selected source points."""

import numpy as np

from zarr_indexing import LazyArray

# --8<-- [start:axis-shape-comparison]
image = np.arange(12).reshape(3, 4)
lazy = LazyArray.from_numpy(image)

integer_view = lazy.lazy[1, :]
slice_view = lazy.lazy[1:2, :]

INTEGER_RESULT = integer_view.result()
SLICE_RESULT = slice_view.result()

assert INTEGER_RESULT.tolist() == [4, 5, 6, 7]
assert INTEGER_RESULT.shape == (4,)
assert SLICE_RESULT.tolist() == [[4, 5, 6, 7]]
assert SLICE_RESULT.shape == (1, 4)
# --8<-- [end:axis-shape-comparison]

# --8<-- [start:axis-insertion]
inserted_view = lazy.lazy[1, :, None]
INSERTED_RESULT = inserted_view.result()

assert INSERTED_RESULT.tolist() == [[4], [5], [6], [7]]
assert INSERTED_RESULT.shape == (4, 1)
# --8<-- [end:axis-insertion]
