"""The canonical basic-selection example used throughout the guide."""

import numpy as np

from zarr_indexing import LazyArray


# --8<-- [start:landing-quickstart]
import numpy as np

from zarr_indexing import LazyArray

image = np.arange(12).reshape(3, 4)
view = LazyArray(image).lazy[1, 0:4]

view.result()
# array([4, 5, 6, 7])
# --8<-- [end:landing-quickstart]

LANDING_QUICKSTART_RESULT = view.result()
assert LANDING_QUICKSTART_RESULT.tolist() == [4, 5, 6, 7]


# --8<-- [start:canonical-slice]
image = np.arange(12).reshape(3, 4)
lazy = LazyArray(image).with_parts((2, 2))
view = lazy.lazy[1, 0:4]

assert view.result().tolist() == [4, 5, 6, 7]
# --8<-- [end:canonical-slice]
