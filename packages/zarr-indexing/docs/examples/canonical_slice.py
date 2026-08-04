"""The canonical basic-selection example used throughout the guide."""

import numpy as np

from zarr_indexing import LazyArray


# --8<-- [start:landing-quickstart]
import numpy as np

from zarr_indexing import LazyArray

source = np.array([10, 11, 12, 13, 14, 15])
view = LazyArray.from_numpy(source).lazy[2:5]

view.result()
# array([12, 13, 14])
# --8<-- [end:landing-quickstart]

LANDING_QUICKSTART_RESULT = view.result()
assert LANDING_QUICKSTART_RESULT.tolist() == [12, 13, 14]


# --8<-- [start:canonical-slice]
source = np.array([10, 11, 12, 13, 14, 15])
lazy = LazyArray.from_numpy(source)
view = lazy.lazy[2:5]

assert view.result().tolist() == [12, 13, 14]
# --8<-- [end:canonical-slice]
