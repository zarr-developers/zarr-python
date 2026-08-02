"""The canonical basic-selection example used throughout the guide."""

import numpy as np
from typing import Any, cast

from zarr_indexing import LazyArray


# --8<-- [start:landing-quickstart]
import numpy as np
from typing import Any, cast

from zarr_indexing import LazyArray

image = np.arange(48).reshape(6, 8)
view = LazyArray(cast(Any, image)).lazy[1:5, 2]

view.result()
# array([10, 18, 26, 34])
# --8<-- [end:landing-quickstart]

LANDING_QUICKSTART_RESULT = view.result()
assert LANDING_QUICKSTART_RESULT.tolist() == [10, 18, 26, 34]


# --8<-- [start:canonical-slice]
image = np.arange(48).reshape(6, 8)
lazy = LazyArray(cast(Any, image)).with_parts((3, 4))
view = lazy.lazy[1:5, 2]

assert view.result().tolist() == [10, 18, 26, 34]
# --8<-- [end:canonical-slice]
