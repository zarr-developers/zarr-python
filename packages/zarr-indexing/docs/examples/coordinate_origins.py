"""Literal coordinate domains and a grid that supports prepending."""

import numpy as np
from numpy.typing import NDArray
from typing import cast

from zarr_indexing import (
    DimensionGridLike,
    IndexDomain,
    IndexTransform,
    LazyArray,
    plan_chunks,
)


# --8<-- [start:coordinate-origin]
domain = IndexDomain(inclusive_min=(-2,), exclusive_max=(3,))
assert domain.contains((-1,))
assert domain.narrow(-1).inclusive_min == (-1,)

values = np.array([10, 20, 30, 40, 50])
assert LazyArray(values).lazy[-1:].result().tolist() == [50]
# --8<-- [end:coordinate-origin]


# --8<-- [start:prepend-grid]
class PrependableGrid:
    """A regular grid whose coordinates may extend below zero."""

    def index_to_chunk(self, index: int) -> int:
        return index // 3

    def chunk_offset(self, chunk: int) -> int:
        return chunk * 3

    def chunk_size(self, chunk: int) -> int:
        return 3

    def indices_to_chunks(self, indices: NDArray[np.intp]) -> NDArray[np.intp]:
        return np.floor_divide(indices, 3).astype(np.intp)


projection, = plan_chunks(
    IndexTransform.identity(IndexDomain((-3,), (0,))),
    cast(tuple[DimensionGridLike], (PrependableGrid(),)),
)
assert projection.chunk_coords == (-1,)
assert projection.chunk_domain == IndexDomain((-3,), (0,))
assert projection.chunk_transform.domain == IndexDomain((0,), (3,))


assert projection.chunk_transform.domain == projection.cell_transform.domain
PREPEND_SHARED_CELL_COORDS = ((0,), (1,), (2,))
prepend_shared_cell_points = np.asarray(PREPEND_SHARED_CELL_COORDS, dtype=np.intp)
PREPEND_CHUNK_LOCAL_COORDS = tuple(
    tuple(point)
    for point in projection.chunk_transform.apply_many(prepend_shared_cell_points).tolist()
)
PREPEND_REQUEST_COORDS = tuple(
    tuple(point)
    for point in projection.cell_transform.apply_many(prepend_shared_cell_points).tolist()
)
assert PREPEND_CHUNK_LOCAL_COORDS == ((0,), (1,), (2,))
assert PREPEND_REQUEST_COORDS == ((-3,), (-2,), (-1,))
# --8<-- [end:prepend-grid]
