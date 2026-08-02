"""Use public chunk projections without choosing a storage backend."""

import numpy as np
from numpy.typing import NDArray
from typing import cast

from zarr_indexing import DimensionGridLike, IndexTransform, plan_chunks


# --8<-- [start:chunk-projection]
class RegularGrid:
    """A small parameterized implementation of the public grid protocol."""

    def __init__(self, size: int) -> None:
        self.size = size

    def index_to_chunk(self, index: int) -> int:
        return index // self.size

    def chunk_offset(self, chunk: int) -> int:
        return chunk * self.size

    def chunk_size(self, chunk: int) -> int:
        return self.size

    def indices_to_chunks(self, indices: NDArray[np.intp]) -> NDArray[np.intp]:
        return np.floor_divide(indices, self.size).astype(np.intp)


transform = IndexTransform.from_shape((6, 8))[1:5, 2]
PROJECTIONS = tuple(
    projection
    for size in (3, 4)
    for projection in plan_chunks(
        transform,
        cast(tuple[DimensionGridLike, DimensionGridLike], (RegularGrid(size), RegularGrid(size))),
    )
)

for size in (3, 4):
    grids = cast(
        tuple[DimensionGridLike, DimensionGridLike], (RegularGrid(size), RegularGrid(size))
    )
    projections = tuple(plan_chunks(transform, grids))
    assert tuple(projection.chunk_coords for projection in projections) == ((0, 0), (1, 0))

PAIRED_DOMAINS = tuple(
    (projection.chunk_transform.domain, projection.cell_transform.domain)
    for projection in PROJECTIONS
)
assert all(chunk_domain == cell_domain for chunk_domain, cell_domain in PAIRED_DOMAINS)
# --8<-- [end:chunk-projection]
