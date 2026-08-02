"""Use public chunk projections without choosing a storage backend."""

import numpy as np
from numpy.typing import NDArray
from typing import Any, cast

from zarr_indexing import DimensionGridLike, IndexTransform, LazyArray, plan_chunks


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
grids = cast(
    tuple[DimensionGridLike, DimensionGridLike],
    (RegularGrid(3), RegularGrid(4)),
)
plan = plan_chunks(transform, grids)
PROJECTIONS = tuple(plan)
assert tuple(projection.chunk_coords for projection in plan) == ((0, 0), (1, 0))

PAIRED_DOMAINS = tuple(
    (projection.chunk_transform.domain, projection.cell_transform.domain)
    for projection in PROJECTIONS
)
assert all(chunk_domain == cell_domain for chunk_domain, cell_domain in PAIRED_DOMAINS)
# --8<-- [end:chunk-projection]


# --8<-- [start:advanced-projection]
image = np.arange(48).reshape(6, 8)
advanced = LazyArray(cast(Any, image)).with_parts((3, 4)).lazy.oindex[[4, 1, 1], 2:6]

ADVANCED_EXPECTED = image[[4, 1, 1]][:, 2:6]
ADVANCED_RESULT = np.empty_like(ADVANCED_EXPECTED)
for part in advanced.parts():
    ADVANCED_RESULT[part.out_selection] = part.view.result()

np.testing.assert_array_equal(ADVANCED_RESULT, ADVANCED_EXPECTED)
# --8<-- [end:advanced-projection]
