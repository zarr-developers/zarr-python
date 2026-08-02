"""Literal coordinate domains and a grid that supports prepending."""

import numpy as np
from numpy.typing import NDArray
from typing import Any, cast

from zarr_indexing import DimensionGridLike, IndexDomain, IndexTransform, LazyArray, plan_chunks


# --8<-- [start:coordinate-origin]
domain = IndexDomain(inclusive_min=(-2,), exclusive_max=(3,))
assert domain.contains((-1,))
assert domain.narrow(-1).inclusive_min == (-1,)

values = np.array([10, 20, 30, 40, 50])
assert LazyArray(cast(Any, values)).lazy[-1].result() == 50
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
# --8<-- [end:prepend-grid]
