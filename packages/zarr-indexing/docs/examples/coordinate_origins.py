"""Literal coordinate domains and a grid that supports prepending."""

import numpy as np
from numpy.typing import NDArray
from typing import Any, cast

from zarr_indexing import (
    ArrayMap,
    ConstantMap,
    DimensionGridLike,
    DimensionMap,
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


def evaluate_point(transform: IndexTransform, point: tuple[int, ...]) -> tuple[int, ...]:
    """Evaluate the public output-map forms at one input coordinate."""
    result: list[int] = []
    for output_map in transform.output:
        if isinstance(output_map, ConstantMap):
            result.append(output_map.offset)
        elif isinstance(output_map, DimensionMap):
            result.append(output_map.offset + output_map.stride * point[output_map.input_dimension])
        else:
            assert isinstance(output_map, ArrayMap)
            index = tuple(
                0
                if output_map.index_array.shape[axis] == 1
                else point[axis] - transform.domain.inclusive_min[axis]
                for axis in range(output_map.index_array.ndim)
            )
            result.append(output_map.offset + output_map.stride * int(output_map.index_array[index]))
    return tuple(result)


assert projection.chunk_transform.domain == projection.cell_transform.domain
PREPEND_SHARED_CELL_COORDS = ((0,), (1,), (2,))
PREPEND_CHUNK_LOCAL_COORDS = tuple(
    evaluate_point(projection.chunk_transform, point) for point in PREPEND_SHARED_CELL_COORDS
)
PREPEND_REQUEST_COORDS = tuple(
    evaluate_point(projection.cell_transform, point) for point in PREPEND_SHARED_CELL_COORDS
)
assert PREPEND_CHUNK_LOCAL_COORDS == ((0,), (1,), (2,))
assert PREPEND_REQUEST_COORDS == ((-3,), (-2,), (-1,))
# --8<-- [end:prepend-grid]
