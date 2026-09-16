"""Follow selections into public chunk projections and assemble their values."""

from typing import Any

import numpy as np

from zarr_indexing import IndexTransform, plan_chunks
from zarr_indexing.grid import dimension_grids_from_chunks


# --8<-- [start:basic]
def assemble(source: np.ndarray[Any, Any], transform: IndexTransform) -> np.ndarray[Any, Any]:
    grids = dimension_grids_from_chunks((4,) * source.ndim, source.shape)
    result = np.empty(transform.domain.shape, dtype=source.dtype)
    for projection in plan_chunks(transform, grids):
        bounds = tuple(
            slice(grid.chunk_offset(c), grid.chunk_offset(c) + grid.data_size(c))
            for grid, c in zip(grids, projection.chunk_coords, strict=True)
        )
        chunk = source[bounds]  # A storage/codec layer supplies this in production.
        domain = projection.chunk_transform.domain
        for position in np.ndindex(domain.shape):
            point = tuple(a + b for a, b in zip(domain.origin, position, strict=True))
            local = projection.chunk_transform.apply(point)
            request = projection.cell_transform.apply(point)
            destination = tuple(
                x - origin for x, origin in zip(request, transform.domain.origin, strict=True)
            )
            result[destination] = chunk[local]
    return result


source = np.arange(80).reshape(8, 10)
selection = (slice(1, 7, 2), slice(2, 6))
transform = IndexTransform.from_shape(source.shape)[selection]
grids = dimension_grids_from_chunks((4, 4), source.shape)
assert [p.chunk_coords for p in plan_chunks(transform, grids)] == [(0, 0), (0, 1), (1, 0), (1, 1)]
result = assemble(source, transform)
np.testing.assert_array_equal(result, source[selection])
assert result.tolist() == [[12, 13, 14, 15], [32, 33, 34, 35], [52, 53, 54, 55]]
# --8<-- [end:basic]

# --8<-- [start:gather]
values = np.arange(8)
coordinates = np.array([6, 1, 6, 4])
gather = IndexTransform.from_shape(values.shape).vindex[coordinates]
gathered = assemble(values, gather)
np.testing.assert_array_equal(gathered, values[coordinates])
pairs: dict[tuple[int, ...], list[tuple[int, int]]] = {}
for projection in plan_chunks(gather, dimension_grids_from_chunks((4,), values.shape)):
    domain = projection.chunk_transform.domain
    pairs[projection.chunk_coords] = [
        (projection.chunk_transform.apply((i,))[0], projection.cell_transform.apply((i,))[0])
        for i in range(domain.origin[0], domain.exclusive_max[0])
    ]
assert pairs == {(0,): [(1, 1)], (1,): [(2, 0), (2, 2), (0, 3)]}
# --8<-- [end:gather]
