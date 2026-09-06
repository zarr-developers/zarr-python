"""Read a plan's factored form — its per-axis tables — instead of its projections."""

from typing import Any

import numpy as np

from zarr_indexing import IndexDomain, IndexedSet, IndexTransform, StridedSet, plan_chunks
from zarr_indexing.output_map import DimensionMap
from zarr_indexing.grid import dimension_grids_from_chunks

# --8<-- [start:strided-tables]
grids = dimension_grids_from_chunks((2, 2), shape=(3, 4))
plan = plan_chunks(IndexTransform.from_shape((3, 4))[1, :], grids)
partition = plan.partition()

rows, columns = partition.sets
assert isinstance(rows, StridedSet) and rows.input_dimension is None  # the fixed row
assert isinstance(columns, StridedSet) and columns.input_dimension == 0
assert columns.chunk.tolist() == [0, 1]
assert columns.local_start.tolist() == [0, 0]
assert columns.extent.tolist() == [2, 2]
assert columns.origin.tolist() == [0, 2]  # request columns 0:2 and 2:4
assert columns.full.tolist() == [True, True]

# One row of each table is one projection; the plan iterates exactly those rows.
assert partition.row_shape == (1, 2)
assert partition.chunk_coords().tolist() == [[0, 0], [0, 1]]
rows = list(partition)
assert rows == list(plan)
assert rows[1].chunk_transform.apply((0,)) == (1, 0)  # cell 0: chunk-local (1, 0) ...
assert rows[1].cell_transform.apply((0,)) == (2,)  # ... which is request column 2
# --8<-- [end:strided-tables]


# --8<-- [start:indexed-and-joint]
grids = dimension_grids_from_chunks((3, 4), shape=(6, 8))

gather = IndexTransform.from_shape((6, 8)).oindex[np.array([4, 1, 1]), 2:6]
rows, columns = plan_chunks(gather, grids).partition().sets
assert isinstance(rows, IndexedSet)
assert rows.chunk.tolist() == [0, 1]  # rows 1 and 1 land in chunk 0, row 4 in chunk 1
assert rows.pointer.tolist() == [0, 2, 3]  # chunk 0 owns entries 0:2, chunk 1 owns 2:3
assert rows.index.tolist() == [1, 1, 4]  # grouped by chunk, request order within a chunk
assert rows.positions.tolist() == [1, 2, 0]  # the request positions those entries fill
assert rows.local.tolist() == [1, 1, 1]  # chunk-local: 1 - 0, 1 - 0, 4 - 3

points = IndexTransform.from_shape((6, 8)).vindex[np.array([0, 5, 5]), np.array([7, 0, 1])]
joint = plan_chunks(points, grids).partition().joint_sets[0]
assert joint is not None
assert joint.chunk.tolist() == [[0, 1], [1, 0]]  # touched chunks, lexicographic
assert joint.pointer.tolist() == [0, 1, 3]  # point 0 alone; points 1 and 2 share a chunk
assert joint.positions.tolist() == [0, 1, 2]
assert joint.local.tolist() == [[0, 3], [2, 0], [2, 1]]
# --8<-- [end:indexed-and-joint]


# --8<-- [start:table-consumer]
def read_box_through_tables(
    source: np.ndarray[Any, Any], chunks: tuple[int, ...], transform: IndexTransform
) -> tuple[np.ndarray[Any, Any], list[tuple[int, ...]]]:
    """Assemble a box selection from the per-axis tables, one slice per chunk.

    No projection is materialized: every column read here is a NumPy array of
    the table, and the only per-chunk Python work is building the two selector
    tuples the copy needs.
    """
    partition = plan_chunks(
        transform, dimension_grids_from_chunks(chunks, shape=source.shape)
    ).partition()
    domain = transform.domain
    out = np.empty(domain.shape, dtype=source.dtype)
    tables = [table for table in partition.sets if isinstance(table, StridedSet)]
    assert len(tables) == len(partition.sets)  # a box: strided tables only
    # A chunk slab comes out in storage-axis order; the result is in request-axis
    # order, and request axes no map reads (`None` in the selection) are length 1.
    read_axes = [table.input_dimension for table in tables if table.input_dimension is not None]
    to_request_order = tuple(np.argsort(read_axes))
    unread_axes = [axis for axis in range(domain.ndim) if axis not in read_axes]
    reads: list[tuple[int, ...]] = []
    for table_rows in np.ndindex(*partition.row_shape):
        chunk_key: list[int] = []
        chunk_sel: list[int | slice] = []
        out_sel: list[slice] = [slice(None)] * domain.ndim
        for table, row in zip(tables, table_rows, strict=True):
            chunk_key.append(int(table.chunk[row]))
            start = int(table.local_start[row])
            count = int(table.extent[row])
            if table.input_dimension is None:
                chunk_sel.append(start)
                continue
            stop: int | None = start + table.stride * count
            if table.stride < 0 and stop < 0:
                stop = None  # a negative stop would count from the end
            chunk_sel.append(slice(start, stop, table.stride))
            first = int(table.origin[row])
            out_sel[table.input_dimension] = slice(first, first + count)
        chunk = source[tuple(slice(k * c, (k + 1) * c) for k, c in zip(chunk_key, chunks, strict=True))]
        values = np.transpose(chunk[tuple(chunk_sel)], to_request_order)
        for axis in unread_axes:
            values = np.expand_dims(values, axis)
        out[tuple(out_sel)] = values
        reads.append(tuple(chunk_key))
    return out, reads


image = np.arange(63).reshape(7, 9)
box = IndexTransform.from_shape((7, 9))[1:6:2, 5:]
values, reads = read_box_through_tables(image, (3, 4), box)
np.testing.assert_array_equal(values, image[1:6:2, 5:])
assert reads == [(0, 1), (0, 2), (1, 1), (1, 2)]

# A reversed axis, an inserted axis, and a transposed transform read the same way.
reversed_box = IndexTransform.from_shape((7, 9))[6:0:-1, None, ::3]
np.testing.assert_array_equal(read_box_through_tables(image, (3, 4), reversed_box)[0], image[6:0:-1, None, ::3])
transposed = IndexTransform(
    domain=IndexDomain.from_shape((9, 7)),
    output=(DimensionMap(input_dimension=1), DimensionMap(input_dimension=0)),
)
np.testing.assert_array_equal(read_box_through_tables(image, (3, 4), transposed)[0], image.T)
# --8<-- [end:table-consumer]


# --8<-- [start:components]
i = np.arange(1000)
transform = IndexTransform.from_shape((1000, 1000, 1000)).vindex[
    i[:, None], i[:, None], i[None, :]
]
partition = plan_chunks(
    transform, dimension_grids_from_chunks((1000, 1000, 1000), shape=(1000, 1000, 1000))
).partition()
assert [group.output_dimensions for group in partition.joint_sets] == [(0, 1), (2,)]
assert sum(group.index.size for group in partition.joint_sets) == 3000
assert partition.row_shape == (1, 1)
(projection,) = partition
assert projection.chunk_transform.apply((5, 9)) == (5, 5, 9)
assert projection.cell_transform.apply((5, 9)) == (5, 9)
# --8<-- [end:components]
