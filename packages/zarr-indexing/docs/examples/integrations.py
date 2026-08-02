"""Boundaries for chunked-source and viewport consumers."""

from typing import Any, cast

import numpy as np

from zarr_indexing import ArrayMap, ConstantMap, DimensionMap, IndexDomain, IndexTransform, LazyArray


# --8<-- [start:zarr-consumer]
class RecordingChunkSource:
    """A decoded-chunk source keyed by public chunk coordinates."""

    def __init__(self, chunks: dict[tuple[int, ...], np.ndarray[Any, Any]]) -> None:
        self.chunks = chunks
        self.reads: list[tuple[int, ...]] = []

    def read(self, chunk_coords: tuple[int, ...]) -> np.ndarray[Any, Any]:
        self.reads.append(chunk_coords)
        return self.chunks[chunk_coords]


def evaluate_point(transform: IndexTransform, point: tuple[int, ...]) -> tuple[int, ...]:
    """Evaluate the three public output-map forms at one input coordinate."""
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


zarr_image = np.arange(48).reshape(6, 8)
zarr_chunks = {
    (chunk_row, chunk_column): zarr_image[
        chunk_row * 3 : (chunk_row + 1) * 3,
        chunk_column * 4 : (chunk_column + 1) * 4,
    ]
    for chunk_row in range(2)
    for chunk_column in range(2)
}
zarr_source = RecordingChunkSource(zarr_chunks)
zarr_view = LazyArray(cast(Any, zarr_image)).with_parts((3, 4)).lazy[1:5, 2]
ZARR_RESULT = np.empty(zarr_view.shape, dtype=zarr_image.dtype)
shared_domains: list[tuple[IndexDomain, IndexDomain]] = []
chunk_local_coords: list[tuple[tuple[int, ...], ...]] = []
request_coords: list[tuple[tuple[int, ...], ...]] = []
read_values: list[tuple[int, ...]] = []

for part in zarr_view.parts():
    projection = part.projection
    assert projection.chunk_transform.domain == projection.cell_transform.domain
    shared_domains.append(
        (projection.chunk_transform.domain, projection.cell_transform.domain)
    )
    domain = projection.chunk_transform.domain
    cell_points = tuple(
        tuple(
            position + origin
            for position, origin in zip(index, domain.inclusive_min, strict=True)
        )
        for index in np.ndindex(*domain.shape)
    )
    local_points = tuple(evaluate_point(projection.chunk_transform, point) for point in cell_points)
    result_points = tuple(evaluate_point(projection.cell_transform, point) for point in cell_points)
    chunk = zarr_source.read(projection.chunk_coords)
    values = tuple(int(chunk[point]) for point in local_points)
    for result_point, value in zip(result_points, values, strict=True):
        ZARR_RESULT[result_point] = value
    chunk_local_coords.append(local_points)
    request_coords.append(result_points)
    read_values.append(values)

ZARR_SOURCE_KEYS = tuple(zarr_source.chunks)
ZARR_SOURCE_READS = tuple(zarr_source.reads)
ZARR_DISPATCHED_CHUNKS = ZARR_SOURCE_READS
ZARR_SHARED_DOMAINS = tuple(shared_domains)
ZARR_CHUNK_LOCAL_COORDS = tuple(chunk_local_coords)
ZARR_REQUEST_COORDS = tuple(request_coords)
ZARR_READ_VALUES = tuple(read_values)
assert ZARR_SOURCE_KEYS == ((0, 0), (0, 1), (1, 0), (1, 1))
assert ZARR_SOURCE_READS == ((0, 0), (1, 0))
assert ZARR_CHUNK_LOCAL_COORDS == (((1, 2), (2, 2)), ((0, 2), (1, 2)))
assert ZARR_REQUEST_COORDS == (((0,), (1,)), ((2,), (3,)))
assert ZARR_READ_VALUES == ((10, 18), (26, 34))
assert ZARR_RESULT.tolist() == [10, 18, 26, 34]
# --8<-- [end:zarr-consumer]


# --8<-- [start:viewport-consumer]
class RecordingArray:
    """An array-like source that records the basic reads it receives."""

    def __init__(self, data: np.ndarray[Any, Any], chunks: tuple[int, ...]) -> None:
        self._data = data
        self.chunks = chunks
        self.keys: list[tuple[slice, ...]] = []

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._data.dtype

    def __getitem__(self, key: tuple[slice, ...]) -> np.ndarray[Any, Any]:
        self.keys.append(key)
        return self._data[key]


viewport_source = RecordingArray(np.arange(48).reshape(6, 8), chunks=(3, 4))
viewport = LazyArray(viewport_source).lazy[1:5, 2]
VIEWPORT_READS_BEFORE_RESULT = tuple(viewport_source.keys)
assert VIEWPORT_READS_BEFORE_RESULT == ()
assert viewport.result().tolist() == [10, 18, 26, 34]

VIEWPORT_SOURCE_KEYS = tuple(viewport_source.keys)
VIEWPORT_SOURCE_CHUNKS = tuple(
    (key[0].start // 3, key[1].start // 4) for key in VIEWPORT_SOURCE_KEYS
)
assert VIEWPORT_SOURCE_KEYS == (
    (slice(1, 3, 1), slice(2, 3, 1)),
    (slice(3, 5, 1), slice(2, 3, 1)),
)
assert VIEWPORT_SOURCE_CHUNKS == ((0, 0), (1, 0))
# --8<-- [end:viewport-consumer]
