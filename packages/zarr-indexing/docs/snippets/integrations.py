"""Boundaries for chunked-source and viewport consumers."""

from typing import Any

import numpy as np

from zarr_indexing import IndexDomain, LazyArray


# --8<-- [start:zarr-consumer]
class RecordingChunkSource:
    """A decoded-chunk source keyed by public chunk coordinates."""

    def __init__(self, chunks: dict[tuple[int, ...], np.ndarray[Any, Any]]) -> None:
        self.chunks = chunks
        self.reads: list[tuple[int, ...]] = []

    def read(self, chunk_coords: tuple[int, ...]) -> np.ndarray[Any, Any]:
        self.reads.append(chunk_coords)
        return self.chunks[chunk_coords]


def _domain_points(domain: IndexDomain) -> np.ndarray[Any, np.dtype[np.intp]]:
    """Enumerate a rectangular domain with a trailing coordinate axis."""
    if domain.ndim == 0:
        return np.empty((1, 0), dtype=np.intp)
    points = np.moveaxis(np.indices(domain.shape, dtype=np.intp), 0, -1).reshape(
        -1, domain.ndim
    )
    points += np.asarray(domain.inclusive_min, dtype=np.intp)
    return points


def _gather_and_scatter(
    destination: np.ndarray[Any, Any],
    source: np.ndarray[Any, Any],
    source_points: np.ndarray[Any, np.dtype[np.intp]],
    destination_points: np.ndarray[Any, np.dtype[np.intp]],
) -> np.ndarray[Any, Any]:
    """Gather and scatter a flattened point batch, including rank zero."""
    values = np.asarray(source[tuple(source_points.T)]).reshape(-1)
    if destination_points.shape[-1] == 0:
        destination[()] = values.reshape(destination.shape)[()]
    else:
        destination[tuple(destination_points.T)] = values
    return values


zarr_image = np.arange(12).reshape(3, 4)
zarr_chunks = {
    (chunk_row, chunk_column): zarr_image[
        chunk_row * 2 : (chunk_row + 1) * 2,
        chunk_column * 2 : (chunk_column + 1) * 2,
    ]
    for chunk_row in range(2)
    for chunk_column in range(2)
}
zarr_source = RecordingChunkSource(zarr_chunks)
zarr_view = LazyArray.from_numpy(zarr_image).with_parts((2, 2)).lazy[1, 0:4]
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
    cell_points = _domain_points(domain)
    local_points_array = projection.chunk_transform.apply_many(cell_points)
    result_points_array = projection.cell_transform.apply_many(cell_points)
    chunk = zarr_source.read(projection.chunk_coords)
    values_array = _gather_and_scatter(
        ZARR_RESULT, chunk, local_points_array, result_points_array
    )
    local_points = tuple(tuple(point) for point in local_points_array.tolist())
    result_points = tuple(tuple(point) for point in result_points_array.tolist())
    values = tuple(int(value) for value in values_array)
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
assert ZARR_SOURCE_READS == ((0, 0), (0, 1))
assert ZARR_CHUNK_LOCAL_COORDS == (((1, 0), (1, 1)), ((1, 0), (1, 1)))
assert ZARR_REQUEST_COORDS == (((0,), (1,)), ((2,), (3,)))
assert ZARR_READ_VALUES == ((4, 5), (6, 7))
assert ZARR_RESULT.tolist() == [4, 5, 6, 7]
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


viewport_source = RecordingArray(np.arange(12).reshape(3, 4), chunks=(2, 2))
viewport = LazyArray(viewport_source).lazy[1, 0:4]
VIEWPORT_READS_BEFORE_RESULT = tuple(viewport_source.keys)
assert VIEWPORT_READS_BEFORE_RESULT == ()
assert viewport.result().tolist() == [4, 5, 6, 7]

VIEWPORT_SOURCE_KEYS = tuple(viewport_source.keys)
VIEWPORT_SOURCE_CHUNKS = tuple(
    (key[0].start // 2, key[1].start // 2) for key in VIEWPORT_SOURCE_KEYS
)
assert VIEWPORT_SOURCE_KEYS == (
    (slice(1, 2, 1), slice(0, 2, 1)),
    (slice(1, 2, 1), slice(2, 4, 1)),
)
assert VIEWPORT_SOURCE_CHUNKS == ((0, 0), (0, 1))
# --8<-- [end:viewport-consumer]


# --8<-- [start:dense-box-repartition]
def materialize(view: LazyArray) -> Any:
    """Read a dense box as one slab; resolve everything else per part."""
    strides = view.strides()
    if view.is_box and strides is not None and all(s == 1 for s in strides):
        view = view.with_parts(view.base_shape)
    return view.result()


slab_source = RecordingArray(np.arange(100).reshape(10, 10), chunks=(4, 4))
slab = LazyArray(slab_source)

dense = slab.lazy[2:9, 1:8]  # a dense box: every stride 1
assert materialize(dense).shape == (7, 7)
assert len(slab_source.keys) == 1  # one slab read; the source dispatches

slab_source.keys.clear()
gather = slab.lazy.oindex[[0, 9], [0, 9]]  # a query: keep the chunk parts
assert materialize(gather).tolist() == [[0, 9], [90, 99]]
assert len(slab_source.keys) == 4  # four covers, each inside one chunk
assert all(
    (key[0].stop - key[0].start) * (key[1].stop - key[1].start) == 1
    for key in slab_source.keys
)
# --8<-- [end:dense-box-repartition]
