"""Boundaries for chunked-source and viewport consumers."""

from typing import Any

import numpy as np

from zarr_indexing import ChunkProjection, LazyArray


# --8<-- [start:zarr-consumer]
class ChunkDispatcher:
    """A zarr-oriented consumer can dispatch public projection coordinates."""

    def __init__(self) -> None:
        self.chunk_coords: list[tuple[int, ...]] = []

    def dispatch(self, projection: ChunkProjection) -> None:
        self.chunk_coords.append(projection.chunk_coords)
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
assert viewport_source.keys == []
assert viewport.result().tolist() == [10, 18, 26, 34]

VIEWPORT_SOURCE_KEYS = tuple(viewport_source.keys)
VIEWPORT_SOURCE_CHUNKS = tuple(
    (key[0].start // 3, key[1].start // 4) for key in VIEWPORT_SOURCE_KEYS
)
assert VIEWPORT_SOURCE_CHUNKS == ((0, 0), (1, 0))
# --8<-- [end:viewport-consumer]
