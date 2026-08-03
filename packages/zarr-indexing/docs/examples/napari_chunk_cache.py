from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, cast

import numpy as np

from zarr_indexing import ArrayMap, ConstantMap, DimensionMap, IndexTransform, LazyArray

type ChunkCoords = tuple[int, ...]


# --8<-- [start:chunk-cache-types]
class ChunkState(StrEnum):
    NEW = "new"
    QUEUED = "queued"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"
    EVICTED = "evicted"


@dataclass(slots=True)
class ChunkRecord:
    state: ChunkState = ChunkState.NEW
    buffer: np.ndarray[Any, Any] | None = None
    error: Exception | None = None
    last_access: int = -1


@dataclass(frozen=True, slots=True)
class ChunkEvent:
    chunk_coords: ChunkCoords
    previous: ChunkState
    current: ChunkState
    reason: str


class ChunkLoadError(RuntimeError):
    pass


# --8<-- [end:chunk-cache-types]


# --8<-- [start:chunk-cache-source]
class RecordingChunkSource:
    def __init__(self, data: np.ndarray[Any, Any], chunks: tuple[int, ...]) -> None:
        self._data = data
        self.chunks = chunks
        self.reads: list[ChunkCoords] = []
        self.failures: set[ChunkCoords] = set()

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._data.dtype

    def __getitem__(self, key: Any) -> np.ndarray[Any, Any]:
        raise AssertionError("the cache must read complete chunks through read_chunk")

    def read_chunk(self, chunk_coords: ChunkCoords) -> np.ndarray[Any, Any]:
        self.reads.append(chunk_coords)
        if chunk_coords in self.failures:
            raise OSError(f"source read failed for chunk {chunk_coords}")
        key = tuple(
            slice(coord * size, min((coord + 1) * size, extent))
            for coord, size, extent in zip(chunk_coords, self.chunks, self.shape, strict=True)
        )
        return self._data[key].copy()


def evaluate_point(transform: IndexTransform, point: tuple[int, ...]) -> tuple[int, ...]:
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


# --8<-- [end:chunk-cache-source]


# --8<-- [start:chunk-cache-wrapper]
LEGAL_TRANSITIONS: dict[ChunkState, frozenset[ChunkState]] = {
    ChunkState.NEW: frozenset({ChunkState.QUEUED}),
    ChunkState.QUEUED: frozenset({ChunkState.LOADING}),
    ChunkState.LOADING: frozenset({ChunkState.READY, ChunkState.FAILED}),
    ChunkState.READY: frozenset({ChunkState.EVICTED}),
    ChunkState.FAILED: frozenset({ChunkState.QUEUED}),
    ChunkState.EVICTED: frozenset({ChunkState.QUEUED}),
}


class SystemMemoryChunkCache:
    def __init__(self, source: RecordingChunkSource, *, capacity: int) -> None:
        self.source = source
        self.capacity = capacity
        self._planner = LazyArray(cast(Any, source)).with_parts(source.chunks)
        self._records: dict[ChunkCoords, ChunkRecord] = {}
        self._queue: list[ChunkCoords] = []
        self._clock = 0
        self.events: list[ChunkEvent] = []
        self.projection_uses: tuple[tuple[str, str], ...] = ()

    @property
    def shape(self) -> tuple[int, ...]:
        return self.source.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.source.dtype

    def state(self, chunk_coords: ChunkCoords) -> ChunkState:
        return self._record(chunk_coords).state

    def resident(self) -> tuple[ChunkCoords, ...]:
        return tuple(
            sorted(
                coords for coords, record in self._records.items() if record.state is ChunkState.READY
            )
        )

    def _record(self, chunk_coords: ChunkCoords) -> ChunkRecord:
        return self._records.setdefault(chunk_coords, ChunkRecord())

    def _transition(self, chunk_coords: ChunkCoords, current: ChunkState, reason: str) -> None:
        record = self._record(chunk_coords)
        if current not in LEGAL_TRANSITIONS[record.state]:
            raise ValueError(f"illegal chunk transition {record.state} -> {current}")
        previous = record.state
        record.state = current
        self.events.append(ChunkEvent(chunk_coords, previous, current, reason))

    def retry(self, chunk_coords: ChunkCoords) -> None:
        record = self._record(chunk_coords)
        if record.state is not ChunkState.FAILED:
            raise ValueError(f"retry requires failed chunk {chunk_coords}, got {record.state}")
        record.error = None
        self._transition(chunk_coords, ChunkState.QUEUED, "explicit retry")
        self._queue.append(chunk_coords)

    def _touch(self, record: ChunkRecord) -> None:
        self._clock += 1
        record.last_access = self._clock

    def _queue_once(self, chunk_coords: ChunkCoords) -> None:
        record = self._record(chunk_coords)
        if record.state in {ChunkState.QUEUED, ChunkState.LOADING, ChunkState.READY}:
            return
        if record.state is ChunkState.FAILED:
            raise ValueError(f"failed chunk {chunk_coords} requires explicit retry")
        self._transition(chunk_coords, ChunkState.QUEUED, "requested")
        self._queue.append(chunk_coords)

    def _drain(self, required: frozenset[ChunkCoords]) -> None:
        pending = self._queue
        self._queue = []
        for index, chunk_coords in enumerate(pending):
            if chunk_coords not in required:
                self._queue.append(chunk_coords)
                continue
            record = self._record(chunk_coords)
            self._transition(chunk_coords, ChunkState.LOADING, "queue drained")
            try:
                record.buffer = self.source.read_chunk(chunk_coords)
            except OSError as error:
                record.buffer = None
                record.error = error
                self._transition(chunk_coords, ChunkState.FAILED, "source read failed")
                self._queue.extend(pending[index + 1 :])
                raise ChunkLoadError(f"could not load chunk {chunk_coords}") from error
            record.error = None
            self._transition(chunk_coords, ChunkState.READY, "source read completed")
            self._touch(record)

    def _evict(self, *, pinned: frozenset[ChunkCoords]) -> None:
        while len(self.resident()) > self.capacity:
            candidates = (
                (record.last_access, chunk_coords)
                for chunk_coords, record in self._records.items()
                if record.state is ChunkState.READY and chunk_coords not in pinned
            )
            _, chunk_coords = min(candidates)
            record = self._record(chunk_coords)
            record.buffer = None
            self._transition(chunk_coords, ChunkState.EVICTED, "LRU capacity")

    def __getitem__(self, key: Any) -> np.ndarray[Any, Any]:
        view = self._planner.lazy.oindex[key]
        projections = tuple(part.projection for part in view.parts())
        required = tuple(dict.fromkeys(projection.chunk_coords for projection in projections))

        for chunk_coords in required:
            record = self._record(chunk_coords)
            if record.state is ChunkState.FAILED:
                assert record.error is not None
                raise ChunkLoadError(f"chunk {chunk_coords} is failed; call retry first") from record.error

        for chunk_coords in required:
            record = self._record(chunk_coords)
            if record.state is ChunkState.READY:
                self._touch(record)
            else:
                self._queue_once(chunk_coords)
        self._drain(frozenset(required))

        result = np.empty(view.shape, dtype=self.dtype)
        uses: list[tuple[str, str]] = []
        for projection in projections:
            record = self._record(projection.chunk_coords)
            assert record.state is ChunkState.READY and record.buffer is not None
            domain = projection.chunk_transform.domain
            for index in np.ndindex(*domain.shape):
                point = tuple(
                    position + origin
                    for position, origin in zip(index, domain.inclusive_min, strict=True)
                )
                chunk_point = evaluate_point(projection.chunk_transform, point)
                request_point = evaluate_point(projection.cell_transform, point)
                result[request_point] = record.buffer[chunk_point]
            uses.append(("chunk_transform", "cell_transform"))
        self.projection_uses = tuple(uses)
        self._evict(pinned=frozenset())
        return result


# --8<-- [end:chunk-cache-wrapper]


# --8<-- [start:chunk-cache-worked-example]
image = np.arange(48).reshape(6, 8)
source = RecordingChunkSource(image, chunks=(3, 4))
cache = SystemMemoryChunkCache(source, capacity=2)

READS_BEFORE_SELECTION = tuple(source.reads)
INITIAL_RESULT = cache[1:5, 2]
INITIAL_READS = tuple(source.reads)

before_overlap = len(source.reads)
OVERLAP_RESULT = cache[3:5, 2]
OVERLAP_NEW_READS = tuple(source.reads[before_overlap:])

before_eviction = len(source.reads)
EVICTION_RESULT = cache[0:2, 5]
EVICTION_NEW_READS = tuple(source.reads[before_eviction:])
AFTER_EVICTION_RESIDENT = cache.resident()

before_reload = len(source.reads)
RELOAD_RESULT = cache[1:5, 2]
RELOAD_NEW_READS = tuple(source.reads[before_reload:])
AFTER_RELOAD_RESIDENT = cache.resident()

source.failures.add((1, 1))
failed_once = False
try:
    cache[3:5, 4:6]
except ChunkLoadError:
    failed_once = True
assert failed_once
FAILED_READ_COUNT = source.reads.count((1, 1))
failed_twice = False
try:
    cache[3:5, 4:6]
except ChunkLoadError:
    failed_twice = True
assert failed_twice
FAILED_REPEAT_READ_COUNT = source.reads.count((1, 1))
FAILURE_READ_COUNTS = (FAILED_READ_COUNT, FAILED_REPEAT_READ_COUNT)

source.failures.remove((1, 1))
cache.retry((1, 1))
before_retry = len(source.reads)
RETRY_RESULT = cache[3:5, 4:6]
RETRY_NEW_READS = tuple(source.reads[before_retry:])
RETRY_STATE = cache.state((1, 1)).value
WORKED_EVENTS = tuple(cache.events)
FAILED_TRANSITIONS = tuple(
    event.current.value for event in WORKED_EVENTS if event.chunk_coords == (1, 1)
)
FAILED_EVENT_ROWS = tuple(
    (event.previous.value, event.current.value, event.reason)
    for event in WORKED_EVENTS
    if event.chunk_coords == (1, 1)
)
# --8<-- [end:chunk-cache-worked-example]

assert READS_BEFORE_SELECTION == ()
assert INITIAL_RESULT.tolist() == [10, 18, 26, 34]
assert INITIAL_READS == ((0, 0), (1, 0))
assert OVERLAP_RESULT.tolist() == [26, 34]
assert OVERLAP_NEW_READS == ()
assert EVICTION_RESULT.tolist() == [5, 13]
assert EVICTION_NEW_READS == ((0, 1),)
assert AFTER_EVICTION_RESIDENT == ((0, 1), (1, 0))
assert RELOAD_RESULT.tolist() == [10, 18, 26, 34]
assert RELOAD_NEW_READS == ((0, 0),)
assert AFTER_RELOAD_RESIDENT == ((0, 0), (1, 0))
assert FAILURE_READ_COUNTS == (1, 1)
assert RETRY_RESULT.tolist() == [[28, 29], [36, 37]]
assert RETRY_NEW_READS == ((1, 1),)
assert RETRY_STATE == "ready"
assert FAILED_TRANSITIONS == (
    "queued",
    "loading",
    "failed",
    "queued",
    "loading",
    "ready",
)
assert FAILED_EVENT_ROWS == (
    ("new", "queued", "requested"),
    ("queued", "loading", "queue drained"),
    ("loading", "failed", "source read failed"),
    ("failed", "queued", "explicit retry"),
    ("queued", "loading", "queue drained"),
    ("loading", "ready", "source read completed"),
)
