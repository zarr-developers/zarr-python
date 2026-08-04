# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "zarr-indexing>=0.1",
#   "numpy==2.4.3",
# ]
# ///
#
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np

from zarr_indexing import (
    IndexDomain,
    LazyArray,
    ReadContext,
    dimension_grids_from_chunks,
    plan_chunks,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


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


def _domain_points(domain: IndexDomain) -> np.ndarray[Any, np.dtype[np.intp]]:
    """Enumerate a rectangular domain with a trailing coordinate axis."""
    if domain.ndim == 0:
        return np.empty((1, 0), dtype=np.intp)
    points = np.moveaxis(np.indices(domain.shape, dtype=np.intp), 0, -1).reshape(-1, domain.ndim)
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


class _OrthogonalIndexer:
    """Expose outer-product indexing without changing ``cache[key]`` semantics."""

    def __init__(self, getitem: Callable[[Any], np.ndarray[Any, Any]]) -> None:
        self._getitem = getitem

    def __getitem__(self, key: Any) -> np.ndarray[Any, Any]:
        return self._getitem(key)


class SystemMemoryChunkReader:
    def __init__(self, *, capacity: int) -> None:
        self.capacity = capacity
        self._records: dict[ChunkCoords, ChunkRecord] = {}
        self._queue: list[ChunkCoords] = []
        self._clock = 0
        self._requests = 0
        self.events: list[ChunkEvent] = []
        self.projection_uses: list[tuple[str, str]] = []

    def state(self, chunk_coords: ChunkCoords) -> ChunkState:
        return self._record(chunk_coords).state

    def resident(self) -> tuple[ChunkCoords, ...]:
        return tuple(
            sorted(
                coords
                for coords, record in self._records.items()
                if record.state is ChunkState.READY
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

    @contextmanager
    def request(self, required: tuple[ChunkCoords, ...]) -> Iterator[None]:
        """Prepare every part and defer eviction until one request completes."""
        self._prepare(required)
        self._requests += 1
        try:
            yield
        except Exception:
            self._requests -= 1
            raise
        else:
            self._requests -= 1
            if self._requests == 0:
                self._evict(pinned=frozenset())

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

    def _prepare(self, required: tuple[ChunkCoords, ...]) -> None:
        for chunk_coords in required:
            record = self._record(chunk_coords)
            if record.state is ChunkState.FAILED:
                assert record.error is not None
                raise ChunkLoadError(
                    f"chunk {chunk_coords} is failed; call retry first"
                ) from record.error

        for chunk_coords in required:
            record = self._record(chunk_coords)
            if record.state is ChunkState.READY:
                self._touch(record)
            else:
                self._queue_once(chunk_coords)

    def _ensure_ready(
        self,
        source: RecordingChunkSource,
        required: tuple[ChunkCoords, ...],
    ) -> None:
        if self._requests == 0:
            self._prepare(required)
        self._drain(source, frozenset(required))

    def _drain(self, source: RecordingChunkSource, required: frozenset[ChunkCoords]) -> None:
        pending = self._queue
        self._queue = []
        for index, chunk_coords in enumerate(pending):
            if chunk_coords not in required:
                self._queue.append(chunk_coords)
                continue
            record = self._record(chunk_coords)
            self._transition(chunk_coords, ChunkState.LOADING, "queue drained")
            try:
                record.buffer = source.read_chunk(chunk_coords)
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

    def read_into(
        self,
        source: RecordingChunkSource,
        context: ReadContext,
        out: np.ndarray[Any, Any],
        /,
    ) -> None:
        grids = dimension_grids_from_chunks(source.chunks, source.shape)
        projections = tuple(plan_chunks(context.transform, grids))
        required = tuple(dict.fromkeys(projection.chunk_coords for projection in projections))
        self._ensure_ready(source, required)
        for projection in projections:
            record = self._record(projection.chunk_coords)
            assert record.buffer is not None
            cell_points = _domain_points(projection.chunk_transform.domain)
            chunk_points = projection.chunk_transform.apply_many(cell_points)
            request_points = projection.cell_transform.apply_many(cell_points)
            _gather_and_scatter(out, record.buffer, chunk_points, request_points)
            self.projection_uses.append(("chunk_transform", "cell_transform"))
        if self._requests == 0:
            self._evict(pinned=frozenset())


class SystemMemoryChunkCache:
    def __init__(self, source: RecordingChunkSource, *, capacity: int) -> None:
        self.source = source
        self.reader = SystemMemoryChunkReader(capacity=capacity)
        self._lazy = LazyArray(source).with_reader(self.reader)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.source.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.source.dtype

    @property
    def oindex(self) -> _OrthogonalIndexer:
        return _OrthogonalIndexer(lambda key: self._read(key, orthogonal=True))

    @property
    def events(self) -> list[ChunkEvent]:
        return self.reader.events

    @property
    def projection_uses(self) -> tuple[tuple[str, str], ...]:
        return tuple(self.reader.projection_uses)

    def state(self, chunk_coords: ChunkCoords) -> ChunkState:
        return self.reader.state(chunk_coords)

    def resident(self) -> tuple[ChunkCoords, ...]:
        return self.reader.resident()

    def retry(self, chunk_coords: ChunkCoords) -> None:
        self.reader.retry(chunk_coords)

    def __getitem__(self, key: Any) -> np.ndarray[Any, Any]:
        return self._read(key, orthogonal=False)

    def _read(self, key: Any, *, orthogonal: bool) -> np.ndarray[Any, Any]:
        self.reader.projection_uses.clear()
        lazy = self._lazy.lazy
        view = lazy.oindex[key] if orthogonal else lazy[key]
        required = tuple(dict.fromkeys(part.base_coords for part in view.parts()))
        with self.reader.request(required):
            return np.asarray(view.result())


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
