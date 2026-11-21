# System-memory chunk cache

This executable reference architecture demonstrates a small, synchronous
system-memory chunk cache for a NumPy-like image consumer. It is inspired by
Neuroglancer's explicit chunk lifecycle:

```text
NEW -> QUEUED -> LOADING -> READY
                      |
                      v
                    FAILED

READY -> EVICTED
FAILED -> QUEUED  (explicit retry)
```

`RecordingChunkSource` owns decoded source-chunk reads and records them for the
example. `LazyArray` converts NumPy-style indexing into transforms and
partitions, then assembles the final result. `SystemMemoryChunkReader`
intercepts each materialized part and owns the lifecycle records, queue
draining, resident ready buffers, LRU eviction, retained load failures, and
explicit retry. The reader owns cache state and source reads, but not result
shape or assembly.

Each request calls `view.parts()` once and keeps the resulting tuple. The cache
queues the tuple's chunk coordinates and defers eviction, then materializes with
`view.result(parts=parts)`, so scheduling and assembly reuse one plan. Every
reader call consumes the exact projection attached to its `ReadContext`; the
reader does not invoke the chunk planner again.

The requests demonstrate lazy selections, paired chunk projections, overlapping
viewport requests that reuse resident chunks, eviction under capacity pressure,
a retained failure that does not retry implicitly, and an explicit retry after
the source is repaired. The integration guide contains the detailed request
table.

Capacity counts resident chunks, not bytes, and is enforced after a successful
request. A request can temporarily exceed it; a failed request skips that eviction
step. Records, event history, and output/coordinate buffers are outside this count.
The example assumes an unchanged source and one source/grid per reader; it has no
cache invalidation for source mutation, and its mutable state is not thread-safe.

This is synchronous system-memory reference architecture, not a
production-ready cache, scheduler, renderer, or complete napari integration.
Its types are intentionally not exported by `zarr_indexing`.

## Running the example

```bash
cd packages/zarr-indexing
uv run --with-editable . examples/system_memory_chunk_cache/system_memory_chunk_cache.py
```
