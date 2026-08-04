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

The requests demonstrate lazy selections, paired chunk projections, overlapping
viewport requests that reuse resident chunks, eviction under capacity pressure,
a retained failure that does not retry implicitly, and an explicit retry after
the source is repaired. The integration guide contains the detailed request
table.

This is synchronous system-memory reference architecture, not a
production-ready cache, scheduler, renderer, or complete napari integration.
Its types are intentionally not exported by `zarr_indexing`.

## Running the example

```bash
cd packages/zarr-indexing
uv run --with-editable . examples/napari_chunk_cache/napari_chunk_cache.py
```
