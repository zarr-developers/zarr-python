# Integration boundaries

This package supplies indexing plans. It does **not** supply scheduling,
caching, codecs, or async orchestration. A consumer decides when projections
run, how decoded chunks are obtained, and where completed values are retained.

## Zarr chunk dispatch

A Zarr-oriented reader can consume each public `ChunkProjection` and use its
`chunk_coords` to obtain one decoded chunk from its own storage and codec
layers. This tiny source keeps four in-memory chunks keyed by their global
chunk coordinates and records the exact reads. For each projection, the
consumer enumerates the shared synthetic input cell domain, evaluates
`chunk_transform` to read zero-origin chunk-local coordinates, and evaluates
`cell_transform` to place each value at its literal request coordinate.

```python
--8<-- "examples/integrations.py:zarr-consumer"
```

The two reads are exactly `(0, 0)` and `(0, 1)`; untouched chunks `(1, 0)` and
`(1, 1)` are never read. The assembled request is `[4, 5, 6, 7]`. The
example intentionally begins with already decoded in-memory chunks: storage
keys, codecs, scheduling, caching, and asynchronous orchestration remain the
consumer's policy rather than responsibilities of the plan.

## napari-like consumer

This is a **napari-like consumer**, not a napari integration. It models the
boundary a viewport could use without importing or claiming support for
napari. `RecordingArray` exposes a chunked, basic-indexing source; composing the
visible slice records no reads. Only `result()` materializes it, with the exact
source selectors `1:2, 0:2` and `1:2, 2:4`; neither selector crosses into an
untouched neighboring chunk.

```python
--8<-- "examples/integrations.py:viewport-consumer"
```

The viewport owns its interaction loop and any cancellation, caching, or
background execution. `LazyArray` contributes the composable selection and
the partition plan, then resolves only when the consumer asks for the result.

### A system-memory chunk cache

Napari accepts NumPy-like array objects and can defer materialization until an
image region is displayed. The indexing plan still deliberately owns no cache
or scheduler. A viewport adapter can place that policy around the plan, as the
executable reference below demonstrates.

This remains a **napari-like consumer, not a napari integration**. It models
only decoded chunks resident in system memory, synchronously.

For setup instructions and the complete executable, see the
[system-memory chunk cache example](../examples/napari_chunk_cache.md).

```text
                         read succeeds
NEW -> QUEUED -> LOADING -------------> READY -> EVICTED
        ^            |
        |            | read fails
        |            v
        +--------- FAILED
             retry

EVICTED -> QUEUED
           reload
```

The example keeps the lifecycle records and transitions explicit:

```python
--8<-- "napari_chunk_cache/napari_chunk_cache.py:chunk-cache-types"
```

Its source represents already decoded chunks and records each read:

```python
--8<-- "napari_chunk_cache/napari_chunk_cache.py:chunk-cache-source"
```

`LazyArray` converts a cache selection into transforms and partitions, then
allocates and assembles the result. `SystemMemoryChunkReader` intercepts each
materialized part, plans its chunks, reads resident buffers, and applies the
paired transforms. The reader owns cache state and source reads; it does not
own result shape or assembly. `SystemMemoryChunkCache` is only the thin
NumPy-style facade that configures that reader:

Its indexing dialects remain explicit: `cache[key]` accepts basic indexing
(integers, slices, ellipsis, and new axes), while `cache.oindex[key]` combines
per-axis index arrays as an outer product. Array keys are not silently treated
as orthogonal by plain square brackets; callers choose that behavior through
the named accessor.

```python
--8<-- "napari_chunk_cache/napari_chunk_cache.py:chunk-cache-wrapper"
```

### Follow one viewport through the cache

```python
--8<-- "napari_chunk_cache/napari_chunk_cache.py:chunk-cache-worked-example"
```

The worked example uses a 6-by-8 image, 3-by-4 chunks, and capacity for two
decoded chunks. Every read delta follows directly from the viewport request:

| Step | Viewport | New reads | Resident afterward | Why |
| --- | --- | --- | --- | --- |
| 1 | `image[1:5, 2]` | `(0, 0)`, `(1, 0)` | `(0, 0)`, `(1, 0)` | Both projected chunks are loaded and assembled as `[10, 18, 26, 34]`. |
| 2 | `image[3:5, 2]` | None | `(0, 0)`, `(1, 0)` | The ready buffer for `(1, 0)` is reused and becomes most recently used. |
| 3 | `image[0:2, 5]` | `(0, 1)` | `(0, 1)`, `(1, 0)` | Placement returns `[5, 13]`, then LRU pressure evicts `(0, 0)`. |
| 4 | `image[1:5, 2]` | `(0, 0)` | `(0, 0)`, `(1, 0)` | The evicted chunk is reloaded while the required ready chunk is retained. |
| 5 | `image[3:5, 4:6]` | `(1, 1)` fails; no repeated read; `(1, 1)` succeeds after retry | `(0, 0)`, `(1, 1)` | Failure is retained until explicit retry; the repaired source then returns `[[28, 29], [36, 37]]`. |

Chunks required by an active request are pinned through assembly, so a request
may temporarily span more chunks than the steady-state capacity. Capacity is
counted in decoded chunks—not records or bytes—and eviction occurs only after
all requested values have been placed.

The event log makes the failure boundary equally explicit:

| Chunk | Transition | Reason |
| --- | --- | --- |
| `(1, 1)` | `NEW -> QUEUED` | requested |
| `(1, 1)` | `QUEUED -> LOADING` | queue drained |
| `(1, 1)` | `LOADING -> FAILED` | source read failed |
| `(1, 1)` | `FAILED -> QUEUED` | explicit retry |
| `(1, 1)` | `QUEUED -> LOADING` | queue drained |
| `(1, 1)` | `LOADING -> READY` | source read completed |

A repeated request while the record is `FAILED` creates no event and performs
no source read. The retained failure forces the caller to choose when retry is
appropriate. A real viewport adapter could drain the queue in workers and
invalidate its canvas when chunks become ready without changing the selection
or projection semantics shown here.

[Napari's image-layer documentation](https://napari.org/dev/howtos/layers/image.html)
describes its NumPy-like array boundary. Neuroglancer's
[`ChunkState`](https://github.com/google/neuroglancer/blob/master/src/chunk_manager/base.ts)
is conceptual prior art for making residency explicit. This example is a
smaller, independently authored, synchronous teaching model; it does not copy
that implementation or reproduce its full worker/GPU lifecycle.

---

<nav aria-label="Reference page navigation">
  <strong>Previous:</strong> <a href="../patterns/">Indexing pattern reference</a>
  ·
  <strong>API:</strong> <a href="../../api/">API reference</a>
</nav>
