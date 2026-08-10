# Integration boundaries

This package supplies indexing plans. It does **not** supply scheduling,
caching, codecs, or async orchestration. A consumer decides when projections
run, how decoded chunks are obtained, and where completed values are retained.
An `IndexTransform` says which source values belong in a result; a `Reader`
lowers that complete transform for one backend. The reader does not choose
indexing semantics or result ownership.

## Zarr chunk dispatch

A Zarr-oriented reader can consume each public `ChunkProjection` and use its
`chunk_coords` to obtain one decoded chunk from its own storage and codec
layers. This tiny source keeps four in-memory chunks keyed by their global
chunk coordinates and records the exact reads. For each projection, the
consumer enumerates the shared synthetic input cell domain, evaluates
`chunk_transform` to read zero-origin chunk-local coordinates, and evaluates
`cell_transform` to place each value at its literal request coordinate.

```python
--8<-- "snippets/integrations.py:zarr-consumer"
```

The two reads are exactly `(0, 0)` and `(0, 1)`; untouched chunks `(1, 0)` and
`(1, 1)` are never read. The assembled request is `[4, 5, 6, 7]`. The
example intentionally begins with already decoded in-memory chunks: storage
keys, codecs, scheduling, caching, and asynchronous orchestration remain the
consumer's policy rather than responsibilities of the plan.

## One slab read or many part reads

A backend with its own native subset read — a Rust or C zarr implementation,
a database, an HTTP range endpoint — resolves a **dense box** (`is_box` with
every stride 1) best as a single read: hand it the whole selection and let it
dispatch to chunks, decode in parallel, and partial-decode shards on its own
side of the boundary. Splitting that read along this library's partitioning
only adds round-trips. Every **other** selection — a strided box, an `oindex`
or `vindex` gather — is where the partitioning earns its keep: each part's
cover stays inside one part, so a sparse selection can never force a read of
its whole bounding hull.

The composed view carries enough to make that call at materialization time,
and re-partitioning is a pure setter, so the policy is three lines:

```python
--8<-- "snippets/integrations.py:dense-box-repartition"
```

The corner gather reads four single cells instead of the 10-by-10 hull, and
the dense box becomes exactly one backend call. Both regimes go through
`result()`; only the partitioning in force differs.

### Sources that accept only unit-step slices

The default `basic_reader` pushes strided and descending selections down as
positive-step slices, which reads the minimum but assumes the source accepts
any step. Many backends do not: FFI bindings and range requests often
support nothing but `slice(start, stop, 1)`. Select
[`unit_step_reader`][zarr_indexing.reader.UnitStepReader] for such a source
and every key it receives is an ascending unit-step slice per axis, with
strides, reversals, and gathers applied to the in-memory block instead:

```python
view = LazyArray(source).with_reader(unit_step_reader)
```

A strided selection then over-reads its cover by the stride factor, which the
partitioning above bounds by one part.

## napari-like consumer

This is a **napari-like consumer**, not a napari integration. It models the
boundary a viewport could use without importing or claiming support for
napari. `RecordingArray` exposes a chunked, basic-indexing source; composing the
visible slice records no reads. Only `result()` materializes it, with the exact
source selectors `1:2, 0:2` and `1:2, 2:4`; neither selector crosses into an
untouched neighboring chunk.

```python
--8<-- "snippets/integrations.py:viewport-consumer"
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
[system-memory chunk cache example](../examples/system_memory_chunk_cache.md).

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
--8<-- "system_memory_chunk_cache/system_memory_chunk_cache.py:chunk-cache-types"
```

Its source represents already decoded chunks and records each read:

```python
--8<-- "system_memory_chunk_cache/system_memory_chunk_cache.py:chunk-cache-source"
```

`LazyArray` converts a cache selection into transforms and partitions, then
allocates and assembles the result. The facade constructs exactly one tuple
from `view.parts()`: it derives the chunk coordinates to pin from that tuple,
then passes the same owned parts to `view.result(parts=parts)`. Planning is
therefore performed once for the request rather than repeated during
materialization. Neither pinning nor the result call rebuilds the plan; both
reuse those prepared `Partition` objects.

`SystemMemoryChunkReader` receives one `ReadContext` for each materialized
part. Its global `context.transform` directly addresses the raw source, while
`context.projection.chunk_transform` addresses the already identified chunk
locally. The reader consumes that supplied projection directly; it never calls
the chunk planner. `LazyArray` retains responsibility for the projection's
result placement and final assembly. The reader owns only cache state and
source reads, while `SystemMemoryChunkCache` remains the thin NumPy-style facade
that prepares and pins the one plan:

Its indexing dialects remain explicit: `cache[key]` accepts basic indexing
(integers, slices, ellipsis, and new axes), while `cache.oindex[key]` combines
per-axis index arrays as an outer product. Array keys are not silently treated
as orthogonal by plain square brackets; callers choose that behavior through
the named accessor.

```python
--8<-- "system_memory_chunk_cache/system_memory_chunk_cache.py:chunk-cache-wrapper"
```

### Follow one viewport through the cache

```python
--8<-- "system_memory_chunk_cache/system_memory_chunk_cache.py:chunk-cache-worked-example"
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
all requested values have been placed. Because pinning and materialization use
the same prepared tuple, those lifecycle decisions cannot drift from the parts
that are actually read, and the cache never has to infer or reconstruct a
projection.

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
