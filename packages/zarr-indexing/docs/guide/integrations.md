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

The two reads are exactly `(0, 0)` and `(1, 0)`; untouched chunks `(0, 1)` and
`(1, 1)` are never read. The assembled request is `[10, 18, 26, 34]`. The
example intentionally begins with already decoded in-memory chunks: storage
keys, codecs, scheduling, caching, and asynchronous orchestration remain the
consumer's policy rather than responsibilities of the plan.

## napari-like consumer

This is a **napari-like consumer**, not a napari integration. It models the
boundary a viewport could use without importing or claiming support for
napari. `RecordingArray` exposes a chunked, basic-indexing source; composing the
visible slice records no reads. Only `result()` materializes it, with the exact
source selectors `1:3, 2:3` and `3:5, 2:3`; neither selector crosses into an
untouched neighboring chunk.

```python
--8<-- "examples/integrations.py:viewport-consumer"
```

The viewport owns its interaction loop and any cancellation, caching, or
background execution. `LazyArray` contributes the composable selection and
the partition plan, then resolves only when the consumer asks for the result.

---

<nav aria-label="Reference page navigation">
  <strong>Previous:</strong> <a href="../patterns/">Indexing pattern reference</a>
  ·
  <strong>API:</strong> <a href="../../api/">API reference</a>
</nav>
