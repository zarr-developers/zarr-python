# Integration boundaries

This package supplies indexing plans. It does **not** supply scheduling,
caching, codecs, or async orchestration. A consumer decides when projections
run, how decoded chunks are obtained, and where completed values are retained.

## Zarr chunk dispatch

A Zarr-oriented reader can consume each public `ChunkProjection` and use its
`chunk_coords` to dispatch work to its own storage and codec layers. This
example stops at that ownership boundary: it records the two planned chunk
coordinates, `(0, 0)` and `(1, 0)`, without fetching, decoding, or scheduling a
chunk.

```python
--8<-- "examples/integrations.py:zarr-consumer"
```

The projection also carries the paired chunk-local and request-side transforms
described in [One cell domain, two projections](05-projections.md). A real Zarr
consumer can lower those transforms after its own chunk dispatch; the plan does
not prescribe that execution policy.

## napari-like consumer

This is a **napari-like consumer**, not a napari integration. It models the
boundary a viewport could use without importing or claiming support for
napari. `RecordingArray` exposes a chunked, basic-indexing source; composing the
visible slice records no reads. Only `result()` materializes it, and that read
touches exactly chunks `(0, 0)` and `(1, 0)`.

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
