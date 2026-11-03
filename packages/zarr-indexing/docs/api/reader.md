---
title: Readers
---

# Readers

An `IndexTransform` defines which source value belongs at every result
position. A `Reader` defines how a particular backend obtains those values.
Readers do not define indexing semantics, partitioning, scheduling, or result
ownership.

`Reader.read_into(source, context, out)` receives a `ReadContext` whose
`transform` maps zero-origin output-buffer coordinates to global coordinates in
`source`, with `context.transform.domain.shape == out.shape`. A view's
transform keeps its literal domain; `ReadContext` re-bases it to origin zero on
construction, so readers never see a view's coordinates. Its optional
`projection` describes one planned read. `LazyArray.result()` always supplies
it, including for partition views and unpartitioned reads. Direct callers of
the reader protocol may omit it when their reader supports that. The projection's
`chunk_transform` remains chunk-local, its `cell_transform` places cells in the
zero-origin result buffer of the view that planned the read, which is what result
placement, and its `chunk_domain` describes the grid cell. The global read
transform and the projection's chunk transform deliberately use different
coordinate frames.

An implementation must fill every cell of `out` in place, preserve the global
transform's exact values, order, and dtype, and return `None`. It must neither
replace nor retain `out`, which may be a strided writable view. Backend
exceptions propagate unchanged. Derived part views share their reader and may
be resolved concurrently, so a stateful reader owns its own synchronization.

Reader wrappers compose by intercepting this one operation and forwarding the
same source, context, and output buffer to an inner reader:

```python
class RecordingReader:
    def __init__(self, inner):
        self.inner = inner
        self.calls = []

    def read_into(self, source, context, out, /):
        self.calls.append((source, context, out.shape, out.dtype))
        self.inner.read_into(source, context, out)


inner = RecordingReader(numpy_reader)
outer = RecordingReader(inner)
view = LazyArray.from_numpy(array).with_reader(outer)
values = view.result()
```

Both wrappers observe the same arguments, in outer-to-inner order, and log
output metadata without retaining the output buffer. This
delegation pattern supports policies such as logging and caching without
library-defined wrapper primitives.

::: zarr_indexing.reader
