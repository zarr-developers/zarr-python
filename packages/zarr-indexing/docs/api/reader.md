---
title: Readers
---

# Readers

An `IndexTransform` defines which source value belongs at every result
position. A `Reader` defines how a particular backend obtains those values.
Readers do not define indexing semantics, partitioning, scheduling, or result
ownership.

`Reader.read_into(source, transform, out)` receives a transform from
zero-origin output-buffer coordinates to global source coordinates, with
`transform.domain.shape == out.shape`. An implementation must fill every cell
of `out` in place, preserve the transform's exact values, order, and dtype, and
return `None`. It must neither replace nor retain `out`, which may be a strided
writable view. Backend exceptions propagate unchanged. Derived part views share
their reader and may be resolved concurrently, so a stateful reader owns its
own synchronization.

Reader wrappers compose by intercepting this one operation and forwarding the
same source, transform, and output buffer to an inner reader:

```python
class RecordingReader:
    def __init__(self, inner):
        self.inner = inner
        self.calls = []

    def read_into(self, source, transform, out, /):
        self.calls.append((source, transform, out))
        self.inner.read_into(source, transform, out)


inner = RecordingReader(numpy_reader)
outer = RecordingReader(inner)
view = LazyArray.from_numpy(array).with_reader(outer)
values = view.result()
```

Both wrappers observe the same three objects, in outer-to-inner order. This
delegation pattern supports policies such as logging and caching without
library-defined wrapper primitives.

::: zarr_indexing.reader
