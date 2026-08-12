---
title: lazy_array
---

`LazyArray.lazy[...]` is metadata-only: every derived view keeps the same
reader and composes its transform without reading data. `result()` allocates
owned system memory, then calls that reader once for each projected part.
Rectangular parts write directly into their final slices; advanced placement
may first use an owned dense temporary. `LazyArray(source)` assumes only basic
indexing, while `LazyArray.from_numpy(array)` explicitly selects NumPy's
optimized reader.

The built-in readers lower through NumPy system memory and support sources
whose basic reads can be converted there. They do not implicitly transfer
device arrays; a device source needs an explicit custom reader that transfers
into the supplied system-memory output. Derived views and parts share their
reader and part views may be materialized concurrently, so stateful readers
must synchronize their own mutable state.

Every public `Partition.view.transform` directly maps that view's zero-origin
coordinates into its raw `Partition.view.array`, including for non-first
partitions. `Partition.projection.chunk_transform` intentionally stays local to
the selected chunk. During materialization the reader receives both frames in
one `ReadContext`: the public global transform in `context.transform` and the
same local plan in `context.projection`.

::: zarr_indexing.lazy_array
