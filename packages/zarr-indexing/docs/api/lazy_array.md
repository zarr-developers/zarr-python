---
title: lazy_array
---

`LazyArray[...]` is metadata-only: every derived view keeps the same
reader and composes its transform without reading data. `result()` allocates
owned system memory, then calls that reader once for each projected part.
Rectangular parts write directly into their final slices; advanced placement
may first use an owned dense temporary. `LazyArray(source)` assumes only basic
indexing, while `LazyArray.from_numpy(array)` selects `numpy_reader`. Both
currently use the same slab-and-gather implementation.

The built-in readers lower through NumPy system memory and support sources
whose basic reads can be converted there. A device source that refuses NumPy
conversion needs a custom reader that transfers into the output buffer. Derived views and parts share their
reader and part views may be materialized concurrently, so stateful readers
must synchronize their own mutable state.

The key's type picks the frame: NumPy keys are relative positions, `IndexDomain`
and `IndexTransform` keys are absolute, and every view keeps its absolute
domain. See [the guide](../guide/index.md#the-keys-type-picks-the-frame).

Every public `Partition.view.transform` directly maps that view's own domain
coordinates into its raw `Partition.view.array`, including for non-first
partitions; a box part's domain is a sub-domain of the parent view's literal
domain. `Partition.projection.chunk_transform` intentionally stays local to
the selected chunk. During parent materialization (`view.result(parts=parts)`) the reader receives
both frames in one `ReadContext`: the public global transform in `context.transform` and the
local plan in `context.projection`. Every view retains the source grid and plans
its reads, so `part.view.result()` also supplies both frames. Its projection's
result placement is relative to that part view, rather than the parent output.
Further indexing and repartitioning use the same source-global coordinate frame.
Even `unpartitioned()` reads carry a projection for the single source-wide cell.

`view[key]`, `view.oindex[key]`, and `view.vindex[key]` return lazy views,
and iteration yields lazy first-axis views. `result()` and `numpy.asarray(view)`
materialize values. `view.write(values)` synchronously writes to the original
source through the composed transform and returns `None`; `view[key] = values`
writes a selected sub-view. Writes require a writable source.

For consumers requiring eager indexing, import `EagerArrayAdapter` from
`zarr_indexing` and wrap the view. The adapter delegates shape, rank, dtype,
NumPy conversion, and tokenization to the view, but materializes each
`adapter[key]`. Use it with `dask.array.from_array`; direct lazy indexing
is not a reliable Dask block-read interface.

::: zarr_indexing.lazy_array
