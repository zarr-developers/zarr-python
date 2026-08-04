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

::: zarr_indexing.lazy_array
