---
title: Eager adapter
---

# Eager adapter

`LazyArray` indexing returns views. Consumers that require indexing to
produce data, such as `dask.array.from_array`, wrap a view in
`EagerArrayAdapter`: the adapter reads each indexed block through the view's
reader and partitioning, while the view itself keeps lazy indexing.

::: zarr_indexing.eager
