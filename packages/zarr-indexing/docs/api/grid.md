---
title: grid
---

`zarr_indexing.grid` owns compact chunk-grid metadata so indexing plans can be
constructed without importing Zarr. `FixedDimension(size, extent)` represents
regular chunks in constant memory, including a clipped final data region;
`VaryingDimension(edges, extent)` represents explicit rectilinear chunk edges.
`ChunkGrid(dimensions=...)` combines these dimensions and returns `ChunkSpec`
objects whose `shape` is the valid data size and whose `codec_shape` preserves
the full codec-buffer size at a regular-grid boundary.

`dimension_grids_from_chunks` returns these compact dimensions: integer chunk
shapes become `FixedDimension` instances and explicit per-axis edge sequences
become `VaryingDimension` instances. `DimensionGridLike` remains the narrow
protocol used by the chunk planner, while `EdgeDimensionGrid` is kept for
explicit edge-based and coordinate-origin examples.

Zarr's array implementation can later import these compact grid types from
`zarr_indexing`; this package intentionally has no import dependency on Zarr.

::: zarr_indexing.grid
