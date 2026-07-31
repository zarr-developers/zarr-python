---
title: API reference
---

# API reference

The modules are layered: the transform algebra at the bottom, chunk resolution
and the wire format built on top of it.

**The transform algebra**

- [`zarr_indexing.domain`](domain.md) — `IndexDomain`, a rectangular region of
  integer coordinates with an explicit (possibly non-zero) origin
- [`zarr_indexing.output_map`](output_map.md) — `ConstantMap`, `DimensionMap`,
  and `ArrayMap`: three representations of a set of integer coordinates, one
  per storage dimension
- [`zarr_indexing.transform`](transform.md) — `IndexTransform`, which pairs a
  domain with output maps, plus the indexing (`[...]`, `.oindex`, `.vindex`),
  `intersect`, and `translate` operations, and `selection_to_transform`
- [`zarr_indexing.composition`](composition.md) — `compose`, which chains two
  transforms into one

**Chunk resolution**

- [`zarr_indexing.chunk_resolution`](chunk_resolution.md) —
  `iter_chunk_transforms` (transform + chunk grid → per-chunk transforms) and
  `sub_transform_to_selections` (the bridge back to the selection tuples the
  current codec pipeline expects)
- [`zarr_indexing.grid`](grid.md) — `DimensionGridLike`, the Protocol
  describing the narrow chunk-grid surface chunk resolution consumes, so that
  nothing here imports `zarr`, plus `EdgeDimensionGrid` and
  `dimension_grids_from_chunks`, a concrete per-axis grid for callers with no
  zarr grid to hand

**Lazy arrays**

- [`zarr_indexing.lazy_array`](lazy_array.md) — `LazyArray`, a wrapper that
  gives any array-API-like array (NumPy, zarr, CuPy, …) a `.lazy` accessor for
  TensorStore-style deferred indexing, plus `Partition` and `parts()` /
  `with_parts()`, which determine the boxes a read is broken into
- [`zarr_indexing.boundary`](boundary.md) — the translation between NumPy's
  positional dialect and the transform algebra's literal coordinates

**The ndsel wire format** (see [the guide](../ndsel.md))

- [`zarr_indexing.messages`](messages.md) — `parse_ndsel` / `normalize_ndsel`,
  the pure JSON→JSON message layer, and `NdselError`
- [`zarr_indexing.json`](json.md) — lowering between canonical ndsel bodies and
  in-memory transforms

**Errors**

- [`zarr_indexing.errors`](errors.md) — the canonical index-error types, which
  `zarr.errors` re-exports by identity

Every name listed in `zarr_indexing.__all__` is re-exported at the top level,
so `from zarr_indexing import IndexTransform` and
`from zarr_indexing.transform import IndexTransform` are equivalent.
