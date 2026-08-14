---
title: API reference
---

# API reference

Choose the guide stopping point that matches your job before following module
links:

- **Use lazy indexing:** finish
  [Lazy views compose](../guide/index.md#lazy-views-compose),
  then open [`zarr_indexing.lazy_array`](lazy_array.md) for `LazyArray`.
- **Integrate a chunked source:** finish
  [One cell domain, two projections](../guide/index.md#one-cell-domain-two-projections),
  then open [`zarr_indexing.chunk_resolution`](chunk_resolution.md) for
  `plan_chunks`. Start with
  [Coordinates are addresses](../guide/index.md#coordinates-are-addresses) if
  literal coordinates are unfamiliar.

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
  transforms into one

**Chunk resolution**

- [`zarr_indexing.chunk_resolution`](chunk_resolution.md) —
  `plan_chunks`, which lazily projects a request through a caller-selected grid,
  plus the reusable `ChunkPlan` and paired-transform `ChunkProjection` values
- [`zarr_indexing.grid`](grid.md) — `DimensionGridLike`, the Protocol
  describing the narrow chunk-grid surface chunk resolution consumes, so that
  nothing here imports `zarr`, plus `EdgeDimensionGrid` and
  `dimension_grids_from_chunks`, a concrete per-axis grid for callers with no
  zarr grid to hand

**Lazy arrays**

- [`zarr_indexing.lazy_array`](lazy_array.md) — `LazyArray`, a wrapper for
  system-memory/basic-indexing sources that adds a `.lazy` accessor for
  TensorStore-style deferred indexing, plus `Partition` and `parts()` /
  `with_parts()`, which determine the boxes a read is broken into. Device
  sources require an explicit custom reader that transfers into the supplied
  system-memory output
- [`zarr_indexing.reader`](reader.md) — `Reader`, the backend execution boundary
  that obtains the values described by a complete transform; `basic_reader`
  serves conservative duck arrays and `numpy_reader` is selected explicitly by
  `LazyArray.from_numpy`
- [`zarr_indexing.boundary`](boundary.md) — the translation between NumPy's
  positional dialect and the transform algebra's literal coordinates

**The ndsel wire format** (see [the guide](../ndsel.md))

- [`zarr_indexing.messages`](messages.md) — `parse_ndsel` / `normalize_ndsel`,
  the pure JSON→JSON message layer, and `NdselError`
- [`zarr_indexing.json`](json.md) — lowering between canonical ndsel bodies and
  in-memory transforms

**Errors**

- [`zarr_indexing.errors`](errors.md) — the index-error types this package
  raises, also exported at the top level. `zarr.errors` defines classes of the
  same names, which are different objects; both subclass `IndexError`

**Test support** (needs the `testing` extra)

- [`zarr_indexing.testing.stateful`](testing_stateful.md) —
  `ChainedIndexingStateMachine`, a Hypothesis state machine that composes
  indexing steps onto a `LazyArray` wrapping your array and checks every step
  against NumPy, plus `apply_selection`, the NumPy model it checks against
- [`zarr_indexing.testing.strategies`](testing_strategies.md) — the selection
  strategies the machine draws from, for a project that has its own harness

Every name listed in `zarr_indexing.__all__` is re-exported at the top level,
so `from zarr_indexing import IndexTransform` and
`from zarr_indexing.transform import IndexTransform` are equivalent.
`zarr_indexing.testing` is deliberately not among them: it imports
`hypothesis`, which the rest of the package does not.
