# Chunk Projection Design

## Goal

Replace the provisional chunk-resolution tuple with a stable, source-independent
projection model based on TensorStore's paired-transform design. The result must be useful
to zarr-python's codec pipeline and to external schedulers such as Dask or napari without
embedding source objects, NumPy-specific scatter indices, caching, or execution policy.

This package is greenfield. There are no dependent projects and the existing public API
does not require compatibility shims.

## Decision

`IndexTransform` remains the logical selection: it maps the caller-visible request domain
to storage coordinates. Chunk planning partitions that transform over a caller-supplied
grid and yields one `ChunkProjection` per touched grid cell.

Each projection uses a synthetic cell domain and carries two transforms with exactly that
input domain:

- `chunk_transform`: synthetic cell coordinates to chunk-local storage coordinates.
- `cell_transform`: synthetic cell coordinates to the original request domain.

This replaces the current `(chunk_coords, sub_transform, out_indices)` result. In
particular, `out_indices` is not exposed: it is an incomplete second representation whose
shape changes between basic, orthogonal, and vectorized indexing. A transform represents
all three uniformly and composes with the rest of the package.

## Public API

```python
from dataclasses import dataclass
from typing import Literal

ChunkCoverage = Literal["full", "partial", "unknown"]


@dataclass(frozen=True, slots=True)
class ChunkProjection:
    chunk_coords: tuple[int, ...]
    chunk_domain: IndexDomain
    chunk_transform: IndexTransform
    cell_transform: IndexTransform
    coverage: ChunkCoverage


@dataclass(frozen=True, slots=True)
class ChunkPlan:
    transform: IndexTransform
    dimension_grids: tuple[DimensionGridLike, ...]

    def projections(self) -> Iterator[ChunkProjection]: ...
    def __iter__(self) -> Iterator[ChunkProjection]: ...


def plan_chunks(
    transform: IndexTransform,
    dimension_grids: Sequence[DimensionGridLike],
) -> ChunkPlan: ...
```

`ChunkPlan` is re-iterable and cheap to construct. It retains only the logical request and
the selected grid; projections remain lazy. The caller chooses the grid appropriate for
the operation. A zarr integration should pass the no-read-amplification grid for reads and
the atomic/no-write-amplification grid for writes. The indexing package does not invent a
read/write layout abstraction before it has one to consume.

`iter_chunk_transforms` and its tuple result cease to be public. The existing intersection
machinery may continue to use temporary survivor arrays internally, but they must be
converted into `cell_transform` before crossing the chunk-resolution boundary.

## Projection Invariants

For every projection:

1. `chunk_transform.domain == cell_transform.domain`.
2. `chunk_transform` maps every cell into `[0, chunk_shape)`.
3. `cell_transform` is one-to-one and maps every cell into the plan transform's input
   domain.
4. Composing the plan transform with `cell_transform`, then subtracting the chunk origin,
   is equivalent to `chunk_transform`.
5. Across one plan, `cell_transform` ranges are disjoint and their union is the request
   domain. Storage coordinates may repeat because fancy indexing may request the same
   element more than once.

Basic and orthogonal intersections retain their natural rank. Correlated intersections may
collapse their broadcast block to a synthetic points dimension, just as the existing
resolver does; `cell_transform` maps those synthetic points back to the original broadcast
axes.

## Coverage

Coverage is computed against the grid supplied to `plan_chunks`:

- `"full"`: the projection is proven to address every cell of the grid cell exactly once.
- `"partial"`: it is proven not to cover the entire grid cell.
- `"unknown"`: exact proof would require materializing or sorting fancy coordinates.

Affine unit-stride projections can be classified exactly. Fancy projections are
`"unknown"` unless the resolver already has a cheap exact proof. Consumers may skip a
read-modify-write only for `"full"`; both other values are conservative.

Coverage is an optimization, not part of selection correctness. A caller planning against
a shard/write grid therefore receives shard coverage, while one planning against an inner
read grid receives inner-chunk coverage.

## `LazyArray` Integration

`LazyArray.parts()` constructs a `ChunkPlan` from the view transform and discovered read
partitioning. Each `Partition` contains its `ChunkProjection` and a source-bound `LazyArray`
view for resolving that projection. Source binding remains above the pure plan.

NumPy placement used by `LazyArray.result()` is derived internally from
`projection.cell_transform`. It is not stored in `ChunkProjection` and is not the public
chunk-planning contract. This keeps execution conveniences available without forcing
NumPy indexing tuples on Dask, napari, codec, or non-NumPy consumers.

Caching, scheduling, cancellation, locks, and I/O are deliberately absent from both
`ChunkPlan` and `ChunkProjection`. Those belong to source/execution adapters. A later async
executor can stream `plan.projections()` and stop between projections without changing the
planning model.

## Error Handling

`plan_chunks` validates that the number of dimension grids equals the transform output
rank. Construction otherwise performs no chunk walk. Errors that depend on coordinates or
grid boundaries arise while iterating projections and identify the offending dimension or
chunk coordinate.

`ChunkProjection` validates its shared-domain invariant at construction. Violations are
internal errors and raise `ValueError` with both domains in the message.

Empty request domains yield an empty projection iterator.

## Testing

Tests exercise observable transform semantics rather than dataclass field presence alone.

- Basic, reversed, orthogonal, correlated, repeated, scalar, empty, and irregular-grid
  selections verify all projection invariants.
- For each selection, applying `chunk_transform` to the chunk and placing the result through
  `cell_transform` reconstructs the direct selection exactly.
- Tests verify that cell ranges tile the request domain once even when storage coordinates
  repeat.
- Coverage tests exercise full, partial, and unknown outcomes, including clipped edge cells.
- A plan is iterated twice to prove it is reusable and lazy.
- `LazyArray.parts()` and `result()` tests prove source binding and assembly use the new
  projection contract.
- Public export and documentation tests ensure the tuple resolver is no longer advertised.

The implementation follows test-driven development: each new public behavior is first
introduced by a focused failing test, then implemented minimally, then covered by the full
`zarr-indexing` test suite and lint/type checks.
