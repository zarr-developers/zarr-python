---
title: Design notes
---

# Design notes

This page records advanced rationale that the API does not state directly: how
this library relates to TensorStore, why rectangular selections are a category
rather than a fast path, and what is deliberately not implemented yet. The
visual guide owns the mechanics of
[literal coordinates](guide/index.md#coordinates-are-addresses),
[view composition](guide/index.md#lazy-views-compose),
[chunk plans](guide/index.md#a-request-becomes-a-chunk-plan), and
[their paired projections](guide/index.md#one-cell-domain-two-projections).

## Relationship to TensorStore

The core is [TensorStore's](https://google.github.io/tensorstore/index_space.html)
index-transform model, implemented here in Python against NumPy. The visual guide
introduces the shared model in
[Coordinates are addresses](guide/index.md#coordinates-are-addresses) and
[Lazy views compose](guide/index.md#lazy-views-compose); the comparison here is
about the deliberately matching semantics:

- **The model.** Both use an `IndexTransform` made of an input domain and one
  output index map per storage dimension, in constant, affine, and index-array
  forms.
- **Slice semantics.** Slice bounds are literal domain coordinates: no
  clamping, no negative wrapping, non-empty intervals must be contained in the
  domain, and a strided slice's domain origin is `trunc(start/step)` rounded
  toward zero. `tests/test_tensorstore_parity.py` compares the enumerated cases with
  TensorStore when that optional dependency is installed.
- **The wire format.** [ndsel](ndsel.md) uses TensorStore's domain and
  output-map field names for transform bodies, with an additional `kind`
  discriminator. `tests/test_ndsel_tensorstore.py` checks interoperability for
  the tested cases. Their validation rules differ, and loading and
  re-emitting a message through the engine can normalize or discard metadata;
  see [lowering to a transform](ndsel.md#lowering-to-a-transform).
- **Chunk partitioning.** Both factor a transform over a grid before visiting
  any cell, rather than intersecting the whole transform with each chunk.
  TensorStore's `IndexTransformGridPartition` holds strided sets and index
  array sets and derives a per-cell transform on iteration;
  [`GridPartition`](api/chunk_resolution.md) also partitions index-array
  dependency components. Its `StridedSet` is per storage axis, where TensorStore's
  is per input dimension and spans every grid axis reading it (which is how
  TensorStore factors diagonals). `joint_sets` holds one `JointSet` per
  connected index-array component, including independent single-array
  components within a mixed request. Both derive the per-chunk transforms
  from the partition ([the guide](guide/index.md#a-plan-is-a-product-of-per-axis-tables)
  shows the tables). TensorStore keeps strided sets implicit, while this
  library materializes their per-axis rows for vectorized consumers. Pure
  affine diagonals need grouping by input dimension; mixed affine/index-array
  dependencies need joint partitioning. TensorStore classifies a connected
  component containing index-array edges as an index-array set
  ([source](https://github.com/google/tensorstore/blob/66b2ce5290fa2ec5c8019682391421062ce767a2/tensorstore/internal/grid_partition.h#L58-L67)).

Four deliberate differences:

| | TensorStore | `zarr-indexing` |
| --- | --- | --- |
| Dialect | Coordinate indices are literal; negative coordinates do not wrap | The algebra keeps that dialect; each public boundary picks its own. [`LazyArray`](api/lazy_array.md) speaks positional NumPy, `IndexTransform` speaks literal. [`zarr_indexing.boundary`](api/boundary.md) is the translation |
| Scheduling | An internal C++ scheduler owns concurrency and chunk ordering | [`parts()`](api/lazy_array.md) exposes the partition structure so the caller's own scheduler — dask, a thread pool, a task queue — drives it |
| Wire format | [Documented JSON schema](https://google.github.io/tensorstore/index_space.html#index-transform) | [ndsel](ndsel.md) is spec-first, with a vendored conformance corpus exercised by this implementation |
| Backends | A driver ecosystem (zarr, N5, neuroglancer, GCS, …) built into the library | No drivers. The default reader needs `shape`, `dtype`, basic integer/slice indexing, and selected slabs convertible to NumPy system memory; other backends use explicit custom readers. A device reader owns transfer into the supplied system-memory output |

The mechanics of a
[chunk plan](guide/index.md#a-request-becomes-a-chunk-plan) and its
[paired projections](guide/index.md#one-cell-domain-two-projections) belong to
the visual guide.
The relevant comparison is that both libraries use the paired-transform
boundary rather than a read key plus scatter indices, so slices, outer products,
and correlated gathers remain ordinary transforms that a consumer can lower to
its own execution vocabulary.

The ownership boundary differs. `plan_chunks` retains only the logical request
and caller-supplied grid; it does not own reads, writes, buffers, locks, or
scheduling. Zarr can therefore plan reads against an inner codec-chunk grid and
writes against an atomic shard grid; napari or dask can turn the same
projections into tasks without putting a dask dependency in this package.
`coverage` describes selection coverage relative to that grid. A `full`
classification can help a writer avoid reading old values, but does not by
itself prove that a write is safe: encoding requirements, conflicts, duplicate
semantics, and concurrency remain consumer responsibilities. `unknown` means
the planner has not established complete or partial coverage.

This implementation performs Python-level bookkeeping over NumPy. This page
provides no benchmark establishing a general performance ordering against
TensorStore; costs depend on the selection and execution backend.

The partition tables can be consumed without constructing a `ChunkProjection`
for each chunk. Materializing projections adds Python object construction.

## Bounding-box selections vs query selections

Every selection this library can express falls into exactly one of two
categories. The boundary between them is structural, not a heuristic:

**A box** is a transform whose output maps are all `ConstantMap` or
`DimensionMap` — no `ArrayMap`. Such a map is affine and monotone: storage
coordinate `offset + stride * i` for `i` running over an interval. The whole
selection is therefore described by `O(ndim)` integers — an interval and a
stride per dimension — composition and intersection are interval arithmetic,
and the coordinates it touches form a regular lattice. Basic indexing produces
one, and composing basic indexing with basic indexing keeps one.

**A query** is a transform with at least one `ArrayMap` — an explicit lookup
table of coordinates. Its stored coordinate arrays cost space proportional to their stored size.
Coordinates may repeat or scatter, but can also be contiguous and local.
Current query-resolution paths inspect these arrays. Fancy indexing can produce
a query, but singleton or constant selections can collapse to `ConstantMap`;
subsequent indexing can therefore make a query affine again. A second query composes onto any axis of an existing one —
including the axes it merely broadcasts along — by evaluating the existing
lookup tables at the new coordinates.

Those coordinate arrays are ordered sequences, never mathematical sets. Their
order and duplicate entries are part of the indexing semantics and must survive
planning and materialization.

[ndsel](ndsel.md) encodes the same split in its message kinds: `point`, `box`,
and `slice` desugar to constant and affine output maps and are always boxes;
`points` desugars to `index_array` maps. A transform without index-array maps
is a box in the engine’s structural classification. Loading can further
simplify degenerate index arrays to constants, so an arbitrary incoming body
with `index_array` fields need not remain a query. For example:

```python
from zarr_indexing import IndexTransform

IndexTransform.from_shape((100, 80))[10:50, ::4].to_json()["output"]
# [{'offset': 0, 'stride': 1, 'input_dimension': 0},
#  {'offset': 0, 'stride': 4, 'input_dimension': 1}]

import numpy as np
gather = IndexTransform.from_shape((100, 80)).oindex[np.array([90, 3, 3]), slice(None)]
gather.to_json()["output"][0]
# {'offset': 0, 'stride': 1, 'index_array': [[90], [3], [3]],
#  'index_array_bounds': ['-inf', '+inf']}
```

The representation helps a consumer choose a lowering strategy. Independent
affine axes can often be read with slices plus reversal, permutation, or
broadcasting. Arbitrary affine maps can also express diagonals, so the absence
of `ArrayMap` alone is not proof of a rectangular slab. Queries may be lowered
through gathers or covers, and can sometimes simplify to slices. Chunk plans
group selected coordinates by chunk while preserving result placement; repeated
coordinates do not imply repeated visits to the same chunk.

[`LazyArray`](api/lazy_array.md) exposes the category directly:

```python
import numpy as np
import zarr

from zarr_indexing import LazyArray

arr = zarr.create_array({}, shape=(100, 80), chunks=(30, 40), dtype="int32")
arr[:] = np.arange(8000).reshape(100, 80)
lazy = LazyArray(arr)

slab = lazy[10:50, ::4]
slab.is_box            # True
slab.bounding_box()    # ((10, 50), (0, 77))
slab.strides()         # (1, 4)
slab.shape             # (40, 20)

gather = lazy.oindex[[90, 3, 3], :]
gather.is_box          # False
gather.bounding_box()  # ((3, 91), (0, 80))
gather.strides()       # None
gather.shape           # (3, 80)
```

`bounding_box()` is defined for both: it is the hull, the smallest interval per
storage dimension containing every coordinate the selection reaches.
`strides()` is defined only for a box and gives the step per dimension.
These summaries omit traversal direction, input-axis correspondence, and
result layout. For example, forward and reversed views have identical bounds
and stride magnitudes but different ordered results. Use the transform for the
complete selection.

For independent axes with multiple selected coordinates, a stride magnitude
greater than one leaves gaps in the hull. Singleton axes are an exception,
and a query can also cover every cell of its hull. The slab above spans a 40x77 hull over the 40x20 cells it selects, so a
consumer that issued one rectangular read of the hull and discarded the rest
would transfer 3.85x the data. A query's hull is looser still and carries no
stride at all: 88 rows of hull over three selected rows. An empty *box* touches
no coordinate to report an interval around, so `bounding_box()` is `None` while
`strides()` still answers — the step is a property of the selection's shape, not
of the region it reaches. An empty query returns `None` from both.

There is deliberately no separate `BoxView` type today. A statically-typed
rectangular-only view is a plausible next step, but it should be introduced by
a consumer that needs the guarantee in its signatures rather than
speculatively; `is_box` is the runtime check until then.

## Negative-origin domains and prependable grids

Literal coordinates let a domain grow at its lower end without changing the
identity of anything already present. Prepending three cells extends `[0, 6)`
to `[-3, 6)`: the new cells receive addresses `-3`, `-2`, and `-1`, while the
old cells keep addresses `0` through `5`. Coordinate `0` does not become
coordinate `3`.

The adjacent intervals `[-3, 0)`, `[0, 3)`, and `[3, 6)` follow the half-open
adjacency rule: each stopping boundary is included exactly once as the next
interval's starting boundary.

```text
before [0, 6):

                  |  0   1   2  |  3   4   5  |
chunk coordinate  |      0      |      1      |

after [-3, 6):

| -3  -2  -1  |  0   1   2  |  3   4   5  |
|     -1      |      0      |      1      |  chunk coordinate
```

The same holds for chunk grids. `EdgeDimensionGrid` is the convenient
concrete grid for a zero-origin array: its chunk offsets are prefix sums
starting at zero. `DimensionGridLike` is the more general protocol consumed
by chunk planning, so it admits grids with negative chunk and cell
coordinates, including this prependable example:

```python
--8<-- "snippets/coordinate_origins.py:prepend-grid"
```

Here the literal cell domain `[-3, 0)` belongs to chunk `-1`. Both public
projection transforms share the same synthetic input cell domain `[0, 3)`.
Evaluating its three points shows the two distinct outputs:
`chunk_transform` produces zero-origin chunk-local coordinates `0, 1, 2`,
while `cell_transform` produces the literal request coordinates `-3, -2, -1`.
The shared input domain is not itself the chunk-local coordinate frame.

## Related work

TensorStore is the prior art for the transform algebra, as described above. At
the execution boundary, this package instead gives each backend a `ReadContext`
through a `Reader`. Its global transform answers **which values?**; the reader
answers **how does this backend obtain them?** A partition view's transform
directly addresses the raw source in global coordinates. Its optional
projection retains the paired planning transforms, of which only
`chunk_transform` addresses zero-origin chunk-local coordinates. The reader
must preserve the global transform exactly, but it does not participate in
indexing semantics, partitioning, scheduling, or result ownership.

Earlier versions used a capability taxonomy modeled on historical indexing
dialects. That model required deciding which fragment of a request a backend
could accept and finishing the rest elsewhere. A reader lowers the complete
transform and can compose through delegation instead. This resembles
[zarrita.js store extensions](https://zarrita.dev/packages/zarrita.html), where
storage-specific behavior is an explicit extension point rather than an
inferred array capability. This is an architectural analogy, not a claim of API or implementation
compatibility.

## Current scope

Negative steps are supported as of ndsel 1.0-draft.2: `a[::-1]` reverses, one
desugaring rule covers both signs, and a reversed interval is an error rather
than a silently empty selection. One consequence: a negative step normally
produces a negative domain origin. Reversing a length-20 zero-origin axis gives
the domain `[-19, 1)`, because the result stays anchored to the source
coordinate frame and a reversing map traverses that frame backwards. `LazyArray`
views keep that literal domain, so `view.transform.domain` shows it; positional
keys are normalized against the domain's origin, so the NumPy dialect never
requires typing it. A caller wanting zero-origin coordinates re-bases
explicitly with `translate_domain_to`.

Supported fancy selections compose across already-fancy views: a second `oindex`/`vindex`/mask
step may land on any axis of an already-fancy view, including axes an existing
index array merely broadcasts along, so
`lazy.oindex[[2, 0], :].oindex[:, [1, 3]]` selects the outer product it
spells. An array-carrying transform is composed — the new selection is applied
to an identity transform over the current domain and chained on with `compose`,
which evaluates the existing lookup tables at the new coordinates — rather than
rewritten in place. Resolution classifies the result by structure
(`index_array_structure`): pure per-axis outer products keep the orthogonal
resolvers, and everything else — correlated maps, mixtures, index arrays
sharing an input axis (as in paired vectorized coordinates) — takes the general reader/intersection path. Chunk planning
factors index arrays into connected dependency components before flattening,
so independent groups do not expand one another. Vectorized selection preserves
broadcast singletons to retain those dependencies.

Some current limits are:

- **Shared affine dependencies.** Planning rejects two affine output maps
  sharing an input axis with `ValueError`. An index array sharing a varying
  input axis with an affine map takes the general classification and raises
  `NotImplementedError`. Pure affine diagonals would need grouping dependent
  storage axes by input dimension; mixed components need joint partitioning.
- **Finite explicit bounds only.** `IndexDomain` has no implicit or unbounded
  dimensions; the message layer will normalize a body with `"-inf"`/`"+inf"`
  bounds, but the engine layer refuses to lower one into a transform.
  TensorStore supports both.
- **Labels are carried, not propagated.** `IndexDomain` holds optional
  dimension labels and the wire format round-trips them, but indexing
  operations build new domains without them, so a label does not survive a
  slice.

## Selection to chunk operations

[The selection-flow guide](guide/selection-flow.md) traces normalization,
partitioning, and paired chunk-local/request projections with executable examples.
