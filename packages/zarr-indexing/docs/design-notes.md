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
index-transform model, reimplemented in Python against NumPy. The visual guide
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
  toward zero. Every one of those rules was executed against tensorstore 0.1.84
  and is pinned in `tests/test_tensorstore_parity.py`.
- **The wire format.** A canonical [ndsel](ndsel.md) transform body is,
  field-for-field, a TensorStore `IndexTransform` minus the `kind`
  discriminator, and `tests/test_ndsel_tensorstore.py` loads our bodies into
  `tensorstore.IndexTransform(json=...)` and round-trips them back through our
  engine layer.

The representations differ in one place: index arrays. Both models want an index
array at the transform's full input rank, with singleton axes for the dimensions
a map does not vary over. TensorStore enforces it — its JSON parser rejects a
rank-1 array over a rank-2 domain outright, with `Index array for output
dimension 0 has rank 1 but must have rank 2` (checked against tensorstore
0.1.84) — while our loader is the more permissive of the two and also accepts a
lower-rank array that broadcasts against the input domain. That is a
compatibility affordance, not a difference in the model: ndsel leaves index-array
rank to [the engine layer](ndsel.md#lowering-to-a-transform), and everything the
algebra builds itself is at full rank.

The reason full rank matters here is that we *derive* meaning from those
singletons rather than merely tolerating them: an array full-sized on one axis
and singleton elsewhere is orthogonal, and one varying over several shared axes
is vectorized, so the distinction is readable off the shape — and the shape is
the *only* place it lives. An earlier `ArrayMap.input_dimension` field pinned
the orthogonal axis redundantly and was retired: the one shape it disambiguated
(a single-coordinate array, all axes singleton) is now normalized away at
construction, collapsed to the `ConstantMap` it equals, exactly as
[the serializer](api/json.md) has always collapsed it on the wire.

Four deliberate differences:

| | TensorStore | `zarr-indexing` |
| --- | --- | --- |
| Dialect | One strict dialect everywhere: literal coordinates, no negative wrapping | The algebra keeps that dialect; each public boundary picks its own. [`LazyArray`](api/lazy_array.md) speaks positional NumPy, `zarr.Array.lazy` speaks literal. [`zarr_indexing.boundary`](api/boundary.md) is the translation |
| Scheduling | An internal C++ scheduler owns concurrency and chunk ordering | [`parts()`](api/lazy_array.md) exposes the partition structure so the caller's own scheduler — dask, a thread pool, a task queue — drives it |
| Wire format | Implementation-defined JSON, specified by what the implementation accepts | [ndsel](ndsel.md) is spec-first, with a vendored language-agnostic conformance corpus every implementation runs |
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
`coverage` is relative to that selected grid: `full` proves a blind replacement
safe, `partial` proves it is not, and `unknown` conservatively covers fancy
selections whose duplicates would require additional work to classify.

The comparison also runs the other way. TensorStore is a mature, heavily
optimized C++ system whose performance this library cannot approach: resolution
here is Python-level bookkeeping over NumPy, and the per-part overhead is
significant. This library is small and depends on nothing beyond NumPy, so the
algebra can be adopted by a Python project that wants the model without the C++
runtime.

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
table of coordinates. It costs `O(n)` to store, it has no locality (the
coordinates may repeat, reverse, or scatter arbitrarily), and intersecting it
with a region means scanning it. `oindex`, `vindex`, and boolean masks all
produce one, and once an axis is a query, subsequent basic indexing cannot make
it a box again. A second query composes onto any axis of an existing one —
including the axes it merely broadcasts along — by evaluating the existing
lookup tables at the new coordinates.

Those coordinate arrays are ordered sequences, never mathematical sets. Their
order and duplicate entries are part of the indexing semantics and must survive
planning and materialization.

[ndsel](ndsel.md) encodes the same split in its message kinds: `point`, `box`,
and `slice` desugar to constant and affine output maps and are always boxes;
`points` desugars to `index_array` maps, and a `transform` body is a box
exactly when none of its output maps carries an `index_array`. A consumer can
therefore classify a selection off the wire without materializing anything:

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

The distinction matters to consumers of a selection. A box can be tiled into
rectangular dask chunks or passed to a viewer or tile server that only accepts
rectangles; a query cannot, and has to be resolved into a gather. A box can also
be served as a single strided slab read, but the read has to be strided: reading
its bounding box and discarding the rest transfers proportionally more data as
soon as any stride exceeds 1. The two also behave differently under
partitioning: a box touches a regularly-spaced run of parts, in increasing
order, each at most once — a stride larger than a part's extent skips parts
outright, so the run is not contiguous — while a query can touch any subset of
them, in any order, more than once.

[`LazyArray`](api/lazy_array.md) exposes the category directly:

```python
import numpy as np
import zarr

from zarr_indexing import LazyArray

arr = zarr.create_array({}, shape=(100, 80), chunks=(30, 40), dtype="int32")
arr[:] = np.arange(8000).reshape(100, 80)
lazy = LazyArray(arr)

slab = lazy.lazy[10:50, ::4]
slab.is_box            # True
slab.bounding_box()    # ((10, 50), (0, 77))
slab.strides()         # (1, 4)
slab.shape             # (40, 20)

gather = lazy.lazy.oindex[[90, 3, 3], :]
gather.is_box          # False
gather.bounding_box()  # ((3, 91), (0, 80))
gather.strides()       # None
gather.shape           # (3, 80)
```

`bounding_box()` is defined for both: it is the hull, the smallest interval per
storage dimension containing every coordinate the selection reaches.
`strides()` is defined only for a box and gives the step per dimension.
Together the two describe a box selection completely.

Both are needed, because a box is dense in its hull only when every stride is
1. The slab above spans a 40x77 hull over the 40x20 cells it selects, so a
consumer that issued one rectangular read of the hull and discarded the rest
would transfer 3.85x the data. A query's hull is looser still and carries no
stride at all: 88 rows of hull over three selected rows. An empty *box* touches
no coordinate to report an interval around, so `bounding_box()` is `None` while
`strides()` still answers — the step is a property of the selection's shape, not
of the region it reaches. Only a query returns `None` from both.

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
inferred array capability. The implementation remains independently authored:
no code is shared with TensorStore, xarray, or zarrita.js.

## Current scope

Negative steps are supported as of ndsel 1.0-draft.2: `a[::-1]` reverses, one
desugaring rule covers both signs, and a reversed interval is an error rather
than a silently empty selection. One consequence: a negative step normally
produces a negative domain origin. Reversing a length-20 zero-origin axis gives
the domain `[-19, 1)`, because the result stays anchored to the source
coordinate frame and a reversing map traverses that frame backwards. `LazyArray`
re-bases every view to origin 0, so the positional dialect never exposes it; a
caller working with `IndexTransform` directly will see it, and re-bases
explicitly with `translate_domain_to` for NumPy-shaped coordinates.

Fancy selections compose without restriction: a second `oindex`/`vindex`/mask
step may land on any axis of an already-fancy view, including axes an existing
index array merely broadcasts along, so
`lazy.oindex[[2, 0], :].lazy.oindex[:, [1, 3]]` selects the outer product it
spells. An array-carrying transform is composed — the new selection is applied
to an identity transform over the current domain and chained on with `compose`,
which evaluates the existing lookup tables at the new coordinates — rather than
rewritten in place. Resolution classifies the result by structure
(`index_array_structure`): pure per-axis outer products keep the orthogonal
resolvers, and everything else — correlated maps, mixtures, index arrays
sharing an input axis (a diagonal gather, reachable only by hand-building a
transform) — takes the pointwise path that collapses the joint block.

Three limits remain, all intentional and all expected to be lifted:

- **Affine diagonals.** A hand-built transform in which an *index array* and a
  *slice map* bind the same input dimension, or two slice maps share one, is
  rejected at resolution with `NotImplementedError`. No selection dialect
  produces one; supporting them means lowering the slice maps into the joint
  block too. *Planned.*
- **Finite explicit bounds only.** `IndexDomain` has no implicit or unbounded
  dimensions; the message layer will normalize a body with `"-inf"`/`"+inf"`
  bounds, but the engine layer refuses to lower one into a transform.
  TensorStore supports both. *Planned.*
- **Labels are carried, not propagated.** `IndexDomain` holds optional
  dimension labels and the wire format round-trips them, but indexing
  operations build new domains without them, so a label does not survive a
  slice. *Planned.*
