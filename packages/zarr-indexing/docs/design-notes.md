---
title: Design notes
---

# Design notes

Three topics the API does not state directly: how this library relates to
TensorStore, why rectangular selections are a category rather than a fast path,
and what is deliberately not implemented yet.

## Relationship to TensorStore

The core is [TensorStore's](https://google.github.io/tensorstore/index_space.html)
index-transform model, reimplemented in Python against NumPy. The following
parts are intentionally identical:

- **The model.** An `IndexTransform` pairs an input `IndexDomain` — a
  rectangular region with an explicit, possibly non-zero origin — with one
  output index map per storage dimension, in the three flavors TensorStore
  defines: constant, single-input-dimension (affine), and index array.
  Composition is the single operation that stacks views.
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

The representations differ in one place: index arrays. Ours are normalized to
the transform's full input rank — full-sized on the axis a map varies over,
singleton elsewhere — so the orthogonal/vectorized distinction is derivable
from the shape. TensorStore permits a lower-rank array that broadcasts
against the input domain; we accept those on load and normalize them.
`ArrayMap` also records the input dimension an *orthogonal* (`oindex`) array
varies over, a field TensorStore's format has no slot for, so
[the serializer](api/json.md) collapses it on the way out and reconstructs it on
the way in.

Four deliberate differences:

| | TensorStore | `zarr-indexing` |
| --- | --- | --- |
| Dialect | One strict dialect everywhere: literal coordinates, no negative wrapping | The algebra keeps that dialect; each public boundary picks its own. [`LazyArray`](api/lazy_array.md) speaks positional NumPy, `zarr.Array.lazy` speaks literal. [`zarr_indexing.boundary`](api/boundary.md) is the translation |
| Scheduling | An internal C++ scheduler owns concurrency and chunk ordering | [`parts()`](api/lazy_array.md) exposes the partition structure so the caller's own scheduler — dask, a thread pool, a task queue — drives it |
| Wire format | Implementation-defined JSON, specified by what the implementation accepts | [ndsel](ndsel.md) is spec-first, with a vendored language-agnostic conformance corpus every implementation runs |
| Backends | A driver ecosystem (zarr, N5, neuroglancer, GCS, …) built into the library | No drivers. The wrapper takes anything with `shape`, `dtype`, and `__getitem__`, and prefers the array's own `__array_namespace__` |

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
it a box again.

[ndsel](ndsel.md) encodes the same split in its message kinds: `point`, `box`,
and `slice` desugar to constant and affine output maps and are always boxes;
`points` desugars to `index_array` maps, and a `transform` body is a box
exactly when none of its output maps carries an `index_array`. A consumer can
therefore classify a selection off the wire without materializing anything:

```python
from zarr_indexing import IndexTransform, transform_to_canonical

transform_to_canonical(IndexTransform.from_shape((100, 80))[10:50, ::4])["output"]
# [{'offset': 0, 'stride': 1, 'input_dimension': 0},
#  {'offset': 0, 'stride': 4, 'input_dimension': 1}]

import numpy as np
gather = IndexTransform.from_shape((100, 80)).oindex[np.array([90, 3, 3]), slice(None)]
transform_to_canonical(gather)["output"][0]
# {'offset': 0, 'stride': 1, 'index_array': [[90], [3], [3]],
#  'index_array_bounds': ['-inf', '+inf']}
```

The distinction matters to consumers of a selection. A box can be tiled into
rectangular dask chunks or passed to a viewer or tile server that only accepts
rectangles; a query cannot, and has to be resolved into a gather. A box can also
be served as a single strided slab read, but the read has to be strided: reading
its bounding box and discarding the rest transfers proportionally more data as
soon as any stride exceeds 1. The two also behave differently under
partitioning: a box touches a contiguous run of parts, while a query can touch
any subset of them, in any order, more than once.

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
stride at all: 88 rows of hull over three selected rows. An empty selection
returns `None` from both, because it touches no coordinate to report an interval
around.

There is deliberately no separate `BoxView` type today. A statically-typed
rectangular-only view is a plausible next step, but it should be introduced by
a consumer that needs the guarantee in its signatures rather than
speculatively; `is_box` is the runtime check until then.

## Related work

The closest neighbour to this package in the Python ecosystem is xarray's
[`xarray/core/indexing.py`](https://github.com/pydata/xarray/blob/main/xarray/core/indexing.py).
It solves the same problem — carrying a selection around as a value, and
deciding how much of it a given backend can be asked to perform — and the
`IndexingSupport` taxonomy here (`BASIC`, `OUTER`, `OUTER_1VECTOR`,
`VECTORIZED`) is taken from it, member names and meanings included, so that a
reader who knows one knows the other. The split of a selection into a part
pushed to the source and a part finished with NumPy follows xarray's
`decompose_indexer`, and the heuristic for choosing which axis keeps its
coordinate array when only one can is xarray's.

This package is a standalone implementation rather than a port: no code is
shared, the selection is carried as an `IndexTransform` rather than as xarray's
explicit indexer classes, and the decomposition is applied per chunk partition
as well as per array, so a source is asked for one key per box rather than one
key per read.

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

Two limits remain, both intentional and expected to be lifted:

- **Finite explicit bounds only.** `IndexDomain` has no implicit or unbounded
  dimensions; the message layer will normalize a body with `"-inf"`/`"+inf"`
  bounds, but the engine layer refuses to lower one into a transform.
  TensorStore supports both. *Planned.*
- **Labels are carried, not propagated.** `IndexDomain` holds optional
  dimension labels and the wire format round-trips them, but indexing
  operations build new domains without them, so a label does not survive a
  slice. *Planned.*
