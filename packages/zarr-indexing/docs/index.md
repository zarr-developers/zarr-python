# zarr-indexing

Composable, lazy coordinate transforms for Zarr array indexing.

`zarr-indexing` is developed in the
[zarr-python repository](https://github.com/zarr-developers/zarr-python/tree/main/packages/zarr-indexing)
and released independently of `zarr` itself. Install it with:

```
pip install zarr-indexing
```

## What this is

An indexing operation — a slice, an integer, a fancy index array — is a mapping
from the coordinates a user asks for to the coordinates in storage. This library
represents that mapping as a value: an [`IndexTransform`](api/transform.md).
Transforms compose, so a view of a view of an array is still a single transform,
and no data is read until a caller asks for it.

The package has three parts:

- **The transform algebra** ([`zarr_indexing.transform`](api/transform.md),
  [`zarr_indexing.domain`](api/domain.md),
  [`zarr_indexing.output_map`](api/output_map.md),
  [`zarr_indexing.composition`](api/composition.md)): an `IndexTransform`
  pairs an input [`IndexDomain`](api/domain.md) — a rectangular region of
  integer coordinates, which unlike NumPy may have a non-zero origin — with
  one output map per storage dimension. `ConstantMap`, `DimensionMap`, and
  `ArrayMap` are three representations of the same thing, a set of integer
  coordinates, traded off against each other for efficiency.
- **Chunk resolution** ([`zarr_indexing.chunk_resolution`](api/chunk_resolution.md)):
  given a transform and a chunk grid, compute which chunks the selection
  touches, which coordinates it touches inside each chunk, and where the values
  land in the output buffer. The resolver is dependency-aware: correlated
  (`vindex`) array maps are enumerated jointly rather than as a cartesian
  product, and orthogonal (`oindex`) array maps contribute only the chunks
  their index arrays actually land in, so resolution scales with the number of
  selected coordinates instead of with the size of the grid.
- **A wire format** ([`zarr_indexing.messages`](api/messages.md),
  [`zarr_indexing.json`](api/json.md)): selections serialize to and from
  [ndsel](https://github.com/zarr-developers/ndsel), a JSON representation of
  NumPy-style n-dimensional selections. See [the ndsel wire format](ndsel.md).

The package depends only on NumPy and the standard library. In particular it
does not import `zarr`: the chunk-grid surface chunk resolution needs is
described by the [`DimensionGridLike`](api/grid.md) Protocol, which zarr's
per-dimension grids satisfy structurally.

## Relationship to TensorStore

The model is [TensorStore's](https://google.github.io/tensorstore/index_space.html)
index transform, reimplemented in Python against NumPy: index domains with
explicit origins, output index maps of constant / single-input-dimension /
index-array flavor, and composition as the single operation that stacks
views. Names and semantics follow TensorStore where they overlap: in
particular, negative indices are literal coordinates, not Python-style offsets
from the end, and the caller is responsible for normalizing them.

The differences are those NumPy compatibility requires. `ArrayMap` records
the input dimension an *orthogonal* (`oindex`) index array varies over, which
TensorStore's format has no field for; the
[serializer collapses or reconstructs that field](api/json.md) so the wire
format stays TensorStore-loadable. Chunk resolution and the `oindex`/`vindex`
helpers exist to serve NumPy-shaped selection semantics, which TensorStore
does not have to model.

The [design notes](design-notes.md) give the comparison in full, including the
four places the two libraries deliberately differ and the areas where
TensorStore is more capable.

## Quickstart

Indexing a transform produces a new transform. No I/O happens, and no
coordinates are materialized:

```python
from zarr_indexing import IndexTransform

transform = IndexTransform.from_shape((100, 100))

view = transform[10:50, 5]
view.domain          # IndexDomain(inclusive_min=(10,), exclusive_max=(50,))
view.selection_repr  # '{ [10, 50), 5 }'
```

The domain describes the coordinates the user sees (here a single dimension of
length 40 with origin 10); the output maps describe the storage coordinates (a
stride-1 `DimensionMap` and a `ConstantMap` for the dropped dimension).

Fancy indexing works the same way in both flavors, and materializes nothing but
the index arrays themselves:

```python
import numpy as np

transform.oindex[np.array([3, 1, 90]), 0:4]   # '{ {3, 1, 90}, [0, 4) }', shape (3, 4)
transform.vindex[np.array([0, 40, 99]), np.array([1, 2, 3])]  # shape (3,)
```

Transforms built independently are combined with
[`compose`](api/composition.md), which is what `transform[...]` uses
internally when indexing an already-indexed view:

```python
from zarr_indexing import compose

inner = IndexTransform.from_shape((100,))[::2]  # storage 0, 2, 4, ... over domain [0, 50)
outer = IndexTransform.from_shape((50,))[10:20]

compose(outer, inner).selection_repr  # '{ [20, 40) step 2 }'
```

Chunk resolution applies a transform to a chunk grid. It asks the grid only for
the per-dimension index-to-chunk mapping described by
[`DimensionGridLike`](api/grid.md), so any object with those four methods
works:

```python
from dataclasses import dataclass

from zarr_indexing import iter_chunk_transforms


@dataclass(frozen=True)
class RegularDimensionGrid:
    chunk: int

    def index_to_chunk(self, idx): return idx // self.chunk
    def chunk_offset(self, chunk_ix): return chunk_ix * self.chunk
    def chunk_size(self, chunk_ix): return self.chunk
    def indices_to_chunks(self, indices): return indices // self.chunk


grids = [RegularDimensionGrid(32), RegularDimensionGrid(32)]

for chunk_coords, sub_transform, out_indices in iter_chunk_transforms(view, grids):
    print(chunk_coords, sub_transform.selection_repr)
# (0, 0) { [10, 32), 5 }
# (1, 0) { [0, 18), 5 }
```

Each yielded `sub_transform` is the original transform restricted to one chunk
and translated into chunk-local coordinates, which is what a codec pipeline
needs to decode that chunk and scatter the result. `out_indices` carries the
output scatter indices for array selections, and is `None` for basic indexing.

## Lazy views over any array

[`LazyArray`](api/lazy_array.md) applies all of the above to an existing array —
NumPy, zarr, CuPy, anything with `shape`, `dtype`, and `__getitem__`. Its
`.lazy` accessor composes transforms instead of reading, and `result()` resolves
the composed view in one pass:

```python
import numpy as np
import zarr

from zarr_indexing import LazyArray

arr = zarr.create_array({}, shape=(100, 80), chunks=(30, 40), dtype="int32")
arr[:] = np.arange(8000).reshape(100, 80)

lazy = LazyArray(arr)

view = lazy.lazy[10:50, ::4].lazy.oindex[[3, 0, 0], :]
view.shape   # (3, 20)
repr(view)   # '<LazyArray Array shape=(3, 20) dtype=int32 view={ {13, 10, 10}, [0, 80) step 4 }>'

view.result()[:, :4]
# array([[1040, 1044, 1048, 1052],
#        [ 800,  804,  808,  812],
#        [ 800,  804,  808,  812]], dtype=int32)
```

In the transform algebra underneath, indices are literal coordinates in a view's
own (possibly shifted) domain. `LazyArray` instead uses **positional NumPy
semantics**: a view is re-zeroed, `-1` is the last element, and boolean masks
must match the view's shape. `.lazy.oindex` and `.lazy.vindex` give the
orthogonal and vectorized flavors:

```python
LazyArray(np.arange(12).reshape(3, 4)).lazy.vindex[[0, 2], [1, 3]].result()
# array([ 1, 11])
```

Two further NumPy rules hold in every mode. A scalar integer drops its axis: it
is a basic index wherever it appears, applied before any advanced index rather
than broadcast against one. Advanced indices are placed as NumPy places them:
next to the axes they replaced when they are adjacent, and leading when a slice
separates them:

```python
x = np.arange(12).reshape(3, 4)

LazyArray(x).lazy.oindex[1].result()
# array([4, 5, 6, 7])

LazyArray(x).lazy.oindex[[2, 0], 1].result()
# array([9, 1])

LazyArray(x).lazy.vindex[..., [3, 0]].result()
# array([[ 3,  0],
#        [ 7,  4],
#        [11,  8]])
```

Use a length-1 list (`oindex[[1]]`) to keep an axis.

Plain `lazy[...]` (no `.lazy`) reads eagerly, which makes a `LazyArray` usable
as a source for `dask.array.from_array`. Any NumPy *function* given a view —
`numpy.sum(view)`, `numpy.add(view, 1)`, `numpy.stack([view, view])` —
materializes the whole thing through `__array__` and works on the result.
Python's arithmetic operators do not: `view + 1` raises `TypeError`, because the
wrapper defines no arithmetic dunders. Laziness here applies to indexing, not to
deferred computation.

### Parts

A read is broken up along a **partitioning**: a grid of boxes the wrapper walks
through [chunk resolution](api/chunk_resolution.md). Each box is read once with
basic slicing, and the selection is applied to the block in memory. `parts()`
exposes that walk, projected through the view:

```python
part = next(iter(lazy.lazy[0:35, 0:10].parts()))

part.base_coords    # (0, 0)                — which box of the base grid
part.box            # ((0, 30), (0, 40))    — that box in the array's own coordinates
part.view.shape    # (30, 10)              — a LazyArray for this box's cells
part.out_selection  # (slice(0, 30, None), slice(0, 10, None))
part.is_complete    # False                 — the view does not cover the whole box
```

Each `part.view` is an ordinary `LazyArray`, so parts can be resolved
independently and concurrently, and `result()` assembles that walk.
`part.view.bounding_box()` is part-local; `part.box` gives the box in the
wrapped array's coordinates, which is what distinguishes two parts.

The partitioning is discovered from the wrapped array — `read_chunk_sizes`, then
`chunks`. Those attribute names belong to the source; this API refers only to
parts. `with_parts` selects a different partitioning without touching the data or
the view:

```python
view = lazy.lazy[10:50, ::4]

[part.base_coords for part in view.with_parts((50, 20)).parts()]
# [(0, 0), (0, 1), (0, 2), (0, 3)]      — uniform boxes, tail clipped

[part.base_coords for part in view.with_parts_per_axis(((50, 50), (20, 20, 20, 20))).parts()]
# [(0, 0), (0, 1), (0, 2), (0, 3)]      — explicit per-axis sizes

[part.base_coords for part in view.unpartitioned().parts()]
# [(0, 0)]                              — one whole-array part; resolve in one shot
```

`result()` returns the same values under any of them: repartitioning changes how
the read is divided, not what it returns, and the result never shares memory
with the wrapped array. Parts that do not line up with the source's own boxes
(to bound peak memory, for example) are permitted; they cost extra I/O but do
not affect correctness.

### Boxes and queries

A selection is either **rectangular** — describable by an interval and a stride
per dimension — or a **query**, an explicit list of coordinates. `is_box`
reports which. `bounding_box()` gives the storage region touched in either case,
and `strides()` gives the step per dimension for a box:

```python
slab = lazy.lazy[10:50, ::4]
slab.is_box            # True
slab.bounding_box()    # ((10, 50), (0, 77))
slab.strides()         # (1, 4)
slab.shape             # (40, 20)

gather = lazy.lazy.oindex[[90, 3, 3], :]
gather.is_box          # False
gather.bounding_box()  # ((3, 91), (0, 80))  — 88 rows of hull over 3 selected
gather.strides()       # None
```

The bounding box is dense — every cell in it selected — only when every stride
is 1. The slab above spans a 40x77 hull over the 40x20 cells it selects, so
reading the hull and discarding the rest would transfer 3.85x the required data;
a consumer that wants one rectangular read has to apply `strides()` too.

The category is structural, not a heuristic: basic indexing always composes to a
box, and one `oindex`, `vindex`, or mask anywhere in the chain makes the
selection a query permanently. A *second* fancy step is limited — it may
re-index the axes the first one varies over, but not the axes it broadcasts
along, and raises `NotImplementedError` if it tries. The
[design notes](design-notes.md) list that limit with the others and explain why
the category matters to consumers of a selection.

## Checking your own array

If your array is one `LazyArray` will be asked to read, you can check that it
holds up under composed selections without writing the generators yourself.
`zarr_indexing.testing` ships the Hypothesis state machine this package tests
itself with: it composes indexing steps onto a view of your array — basic,
orthogonal, vectorized, each applied to the view the last one produced — and
after every step checks the view's shape, its `result()`, and the assembly of
its `parts()` against a NumPy array holding the same values.

```python
from zarr_indexing.testing import ChainedIndexingStateMachine, state_machine_test

class MyArrayIndexing(ChainedIndexingStateMachine):
    def make_source(self, data):
        array = my_format.create(shape=data.shape, dtype=data.dtype)
        array[:] = data
        return array

TestMyArrayIndexing = state_machine_test(MyArrayIndexing)
```

That is the whole subclass. `data`, `partitionings`, and `supports` are class
attributes if you want to widen or narrow what is drawn — `supports` in
particular names the [`IndexingSupport`](api/support.md) levels your source
really serves, and a chain read at one level must answer what the same chain
answers at every other. A failure shrinks to the shortest chain that causes it.

Install with `pip install zarr-indexing[testing]`; the rest of the package does
not depend on Hypothesis.

## Reference

- [The ndsel wire format](ndsel.md)
- [Design notes](design-notes.md)
- [API reference](api/index.md)
- [Changelog](https://github.com/zarr-developers/zarr-python/blob/main/packages/zarr-indexing/CHANGELOG.md)
- [License (MIT)](https://github.com/zarr-developers/zarr-python/blob/main/packages/zarr-indexing/LICENSE.txt)
