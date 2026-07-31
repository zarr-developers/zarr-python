# zarr-indexing

Composable, lazy coordinate transforms for Zarr array indexing.

`zarr-indexing` is developed in the
[zarr-python repository](https://github.com/zarr-developers/zarr-python/tree/main/packages/zarr-indexing)
and released independently of `zarr` itself. Install it with:

```
pip install zarr-indexing
```

## What this is

An indexing operation — a slice, an integer, a fancy index array — is a
*mapping* from the coordinates a user asks for to the coordinates that live in
storage. This library makes that mapping a first-class value: an
[`IndexTransform`](api/transform.md). Transforms compose, so a view of a view
of an array is still a single transform, and nothing is read until someone
asks for data.

Three pieces do the work:

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
  given a transform and a chunk grid, which chunks does this selection touch,
  which coordinates does it touch *inside* each chunk, and where do the values
  land in the output buffer? The resolver is dependency-aware: correlated
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
index-array flavour, and composition as the single operation that stacks
views. Names and semantics follow TensorStore where they overlap — notably,
negative indices are literal coordinates, not Python-style offsets from the
end, and it is the caller's job to normalize them.

The differences are the ones NumPy compatibility forces. `ArrayMap` records
the input dimension an *orthogonal* (`oindex`) index array varies over, which
TensorStore's format has no field for; the
[serializer collapses or reconstructs that field](api/json.md) so the wire
format stays TensorStore-loadable. Chunk resolution and the `oindex`/`vindex`
helpers exist to serve NumPy-shaped selection semantics, which TensorStore
does not have to model.

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

The domain describes what the *user* sees (here a single dimension, 40 long,
with origin 10); the output maps describe what *storage* sees (a stride-1
`DimensionMap` and the `ConstantMap` for the dropped dimension).

Fancy indexing works the same way, in both flavours, and still materializes
nothing but the index arrays themselves:

```python
import numpy as np

transform.oindex[np.array([3, 1, 90]), 0:4]   # '{ {3, 1, 90}, [0, 4) }', shape (3, 4)
transform.vindex[np.array([0, 40, 99]), np.array([1, 2, 3])]  # shape (3,)
```

Transforms built independently stack with
[`compose`](api/composition.md), which is what `transform[...]` uses
internally when you index an already-indexed view:

```python
from zarr_indexing import compose

inner = IndexTransform.from_shape((100,))[::2]  # storage 0, 2, 4, ... over domain [0, 50)
outer = IndexTransform.from_shape((50,))[10:20]

compose(outer, inner).selection_repr  # '{ [20, 40) step 2 }'
```

Resolution against a chunk grid is where a transform finally meets storage.
Chunk resolution asks the grid only for the per-dimension index-to-chunk
mapping described by [`DimensionGridLike`](api/grid.md), so any object with
those four methods will do:

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
and translated into chunk-local coordinates — exactly what a codec pipeline
needs to decode that chunk and scatter the result. `out_indices` carries the
output scatter indices for array selections, and is `None` for basic indexing.

## Reference

- [The ndsel wire format](ndsel.md)
- [API reference](api/index.md)
- [Changelog](https://github.com/zarr-developers/zarr-python/blob/main/packages/zarr-indexing/CHANGELOG.md)
- [License (MIT)](https://github.com/zarr-developers/zarr-python/blob/main/packages/zarr-indexing/LICENSE.txt)
