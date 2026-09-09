# From a selection to chunk operations

A selection describes values in an array. A chunk operation identifies a chunk,
which values to select inside its decoded buffer, and where those values belong
in the result. This page follows that conversion through the current planner and
explains where the proposed sharding integration would fit.

The public planning API is `plan_chunks(transform, grids)`. The plan describes
coordinate mappings; a consumer decides how to turn them into selectors for its
buffer implementation. Zarr's default indexers have not been replaced.

## The flow

```text
selection + source domain + indexing dialect
                    |
          normalize / compose
                    |
       request-to-storage mapping
                    |  + caller-selected chunk grid
                    v
       factor and partition the request
                    |
          +---------+-----------------+
          |                           |
  declarative projections       execution lowering
  chunk coordinates             chunk coordinates
  chunk_transform               chunk_selection
  cell_transform                out_selection
  coverage                      selected-value shape / coverage flag
          |                           |
          +------------+--------------+
                       |
             consumer fetches, decodes,
             gathers/scatters, encodes
```

The right-hand branch describes consumer responsibilities, not a second public
planning API. A consumer can lower table rows directly without first allocating
a transform pair for every chunk.

## 1. Interpret the selection

The indexing boundary chooses the dialect before chunk planning begins:

- A positional boundary, such as `LazyArray`, interprets NumPy-style negative
  indices and slice clipping against the current view.
- `IndexTransform` uses literal
  coordinates. Bounds are addresses; they do not wrap or clip like NumPy bounds.
- Basic, orthogonal, and vectorized indexing specify different mappings.
  Orthogonal arrays form an outer product; vectorized arrays select paired,
  broadcast coordinates.

Normalization resolves omitted dimensions and slice bounds and checks the
selection. Composing another selection onto a view produces a mapping directly
to the original source, rather than reading the intermediate view first. See
[Indexing patterns](patterns.md) for the dialect examples.

An `IndexTransform` represents this mapping with a request domain and one output
map per storage dimension: a constant, an affine function of a request axis, or
an index-array lookup. An integer selection can therefore remove a result axis
while retaining a constant coordinate for that storage dimension.

## 2. Keep the coordinate spaces distinct

| Space | Meaning | Example for `a[1:7:2, 2:6]` |
| --- | --- | --- |
| Global storage | Addresses in the source array | Rows `1, 3, 5`; columns `2, 3, 4, 5` |
| Chunk grid | Which grid cell contains an address | With `(4, 4)` chunks, address `(5, 4)` belongs to chunk `(1, 1)` |
| Chunk local | Address relative to that chunk's origin | `(5, 4) - (4, 4) = (1, 0)` |
| Request domain | Coordinates accepted by the request transform | May have a nonzero origin under literal slice semantics |
| Result buffer | Zero-based positions in the materialized result | Source `(5, 4)` goes to result `(2, 2)` |

For each storage dimension, the grid supplies `index_to_chunk`, `chunk_offset`,
and `chunk_size`. Uniform grids can use integer division; irregular grids use
their actual boundaries. The planner consumes this grid protocol rather than
assuming that all chunks have the same size.

A declarative `cell_transform` returns **request-domain coordinates**, not
necessarily zero-based buffer positions. A consumer placing values into a NumPy
result subtracts the request domain origin. An execution consumer must translate these into zero-based buffer positions.

## 3. Partition without expanding the entire request

Planning groups selected values by the chunks that contain them, retaining the
mapping back to their original request positions.

For affine selections, the planner finds each run that falls
inside one chunk. A run records the chunk coordinate, chunk origin, local start,
number of selected values, and starting request position. For a rectangular
request, the multidimensional chunk operations are products of these axis runs.
`StridedSet` tables materialize these per-axis rows, without materializing
their full Cartesian product.

For index arrays, planning groups coordinates by chunk while retaining their
request positions. Arrays that share request axes must be grouped together:
those coordinates are correlated. Independent connected components stay
separate until chunk operations are visited. This avoids expanding an outer
product into one coordinate tuple per requested value during preparation.

`ChunkPlan.partition()` memoizes the factored `GridPartition`. Its tables include
strided sets, indexed sets, and joint sets for connected index-array components.
Plan iteration builds paired projections from those tables. Affine diagonals (two slice maps reading one request axis) are rejected with `ValueError`; see [the current limitations](../design-notes.md#current-scope).

An empty request produces no chunk operations. Repeated coordinates remain
repeated request positions for reads: the selection is an ordered mapping, not
a mathematical set of source points.

## 4. Describe each chunk's contribution

The declarative representation is a `ChunkProjection`:

- `chunk_coords` identifies the grid cell, and `chunk_domain` gives its global
  storage bounds.
- `chunk_transform` maps a shared synthetic domain to chunk-local coordinates.
- `cell_transform` maps the same synthetic domain to request-domain coordinates.

The shared domain pairs each source value with its destination. Its coordinates
are neither automatically chunk-local nor automatically result positions.
For every point `p` in that domain, the defining relationship is:

```text
request_transform(cell_transform(p))
    == chunk_origin + chunk_transform(p)
```

A consumer can evaluate these transforms, consume the factored tables directly,
or lower them to its own selection vocabulary.

For a NumPy reader, the eventual operation has this meaning:

```text
result[result_selection] = decoded_chunk[chunk_selection]
```

The chunk coordinates identify which decoded chunk to obtain. For writes,
values flow in the opposite direction. Basic selectors use slice/integer
semantics; coordinate selectors are paired arrays that broadcast together.
The consumer must preserve both the selected-value shape and the coordinate
correspondence when choosing between these representations.

### A strided rectangle crossing four chunks

For an `(8, 10)` source with `(4, 4)` chunks, `a[1:7:2, 2:6]` has result shape
`(3, 4)`. The following slices describe the four contributions; slice stops are
shown clipped to each chunk's extent, so equivalent generated stops may differ.

| Chunk coordinates | Chunk origin | Chunk-local selection | Result selection |
| --- | --- | --- | --- |
| `(0, 0)` | `(0, 0)` | `[1:4:2, 2:4]` | `[0:2, 0:2]` |
| `(0, 1)` | `(0, 4)` | `[1:4:2, 0:2]` | `[0:2, 2:4]` |
| `(1, 0)` | `(4, 0)` | `[1:2:2, 2:4]` | `[2:3, 0:2]` |
| `(1, 1)` | `(4, 4)` | `[1:2:2, 0:2]` | `[2:3, 2:4]` |

This executable example checks both the touched chunks and the assembled values.
The example enumerates the small synthetic domains to make both projection
directions explicit. Production consumers can instead lower the factored tables
to slices and coordinate arrays. Array slicing stands in for storage and codecs:

```python
--8<-- "snippets/selection_flow.py:basic"
```

### A gather with repeated, out-of-order coordinates

For `a[[6, 1, 6, 4]]` and chunk size `4`, grouping by chunk must preserve these
pairs:

| Chunk | Chunk-local coordinates | Result positions |
| --- | --- | --- |
| `(0,)` | `[1]` | `[1]` |
| `(1,)` | `[2, 2, 0]` | `[0, 2, 3]` |

Chunk `(1,)` is fetched once, but its local value `2` contributes to two result
positions. Chunk visitation order need not be result order; `out_selection`
restores the requested arrangement.

```python
--8<-- "snippets/selection_flow.py:gather"
```

## 5. Coverage and execution

`StridedSet.full` and `ChunkProjection.coverage` describe coverage of the valid
**data extent** of a grid cell, including a clipped boundary cell. A singleton
cell selected once is full even when the request stride has magnitude greater
than one. Repeated coverage is not full; fancy coverage remains conservative.

Zarr's merge operation can skip a read for a full data-extent write and allocate
a fill-valued codec buffer when the selected data is smaller than that buffer.
Touching every chunk alone does not prove full coverage of each chunk.

Planning does not fetch bytes, decode buffers, choose concurrency, or define
write-conflict handling. Consumers must preserve request ordering where their
write API requires it, and account for memory retained by emitted operations.

## 6. Applying the flow to sharding

A sharded array has an outer shard grid and an inner codec-chunk grid. A consumer
can partition the request first by outer shard, then map each contribution into
the shard's coordinate frame and partition against the inner grid:

```text
request -> global storage
        -> outer shard + request-to-shard-local mapping
        -> inner chunk + request-to-inner-local mapping
        -> inner selection + destination in the result
```

This is a proposed integration, not implemented by this PR. The existing
sharding codec receives selectors and calls its own indexer against the inner
grid. A future consumer could retain compact transforms through both levels,
collect touched chunk coordinates for byte-range retrieval, and lower value
selectors only when needed. Direct scattering into the final result would also
require a pipeline interface carrying that buffer and its destination mapping.
No on-disk format change is needed. Fill values, missing chunks, partial-write
coverage, and the consumer's duplicate-write policy must survive both levels.

See [Integration boundaries](integrations.md) for consumers of projections and
tables, and [Design notes](../design-notes.md) for implementation constraints.
