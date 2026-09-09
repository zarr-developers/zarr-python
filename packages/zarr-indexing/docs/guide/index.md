# Visual guide

The whole model in one sentence: indexing through `LazyArray.lazy` builds a
view, chunk planning partitions its coordinates, and `result()` materializes
the view. This page follows one familiar NumPy selection, `source[2:5]`,
through those stages.

The first four sections are for anyone indexing arrays: coordinates,
transforms, composition, and result axes. **If you are using lazy indexing
rather than building a storage backend, you can stop after section four.**
The last three sections are for integrators: they turn a request into a chunk
plan, pair each chunk read with its place in the result, and show the per-axis
tables the plan is built from.

Throughout, one division of labor holds: the transform answers **which
values?** and is independent of the backend; the reader answers **how do I
obtain them?** and must preserve the transform exactly.

## An index selects coordinates {#an-index-selects-coordinates}

Begin with an ordinary NumPy array. `source` contains the values 10 through 15.
The selection `source[2:5]` takes source coordinates 2, 3, and 4, containing
the values 12, 13, and 14.

```text
source coordinate |  0   1   2   3   4   5
source value      | 10  11  12  13  14  15
selection         |        [12  13  14]
                            source[2:5]

result coordinate |  0   1   2
source coordinate |  2   3   4
result value      | 12  13  14
```

The result defines its own coordinates: `0`, `1`, and `2`. The aligned rows
make the correspondence explicit: those result coordinates receive values
`12`, `13`, and `14` from source coordinates `2`, `3`, and `4`.

The wrapper below gives the same familiar selection a lazy spelling. Indexing
through `.lazy` creates `view`; the last line asks for its values and checks the
observable NumPy result.

```python
--8<-- "snippets/canonical_slice.py:canonical-slice"
```

The important first step is simply that an index describes which source values
fill a result in a particular order. The next section gives the numbers on both
sides of that description a precise meaning.

## Coordinates are addresses {#coordinates-are-addresses}

### How to read a half-open interval

`[0, 1)` is a **half-open interval**: start at 0, inclusive, and stop at 1, exclusive.
The `[` includes the lower boundary, while the `)` excludes the upper boundary.
For integer coordinates, `[0, 1)` therefore enumerates the ordered sequence
`[0]`.

Half-openness lets adjacent slices and chunks meet without a gap or overlap.
Concatenation is ordered: the first interval is followed by the second. When
the first interval's exclusive stop matches the second interval's inclusive
start, the shared boundary coordinate appears exactly once.

- `[0, 1) -> [0]` — Start at 0 and stop before 1, so the sequence contains only 0.
- `[1, 3) -> [1, 2]` — Start at 1 and stop before 3, so the sequence contains 1 and 2.
- `concat([0, 1), [1, 3)) = [0, 3)` — Append the second interval after the
  first. Their matching exclusive/inclusive boundary produces one continuous
  interval without a gap or duplicated coordinate.

Explicit coordinates, including coordinate arrays, are always ordered
sequences rather than mathematical sets. Their order is semantic, and repeated
coordinates remain repeated in the result.

In the transform algebra, **coordinates are just integers**. A negative
coordinate is a real address in a domain, with the same status as zero or a
positive coordinate; it is not automatically shorthand for counting backward
from an array's end.

```text
domain [-2, 3)

coordinate |   -2     -1      0      1      2
status     | address address address address address
```

`IndexDomain` makes those bounds explicit. In the example below, narrowing the
domain at `-1` selects the literal address `-1`; the wrapper at the end treats
`-1` the way NumPy does — as the last position.

```python
--8<-- "snippets/coordinate_origins.py:coordinate-origin"
```

Why carry literal coordinates at all? They let independently described
regions keep stable addresses — a domain can even grow at its lower end
without renumbering what is already there. The [design
notes](../design-notes.md#negative-origin-domains-and-prependable-grids) work
through that prepending example; nothing else in this guide depends on it.

The literal model and NumPy's positional model are both useful, but they answer
different questions:

| Surface | Meaning of an integer index | Meaning of `-1` |
| --- | --- | --- |
| `IndexDomain` and `IndexTransform` | A literal coordinate in the current domain | The actual address `-1`, if the domain contains it |
| `LazyArray.lazy` | A NumPy-style position in the current view | The last position, normalized before it reaches the transform algebra |

`LazyArray` uses positions because it is an array-like wrapper: each derived
view starts at position zero and negative indices wrap exactly as they do in
NumPy. The lower-level domain and transform types keep literal coordinates.

### A transform points from the request to the source

An `IndexTransform` records how every coordinate in a request finds its source
coordinate. For the slice from the first section, request coordinate `i` maps
to source coordinate `i + 2`. This direction is deliberate: request to source,
not source to request.

```text
request coordinate | 0   1   2
                   | |   |   |
             i + 2 | v   v   v
source coordinate  | 2   3   4
source value       | 12  13  14
```

A transform speaks function vocabulary while this guide speaks array
vocabulary. The two line up like this:

| the API says | this guide says |
| --- | --- |
| input space (`domain`, `input_rank`) | request coordinates — the result being built |
| output space (`output`, one map per dimension) | source coordinates — where values are read |

`output` names the output side of the coordinate *function*, not the data:
values flow source → request, against the arrow. The neutral names exist
because transforms compose — in a chain, an interior transform's output space
is just the next transform's input space, neither a request nor a source.

### The three map kinds, in NumPy terms

Every output dimension is produced by one of three map forms. Each has a
NumPy counterpart, shown executably below. The examples share one helper —
and it doubles as the answer to how a bare transform meets data at all: a
reader materializes it into a buffer.

```python
--8<-- "snippets/output_maps.py:resolve-helper"
```

`DimensionMap` is an arithmetic rule — the slice above is one, mapping
request `i` to source coordinate `i + 2`:

```python
--8<-- "snippets/output_maps.py:dimension-map"
```

`ArrayMap` stores explicit source coordinates for irregular or fancy
indexing; order and repeats survive into the result:

```python
--8<-- "snippets/output_maps.py:array-map"
```

`ConstantMap` fixes one source coordinate for every request cell. Whether an
axis appears in the result is decided by the **domain**, never by the map:
`image[2, :]` compiles to a `ConstantMap(2)` with no corresponding domain
axis (the axis is dropped), while pairing a constant map with a length-`n`
domain axis that no map consumes yields `n` cells all reading one
coordinate — a broadcast, the one arrangement with no NumPy index
counterpart:

```python
--8<-- "snippets/output_maps.py:constant-map"
```

Together, the request domain and these per-source-dimension maps are the
complete reusable description of an index.

## Lazy views compose {#lazy-views-compose}

A lazy view can be indexed again. Each step changes the request-to-source
description, but it does not read an intermediate array. The chain is reduced
to one direct transform from the newest request to the original source.

```text
source[2:5][::-1][1:]

new request | intermediate view | original source
------------+-------------------+----------------
     0      |         1         |       3
     1      |         2         |       2

direct map: request i -> source (3 - i)
```

The executable example first selects `source[2:5]`, then reverses that view
and trims its first element:

```python
--8<-- "snippets/lazy_composition.py:lazy-composition"
```

Immediately after `composed` is created—and before the final `result()` call—its
metadata is ready to inspect:

| Available without reading | Value in this example |
| --- | --- |
| `composed.shape` | `(2,)` |
| `composed.transform` | One transform mapping request `i` to source `3 - i` |

Neither property needs source values. Composition works only on the coordinate
description; the assertion's call to `result()` is the first operation in the
example that materializes the selected data.

!!! warning "Stop here: the materialization boundary"
    Indexing through `.lazy[...]` never reads. These do:

    - `result()`
    - eager indexing of the wrapper: `view[...]`
    - `numpy.asarray(view)`, or passing the view to any NumPy function
      (`numpy.add(view, 1)` converts, and therefore materializes, the view)

    Python arithmetic such as `view + 1` raises `TypeError` instead: this
    wrapper defers indexing, not a general compute graph.

    Nor does it write. There is no `__setitem__`, so `view[...] = values`
    raises `TypeError` too, and a wrapped source needs no `__setitem__` of
    its own. A consumer that writes plans the selection with `plan_chunks`
    and performs its own read-modify-write, keeping chunk atomicity and
    concurrent-writer policy on the backend's side of the boundary.

## An index defines a result array {#an-index-defines-a-result-array}

An index chooses source points and also defines how those points are arranged in
the result. In the 3-by-4 image below, `image[1, :]` and `image[1:2, :]` choose
the same four source points: values `4`, `5`, `6`, and `7`.

```text
same selected source cells

source coordinate | (1, 0)  (1, 1)  (1, 2)  (1, 3)
value             |    4       5       6       7

image[1, :]

result coordinate |    0       1       2       3
value             |    4       5       6       7
shape             | (4,); source axis 0 is omitted

image[1:2, :]

result coordinate | (0, 0)  (0, 1)  (0, 2)  (0, 3)
value             |    4       5       6       7
shape             | (1, 4); source axis 0 is retained with length 1
```

The integer in `image[1, :]` fixes source axis 0. No result coordinate varies
along that axis, so it is omitted and the result shape is `(4,)`. The slice in
`image[1:2, :]` preserves source axis 0 as a length-one result axis, so the
result shape is `(1, 4)`.

```python
--8<-- "snippets/axis_manipulation.py:axis-shape-comparison"
```

`None` inserts a new length-one axis without selecting different source points.
Here it produces the shape `(4, 1)`:

```python
--8<-- "snippets/axis_manipulation.py:axis-insertion"
```

## A request becomes a chunk plan {#a-request-becomes-a-chunk-plan}

Continue with the 3-by-4 image and `image[1, :]` introduced above. Giving the
image a 2-by-2 chunk shape does not change the four selected values or their
order. It changes only how the work is divided: columns 0 and 1 come from chunk
`(0, 0)`, while columns 2 and 3 come from chunk `(0, 1)`.

```text
                    column
                0    1  |  2    3
              ----------+----------
row 0            0    1 |   2    3
row 1           [4]  [5]|  [6]  [7]   <- image[1, :]
              ----------+----------
row 2            8    9 |  10   11

                       left part             right part
chunk_coords           (0, 0)                (0, 1)
global chunk_domain    [0,2) x [0,2)         [0,2) x [2,4)
selected global cells  (1,0), (1,1)          (1,2), (1,3)
chunk-local cells      (1,0), (1,1)          (1,0), (1,1)
request coordinates    0, 1                  2, 3
```

Every planned chunk keeps three coordinate frames distinct:

- `chunk_coords` identifies a cell in the chunk grid. Chunk coordinates are
  literal integers, so a grid that grows at its lower end can hold a chunk
  whose coordinate really is `-1` — not an alias for the final chunk (see the
  [design notes](../design-notes.md#negative-origin-domains-and-prependable-grids)).
- `chunk_domain` gives that chunk's bounds in **global source coordinates**.
  Here the two domains are `[0, 2) × [0, 2)` and `[0, 2) × [2, 4)`.
- Chunk-local positions start from zero inside each chunk. Global column 2 is
  therefore local column 0 in chunk `(0, 1)`. This zero-origin local frame is
  separate from both the global `chunk_domain` and the possibly negative
  chunk coordinate.

`plan_chunks` needs only a transform and the chunk layout: one grid object
per source dimension. A per-dimension grid answers four questions — which
chunk contains a source index, where a chunk starts, how long it is, and
the vectorized form of the first (`index_to_chunk`, `chunk_offset`,
`chunk_size`, `indices_to_chunks`). The library builds these from chunk
sizes via `dimension_grids_from_chunks`; the executable example hand-rolls
one instead, to show that the whole contract is those four answers. It
plans the canonical request over 2-by-2 chunks, and iterates the same plan
again to show that planning is reusable. (The two transforms it inspects on
each projection are the next section's subject.)

```python
--8<-- "snippets/chunk_projection.py:chunk-projection"
```

The plan describes work but does not perform it. It contains no array source,
storage backend, codec pipeline, buffer, or scheduler. A Zarr reader, a task
queue, or a viewport can consume the same logical plan and decide independently
how and when to fetch its two chunks.

On the wrapper, this partitioning is called **parts**: `with_parts(shape)`
gives a `LazyArray` a grid of uniform boxes to divide its reads along
(re-partitioning is a pure setter — it changes how a read is divided, never
what `result()` returns), and a wrapped array advertising its own `chunks`
is partitioned that way automatically.

A zero-length source axis has no chunks. `LazyArray` accepts a positive uniform
part shape for that axis, or explicit per-axis spellings `()`, `(0,)`, and
`(0, 0)`; each produces no parts and the same empty result. Zero-sized parts
remain invalid on a nonempty axis.

## One cell domain, two projections {#one-cell-domain-two-projections}

A chunk read has to answer two questions at once: which cells belong to this
chunk, and where does each of those cells belong in the requested result?

Think of a projection as a small table with one row per selected cell. For
each row, `chunk_transform` gives the cell's zero-origin address inside the
chunk, and `cell_transform` gives the position in the requested result that
receives its value. The row numbers of that table are the shared **cell
domain** — a synthetic input space both transforms accept, which is why one
input point can be evaluated on both sides.

```text
left chunk (0, 0)

shared cell coordinate |    0       1
cell_transform         |    v       v
request coordinate     |    0       1

shared cell coordinate |    0       1
chunk_transform        |    v       v
chunk-local coordinate | (1, 0)  (1, 1)

right chunk (0, 1)

shared cell coordinate |    0       1
request coordinate     |    2       3
chunk-local coordinate | (1, 0)  (1, 1)
```

The directions are exact: **shared synthetic input cell domain → request via
`cell_transform`**, and **shared cell domain → chunk-local via
`chunk_transform`**. Neither arrow starts at the request or maps one output
space into the other.

On the wrapper, `view.parts()` returns one `Partition` per planned chunk;
each bundles a sub-view of the request (`.view`), that chunk's projection
(`.projection`), and the NumPy selection placing its values in the result
(`.out_selection`).

Within one `Partition`, the frames divide: `Partition.view.transform` is a
different, global transform — it maps the part view directly into the raw
wrapped source — while only `Partition.projection.chunk_transform` uses
zero-origin chunk-local coordinates. Readers receive both so the global
source address and the local planning frame cannot be confused.

| Projection field | What its output coordinates mean |
| --- | --- |
| `cell_transform` | Literal coordinates in the original request; its output rank is the request rank |
| `chunk_transform` | Zero-origin coordinates in the selected chunk's local frame; its output rank is the source rank |

The cell domain enumerates corresponding cells; it is not itself either
output coordinate space. The canonical row selection has a one-dimensional
request and a two-dimensional source, so its paired projections have request
rank one and source rank two.

### Order and duplicates need the request-side projection

Orthogonal indexing (`.lazy.oindex`) applies each axis's indexer
independently, like `numpy.ix_` — an outer product; the
[pattern reference](patterns.md) develops the dialects. It can visit source
cells in an order that does not match chunk order, and it can visit one
source cell more than once. In the request below, row 4 comes first and
row 1 appears twice.

```text
request position |   0      1      2
source row       |   4      1      1
result row       | row 4  row 1  row 1
```

A source bounding box cannot reconstruct this result. The box spanning rows 1
through 4 also includes unrequested rows 2 and 3, and its increasing coordinate
order does not record that row 4 comes first. Narrowing the read to just rows 1
and 4 still does not record the second use of row 1. For the same reason, a
chunk-local selector alone says which cells to read inside a chunk but cannot
say which request positions receive them, especially when the chunks are
processed in a different order.

The executable example assembles the 3-by-4 request from a 6-by-8 source with
3-by-4 chunks. Each `Partition` resolves its own sub-view — the global
transform addressing the raw source — and `out_selection` places those
values at their request-side positions; the paired projection stays
available on `part.projection` for consumers that read chunks directly.
The assertion checks the reordered, duplicated result against direct NumPy
indexing.

```python
--8<-- "snippets/chunk_projection.py:advanced-projection"
```

The paired representation preserves information that a bounding box or local
selector discards: exact request order, duplicate destinations, and the
correspondence between every request position and its chunk-local source cell.

## A plan is a product of per-axis tables {#a-plan-is-a-product-of-per-axis-tables}

Look again at the two projections of `image[1, :]`. Each chunk transform has
one map per source axis, and every one of those maps came from restricting
the request's map for *that axis alone* to *that axis's* chunk: the fixed row
`ConstantMap(1)` lands in row-chunk 0 whatever the column chunk is, and the
column slice meets column-chunk 0 as local columns `0:2` and column-chunk 1 as
local columns `0:2` whatever the row chunk is. Restricting a transform to a
chunk distributes over axes whenever each output map reads its own request
axis — which every basic and orthogonal selection satisfies. So the plan does
not intersect the whole transform with every chunk. It resolves each axis
once, into a table with one row per chunk that axis touches, and a projection
is one row of each table combined.

```text
image[1, :] over 2-by-2 chunks

axis 0 (rows): ConstantMap(1)          axis 1 (columns): DimensionMap
row | chunk  start  local  extent      row | chunk  start  local_start  extent  origin  full
 0  |   0      0      1      1          0  |   0      0        0          2       0     yes
                                        1  |   1      2        0          2       2     yes

row_shape (1, 2): 1 x 2 = 2 projections
projection (0, 0) = axis-0 row 0 x axis-1 row 0 -> chunk (0, 0), local (1, 0:2), request columns 0:2
projection (0, 1) = axis-0 row 0 x axis-1 row 1 -> chunk (0, 1), local (1, 0:2), request columns 2:4
```

`ChunkPlan.partition()` returns this factored form, a `GridPartition`. Its
`sets` hold one table per source axis the request reads independently, in
axis order, and `joint_sets` one table per connected group of index arrays; `row_shape` is the
number of rows in each; and the plan walks the rows in row-major order over
it. The executable example reads the two tables above off the plan, checks
that the plan's projections are exactly the partition's rows, and evaluates
the second row on both of its transforms:

```python
--8<-- "snippets/grid_partition.py:strided-tables"
```

There are three kinds of table, matching the three map kinds and the one
arrangement that does not factor. A `StridedSet` holds a `ConstantMap` or
`DimensionMap` axis; an `IndexedSet` holds an orthogonal `ArrayMap` axis
(`.oindex`) with its coordinates grouped by chunk; and each `JointSet` holds one connected group of index arrays. Arrays
sharing request axes are sorted together; independent groups are separate
tables in `joint_sets`. Singleton broadcast axes retain this independence. The
[API reference](../api/chunk_resolution.md) documents every column.

The gather from the previous section, `oindex[[4, 1, 1], 2:6]` over 3-by-4
chunks, groups rows `1, 1` into chunk 0 and row `4` into chunk 1 while
remembering that row `4` fills request position 0. A `vindex` selection keeps
its points paired in the joint table:

```python
--8<-- "snippets/grid_partition.py:indexed-and-joint"
```

Building independent strided tables costs the *sum* of touched chunks per
axis. Index-array grouping also scales with the selected point count; the
planner flattens each connected component separately. Independent components
do not expand into their Cartesian product until their rows are iterated. And projections are derived from rows only when asked for:
`chunk_coords()` lists every chunk the plan touches without materializing a
row, and a consumer can read the columns directly, as
[Integration boundaries](integrations.md#reading-the-tables-directly) shows.
For example, `(u, v) -> (a[u], b[u], c[v])` has two independent components.
The first two storage axes share `u`; the third reads `v`. With 1,000 values
per input axis, planning stores 3,000 index values rather than 3,000,000.
Each projection has one synthetic axis per nonconstant component, followed
by residual request axes, including axes no output reads.

```python
--8<-- "snippets/grid_partition.py:components"
```

A hand-built affine diagonal — two `DimensionMap`s reading one request axis,
which no selection produces — would need a table spanning several storage
axes; the plan rejects it with `ValueError`.

---

<nav aria-label="Guide navigation" markdown>
**Previous:** [zarr-indexing](../index.md)
·
**Next:** [Indexing patterns](patterns.md)
</nav>
