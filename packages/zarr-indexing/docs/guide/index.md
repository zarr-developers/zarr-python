# Visual guide

Start with a familiar NumPy selection and follow it through the five ideas that
make indexing lazy and partitionable: coordinates, transforms, composition,
chunk planning, and paired projections. The opening slice keeps the coordinate
story one-dimensional before later sections introduce higher-dimensional chunk
planning.

1. [An index selects coordinates](#an-index-selects-coordinates) begins with
   the values and order of `source[2:5]`.
2. [Coordinates are addresses](#coordinates-are-addresses) explains literal
   coordinate domains, NumPy positions, and the request-to-source mapping
   between them.
3. [Lazy views compose](#lazy-views-compose) shows how repeated indexing
   becomes one description and identifies exactly when data is read.
4. [A request becomes a chunk plan](#a-request-becomes-a-chunk-plan) divides
   a higher-dimensional request over a 2-by-2 chunk grid without choosing a source or
   scheduler.
5. [One cell domain, two projections](#one-cell-domain-two-projections) pairs
   chunk-local reads with their exact positions in the requested result.

You can finish the tour with a practical mental model: indexing through
`LazyArray.lazy` builds a view, chunk planning partitions its coordinates, and
`result()` materializes that view.

## An index selects coordinates {#an-index-selects-coordinates}

Begin with an ordinary NumPy array. `source` contains the values 10 through 15.
The selection `source[2:5]` takes source coordinates 2, 3, and 4, containing
the values 12, 13, and 14.

<figure>
<div class="zi-figure-scroll" role="region" aria-label="Scrollable basic indexing selection diagram" tabindex="0">
--8<-- "_static/diagrams/indexing-selection.svg"
</div>
<figcaption>Result coordinates <code>0</code>, <code>1</code>, and <code>2</code> receive source values at coordinates <code>2</code>, <code>3</code>, and <code>4</code>.</figcaption>
</figure>

The result defines its own coordinates: `0`, `1`, and `2`. The aligned rows
make the correspondence explicit: those result coordinates receive values
`12`, `13`, and `14` from source coordinates `2`, `3`, and `4`. This is logical
placement, not physical read order.

The wrapper below gives the same familiar selection a lazy spelling. Indexing
through `.lazy` creates `view`; the last line asks for its values and checks the
observable NumPy result.

```python
--8<-- "examples/canonical_slice.py:canonical-slice"
```

The important first step is simply that an index describes which source values
fill a result in a particular order. The next section gives the numbers on both
sides of that description a precise meaning.

## Coordinates are addresses {#coordinates-are-addresses}

### How to read a half-open interval

`[0, 1)` is a **half-open interval**: start at 0, inclusive, and stop at 1, exclusive.
The `[` includes the lower boundary, while the `)` excludes the upper boundary.
For integer coordinates, `[0, 1)` therefore contains exactly one coordinate: `0`.

Half-openness lets adjacent slices and chunks meet without a gap or overlap.
Concatenation is ordered: the first interval is followed by the second. When
the first interval's exclusive stop matches the second interval's inclusive
start, the shared boundary coordinate appears exactly once.

- `[0, 1) = {0}` — Start at 0 and stop before 1, so the interval contains only 0.
- `[1, 3) = {1, 2}` — Start at 1 and stop before 3, so the interval contains 1 and 2.
- `concat([0, 1), [1, 3)) = [0, 3)` — Append the second interval after the
  first. Their matching exclusive/inclusive boundary produces one continuous
  interval without a gap or duplicated coordinate.

In the transform algebra, **coordinates are just integers**. A negative
coordinate is a real address in a domain, with the same status as zero or a
positive coordinate; it is not automatically shorthand for counting backward
from an array's end.

<figure>
<div class="zi-figure-scroll" role="region" aria-label="Scrollable coordinate addresses diagram" tabindex="0">
--8<-- "_static/diagrams/coordinate-addresses.svg"
</div>
<figcaption>The domain <code>[-2, 3)</code> contains five peer addresses, including the real negative addresses <code>-2</code> and <code>-1</code>.</figcaption>
</figure>

`IndexDomain` makes those bounds explicit. In the first half of this executable
example, narrowing at `-1` means selecting the literal address `-1`. The final
assertion deliberately contrasts that with a NumPy-shaped wrapper.

```python
--8<-- "examples/coordinate_origins.py:coordinate-origin"
```

### Prepending does not renumber

Literal coordinates let a domain grow at its lower end without changing the
identity of anything already present. Prepending three cells extends `[0, 6)`
to `[-3, 6)`: the new cells receive addresses `-3`, `-2`, and `-1`, while the
old cells keep addresses `0` through `5`. Coordinate `0` does not become
coordinate `3`.

The adjacent intervals `[-3, 0)`, `[0, 3)`, and `[3, 6)` follow the same
half-open adjacency rule: each stopping boundary is included exactly once as
the next interval's starting boundary.

<figure>
<div class="zi-figure-scroll" role="region" aria-label="Scrollable prepended chunk coordinates diagram" tabindex="0">
--8<-- "_static/diagrams/prepend-chunk.svg"
</div>
<figcaption>Prepending extends <code>[0, 6)</code> to <code>[-3, 6)</code> without renumbering any existing coordinate.</figcaption>
</figure>

That literal model and NumPy's positional model are both useful, but they answer
different questions:

| Surface | Meaning of an integer index | Meaning of `-1` |
| --- | --- | --- |
| `IndexDomain` and `IndexTransform` | A literal coordinate in the current domain | The actual address `-1`, if the domain contains it |
| `LazyArray.lazy` | A NumPy-style position in the current view | The last position, normalized before it reaches the transform algebra |

`LazyArray` uses positions because it is an array-like wrapper: each derived
view starts at position zero and negative indices wrap exactly as they do in
NumPy. The lower-level domain and transform types keep literal coordinates so
independently described regions can retain stable addresses.

The same distinction matters for chunk grids. `EdgeDimensionGrid` is the
convenient concrete grid for a zero-origin array: its chunk offsets are prefix
sums starting at zero. `DimensionGridLike` is the more general protocol
consumed by chunk planning, so it admits grids with negative chunk and cell
coordinates, including the prependable example below.

```python
--8<-- "examples/coordinate_origins.py:prepend-grid"
```

Here the literal cell domain `[-3, 0)` belongs to chunk `-1`. Both public
projection transforms share the same synthetic input cell domain `[0, 3)`.
Evaluating its three points shows the two distinct outputs:
`chunk_transform` produces zero-origin chunk-local coordinates `0, 1, 2`,
while `cell_transform` produces the literal request coordinates `-3, -2, -1`.
The shared input domain is not itself the chunk-local coordinate frame.

### A transform points from the request to the source

An `IndexTransform` records how every coordinate in a request finds its source
coordinate. For the slice from the first section, request coordinate `i` maps
to source coordinate `i + 2`. This direction is deliberate: request to source,
not source to request.

<figure>
<div class="zi-figure-scroll" role="region" aria-label="Scrollable transform mapping diagram" tabindex="0">
--8<-- "_static/diagrams/transform-mapping.svg"
</div>
<figcaption>The transform sends each request coordinate <code>i</code> to source address <code>i + 2</code>.</figcaption>
</figure>

This slice needs one `DimensionMap`, which describes how its one request
coordinate becomes the source coordinate `i + 2`. The transform system also
has three map forms:

- `ConstantMap` is a map whose source coordinate does not vary; the later
  result-array section returns to that form.
- `DimensionMap` describes an arithmetic rule, such as request `i` becoming
  source coordinate `i + 2`.
- `ArrayMap` stores explicit coordinates for irregular or fancy indexing.

Together, the request domain and these per-source-dimension maps are the
complete reusable description of an index.

## Lazy views compose {#lazy-views-compose}

A lazy view can be indexed again. Each step changes the request-to-source
description, but it does not read an intermediate array. The chain is reduced
to one direct transform from the newest request to the original source.

<figure>
<div class="zi-figure-scroll" role="region" aria-label="Scrollable composed views diagram" tabindex="0">
--8<-- "_static/diagrams/compose-views.svg"
</div>
<figcaption>The two view steps collapse into one direct request-to-source map, so composition adds no intermediate read.</figcaption>
</figure>

The executable example first selects `source[2:5]`, then reverses that view
and trims its first element:

```python
--8<-- "examples/lazy_composition.py:lazy-composition"
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
    Indexing through `.lazy[...]` is lazy. Calling `result()`, indexing the
    wrapper eagerly with `view[...]`, converting it with `numpy.asarray`, or
    passing it to a NumPy function materializes data. Python arithmetic such as
    `view + 1` is outside this wrapper's deferred scope and raises `TypeError`;
    NumPy arithmetic such as `numpy.add(view, 1)` converts and materializes the
    view. This wrapper defers indexing, not a general compute graph.

## A request becomes a chunk plan {#a-request-becomes-a-chunk-plan}

Now move to a 3-by-4 source and its selection `image[1, 0:4]`. Giving the
image a 2-by-2 chunk shape does not change the four selected values or their
order. It changes only how the work is divided: columns 0 and 1 come from chunk
`(0, 0)`, while columns 2 and 3 come from chunk `(0, 1)`.

<figure>
<div class="zi-figure-scroll" role="region" aria-label="Scrollable chunk projection overlay diagram" tabindex="0">
--8<-- "_static/diagrams/chunk-overlay.svg"
</div>
<figcaption>The unchanged selection crosses one 2-column chunk boundary, so it touches exactly chunks <code>(0, 0)</code> and <code>(0, 1)</code> of the 3-by-4 source.</figcaption>
</figure>

Every planned chunk keeps three coordinate frames distinct:

- `chunk_coords` identifies a cell in the chunk grid. Chunk coordinates are
  integers, so the prepended chunk from the coordinates section really has
  coordinate `-1`; it is not an alias for the final chunk.
- `chunk_domain` gives that chunk's bounds in **global source coordinates**.
  Here the two domains are `[0, 2) × [0, 2)` and
  `[0, 2) × [2, 4)`. For the prepended one-dimensional example, chunk `-1`
  has the global domain `[-3, 0)`.
- Chunk-local positions start from zero inside each chunk. Global column 2 is
  therefore local column 0 in chunk `(0, 1)`. This zero-origin local frame is
  separate from both the global `chunk_domain` and the possibly negative
  chunk coordinate.

`plan_chunks` needs only a transform and a grid. The executable example builds
the canonical 2-by-2 chunk grid, plans the request, and iterates the same plan again
to show that planning is reusable.

```python
--8<-- "examples/chunk_projection.py:chunk-projection"
```

The plan describes work but does not perform it. It contains no array source,
storage backend, codec pipeline, buffer, or scheduler. A Zarr reader, a task
queue, or a viewport can consume the same logical plan and decide independently
how and when to fetch its two chunks.

## One cell domain, two projections {#one-cell-domain-two-projections}

A chunk read has to answer two questions at once: which cells belong to this
chunk, and where does each of those cells belong in the requested result? A
paired projection answers both from one shared cell domain.

<figure>
<div class="zi-figure-scroll" role="region" aria-label="Scrollable paired chunk projections diagram" tabindex="0">
--8<-- "_static/diagrams/projection-pair.svg"
</div>
<figcaption>Each point in the shared cell domain identifies the same logical cell on both arrows, once in request space and once in zero-origin chunk-local space.</figcaption>
</figure>

The directions are exact: **shared synthetic input cell domain → request via
`cell_transform`**, and **shared cell domain → chunk-local via
`chunk_transform`**. Neither arrow starts at the request or maps one output
space into the other.

| Projection field | What its output coordinates mean |
| --- | --- |
| `cell_transform` | Literal coordinates in the original request; its output rank is the request rank |
| `chunk_transform` | Zero-origin coordinates in the selected chunk's local frame; its output rank is the source rank |

Both transforms have exactly the same synthetic input cell domain. That domain
enumerates corresponding cells; it is not either output coordinate space. A
consumer can therefore evaluate one input point on both sides: the
`chunk_transform` result says which zero-origin chunk-local coordinate to read,
and the `cell_transform` result says which literal request coordinate receives
that value. The canonical row selection has a one-dimensional request and a
two-dimensional source, so its paired projections have request rank one and
source rank two.

### Order and duplicates need the request-side projection

Orthogonal indexing can visit source cells in an order that does not match
chunk order, and it can visit one source cell more than once. In the request
below, row 4 comes first and row 1 appears twice.

<figure>
<div class="zi-figure-scroll" role="region" aria-label="Scrollable orthogonal indexing contrast diagram" tabindex="0">
--8<-- "_static/diagrams/orthogonal-contrast.svg"
</div>
<figcaption>The request <code>[4, 1, 1]</code> produces row 4 first, then two independently placed copies of row 1.</figcaption>
</figure>

A source bounding box cannot reconstruct this result. The box spanning rows 1
through 4 also includes unrequested rows 2 and 3, and its increasing coordinate
order does not record that row 4 comes first. Narrowing the read to just rows 1
and 4 still does not record the second use of row 1. For the same reason, a
chunk-local selector alone says which cells to read inside a chunk but cannot
say which request positions receive them, especially when the chunks are
processed in a different order.

The executable example assembles the 3-by-4 request from a 6-by-8 source with
3-by-4 chunks. Each part reads through the chunk-local side; `out_selection` is
the wrapper's NumPy lowering of the request-side placement. The assertion
checks the reordered, duplicated result against direct NumPy indexing.

```python
--8<-- "examples/chunk_projection.py:advanced-projection"
```

The paired representation preserves information that a bounding box or local
selector discards: exact request order, duplicate destinations, and the
correspondence between every request position and its chunk-local source cell.

---

<nav aria-label="Guide navigation">
  <strong>Previous:</strong> <a href="../">zarr-indexing</a>
  ·
  <strong>Next:</strong> <a href="../api/">API reference</a>
</nav>
