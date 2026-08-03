# Coordinates are addresses

## How to read a half-open interval

`[0, 1)` is a **half-open interval**: start at 0, inclusive, and stop at 1, exclusive.
The `[` includes the lower boundary, while the `)` excludes the upper boundary.
For integer coordinates, `[0, 1)` therefore contains exactly one coordinate: `0`.

Half-openness lets adjacent slices and chunks meet without a gap or overlap.
The stopping boundary of one interval is the starting boundary of the next:
`[0, 1) ∪ [1, 3) = [0, 3)`. Coordinate `1` appears exactly once—in the
second interval.

```text
[0, 1) = {0}
[1, 3) = {1, 2}
[0, 1) ∪ [1, 3) = [0, 3) = {0, 1, 2}
```

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

## Prepending does not renumber

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

## A transform points from the request to the source

An `IndexTransform` records how every coordinate in a request finds its source
coordinate. For the slice from chapter 1, request coordinate `i` maps to source
coordinate `(1, i)`. This direction is deliberate: request to source, not
source to request.

<figure>
<div class="zi-figure-scroll" role="region" aria-label="Scrollable transform mapping diagram" tabindex="0">
--8<-- "_static/diagrams/transform-mapping.svg"
</div>
<figcaption>The transform sends each request coordinate <code>i</code> to its source address <code>(1, i)</code>.</figcaption>
</figure>

One output map describes each source dimension. There are three map forms:

- `ConstantMap` fixes a source coordinate, such as row 1 after an integer
  index removes that request axis.
- `DimensionMap` describes an arithmetic rule, such as request `i` becoming
  source column `i`.
- `ArrayMap` stores explicit coordinates for irregular or fancy indexing.

Together, the request domain and these per-source-dimension maps are the
complete reusable description of an index.

---

<nav aria-label="Guide chapter navigation">
  <strong>Previous:</strong> <a href="../01-indexing/">An index selects coordinates</a>
  ·
  <strong>Next:</strong> <a href="../03-composition/">Lazy views compose</a>
</nav>
