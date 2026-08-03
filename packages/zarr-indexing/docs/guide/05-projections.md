# One cell domain, two projections

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

## Order and duplicates need the request-side projection

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

<nav aria-label="Guide chapter navigation">
  <strong>Previous:</strong> <a href="../04-chunks/">A request becomes a chunk plan</a>
  ·
  <strong>Next:</strong> <a href="../../api/">API reference</a>
</nav>
