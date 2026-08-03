# A request becomes a chunk plan

Return to the same 6-by-8 image and the same selection, `image[1:5, 2]`.
Giving the image a 3-by-4 chunk shape does not change the four selected values
or their order. It changes only how the work is divided: rows 1 and 2 come
from chunk `(0, 0)`, while rows 3 and 4 come from chunk `(1, 0)`.

<figure>
<div class="zi-figure-scroll" role="region" aria-label="Scrollable chunk projection overlay diagram" tabindex="0">
--8<-- "_static/diagrams/chunk-overlay.svg"
</div>
<figcaption>The unchanged selection crosses one 3-row chunk boundary, so it touches exactly chunks <code>(0, 0)</code> and <code>(1, 0)</code> of the 6-by-8 source.</figcaption>
</figure>

Every planned chunk keeps three coordinate frames distinct:

- `chunk_coords` identifies a cell in the chunk grid. Chunk coordinates are
  integers, so the prepended chunk from chapter 2 really has coordinate `-1`;
  it is not an alias for the final chunk.
- `chunk_domain` gives that chunk's bounds in **global source coordinates**.
  Here the two domains are `[0, 3) × [0, 4)` and
  `[3, 6) × [0, 4)`. For the prepended one-dimensional example, chunk `-1`
  has the global domain `[-3, 0)`.
- Chunk-local positions start from zero inside each chunk. Global row 3 is
  therefore local row 0 in chunk `(1, 0)`. This zero-origin local frame is
  separate from both the global `chunk_domain` and the possibly negative
  chunk coordinate.

`plan_chunks` needs only a transform and a grid. The executable example builds
the canonical 3-by-4 grid, plans the request, and iterates the same plan again
to show that planning is reusable.

```python
--8<-- "examples/chunk_projection.py:chunk-projection"
```

The plan describes work but does not perform it. It contains no array source,
storage backend, codec pipeline, buffer, or scheduler. A Zarr reader, a task
queue, or a viewport can consume the same logical plan and decide independently
how and when to fetch its two chunks.

---

<nav aria-label="Guide chapter navigation">
  <strong>Previous:</strong> <a href="../03-composition/">Lazy views compose</a>
  ·
  <strong>Next:</strong> <a href="../05-projections/">One cell domain, two projections</a>
</nav>
