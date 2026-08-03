# An index selects coordinates

Begin with an ordinary NumPy array. `image` has 3 rows and 4 columns, filled
row-by-row with the values 0 through 11. The selection `image[1, 0:4]` takes
all four columns from row 1. Four source cells are selected, the row index
removes one axis, and the request therefore has shape `(4,)`.

<figure>
<div class="zi-figure-scroll" role="region" aria-label="Scrollable basic indexing selection diagram" tabindex="0">
--8<-- "_static/diagrams/indexing-selection.svg"
</div>
<figcaption>A one-dimensional request of shape <code>(4,)</code> visits four source cells in left-to-right result order.</figcaption>
</figure>

The diagram labels addresses as `coord:` and stored data as `value:`. Source
cell `coord: (1, 0)` stores `value: 4`, and request `coord: 0` receives that
value. Request coordinates 1, 2, and 3 similarly receive values 5, 6, and 7
from source coordinates `(1, 1)`, `(1, 2)`, and `(1, 3)`. The request sequence
therefore shows both where each result value lives and where it came from.

The wrapper below gives the same familiar selection a lazy spelling. Indexing
through `.lazy` creates `view`; the last line asks for its values and checks the
observable NumPy result.

```python
--8<-- "examples/canonical_slice.py:canonical-slice"
```

The important first step is simply that an index describes which source cells
fill a result of a particular shape and order. The next chapter gives the
numbers on both sides of that description a precise meaning.

---

<nav aria-label="Guide chapter navigation">
  <strong>Previous:</strong> <a href="../">Visual guide</a>
  ·
  <strong>Next:</strong> <a href="../02-transforms/">Coordinates are addresses</a>
</nav>
