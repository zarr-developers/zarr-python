# An index selects coordinates

Begin with an ordinary NumPy array. `image` has 6 rows and 8 columns, and the
selection `image[1:5, 2]` takes rows 1 through 4 from column 2. Four source cells
are selected, the column index removes one axis, and the request therefore has
shape `(4,)`.

<figure>
--8<-- "_static/diagrams/indexing-selection.svg"
<figcaption>A one-dimensional request of shape <code>(4,)</code> visits four source cells in top-to-bottom result order.</figcaption>
</figure>

The selected source values are `10`, `18`, `26`, and `34`. Their order in the
result is the order of the request: position 0 gets source cell `(1, 2)`,
position 1 gets `(2, 2)`, and so on.

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
