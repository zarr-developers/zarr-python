# Indexing pattern reference

Every NumPy indexing pattern compiles to an `IndexTransform`. This page is
the matrix: one row per pattern, each showing the transform it produces —
its result shape and whether it is a **box** (constant and affine maps
only: an interval and stride per dimension) or a **query** (at least one
explicit coordinate list, whose order and repeats are semantic).

## Selection matrix

The table uses a 6-by-8 `image` and the names from the executable matrix
below. `t` is `IndexTransform.from_shape((6, 8))`; each expression compiles
a selection into a transform without touching data.

| Case | Expression | Result shape | Category |
| --- | --- | --- | --- |
| Basic slice | `t[1:5, ::2]` | `(4, 4)` | box |
| Integer axis removal | `t[2, :]` | `(8,)` | box |
| Negative stride | `t[::-2, :]` | `(3, 8)` | box |
| Empty selection | `t[2:2, :]` | `(0, 8)` | box |
| Boolean mask | `t.vindex[mask]` | `(10,)` | query |
| Orthogonal | `t.oindex[rows, columns]` | `(3, 2)` | query |
| Vectorized | `t.vindex[vector_rows, vector_columns]` | `(3,)` | query |
| Broadcasting | `t.vindex[broadcast_rows, broadcast_columns]` | `(2, 3)` | query |
| Repeated, out of order | `t.oindex[rows, 2:6]` | `(3, 4)` | query |

The category is structural, read straight off the transform's output maps:
basic indexing composes to `ConstantMap` and `DimensionMap` entries and
stays a box at any depth; one `ArrayMap` — from `oindex`, `vindex`, or a
mask — makes a query permanently. See the
[design notes](../design-notes.md#bounding-box-selections-vs-query-selections)
for why consumers dispatch on the distinction.

## Executable NumPy references

Every expected value is evaluated directly with NumPy (`numpy.ix_` for the
orthogonal rows; direct indexing for the rest). The same matrix compiles
each selection through the transform accessors, then checks shape,
category, and — resolved through the public reader — values:

```python
--8<-- "snippets/indexing_patterns.py:indexing-patterns"
```

`LazyArray` adds nothing to these semantics: it is a regular array-like
API whose `.lazy`, `.lazy.oindex`, and `.lazy.vindex` accessors compile the
same dialects to the same transforms — the only difference is the return
type, a view instead of an array. The test suite holds the wrapper to this
matrix.

## Positions vs literal coordinates

| Surface | Meaning of an integer index | Meaning of `-1` |
| --- | --- | --- |
| `IndexDomain` and `IndexTransform` | A literal coordinate in the current domain | The address `-1`, when the domain contains it |
| `LazyArray.lazy` | A NumPy-style position in the current view | The last position, normalized before transform composition |

The wrapper's three indexing modes all use positions in the current view. Each
derived view begins at position zero, while the transform algebra underneath
retains literal coordinates. The
[Coordinates are addresses](index.md#coordinates-are-addresses) section develops
that distinction with non-zero and negative-origin domains.

---

<nav aria-label="Reference page navigation">
  <strong>Next:</strong> <a href="../integrations/">Integration boundaries</a>
</nav>
