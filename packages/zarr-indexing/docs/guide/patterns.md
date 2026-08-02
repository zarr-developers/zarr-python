# Indexing pattern reference

| Surface | Meaning of an integer index | Meaning of `-1` |
| --- | --- | --- |
| `IndexDomain` and `IndexTransform` | A literal coordinate in the current domain | The address `-1`, when the domain contains it |
| `LazyArray.lazy` | A NumPy-style position in the current view | The last position, normalized before transform composition |

The wrapper's three indexing modes all use positions in the current view. Each
derived view begins at position zero, while the transform algebra underneath
retains literal coordinates. The [coordinate chapter](02-transforms.md) develops
that distinction with non-zero and negative-origin domains.

## Selection matrix

The table uses the 6-by-8 `image` and names from the executable matrix below.
`lazy` means `LazyArray(image)`. A **box** has an interval and stride per source
dimension; a **query** carries explicit source coordinates.

| Case | Expression | Result shape | Category | API mode |
| --- | --- | --- | --- | --- |
| Basic slice | `lazy.lazy[1:5, ::2]` | `(4, 4)` | box | `lazy` |
| Integer axis removal | `lazy.lazy[2, :]` | `(8,)` | box | `lazy` |
| Negative stride | `lazy.lazy[::-2, :]` | `(3, 8)` | box | `lazy` |
| Empty selection | `lazy.lazy[2:2, :]` | `(0, 8)` | box | `lazy` |
| Boolean mask | `lazy.lazy.vindex[mask]` | `(10,)` | query | `vindex` |
| Orthogonal | `lazy.lazy.oindex[rows, columns]` | `(3, 2)` | query | `oindex` |
| Vectorized | `lazy.lazy.vindex[vector_rows, vector_columns]` | `(3,)` | query | `vindex` |
| Broadcasting | `lazy.lazy.vindex[broadcast_rows, broadcast_columns]` | `(2, 3)` | query | `vindex` |
| Repeated, out of order | `lazy.lazy.oindex[rows, 2:6]` | `(3, 4)` | query | `oindex` |

The [LazyArray API](../api/lazy_array.md) is the source of truth for negative
indices, axis removal, empty views, broadcasting, and the placement of advanced
indices. See the [design notes](../design-notes.md) for intentionally unsupported
compositions. Those edge cases are linked here rather than restating their
normalization and lowering rules.

## Executable NumPy references

Every expected value is evaluated directly with NumPy. Orthogonal arrays use
`numpy.ix_`; the basic and vectorized rows use NumPy's direct indexing. The
same matrix applies each selection to `LazyArray`, then checks values, result
shape, and box/query category.

```python
--8<-- "examples/indexing_patterns.py:indexing-patterns"
```

---

<nav aria-label="Reference page navigation">
  <strong>Next:</strong> <a href="../integrations/">Integration boundaries</a>
</nav>
