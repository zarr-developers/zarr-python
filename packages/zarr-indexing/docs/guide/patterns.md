# Indexing pattern reference

Every NumPy indexing idiom is modeled by an `IndexTransform`: a domain (the
result's coordinates) and one output map per source dimension. This page
builds that model **by hand for each idiom**, so the anatomy is explicit —
which map kind an idiom needs, where the offset and stride go, and how an
index array's shape spells outer-product versus pointwise. Each model is
then proven equal to what the selection compiler derives, and its values
are checked against NumPy.

## The idiom-to-model matrix

Over a 6-by-8 `image`; the full constructions, with per-idiom commentary,
are in the executable matrix below. Each row is the complete model — both
constructor halves. The domain is the result's coordinates, so its shape is
the result shape; every one here is zero-origin, spelled
`IndexDomain.from_shape(...)`.

| NumPy idiom | Domain | Category | Output maps |
| --- | --- | --- | --- |
| `image[1:5, ::2]` | `from_shape((4, 4))` | box | `DimensionMap(0, offset=1)`, `DimensionMap(1, stride=2)` |
| `image[2, :]` | `from_shape((8,))` | box | `ConstantMap(2)`, `DimensionMap(0)` |
| `image[::-2, :]` | `from_shape((3, 8))` | box | `DimensionMap(0, offset=5, stride=-2)`, `DimensionMap(1)` |
| `image[2:2, :]` | `from_shape((0, 8))` | box | `DimensionMap(0, offset=2)`, `DimensionMap(1)` — emptiness lives in the domain |
| `image[mask]` | `from_shape((10,))` | query | two correlated `ArrayMap`s: the mask's nonzero rows and columns |
| `image[np.ix_(rows, columns)]` | `from_shape((3, 2))` | query | `ArrayMap` shaped `(3, 1)`, `ArrayMap` shaped `(1, 2)` — distinct axes |
| `image[vector_rows, vector_columns]` | `from_shape((3,))` | query | two 1-D `ArrayMap`s on one shared axis — pointwise |
| `image[broadcast_rows, broadcast_columns]` | `from_shape((2, 3))` | query | two `ArrayMap`s carrying the full `(2, 3)` broadcast block |
| `image[rows, 2:6]` | `from_shape((3, 4))` | query | `ArrayMap` shaped `(3, 1)`, `DimensionMap(1, offset=2)` |

Two structural rules do all the work:

- **Category**: `ConstantMap` and `DimensionMap` entries keep a selection a
  box at any composition depth; one `ArrayMap` makes it a query permanently.
  See the [design notes](../design-notes.md#bounding-box-selections-vs-query-selections)
  for why consumers dispatch on this.
- **Fancy flavor is spelled by shape**: index arrays varying over distinct
  axes (singleton elsewhere) form an outer product; arrays sharing their
  non-singleton axes pair pointwise.

## The executable matrix

Each case hand-builds the model, checks shape, category, and NumPy values
(resolved through the public reader), then proves the selection compiler
derives the same transform. One wrinkle the last assert documents: compiled
*basic* selections keep literal domains (`t[1:5, ...]` starts at
coordinate 1 — see [Positions vs literal coordinates](#positions-vs-literal-coordinates)),
so they equal the zero-origin models after `translate_domain_to`:

```python
--8<-- "snippets/indexing_patterns.py:indexing-patterns"
```

`LazyArray` adds nothing to these semantics: it is a regular array-like API
whose `.lazy`, `.lazy.oindex`, and `.lazy.vindex` accessors compile the same
dialects to the same transforms — the only difference is the return type, a
view instead of an array. The test suite holds the wrapper to this matrix.

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
