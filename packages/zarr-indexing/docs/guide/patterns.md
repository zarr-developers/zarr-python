# Indexing pattern reference

Every NumPy indexing idiom is modeled by an `IndexTransform`: a domain (the
result's coordinates) and one output map per source dimension. This page
builds that model **by hand for each idiom**, so the anatomy is explicit —
which map kind an idiom needs, where the offset and stride go, and how an
index array's shape spells outer-product versus pointwise. Each model is
then proven equal to what the selection compiler derives, and its values
are checked against NumPy.

## The idiom-to-model matrix

Each idiom over a 6-by-8 `image`, shown two ways: the Python construction,
and the same transform in **wire form** — the [ndsel](../ndsel.md)
canonical body `to_json` produces. Both spell the whole
object: a domain whose extent is the result shape, then one output map per
source dimension. The index-array variables (`rows`, `columns`,
`mask_rows, mask_columns = np.nonzero(mask)`, and friends) are defined in
the executable matrix at the end of the page.

**`image[1:5, ::2]`** — box. The offset picks where cell 0 reads; the
stride skips:

=== "Python"

    ```python
    IndexTransform(
        domain=IndexDomain.from_shape((4, 4)),
        output=(
            DimensionMap(input_dimension=0, offset=1),
            DimensionMap(input_dimension=1, stride=2),
        ),
    )
    ```

=== "JSON"

    ```json
    {
     "input_rank": 2,
     "input_inclusive_min": [0, 0],
     "input_exclusive_max": [4, 4],
     "input_labels": ["", ""],
     "output": [
      {"offset": 1, "stride": 1, "input_dimension": 0},
      {"offset": 0, "stride": 2, "input_dimension": 1}
     ]
    }
    ```

**`image[2, :]`** — box. A rank-1 domain with two output maps: the dropped
axis survives as the wire's constant form, a bare `{"offset": 2}`:

=== "Python"

    ```python
    IndexTransform(
        domain=IndexDomain.from_shape((8,)),
        output=(ConstantMap(2), DimensionMap(input_dimension=0)),
    )
    ```

=== "JSON"

    ```json
    {
     "input_rank": 1,
     "input_inclusive_min": [0],
     "input_exclusive_max": [8],
     "input_labels": [""],
     "output": [
      {"offset": 2},
      {"offset": 0, "stride": 1, "input_dimension": 0}
     ]
    }
    ```

**`image[::-2, :]`** — box. Reversal is nothing but a negative stride, and
the offset is where cell 0 reads (row 5):

=== "Python"

    ```python
    IndexTransform(
        domain=IndexDomain.from_shape((3, 8)),
        output=(
            DimensionMap(input_dimension=0, offset=5, stride=-2),
            DimensionMap(input_dimension=1),
        ),
    )
    ```

=== "JSON"

    ```json
    {
     "input_rank": 2,
     "input_inclusive_min": [0, 0],
     "input_exclusive_max": [3, 8],
     "input_labels": ["", ""],
     "output": [
      {"offset": 5, "stride": -2, "input_dimension": 0},
      {"offset": 0, "stride": 1, "input_dimension": 1}
     ]
    }
    ```

**`image[2:2, :]`** — box. Emptiness lives in the domain
(`input_exclusive_max[0]` equals the minimum); the maps are ordinary:

=== "Python"

    ```python
    IndexTransform(
        domain=IndexDomain.from_shape((0, 8)),
        output=(
            DimensionMap(input_dimension=0, offset=2),
            DimensionMap(input_dimension=1),
        ),
    )
    ```

=== "JSON"

    ```json
    {
     "input_rank": 2,
     "input_inclusive_min": [0, 0],
     "input_exclusive_max": [0, 8],
     "input_labels": ["", ""],
     "output": [
      {"offset": 2, "stride": 1, "input_dimension": 0},
      {"offset": 0, "stride": 1, "input_dimension": 1}
     ]
    }
    ```

**`image[mask]`** — query. A mask is its nonzero coordinates: two
correlated index arrays over one flat axis, entry `i` of each pairing into
one cell:

=== "Python"

    ```python
    IndexTransform(
        domain=IndexDomain.from_shape((10,)),
        output=(ArrayMap(mask_rows), ArrayMap(mask_columns)),
    )
    ```

=== "JSON"

    ```json
    {
     "input_rank": 1,
     "input_inclusive_min": [0],
     "input_exclusive_max": [10],
     "input_labels": [""],
     "output": [
      {"offset": 0, "stride": 1, "index_array": [0, 0, 1, 1, 2, 3, 3, 4, 5, 5], "index_array_bounds": ["-inf", "+inf"]},
      {"offset": 0, "stride": 1, "index_array": [0, 5, 2, 7, 4, 1, 6, 3, 0, 5], "index_array_bounds": ["-inf", "+inf"]}
     ]
    }
    ```

**`image[np.ix_(rows, columns)]`** — query. The outer product is spelled by
nesting: `[[4], [1], [1]]` varies down the first axis, `[[2, 5]]` across
the second, each singleton along the other:

=== "Python"

    ```python
    IndexTransform(
        domain=IndexDomain.from_shape((3, 2)),
        output=(ArrayMap(rows.reshape(3, 1)), ArrayMap(columns.reshape(1, 2))),
    )
    ```

=== "JSON"

    ```json
    {
     "input_rank": 2,
     "input_inclusive_min": [0, 0],
     "input_exclusive_max": [3, 2],
     "input_labels": ["", ""],
     "output": [
      {"offset": 0, "stride": 1, "index_array": [[4], [1], [1]], "index_array_bounds": ["-inf", "+inf"]},
      {"offset": 0, "stride": 1, "index_array": [[2, 5]], "index_array_bounds": ["-inf", "+inf"]}
     ]
    }
    ```

**`image[vector_rows, vector_columns]`** — query. Pointwise: two flat
arrays over one shared axis:

=== "Python"

    ```python
    IndexTransform(
        domain=IndexDomain.from_shape((3,)),
        output=(ArrayMap(vector_rows), ArrayMap(vector_columns)),
    )
    ```

=== "JSON"

    ```json
    {
     "input_rank": 1,
     "input_inclusive_min": [0],
     "input_exclusive_max": [3],
     "input_labels": [""],
     "output": [
      {"offset": 0, "stride": 1, "index_array": [4, 1, 1], "index_array_bounds": ["-inf", "+inf"]},
      {"offset": 0, "stride": 1, "index_array": [2, 5, 2], "index_array_bounds": ["-inf", "+inf"]}
     ]
    }
    ```

**`image[broadcast_rows, broadcast_columns]`** — query. NumPy broadcasting,
kept compact: singleton axes retain the independent dependencies:

=== "Python"

    ```python
    IndexTransform(
        domain=IndexDomain.from_shape((2, 3)),
        output=(
            ArrayMap(broadcast_rows.reshape(2, 1)),
            ArrayMap(broadcast_columns.reshape(1, 3)),
        ),
    )
    ```

=== "JSON"

    ```json
    {
     "input_rank": 2,
     "input_inclusive_min": [0, 0],
     "input_exclusive_max": [2, 3],
     "input_labels": ["", ""],
     "output": [
      {"offset": 0, "stride": 1, "index_array": [[0], [3]], "index_array_bounds": ["-inf", "+inf"]},
      {"offset": 0, "stride": 1, "index_array": [[1, 4, 6]], "index_array_bounds": ["-inf", "+inf"]}
     ]
    }
    ```

**`image[rows, 2:6]`** — query. One lookup table (order and repeats kept)
beside one ordinary affine map — one index array makes the whole selection
a query:

=== "Python"

    ```python
    IndexTransform(
        domain=IndexDomain.from_shape((3, 4)),
        output=(
            ArrayMap(rows.reshape(3, 1)),
            DimensionMap(input_dimension=1, offset=2),
        ),
    )
    ```

=== "JSON"

    ```json
    {
     "input_rank": 2,
     "input_inclusive_min": [0, 0],
     "input_exclusive_max": [3, 4],
     "input_labels": ["", ""],
     "output": [
      {"offset": 0, "stride": 1, "index_array": [[4], [1], [1]], "index_array_bounds": ["-inf", "+inf"]},
      {"offset": 2, "stride": 1, "input_dimension": 1}
     ]
    }
    ```

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

<nav aria-label="Reference page navigation" markdown>
**Next:** [Integration boundaries](integrations.md)
</nav>
