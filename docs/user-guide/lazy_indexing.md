# Lazy indexing

Zarr arrays support *lazy* indexing through the `Array.lazy` accessor. Where
ordinary indexing reads or writes data immediately, `a.lazy[selection]` returns a
lightweight **view** — itself a `zarr.Array` — without touching storage. Views
compose, support orthogonal and coordinate selection, write through to the
backing array, and materialize on demand.

```python exec="true" session="lazy"
import numpy as np
import zarr
```

```python exec="true" session="lazy" source="above" result="ansi"
a = zarr.create_array(store="memory://lazy-demo", shape=(12,), chunks=(3,), dtype="int32")
a[...] = np.arange(12)

view = a.lazy[2:10]        # no I/O happens here
print(view)
print(view.result())       # I/O happens here
```

## Theory: indexing as a declaration

Eager indexing is an *action*: `a[2:10]` performs I/O now and hands back the
bytes. Lazy indexing is a *declaration*: `a.lazy[2:10]` records **which cells
you mean** — an index transform mapping the view's coordinates to storage
coordinates — and defers the I/O. Because the selection is data rather than an
action, zarr can:

- **compose** it with further selections without reading anything,
- **write through** it (the same transform routes values back to storage),
- **plan** with it (`chunk_projections` enumerates exactly the stored chunks
  the declaration touches).

A view is a *window*: its shape is the selection's shape, and indexing a view is
always relative to the view itself, starting at zero — exactly like indexing a
NumPy view:

```python exec="true" session="lazy" source="above" result="ansi"
v = a.lazy[2:10]           # window onto cells 2..9
print(v.shape)             # the window's shape
print(v.lazy[0].result())  # the window's first element -> base cell 2
print(v.lazy[3:5].result())  # window cells 3,4 -> base cells 5,6
```

The `domain={ [2, 10) }` shown in a view's repr is **provenance** — the region
of the backing store the view maps onto. It is *not* the coordinate system you
index with; that is always `[0, shape)`.

### Literal coordinates: `-1` is just another index

A lazy selection is a declaration in **literal coordinates**. NumPy's negative
indexing ("count from the end") is sugar applied at *evaluation* time; in a
deferred, composable setting it would silently re-bind meaning as views compose.
Zarr therefore treats every coordinate in a lazy selection literally, and since
view domains start at 0, **a negative value is never in bounds — no matter the
syntactic form** (integer, slice bound, or index array):

```python exec="true" session="lazy" source="above" result="ansi"
for make in (lambda: a.lazy[-1], lambda: a.lazy[-3:], lambda: a.lazy.oindex[[-1]]):
    try:
        make()
    except IndexError as e:
        print(type(e).__name__, "-", e)
```

To select from the end, say what you mean literally:

```python exec="true" session="lazy" source="above" result="ansi"
n = a.shape[0]
print(a.lazy[n - 3 :].result())   # the last three elements
```

Positive out-of-range *slice* bounds are still fine — a slice denotes a range,
and ranges intersect the domain (as in Python and NumPy). Clamping can only
shorten a result; it can never silently select different data the way
wraparound can:

```python exec="true" session="lazy" source="above" result="ansi"
print(a.lazy[5:100].result())     # clamps to [5, 12)
```

### Two dialects on one object

A view is an ordinary `zarr.Array`, so it also has the ordinary *eager* methods
— and those keep full NumPy semantics, negatives included. The two dialects
split cleanly by intent:

| You are...  | Spelling                    | Negative indices        |
| ----------- | --------------------------- | ----------------------- |
| declaring   | `v.lazy[...]` (a new view)  | literal — raise         |
| accessing   | `v[...]`, `v.oindex[...]`   | NumPy — from the end    |

```python exec="true" session="lazy" source="above" result="ansi"
print(v[-1])                  # eager access: NumPy dialect -> base cell 9
print(v.lazy[v.shape[0] - 1].result())  # lazy declaration: same cell, said literally
```

## Common patterns

### Crop, analyze, crop again

```python exec="true" session="lazy" source="above" result="ansi"
img = zarr.create_array(store="memory://lazy-img", shape=(100, 100), chunks=(10, 10), dtype="float64")
img[...] = np.arange(100 * 100).reshape(100, 100)

crop = img.lazy[25:75, 25:75]          # 50x50 window, no I/O
inner = crop.lazy[10:40, 10:40]        # 30x30 window of the window
print(crop.shape, inner.shape)
print(float(np.mean(inner)))           # I/O happens here, for the inner crop only
```

### Write through a view

Assignment through the accessor, or through a view, routes values back to
storage — including strided and composed selections:

```python exec="true" session="lazy" source="above" result="ansi"
img.lazy[30:50, 40:60] = 0.0           # region write, no read-back needed here
tile = img.lazy[30:50, 40:60]
tile[0:5, 0:5] = 7.0                   # eager write through the view
img.lazy[::2, ::2] = -1.0              # strided write, NumPy-equivalent
print(img[29:33, 39:43])
```

### Orthogonal and coordinate selection

`lazy.oindex` selects an outer product per axis; `lazy.vindex` selects points.
Both return views that compose:

```python exec="true" session="lazy" source="above" result="ansi"
rows = img.lazy.oindex[[3, 17, 42], :]     # three full rows -> shape (3, 100)
sub = rows.lazy[:, 10:20]                  # then a column window of those rows
print(rows.shape, sub.shape)

pts = img.lazy.vindex[[3, 17], [5, 9]]     # two points -> shape (2,)
print(pts.result())
```

Boolean masks are array selections, so they go through `oindex`/`vindex`:

```python exec="true" session="lazy" source="above" result="ansi"
mask = np.zeros(12, dtype=bool)
mask[[1, 4, 6]] = True
print(a.lazy.oindex[mask].result())
```

### Materializing

`view.result()`, `view[...]`, and `np.asarray(view)` are equivalent reads of the
whole view; views also work directly with NumPy reductions:

```python exec="true" session="lazy" source="above" result="ansi"
w = a.lazy[3:9]
print(w.result(), np.asarray(w), float(np.mean(w)))
```

### Chunk-aware processing

`chunk_projections` enumerates the stored chunks a view touches: which store
object (`coord`, `key`), its stored `shape`, the region *within the chunk* the
view covers (`chunk_selection`), the region *of the view* it maps to
(`array_selection`), and whether the chunk is only partially covered
(`is_partial` — a partial write requires a read-modify-write):

```python exec="true" session="lazy" source="above" result="ansi"
for p in a.lazy[2:10].chunk_projections():
    print(p.key, p.chunk_selection, p.array_selection, p.is_partial)
print(a.lazy[2:10].is_chunk_aligned())
print(a.lazy[3:9].is_chunk_aligned())   # starts and ends on chunk boundaries
```

This is the supported way to partition any selection for parallel or
chunk-at-a-time work — compose the selection through `.lazy`, then project:

```python exec="true" session="lazy" source="above" result="ansi"
total = 0.0
crop = img.lazy[25:75, 25:75]
for p in crop.chunk_projections():
    total += float(np.sum(crop[p.array_selection]))
print(total == float(np.sum(crop)))
```

For sharded arrays, pass `unit="write"` to enumerate at shard (write-unit)
granularity; read-unit projections for sharded arrays are not yet implemented.

### What a view will not tell you

Members that describe the chunk grid assume the array *fills* its grid, which a
view generally does not. On a view they raise `zarr.errors.LazyViewError`
instead of silently describing the backing array:

```python exec="true" session="lazy" source="above" result="ansi"
try:
    v.chunks
except zarr.errors.LazyViewError as e:
    print(e)
```

Logical members (`shape`, `size`, `nbytes`, `dtype`, `attrs`, ...) reflect the
view; `metadata` and `chunk_grid` remain available and describe the *backing*
array.

## Coming from NumPy

- **Negative indices raise on the lazy path** — in every form (integer, slice
  bound, index array). Use `shape[dim] - k`, or the eager methods, which keep
  NumPy semantics.
- **No negative steps**: `a.lazy[::-1]` raises; reversal is not supported by
  zarr indexing generally.
- **No `newaxis`**: `a.lazy[None]` raises; insert axes on the materialized
  result instead.
- **The basic accessor takes basic selections only** (integers, slices,
  ellipsis). Lists, arrays, and boolean masks go through `lazy.oindex` /
  `lazy.vindex`.
- **Scalar reads return NumPy scalars**: `a.lazy[3].result()` is `np.int32(3)`,
  matching `a[3]`'s value with scalar type.

## Current limitations

- Integer indexing a dimension *created by* an `oindex`/`vindex` selection
  (e.g. `rows.lazy[0]` after `rows = a.lazy.oindex[[3, 17, 42], :]`) is not yet
  supported reliably; slice the view instead (`rows.lazy[0:1]`).
- `chunk_projections(unit="read")` on sharded arrays (inner-chunk granularity)
  is not yet implemented; use `unit="write"`.
- Views cannot be resized or appended to, and block selection is not defined
  for views.
