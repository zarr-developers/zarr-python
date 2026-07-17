# Lazy indexing

Zarr arrays support *lazy* indexing through the `Array.lazy` accessor. Where
ordinary indexing reads or writes data immediately, `a.lazy[selection]` returns a
lightweight **view** — itself a `zarr.Array` — without touching storage. Views
compose, support orthogonal and coordinate selection, write through to the
backing array, and materialize on demand.

Zarr's lazy indexing follows [TensorStore's indexing
model](https://google.github.io/tensorstore/python/indexing.html): a view has a
**domain** — a box of coordinates — and a positive index is a **literal
coordinate** in that domain (a negative index counts from the domain's end, as
in NumPy).

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

### Domains are preserved: an index is a name, not a position

A view keeps the coordinates of the cells it selects. Slicing `[2:10]` does not
renumber anything — the view's domain *is* `[2, 10)`, and coordinate 3 still
means what it meant on the parent:

```python exec="true" session="lazy" source="above" result="ansi"
v = a.lazy[2:10]
print(v.shape)                 # (8,) — eight cells...
print(v.lazy[3].result())      # ...and coordinate 3 is still base cell 3
print(v.lazy[3:7].result())    # coordinates [3, 7) — literal, stable
```

This is what makes composition safe: a coordinate means the same cell no matter
how many views deep you are. `a.lazy[2:10].lazy[3:7]` and `a.lazy[3:7]` are the
same selection.

The price of stable names: **positions are not valid indices.** The first
element of `v` is coordinate 2, not 0 — and `v[0]` is an error, not the first
element:

```python exec="true" session="lazy" source="above" result="ansi"
try:
    v[0]
except IndexError as e:
    print(e)
```

To renumber a view explicitly, move its domain with `translate_to` (or shift it
with `translate_by`) — the data does not move, only the labels:

```python exec="true" session="lazy" source="above" result="ansi"
z = v.translate_to((0,))       # same cells, coordinates now [0, 8)
print(z.lazy[0].result())      # coordinate 0 -> base cell 2
```

### Negative indices count from the end

*Positive* indices are literal coordinates (so positions still are not indices —
see above), but a **negative** index counts from the end of the current view's
domain, exactly like NumPy: `k` maps to `exclusive_max + k`. This holds in every
syntactic form — integers, slice bounds, and index arrays — at the public
boundary (`a.lazy[...]`, `v.lazy[...]`, `v[...]`, `.oindex`, `.vindex`, and the
`get_/set_*_selection` methods). For a fresh, zero-origin array both rules
coincide with NumPy in full:

```python exec="true" session="lazy" source="above" result="ansi"
print(a.lazy[-1].result())               # last element, like NumPy
print(a.lazy[-3:].result())              # the last three
print(a.lazy.oindex[[-1, -2]].result())  # index arrays wrap too
```

On a view the wrap is relative to the *view's* end: on the domain-`[2, 10)`
view `v`, `-1` names coordinate `9`, and `-8` names the first cell (coordinate
`2`). Only an out-of-range index raises — a positive past the end, or a
negative below `-size` (one that would wrap past the domain origin):

```python exec="true" session="lazy" source="above" result="ansi"
n = a.shape[0]
for make in (lambda: a.lazy[n], lambda: a.lazy[-n - 1], lambda: a.lazy.oindex[[-n - 1]]):
    try:
        make()
    except IndexError as e:
        print(type(e).__name__, "-", str(e).split(";")[0])
```

### No clamping: intervals must fit the domain

A slice interval must be contained in the domain — an out-of-range bound is an
error, not a silently shorter result. Empty intervals are the one exception:
they are valid anywhere. Reversed bounds are an error, not an empty result.

```python exec="true" session="lazy" source="above" result="ansi"
for sel in (slice(5, 100), slice(5, 2)):
    try:
        a.lazy[sel]
    except IndexError as e:
        print(type(e).__name__, "-", str(e).split(";")[0])
print(a.lazy[5:5].shape)          # empty is fine, anywhere
```

### Strided views renumber by division

A strided slice produces a domain in *strided units*: for step `k`, the new
origin is `start / k` rounded toward zero, and coordinate `origin + i` maps to
base cell `start + i*k` (TensorStore's rule):

```python exec="true" session="lazy" source="above" result="ansi"
s = a.lazy[1:10:3]              # base cells 1, 4, 7
print(s)                        # domain [0, 3)
print(s.lazy[1].result())       # coordinate 1 -> base cell 4
```

### One coordinate system per view

Every way of indexing a view — `v[...]`, `v.lazy[...]`, `v.oindex`, `v.vindex`
— uses the same rule: positive indices are literal domain coordinates, and
negative indices wrap from the domain's end. (The base array's ordinary `a[...]`
API is unchanged: it keeps full NumPy semantics.) NumPy-style zero-based access
to a view's *positive* range is spelled explicitly: materialize with `result()`
/ `np.asarray`, or renumber with `translate_to`.

```python exec="true" session="lazy" source="above" result="ansi"
print(v[3], v.lazy[3].result())     # same coordinate, same cell
print(np.asarray(v)[0])             # materialized: NumPy rules apply
print(a[-1])                        # base arrays keep NumPy semantics
```

## Common patterns

### Crop, analyze, crop again

```python exec="true" session="lazy" source="above" result="ansi"
img = zarr.create_array(store="memory://lazy-img", shape=(100, 100), chunks=(10, 10), dtype="float64")
img[...] = np.arange(100 * 100).reshape(100, 100)

crop = img.lazy[25:75, 25:75]          # no I/O; domain [25,75) x [25,75)
inner = crop.lazy[35:65, 35:65]        # coordinates are literal: this is img[35:65, 35:65]
print(crop.shape, inner.shape)
print(float(np.mean(inner)))           # I/O happens here, for the inner crop only
```

### Write through a view

Assignment through the accessor, or through a view, routes values back to
storage — including strided and composed selections:

```python exec="true" session="lazy" source="above" result="ansi"
img.lazy[30:50, 40:60] = 0.0           # region write
tile = img.lazy[30:50, 40:60]
tile[30:35, 40:45] = 7.0               # write through the view, same coordinates
img.lazy[::2, ::2] = -1.0              # strided write, NumPy-equivalent cells
print(img[29:33, 39:43])
```

### Orthogonal and coordinate selection

`lazy.oindex` selects an outer product per axis; `lazy.vindex` selects points.
Index-array *values* are domain coordinates; the dimension a fancy selection
*creates* gets a fresh `[0, n)` domain (there is no meaningful coordinate to
preserve for "the i-th pick"):

```python exec="true" session="lazy" source="above" result="ansi"
rows = img.lazy.oindex[[3, 17, 42], :]     # picked dim: domain [0, 3); row dim preserved
sub = rows.lazy[:, 10:20]                  # column window, literal coords
print(rows.shape, sub.shape)

pts = img.lazy.vindex[[3, 17], [5, 9]]     # two points -> fresh domain [0, 2)
print(pts.result())
```

Boolean masks are array selections, so they go through `oindex`/`vindex`; the
positions of `True` values become **coordinates, counted from 0** — not offsets
from the view's origin. On a view whose domain starts at 2, a mask `True` at
position 3 addresses coordinate 3, and a `True` at position 0 or 1 is out of
the domain (matching TensorStore, where a mask is sugar for the coordinate
array of its `True` positions):

```python exec="true" session="lazy" source="above" result="ansi"
mask = np.zeros(12, dtype=bool)
mask[[1, 4, 6]] = True
print(a.lazy.oindex[mask].result())
```

### Materializing

`view.result()`, `view[...]`, and `np.asarray(view)` are equivalent whole-view
reads; views also work directly with NumPy reductions. Views are **not**
iterable (iterate the materialized result instead):

```python exec="true" session="lazy" source="above" result="ansi"
w2 = a.lazy[3:9]
print(w2.result(), float(np.mean(w2)))
try:
    iter(w2)
except TypeError as e:
    print(e)
```

### Chunk-aware processing

`chunk_projections` enumerates the stored chunks a view touches: which store
object (`coord`, `key`), its stored `shape`, the region *within the chunk* the
view covers (`chunk_selection`), the region *of the view* it maps to
(`array_selection`, positional — 0-based into the view's extent), and whether
the chunk is only partially covered (`is_partial` — a partial write requires a
read-modify-write):

```python exec="true" session="lazy" source="above" result="ansi"
for p in a.lazy[2:10].chunk_projections():
    print(p.key, p.chunk_selection, p.array_selection, p.is_partial)
print(a.lazy[2:10].is_chunk_aligned())
print(a.lazy[3:9].is_chunk_aligned())   # starts and ends on chunk boundaries
```

This is the supported way to partition any selection for parallel or
chunk-at-a-time work — compose the selection through `.lazy`, then project.
Since `array_selection` is positional, re-zero the view (or materialize) to use
it:

```python exec="true" session="lazy" source="above" result="ansi"
crop0 = img.lazy[25:75, 25:75].translate_to((0, 0))
total = 0.0
for p in crop0.chunk_projections():
    total += float(np.sum(crop0[tuple(slice(s.start, s.stop) for s in p.array_selection)]))
print(total == float(np.sum(crop0)))
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

- **A view's positive indices are coordinates, not positions.** `a.lazy[2:10]`
  is indexed with 2..9, not 0..7. Renumber explicitly with
  `view.translate_to((0, ...))` if you want positions.
- **Negative indices count from the end**, exactly like NumPy — in every form
  (integer, slice bound, index array). `k` maps to `exclusive_max + k` in the
  current view's domain; only a negative below `-size` (or a positive past the
  end) is out of bounds.
- **No clamping**: out-of-range slice bounds raise (including a negative that
  would wrap past the domain origin); reversed bounds raise; only empty
  intervals are allowed anywhere.
- **No negative steps**: `a.lazy[::-1]` raises; reversal is not yet supported.
- **No `newaxis`**: `a.lazy[None]` raises; insert axes on the materialized
  result instead.
- **The basic accessor takes basic selections only** (integers, slices,
  ellipsis). Lists, arrays, and boolean masks go through `lazy.oindex` /
  `lazy.vindex`.
- **Views are not iterable**; iterate `view.result()`.
- **Base arrays are unchanged**: `a[-1]`, `a[5:100]`, and friends keep full
  NumPy semantics on non-view arrays.

## Current limitations

- Negative slice steps (reversal) are not yet supported.
- Integer indexing a dimension *created by* an `oindex`/`vindex` selection
  (e.g. `rows.lazy[0]` after `rows = a.lazy.oindex[[3, 17, 42], :]`) is not yet
  supported reliably; slice the view instead (`rows.lazy[0:1]`).
- Composing a fancy (`oindex`/`vindex`) selection onto a view that already has a
  fancy-indexed axis (fancy-after-fancy — e.g. `a.lazy.oindex[[0, 2, 5], :]`
  then `.oindex[:, [5]]`) is not supported and raises `NotImplementedError`.
  Materialize the view first with `.result()` and index the array, or reorder
  the selections so the fancy step is applied last.
- `chunk_projections(unit="read")` on sharded arrays (inner-chunk granularity)
  is not yet implemented; use `unit="write"`.
- Views cannot be resized or appended to, and block selection is not defined
  for views.
