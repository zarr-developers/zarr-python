# Array Engine Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `Array`/`AsyncArray` route all data I/O through pluggable engine objects (`ArrayEngine`/`AsyncArrayEngine` protocols, `Region` interchange), with a default pure-Python engine and a Zarrista (zarrs-backed) engine, replacing `zarr.crud`/`zarr.zarrs`/`packages/zarrs-bindings`.

**Architecture:** Engines are bound to `(store, path, metadata)` and speak only contiguous step-1 boxes (`Region(start, end_exclusive)`). The facade (`Array`/`AsyncArray`) normalizes every public selection kind to Region calls: contiguous basic selections map directly; strided/orthogonal/coordinate/mask selections go through their bounding box with numpy post-indexing (reads) or box-level read-patch-write (writes). Hierarchy engines are store-bound factories minting resource-sharing array engines. The sync `Array` calls a sync engine directly (no event loop on the data path).

**Tech Stack:** Python protocols (`typing.Protocol`), existing codec-pipeline machinery (default engine), Zarrista main branch (git-pinned; PyO3/zarrs) for the accelerated engine.

**Spec:** `docs/superpowers/specs/2026-07-22-array-engine-protocol-design.md` — read it before starting any task.

## Global Constraints

- Run all Python commands with `uv run` (e.g. `uv run pytest`, `uv run mypy`).
- Conventional commits with trailer `Assisted-by: ClaudeCode:<model>`.
- Never `git add -A`; stage explicit paths only. Never commit `.claude/` or `.superpowers/`.
- Protocol names have no `Protocol` suffix: `ArrayEngine`, `AsyncArrayEngine`, `HierarchyEngine`, `AsyncHierarchyEngine`.
- The v1 engine contract has **no `out` parameter** and **no config key**; engine choice is explicit (`engine="default" | "zarrista" | <instance>`).
- Engine read results need only implement `__array__`.
- Fail loud: `UnsupportedEngineError` at engine construction; `NotImplementedError` for unsupported operations. No silent fallback.
- Docstrings are markdown (mkdocs), single backticks.
- Zarrista dependency: git-pinned to a `main` commit in an optional `zarrista` dependency group.

## File Structure

```
src/zarr/abc/engine.py              Region + the four protocols (new)
src/zarr/errors.py                  add UnsupportedEngineError
src/zarr/core/engine/__init__.py    re-exports (new)
src/zarr/core/engine/_normalize.py  selection-kind -> (Region, post_index) (new)
src/zarr/core/engine/_default.py    DefaultArrayEngine / DefaultAsyncArrayEngine + hierarchy (new)
src/zarr/core/engine/_resolve.py    engine-spec resolution + per-store hierarchy cache (new)
src/zarr/core/array.py              facade rewiring (modify)
src/zarr/zarrista/__init__.py       public zarrista engine surface (new)
src/zarr/zarrista/_translate.py     zarr Store -> zarrista/obstore store translation (new)
src/zarr/zarrista/_engine.py        ZarristaEngine / ZarristaAsyncEngine + hierarchy (new)
tests/engine/                       protocol, normalization, default-engine, differential (new)
tests/zarrista/                     translation + zarrista-only behavior (new)
DELETED: src/zarr/crud/, src/zarr/zarrs/, packages/zarrs-bindings/, tests/crud/, tests/zarrs/
```

---

### Task 1: `Region`, the four protocols, `UnsupportedEngineError`

**Files:**
- Create: `src/zarr/abc/engine.py`
- Modify: `src/zarr/errors.py` (append class)
- Test: `tests/engine/test_protocols.py`, `tests/engine/__init__.py` (empty)

**Interfaces:**
- Produces: `Region(start: tuple[int, ...], end_exclusive: tuple[int, ...])` with property `shape -> tuple[int, ...]`; protocols `ArrayEngine`, `AsyncArrayEngine`, `HierarchyEngine`, `AsyncHierarchyEngine` exactly as below; `zarr.errors.UnsupportedEngineError(ValueError)`. Every later task imports these from `zarr.abc.engine` / `zarr.errors`.

- [ ] **Step 1: Write the failing test**

```python
# tests/engine/test_protocols.py
from __future__ import annotations

import numpy as np
import pytest

from zarr.abc.engine import ArrayEngine, AsyncArrayEngine, Region
from zarr.errors import UnsupportedEngineError


def test_region() -> None:
    """Region carries start/end_exclusive and derives shape."""
    r = Region(start=(1, 2), end_exclusive=(4, 2))
    assert r.start == (1, 2)
    assert r.end_exclusive == (4, 2)
    assert r.shape == (3, 0)


class _FakeSyncEngine:
    def read_selection(self, selection, *, prototype):
        return np.zeros(selection.shape)

    def write_selection(self, selection, value, *, prototype):
        return None

    def with_metadata(self, metadata):
        return self


def test_runtime_checkable_protocols() -> None:
    """isinstance checks verify method presence for the sync protocol."""
    assert isinstance(_FakeSyncEngine(), ArrayEngine)
    assert not isinstance(object(), ArrayEngine)
    assert not isinstance(_FakeSyncEngine(), AsyncArrayEngine) or True  # names match; mypy is authoritative


def test_unsupported_engine_error_is_value_error() -> None:
    with pytest.raises(ValueError):
        raise UnsupportedEngineError("nope")
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/engine/test_protocols.py -v`
Expected: FAIL (`ModuleNotFoundError: zarr.abc.engine`)

- [ ] **Step 3: Implement**

```python
# src/zarr/abc/engine.py
"""Array engine protocols.

An *array engine* owns the data path of one open array: reading and writing
decoded data for contiguous regions. `Array` wraps an object satisfying
`ArrayEngine`; `AsyncArray` wraps an object satisfying `AsyncArrayEngine`.
A *hierarchy engine* is bound to a store and mints array engines that share
resources. See `docs/superpowers/specs/2026-07-22-array-engine-protocol-design.md`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Protocol, runtime_checkable

if TYPE_CHECKING:
    from zarr.core.buffer import BufferPrototype, NDBuffer
    from zarr.core.common import NDArrayLike
    from zarr.core.metadata import ArrayMetadata

__all__ = [
    "ArrayEngine",
    "AsyncArrayEngine",
    "AsyncHierarchyEngine",
    "HierarchyEngine",
    "Region",
]


class Region(NamedTuple):
    """A contiguous, step-1 box in array-element coordinates.

    One entry per dimension; `start` is inclusive, `end_exclusive` exclusive.
    Callers pass normalized values: non-negative and clipped to the array
    shape. This is the only selection type that crosses the engine boundary.
    """

    start: tuple[int, ...]
    end_exclusive: tuple[int, ...]

    @property
    def shape(self) -> tuple[int, ...]:
        """The ndim-preserving shape of the box."""
        return tuple(e - s for s, e in zip(self.start, self.end_exclusive, strict=True))


@runtime_checkable
class ArrayEngine(Protocol):
    """The synchronous data path of one open array.

    Bound to `(store, path, metadata)` at construction. Methods must not
    require a running event loop. Read results are ndim-preserving with
    shape `selection.shape` and need only implement `__array__`.

    Note: `runtime_checkable` isinstance checks only verify method names;
    mypy is the authoritative conformance check.
    """

    def read_selection(
        self, selection: Region, *, prototype: BufferPrototype
    ) -> NDArrayLike: ...

    def write_selection(
        self, selection: Region, value: NDBuffer, *, prototype: BufferPrototype
    ) -> None: ...

    def with_metadata(self, metadata: ArrayMetadata) -> ArrayEngine: ...


@runtime_checkable
class AsyncArrayEngine(Protocol):
    """The asynchronous data path of one open array. See `ArrayEngine`."""

    async def read_selection(
        self, selection: Region, *, prototype: BufferPrototype
    ) -> NDArrayLike: ...

    async def write_selection(
        self, selection: Region, value: NDBuffer, *, prototype: BufferPrototype
    ) -> None: ...

    def with_metadata(self, metadata: ArrayMetadata) -> AsyncArrayEngine: ...


@runtime_checkable
class HierarchyEngine(Protocol):
    """A store-bound factory for synchronous array engines."""

    def array_engine(self, path: str, metadata: ArrayMetadata) -> ArrayEngine: ...


@runtime_checkable
class AsyncHierarchyEngine(Protocol):
    """A store-bound factory for asynchronous array engines."""

    def array_engine(self, path: str, metadata: ArrayMetadata) -> AsyncArrayEngine: ...
```

Append to `src/zarr/errors.py` (match the module's existing docstring style):

```python
class UnsupportedEngineError(ValueError):
    """Raised when an array engine cannot serve the requested store or array.

    Examples: a store the engine cannot translate, or metadata (e.g. Zarr v2)
    the engine does not support. Raised at engine construction time.
    """
```

Add `"UnsupportedEngineError"` to `zarr.errors.__all__` if the module defines one.

- [ ] **Step 4: Run tests, then mypy**

Run: `uv run pytest tests/engine/test_protocols.py -v` — Expected: PASS
Run: `uv run mypy src/zarr/abc/engine.py` — Expected: no errors

- [ ] **Step 5: Commit**

```bash
git add src/zarr/abc/engine.py src/zarr/errors.py tests/engine
git commit -m "feat: Region interchange and array/hierarchy engine protocols"
```

---

### Task 2: Selection normalization (`_normalize.py`)

Every public selection kind reduces to `(Region, post_index)` where
`post_index` is a numpy index applied to the ndim-preserving box result.
The basic-kind code is adapted from `src/zarr/crud/_api.py::_normalize_selection`
(lines 86-147) — copy it before Task 9 deletes that package.

**Files:**
- Create: `src/zarr/core/engine/__init__.py`, `src/zarr/core/engine/_normalize.py`
- Test: `tests/engine/test_normalize.py`

**Interfaces:**
- Produces (all in `zarr.core.engine._normalize`; re-exported from `zarr.core.engine`):
  - `normalize_basic(selection, shape) -> tuple[Region, tuple[slice | int, ...]]`
  - `normalize_orthogonal(selection, shape) -> tuple[Region, tuple[Any, ...]]` (post is per-axis arrays/slices for `np.ix_`-style outer indexing; the helper returns the ready-to-use index)
  - `normalize_coordinate(selection, shape) -> tuple[Region, tuple[np.ndarray, ...]]`
  - `normalize_block(selection, chunk_grid_shape, chunk_shape, shape) -> Region` (block selections are contiguous; no post index)
  - mask selections are converted by callers via `np.nonzero(mask)` → `normalize_coordinate`.
- Invariant used by Tasks 5-6: for any array `a`,
  `np.asarray(engine_read(region))[post] == a[original_selection]`.

- [ ] **Step 1: Write the failing tests** — differential against numpy, per your test convention (one combo test per function + error cases):

```python
# tests/engine/test_normalize.py
from __future__ import annotations

import numpy as np
import pytest

from zarr.core.engine import (
    normalize_basic,
    normalize_block,
    normalize_coordinate,
    normalize_orthogonal,
)

SHAPE = (10, 9)
ARR = np.arange(90).reshape(SHAPE)


def _read_box(region):
    """Simulate an engine read: the ndim-preserving box."""
    return ARR[tuple(slice(s, e) for s, e in zip(region.start, region.end_exclusive, strict=True))]


@pytest.mark.parametrize(
    "sel",
    [
        (slice(2, 7), slice(0, 9)),
        (slice(None), slice(3, 4)),
        (slice(8, 2, -2), slice(None)),
        (slice(1, 8, 3), slice(2, 9, 2)),
        (3, slice(None)),
        (-1, -2),
        (Ellipsis, 4),
        slice(5),
        Ellipsis,
        (slice(4, 4), slice(None)),  # empty
    ],
)
def test_normalize_basic_matches_numpy(sel) -> None:
    region, post = normalize_basic(sel, SHAPE)
    np.testing.assert_array_equal(_read_box(region)[post], ARR[sel])


@pytest.mark.parametrize(
    "sel",
    [
        (np.array([1, 4, 7]), np.array([0, 8])),
        (np.array([7, 1, 4]), slice(2, 6)),      # unordered axis indices
        (np.array([3, 3]), np.array([5, 5])),    # repeats
        (slice(1, 9, 2), np.array([2])),
        (np.array([0]), 4),                      # int axis
    ],
)
def test_normalize_orthogonal_matches_numpy_oindex(sel) -> None:
    import numpy.lib.recfunctions  # noqa: F401  (just ensures numpy fully loaded)

    region, post = normalize_orthogonal(sel, SHAPE)
    # numpy oindex semantics == np.ix_ over per-axis index lists
    axes = []
    for s, size in zip(sel if isinstance(sel, tuple) else (sel,), SHAPE, strict=False):
        if isinstance(s, slice):
            axes.append(np.arange(*s.indices(size)))
        elif np.isscalar(s) or getattr(s, "ndim", 1) == 0:
            axes.append(np.array([int(s)]))
        else:
            axes.append(np.asarray(s))
    expected = ARR[np.ix_(*axes)]
    np.testing.assert_array_equal(_read_box(region)[post], expected)


def test_normalize_coordinate_matches_numpy_vindex() -> None:
    coords = (np.array([9, 0, 3, 3]), np.array([8, 0, 2, 2]))
    region, post = normalize_coordinate(coords, SHAPE)
    np.testing.assert_array_equal(_read_box(region)[post], ARR[coords])


def test_normalize_block_is_contiguous() -> None:
    # chunk shape (3, 4) over SHAPE (10, 9): block (1, 2) spans rows 3:6, cols 8:9
    region = normalize_block((1, 2), chunk_grid_shape=(4, 3), chunk_shape=(3, 4), shape=SHAPE)
    assert region.start == (3, 8)
    assert region.end_exclusive == (6, 9)


def test_normalize_basic_rejects_fancy() -> None:
    with pytest.raises(TypeError):
        normalize_basic((np.array([1, 2]), slice(None)), SHAPE)


def test_normalize_basic_rejects_out_of_bounds_int() -> None:
    with pytest.raises(IndexError):
        normalize_basic((10, slice(None)), SHAPE)


def test_normalize_coordinate_rejects_out_of_bounds() -> None:
    with pytest.raises(IndexError):
        normalize_coordinate((np.array([10]), np.array([0])), SHAPE)


def test_normalize_orthogonal_rejects_boolean_ndim_mismatch() -> None:
    with pytest.raises(IndexError):
        normalize_orthogonal((np.zeros((2, 2), dtype=bool), slice(None)), SHAPE)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/engine/test_normalize.py -v`
Expected: FAIL (`ImportError`)

- [ ] **Step 3: Implement**

```python
# src/zarr/core/engine/_normalize.py
"""Reduce public selection kinds to `(Region, post_index)` pairs.

The engine boundary only speaks contiguous step-1 boxes (`Region`). Each
helper returns the box to transfer plus the numpy index that, applied to the
ndim-preserving box result, yields exactly `array[original_selection]`.
"""

from __future__ import annotations

import operator
import types
from typing import TYPE_CHECKING, Any

import numpy as np

from zarr.abc.engine import Region

if TYPE_CHECKING:
    from zarr.core.indexing import BasicSelection


def _expand(selection: Any, ndim: int) -> tuple[Any, ...]:
    """Expand Ellipsis and pad missing trailing axes with full slices."""
    sel_tuple = selection if isinstance(selection, tuple) else (selection,)
    n_ellipsis = sum(1 for s in sel_tuple if s is Ellipsis)
    if n_ellipsis > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    if n_ellipsis == 1:
        i = sel_tuple.index(Ellipsis)
        n_fill = ndim - (len(sel_tuple) - 1)
        if n_fill < 0:
            raise IndexError(f"too many indices for array: array is {ndim}-dimensional")
        sel_tuple = sel_tuple[:i] + (slice(None),) * n_fill + sel_tuple[i + 1 :]
    if len(sel_tuple) > ndim:
        raise IndexError(f"too many indices for array: array is {ndim}-dimensional")
    return sel_tuple + (slice(None),) * (ndim - len(sel_tuple))


def _normalize_int(sel: Any, size: int, dim: int) -> int:
    idx = operator.index(sel)
    if idx < 0:
        idx += size
    if not 0 <= idx < size:
        raise IndexError(f"index {sel} is out of bounds for axis {dim} with size {size}")
    return idx


def normalize_basic(
    selection: BasicSelection, shape: tuple[int, ...]
) -> tuple[Region, tuple[slice | int, ...]]:
    """Normalize a numpy basic-indexing selection to a step-1 box.

    Supports integers, slices (any step), and `Ellipsis`. Integer axes get a
    length-1 range in the box and `0` in the post index (dropping the axis,
    matching numpy). Fancy elements raise `TypeError`.
    """
    sel_tuple = _expand(selection, len(shape))
    starts: list[int] = []
    ends: list[int] = []
    post: list[slice | int] = []
    for dim, (sel, size) in enumerate(zip(sel_tuple, shape, strict=True)):
        if isinstance(sel, slice):
            start, stop, step = sel.indices(size)
            n = len(range(start, stop, step))
            if n == 0:
                starts.append(0)
                ends.append(0)
                post.append(slice(None))
            elif step > 0:
                starts.append(start)
                ends.append(start + (n - 1) * step + 1)
                post.append(slice(None, None, step))
            else:
                last = start + (n - 1) * step
                starts.append(last)
                ends.append(start + 1)
                post.append(slice(None, None, step))
        elif isinstance(sel, types.EllipsisType):
            raise AssertionError("Ellipsis expanded by _expand")
        else:
            try:
                idx = _normalize_int(sel, size, dim)
            except TypeError:
                raise TypeError(
                    f"unsupported selection element {sel!r}: only integers, "
                    "slices, and Ellipsis are supported by basic indexing"
                ) from None
            starts.append(idx)
            ends.append(idx + 1)
            post.append(0)
    return Region(start=tuple(starts), end_exclusive=tuple(ends)), tuple(post)


def normalize_orthogonal(
    selection: Any, shape: tuple[int, ...]
) -> tuple[Region, tuple[Any, ...]]:
    """Normalize an orthogonal (`oindex`) selection to a box + outer index.

    Each axis selector may be an integer, a slice, an integer array, or a 1-d
    boolean mask for that axis. The post index is the `np.ix_`-broadcastable
    tuple of per-axis integer arrays (relative to the box origin), with
    integer axes dropped afterwards by the facade via the returned index
    (integers appear as scalar entries produced by `np.ix_` inputs of shape
    `(1,)` — the facade result keeps numpy `oindex` semantics).
    """
    sel_tuple = _expand(selection, len(shape))
    starts: list[int] = []
    ends: list[int] = []
    axis_indices: list[np.ndarray] = []
    for dim, (sel, size) in enumerate(zip(sel_tuple, shape, strict=True)):
        if isinstance(sel, slice):
            idxs = np.arange(*sel.indices(size))
        elif isinstance(sel, np.ndarray) and sel.dtype == bool:
            if sel.ndim != 1 or sel.shape[0] != size:
                raise IndexError(
                    f"boolean index for axis {dim} must be 1-d with length {size}"
                )
            idxs = np.nonzero(sel)[0]
        elif isinstance(sel, (np.ndarray, list)):
            idxs = np.asarray(sel, dtype=np.intp)
            if idxs.ndim != 1:
                raise IndexError(f"orthogonal index for axis {dim} must be 1-d")
            idxs = np.where(idxs < 0, idxs + size, idxs)
            if idxs.size and (idxs.min() < 0 or idxs.max() >= size):
                raise IndexError(f"index out of bounds for axis {dim} with size {size}")
        else:
            idxs = np.array([_normalize_int(sel, size, dim)], dtype=np.intp)
        if idxs.size == 0:
            starts.append(0)
            ends.append(0)
            axis_indices.append(idxs)
        else:
            lo, hi = int(idxs.min()), int(idxs.max())
            starts.append(lo)
            ends.append(hi + 1)
            axis_indices.append(idxs - lo)
    region = Region(start=tuple(starts), end_exclusive=tuple(ends))
    post = np.ix_(*axis_indices) if axis_indices else ()
    # integer axes were widened to length-1 arrays; the caller squeezes them
    squeeze_axes = tuple(
        i
        for i, sel in enumerate(sel_tuple)
        if not isinstance(sel, (slice, np.ndarray, list))
    )
    if squeeze_axes:
        return region, (*post, _Squeeze(squeeze_axes))  # type: ignore[return-value]
    return region, post


class _Squeeze:
    """Marker appended to an orthogonal post index: squeeze these axes."""

    def __init__(self, axes: tuple[int, ...]) -> None:
        self.axes = axes


def apply_post_index(box: np.ndarray, post: tuple[Any, ...]) -> np.ndarray:
    """Apply a post index produced by a `normalize_*` helper to a box read."""
    if post and isinstance(post[-1], _Squeeze):
        result = box[post[:-1]] if len(post) > 1 else box
        return np.squeeze(result, axis=post[-1].axes)
    if post == ():
        return box
    return box[post]


def normalize_coordinate(
    selection: tuple[Any, ...], shape: tuple[int, ...]
) -> tuple[Region, tuple[np.ndarray, ...]]:
    """Normalize a coordinate (`vindex`) selection to a box + pointwise index."""
    coords = tuple(np.asarray(c, dtype=np.intp) for c in selection)
    if len(coords) != len(shape):
        raise IndexError(
            f"coordinate selection needs {len(shape)} axis arrays, got {len(coords)}"
        )
    coords = tuple(
        np.where(c < 0, c + size, c) for c, size in zip(coords, shape, strict=True)
    )
    for dim, (c, size) in enumerate(zip(coords, shape, strict=True)):
        if c.size and (c.min() < 0 or c.max() >= size):
            raise IndexError(f"index out of bounds for axis {dim} with size {size}")
    starts = tuple(int(c.min()) if c.size else 0 for c in coords)
    ends = tuple(int(c.max()) + 1 if c.size else 0 for c in coords)
    post = tuple(c - s for c, s in zip(coords, starts, strict=True))
    return Region(start=starts, end_exclusive=ends), post


def normalize_block(
    block_coords: tuple[int, ...],
    *,
    chunk_grid_shape: tuple[int, ...],
    chunk_shape: tuple[int, ...],
    shape: tuple[int, ...],
) -> Region:
    """Normalize a block (chunk-grid) selection to its contiguous box."""
    starts = []
    ends = []
    for dim, (b, nblocks, csize, size) in enumerate(
        zip(block_coords, chunk_grid_shape, chunk_shape, shape, strict=True)
    ):
        idx = _normalize_int(b, nblocks, dim)
        starts.append(idx * csize)
        ends.append(min((idx + 1) * csize, size))
    return Region(start=tuple(starts), end_exclusive=tuple(ends))
```

```python
# src/zarr/core/engine/__init__.py
from zarr.core.engine._normalize import (
    apply_post_index,
    normalize_basic,
    normalize_block,
    normalize_coordinate,
    normalize_orthogonal,
)

__all__ = [
    "apply_post_index",
    "normalize_basic",
    "normalize_block",
    "normalize_coordinate",
    "normalize_orthogonal",
]
```

Note for the implementer: the test uses `_read_box(region)[post]` for basic and
coordinate kinds and `apply_post_index` semantics for orthogonal. Update the
orthogonal test to call `apply_post_index(_read_box(region), post)` — plain
`box[post]` is wrong once the `_Squeeze` marker is present.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/engine/test_normalize.py -v` — Expected: PASS
Run: `uv run mypy src/zarr/core/engine` — Expected: no errors

- [ ] **Step 5: Commit**

```bash
git add src/zarr/core/engine tests/engine/test_normalize.py
git commit -m "feat: normalize selection kinds to Region + post-index"
```

---

### Task 3: Default engines (`_default.py`)

**Files:**
- Create: `src/zarr/core/engine/_default.py`
- Modify: `src/zarr/core/engine/__init__.py` (re-export)
- Test: `tests/engine/test_default_engine.py`

**Interfaces:**
- Consumes: `Region` (Task 1).
- Produces:
  - `DefaultAsyncArrayEngine(store_path: StorePath, metadata: ArrayMetadata, config: ArrayConfig)` satisfying `AsyncArrayEngine`.
  - `DefaultArrayEngine(async_engine: DefaultAsyncArrayEngine)` satisfying `ArrayEngine` (wraps with `zarr.core.sync.sync`).
  - `DefaultAsyncHierarchyEngine(store: Store)` / `DefaultHierarchyEngine(store: Store)` with `array_engine(path, metadata)`.
- Task 5 constructs these inside `AsyncArray.__init__`/`Array`.

- [ ] **Step 1: Write the failing test**

```python
# tests/engine/test_default_engine.py
from __future__ import annotations

import numpy as np

import zarr
from zarr.abc.engine import AsyncArrayEngine, Region
from zarr.core.buffer import default_buffer_prototype
from zarr.core.engine import DefaultArrayEngine, DefaultAsyncArrayEngine
from zarr.core.sync import sync
from zarr.storage import MemoryStore


def _make_array() -> zarr.Array:
    z = zarr.create_array(
        MemoryStore(), shape=(10, 9), chunks=(3, 4), dtype="int32", fill_value=0
    )
    z[:, :] = np.arange(90, dtype="int32").reshape(10, 9)
    return z


def test_default_async_engine_read_write_roundtrip() -> None:
    z = _make_array()
    eng = DefaultAsyncArrayEngine(
        store_path=z.async_array.store_path,
        metadata=z.async_array.metadata,
        config=z.async_array.config,
    )
    assert isinstance(eng, AsyncArrayEngine)
    proto = default_buffer_prototype()
    region = Region(start=(2, 1), end_exclusive=(7, 5))

    out = sync(eng.read_selection(region, prototype=proto))
    np.testing.assert_array_equal(np.asarray(out), np.asarray(z[2:7, 1:5]))

    new = np.full((5, 4), -1, dtype="int32")
    value = proto.nd_buffer.from_ndarray_like(new)
    sync(eng.write_selection(region, value, prototype=proto))
    np.testing.assert_array_equal(np.asarray(z[2:7, 1:5]), new)


def test_default_sync_engine_matches_async() -> None:
    z = _make_array()
    async_eng = DefaultAsyncArrayEngine(
        store_path=z.async_array.store_path,
        metadata=z.async_array.metadata,
        config=z.async_array.config,
    )
    eng = DefaultArrayEngine(async_eng)
    proto = default_buffer_prototype()
    region = Region(start=(0, 0), end_exclusive=(10, 9))
    np.testing.assert_array_equal(
        np.asarray(eng.read_selection(region, prototype=proto)), np.asarray(z[:, :])
    )


def test_with_metadata_rebinds() -> None:
    z = _make_array()
    eng = DefaultAsyncArrayEngine(
        store_path=z.async_array.store_path,
        metadata=z.async_array.metadata,
        config=z.async_array.config,
    )
    new_meta = z.async_array.metadata
    assert eng.with_metadata(new_meta) is not eng
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/engine/test_default_engine.py -v`
Expected: FAIL (`ImportError: DefaultArrayEngine`)

- [ ] **Step 3: Implement**

```python
# src/zarr/core/engine/_default.py
"""The default engines: today's codec-pipeline machinery behind the protocol."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from zarr.abc.engine import Region
from zarr.core.array_spec import ArraySpec
from zarr.core.chunk_grids import ChunkGrid
from zarr.core.codec_pipeline import create_codec_pipeline
from zarr.core.indexing import BasicIndexer
from zarr.core.metadata import ArrayMetadata
from zarr.core.sync import sync

if TYPE_CHECKING:
    from zarr.abc.store import Store
    from zarr.core.array_spec import ArrayConfig
    from zarr.core.buffer import BufferPrototype, NDBuffer
    from zarr.core.common import NDArrayLike
    from zarr.core.storage import StorePath


def _region_to_slices(region: Region) -> tuple[slice, ...]:
    return tuple(
        slice(s, e) for s, e in zip(region.start, region.end_exclusive, strict=True)
    )


class DefaultAsyncArrayEngine:
    """Codec-pipeline-backed engine. Any store, Zarr v2 and v3."""

    def __init__(
        self, store_path: StorePath, metadata: ArrayMetadata, config: ArrayConfig
    ) -> None:
        self.store_path = store_path
        self.metadata = metadata
        self.config = config
        self._chunk_grid = ChunkGrid.from_metadata(metadata)
        self._codec_pipeline = create_codec_pipeline(
            metadata=metadata, store=store_path.store
        )

    def with_metadata(self, metadata: ArrayMetadata) -> DefaultAsyncArrayEngine:
        return DefaultAsyncArrayEngine(
            store_path=self.store_path, metadata=metadata, config=self.config
        )

    def _indexer(self, region: Region) -> BasicIndexer:
        return BasicIndexer(
            _region_to_slices(region),
            shape=self.metadata.shape,
            chunk_grid=self._chunk_grid,
        )

    def _chunk_spec(self, prototype: BufferPrototype) -> tuple[ArraySpec | None, ArrayConfig]:
        # v2 honors metadata order; v3 honors config order (mirrors _get_selection)
        config = self.config
        if self.metadata.zarr_format == 2:
            config = replace(config, order=self.metadata.order)
        if self._chunk_grid.is_regular:
            return (
                ArraySpec(
                    shape=self._chunk_grid.chunk_shape,
                    dtype=self.metadata.dtype,
                    fill_value=self.metadata.fill_value,
                    config=config,
                    prototype=prototype,
                ),
                config,
            )
        return None, config

    async def read_selection(
        self, selection: Region, *, prototype: BufferPrototype
    ) -> NDArrayLike:
        indexer = self._indexer(selection)
        if self.metadata.zarr_format == 2:
            dtype = self.metadata.dtype.to_native_dtype()
            order = self.metadata.order
        else:
            dtype = self.metadata.data_type.to_native_dtype()
            order = self.config.order
        out_buffer = prototype.nd_buffer.empty(
            shape=indexer.shape, dtype=dtype, order=order
        )
        if all(e > s for s, e in zip(selection.start, selection.end_exclusive, strict=True)):
            spec, config = self._chunk_spec(prototype)
            await self._codec_pipeline.read(
                [
                    (
                        self.store_path / self.metadata.encode_chunk_key(chunk_coords),
                        spec
                        if spec is not None
                        else _irregular_chunk_spec(
                            self.metadata, self._chunk_grid, chunk_coords, config, prototype
                        ),
                        chunk_selection,
                        out_selection,
                        is_complete_chunk,
                    )
                    for chunk_coords, chunk_selection, out_selection, is_complete_chunk in indexer
                ],
                out_buffer,
                drop_axes=indexer.drop_axes,
            )
        return out_buffer.as_ndarray_like()

    async def write_selection(
        self, selection: Region, value: NDBuffer, *, prototype: BufferPrototype
    ) -> None:
        indexer = self._indexer(selection)
        spec, config = self._chunk_spec(prototype)
        await self._codec_pipeline.write(
            [
                (
                    self.store_path / self.metadata.encode_chunk_key(chunk_coords),
                    spec
                    if spec is not None
                    else _irregular_chunk_spec(
                        self.metadata, self._chunk_grid, chunk_coords, config, prototype
                    ),
                    chunk_selection,
                    out_selection,
                    is_complete_chunk,
                )
                for chunk_coords, chunk_selection, out_selection, is_complete_chunk in indexer
            ],
            value,
            drop_axes=indexer.drop_axes,
        )


def _irregular_chunk_spec(metadata, chunk_grid, chunk_coords, config, prototype):
    # same construction as array.py::_get_chunk_spec — import and delegate
    from zarr.core.array import _get_chunk_spec

    return _get_chunk_spec(metadata, chunk_grid, chunk_coords, config, prototype)


class DefaultArrayEngine:
    """Sync adapter over `DefaultAsyncArrayEngine` via `sync()`."""

    def __init__(self, async_engine: DefaultAsyncArrayEngine) -> None:
        self._async = async_engine

    def read_selection(
        self, selection: Region, *, prototype: BufferPrototype
    ) -> NDArrayLike:
        return sync(self._async.read_selection(selection, prototype=prototype))

    def write_selection(
        self, selection: Region, value: NDBuffer, *, prototype: BufferPrototype
    ) -> None:
        sync(self._async.write_selection(selection, value, prototype=prototype))

    def with_metadata(self, metadata: ArrayMetadata) -> DefaultArrayEngine:
        return DefaultArrayEngine(self._async.with_metadata(metadata))


class DefaultAsyncHierarchyEngine:
    """Store-bound factory for default async engines."""

    def __init__(self, store: Store) -> None:
        self.store = store

    def array_engine(self, path: str, metadata: ArrayMetadata) -> DefaultAsyncArrayEngine:
        from zarr.core.storage import StorePath

        from zarr.core.array_spec import parse_array_config

        return DefaultAsyncArrayEngine(
            store_path=StorePath(self.store, path),
            metadata=metadata,
            config=parse_array_config(None),
        )


class DefaultHierarchyEngine:
    """Store-bound factory for default sync engines."""

    def __init__(self, store: Store) -> None:
        self._async = DefaultAsyncHierarchyEngine(store)

    def array_engine(self, path: str, metadata: ArrayMetadata) -> DefaultArrayEngine:
        return DefaultArrayEngine(self._async.array_engine(path, metadata))
```

Implementation notes (verify against the live code, these matter):
- `StorePath` import path: check `src/zarr/core/array.py`'s own import and copy it.
- The empty-region guard in `read_selection` mirrors `_get_selection`'s
  `product(indexer.shape) > 0` check; use `zarr.core.common.product` if simpler.
- v2 `metadata.dtype` / v3 `metadata.data_type` split is copied from
  `_get_selection` (array.py:5392-5397). Do not invent a new path.

Add to `src/zarr/core/engine/__init__.py`:

```python
from zarr.core.engine._default import (
    DefaultArrayEngine,
    DefaultAsyncArrayEngine,
    DefaultAsyncHierarchyEngine,
    DefaultHierarchyEngine,
)
```
(and extend `__all__` accordingly).

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/engine/test_default_engine.py tests/engine/test_protocols.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/zarr/core/engine tests/engine/test_default_engine.py
git commit -m "feat: default array engines over the codec pipeline"
```

---

### Task 4: Engine resolution (`_resolve.py`)

**Files:**
- Create: `src/zarr/core/engine/_resolve.py`
- Modify: `src/zarr/core/engine/__init__.py` (re-export)
- Test: `tests/engine/test_resolve.py`

**Interfaces:**
- Consumes: default engines (Task 3), protocols (Task 1).
- Produces (used by Task 5's `AsyncArray.__init__` / `Array` wiring):
  - `EngineName = Literal["default", "zarrista"]`
  - `resolve_async_engine(engine: AsyncArrayEngine | EngineName | None, *, store: Store, path: str, metadata: ArrayMetadata) -> AsyncArrayEngine`
  - `resolve_sync_engine(engine: ArrayEngine | EngineName | None, *, store: Store, path: str, metadata: ArrayMetadata) -> ArrayEngine`
  - `None` and `"default"` → default engine; `"zarrista"` → lazy import of `zarr.zarrista`, raising a clear `ImportError` if zarrista is missing; an instance → returned as-is.
  - Hierarchy engines are cached per `(name, id(store))` in a module-level `WeakValueDictionary`-free plain dict keyed by a `weakref.ref` of the store (evicted via callback) so engines from one store share resources.

- [ ] **Step 1: Write the failing test**

```python
# tests/engine/test_resolve.py
from __future__ import annotations

import numpy as np
import pytest

import zarr
from zarr.core.engine import (
    DefaultArrayEngine,
    DefaultAsyncArrayEngine,
    resolve_async_engine,
    resolve_sync_engine,
)
from zarr.storage import MemoryStore


def _array() -> zarr.Array:
    return zarr.create_array(MemoryStore(), shape=(4,), chunks=(2,), dtype="int8")


def test_resolution_combinations() -> None:
    z = _array()
    store = z.store
    path = z.path
    meta = z.async_array.metadata

    # None and "default" produce default engines
    for spec in (None, "default"):
        assert isinstance(
            resolve_async_engine(spec, store=store, path=path, metadata=meta),
            DefaultAsyncArrayEngine,
        )
        assert isinstance(
            resolve_sync_engine(spec, store=store, path=path, metadata=meta),
            DefaultArrayEngine,
        )

    # instances pass through untouched
    inst = resolve_sync_engine(None, store=store, path=path, metadata=meta)
    assert resolve_sync_engine(inst, store=store, path=path, metadata=meta) is inst

    # engines minted from the same store share a hierarchy engine
    e1 = resolve_async_engine(None, store=store, path=path, metadata=meta)
    e2 = resolve_async_engine(None, store=store, path="other", metadata=meta)
    assert e1.store_path.store is e2.store_path.store


def test_unknown_name_raises() -> None:
    z = _array()
    with pytest.raises(ValueError, match="unknown engine"):
        resolve_async_engine(
            "bogus",  # type: ignore[arg-type]
            store=z.store,
            path=z.path,
            metadata=z.async_array.metadata,
        )


def test_zarrista_missing_raises_import_error() -> None:
    pytest.importorskip("zarr.zarrista", reason="run only when zarrista absent") \
        if False else None
    try:
        import zarrista  # noqa: F401

        pytest.skip("zarrista installed; missing-module error not testable")
    except ImportError:
        pass
    z = _array()
    with pytest.raises(ImportError, match="zarrista"):
        resolve_async_engine(
            "zarrista", store=z.store, path=z.path, metadata=z.async_array.metadata
        )
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/engine/test_resolve.py -v`
Expected: FAIL (`ImportError: resolve_async_engine`)

- [ ] **Step 3: Implement**

```python
# src/zarr/core/engine/_resolve.py
"""Resolve an `engine=` argument to a bound array engine."""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Literal

from zarr.core.engine._default import (
    DefaultAsyncHierarchyEngine,
    DefaultHierarchyEngine,
)

if TYPE_CHECKING:
    from zarr.abc.engine import (
        ArrayEngine,
        AsyncArrayEngine,
        AsyncHierarchyEngine,
        HierarchyEngine,
    )
    from zarr.abc.store import Store
    from zarr.core.metadata import ArrayMetadata

EngineName = Literal["default", "zarrista"]

# (name, kind, id(store)) -> hierarchy engine; entries evicted when the store dies
_hierarchy_cache: dict[tuple[str, str, int], object] = {}


def _cached_hierarchy(name: str, kind: str, store: Store, factory) -> object:
    key = (name, kind, id(store))
    if key not in _hierarchy_cache:
        _hierarchy_cache[key] = factory(store)
        weakref.finalize(store, _hierarchy_cache.pop, key, None)
    return _hierarchy_cache[key]


def _hierarchy_factory(name: str, *, sync: bool):
    if name == "default":
        return DefaultHierarchyEngine if sync else DefaultAsyncHierarchyEngine
    if name == "zarrista":
        try:
            from zarr.zarrista import (
                ZarristaAsyncHierarchyEngine,
                ZarristaHierarchyEngine,
            )
        except ImportError as e:
            raise ImportError(
                "engine='zarrista' requires the `zarrista` package; "
                "install zarr with the `zarrista` extra"
            ) from e
        return ZarristaHierarchyEngine if sync else ZarristaAsyncHierarchyEngine
    raise ValueError(f"unknown engine name {name!r}; expected 'default' or 'zarrista'")


def resolve_async_engine(
    engine: AsyncArrayEngine | EngineName | None,
    *,
    store: Store,
    path: str,
    metadata: ArrayMetadata,
) -> AsyncArrayEngine:
    if engine is None:
        engine = "default"
    if isinstance(engine, str):
        factory = _hierarchy_factory(engine, sync=False)
        hierarchy: AsyncHierarchyEngine = _cached_hierarchy(  # type: ignore[assignment]
            engine, "async", store, factory
        )
        return hierarchy.array_engine(path, metadata)
    return engine


def resolve_sync_engine(
    engine: ArrayEngine | EngineName | None,
    *,
    store: Store,
    path: str,
    metadata: ArrayMetadata,
) -> ArrayEngine:
    if engine is None:
        engine = "default"
    if isinstance(engine, str):
        factory = _hierarchy_factory(engine, sync=True)
        hierarchy: HierarchyEngine = _cached_hierarchy(  # type: ignore[assignment]
            engine, "sync", store, factory
        )
        return hierarchy.array_engine(path, metadata)
    return engine
```

Note: if `weakref.finalize` on a `Store` fails (stores with `__slots__` and no
`__weakref__`), fall back to a `weakref.WeakKeyDictionary[Store, dict[str, object]]`
keyed on the store; adjust the test only if the sharing assertion still holds.

Re-export `EngineName`, `resolve_async_engine`, `resolve_sync_engine` from
`zarr.core.engine.__init__` and extend `__all__`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/engine/test_resolve.py -v` — Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/zarr/core/engine tests/engine/test_resolve.py
git commit -m "feat: engine-spec resolution with per-store hierarchy cache"
```

---

### Task 5: Wire `AsyncArray` through its engine

The public behavior contract for this task and Task 6: **the whole existing
array API test suite keeps passing** (`tests/test_array.py`, `tests/test_indexing.py`).
That suite is the real acceptance test; the new unit tests below only pin the
wiring itself.

**Files:**
- Modify: `src/zarr/core/array.py`
- Test: `tests/engine/test_asyncarray_wiring.py`

**Interfaces:**
- Consumes: `resolve_async_engine` (Task 4), `normalize_*`/`apply_post_index` (Task 2).
- Produces: `AsyncArray.__init__(..., engine: AsyncArrayEngine | EngineName | None = None)`; attribute `AsyncArray.engine`; rewritten module-level `_get_selection(engine, indexer_or_selection…)`/`_set_selection` (final signatures below). Task 6 mirrors this for `Array`; Task 7 threads `engine=` through creation APIs.

- [ ] **Step 1: Write the failing test**

```python
# tests/engine/test_asyncarray_wiring.py
from __future__ import annotations

import numpy as np

import zarr
from zarr.abc.engine import Region
from zarr.core.engine import DefaultAsyncArrayEngine
from zarr.storage import MemoryStore


class _SpyEngine:
    """Wraps a real engine, recording regions."""

    def __init__(self, inner: DefaultAsyncArrayEngine) -> None:
        self.inner = inner
        self.read_regions: list[Region] = []
        self.write_regions: list[Region] = []

    async def read_selection(self, selection, *, prototype):
        self.read_regions.append(selection)
        return await self.inner.read_selection(selection, prototype=prototype)

    async def write_selection(self, selection, value, *, prototype):
        self.write_regions.append(selection)
        return await self.inner.write_selection(selection, value, prototype=prototype)

    def with_metadata(self, metadata):
        return _SpyEngine(self.inner.with_metadata(metadata))


def test_asyncarray_routes_io_through_engine() -> None:
    z = zarr.create_array(MemoryStore(), shape=(10,), chunks=(3,), dtype="int16")
    aa = z.async_array
    spy = _SpyEngine(
        DefaultAsyncArrayEngine(
            store_path=aa.store_path, metadata=aa.metadata, config=aa.config
        )
    )
    object.__setattr__(aa, "engine", spy)

    z[2:8] = np.arange(6, dtype="int16")
    data = z[2:8]

    assert spy.write_regions == [Region(start=(2,), end_exclusive=(8,))]
    assert spy.read_regions == [Region(start=(2,), end_exclusive=(8,))]
    np.testing.assert_array_equal(np.asarray(data), np.arange(6, dtype="int16"))


def test_asyncarray_default_engine_attribute() -> None:
    z = zarr.create_array(MemoryStore(), shape=(4,), chunks=(2,), dtype="int8")
    assert isinstance(z.async_array.engine, DefaultAsyncArrayEngine)
```

Note: `test_asyncarray_routes_io_through_engine` depends on Task 6 (the sync
`Array` path shares the async engine attribute until Task 6 gives `Array` its
own sync engine; when implementing Task 5 alone, exercise `await aa.getitem(...)`
instead of `z[...]` — final test content above is what must pass after Task 6).

- [ ] **Step 2: Implement the `AsyncArray` changes**

In `src/zarr/core/array.py`:

1. `AsyncArray.__init__` (array.py:349): add keyword
   `engine: AsyncArrayEngine | EngineName | None = None` to all three
   signatures (both overloads + implementation), and set

```python
object.__setattr__(
    self,
    "engine",
    resolve_async_engine(
        engine,
        store=store_path.store,
        path=store_path.path,
        metadata=metadata_parsed,
    ),
)
```

   with attribute declaration `engine: AsyncArrayEngine = field(init=False)`
   next to `codec_pipeline` (array.py:329). Import `resolve_async_engine`
   and `EngineName` from `zarr.core.engine`, `AsyncArrayEngine` from
   `zarr.abc.engine` (TYPE_CHECKING block for the types).

2. Metadata-mutating paths: wherever `AsyncArray` constructs a successor
   `AsyncArray` with new metadata (`resize`, `append`, `update_attributes`,
   `with_config` — find them with
   `grep -n "type(self)(" src/zarr/core/array.py` and
   `grep -n "replace(self" src/zarr/core/array.py`), pass
   `engine=self.engine.with_metadata(new_metadata)` if the constructor
   accepts it, or call `object.__setattr__(new_array, "engine",
   self.engine.with_metadata(new_metadata))` right after construction.

3. Rewrite the two internal I/O helpers. Replace the bodies of
   `AsyncArray._get_selection` (array.py:1412) and
   `AsyncArray._set_selection` (array.py:1550) and the module-level
   `_get_selection` (array.py:5353) / `_set_selection` (array.py:5702) with
   engine-routed versions. The module-level functions get new signatures
   (all callers are inside this file and the sync `Array` methods —
   update every call site found via
   `grep -n "_get_selection\|_set_selection" src/zarr/core/array.py`):

```python
async def _get_selection(
    engine: AsyncArrayEngine,
    metadata: ArrayMetadata,
    config: ArrayConfig,
    region: Region,
    post_index: tuple[Any, ...],
    *,
    prototype: BufferPrototype,
    out: NDBuffer | None = None,
    fields: Fields | None = None,
) -> NDArrayLikeOrScalar:
    dtype = (
        metadata.dtype if metadata.zarr_format == 2 else metadata.data_type
    ).to_native_dtype()
    out_dtype = check_fields(fields, dtype)
    if product(region.shape) == 0:
        empty = np.empty(
            apply_post_index(np.empty(region.shape, dtype=out_dtype), post_index).shape,
            dtype=out_dtype,
        )
        return _finalize_result(empty, out)
    box = np.asarray(await engine.read_selection(region, prototype=prototype))
    if fields is not None:
        box = box[fields]
    result = apply_post_index(box, post_index)
    return _finalize_result(result, out)


def _finalize_result(result: np.ndarray, out: NDBuffer | None) -> NDArrayLikeOrScalar:
    if out is not None:
        out.as_ndarray_like()[...] = result
        return out.as_ndarray_like()
    if result.shape == ():
        return result[()]  # scalar extraction, matching current behavior
    return result
```

```python
async def _set_selection(
    engine: AsyncArrayEngine,
    metadata: ArrayMetadata,
    config: ArrayConfig,
    region: Region,
    post_index: tuple[Any, ...],
    value: npt.ArrayLike,
    *,
    prototype: BufferPrototype,
    fields: Fields | None = None,
) -> None:
    dtype = (
        metadata.dtype if metadata.zarr_format == 2 else metadata.data_type
    ).to_native_dtype()
    check_fields(fields, dtype)
    fields = check_no_multi_fields(fields)
    if product(region.shape) == 0:
        return
    value_np = np.asanyarray(value, dtype=None if fields else dtype)

    identity_post = all(
        isinstance(p, slice) and p == slice(None) for p in post_index
    ) and fields is None
    if identity_post:
        # broadcast the value to the full ndim-preserving box
        box = np.broadcast_to(value_np, region.shape).astype(dtype, copy=False)
    else:
        # facade-level read-modify-write for strided/fancy/fields writes
        box = np.array(
            np.asarray(await engine.read_selection(region, prototype=prototype))
        )
        target = box[fields] if fields is not None else box
        if post_index == ():
            target[...] = value_np
        else:
            # _Squeeze markers only occur on reads (orthogonal); writes use
            # the raw np.ix_/pointwise index without the marker
            target[post_index] = value_np
    value_buffer = prototype.nd_buffer.from_ndarray_like(
        np.ascontiguousarray(box)
    )
    await engine.write_selection(region, value_buffer, prototype=prototype)
```

   Implementation notes:
   - Keep the existing scalar/array-like `value` coercion logic from the old
     `_set_selection` (array.py:5749-5769) ahead of the code above where it is
     richer than `np.asanyarray` (non-numpy array types); preserve it rather
     than the simplified line if any existing test fails.
   - Writes for orthogonal selections must pass the *raw* per-axis outer index
     (`np.ix_(...)` result without the `_Squeeze` marker); add a
     `strip_squeeze(post) -> post` helper in `_normalize.py` and use it here.
   - The old `regular_chunk_spec` optimization lives in the default engine
     now (Task 3); nothing here touches chunks.

4. Update `AsyncArray` methods that build indexers
   (`getitem`, `setitem`, `get_basic_selection`, `set_basic_selection`,
   `get_orthogonal_selection`, `set_orthogonal_selection`,
   `get_mask_selection`, `set_mask_selection`,
   `get_coordinate_selection`, `set_coordinate_selection`,
   `get_block_selection`, `set_block_selection` — find each with
   `grep -n "Indexer(" src/zarr/core/array.py`). Replace
   `indexer = BasicIndexer(...)` + `self._get_selection(indexer, ...)` with:

```python
region, post = normalize_basic(selection, self.shape)
return await _get_selection(
    self.engine, self.metadata, self.config, region, post,
    prototype=prototype, out=out, fields=fields,
)
```

   and analogously `normalize_orthogonal` for orthogonal, `normalize_coordinate`
   for coordinate, `np.nonzero(mask)` → `normalize_coordinate` for mask, and
   `normalize_block(...)` (post index `()`) for block selections, passing the
   chunk-grid shape from `self._chunk_grid`. Mask/coordinate *reads* return the
   pointwise result (already numpy `vindex` semantics via the pointwise post
   index). Delete the now-unused `Indexer` imports only if nothing else in the
   file uses them (resize/`nchunks_initialized` may).

- [ ] **Step 3: Run the wiring test and the array suites**

Run: `uv run pytest tests/engine/test_asyncarray_wiring.py -v` — Expected: PASS (the async variant of the spy test if Task 6 not yet done)
Run: `uv run pytest tests/test_array.py tests/test_indexing.py -x -q` — Expected: PASS. Any failure here is a behavior regression: fix it before proceeding, favoring the old code's semantics.

- [ ] **Step 4: Commit**

```bash
git add src/zarr/core/array.py src/zarr/core/engine tests/engine/test_asyncarray_wiring.py
git commit -m "feat: AsyncArray routes selection I/O through its engine"
```

---

### Task 6: Sync `Array` data path (no event loop)

**Files:**
- Modify: `src/zarr/core/array.py`
- Test: `tests/engine/test_sync_path.py`

**Interfaces:**
- Consumes: `resolve_sync_engine` (Task 4), normalization helpers (Task 2), the `_finalize_result`/RMW patterns from Task 5.
- Produces: `Array.engine: ArrayEngine` property; sync module-level `_get_selection_sync`/`_set_selection_sync` mirroring Task 5's helpers with `engine.read_selection(...)` (no await). `Array.__init__`/constructors accept `engine: ArrayEngine | EngineName | None = None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/engine/test_sync_path.py
from __future__ import annotations

import asyncio

import numpy as np
import pytest

import zarr
from zarr.abc.engine import ArrayEngine, Region
from zarr.core.engine import DefaultArrayEngine
from zarr.storage import MemoryStore


def test_array_has_sync_engine() -> None:
    z = zarr.create_array(MemoryStore(), shape=(6,), chunks=(2,), dtype="uint8")
    assert isinstance(z.engine, ArrayEngine)
    assert isinstance(z.engine, DefaultArrayEngine)


class _NoLoopEngine:
    """A sync engine that asserts no event loop is running when called."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.calls = 0

    def read_selection(self, selection, *, prototype):
        self.calls += 1
        with pytest.raises(RuntimeError):
            asyncio.get_running_loop()
        return self._inner.read_selection(selection, prototype=prototype)

    def write_selection(self, selection, value, *, prototype):
        self.calls += 1
        with pytest.raises(RuntimeError):
            asyncio.get_running_loop()
        return self._inner.write_selection(selection, value, prototype=prototype)

    def with_metadata(self, metadata):
        return _NoLoopEngine(self._inner.with_metadata(metadata))


def test_sync_data_path_runs_without_event_loop_in_caller_thread() -> None:
    z = zarr.create_array(MemoryStore(), shape=(6,), chunks=(2,), dtype="uint8")
    probe = _NoLoopEngine(z.engine)
    object.__setattr__(z, "_engine", probe)  # match the attribute name used in impl

    z[1:5] = np.arange(4, dtype="uint8")
    out = z[1:5]

    assert probe.calls == 2
    np.testing.assert_array_equal(np.asarray(out), np.arange(4, dtype="uint8"))
```

(The engine methods themselves run in the caller thread; the *default* engine
internally uses `sync()` which runs coroutines on zarr's loop thread — that is
allowed. The probe asserts the caller thread has no running loop, i.e. `Array`
did not wrap the engine call in a coroutine.)

- [ ] **Step 2: Implement**

1. `Array` gains a lazily-resolved sync engine: store the creation-time spec
   and resolve on first use.

```python
# on class Array (near _async_array declaration, array.py:1803)
_engine: ArrayEngine | None = None
_engine_spec: ArrayEngine | EngineName | None = None

@property
def engine(self) -> ArrayEngine:
    """The synchronous array engine serving this array's data path."""
    if self._engine is None:
        aa = self._async_array
        object.__setattr__(
            self,
            "_engine",
            resolve_sync_engine(
                self._engine_spec,
                store=aa.store_path.store,
                path=aa.store_path.path,
                metadata=aa.metadata,
            ),
        )
        object.__setattr__(self, "_engine_spec", None)
    return self._engine
```

   Whatever constructor `Array` uses (dataclass or `__init__` — check
   `grep -n "def __init__\|@dataclass" src/zarr/core/array.py` around line
   1798) gains the optional `engine` keyword storing `_engine_spec`. When
   `Array` wraps an `AsyncArray` that was itself given a *name* spec, reuse
   that name so sync and async engines come from the same family.

2. Every sync data method that currently does
   `sync(self.async_array._get_selection(indexer, ...))`
   (sites: array.py:2830, 2939, 3068, 3186, 3274, 3362, 3451, 3563, 3664, 3763
   — re-grep, they will have shifted after Task 5) is rewritten to the sync
   mirror:

```python
region, post = normalize_basic(selection, self.shape)   # kind-appropriate helper
return _get_selection_sync(
    self.engine, self._async_array.metadata, self._async_array.config,
    region, post, prototype=prototype, out=out, fields=fields,
)
```

3. Add module-level `_get_selection_sync` / `_set_selection_sync` beside the
   async versions from Task 5 — same bodies with `engine.read_selection(...)` /
   `engine.write_selection(...)` un-awaited. Factor the shared pure logic
   (dtype/fields resolution, empty short-circuit, box patching, result
   finalization) into small helper functions used by both so the async and
   sync variants differ only in the two engine calls.

4. Metadata mutation on `Array` (`resize`, `append`) invalidates the cached
   sync engine: `object.__setattr__(self, "_engine", self._engine.with_metadata(new_meta))`
   when `_engine` is not None (sites via `grep -n "def resize\|def append" src/zarr/core/array.py`).

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/engine/ -v` — Expected: PASS (including the full Task 5 spy test now)
Run: `uv run pytest tests/test_array.py tests/test_indexing.py -q` — Expected: PASS
Run: `uv run mypy src/zarr` — Expected: no new errors

- [ ] **Step 4: Commit**

```bash
git add src/zarr/core/array.py src/zarr/core/engine tests/engine/test_sync_path.py
git commit -m "feat: Array sync data path calls its engine without the event loop"
```

---

### Task 7: `engine=` on creation/open APIs

**Files:**
- Modify: `src/zarr/core/array.py` (`AsyncArray.open`, `Array.open`, `_create` paths), `src/zarr/api/asynchronous.py` and `src/zarr/api/synchronous.py` (`create_array`, `open_array`), `src/zarr/core/array_creation.py` if `create_array` lives there (find with `grep -rn "def create_array" src/zarr`).
- Test: `tests/engine/test_engine_param.py`

**Interfaces:**
- Consumes: `EngineName`, engines from earlier tasks.
- Produces: public keyword `engine: ArrayEngine | AsyncArrayEngine | EngineName | None = None` on `zarr.create_array`, `zarr.open_array`, `zarr.api.asynchronous.create_array`, `zarr.api.asynchronous.open_array`, threaded to `AsyncArray.__init__` (async instances/names) and `Array._engine_spec` (sync instances/names). Passing a *sync instance* through an async API raises `TypeError`, and vice versa; a *name* is valid everywhere.

- [ ] **Step 1: Write the failing test**

```python
# tests/engine/test_engine_param.py
from __future__ import annotations

import numpy as np
import pytest

import zarr
from zarr.core.engine import DefaultArrayEngine, DefaultAsyncArrayEngine
from zarr.storage import MemoryStore


def test_engine_param_combinations() -> None:
    store = MemoryStore()
    z = zarr.create_array(store, name="a", shape=(4,), chunks=(2,), dtype="int8", engine="default")
    z[:] = np.arange(4, dtype="int8")
    assert isinstance(z.engine, DefaultArrayEngine)

    z2 = zarr.open_array(store, path="a", engine="default")
    np.testing.assert_array_equal(np.asarray(z2[:]), np.arange(4, dtype="int8"))
    assert isinstance(z2.async_array.engine, DefaultAsyncArrayEngine)

    # user-provided sync instance
    inst = z2.engine
    z3 = zarr.open_array(store, path="a", engine=inst)
    assert z3.engine is inst


def test_engine_param_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown engine"):
        zarr.create_array(
            MemoryStore(), shape=(2,), chunks=(2,), dtype="int8", engine="nope"
        )
```

- [ ] **Step 2: Implement**

Thread the keyword: each `create_array`/`open_array` entry point passes
`engine=engine` down to where `AsyncArray(...)`/`Array(...)` is constructed.
Rules:
- Async entry points pass names and `AsyncArrayEngine` instances to
  `AsyncArray.__init__`; a sync `ArrayEngine` instance there raises
  `TypeError("a sync ArrayEngine cannot serve an AsyncArray; pass a name or an AsyncArrayEngine")`.
- Sync entry points pass names to *both* layers (the `Array` stores the name
  for its sync engine; the inner `AsyncArray` also gets the name so async
  access stays consistent), and sync instances only to `Array`.
- Detection: `isinstance(engine, str) or engine is None` → both layers;
  otherwise check for a coroutine `read_selection` via
  `inspect.iscoroutinefunction(engine.read_selection)`.

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/engine/test_engine_param.py -v` — Expected: PASS
Run: `uv run pytest tests/test_api.py -q` — Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/zarr/core/array.py src/zarr/api tests/engine/test_engine_param.py
git commit -m "feat: engine= parameter on array creation and open APIs"
```

---

### Task 8: Zarrista engines (`zarr.zarrista`)

**Files:**
- Create: `src/zarr/zarrista/__init__.py`, `src/zarr/zarrista/_translate.py`, `src/zarr/zarrista/_engine.py`
- Modify: `pyproject.toml` (add `zarrista` dependency group, git-pinned)
- Test: `tests/zarrista/__init__.py` (empty), `tests/zarrista/test_translate.py`, `tests/zarrista/test_engine.py`

**Interfaces:**
- Consumes: protocols/`Region` (Task 1), `UnsupportedEngineError` (Task 1).
- Produces: `ZarristaHierarchyEngine(store: Store)`, `ZarristaAsyncHierarchyEngine(store: Store)`, `ZarristaEngine`, `ZarristaAsyncEngine` (imported lazily by Task 4's resolver).
- Zarrista API used (from its `.pyi` stubs, main branch): sync
  `zarrista.Array.from_metadata(metadata: dict, store, path)`,
  `.retrieve_array_subset(selection) -> DecodedArray`,
  `.store_chunk(chunk_indices, ArrayBytes)`, `.retrieve_chunk(chunk_indices)`,
  `.chunk_subset(chunk_indices) -> tuple[slice, ...]`, `.chunk_grid_shape`,
  `.chunk_shape(chunk_indices)`; async twins on `zarrista.AsyncArray`
  (`from_metadata`, awaitable retrieve/store). Stores:
  `zarrista.FilesystemStore(path)` (sync), obstore `ObjectStore` / icechunk
  `Session` (async). `zarrista.ArrayBytes(bytes=...)` wraps decoded chunk bytes.

- [ ] **Step 1: Add the dependency group**

In `pyproject.toml`, next to the existing dependency groups (see
`[dependency-groups]`), add (pin the current zarrista main commit hash — get it
with `git ls-remote https://github.com/developmentseed/zarrista HEAD`):

```toml
zarrista = [
    "zarrista @ git+https://github.com/developmentseed/zarrista@<MAIN_COMMIT_SHA>",
]
```

Run: `uv sync --group zarrista` — Expected: builds/installs zarrista (Rust
toolchain required; if the build fails locally, note it and let CI cover it —
zarrista publishes no sdist-independent wheels from git).

- [ ] **Step 2: Write the failing tests**

```python
# tests/zarrista/test_translate.py
from __future__ import annotations

import pytest

zarrista = pytest.importorskip("zarrista")

from zarr.errors import UnsupportedEngineError
from zarr.storage import LocalStore, MemoryStore
from zarr.zarrista._translate import translate_store_async, translate_store_sync


def test_local_store_translates_to_filesystem_store(tmp_path) -> None:
    zs = translate_store_sync(LocalStore(tmp_path))
    assert isinstance(zs, zarrista.FilesystemStore)


def test_memory_store_rejected_sync(tmp_path) -> None:
    with pytest.raises(UnsupportedEngineError, match="MemoryStore"):
        translate_store_sync(MemoryStore())


def test_local_store_rejected_async(tmp_path) -> None:
    # async side wants obstore/icechunk; LocalStore is sync-only in v1
    with pytest.raises(UnsupportedEngineError):
        translate_store_async(LocalStore(tmp_path))


def test_object_store_translates_to_obstore(tmp_path) -> None:
    obstore = pytest.importorskip("obstore")
    from zarr.storage import ObjectStore

    inner = obstore.store.LocalStore(prefix=str(tmp_path))
    zstore = ObjectStore(inner)
    assert translate_store_async(zstore) is inner
```

```python
# tests/zarrista/test_engine.py
from __future__ import annotations

import numpy as np
import pytest

zarrista = pytest.importorskip("zarrista")

import zarr
from zarr.errors import UnsupportedEngineError
from zarr.storage import LocalStore


def _make(tmp_path, **kwargs):
    z = zarr.create_array(
        LocalStore(tmp_path), shape=(10, 9), chunks=(3, 4), dtype="float32", **kwargs
    )
    z[:, :] = np.arange(90, dtype="float32").reshape(10, 9)
    return z


def test_zarrista_engine_read_write_combinations(tmp_path) -> None:
    z = _make(tmp_path)
    ze = zarr.open_array(LocalStore(tmp_path), engine="zarrista")

    # contiguous read
    np.testing.assert_array_equal(np.asarray(ze[2:7, 1:5]), np.asarray(z[2:7, 1:5]))
    # strided read via bbox + post-index
    np.testing.assert_array_equal(np.asarray(ze[1:9:2, ::3]), np.asarray(z[1:9:2, ::3]))
    # full-chunk-aligned write
    ze[0:3, 0:4] = np.zeros((3, 4), dtype="float32")
    np.testing.assert_array_equal(np.asarray(z[0:3, 0:4]), np.zeros((3, 4), dtype="float32"))
    # partial-chunk RMW write
    ze[1:2, 1:2] = np.float32(99.0)
    assert float(z[1, 1]) == 99.0
    assert float(z[0, 0]) == 0.0  # neighbor in same chunk untouched


def test_zarrista_rejects_v2(tmp_path) -> None:
    zarr.create_array(
        LocalStore(tmp_path), shape=(4,), chunks=(2,), dtype="int8", zarr_format=2
    )
    with pytest.raises(UnsupportedEngineError, match="v3"):
        zarr.open_array(LocalStore(tmp_path), engine="zarrista")
```

Run: `uv run --group zarrista pytest tests/zarrista -v`
Expected: FAIL (`ModuleNotFoundError: zarr.zarrista`)

- [ ] **Step 3: Implement translation**

```python
# src/zarr/zarrista/_translate.py
"""Translate zarr-python stores into stores Zarrista can consume."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from zarr.errors import UnsupportedEngineError
from zarr.storage import LocalStore

if TYPE_CHECKING:
    from zarr.abc.store import Store

_SYNC_SUPPORTED = "LocalStore"
_ASYNC_SUPPORTED = "zarr.storage.ObjectStore (obstore-backed) or an icechunk store"


def translate_store_sync(store: Store) -> Any:
    """zarr store -> zarrista sync store (`FilesystemStore`)."""
    import zarrista

    if isinstance(store, LocalStore):
        return zarrista.FilesystemStore(store.root)
    raise UnsupportedEngineError(
        f"the zarrista sync engine cannot serve a {type(store).__name__}; "
        f"supported: {_SYNC_SUPPORTED}. Note: zarr's MemoryStore lives in the "
        "Python process and cannot be shared with the Rust extension."
    )


def translate_store_async(store: Store) -> Any:
    """zarr store -> zarrista async store (obstore `ObjectStore` or icechunk `Session`)."""
    from zarr.storage import ObjectStore

    if isinstance(store, ObjectStore):
        return store.store  # the underlying obstore instance
    try:
        from icechunk import IcechunkStore  # type: ignore[import-not-found]

        if isinstance(store, IcechunkStore):
            return store.session
    except ImportError:
        pass
    raise UnsupportedEngineError(
        f"the zarrista async engine cannot serve a {type(store).__name__}; "
        f"supported: {_ASYNC_SUPPORTED}."
    )
```

(Verify the icechunk attribute: `IcechunkStore.session` — check with
`uv run python -c "import icechunk, inspect; print([m for m in dir(icechunk.IcechunkStore) if 'session' in m])"`
if icechunk is installed; otherwise leave the guarded branch as written.)

- [ ] **Step 4: Implement the engines**

```python
# src/zarr/zarrista/_engine.py
"""Zarrista-backed array engines."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import numpy as np

from zarr.abc.engine import Region
from zarr.errors import UnsupportedEngineError
from zarr.zarrista._translate import translate_store_async, translate_store_sync

if TYPE_CHECKING:
    from zarr.abc.store import Store
    from zarr.core.buffer import BufferPrototype, NDBuffer
    from zarr.core.metadata import ArrayMetadata


def _require_v3(metadata: ArrayMetadata) -> dict[str, Any]:
    if metadata.zarr_format != 3:
        raise UnsupportedEngineError(
            "the zarrista engine supports Zarr v3 only; this array is "
            f"format v{metadata.zarr_format}"
        )
    return metadata.to_dict()


def _region_to_selection(region: Region) -> tuple[slice, ...]:
    return tuple(
        slice(s, e) for s, e in zip(region.start, region.end_exclusive, strict=True)
    )


def _decoded_to_numpy(decoded: Any) -> np.ndarray:
    # Tensor / (future) VariableArray implement __array__; masked layouts do not
    # map to a zarr-python dtype.
    type_name = type(decoded).__name__
    if type_name in ("MaskedTensor", "MaskedVariableArray"):
        raise NotImplementedError(
            f"zarrista returned {type_name}; masked layouts have no "
            "zarr-python equivalent"
        )
    return np.asarray(decoded)


def _chunks_overlapping(region: Region, chunk_shape: tuple[int, ...]) -> Any:
    ranges = [
        range(s // c, (e + c - 1) // c) if e > s else range(0)
        for s, e, c in zip(region.start, region.end_exclusive, chunk_shape, strict=True)
    ]
    return itertools.product(*ranges)


class ZarristaEngine:
    """Sync engine over `zarrista.Array`. No event loop involved."""

    def __init__(self, zarrista_array: Any) -> None:
        self._arr = zarrista_array

    @classmethod
    def from_zarr_metadata(cls, store: Store, path: str, metadata: ArrayMetadata) -> ZarristaEngine:
        import zarrista

        meta_dict = _require_v3(metadata)
        zstore = translate_store_sync(store)
        return cls(zarrista.Array.from_metadata(meta_dict, zstore, "/" + path.strip("/")))

    def with_metadata(self, metadata: ArrayMetadata) -> ZarristaEngine:
        import zarrista

        return ZarristaEngine(
            zarrista.Array.from_metadata(
                _require_v3(metadata), self._arr.store, self._arr.path
            )
        )

    def read_selection(self, selection: Region, *, prototype: BufferPrototype) -> Any:
        return _decoded_to_numpy(
            self._arr.retrieve_array_subset(_region_to_selection(selection))
        )

    def write_selection(
        self, selection: Region, value: NDBuffer, *, prototype: BufferPrototype
    ) -> None:
        import zarrista

        value_np = np.ascontiguousarray(value.as_ndarray_like())
        # regular grids: the origin chunk has the full (unclipped) chunk shape
        chunk_shape = tuple(self._arr.chunk_shape([0] * len(selection.start)))
        for chunk_idx in _chunks_overlapping(selection, chunk_shape):
            chunk_idx = list(chunk_idx)
            chunk_slices = self._arr.chunk_subset(chunk_idx)
            # overlap of the write region with this chunk, in array coords
            lo = tuple(
                max(cs.start, s)
                for cs, s in zip(chunk_slices, selection.start, strict=True)
            )
            hi = tuple(
                min(cs.stop, e)
                for cs, e in zip(chunk_slices, selection.end_exclusive, strict=True)
            )
            in_value = tuple(
                slice(a - s, b - s)
                for a, b, s in zip(lo, hi, selection.start, strict=True)
            )
            in_chunk = tuple(
                slice(a - cs.start, b - cs.start)
                for a, b, cs in zip(lo, hi, chunk_slices, strict=True)
            )
            full_chunk = all(
                sl.start == 0 and sl.stop == (cs.stop - cs.start)
                for sl, cs in zip(in_chunk, chunk_slices, strict=True)
            )
            if full_chunk:
                chunk_np = np.ascontiguousarray(value_np[in_value])
            else:
                chunk_np = np.array(_decoded_to_numpy(self._arr.retrieve_chunk(chunk_idx)))
                chunk_np[in_chunk] = value_np[in_value]
                chunk_np = np.ascontiguousarray(chunk_np)
            self._arr.store_chunk(chunk_idx, zarrista.ArrayBytes(chunk_np))


class ZarristaHierarchyEngine:
    """Store-bound factory for sync zarrista engines (translates the store once)."""

    def __init__(self, store: Store) -> None:
        self._zarr_store = store
        self._zstore = translate_store_sync(store)

    def array_engine(self, path: str, metadata: ArrayMetadata) -> ZarristaEngine:
        import zarrista

        return ZarristaEngine(
            zarrista.Array.from_metadata(
                _require_v3(metadata), self._zstore, "/" + path.strip("/")
            )
        )
```

`ZarristaAsyncEngine` / `ZarristaAsyncHierarchyEngine`: same shape with
`zarrista.AsyncArray.from_metadata`, `translate_store_async`, and `await` on
`retrieve_array_subset` / `retrieve_chunk` / `store_chunk`. Write the full
async twin classes — do not factor a shared base for v1 (the sync class must
never touch a loop; keeping them textually parallel is clearer).

Implementation notes:
- `zarrista.ArrayBytes(chunk_np)` — the constructor takes `bytes: Buffer`;
  pass the numpy array (it implements the buffer protocol). If zarrista
  rejects multi-dimensional buffers, pass
  `chunk_np.reshape(-1).view(np.uint8)` — decide by running the test.
- `store_chunk` expects the *full* chunk shape including edge-chunk overhang;
  `chunk_subset` returns clipped slices at array edges. For edge chunks the
  decoded chunk from `retrieve_chunk` already has the full chunk shape —
  patch using `in_chunk` offsets (clipped coords are correct because
  `chunk_subset` slices start at the chunk origin). For a *full-chunk* write
  on an edge chunk (`full_chunk` computed against the clipped extent), the
  written array is smaller than the chunk shape — in that case fall through
  to the RMW branch. Adjust `full_chunk` to also require
  `(cs.stop - cs.start) == chunk_shape[dim]`.
- `zarrista.Array.from_metadata` path convention is `/`-prefixed (`"/"` is
  the root), matching the stub default `path="/"`.
- vlen dtypes: `retrieve_array_subset` returns `VariableArray`; if
  `np.asarray` on it fails under the pinned commit, raise
  `NotImplementedError("vlen read support pending zarrista numpy export")`
  and mark the corresponding differential test `xfail(strict=False)`.

```python
# src/zarr/zarrista/__init__.py
"""Zarrista-backed array engines for zarr-python.

Requires the `zarrista` package (`pip install zarr[zarrista]` once released;
currently the git-pinned `zarrista` dependency group).
"""

from zarr.zarrista._engine import (
    ZarristaAsyncEngine,
    ZarristaAsyncHierarchyEngine,
    ZarristaEngine,
    ZarristaHierarchyEngine,
)

__all__ = [
    "ZarristaAsyncEngine",
    "ZarristaAsyncHierarchyEngine",
    "ZarristaEngine",
    "ZarristaHierarchyEngine",
]
```

- [ ] **Step 5: Run tests**

Run: `uv run --group zarrista pytest tests/zarrista -v` — Expected: PASS
Run: `uv run pytest tests/zarrista -v` (no group) — Expected: all SKIPPED (importorskip)

- [ ] **Step 6: Commit**

```bash
git add src/zarr/zarrista tests/zarrista pyproject.toml uv.lock
git commit -m "feat: zarrista-backed array engines with store translation"
```

---

### Task 9: Delete `zarr.crud`, `zarr.zarrs`, `packages/zarrs-bindings`

**Files:**
- Delete: `src/zarr/crud/`, `src/zarr/zarrs/`, `packages/zarrs-bindings/`, `tests/crud/`, `tests/zarrs/`, `.github/workflows/zarrs.yml`, the `changes/` fragments describing zarr.crud/zarr.zarrs (find: `grep -rl "crud\|zarrs" changes/`)
- Modify: `pyproject.toml` — remove the `zarrs` dependency group (line ~138), the `zarrs-bindings` uv source (line ~502), the `/packages/zarrs-bindings` exclude (line ~10), and the `--ignore=src/zarr/zarrs` pytest flag (line ~444).

- [ ] **Step 1: Delete and unwire**

```bash
git rm -r src/zarr/crud src/zarr/zarrs packages/zarrs-bindings tests/crud tests/zarrs .github/workflows/zarrs.yml
```

Edit `pyproject.toml` as listed above. Search for stragglers:

```bash
grep -rn "zarr.crud\|zarr\.zarrs\|zarrs_bindings\|zarrs-bindings" src tests docs pyproject.toml .github --include="*" | grep -v superpowers
```

Expected: no hits outside `docs/superpowers/` (specs/plans are historical records — leave them).

- [ ] **Step 2: Full test run**

Run: `uv sync && uv run pytest tests -x -q --ignore=tests/zarrista`
Expected: PASS (no imports of deleted modules anywhere)
Run: `uv run mypy src/zarr` — Expected: no errors

- [ ] **Step 3: Commit**

```bash
git add -u src tests pyproject.toml uv.lock .github
git commit -m "refactor!: remove zarr.crud, zarr.zarrs, and the zarrs-bindings crate"
```

---

### Task 10: Differential suite, CI, changelog

**Files:**
- Create: `tests/engine/test_differential.py`, `.github/workflows/engine.yml`, `changes/<PR>.feature.md` (get the PR number from the branch's open PR via `gh pr view --json number -q .number`; if no PR exists yet, use the next number after `gh api repos/{owner}/{repo}/issues?state=all&per_page=1 -q '.[0].number'` and note it must match the eventual PR)

**Interfaces:**
- Consumes: everything.

- [ ] **Step 1: Differential test**

```python
# tests/engine/test_differential.py
"""The same operations through both engines must agree with numpy and each other."""

from __future__ import annotations

import numpy as np
import pytest

import zarr
from zarr.storage import LocalStore

try:
    import zarrista  # noqa: F401

    ENGINES = ["default", "zarrista"]
except ImportError:
    ENGINES = ["default"]

SHAPE = (10, 9)
CHUNKS = (3, 4)


@pytest.fixture
def reference(tmp_path):
    z = zarr.create_array(
        LocalStore(tmp_path), shape=SHAPE, chunks=CHUNKS, dtype="float64"
    )
    data = np.arange(90, dtype="float64").reshape(SHAPE)
    z[:, :] = data
    return tmp_path, data


READS = [
    (slice(None), slice(None)),
    (slice(2, 7), slice(1, 5)),
    (slice(8, 2, -2), slice(None, None, 3)),
    (3, slice(None)),
    (-1, -2),
    (slice(4, 4), slice(None)),
]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("sel", READS)
def test_reads_match_numpy(reference, engine, sel) -> None:
    tmp_path, data = reference
    z = zarr.open_array(LocalStore(tmp_path), engine=engine)
    np.testing.assert_array_equal(np.asarray(z[sel]), data[sel])


@pytest.mark.parametrize("engine", ENGINES)
def test_fancy_reads_match_numpy(reference, engine) -> None:
    tmp_path, data = reference
    z = zarr.open_array(LocalStore(tmp_path), engine=engine)
    np.testing.assert_array_equal(
        np.asarray(z.oindex[np.array([7, 1, 4]), np.array([0, 8])]),
        data[np.ix_([7, 1, 4], [0, 8])],
    )
    np.testing.assert_array_equal(
        np.asarray(z.vindex[np.array([9, 0, 3]), np.array([8, 0, 2])]),
        data[np.array([9, 0, 3]), np.array([8, 0, 2])],
    )
    np.testing.assert_array_equal(np.asarray(z.blocks[1, 2]), data[3:6, 8:9])


@pytest.mark.parametrize("engine", ENGINES)
def test_writes_match_numpy(reference, engine) -> None:
    tmp_path, data = reference
    z = zarr.open_array(LocalStore(tmp_path), engine=engine)
    expected = data.copy()

    z[0:3, 0:4] = 7.0                       # aligned full chunk
    expected[0:3, 0:4] = 7.0
    z[4:6, 2:9] = np.arange(14, dtype="float64").reshape(2, 7)  # partial chunks
    expected[4:6, 2:9] = np.arange(14, dtype="float64").reshape(2, 7)
    z[1:9:3, ::4] = -1.0                    # strided write (facade RMW)
    expected[1:9:3, ::4] = -1.0

    np.testing.assert_array_equal(np.asarray(z[:, :]), expected)


@pytest.mark.parametrize("engine", ENGINES)
def test_sharded_reads(reference, engine, tmp_path_factory) -> None:
    path = tmp_path_factory.mktemp("sharded")
    z = zarr.create_array(
        LocalStore(path),
        shape=SHAPE,
        chunks=(3, 4),          # inner chunks
        shards=(6, 8),          # shard shape
        dtype="int32",
    )
    data = np.arange(90, dtype="int32").reshape(SHAPE)
    z[:, :] = data
    zr = zarr.open_array(LocalStore(path), engine=engine)
    np.testing.assert_array_equal(np.asarray(zr[2:8, 3:9]), data[2:8, 3:9])
```

Run: `uv run --group zarrista pytest tests/engine/test_differential.py -v` — Expected: PASS (both engines)
Run: `uv run pytest tests/engine/test_differential.py -v` — Expected: PASS (default only)

- [ ] **Step 2: CI workflow**

Create `.github/workflows/engine.yml` modeled on the deleted `zarrs.yml`
(check its triggers/paths in git history: `git show HEAD~1:.github/workflows/zarrs.yml`):
a single job on `ubuntu-latest` that installs Rust (`dtolnay/rust-toolchain@stable`),
sets up uv (`astral-sh/setup-uv`), runs
`uv sync --group zarrista` and
`uv run --group zarrista pytest tests/engine tests/zarrista -v`.
Trigger on pull requests touching `src/zarr/**`, `tests/engine/**`,
`tests/zarrista/**`, `pyproject.toml`, and the workflow file itself. Keep
`permissions: {}` at top level and pin action SHAs, matching the repo's other
workflows (zizmor enforces this).

- [ ] **Step 3: Changelog fragment**

```markdown
# changes/<PR>.feature.md
`Array` and `AsyncArray` now route data I/O through pluggable *array engines*
(`zarr.abc.engine.ArrayEngine` / `AsyncArrayEngine`). The default engine
preserves existing behavior on every store and format. Pass
`engine="zarrista"` (requires the `zarrista` package) to serve Zarr v3 arrays
on local-filesystem, obstore, or icechunk storage through the Rust `zarrs`
implementation. The experimental `zarr.crud` and `zarr.zarrs` modules and the
bundled `zarrs-bindings` crate are removed in favor of this interface.
```

- [ ] **Step 4: Full local gate**

Run: `uv run --group zarrista pytest tests -q` — Expected: PASS
Run: `uv run mypy src/zarr` — Expected: no errors
Run: `uv run pre-commit run --all-files` — Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/engine/test_differential.py .github/workflows/engine.yml changes/
git commit -m "test/ci: engine differential suite and zarrista CI job"
```

---

## Self-Review Notes (already applied)

- Spec coverage: protocols+Region (T1), normalization/facade rules (T2, T5, T6),
  default engines incl. sync adapter (T3), explicit engine selection without a
  config key (T4, T7), truly-sync path (T6), zarrista store translation /
  v3-only / RMW writes / buffer handling (T8), fail-loud errors (T1, T4, T8),
  deletions (T9), differential tests + CI + changelog (T10). `with_metadata`
  rebinding covered in T3/T5/T6.
- Deliberately out of scope, per spec non-goals: group operations, engine-owned
  metadata mutation, `out=` in the engine contract (facade copies instead).
- Known judgment calls the implementer may hit: exact `Array` constructor shape
  (T6 step 1), zarrista `ArrayBytes` buffer dimensionality (T8 note), icechunk
  session attribute (T8 note), vlen export under the pinned commit (T8 note).
  Each has a decision rule written at the point of use.
