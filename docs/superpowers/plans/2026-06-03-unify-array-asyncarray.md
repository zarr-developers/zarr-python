# Unify `Array` and `AsyncArray` via a pluggable `runner` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `Array` a self-contained class that owns its own state plus a pluggable `_runner`, exposes every async operation as a `*_async` coroutine method, and implements its synchronous methods as `self._runner.run(self.foo_async(...))` — while leaving `AsyncArray` untouched as a compatibility shim.

**Architecture:** A `Runner` protocol (with default `SyncRunner` wrapping the existing `sync()`) is added to `zarr/core/sync.py`. `Array` stops wrapping an `AsyncArray`; it stores `metadata`, `store_path`, `config`, `codec_pipeline`, `_chunk_grid`, and `_runner` directly, and reuses the already-existing module-level async free functions (`_getitem`, `_setitem`, `_resize`, `_append`, `_update_attributes`, `_info_complete`, `_nchunks_initialized`, `_nshards_initialized`, `_nbytes_stored`, etc.). Several of those functions accept an array object and use its property surface; their parameter type is widened to a structural `SupportsArrayState` Protocol that both `Array` and `AsyncArray` satisfy, so no function bodies change. `Array.async_array` / `_async_array` are deprecated and reconstructed on demand.

**Tech Stack:** Python 3.11+, `asyncio`, `typing.Protocol`, pytest. Run everything with `uv run`.

---

## File Structure

- `src/zarr/core/sync.py` — add `Runner` Protocol + `SyncRunner` class.
- `src/zarr/core/array.py` — add `SupportsArrayState` Protocol; widen free-function annotations; rewrite `Array` (`__init__`, state, `*_async` methods, sync methods, `_from_async_array`, deprecated `async_array`).
- `tests/test_runner.py` — new tests for the runner protocol, custom-runner injection, sync/async equivalence, and `async_array` deprecation.
- `tests/test_array.py` — update the few spots that rely on the old construction / `async_array` access (only where they now warn).

---

## Conventions for this plan

- Always run tooling via `uv run` (e.g. `uv run pytest`, `uv run mypy`).
- Docstrings use single-backtick markdown (mkdocs), not RST double-backticks.
- Commit after each task once its tests pass.
- The type parameter on both classes is `T_ArrayMetadata: (ArrayV2Metadata, ArrayV3Metadata)`.

---

### Task 1: Add the `Runner` protocol and `SyncRunner` to `sync.py`

**Files:**
- Modify: `src/zarr/core/sync.py`
- Test: `tests/test_runner.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_runner.py`:

```python
from __future__ import annotations

import asyncio

from zarr.core.sync import Runner, SyncRunner


async def _coro() -> int:
    await asyncio.sleep(0)
    return 42


def test_sync_runner_runs_coroutine() -> None:
    runner = SyncRunner()
    assert runner.run(_coro()) == 42


def test_sync_runner_is_runner() -> None:
    assert isinstance(SyncRunner(), Runner)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_runner.py -v`
Expected: FAIL with `ImportError: cannot import name 'Runner'` (or `SyncRunner`).

- [ ] **Step 3: Implement `Runner` and `SyncRunner`**

In `src/zarr/core/sync.py`, add `Protocol`, `runtime_checkable`, and `TypeVar`/`T` to the imports as needed, then add near the bottom of the module (after `sync`):

```python
@runtime_checkable
class Runner(Protocol):
    """A `Runner` executes a coroutine and returns the awaited result.

    Implement this protocol to plug a custom event loop into `Array`.
    """

    def run(self, coro: Coroutine[Any, Any, T]) -> T: ...


class SyncRunner:
    """The default `Runner`. Runs coroutines on Zarr's shared background event
    loop via `sync`.
    """

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        return sync(coro)
```

Add the supporting imports at the top of `sync.py`:

```python
from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable
```

and (outside `TYPE_CHECKING`, because `T` is used at runtime in the protocol/class bodies' annotations only — annotations are lazy under `from __future__ import annotations`, so a `TYPE_CHECKING`-only `T` is fine, but `Coroutine`/`Any` are referenced only in annotations too). Keep `Coroutine` and `Any` in the existing `TYPE_CHECKING` block. Define `T` next to `P`:

```python
T = TypeVar("T")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_runner.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Type-check**

Run: `uv run mypy src/zarr/core/sync.py`
Expected: no new errors.

- [ ] **Step 6: Commit**

```bash
git add src/zarr/core/sync.py tests/test_runner.py
git commit -m "feat(sync): add Runner protocol and SyncRunner"
```

---

### Task 2: Add the `SupportsArrayState` protocol and widen free-function annotations

The free functions `_resize`, `_append`, `_update_attributes`, `_info_complete`, `_nchunks_initialized`, `_nshards_initialized`, and `_shards_initialized` currently take `AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]`. They only use the array's property surface (`metadata`, `store_path`, `codec_pipeline`, `config`, `_chunk_grid`, `shape`, `shards`, `chunks`, `_info`) and mutate via `object.__setattr__`. Widen the annotation to a structural protocol so `Array` can pass `self`.

**Files:**
- Modify: `src/zarr/core/array.py`

- [ ] **Step 1: Add the protocol**

Near the top-level definitions in `src/zarr/core/array.py` (after imports, before `AsyncArray`), add:

```python
@runtime_checkable
class SupportsArrayState(Protocol):
    """The structural surface the module-level array helpers rely on.

    Both `AsyncArray` and `Array` satisfy this protocol, which lets the
    helper functions operate on either class.
    """

    metadata: ArrayMetadata
    store_path: StorePath
    codec_pipeline: CodecPipeline
    config: ArrayConfig
    _chunk_grid: ChunkGrid

    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def chunks(self) -> tuple[int, ...]: ...
    @property
    def shards(self) -> tuple[int, ...] | None: ...

    def _info(
        self,
        count_chunks_initialized: int | None = None,
        count_bytes_stored: int | None = None,
    ) -> Any: ...
```

Ensure `Protocol` and `runtime_checkable` are imported from `typing` at the top of `array.py`.

- [ ] **Step 2: Widen the free-function signatures**

In `src/zarr/core/array.py`, change the first parameter annotation of each of these functions from
`array: AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]`
to
`array: SupportsArrayState`:

- `_nchunks_initialized` (~line 5267)
- `_nshards_initialized` (~line 5297)
- `_shards_initialized` (the function `_nshards_initialized` calls — find via `grep -n "async def _shards_initialized" src/zarr/core/array.py`)
- `_resize` (~line 5873)
- `_append` (~line 5925)
- `_update_attributes` (~line 5996) — also change its return annotation from the `AsyncArray[...]` union to `SupportsArrayState`
- `_info_complete` (~line 6023)

Do NOT change `_nbytes_stored` (it already takes `store_path: StorePath`).

- [ ] **Step 3: Type-check**

Run: `uv run mypy src/zarr/core/array.py`
Expected: no new errors. (AsyncArray still satisfies the protocol structurally.)

- [ ] **Step 4: Run the existing async-array tests to confirm no behavior change**

Run: `uv run pytest tests/test_array.py -k "async" -q`
Expected: PASS (AsyncArray behavior unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/zarr/core/array.py
git commit -m "refactor(array): widen array-helper params to SupportsArrayState protocol"
```

---

### Task 3: Give `Array` its own state and `__init__` (with `runner`)

Replace the `@dataclass(frozen=False)` `Array` that wraps `_async_array` with an explicit class that owns its state. This task ONLY changes construction + state + the `async_array`/`_async_array` deprecation + a `_from_async_array` helper. Property/method rewiring happens in later tasks; to keep this task self-contained, the existing property bodies that read `self.async_array.X` will keep working because the deprecated `async_array` property reconstructs an `AsyncArray` on demand.

**Files:**
- Modify: `src/zarr/core/array.py` (the `Array` class, starting ~line 1800)
- Test: `tests/test_runner.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_runner.py`:

```python
import warnings

import numpy as np
import pytest

import zarr
from zarr.core.array import Array, AsyncArray
from zarr.core.sync import SyncRunner
from zarr.storage import MemoryStore


def _make_array() -> Array:
    return zarr.create_array(
        store=MemoryStore(), shape=(8,), chunks=(4,), dtype="i4", fill_value=0
    )


def test_array_has_default_sync_runner() -> None:
    arr = _make_array()
    assert isinstance(arr._runner, SyncRunner)


def test_array_owns_state() -> None:
    arr = _make_array()
    # state lives on Array directly, not via a wrapped AsyncArray
    assert arr.metadata is not None
    assert arr.store_path is not None
    assert arr.codec_pipeline is not None


def test_array_accepts_custom_runner() -> None:
    class RecordingRunner:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, coro):  # type: ignore[no-untyped-def]
            self.calls += 1
            return SyncRunner().run(coro)

    runner = RecordingRunner()
    aa = _make_array()._as_async()  # helper defined below; or build AsyncArray directly
    arr = Array(
        metadata=aa.metadata,
        store_path=aa.store_path,
        config=aa.config,
        runner=runner,
    )
    _ = arr[:]
    assert runner.calls > 0


def test_async_array_property_deprecated() -> None:
    arr = _make_array()
    with pytest.warns(DeprecationWarning):
        aa = arr.async_array
    assert isinstance(aa, AsyncArray)


def test_from_async_array_roundtrip() -> None:
    arr = _make_array()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        aa = arr.async_array
    arr2 = Array._from_async_array(aa)
    assert arr2.metadata == arr.metadata
    assert isinstance(arr2._runner, SyncRunner)
```

Note: remove the `._as_async()` reference — construct the `AsyncArray` for `test_array_accepts_custom_runner` directly instead:

```python
    base = _make_array()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        aa = base.async_array
    arr = Array(metadata=aa.metadata, store_path=aa.store_path, config=aa.config, runner=runner)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_runner.py -v`
Expected: FAIL (e.g. `Array` has no `_runner`, `__init__` signature mismatch, no `_from_async_array`).

- [ ] **Step 3: Rewrite the `Array` class header, state, and `__init__`**

Replace the dataclass decorator + `_async_array` field + `async_array` property (`src/zarr/core/array.py`, ~lines 1800–1817) with:

```python
class Array[T_ArrayMetadata: (ArrayV2Metadata, ArrayV3Metadata)]:
    """
    A Zarr array.
    """

    metadata: T_ArrayMetadata
    store_path: StorePath
    config: ArrayConfig
    codec_pipeline: CodecPipeline
    _chunk_grid: ChunkGrid
    _runner: Runner

    def __init__(
        self,
        metadata: ArrayMetadata | ArrayMetadataDict,
        store_path: StorePath,
        config: ArrayConfigLike | None = None,
        *,
        runner: Runner | None = None,
    ) -> None:
        metadata_parsed = parse_array_metadata(metadata)
        config_parsed = parse_array_config(config)
        object.__setattr__(self, "metadata", metadata_parsed)
        object.__setattr__(self, "store_path", store_path)
        object.__setattr__(self, "config", config_parsed)
        object.__setattr__(self, "_chunk_grid", ChunkGrid.from_metadata(metadata_parsed))
        object.__setattr__(
            self,
            "codec_pipeline",
            create_codec_pipeline(metadata=metadata_parsed, store=store_path.store),
        )
        object.__setattr__(self, "_runner", runner if runner is not None else SyncRunner())

    @classmethod
    def _from_async_array(
        cls,
        async_array: AsyncArray[T_ArrayMetadata],
        *,
        runner: Runner | None = None,
    ) -> Self:
        return cls(
            metadata=async_array.metadata,
            store_path=async_array.store_path,
            config=async_array.config,
            runner=runner,
        )

    @property
    def async_array(self) -> AsyncArray[T_ArrayMetadata]:
        """An asynchronous version of this array.

        .. deprecated::
            Use the `*_async` methods on `Array` instead. This property will be
            removed in a future release.
        """
        warnings.warn(
            "Array.async_array is deprecated; use the *_async methods on Array instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return AsyncArray(self.metadata, self.store_path, self.config)
```

Notes:
- Keep using `object.__setattr__` so the class can later become frozen again without churn (matches `AsyncArray`).
- Remove the `_async_array` class attribute entirely.
- Ensure `Runner`, `SyncRunner` are imported from `zarr.core.sync`, and `warnings` is imported at the top of `array.py`.
- `create_codec_pipeline` is already defined in this module (line ~224).

- [ ] **Step 4: Fix the `_chunk_grid` and `config` properties**

The existing `config` property (~line 1819) returns `self.async_array.config` and `_chunk_grid` property (~line 1832) returns `self.async_array._chunk_grid`. These now collide with the real attributes. DELETE both property definitions — `config` and `_chunk_grid` are now plain attributes set in `__init__`.

- [ ] **Step 5: Add an `_info` method to `Array`**

`Array` needs `_info` for the `SupportsArrayState` protocol (used in Task 6). Add it to `Array` (mirroring `AsyncArray._info`, ~line 1777):

```python
    def _info(
        self, count_chunks_initialized: int | None = None, count_bytes_stored: int | None = None
    ) -> Any:
        chunk_shape = self.chunks if self._chunk_grid.is_regular else None
        return ArrayInfo(
            _zarr_format=self.metadata.zarr_format,
            _data_type=self._zdtype,
            _fill_value=self.metadata.fill_value,
            _shape=self.shape,
            _order=self.order,
            _shard_shape=self.shards,
            _chunk_shape=chunk_shape,
            _read_only=self.read_only,
            _compressors=self.compressors,
            _filters=self.filters,
            _serializer=self.serializer,
            _store_type=type(self.store_path.store).__name__,
            _count_bytes=self.nbytes,
            _count_bytes_stored=count_bytes_stored,
            _count_chunks_initialized=count_chunks_initialized,
        )
```

This requires `Array` to expose `_zdtype`, which currently lives only on `AsyncArray` (~line 972). Add an `Array._zdtype` property copied verbatim from `AsyncArray._zdtype`:

```python
    @property
    def _zdtype(self) -> ZDType[TBaseDType, TBaseScalar]:
        """
        The zarr-specific representation of the array data type
        """
        if self.metadata.zarr_format == 2:
            return self.metadata.dtype
        else:
            return self.metadata.data_type
```

- [ ] **Step 6: Fix `with_config` to thread the runner**

Replace `with_config` body (~line 2233) `return type(self)(self._async_array.with_config(config))` with construction from new state. IMPORTANT: `AsyncArray.with_config` (~line 1156) does NOT use `parse_array_config`; it merges the new config over the existing one. Replicate that merge exactly and thread the runner:

```python
        if isinstance(config, ArrayConfig):
            new_config = config
        else:
            # Merge new config with existing config, so missing keys are inherited
            # from the current array rather than from global defaults
            new_config = ArrayConfig(**{**self.config.to_dict(), **config})  # type: ignore[arg-type]
        return type(self)(
            metadata=self.metadata,
            store_path=self.store_path,
            config=new_config,
            runner=self._runner,
        )
```

- [ ] **Step 7: Update the three `Array` classmethods that build via `cls(async_array)`**

`Array._create` (~line 1892), `Array.from_dict` (~line 1923), `Array.open` (~line 1945) end with `return cls(async_array)`. Change each to `return cls._from_async_array(async_array)`.

- [ ] **Step 8: Run the new runner tests**

Run: `uv run pytest tests/test_runner.py -v`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/zarr/core/array.py tests/test_runner.py
git commit -m "feat(array): Array owns its own state + runner; deprecate async_array"
```

---

### Task 4: Rewire `Array`'s property delegations to its own state

There are ~48 `self.async_array.X` reads in property bodies (verify count: `grep -c "self\.async_array\." src/zarr/core/array.py`). With `async_array` now warning, these MUST be repointed to `Array`'s own state or the deprecation warning fires on every property access.

**Files:**
- Modify: `src/zarr/core/array.py` (the `Array` property bodies, ~lines 1947–2248)

- [ ] **Step 1: Run the property tests first (capture current passing state)**

Run: `uv run pytest tests/test_array.py -q -k "property or shape or dtype or attrs or nbytes or chunks or shards" 2>&1 | tail -20`
Expected: PASS now (baseline).

- [ ] **Step 2: Repoint each delegating property**

For every `Array` property/method whose body reads `self.async_array.X`, replace with the direct equivalent. The mapping is mechanical because `Array` now holds the same state. Examples (apply the same pattern to all):

- `store` → `return self.store_path.store`
- `ndim` → `return self.metadata.ndim`
- `shape` getter → `return self.metadata.shape`
- `chunks` → copy the `AsyncArray.chunks` body (reads `self.metadata`)
- `shards` → copy `AsyncArray.shards` body
- `size` → copy `AsyncArray.size` body
- `dtype` → copy `AsyncArray.dtype` body
- `attrs` → `return Attributes(self)` (match current `Array.attrs` semantics; check ~line 2080)
- `path`, `name`, `basename`, `order`, `read_only`, `fill_value`, `filters`, `serializer`, `compressor`, `compressors`, `cdata_shape`, `_chunk_grid_shape`, `_shard_grid_shape`, `nchunks`, `_nshards`, `nbytes` → copy each corresponding `AsyncArray` property body (all read from `self.metadata` / `self.config` / `self._chunk_grid` / `self.store_path`).
- `metadata` property (~line 2111) — `Array` had a `metadata` property returning `self.async_array.metadata`; now `metadata` is a plain attribute. DELETE the property.
- `store_path` property (~line 2115) — same: DELETE the property; it's a plain attribute now.

Strategy: for each `AsyncArray` property, the body already uses exactly `self.metadata`/`self.config`/`self._chunk_grid`/`self.store_path`. Copy the body verbatim into the `Array` property of the same name. Use `grep -n "self\.async_array\." src/zarr/core/array.py` to enumerate remaining sites and confirm ZERO remain in property bodies when done (the only acceptable remaining `async_array` reference is inside the deprecated `async_array` property itself, which doesn't reference `self.async_array`).

- [ ] **Step 3: Verify no stray `self.async_array.` reads remain**

Run: `grep -n "self\.async_array\." src/zarr/core/array.py`
Expected: NO output (zero matches).

- [ ] **Step 4: Run the array property tests**

Run: `uv run pytest tests/test_array.py -q -k "property or shape or dtype or attrs or nbytes or chunks or shards" 2>&1 | tail -20`
Expected: PASS, and no `DeprecationWarning` emitted (run with `-W error::DeprecationWarning` to be strict):
Run: `uv run pytest tests/test_array.py -q -W error::DeprecationWarning -k "property or shape or dtype" 2>&1 | tail -20`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/zarr/core/array.py
git commit -m "refactor(array): repoint Array properties to own state"
```

---

### Task 5: Add the read/write `*_async` methods and rewire sync selection methods

Add `getitem_async`, `setitem_async`, and the basic/orthogonal/mask/coordinate/block selection `*_async` twins to `Array`, calling the existing free functions. Rewrite the synchronous selection methods to use `self._runner.run(self.<x>_async(...))` instead of `sync(self.async_array.<x>(...))`.

**Files:**
- Modify: `src/zarr/core/array.py` (`Array` selection methods, ~lines 2426–3767)
- Test: `tests/test_runner.py`

- [ ] **Step 1: Write the failing equivalence test**

Append to `tests/test_runner.py`:

```python
def test_getitem_sync_async_equivalence() -> None:
    arr = _make_array()
    arr[:] = np.arange(8, dtype="i4")
    sync_result = arr[2:6]
    async_via_runner = arr._runner.run(arr.getitem_async(slice(2, 6)))
    np.testing.assert_array_equal(sync_result, async_via_runner)


def test_setitem_async_roundtrip() -> None:
    arr = _make_array()
    arr._runner.run(arr.setitem_async(slice(0, 4), np.arange(4, dtype="i4")))
    np.testing.assert_array_equal(arr[0:4], np.arange(4, dtype="i4"))
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_runner.py -k "equivalence or setitem_async" -v`
Expected: FAIL with `AttributeError: 'Array' object has no attribute 'getitem_async'`.

- [ ] **Step 3: Add the core `*_async` methods to `Array`**

Add to `Array` (place near the selection methods). Each delegates to the existing module-level free functions exactly as `AsyncArray` does:

```python
    async def _get_selection_async(
        self,
        indexer: Indexer,
        *,
        prototype: BufferPrototype,
        out: NDBuffer | None = None,
        fields: Fields | None = None,
    ) -> NDArrayLikeOrScalar:
        return await _get_selection(
            self.store_path,
            self.metadata,
            self.codec_pipeline,
            self.config,
            self._chunk_grid,
            indexer,
            prototype=prototype,
            out=out,
            fields=fields,
        )

    async def _set_selection_async(
        self,
        indexer: Indexer,
        value: npt.ArrayLike,
        *,
        prototype: BufferPrototype,
        fields: Fields | None = None,
    ) -> None:
        return await _set_selection(
            self.store_path,
            self.metadata,
            self.codec_pipeline,
            self.config,
            self._chunk_grid,
            indexer,
            value,
            prototype=prototype,
            fields=fields,
        )

    async def getitem_async(
        self,
        selection: BasicSelection,
        *,
        prototype: BufferPrototype | None = None,
    ) -> NDArrayLikeOrScalar:
        return await _getitem(
            self.store_path,
            self.metadata,
            self.codec_pipeline,
            self.config,
            self._chunk_grid,
            selection,
            prototype=prototype,
        )

    async def setitem_async(
        self,
        selection: BasicSelection,
        value: npt.ArrayLike,
        *,
        prototype: BufferPrototype | None = None,
    ) -> None:
        return await _setitem(
            self.store_path,
            self.metadata,
            self.codec_pipeline,
            self.config,
            self._chunk_grid,
            selection,
            value,
            prototype=prototype,
        )
```

Confirm the exact arg order of `_get_selection`, `_set_selection`, `_getitem`, `_setitem` by reading their definitions (`grep -n "^async def _getitem\|^async def _setitem\|^async def _get_selection\|^async def _set_selection" src/zarr/core/array.py`) and match them. The `AsyncArray._get_selection`/`getitem`/`setitem` bodies (lines ~1416, ~1436, ~1574) are the reference — copy their call shape.

- [ ] **Step 4: Rewrite the synchronous selection methods**

Replace every `sync(self.async_array.X(...))` and `sync(self.async_array._set_selection(...))` / `sync(self.async_array._get_selection(...))` call inside `Array`'s sync selection methods with the runner + `*_async` equivalent. Concretely, for the selection getters/setters (`__getitem__`, `__setitem__`, `get_basic_selection`, `set_basic_selection`, `get_orthogonal_selection`, `set_orthogonal_selection`, `get_mask_selection`, `set_mask_selection`, `get_coordinate_selection`, `set_coordinate_selection`, `get_block_selection`, `set_block_selection`):

- Where the body did `sync(self.async_array._get_selection(indexer, ...))`, change to `self._runner.run(self._get_selection_async(indexer, ...))`.
- Where the body did `sync(self.async_array._set_selection(indexer, value, ...))`, change to `self._runner.run(self._set_selection_async(indexer, value, ...))`.

The indexer-construction logic in these sync methods stays exactly as-is; only the terminal `sync(self.async_array._..._selection(...))` call changes. Enumerate them with `grep -n "sync(self.async_array._set_selection\|sync(self.async_array._get_selection\|sync($" src/zarr/core/array.py` and the broader `grep -n "self.async_array._get_selection\|self.async_array._set_selection" src/zarr/core/array.py`.

- [ ] **Step 5: Run the equivalence tests**

Run: `uv run pytest tests/test_runner.py -k "equivalence or setitem_async" -v`
Expected: PASS.

- [ ] **Step 6: Run the full selection test suite**

Run: `uv run pytest tests/test_array.py -q -k "selection or getitem or setitem or basic or orthogonal or mask or coordinate or block" 2>&1 | tail -25`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/zarr/core/array.py tests/test_runner.py
git commit -m "feat(array): add selection *_async methods; route sync selection via runner"
```

---

### Task 6: Add remaining `*_async` methods (`resize`, `append`, `update_attributes`, `info_complete`, `nchunks_initialized`, `nbytes_stored`) and rewire their sync wrappers

These reuse the free functions widened in Task 2 (`_resize`, `_append`, `_update_attributes`, `_info_complete`, `_nchunks_initialized`, `_nshards_initialized`, `_nbytes_stored`).

**Files:**
- Modify: `src/zarr/core/array.py`
- Test: `tests/test_runner.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_runner.py`:

```python
def test_resize_async() -> None:
    arr = _make_array()
    arr._runner.run(arr.resize_async((16,)))
    assert arr.shape == (16,)


def test_update_attributes_async() -> None:
    arr = _make_array()
    arr._runner.run(arr.update_attributes_async({"foo": "bar"}))
    assert arr.metadata.attributes["foo"] == "bar"


def test_nchunks_initialized_async() -> None:
    arr = _make_array()
    arr[:] = np.arange(8, dtype="i4")
    n = arr._runner.run(arr.nchunks_initialized_async())
    assert n == arr.nchunks_initialized  # sync property matches async result
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_runner.py -k "resize_async or update_attributes_async or nchunks_initialized_async" -v`
Expected: FAIL (`AttributeError`).

- [ ] **Step 3: Add the `*_async` methods to `Array`**

```python
    async def resize_async(
        self, new_shape: ShapeLike, delete_outside_chunks: bool = True
    ) -> None:
        return await _resize(self, new_shape, delete_outside_chunks)

    async def append_async(self, data: npt.ArrayLike, axis: int = 0) -> tuple[int, ...]:
        return await _append(self, data, axis)

    async def update_attributes_async(self, new_attributes: dict[str, JSON]) -> Self:
        await _update_attributes(self, new_attributes)
        return self

    async def nchunks_initialized_async(self) -> int:
        return await _nchunks_initialized(self)

    async def _nshards_initialized_async(self) -> int:
        return await _nshards_initialized(self)

    async def nbytes_stored_async(self) -> int:
        return await _nbytes_stored(self.store_path)

    async def info_complete_async(self) -> Any:
        return await _info_complete(self)
```

Note `_resize` mutates `self` in place via `object.__setattr__` — this works on `Array` because it is not frozen.

- [ ] **Step 4: Rewrite the corresponding sync methods**

Replace the sync bodies (currently `sync(self.async_array.X(...))`, ~lines 2274, 2296, 2306, 3831, 3867, 3894, 3952):

- `nchunks_initialized` property (~2274): `return self._runner.run(self.nchunks_initialized_async())`
- `_nshards_initialized` property (~2296): `return self._runner.run(self._nshards_initialized_async())`
- `nbytes_stored` (~2306): `return self._runner.run(self.nbytes_stored_async())`
- `resize` (~3831): `self._runner.run(self.resize_async(new_shape))` (keep the existing wrapper's return/type; check whether `Array.resize` returns a new array or `None` — read ~line 3800–3835 and preserve its current return contract)
- `append` (~3867): `return self._runner.run(self.append_async(data, axis=axis))`
- `update_attributes` (~3894): the current body does `new_array = sync(self.async_array.update_attributes(new_attributes))` then `return type(self)(new_array)` (wrapping the returned `AsyncArray`). Under the new design `update_attributes_async` mutates `self` in place and returns `self` (an `Array`), so replace the whole body with:
  ```python
          self._runner.run(self.update_attributes_async(new_attributes))
          return self
  ```
  This matches the prior observable behavior: attributes are persisted and an `Array` with the updated metadata is returned. (Previously a fresh wrapper was returned; now `self` is mutated and returned. If a test asserts the returned object is a *distinct* instance, adjust to `return type(self)(metadata=self.metadata, store_path=self.store_path, config=self.config, runner=self._runner)` — check `tests/test_array.py` for such an assertion in Task 8.)
- `info_complete` (~3952): `return self._runner.run(self.info_complete_async())`

For each, read the surrounding 10 lines first to preserve the exact return type and any post-processing.

- [ ] **Step 5: Confirm zero `self.async_array.` references remain anywhere**

Run: `grep -n "self\.async_array\." src/zarr/core/array.py`
Expected: NO output.

- [ ] **Step 6: Run the new tests + relevant suite**

Run: `uv run pytest tests/test_runner.py -v`
Run: `uv run pytest tests/test_array.py -q -k "resize or append or update_attributes or info or nchunks or nbytes" 2>&1 | tail -25`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/zarr/core/array.py tests/test_runner.py
git commit -m "feat(array): add remaining *_async methods; route sync wrappers via runner"
```

---

### Task 7: Update external construction sites to `_from_async_array`

The `Array(async_array)` call sites outside `array.py` must use the new construction path.

**Files:**
- Modify: `src/zarr/api/synchronous.py` (lines ~763, ~947, ~1168, ~1359)
- Modify: `src/zarr/core/group.py` (lines ~2272, ~2656, ~2779)

- [ ] **Step 1: Find all external `Array(` construction-from-async sites**

Run: `grep -rn "Array(async_array\|Array($" src/zarr/api/synchronous.py src/zarr/core/group.py`
Also check each `Array(` call's argument: only those passing an `AsyncArray` positionally need changing.

- [ ] **Step 2: Replace each with `_from_async_array`**

For each site that does `Array(<async_array_expr>)`, change to `Array._from_async_array(<async_array_expr>)`. For example in `group.py`:
`yield name, Array(async_array)` → `yield name, Array._from_async_array(async_array)`.

Read each call's surrounding lines to confirm the single positional arg is an `AsyncArray` (the variable is usually named `async_array` or is a `await AsyncArray.open(...)` result).

- [ ] **Step 3: Type-check the modified modules**

Run: `uv run mypy src/zarr/api/synchronous.py src/zarr/core/group.py`
Expected: no new errors.

- [ ] **Step 4: Run the api + group test suites**

Run: `uv run pytest tests/test_api.py tests/test_group.py -q 2>&1 | tail -25`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/zarr/api/synchronous.py src/zarr/core/group.py
git commit -m "refactor: construct Array via _from_async_array at external call sites"
```

---

### Task 8: Update tests that touch `async_array` directly + add changelog

Existing tests read `arr.async_array.X` (e.g. `tests/test_array.py:384`, `:418`, `:419`). These now emit `DeprecationWarning`. Update them to use the new sync/async surface, or wrap in a warning filter where the intent is specifically to test the deprecated property.

**Files:**
- Modify: `tests/test_array.py`
- Create: `changes/<PR-number>.feature.md` (the repo uses towncrier with `.md` fragments, e.g. `changes/3826.feature.md`)

- [ ] **Step 1: Find tests referencing `async_array`**

Run: `grep -rn "\.async_array" tests/`

- [ ] **Step 2: Update each reference**

- `arr.async_array.nchunks` → `arr.nchunks` (the sync property).
- `await arr.async_array._nshards_initialized()` → `arr._runner.run(arr._nshards_initialized_async())` or, if the test is async, `await arr._nshards_initialized_async()`.
- `await arr.async_array.nchunks_initialized()` → `await arr.nchunks_initialized_async()` (async test) or `arr.nchunks_initialized` (sync property).

For any test whose explicit purpose is to verify `async_array` still works as a deprecated shim, wrap access in `pytest.warns(DeprecationWarning)` instead of removing it.

- [ ] **Step 3: Add the changelog fragment**

The repo uses towncrier with markdown fragments named `changes/<PR-number>.<type>.md` (e.g. `changes/3826.feature.md`). Create `changes/<PR-number>.feature.md` (substitute the actual PR number) containing:

```markdown
`Array` now owns its own state and accepts a keyword-only `runner` argument for plugging in a custom event loop. Every async operation is available as a `*_async` method on `Array`. `Array.async_array` is deprecated; use the `*_async` methods instead.
```

- [ ] **Step 4: Run the updated tests with deprecation-as-error**

Run: `uv run pytest tests/test_array.py -q -W error::DeprecationWarning 2>&1 | tail -30`
Expected: PASS (no un-suppressed deprecation warnings escape).

- [ ] **Step 5: Commit**

```bash
git add tests/test_array.py changes/
git commit -m "test: migrate async_array usages; add changelog for Array unification"
```

---

### Task 9: Full verification sweep

**Files:** none (verification only)

- [ ] **Step 1: Run the full array + sync + runner + api + group suites**

Run: `uv run pytest tests/test_array.py tests/test_runner.py tests/test_sync.py tests/test_api.py tests/test_group.py -q 2>&1 | tail -30`
Expected: all PASS.

- [ ] **Step 2: Run the complete test suite**

Run: `uv run pytest -q 2>&1 | tail -40`
Expected: PASS (or only pre-existing unrelated failures — compare against a clean `main` run if anything fails).

- [ ] **Step 3: Type-check the whole package**

Run: `uv run mypy src/zarr 2>&1 | tail -30`
Expected: no new errors.

- [ ] **Step 4: Run the linters / pre-commit**

Run: `uv run pre-commit run --all-files 2>&1 | tail -40`
Expected: PASS.

- [ ] **Step 5: Confirm the invariant holds**

Run: `grep -rn "self\.async_array\." src/zarr/core/array.py`
Expected: NO output. The only `async_array` reference in `array.py` is the deprecated property definition itself.

- [ ] **Step 6: Final commit (if any lint fixes were applied)**

```bash
git add -A
git commit -m "chore: lint/type fixes for Array unification"
```

---

## Self-Review notes

- **Spec coverage:** Runner protocol (Task 1) ✓; SupportsArrayState + widened helpers (Task 2) ✓; Array owns state + `runner` + deprecated `async_array` + `_from_async_array` (Task 3) ✓; property repoint (Task 4) ✓; selection `*_async` + runner routing (Task 5) ✓; remaining `*_async` + sync routing (Task 6) ✓; external construction sites (Task 7) ✓; test migration + changelog (Task 8) ✓; full regression + new runner tests (Task 9 + Tasks 1/3/5/6) ✓.
- **Discovery vs. spec:** The spec assumed several methods had inline bodies needing extraction; in fact the free functions already exist but take an array object. Section 3's "extract free functions" is therefore replaced by "widen existing free-function signatures to a Protocol" (Task 2) — a smaller, safer change that still achieves the single-source-of-truth goal.
- **Type consistency:** `getitem_async`/`setitem_async`/`resize_async`/`append_async`/`update_attributes_async`/`nchunks_initialized_async`/`_nshards_initialized_async`/`nbytes_stored_async`/`info_complete_async` and `_from_async_array`, `SupportsArrayState`, `Runner`, `SyncRunner` are used consistently across tasks.
- **Verification points where exact line numbers are approximate:** each such step instructs reading the surrounding lines first and preserving the current return contract, since line numbers will drift as edits land.
