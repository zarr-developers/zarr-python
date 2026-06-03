# Unifying `Array` and `AsyncArray` via a pluggable `runner`

Date: 2026-06-03
Branch: `one-array-class`

## Goal

Unify the `Array` and `AsyncArray` classes so that a single `Array` class owns
all array logic. Today `Array` is a thin synchronous wrapper that holds an
`AsyncArray` and delegates every operation through `sync(self.async_array.foo(...))`.
After this work, `Array` owns its own state and exposes every async operation as
a `*_async` coroutine method, with the synchronous methods implemented by running
that coroutine through a pluggable `_runner`.

The `runner` lets a user wire in their own event loop for the async side of array
operations, instead of being locked into Zarr's background-thread event loop.

## Scope and non-goals

- **In scope:** add a keyword-only `runner` argument to `Array.__init__`; give
  `Array` its own state and `*_async` methods; rewrite the synchronous methods to
  go through `_runner`; extract remaining inline async logic into shared
  module-level free functions; deprecate `Array.async_array` / `Array._async_array`.
- **Out of scope / non-goals:**
  - We do **not** modify `AsyncArray`. It stays exactly as it is and remains the
    public async class (compatibility shim). We simply stop wiring `Array` to it.
  - We do not remove `AsyncArray` or its public API in this PR. Future deprecation
    of `AsyncArray` itself is a separate effort.
  - No behavior change to the default synchronous path — the default runner
    preserves today's `sync()` semantics exactly.

## Design

### 1. The `Runner` protocol

Add to `zarr/core/sync.py`:

```python
@runtime_checkable
class Runner(Protocol):
    def run(self, coro: Coroutine[Any, Any, T]) -> T: ...
```

A `Runner` takes a coroutine and returns the value obtained by awaiting it.

Concrete default implementation, also in `zarr/core/sync.py`:

```python
class SyncRunner:
    """Run coroutines on Zarr's shared background event loop via sync()."""

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        return sync(coro)
```

There is **no** module-level mutable `DEFAULT_RUNNER` singleton baked into the
signature. `Array.__init__` accepts `runner: Runner | None = None`, and `None`
means "use the default", resolved to a `SyncRunner()` inside `__init__`. This
keeps the default-resolution logic in one place and avoids a shared mutable
default argument.

### 2. `Array.__init__` and state ownership

`Array` stops being a wrapper around `AsyncArray`. It owns the same state
`AsyncArray` holds, plus a `_runner`:

- `metadata: T_ArrayMetadata`
- `store_path: StorePath`
- `config: ArrayConfig`
- `codec_pipeline: CodecPipeline`
- `_chunk_grid: ChunkGrid`
- `_runner: Runner`

New signature (keyword-only `runner`):

```python
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
    # store metadata_parsed, store_path, config_parsed,
    # codec_pipeline, _chunk_grid, and (runner or SyncRunner())
```

`Array`'s field set no longer maps cleanly to a single `_async_array` field, so
the `@dataclass(frozen=False)` decorator is dropped in favor of this explicit
`__init__` (mirroring `AsyncArray`'s construction style). The `with_config`
overloads and other internals are updated to construct via the new signature.

#### Construction helper

Many internal call sites currently do `Array(async_array)` (≈10 sites across
`zarr/api/synchronous.py`, `zarr/core/group.py`, and `Array._create` /
`from_dict` / `open`). To keep these ergonomic and to handle the common
"I already have an `AsyncArray`" case, add:

```python
@classmethod
def _from_async_array(
    cls, async_array: AsyncArray[T_ArrayMetadata], *, runner: Runner | None = None
) -> Self:
    return cls(
        metadata=async_array.metadata,
        store_path=async_array.store_path,
        config=async_array.config,
        runner=runner,
    )
```

All existing `Array(async_array)` call sites are updated to
`Array._from_async_array(async_array)`. This is a mechanical change.

#### Deprecating `async_array` / `_async_array`

The `async_array` property is deprecated. On access it emits a
`DeprecationWarning` and constructs a fresh `AsyncArray` on demand from `Array`'s
own state:

```python
@property
def async_array(self) -> AsyncArray[T_ArrayMetadata]:
    warnings.warn(
        "Array.async_array is deprecated; ...",
        DeprecationWarning,
        stacklevel=2,
    )
    return AsyncArray(self.metadata, self.store_path, self.config)
```

The `_async_array` field is removed; any remaining internal uses are migrated to
`Array`'s own state.

### 3. Shared free functions (single source of truth)

The async selection methods on `AsyncArray` already delegate to module-level free
functions that take explicit state: `_getitem`, `_get_selection`,
`_get_orthogonal_selection`, `_get_mask_selection`, `_get_coordinate_selection`,
`_set_selection`, `_setitem`. These functions take `(store_path, metadata,
codec_pipeline, config, chunk_grid, ...)`.

For the `AsyncArray` methods whose logic is currently **inline** (no free
function yet), extract the body into a new module-level async free function taking
explicit state, then have both classes call it. Methods to extract:

- `resize`, `append`, `update_attributes`
- `nchunks_initialized`, `_nshards_initialized`, `nbytes_stored`
- `info_complete`, `_save_metadata`
- the classmethods/loaders as appropriate (`open`, `_create*`,
  `get_array_metadata` already exists as a free function)

Resulting call structure for each operation `foo`:

- `AsyncArray.foo(...)`  → `await _foo(self.metadata, self.store_path, ...)`
- `Array.foo_async(...)` → `await _foo(self.metadata, self.store_path, ...)`
- `Array.foo(...)`       → `self._runner.run(self.foo_async(...))`

For operations that return a *new* array (`resize` on v2, `update_attributes`,
`append`), the free function returns the new metadata/state; each class wraps the
result in its own type (`AsyncArray` vs `Array`, the latter preserving its
`_runner`).

This guarantees `AsyncArray` and `Array` cannot drift, because they share one
implementation per operation.

### 4. The `*_async` surface on `Array`

Every current public async method on `AsyncArray` gets a `*_async` twin on
`Array`:

- `getitem_async`, `setitem_async`
- async twins for each selection getter/setter:
  `get_orthogonal_selection_async` / `set_orthogonal_selection_async`,
  `get_mask_selection_async` / `set_mask_selection_async`,
  `get_coordinate_selection_async` / `set_coordinate_selection_async`,
  `get_block_selection_async` / `set_block_selection_async`,
  `get_basic_selection_async` / `set_basic_selection_async`
- `resize_async`, `append_async`, `update_attributes_async`
- `nchunks_initialized_async`, `nbytes_stored_async`, `info_complete_async`

The existing synchronous methods (`__getitem__`, `__setitem__`,
`get_basic_selection`, `set_basic_selection`, the orthogonal/mask/coordinate/block
selection getters and setters, `resize`, `append`, `update_attributes`,
`nchunks_initialized`, `nbytes_stored`, `info_complete`, …) are rewritten from
`sync(self.async_array.foo(...))` to `self._runner.run(self.foo_async(...))`.

### 5. Testing and verification

Verification bar: **full existing suite passes unchanged + new runner tests.**

- **Regression:** the entire existing array test suite passes without
  modification, proving the default synchronous behavior is preserved. Run with
  `uv run pytest`.
- **New tests:**
  - `Runner` protocol conformance / `SyncRunner` behaves as a `Runner`.
  - Injecting a custom runner: a recording runner that captures the coroutine it
    receives, asserts it is the expected coroutine, runs it, and returns the
    awaited value; assert `Array(..., runner=recording)` uses it.
  - Equivalence: `arr.getitem(sel)` equals `arr._runner.run(arr.getitem_async(sel))`
    and equals the value via a directly-awaited `getitem_async`.
  - Deprecation: accessing `Array.async_array` (and `_async_array` if still
    reachable) emits a `DeprecationWarning`.

## Risks

- Large mechanical diff in a 6000-line file; risk of missing a `sync(...)` call
  site. Mitigated by grepping all `sync(self.async_array` occurrences and by the
  unchanged regression suite.
- Free-function extraction for new-array-returning methods must correctly
  reconstruct per-class types. Covered by existing `resize`/`append`/
  `update_attributes` tests.
- `with_config` and other `Self`-returning methods must thread `_runner` through
  so a derived `Array` keeps the user's runner.
