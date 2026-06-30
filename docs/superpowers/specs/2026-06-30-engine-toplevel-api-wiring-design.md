# Wiring the top-level zarr-python API to `zarr.crud` via an `engine`

Date: 2026-06-30
Status: implemented
Branch: `zarrs-bindings`

## Goal

Let users drive zarr-python's ordinary top-level API (`create_array`,
`open_array`, `Array.__getitem__`/`__setitem__`, …) through a selectable
**engine**, so the same code runs on the native Python path or on a `zarr.crud`
backend (`reference`, `zarrs`, `zarrista`) without learning the low-level CRUD
API. This realizes the CRUD design doc's deliberately-deferred non-goal,
"wiring the CRUD layer under zarr-python's own Array/Group classes," and gives a
home to the also-deferred `write_region` operation.

## Scope

In scope (approved):

- A top-level `engine` config key (replacing `crud.backend`) plus a per-call
  `engine=` argument on the relevant top-level functions.
- Routing **data access** (`getitem`/`setitem`) and **creation/open**
  (`create_array`/`open_array`) through `zarr.crud` when a non-native engine is
  selected.
- A new `crud.write_region` facade function (the write counterpart to
  `read_region`).

Out of scope:

- Group-level CRUD wiring beyond what `create_array`/`open_array` need
  (`create_group`/`open_group` engine routing is a follow-up).
- Consolidated metadata, copy/save/load helpers.
- Any new indexing capability in the crud layer (it stays basic-indexing only).

## Decisions

- **Selection:** the engine is a per-array execution setting living in
  `ArrayConfig` (alongside `order`/`write_empty_chunks`/`read_missing_chunks`),
  defaulted from the `array.engine` config key (default `"zarr"`) and overridable
  per-array via a per-call `engine=` kwarg. Both supported, mirroring how the
  existing `ArrayConfig` fields work.
- **Error policy: strict.** When the selected engine cannot express an
  operation, raise — never silently fall back to native.
- **Advanced indexing** (`.oindex`/`.vindex`/`.blocks`) under a non-`"zarr"`
  engine raises `NotImplementedError`.
- **Phasing:** one spec, two implementation phases. Phase A lands and is
  verifiable on its own; Phase B adds writes.

## Background

`AsyncArray.getitem`/`setitem` are the **basic-indexing** entry points;
orthogonal/coordinate/block indexing live on separate accessors
(`.oindex`/`.vindex`/`.blocks`, which call `_get_selection`/`_set_selection`
with non-basic indexers). The `zarr.crud` facade is **basic-indexing only**
(`read_region` takes a `BasicSelection`; fancy indexing raises `TypeError`) and
operates on an explicit metadata document. These two facts make `getitem` /
`setitem` clean routing seams and make the strict policy natural: the accessors
crud cannot express simply raise.

The sync `zarr.Array` wraps an `AsyncArray` and forwards `__getitem__` /
`__setitem__`; storing the engine on the async object means the sync layer
inherits routing for free.

## Architecture

### 1. The `array.engine` config key

`src/zarr/core/config.py`: remove `"crud": {"backend": "reference"}`; add
`"engine": "zarr"` to the existing `"array"` namespace (next to `order`,
`write_empty_chunks`, `read_missing_chunks`). Valid values: `"zarr"` (native) and
the registered crud backend names (`"reference"`, `"zarrs"`, `"zarrista"`).

`zarr.crud._registry.get_backend(None)` resolves its default from
`config.get("array.engine")`, mapping `"zarr" → "reference"` (so direct crud
callers still get the pure-Python backend; the Array layer never routes to crud
with engine `"zarr"`).

### 2. Engine in `ArrayConfig`

Add an `engine: str` field to `ArrayConfig` (and `ArrayConfigParams`). Because
`ArrayConfig.from_dict` fills missing fields from `array.<field>` in the global
config, `engine` is resolved from `array.engine` at construction exactly like the
other fields. `AsyncArray` already stores `config: ArrayConfig`, so the engine
rides along with no new bespoke field. The per-call `engine: str | None = None`
parameter on the factory functions (async + sync `create_array`, `open_array`,
`create`, `open`) is folded into the array's `ArrayConfigParams`. The sync
`Array` reads the engine from its wrapped `AsyncArray`'s `config`.

**Re-entrancy guard (explicit).** Because `engine` defaults *from global config*,
a non-native global `array.engine` would otherwise propagate into the
`AsyncArray` that `ReferenceBackend.read_subset` constructs internally, causing
`getitem → crud → reference → getitem` to loop. The single re-entry site
(`ReferenceBackend.read_subset`, the only backend method that builds an
`AsyncArray` and calls `.getitem`) therefore constructs that array with an
explicit `engine="zarr"` config, pinning it to the native path regardless of the
global default. No other backend constructs an `AsyncArray`.

### 3. Read path — `AsyncArray.getitem` (Phase A)

```
self.config.engine == "zarr":  native path (unchanged)
otherwise:                      crud.read_region(self.metadata.to_dict(),
                                    self.store_path.store, self.store_path.path,
                                    selection, backend=self.config.engine)
```

`.oindex`/`.vindex`/`.blocks` getters: if engine != "zarr", raise
`NotImplementedError("engine {engine!r} supports only basic indexing; use the "
"native engine for orthogonal/coordinate/block indexing")`. Store/format limits
(e.g. `UnsupportedStoreError` from zarrista on a `MemoryStore`, or a v2 array
outside the supported subset) propagate from the backend unchanged.

### 4. Creation / open (Phase A)

- `create_array(..., engine=e)`: build the metadata document with the existing
  native machinery (param parsing is unchanged), then:
  - `e == "zarr"`: native creation (unchanged).
  - else: `await crud.create_new_array(metadata, store, path, backend=e)` and
    return an `AsyncArray` with `_engine=e`.
- `open_array(..., engine=e)`:
  - `e == "zarr"`: native open (unchanged).
  - else: `doc = await crud.read_metadata(store, path, backend=e)`, parse it, and
    construct the `AsyncArray` with `_engine=e`.

### 5. Write path — `crud.write_region` + `AsyncArray.setitem` (Phase B)

New facade function:

```python
async def write_region(metadata, store, path, selection, value, *, backend=None): ...
```

It decomposes the basic `selection` into chunk projections using zarr-python's
`BasicIndexer` over the metadata's chunk grid, then per chunk:

- **fully covered** by the selection: encode the slice of `value` and
  `backend.write_chunk(...)`.
- **partially covered** (boundary chunk): read-modify-write —
  `backend.read_chunk(...)` (fill value if absent), splice in the covered
  sub-region, `backend.write_chunk(...)`.

`AsyncArray.setitem`: `self.config.engine == "zarr"` native; else
`crud.write_region(..., backend=self.config.engine)`. Advanced accessors raise as
in the read path.

### 6. Sync layer

`zarr.Array.__getitem__`/`__setitem__` already forward to the async object via
`sync(...)`; no change beyond threading `engine=` through the sync factory
functions to their async counterparts.

## Error handling

| Situation | Behavior |
| --- | --- |
| `config.engine == "zarr"` (default) | native path, unchanged |
| basic getitem/setitem, non-native engine | route through crud |
| `.oindex`/`.vindex`/`.blocks`, non-native engine | `NotImplementedError` |
| un-ingestable store under `zarrista` | `UnsupportedStoreError` (from backend) |
| unknown engine name | `KeyError` from `crud.get_backend` |
| v2 array outside zarrs' supported subset | backend's own error |

## Testing (TDD, differential)

- **Config:** `array.engine` default is `"zarr"`; per-call `engine=` overrides
  config and lands in the array's `ArrayConfig`; unknown engine raises.
- **Read parity:** for a `LocalStore` array, `arr.getitem(sel)` under
  `engine in {"reference", "zarrista"}` equals the native result, across several
  basic selections (full, slices, integer axis, steps). `zarrs` param skipped
  when the extension is absent.
- **Strict raise:** `arr.oindex[...]`/`.vindex[...]`/`.blocks[...]` under a
  non-native engine raise `NotImplementedError`.
- **Store limits:** `engine="zarrista"` on a `MemoryStore` raises
  `UnsupportedStoreError`.
- **Re-entrancy:** with `array.engine="reference"` set globally, a getitem
  completes (no infinite recursion) — exercises the explicit `engine="zarr"` pin
  in `ReferenceBackend.read_subset`.
- **Creation/open:** `create_array(engine=e)` then native `open_array` reads the
  same data back; `open_array(engine=e)` carries the engine onto the returned
  array.
- **Write parity (Phase B):** `arr[sel] = v` under a non-native engine matches
  the native result, including a partial-chunk (read-modify-write) selection and
  a fill-value-only write (chunk erased).

## Phasing

- **Phase A:** `engine` config + resolver, `_engine` on `AsyncArray`, read-path
  routing, advanced-accessor raises, creation/open routing, and all Phase-A
  tests. Lands independently.
- **Phase B:** `crud.write_region`, setitem routing, write-path tests.

## Migration

`crud.backend` is unreleased on this branch, so replacing it with `array.engine`
needs no deprecation. Update the existing `tests/crud` and
`tests/zarrs`/`tests/zarrista` suites that read/set `crud.backend` to use
`array.engine` (or keep passing `backend=` per call, which is unaffected).
