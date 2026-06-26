# Backend-agnostic CRUD layer for zarr-python

Date: 2026-06-15
Status: approved
Branch: `zarrs-bindings`

## Goal

Turn the low-level functional CRUD API (introduced as `zarr.zarrs` earlier on
this branch) into a backend-agnostic layer, with the Rust zarrs bindings as one
of several interchangeable implementations. Define the CRUD contract abstractly,
provide a pure-Python reference backend (no Rust required), and make the zarrs
bindings conform to the same contract.

This validates the abstraction by having two real backends agree with each other
and with zarr-python, and it gives users a no-Rust fallback.

Non-goals for this change (deliberately deferred):

- Wiring the CRUD layer under zarr-python's own `Array`/`Group` classes.
- Entrypoint-based backend discovery (this change uses explicit import-time
  registration).
- A write-side region operation (`write_region`) remains future work.

## Background

The current `zarr.zarrs._api` is a flat module of 13 async functions that
delegate to the `_zarrs_bindings` Rust extension. It already separates two
concerns that this design formalizes into a hard boundary:

- **Backend-neutral glue:** `_normalize_selection`, `_array_shape`,
  `_chunk_dtype_and_shape`, numpy assembly (`np.frombuffer`/reshape/strided
  views), native-dtype coercion, options handling, error translation.
- **Genuinely zarrs-specific work:** producing/consuming raw chunk bytes,
  reading array subsets as bytes, writing metadata documents — all via
  `_zarrs_bindings` and the `_bridge.StoreShim`/`resolve_store` plumbing.

The public surface (`zarr.zarrs.decode_region`, etc.) is unreleased on this
branch, so it can move without backward-compatibility constraints.

zarr-python already contains everything a pure-Python backend needs:
`BatchedCodecPipeline` (`src/zarr/core/codec_pipeline.py`), `BasicIndexer`
(`src/zarr/core/indexing.py`), `save_metadata` (`src/zarr/core/metadata/io.py`),
metadata parsing (`ArrayV3Metadata.from_dict` / `ArrayV2Metadata.from_dict`),
and chunk-key encoding (`src/zarr/core/chunk_key_encodings.py`).

## Architecture

Two packages with a hard boundary.

### `zarr.crud` (new, backend-neutral)

- `_backend.py` — the `CrudBackend` `Protocol` (the narrow byte/metadata
  contract below) plus the canonical exceptions.
- `_api.py` — the shared async facade: the 13 public functions moved out of
  `zarr.zarrs`, holding all backend-neutral logic. Each function resolves a
  backend (from the `backend` argument or the registry default) and calls its
  byte/metadata methods, then does selection normalization, dtype handling, and
  numpy assembly.
- `_reference.py` — `ReferenceBackend`, pure Python, wrapping zarr-python's own
  codec/indexing/metadata machinery. Always importable; the default backend.
- `_registry.py` — `register_backend(name, backend)`, `get_backend(name)`, and
  the config-driven default resolution.
- `__init__.py` — re-exports the facade functions, `CrudBackend`,
  `ZarrsOptions`, the exceptions, and `register_backend`.

### `zarr.zarrs` (shrinks to the zarrs provider)

- `_backend.py` — `ZarrsBackend`, implementing `CrudBackend` by wrapping
  `_zarrs_bindings`. Owns the zarrs-isms that move out of the facade:
  `json.dumps` of the metadata dict, the `/`-prefixed zarrs node-path form
  (formerly `_node_path`), `_bridge.resolve_store`, and translation of
  `_zarrs_bindings` exceptions into the `zarr.crud` canonical exceptions.
- `_bridge.py` — unchanged (`StoreShim`, `resolve_store`).
- the Rust crate `zarrs-bindings/` and the construction cache — unchanged.
- registers itself as backend `"zarrs"` at import time.

## The `CrudBackend` contract

Narrow, byte/metadata level. Methods pass neutral types — the metadata document
as a `dict`, the zarr `Store`, and plain zarr paths (`""`, `"foo/bar"`) — and
return raw bytes / JSON-as-dict / `None`. Each backend serializes and bridges as
it needs.

```python
class CrudBackend(Protocol):
    async def create_array(self, store, path, metadata, *, overwrite: bool) -> None: ...
    async def create_group(self, store, path, metadata, *, overwrite: bool) -> None: ...
    async def read_metadata(self, store, path) -> dict[str, JSON]: ...
    async def read_chunk(self, store, path, metadata, coords) -> bytes: ...
    async def read_subset(self, store, path, metadata, start, shape) -> bytes: ...
    async def write_chunk(self, store, path, metadata, coords, data: bytes) -> None: ...
    async def delete_chunk(self, store, path, metadata, coords) -> None: ...
    async def delete_node(self, store, path) -> None: ...
    async def list_children(self, store, path) -> list[tuple[str, dict[str, JSON]]]: ...
```

Nine methods. `read_encoded_chunk` is deliberately **not** a backend method —
see below; it is a backend-independent facade helper over `store.get`.

Byte conventions: `read_chunk`/`read_subset` return C-contiguous raw bytes in the
array's native byte order for the requested chunk / step-1 bounding box;
`write_chunk` takes the same. `read_metadata`/`list_children` return parsed JSON
documents as dicts.

### Two read-addressing axes (no overlap)

Reads are addressed in one of two coordinate spaces, and the two never overlap:

- **Chunk-grid coordinates** — `read_chunk(coords)` / `read_encoded_chunk(coords)`
  return a whole chunk addressed by its grid position. `read_chunk` (a backend
  method) decodes and returns the *full* chunk shape, including the fill-padded
  overhang of edge chunks; `read_encoded_chunk` (a facade helper, not a backend
  method) returns the raw stored bytes or `None`. These pair with `write_chunk` /
  `delete_chunk`, which are chunk-grid-addressed backend methods.
- **Array-element coordinates** — `read_subset(start, shape)` returns an
  arbitrary box in array space, which generally spans multiple chunks and is
  clipped to the array bounds. The facade's `read_region(selection)` normalizes
  a numpy selection to a step-1 bounding box and calls it.

`read_chunk` takes no `selection` parameter. A sub-region *within* a single
chunk is simply a `read_region` whose bounding box lies inside one chunk; the
backend already decodes only the overlapping chunk(s) (sharding-aware in the
zarrs backend), so a chunk-relative partial-read needs no separate API. The
`Store.get(key, byte_range=)` analogue is therefore `read_region` over a
single-chunk box, not a parameter on `read_chunk`; `read_subset` itself has no
single-`get` analogue — it is closer to "`get_partial_values` across many keys,
stitched into one array."

### `read_encoded_chunk` is facade-level, not a backend method

Reading a chunk's raw stored bytes is just `store.get(chunk_key)`, and the chunk
key is computable from the metadata document alone via zarr-python's
`chunk_key_encoding` — no decoding, no codec pipeline, nothing backend-specific.
Both backends would implement it identically. So the facade implements
`read_encoded_chunk(store, path, metadata, coords)` directly as: encode the chunk
key from the metadata, `store.get` it, return the bytes or `None`. It works the
same regardless of which backend (or none) is selected, which is correct since it
is pure store I/O. Under sharding the chunk key holds the whole shard blob, and
this returns exactly that raw object.

This raw read can also be *expressed* through `read_chunk` by supplying a view
metadata document (`data_type: uint8` + a single `bytes` codec, identity decode)
— a nice demonstration that the read-only-view mechanism is general — but that
route requires knowing the encoded byte length up front to set the chunk shape (a
`store.getsize` round-trip) and would synthesize a fill-valued array for a missing
chunk instead of returning `None`. So `store.get` is the correct implementation
for fetching stored bytes; the view trick is the general tool for *reinterpreting*
decoded data under a different dtype/shape, which `read_chunk` already supports.

## Method naming

Both the public facade and the backend contract use a single, consistent verb
set: **create / read / write / delete / list**. No `decode`/`encode`/`retrieve`/
`store`/`erase` synonyms.

Public facade (`zarr.crud`):

| Function | Verb | Notes |
|---|---|---|
| `create_new_group` / `create_overwrite_group` | create | node lifecycle |
| `create_new_array` / `create_overwrite_array` | create | node lifecycle |
| `read_metadata` | read | array or group document |
| `read_chunk` | read | decoded chunk → `ndarray` |
| `read_encoded_chunk` | read | raw stored bytes, no decode (facade-only, `store.get`) |
| `read_region` | read | numpy basic-indexing selection → `ndarray` |
| `write_chunk` | write | encode + store a chunk |
| `delete_chunk` | delete | remove one chunk |
| `delete_node` | delete | remove a node + descendants |
| `list_children` | list | direct children of a group |

Facade → backend mapping for the byte-level methods: `read_chunk` →
`backend.read_chunk`, `read_region` → `backend.read_subset` (the facade
normalizes the selection to a step-1 bounding box `(start, shape)`),
`write_chunk` → `backend.write_chunk`, `delete_chunk` → `backend.delete_chunk`.
`read_encoded_chunk` maps to no backend method — the facade serves it from
`store.get`. The two distinct names `read_region` (selection-based, public) and
`read_subset` (bounding-box bytes, backend) are intentional: they have different
signatures and the facade is the adapter between them.

## Facade / backend split

What stays in the `zarr.crud` facade (written once, backend-neutral):

- selection normalization (`_normalize_selection`), shape/dtype resolution
  (`_array_shape`, `_chunk_dtype_and_shape`), native-dtype coercion;
- numpy assembly: `np.frombuffer(...).reshape(...)` and the strided/reversed/
  integer-axis post-index views; read-only result guarantee;
- the empty-selection short circuit (no backend call);
- `read_encoded_chunk`: encode the chunk key from the metadata and `store.get`
  it (no backend involved);
- `ZarrsOptions` acceptance (still a placeholder) and backend resolution.

What moves into each backend:

- `ZarrsBackend`: `json.dumps`, the `/`-prefixed node-path form,
  `resolve_store`, calling `_zarrs_bindings`, exception translation.
- `ReferenceBackend`: `ArrayV3Metadata.from_dict`/`ArrayV2Metadata.from_dict`,
  building a `BatchedCodecPipeline` and `ChunkGrid`/`BasicIndexer`, assembling
  `batch_info` and calling `codec_pipeline.read`/`write`, `save_metadata`,
  `store.delete_dir`, and `list_dir` + per-child metadata reads.

## Backend selection

- A registry in `zarr.crud._registry`: `register_backend(name, backend)`,
  `get_backend(name) -> CrudBackend`.
- A `zarr.config` key `crud.backend`, default `"reference"`. The pure-Python
  backend always works and is predictable; `"zarrs"` opts into the accelerator
  and is registered when `zarr.zarrs` is imported.
- Every facade function accepts `backend: CrudBackend | str | None = None`.
  `None` → registry default; a string → registry lookup; an instance → used
  directly. This enables side-by-side testing of backends.

## Error handling

`zarr.crud` defines the canonical exceptions: reuse
`zarr.errors.NodeNotFoundError`, and keep a `NodeExistsError` (exposed as
`zarr.crud.NodeExistsError`). Each backend raises these directly:

- `ReferenceBackend` raises them at the point of detection.
- `ZarrsBackend` translates `_zarrs_bindings.NodeExistsError` /
  `_zarrs_bindings.NodeNotFoundError` into the canonical types.

The facade therefore no longer needs the `_translate_errors` shim. Phase-1
fidelity limits (store-callback exceptions flattened to `RuntimeError` across the
Rust boundary) are unchanged for the zarrs backend; the reference backend
surfaces native exceptions directly.

## Testing

- Shared differential suite moves to `tests/crud/`, parametrized over
  `backend ∈ {reference, zarrs}` × `store ∈ {memory, local}`. Each test writes
  with zarr-python and reads through the facade (and vice versa), so the two
  backends are checked against zarr-python *and*, transitively, against each
  other. The zarrs-parametrized cases skip when `_zarrs_bindings` is not
  installed (xdist-safe module-level `importorskip` in a zarrs-only conftest
  helper, or a skip marker on the zarrs param).
- Zarrs-only tests stay in `tests/zarrs/`: the construction cache
  (`test_cache.py`) and the store bridge (`test_bridge.py`).
- A focused `tests/crud/test_registry.py`: default resolution, `register_backend`,
  string vs instance `backend=` override.
- `uv run --group zarrs pytest tests/crud tests/zarrs` is the full local check;
  `uv run pytest tests/crud` (no zarrs group) must pass with the reference
  backend alone and skip the zarrs params.

## Migration notes

- Move the 13 functions and the neutral helpers from `zarr.zarrs._api` into
  `zarr.crud._api`; delete `zarr.zarrs._api`. No aliases in `zarr.zarrs`.
- Rename to the consistent verb set in the move (no compatibility aliases, since
  the surface is unreleased): `decode_chunk` → `read_chunk`, `decode_region` →
  `read_region`, `encode_chunk` → `write_chunk`, `erase_chunk` → `delete_chunk`.
  `read_metadata`, `read_encoded_chunk`, `delete_node`, `list_children`, and the
  `create_*` functions keep their names.
- `zarr.zarrs.__init__` exports only what is needed to register and identify the
  zarrs backend (`ZarrsBackend`, and re-registers `"zarrs"` on import).
- The changelog fragment is updated to describe `zarr.crud` as the public CRUD
  surface with pluggable backends, and `zarr.zarrs` as the zarrs backend.
- The CI job continues to build the crate and now runs `tests/crud tests/zarrs`.
