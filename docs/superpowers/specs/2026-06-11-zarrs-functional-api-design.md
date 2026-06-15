# zarrs-backed low-level functional API for zarr-python

Date: 2026-06-11
Status: approved
Branch: `zarrs-bindings`

## Goal

Give zarr-python a low-level, functional API for zarr hierarchy CRUD whose
implementation delegates to the Rust [`zarrs`](https://docs.rs/zarrs) crate via
new PyO3 bindings. Every array routine takes a metadata document as an explicit
parameter, so callers can operate on read-only or virtual views of arrays
(e.g. decode a chunk with metadata the store never saw, or read a chunk as raw
bytes without decoding).

Non-goals for this work: rewiring zarr-python's `Array`/`Group` classes or the
codec-pipeline registry through this API (possible later), fancy
(non-slice) indexing, and use of zarrs's experimental async feature.

## Background

- zarr-python is pure Python (hatchling). Its `Store` ABC
  (`src/zarr/abc/store.py`) is async; metadata classes live under
  `src/zarr/core/metadata/`.
- The Rust `zarrs` crate (~0.23) supports exactly the metadata-driven shape we
  need: `Array::new_with_metadata(storage, path, metadata)` and
  `Group::new_with_metadata(...)` construct nodes from a metadata document
  without touching the store; `store_metadata()` persists separately. Chunk and
  region I/O: `retrieve_chunk`, `retrieve_encoded_chunk` (raw bytes),
  `retrieve_array_subset`, `partial_decoder` (sharding-aware), and the
  corresponding `store_*` methods. `ArrayMetadata`/`GroupMetadata` parse
  directly from JSON strings (v2 or v3; v2 converts internally).
- The existing `zarrs` PyPI package (github.com/zarrs/zarrs-python) exposes only
  a codec pipeline (`CodecPipelineImpl`) and supports only a fixed set of
  native stores. It cannot provide the API designed here, but its build setup
  (maturin, PyO3 abi3, tokio/rayon) is the reference for ours.

## Architecture

Two distributions in this repo, hard boundary between them:

1. **Rust crate `zarrs-bindings`** at the repo root (`zarrs-bindings/`),
   built with maturin (PyO3, `abi3-py312`), publishing wheel `zarrs-bindings`
   with native module `_zarrs_bindings`. It is a thin, mechanical binding over
   `zarrs`: functions/pyclasses take metadata as a **JSON string**, a
   store-config object, a node path, and return bytes / numpy arrays. It knows
   nothing about zarr-python except the store sniffing described below.
2. **Python subpackage `zarr.zarrs`** in zarr-python: the public functional
   API. Owns conversion between zarr-python types (`dict` metadata documents,
   `zarr.abc.store.Store`, numpy arrays) and the binding layer, plus
   validation, ergonomics, and error translation. Imports `_zarrs_bindings`
   lazily and raises a helpful `ImportError` naming the `zarr[zarrs]` extra if
   it is missing.

zarr-python's own wheel remains pure Python; `zarrs-bindings` becomes an
optional dependency (`zarr[zarrs]`).

## Public API (`zarr.zarrs`)

All functions are `async def`. Parameters:

- `metadata`: `dict[str, JSON]` — the literal metadata document (`zarr.json`,
  or v2 `.zarray`/`.zgroup` equivalents). Never read from the store by the
  array routines.
- `store`: `zarr.abc.store.Store`.
- `path`: node path within the store (str, `""` = root).
- `chunk_coords`: `tuple[int, ...]` grid coordinates.
- `selection`: numpy-style basic indexing — integers, slices (including steps; strided/reversed selections fetch the step-1 bounding box in one call and apply numpy views), and `Ellipsis`. Fancy indexing (integer/boolean arrays) and `np.newaxis` are not supported.
- `options`: every function also accepts keyword-only
  `options: ZarrsOptions | None = None` (omitted from the signatures below for
  brevity) — a dataclass holding concurrency limits and checksum validation
  flags. Defaults are applied when omitted; in Phase 1 the dataclass exists
  but carries only defaults (fields become meaningful in Phase 3).

```python
# node lifecycle
async def create_new_group(metadata, store, path) -> None          # error if node exists
async def create_overwrite_group(metadata, store, path) -> None
async def create_new_array(metadata, store, path) -> None
async def create_overwrite_array(metadata, store, path) -> None
async def read_metadata(store, path) -> dict[str, JSON]            # array or group doc
async def delete_node(store, path) -> None
async def list_children(store, path) -> list[tuple[str, dict]]     # (path, metadata)

# chunk-level I/O
async def decode_chunk(metadata, store, path, chunk_coords, *, selection=None) -> np.ndarray
async def read_encoded_chunk(metadata, store, path, chunk_coords) -> bytes | None
async def encode_chunk(metadata, store, path, chunk_coords, value) -> None
async def erase_chunk(metadata, store, path, chunk_coords) -> None

# region-level I/O (selection in array coordinates, may span chunks)
async def decode_region(metadata, store, path, selection) -> np.ndarray
async def encode_region(metadata, store, path, selection, value) -> None
```

Mapping to zarrs primitives:

| API function | zarrs primitive |
|---|---|
| `create_new_group` / `create_overwrite_group` | `Group::new_with_metadata` + `store_metadata` (existence check first for `new`) |
| `create_new_array` / `create_overwrite_array` | `Array::new_with_metadata` + `store_metadata` |
| `read_metadata` | `Array::open` / `Group::open` metadata retrieval |
| `delete_node` | `erase_metadata` + chunk erasure / prefix delete |
| `list_children` | `Group::children` / `traverse` |
| `decode_chunk` (no selection) | `retrieve_chunk` |
| `decode_chunk` (selection) | `partial_decoder(chunk).partial_decode` (sharding-aware) |
| `read_encoded_chunk` | `retrieve_encoded_chunk` |
| `encode_chunk` | `store_chunk` |
| `erase_chunk` | `erase_chunk` |
| `decode_region` | `retrieve_array_subset` |
| `encode_region` | `store_array_subset` |

## Store bridge

A Rust-side `StoreConfig` resolver, tried in priority order:

1. `zarr.storage.LocalStore` → native `zarrs_filesystem` store.
2. obstore-backed `ObjectStore` → `zarrs_object_store` (Phase 3).
3. **Anything else** → generic `PyStore`: a Rust struct implementing
   `ReadableStorageTraits` / `WritableStorageTraits` /
   `ListableStorageTraits` over a Python callback object.

The callback path: the async API function wraps the user's `Store` in a small
sync Python shim whose methods submit coroutines to zarr-python's existing
sync event-loop thread (`zarr.core.sync`,
`asyncio.run_coroutine_threadsafe(...)` + blocking result). Rust calls the
shim while holding no locks of its own. This makes any conformant `Store`
(Memory, Zip, Logging, Wrapper, user-defined) work without Rust knowing its
type. Deadlock safety relies on the existing invariant that code running on
the zarr sync loop never blocks on these Rust entry points.

## Sync/async seam

The public API is async to match zarr-python conventions. Internally each
function calls a blocking Rust entry point via `asyncio.to_thread`; the Rust
side releases the GIL during I/O and compute (reacquiring it only inside
`PyStore` callbacks). zarrs's experimental async feature is not used.

## Array construction cache

`Array::new_with_metadata` (serde-parsing the metadata document and building the
codec chain) is the dominant per-call cost on the native path — measured at
~20µs for a bytes-only array up to ~80µs for sharded+blosc, against single-digit
µs of actual chunk I/O on a warm filesystem. To amortize it across the common
"open one array, then do many chunk operations" pattern, the chunk/region
routines memoize the constructed `Array` in a process-global LRU cache
(capacity 128) keyed on `(filesystem root, node path, metadata JSON)`.

This is safe because a zarrs `Array` caches no chunk data — it is metadata plus
codec chain plus a storage handle — so every read/write still goes through to
the store, and a correctly-keyed hit is behaviorally identical to a fresh build.
The key must include all three components: the same document at a different path
or store is a different array. Only native filesystem stores are cached; the
generic `PyStore` callback path has no stable cross-call identity to key on and
is left uncached (a future change may cache it if a store can supply a stable
value-based token). No invalidation hook is needed: delete/overwrite with
different metadata yields a different key, and an entry for a deleted-and-rebuilt
array with identical metadata stays valid because reads go through to the store.
A poisoned cache mutex is recovered rather than propagated, so the cache can
never wedge array I/O. Measured win: 14–20% faster per repeated call on a local
store, free on every hit.

## Error handling

The binding layer raises a small set of typed exceptions defined in one place:
`NodeExistsError`, `NodeNotFoundError`, and `ValueError` subclasses for
metadata-parse failures. In Phase 1 the translation surface is deliberately
small: `zarr.zarrs` re-raises the bindings' `NodeNotFoundError` as
`zarr.errors.NodeNotFoundError`; `NodeExistsError` is exposed as
`zarr.zarrs.NodeExistsError`. Exceptions raised by Python store callbacks are
flattened to a `RuntimeError` carrying the original message — the original
exception type and traceback are lost crossing the Rust boundary. Faithful
propagation of store-callback exceptions (and richer mapping onto
`zarr.errors` types) is deferred to a later phase.

## Testing

`tests/zarrs/`, module-level skip when `_zarrs_bindings` is not importable.

- **Differential tests** are the core: every operation checked against
  zarr-python's own implementation on the same store — write with zarr-python,
  read with zarrs, and vice versa; metadata documents produced by both must
  round-trip.
- Parametrized over: `MemoryStore` (exercises generic bridge) and `LocalStore`
  (native path); zarr formats v2 and v3; a codec matrix including
  `sharding_indexed`.
- Read-only-view tests: decode a chunk using a metadata dict not present in
  the store; `read_encoded_chunk` returns bytes identical to `store.get`.
- A CI job builds the crate with `maturin develop` and runs `tests/zarrs/`.
  Existing CI jobs are untouched (the suite skips without the extension).

## Phasing

1. **Phase 1**: crate scaffolding (maturin, CI build), store bridge (native
   LocalStore + generic PyStore), node lifecycle functions, whole-chunk
   `decode_chunk` / `read_encoded_chunk` / `encode_chunk` / `erase_chunk`.
2. **Phase 2**: `decode_region` (read side of region I/O) is implemented on
   this branch. `encode_region` and chunk-subset `selection` for `decode_chunk`
   via partial decoders remain Phase 2.
3. **Phase 3**: `ZarrsOptions` surface (concurrency, checksum validation,
   direct IO), obstore native path, benchmarks vs. the pure-Python pipeline.

## Naming decisions

- Python API: `zarr.zarrs`.
- Rust crate / PyPI distribution: `zarrs-bindings` (PyPI name `zarrs` is taken
  by the existing project); native module `_zarrs_bindings`.
- Function names follow the requested `create_new_*` / `create_overwrite_*`
  pattern; reads are `decode_*` / `read_*`, writes `encode_*`.
