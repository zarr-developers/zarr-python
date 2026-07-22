# Array engine protocol for zarr-python

Date: 2026-07-22
Status: approved
Branch: `claude/zarr-array-engine-protocol-239cfa`
Reviewed-by: @kylebarron (Zarrista developer), 2026-07-22 — feedback integrated

## Goal

Move the pluggable-backend boundary inside zarr-python's array classes:
`Array` wraps an object satisfying the `ArrayEngine` protocol, and
`AsyncArray` wraps an object satisfying the `AsyncArrayEngine` protocol (the
same contract with async methods). The engine owns the I/O data path — reading and writing
decoded data for selections — while `Array`/`AsyncArray` keep metadata
management, indexing normalization, and resize/append logic.

Two engine families ship:

- **Default engines** — the existing codec-pipeline machinery, moved behind
  the protocol. Always available, every store, Zarr v2 and v3. Behavior and
  performance are unchanged.
- **Zarrista engines** — powered by [Zarrista](https://pypi.org/project/zarrista/)
  (Development Seed's zarrs-backed low-level Zarr API). We build against
  Zarrista's `main` branch (unreleased; approved for use), pinned to a git
  commit; the Zarrista team can publish a new beta release on request, at
  which point the pin becomes `>=` that beta.

This strategy **replaces** the `zarr.crud` layer and the homemade Rust
bindings from earlier on this branch. `zarr.crud`, `zarr.zarrs`, and the
`packages/zarrs-bindings` crate are deleted (see Deletions).

Non-goals for this change:

- Group-level engine operations (listing, group creation). The hierarchy
  engine exists only to mint array engines in v1.
- Engine-owned metadata mutation (resize/append/update_attributes stay in
  `Array`/`AsyncArray`).
- Silent fallback between engines (see Error handling).

## Background: what Zarrista provides (main branch)

Sync `Array` and async `AsyncArray` are separate native classes:

- Construction: `open(store, path)` and `from_metadata(metadata, store, path)`
  (constructs without writing; `store_metadata()` persists). Metadata is typed
  as `zarr_metadata.ArrayMetadataV3` — our in-repo `packages/zarr-metadata`
  package, so `ArrayV3Metadata.to_dict()` output is already the right currency.
- Reads: `retrieve_array_subset(selection)` (numpy basic indexing, step-1
  slices, ndim-preserving), `retrieve_chunk`, `retrieve_encoded_chunk`, and
  sharding-aware `retrieve_subchunk` variants.
- Writes: `store_chunk` (decoded input as `ArrayBytes`; drops fill-value
  chunks), `store_encoded_chunk`, `erase_chunk`, `compact_chunk`. **No
  multi-chunk write** (`store_array_subset` is not exposed yet).
- Results: `DecodedArray = Tensor | VariableArray | MaskedTensor |
  MaskedVariableArray`. `Tensor` is zero-copy to numpy (buffer protocol,
  `__array__`, DLPack); `VariableArray` exports Arrow capsules.
- Stores: sync side takes Rust-native `FilesystemStore` / `MemoryStore`;
  async side takes an obstore `ObjectStore` or an icechunk `Session`. There is
  no bridge for arbitrary Python store objects.
- Zarr v3 only.

## Architecture

### Protocols (`zarr.abc.engine`)

Four runtime-checkable protocols, named without a `Protocol` suffix
(`ArrayEngine(Protocol)` is enough; implementations get names like
`ZarristaEngine`). The async pair:

```python
class AsyncArrayEngine(Protocol):
    async def read_selection(
        self,
        indexer: Indexer,
        *,
        prototype: BufferPrototype,
    ) -> NDArrayLike: ...

    async def write_selection(
        self,
        indexer: Indexer,
        value: NDBuffer,
        *,
        prototype: BufferPrototype,
    ) -> None: ...

    def with_metadata(self, metadata: ArrayMetadata) -> AsyncArrayEngine: ...


class AsyncHierarchyEngine(Protocol):
    def array_engine(
        self, path: str, metadata: ArrayMetadata
    ) -> AsyncArrayEngine: ...
```

`ArrayEngine` and `HierarchyEngine` are identical with sync methods and sync
return types.

Contract details:

- An **array engine is bound** to `(store, path, metadata)` at construction.
  Methods take only selection-level arguments, so engines can hold expensive
  per-array state (a constructed `zarrista.Array`, codec pipelines, caches).
- A **hierarchy engine is bound to a store** and mints array engines that
  share resources (the translated store handle, runtime, caches). This is the
  factory the registry uses; users may also construct an array engine directly
  and pass it to array creation/open.
- `read_selection` returns a buffer of the engine's own making: any object
  implementing at least `__array__` (numpy coercion), zero-copy where the
  backend allows. Additional export protocols (DLPack, Arrow) are welcome on
  concrete engines but are not part of the contract. There is **no `out`
  parameter** in v1; one may be added later as a performance improvement, and
  the facade meanwhile serves user-supplied `out=` arguments by copying the
  engine's result into them.
- The **engine speaks `Indexer`**: zarr-python's normalized selection plan
  (iterable of `(chunk_coords, chunk_selection, out_selection,
  is_complete_chunk)` projections with `shape` / `drop_axes`). Engines may
  consume the indexer wholesale (fast paths) or iterate its projections.
  (Reviewer note: Kyle raised chunk-level indexing as a simpler v1 contract;
  we are staying indexer-level per the approach decision — it is the only
  shape that keeps the default path unchanged.)
- `with_metadata` returns a rebound engine for the same store/path with new
  metadata. `resize`/`append`/`update_attributes` use it, so rebinding works
  uniformly for factory-made and user-provided engines.
- Facade-side responsibilities (not the engine's): building the indexer,
  `fields` handling, output dtype/order resolution, scalar extraction,
  `drop_axes` squeezing, copying into a user-supplied `out=` buffer, and the
  empty-selection short circuit.

### Wiring into `Array`/`AsyncArray`

- `AsyncArray` gains an `engine: AsyncArrayEngine` attribute. The
  module-level `_get_selection`/`_set_selection` shrink to: resolve output
  buffer arguments, call the engine, post-process. The current codec-pipeline
  body of those functions **becomes** the default async engine's
  implementation, byte-for-byte.
- `Array` gains `engine: ArrayEngine`. Its data methods
  (`__getitem__`/`__setitem__`, `get/set_basic_selection`,
  `get/set_orthogonal_selection`, `get/set_mask_selection`,
  `get/set_coordinate_selection`, `get/set_block_selection`, and the
  `oindex`/`vindex`/`blocks` accessors) call the sync engine directly — **the
  sync data path no longer touches the event loop**. Non-I/O operations
  (metadata, resize, create) keep the existing sync-over-async route.
- `DefaultAsyncArrayEngine` wraps store_path + metadata + codec pipeline +
  config. `DefaultArrayEngine` adapts it with `sync()` — so the default sync
  path is exactly as fast as today, while engines with native sync
  implementations (Zarrista on local files) skip asyncio entirely.

### Engine selection

- `engine=` parameter on array creation/open APIs, with three accepted forms:
  the string `"default"`, the string `"zarrista"`, or an
  `ArrayEngine`/`AsyncArrayEngine` instance the user constructed themselves
  (used as-is, sync vs async matching the entry point):

  ```python
  engine = ZarristaEngine(...)          # user-constructed, full control
  zarr.open_array(store, engine=engine)
  zarr.open_array(store, engine="zarrista")  # named, we construct it
  ```

- **No config key in v1.** Omitting `engine=` means the default (pure-Python)
  engine — today's behavior. A global config default can be added later.
- Name resolution is a small internal mapping from name to hierarchy-engine
  factory taking the zarr store; the resulting hierarchy engine is cached per
  `(name, store)` so array engines opened from one store share resources.
  This replaces the `zarr.crud` registry.

### Zarrista engines (`zarr.zarrista`)

- Array engines `ZarristaEngine` (sync) / `ZarristaAsyncEngine`, minted by
  `ZarristaHierarchyEngine` / `ZarristaAsyncHierarchyEngine`: the hierarchy
  engine is constructed with a zarr-python store, translating it at
  construction time:
  - sync: `LocalStore → zarrista.FilesystemStore`.
  - async: `zarr.storage.ObjectStore →` its underlying obstore instance;
    an icechunk store/session → `icechunk.Session`.
  - anything else (including zarr's `MemoryStore`, whose contents live in the
    Python process): raise immediately with a message naming the store type
    and the supported set.
- Array engines are built with
  `zarrista.Array.from_metadata(metadata.to_dict(), store, path)` (async:
  `AsyncArray.from_metadata`). Zarr v2 metadata raises immediately (Zarrista
  is v3-only).
- **Reads**: an indexer that is basic with all-step-1 slices maps directly to
  a zarrista `Selection` and one `retrieve_array_subset` call (all-Rust,
  sharding-aware, parallel). Any other indexer (steps, orthogonal,
  coordinate, mask, block) is served per-chunk: for each projection,
  `retrieve_chunk` → zero-copy numpy view → scatter into the output buffer.
  Rust still does all decoding; Python does the gather.
- **Writes**: per-chunk over the indexer's projections. A complete-chunk
  projection encodes the value directly via `store_chunk`; a partial chunk
  does read-modify-write: `retrieve_chunk` → patch with numpy → `store_chunk`.
  Kyle confirmed exposing `store_array_subset` (multi-chunk write) is easy on
  the Zarrista side; when it lands, basic-selection writes collapse to one
  Rust call with no protocol change.
- **Buffers**: `Tensor` → zero-copy numpy (`to_numpy`); `VariableArray` →
  numpy via Zarrista's upcoming numpy export (a copy for now — confirmed
  planned); `MaskedTensor`/`MaskedVariableArray` unsupported (zarr-python has
  no masked dtype) — raise.
- Registered under the name `"zarrista"` on import of `zarr.zarrista`; import
  errors cleanly when `zarrista` is not installed.
- Dependency: optional dependency group `zarrista`, git-pinned to a Zarrista
  `main` commit until its next release, then `>=` that release.

## Error handling

Fail loud, no silent fallback:

- Engine/hierarchy-engine construction raises `UnsupportedEngineError` (new,
  in `zarr.errors`) when the engine cannot serve the store or metadata
  (untranslatable store, v2 metadata, a dtype that decodes to a masked
  layout). The message names the incompatibility and the supported set.
- Operations an engine cannot perform raise `NotImplementedError`.
- Engines raise zarr-python's canonical exceptions where they exist
  (`zarr.errors` types); Zarrista exceptions are translated at the engine
  boundary.

## Testing

- `tests/engine/` differential suite: the same operations executed through
  the default and zarrista engines with results compared — reads/writes over
  basic (with and without steps), orthogonal, coordinate, mask, and block
  selections; partial-chunk RMW; fill-value handling; sharded arrays;
  vlen dtypes. Sync engines on `LocalStore`; async engines on an
  obstore local store. Zarrista params skip when `zarrista` is not installed
  (xdist-safe importorskip).
- Per the testing convention: one test function per operation family covering
  reasonable input combinations, plus one test function per error case
  (untranslatable store, v2 metadata, masked dtype, unsupported operation).
- Protocol conformance is checked statically: mypy verifies both default and
  zarrista engines satisfy the protocols (`runtime_checkable` isinstance
  checks only verify method presence).
- The sync zarrista path gets a no-event-loop regression test (engine methods
  run with no running loop and never create one).
- CI: drop the Rust-crate build job; add a job installing the pinned Zarrista
  and running `tests/engine`.

## Deletions

- `src/zarr/crud/` (protocol, facade, reference backend, registry).
- `src/zarr/zarrs/` and `packages/zarrs-bindings/` (the homemade PyO3 crate,
  its construction cache, store shim, and bridge).
- `tests/crud/`, `tests/zarrs/`, and the crate-build CI wiring.
- Changelog fragments describing `zarr.crud`/`zarr.zarrs` are replaced by one
  describing the engine protocol and the zarrista engine.

The differential-testing idea from `zarr.crud` survives as `tests/engine/`;
the store-translation and metadata-handoff learnings carry into
`zarr.zarrista`.

## Feature requests for Zarrista (tracked, not blocking)

Kyle has reviewed this list and is open to all of them; (1) is confirmed easy
and (a copying) numpy export for `VariableArray` is already planned.

1. **`store_array_subset`** — zarrs has it natively; exposing it moves
   partial-chunk read-modify-write into Rust and collapses the per-chunk
   write path for basic selections.
2. **Python store bridge** — would lift the store-translation restriction and
   let any zarr-python `Store` back a zarrista engine.
3. **Step > 1 slices in `Selection`** — widens the read fast path to strided
   basic selections.
4. **Zarr v2 read support** — zarrs supports a v2 subset; exposing it would
   let the zarrista engine serve v2 arrays.

## Decision log

- Engine protocols replace `zarr.crud`/`CrudBackend` entirely; homemade Rust
  bindings dropped in favor of Zarrista (main branch approved).
- Engine scope is I/O only; metadata management stays in the array classes.
- Both acquisition paths: hierarchy engine as factory (resource sharing) and
  user-provided array engines at creation time.
- Sync path is truly sync: `Array` calls its sync engine directly.
- Store bridging by translating known stores; fail loud otherwise.
- No fallback between engines; errors are immediate and specific.
- Region writes: protocol is selection-level; zarrista engine does
  Python-side per-chunk RMW now, upgradeable when `store_array_subset` lands.
- Protocol granularity: engines speak `Indexer` (approach A), keeping the
  default path unchanged and giving engines full selection information.

Post-review (kylebarron, on the shared gist):

- Protocol names drop the `Protocol` suffix: `ArrayEngine`,
  `AsyncArrayEngine`, `HierarchyEngine`, `AsyncHierarchyEngine`.
- No `out` parameter in the v1 protocol; revisit later for performance.
- Engine read results only need `__array__`; other export protocols
  (DLPack, Arrow) are optional extras on implementations.
- No config key in v1: engine choice is explicit via
  `engine="default" | "zarrista" | <instance>`; a global config default may
  come later.
- Zarrista can cut a beta release on request, so the git pin is temporary.
- Kyle's open question — chunk-level indexing as a simpler v1 engine
  contract — noted; staying indexer-level (approach A) to keep the default
  path byte-for-byte unchanged.
