# What Zarrista should change for a fluent zarr-python binding

Date: 2026-07-23
Status: discussion draft (audience: the Zarrista team)
Context: companion to `2026-07-23-zarr-python-blockers-for-engine-integration.md`.
zarr-python now routes array I/O through pluggable engines, and
`zarr.zarrista` implements them over Zarrista's `main` branch (pinned at
`95e47ad`). The binding works and passes a differential suite against the
pure-Python engine, but several Zarrista-side gaps forced Python-side
workarounds, per-read copies, or outright feature rejection. Each item below
states current behavior, the concrete cost it imposed on the binding, and
the ask — ranked by impact.

## What already works well (please don't change it)

- **`zarr_metadata` as the metadata currency.** `from_metadata` accepting
  the `ArrayMetadataV3` TypedDict means zarr-python's
  `ArrayV3Metadata.to_dict()` output is directly consumable — no
  serialization shim, no divergent metadata model. This was the single
  biggest reason the binding is thin.
- **Native sync and async classes.** zarr-python's sync `Array` now calls
  Zarrista's sync API with no event loop anywhere in the caller thread —
  only possible because the sync side is genuinely sync.
- **Ndim-preserving `Selection` semantics.** Matches the binding's `Region`
  interchange exactly; `read_selection` is a single
  `retrieve_array_subset` call with zero dispatch.
- **`from_metadata` constructing without writing.** Lets the binding open
  arrays zarr-python already manages without touching the store.

## 1. Multi-chunk writes: expose `store_array_subset`

**Today.** Writes are chunk-at-a-time (`store_chunk`). The binding's
`write_selection` therefore decomposes the region over the chunk grid in
Python: full chunks encode directly, partial chunks read-modify-write via
`retrieve_chunk` → numpy patch → `store_chunk`.

**Cost.** The decomposition arithmetic (overlap slices, edge-chunk
clipping, the "full chunk" test that must also compare against the nominal
chunk shape) was the highest-risk code in the binding — one review
correction and a 65-region differential fuzz were needed to trust it. All
of it is a Python re-implementation of what zarrs already does natively,
serially, with a Python round-trip per chunk.

**Ask.** Expose zarrs's `store_array_subset` on `Array`/`AsyncArray`. The
binding's write path collapses to one call, RMW moves into Rust with
proper parallelism, and the edge-case surface disappears. (Confirmed easy
in earlier review — this is the formal request.)

## 2. Unify the sync and async store worlds

**Today.** Sync `Array` accepts `FilesystemStore | MemoryStore`; async
`AsyncArray` accepts obstore `ObjectStore | icechunk Session`. The sets are
disjoint.

**Cost.** The disjoint sets leak all the way up zarr-python's stack. A
zarr-python array always carries both a sync and an async face; because no
single store satisfies both Zarrista sides, the binding cannot resolve both
engines eagerly and had to make engine construction lazy on both sides
(deferred store translation, nullable cached engines). Sync users get no
object storage; async users get local files only through obstore's
`LocalStore`.

**Ask.** Accept the same store set on both sides — most simply, blocking
sync wrappers over the async stores (sync `Array.open` taking an obstore
`ObjectStore`), and/or `FilesystemStore` on the async side. One store
handle usable from both `Array` and `AsyncArray` would let the binding
translate once and bind both engines eagerly.

## 3. A Python store bridge (callback store)

**Today.** Only Zarrista-native stores work. Arbitrary zarr-python `Store`
implementations — `MemoryStore`, fsspec-backed stores, test doubles,
anything third-party — cannot back a Zarrista array at all; the binding
fails loudly with "unsupported store."

**Cost.** `engine="zarrista"` silently partitions the store ecosystem.
`MemoryStore` needs a bespoke error explaining Rust-vs-Python address
spaces; every in-memory test of the binding must use `LocalStore` +
tmp-dirs instead.

**Ask.** A `PyObjectStore`-style adapter: a Zarrista store that calls back
into a user-supplied Python object implementing get/set/delete/list (zarrs
supports custom stores; our deleted homemade bindings had exactly this
shim). Slower than native paths, but it converts "unsupported store" into
"works, at Python-callback speed" — and makes the whole binding testable
in memory.

## 4. Writable (or ownership-transferring) read buffers

**Today.** `Tensor`'s buffer-protocol export is read-only;
`np.asarray(tensor)` yields `writeable=False`.

**Cost.** zarr-python's API contract is that reads return writable arrays.
The binding now detects read-only numpy results on the fast path and
copies — so every full-box zarrista read pays a copy that the zero-copy
design was meant to avoid. (Latent-bug history: before the guard, `arr[:]`
via zarrista returned read-only arrays and in-place user code broke.)

**Ask.** Use DLPack's existing semantics — no new protocol. Of the three
standard interchange models, only one fits writable hand-off:

- The **buffer protocol** is shared-ownership-by-refcount; writability is a
  per-export decision (`PyBUF_WRITABLE`). A writable export is sound iff
  the Rust side never touches the allocation after exporting — acceptable
  as a fallback, but the safety condition is invisible convention.
- **Arrow C data** (already used for `VariableArray`) is release-callback
  ownership over **immutable-by-contract** buffers; it has no writable
  story and should not be bent into one.
- **DLPack** is built for this. `Tensor.__dlpack__` already exists and
  speaks the versioned protocol (`max_version`). Two spec-compliant
  options, both with heavy precedent:
  1. *Shared writable view (the PyTorch/CuPy pattern)*: export the
     `DLManagedTensorVersioned` **without** `DLPACK_FLAG_BITMASK_READ_ONLY`,
     with a deleter that holds a strong reference to the `Tensor`.
     `np.from_dlpack` then yields a writable ndarray aliasing the Rust
     allocation; sound because a freshly decoded result is never touched
     by Rust again. Smallest change.
  2. *True move*: the deleter owns the allocation directly (drops the
     `Vec`); the `Tensor` tombstones itself and further access raises. The
     capsule's one-shot convention (`used_dltensor` rename) already
     enforces single consumption on the consumer side.

Either way, the binding's read path becomes `np.from_dlpack(result)` on
the fast path and the writability copy-guard never fires for zarrista.

## 5. A missing-chunk policy on reads

**Today.** `retrieve_chunk`/`retrieve_array_subset` silently materialize
fill values for absent chunks. There is no way to ask "and tell me (or
raise) if a chunk was missing."

**Cost.** zarr-python has `read_missing_chunks=False` (raise
`ChunkNotFoundError` on uninitialized chunks). The default engine honors
it; the zarrista engine cannot, so the binding *rejects* that config with
`UnsupportedEngineError` — a documented capability gap between engines.

**Ask.** A codec-option (or retrieve variant) with a missing-chunk policy:
`fill` (today's behavior) | `error`, or alternatively return a
chunks-present mask alongside the data. Either form lets the binding honor
the config instead of refusing it.

## 6. Make `VariableArray → numpy` either work or raise

**Today.** vlen results (`VariableArray`) export Arrow capsules; the
planned numpy export (copying) has not landed. Critically,
`np.asarray(variable_array)` today neither works nor raises — it silently
produces a wrong-shape 0-d object array.

**Cost.** The binding had to type-gate every decode (`Tensor`-only) because
the generic numpy coercion path silently corrupts vlen reads — this was
caught by a review probe, not by an exception.

**Ask.** Two parts: (a) land the numpy export for `VariableArray` (a copy
is fine to start) so vlen dtypes work through the binding; (b) until then,
make `__array__` on non-exportable types raise `TypeError` rather than
silently misconverting — silent wrong answers are the worst failure mode a
binding can inherit.

## 7. Strided selections (`step > 1`)

**Today.** `Selection` supports integers and step-1 slices only.

**Cost.** Any strided read (`arr[::2]`) leaves the fast path: the binding
fetches the step-1 bounding box and post-indexes in Python — transferring
and decoding up to `step×` more data than needed.

**Ask.** Accept arbitrary positive steps in `Selection` (zarrs subsets can
express this via per-chunk mapping even if the core type is contiguous).
Negative steps stay a Python-side concern; positive strides are the common
scientific-access pattern worth doing in Rust.

## 8. Subchunk-level writes for sharded arrays

**Today.** Reads are shard-aware (`retrieve_subchunk` reads one inner
chunk without touching the rest of the shard), but there is no
`store_subchunk` — writes operate on whole chunks, i.e. whole shards.

**Cost.** A one-element write into a large shard round-trips the entire
shard through Python (`retrieve_chunk` of the shard → patch →
`store_chunk`). The read side's nice granularity has no write mirror.

**Ask.** `store_subchunk` (or make `store_array_subset` from §1
shard-aware, which subsumes this). Sharded-append workloads are where
zarrs's performance story is strongest; the write path should match.

## 9. Zarr v2 read support

**Today.** `from_metadata` accepts `ArrayMetadataV3` only.

**Cost.** The binding hard-rejects v2 arrays (`UnsupportedEngineError`),
so mixed-version hierarchies silently split into "accelerated" and
"not accelerated" populations.

**Ask.** zarrs supports a V3-compatible subset of V2; exposing
`from_metadata` for the v2 document (or an internal v2→v3-view conversion)
would let the binding serve most real-world v2 data read-only, which is
the dominant v2 use case.

## 10. Smaller ergonomic papercuts

- **`ArrayBytes` rejects multi-dimensional buffers.** The binding flattens
  every chunk with `reshape(-1).view(np.uint8)` before `store_chunk`.
  Accepting any C-contiguous buffer (validating length against the chunk
  shape) would remove a copy-risk and a line of ceremony per write.
- **Path convention.** Zarrista paths are `/`-prefixed with `/` as root;
  zarr-python paths are `""`-rooted relatives. The binding normalizes with
  `"/" + path.strip("/")` everywhere — accepting both forms would delete a
  papercut every consumer rediscovers.
- **Icechunk sessions are reconstructed, not shared.** The session is
  serialized into the Rust extension and runs as a separate icechunk
  instance; `in_memory_storage()` cannot work, and writes made through the
  reconstructed session interact opaquely with the Python-side session's
  transaction state. A shared-instance handoff (or a documented
  commit-visibility contract) would make icechunk + zarrista trustworthy
  for read-write use.
- **Release cadence.** The binding pins a git commit. A beta release with
  §1 (and ideally §4/§6) would let zarr-python depend on
  `zarrista>=0.1.0bN` instead of a SHA.

## Priority from the binding's perspective

1. §1 `store_array_subset` — deletes the riskiest Python code in the
   binding and is already agreed to be easy.
2. §4 writable/ownership read buffers — removes a copy from every fast-path
   read; the zero-copy story is currently negated by the writability gap.
3. §2 + §3 store-world unification and the Python bridge — these two
   dissolve the store-translation layer and most of the lazy-engine
   machinery on the zarr-python side.
4. §5 missing-chunk policy and §6 vlen export — close the two documented
   capability gaps between the zarrista and default engines.
5. §7–§9 — performance breadth (strided, sharded writes, v2).
6. §10 — papercuts, any time.
