# What zarr-python should change for engines to fit naturally

Date: 2026-07-23
Status: discussion draft
Context: post-mortem of the array-engine-protocol branch
(`claude/zarr-array-engine-protocol-239cfa`, spec:
`2026-07-22-array-engine-protocol-design.md`). The engine layer works and is
fully tested, but building it meant working *around* several patterns baked
into zarr-python rather than *with* them. This document names those patterns,
shows the concrete friction each caused on the branch, and proposes the
change that would have made — and would still make — the integration smooth.

Ranked by how much friction each caused.

## 1. The `Array` / `AsyncArray` dual-class architecture

**The pattern.** `Array` is a wrapper holding an `AsyncArray`; every sync
operation historically round-tripped through `sync()` onto the loop thread
(22 `sync()` call sites remain in `array.py`). Every open constructs both
layers unconditionally.

**The friction.** The engine concept is *exactly* a sync/async boundary — so
the class pair duplicates it. Consequences on the branch:

- Every array conceptually needs **two** engines (sync for `Array`, async for
  `AsyncArray`), but constructing both eagerly is impossible for zarrista:
  its sync side translates only `LocalStore`, its async side only
  obstore/icechunk — disjoint sets, so dual eager resolution fails on
  *every* store. We papered over this twice: `Array._engine` is a lazily
  resolved nullable cache slot, and `ZarristaAsyncEngine` defers store
  translation to first I/O. Both are workarounds for "the wrapper class
  forces two engines where one is ever used."
- `Array` is a dataclass, so a constructor keyword `engine=` collides with
  the `engine` property (verified: it silently corrupts the field default) —
  hence the awkward internal `engine_spec: InitVar` spelling.
- Sync entry points must thread the engine spec to *both* layers so sync and
  async access agree — more plumbing per entry point.

**The change.** Make the engine the *only* sync/async boundary. One `Array`
class parameterized by its engine; `.async_` / `.sync_` access becomes a
property of the engine binding, not a parallel class hierarchy. Short of
that: stop having `Array` eagerly construct an `AsyncArray` (peer classes
over shared `(store, path, metadata, config)` state, each resolving only its
own engine on demand).

## 2. Creation/open API sprawl

**The pattern.** Twelve `create_array`/`open_array`/`create`/`from_array`/
`init_array` entry points across five files (`zarr.api.synchronous`,
`zarr.api.asynchronous`, `zarr.core.array`, `zarr.core.group`), each
duplicating a ~25-keyword signature.

**The friction.** Adding one keyword (`engine=`) meant editing every entry
point and every intermediate hop (15 mentions in `asynchronous.py`, 24 in
`synchronous.py`). Task 7 was an entire plan task whose content was pure
signature threading, and the old branch tip independently hit a bug in
exactly this plumbing (`open_array`'s create-fallback dropping `engine`).
The still-open `open_array(config=...)`-ignored-on-existing-arrays gap is
the same disease: `AsyncArray.open` simply has no `config` parameter.

**The change.** Collapse creation/open onto one internal path taking a
single spec object (an `ArrayParams`/builder holding shape, dtype, chunks,
codecs, config, engine, ...). Public functions become thin adapters that
construct the spec; the sync API is generated mechanically from the async
one. New cross-cutting parameters then land in exactly one place.

## 3. `Store` hides the native resource

**The pattern.** `zarr.abc.store.Store` is an async Python byte interface.
That is the right abstraction for Python codecs, but a native engine does
not want Python byte callbacks — it wants the *underlying handle*: a
filesystem root, an obstore instance, an icechunk session.

**The friction.** `zarr.zarrista._translate` is isinstance-sniffing:
`LocalStore → FilesystemStore` via `.root`, `ObjectStore → .store`,
icechunk via a guarded import. Third-party stores cannot participate at
all; `MemoryStore` is fundamentally untranslatable (process-local Python
dict vs Rust address space) and needs a bespoke error message; and the
sync/async translatable sets being disjoint caused the lazy-resolution
contortions of §1.

**The change.** A capability protocol on `Store` — e.g.
`def native_handle(self) -> NativeStoreDescriptor | None`, where the
descriptor is a small tagged union (`kind: "filesystem" | "obstore" |
"icechunk" | ...`, plus the handle). Stores opt in; engines match on the
descriptor instead of on zarr-python class identity; third-party stores
become translatable without `zarr.zarrista` knowing their types. A stronger
version: standardize object storage on obstore internally so "the handle"
exists for every remote store by construction.

## 4. Runtime config has no single home

**The pattern.** `ArrayConfig` (order, `read_missing_chunks`, ...) is parsed
per-array but threaded ad hoc; engines are bound to `(store, path,
metadata)` and got `config` bolted onto `HierarchyEngine.array_engine`
mid-project when the default engine turned out to need it (it owns
`read_missing_chunks` enforcement).

**The friction.** The protocol churned after Kyle's review (the spec's
hierarchy snippet had to be amended); zarrista engines must now *reject*
config they cannot honor (`read_missing_chunks=False`) rather than share an
enforcement point; `with_config` silently dropped the engine until the
final review caught it, because config, metadata, and engine rebind through
three different mechanisms.

**The change.** Bind arrays and engines to one immutable context object:
`ArrayContext(store, path, metadata, config)`. Engine construction takes
the context; every metadata/config change produces a successor context and
rebinding becomes one operation (`engine.with_context(ctx)`) instead of
`with_metadata` + config threading + per-site `_rebind_engine()` calls.

## 5. Metadata mutation has three inconsistent successor patterns

**The pattern.** `resize` mutates `metadata` in place on the same
`AsyncArray`; `with_config` constructs a fresh wrapper; `update_attributes`
mutates a shared dict *and* returns a new `Array`.

**The friction.** Engines cache per-array state keyed on metadata, so every
mutation site must remember to rebind. We fixed three separate bugs of this
class on the branch (stale engine after `with_config`, after
`update_attributes`, and the `_rebind_engine` sprinkling after
`resize`/`append`). Any future mutation site will silently reintroduce the
bug — the pattern invites it.

**The change.** One canonical rule: arrays are immutable; every metadata or
config change flows through a single `_with_context` chokepoint that
returns a successor array with a rebound engine. `resize` and friends
become thin wrappers over it.

## 6. Indexing is fused to the chunk/codec machinery

**The pattern.** `Indexer` subclasses emit per-chunk projections
(`chunk_coords, chunk_selection, out_selection`) consumed directly by
`CodecPipeline.read/write`. There is no neutral, backend-independent
representation of "what the user selected."

**The friction.** The engine boundary needed one, so we invented `Region` +
the normalize/post-index sandwich. That works, but the contiguous-box
interchange forces the accepted bounding-box amplification for sparse fancy
selections *on every engine including the default*, and hypothesis promptly
found two facade regressions (integer-axis widening, empty block slices) in
code that re-derives what the indexers already knew. A richer interchange
(point lists, per-chunk plans) has nowhere natural to live because the
indexers speak only "chunk projections."

**The change.** Promote a selection IR as the internal currency: indexers
*produce* it (Region today; later a small union — box, strided box, point
set), and both the codec pipeline and external engines *consume* it. The
facade then translates user selections once, in one direction, instead of
zarr-python decomposing to chunk projections while the engine layer
re-normalizes to boxes.

## 7. The data path is numpy-centric despite the buffer abstraction

**The pattern.** `NDBuffer`/`BufferPrototype` exist precisely to abstract
device buffers, but the surrounding code freely calls `np.asarray`,
`np.broadcast_to`, `np.ascontiguousarray` — and the engine spec itself
settled on "results implement `__array__`".

**The friction.** The single Critical finding of the final review: the new
facade broke the GPU data path in six places, then twice more in residual
sweeps (`np.generic` scalars, `ascontiguousarray`'s C-level coercion). Each
fix was "keep the operation in the result's own array namespace" — done by
hand, site by site, guarded by a stand-in test because CI has no GPU.

**The change.** Make the buffer story the lingua franca of the data path:
either extend `BufferPrototype`/`NDBuffer` with the operations the facade
needs (broadcast, patch-assign, astype, scalar extraction) so raw `np.*`
never appears there, or adopt the array-API namespace
(`array_api_compat`) as the internal convention. Plus: keep the
raising-`__array__` stand-in test as a permanent CPU-run guard (done on
this branch — worth upstreaming as a pattern).

## 8. Two competing accelerator hooks

**The pattern.** The historical acceleration point is codec-pipeline
injection (`config codec_pipeline.path`, the zarrs-python approach). The
engine layer now coexists with it at a different altitude.

**The friction.** Conceptual, not mechanical, but real: the codec-pipeline
hook cannot express what engines exist for (native store I/O, sharding-aware
subset reads, sync-native paths), while the engine hook subsumes the
pipeline hook's use case. Documentation and user mental models now carry
both.

**The change.** Declare engines the public accelerator boundary. Reframe
codec-pipeline injection as an implementation detail of the *default
engine* (or deprecate it once zarrs-python ships an engine), so there is
one story: "an engine owns an array's data path."

## 9. Pervasive v2/v3 branching

**The pattern.** `zarr_format == 2` branches at 19 sites in `array.py`
alone (dtype vs `data_type`, order-from-metadata vs order-from-config,
key encoding).

**The friction.** Every one of those branches had to be faithfully cloned
into the default engine (a review caught the `read_missing_chunks` block
that wasn't), and every future engine author confronts them again. The
zarrista engine adds its own "v3 only" gate on top.

**The change.** Normalize at the metadata boundary: uniform accessors on
`ArrayMetadata` (`.native_dtype`, `.effective_order(config)`, ...) so the
data path — and engine implementers — never branch on `zarr_format`.

## 10. Dataclass ergonomics for stateful core classes

Smaller, but it taxed every task that touched `array.py`: frozen-style
dataclasses with 10 `object.__setattr__` calls, `field(init=False)` slots,
and the `engine`/`engine_spec` name collision. These classes have
constructor logic, lazy state, and invariants — a plain class with an
explicit `__init__` (or attrs) would say what it means.

## Priority if zarr-python adopts these

1. §1 + §4 together (single class parameterized by engine, bound to one
   context object) — they eliminate the two largest workaround clusters on
   the branch (lazy nullable engines, rebinding bugs).
2. §3 (store native-handle protocol) — unlocks third-party stores for
   native engines and deletes the isinstance translation layer.
3. §2 (creation-path consolidation) — makes every future cross-cutting
   parameter a one-file change.
4. §6 + §7 — quality-of-implementation: kill the double normalization and
   the hand-audited GPU path.
5. §8, §9, §10 — cleanups that reduce ongoing tax.

## Companion asks on the Zarrista side (tracked elsewhere)

For completeness, the mirror list lives in the design spec: multi-chunk
`store_array_subset` (confirmed easy), a Python store bridge (would relax
§3 from "required" to "optimization"), step>1 selections, v2 reads, and
config hooks so a zarrista engine could honor `read_missing_chunks` instead
of rejecting it.
