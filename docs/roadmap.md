# Roadmap

This page describes where Zarr-Python is headed: the goals for the next major
cycle of work, the changes we intend to make, and how those changes will be
released. It is a living document; discussion and counter-proposals are welcome
on the
[zarr-python issue tracker](https://github.com/zarr-developers/zarr-python/issues).

*The history of this roadmap, including the detailed technical proposals it
was distilled from, can be traced in the
[zarr-python-planning](https://github.com/d-v-b/zarr-python-planning)
repository.*

!!! note

    This roadmap reflects the current thinking of the core developers. It is a
    statement of direction, not a schedule: the work ships when it is ready,
    and individual items may change shape as the proposals are discussed and
    refined.

## Where we are

The [3.0 release](https://github.com/zarr-developers/zarr-python/releases/tag/v3.0.0)
was a total redesign of the library's internals, with three goals: full support
for the Zarr V2 and V3 storage formats, storage APIs ergonomic for high-latency
(cloud) storage, and backwards compatibility with Zarr-Python 2.x where
possible. Those goals were largely achieved, and more than a year on, the
2.x → 3.x transition is effectively resolved.

The 3.x redesign was carried out under hard backwards-compatibility
constraints, and it inherited many structural patterns from the 2.x
implementation it replaced. The library has never had a release cycle whose
primary goal was the *shape* of the internals. The next body of work — which we
call **"v4"** — is that overdue investment.

## Goals

If the 3.0 goals could be sloganized as "migrate to Zarr V3, and improve cloud
storage support", the slogan for the v4 goals is:
**"a frictionless Zarr-based Python ecosystem for chunked arrays"**. Zarr-Python
should be *foundational* for the growing number of Python packages that work
with data in the Zarr format. Concretely, that means pushing in these
directions:

- Give Zarr-Python users excellent performance, out of the box.
- Make Zarr-Python APIs ergonomic and useful for developers.
- Expand our scope to cover vital quality-of-life routines like data copying,
  rechunking, and the like.
- Ease the growth of Python tools across all levels of the Zarr stack.
- Accelerate the implementation of new codecs, chunk grids, chunk key
  encodings, etc.

An important design input: [zarrs](https://github.com/zarrs/zarrs) (Rust) and
[TensorStore](https://github.com/google/tensorstore) (C++) are two independent
Zarr implementations that have converged on the same architectural patterns —
sync-first codec APIs, per-codec concurrency budgets, adaptive sharded-read
strategies, request deduplication, conditional reads. We treat them as
complementary rather than competitive: Zarr-Python aims to be the best
pure-Python Zarr implementation *and* the best wrapper around the
compiled-language implementations, so that users who need native throughput can
get it without leaving the Zarr-Python API surface.

## The Zarr stack

Different applications need different levels of Zarr support: a convention
validator only needs to read metadata documents; a visualization tool may only
need read-only array access; other tools need everything. We think of this as a
"Zarr stack", from most abstract to most concrete:

1. **Conventions** — application and/or domain-specific schemas built on top of Zarr (OME-NGFF,
   GeoZarr, anndata-zarr, multiscales).
2. **Groups** — Zarr hierarchies, traversal, group-level attributes.
3. **Arrays** — the user-facing array object, plus indexing and slicing.
4. **Chunk decoding** — the codec pipeline.
5. **Chunk addressing** — chunk grids and key encodings that map array
   coordinates to store keys.
6. **Stores** — the key-value layer.
7. **Metadata** — pure data documents describing arrays and groups.

Today, Zarr-Python is a monolith that serves every level: a consumer who only
needs metadata handling has to install the full dependency footprint of the
whole library, and a faster chunk-decoding implementation cannot plug in
without re-implementing the layers above it. The v4 direction is to re-shape
Zarr-Python around the stack, so that each level is something you can depend
on, conform to, or replace, without buying every other level:

- **A focused package per level** — `zarr-metadata`, `zarr-store`,
  `zarr-codec`, `zarr-dtype`, with `zarr` as the facade that composes them.
  The first of these,
  [`zarr-metadata`](https://pypi.org/project/zarr-metadata/), is already
  published.
- **A documented interface per level** — capability protocols for stores, a
  small stateless codec API, pure-data dtypes.
- **A conformance suite per level** — so that alternative implementations of a
  level can verify they behave correctly.
- **Engine pluggability at the chunk-decoding level** — alternative engines
  (zarrs, TensorStore) can take over IO without re-implementing hierarchy
  traversal, indexing, or metadata handling.

## What we intend to change

Each theme below is backed by a detailed technical proposal; the summaries
here describe the intended end state.

### Foundation: a functional core

Refactor the internals around a *functional core* — pure data structures and
pure functions for the algebra of Zarr (metadata, chunk layouts, slice
planning, codec walking) — with the side-effecting protocols (stores, codecs)
at the edges. This is an internal change that makes the per-level package split
implementable and provides a clean substrate for engine pluggability.

### Foundation: a formal hierarchy layer

Name and specify the layer that sits between the store API (key-agnostic
bytes) and the user-facing `Array` / `Group` facade, as a small set of typed
verbs (`read_array_metadata`, `write_chunk`, `list_children`,
`read_selection`, ...). Alternative engines implement the verbs end-to-end;
hierarchy-aware caching wraps them; chunk-introspection APIs expose them.

### Codecs

The current codec API wraps every codec in an unnecessary async layer (a
profiling hotspot), bakes batching into every signature, and forces output
allocation even when the caller has a buffer ready. Rewrite the codec API as a
small, stateless capability bundle — sync-first encode/decode, single-element
signatures, optional `decode_into`, capability flags — decoupled from the rest
of the library, with a compatibility shim for existing codecs and clear paths
for migrating Zarr V2 codecs that still have no V3 equivalent.

### Stores

The store abstraction conflates lifecycle, path handling, sync/async,
capability advertisement, and read-only semantics into one inheritance
hierarchy, and the resulting friction has produced a recurring stream of
regressions. Redesign stores as composable capability protocols (`Get`, `Put`,
`List`, ...) with composable wrappers (caching, range coalescing, retries),
transactional semantics, and a shared conformance suite that backends and
wrappers parameterize.

### Performance

A cross-cutting theme that ties the codec, store, and functional-core work
into one performance story: typed, library-owned concurrency resources with
dask-safe defaults; synchronous codec encode/decode on the default read path;
range coalescing; pre-allocated decode buffers; in-flight request
deduplication; ETag-style conditional reads; a unified caching substrate with
sensible defaults; an adaptive whole-shard-vs-coalesced read strategy; and
pluggable high-performance backends (zarrs, TensorStore) selectable with a
keyword argument, so the same `Array` and `Group` — and the same Xarray, Dask,
and napari integrations — work at native throughput. A benchmark suite for the
target access patterns lands first, so every performance lever ships with
before/after numbers.

### Lazy indexing

`Array.__getitem__` performs IO eagerly and returns NumPy arrays, which makes Zarr
arrays the odd one out among modern array libraries and blocks compliance with
the [Python Array API](https://data-apis.org/array-api/) standard. Add an
opt-in `array.lazy[...]` accessor backed by a stable coordinate-mapping algebra
(the `IndexTransform` work in
[#3906](https://github.com/zarr-developers/zarr-python/pull/3906)), plus a
small query planner that turns chained selections into a single IO plan before
any chunks are fetched. No new array type is introduced. Whether the *default*
of bare `array[...]` ever flips from eager to lazy is an explicit, separate
decision — see [decision points](#decision-points) below.

### Data types

First-class support for ML-specific dtypes — `bfloat16`, the `float8`
variants, packed `int4`/`uint4` — via
[`ml_dtypes`](https://github.com/jax-ml/ml_dtypes), using the exact identifiers
registered in `zarr-extensions` so the data stays readable by other
implementations. Ragged arrays, variable-length strings, and an investigation
of Apache Arrow as a substrate for the dtypes the Array API cannot express are
follow-on work on the same substrate.

### Device-agnostic IO

Make Zarr-Python's IO surfaces device-agnostic rather than adding GPU support
as a bolted-on feature: stores and codecs grow APIs for writing into a
caller-provided buffer (`read_into`, `decode_into`), and the `Array` facade
returns array-like objects in the user's chosen Array API namespace. GPU
support falls out once the assumption of CPU destinations is removed, and CPU
paths get faster too, because pre-allocated output buffers eliminate per-chunk
allocation.

### Observability

Two pillars: **performance metrics and tracing** (a small library-owned
`Metrics` object plus OpenTelemetry auto-instrumentation across stores, codecs,
caches, and the engine boundary) and **stored-state introspection** (public
APIs for asking about chunk-level structure, materialization, byte ranges, and
storage footprint without reading the chunks — the surface projects like
VirtualiZarr and Kerchunk have been asking for).

### Configuration, registries, and plugins

Move configuration from "global mutable state read implicitly" to "typed data
passed explicitly": a typed config object replacing the untyped global `donfig`
dict, array-scoped runtime config passed at open time, a registry redesign that
addresses implementations by stable identity and resolves plugin name-conflicts
deliberately, and named profiles replacing global mutators. This substrate is
where the performance-lever defaults (concurrency, caching, engine selection)
will live, so it lands early.


### Coordinated and distributed writes

Give the two patterns that actually produce large Zarr archives — parallel
disjoint-region writes and append-along-axis growth — a design home: disjoint
chunk-aligned region writes with alignment *checked* rather than assumed, a
create-then-hand-out-regions primitive, and single-writer resize/append, all on
plain Zarr V3. Stronger guarantees (atomicity, reader isolation, concurrent
appenders) are enabled through the seam a transactional engine such as
[Icechunk](https://icechunk.io/) builds on, rather than implemented in
Zarr-Python itself.

### Missing APIs

User-facing conveniences that users have been asking for, in some cases for
years: hierarchy navigation helpers, chunk introspection, explicit constructors
replacing `mode=`, a typed exception hierarchy, rich reprs, context-manager
support, data copying, and an in-library rechunking primitive.

## How the work will be released

**"v4" names this whole body of work, delivered across many releases — it is
not a single "4.0" feature release.** The work is organized into three streams
that run in parallel:

| Stream | Release vehicle | Scope |
|---|---|---|
| **Additive value** | 3.x minor releases, shipping continuously | The overwhelming majority of the plan, including the entire foundation. No migration required. |
| **Deprecation accumulation** | Warnings across the 3.x line | Each surface is deprecated only *after* its additive replacement has shipped, so users always have a migration target before they see a warning. |
| **Breaking removals** | One minimal, late major release (4.0.0) | Removal of the deprecated surfaces, and *only* those, after deprecation windows have elapsed and downstream libraries have had release windows to adapt. |

The additive stream is itself roughly ordered:

1. **Ship-now wins** — dependency-free improvements that land first: the
   benchmark suite, store-layer range coalescing, in-flight request
   deduplication, the sync codec path on default reads, ML dtype support,
   constructor and display UX.
2. **Foundation** — the functional-core refactor, the per-level package split,
   the new stores API, the hierarchy verbs, the typed configuration substrate,
   the full concurrency and caching rework, and the codec API rewrite. Mostly
   invisible to users, all additive.
3. **User-facing surface** — opt-in lazy indexing and the query planner,
   device-agnostic IO, observability, chunk introspection, and the zarrs and
   TensorStore engine wrappers, built on the foundation.

The eventual 4.0.0 release contains only removals whose replacements shipped
earlier: the legacy `Store` ABC and the `Buffer`/`prototype` read contract, the
`mode=` constructors, the internal `sync()` bridge, and — conditionally — the
eager `array[...]` path. Nothing new is delivered there; it is the only release
downstream maintainers must treat as breaking, and it arrives after the value
has already been delivered additively.

### Backwards-compatibility commitments

The v4 work changes the public API: methods will be renamed, signatures will
change, deprecated patterns will be removed, and the codec and store APIs will
be rewritten. We believe the changes are worth the cost, and we commit to the
following:

- **Conformance with community standards.** Where a relevant cross-language
  standard exists, we conform to it: the Python Array API at the array surface,
  the Zarr V3 spec and its extensions at the storage layer, OpenTelemetry for
  tracing, and standard buffer-protocol and device-interop conventions for
  device-agnostic IO.
- **Functional coverage.** Anything you can do in Zarr-Python 3.x you will
  still be able to do once the v4 work has landed — sometimes through a renamed
  API, but the capability is preserved. We will not remove the ability to read
  or write any Zarr-format data that 3.x supports.
- **A deprecation window for every change.** Renames and removals land through
  deprecation cycles, and downstream libraries (Xarray, Dask, napari) get
  release windows to absorb each change before the next one lands.
- **Generous legacy support** If necessary, we can keep old code around in a `legacy` module. Pydantic used a similar strategy to manage their 2.0 release: see https://pydantic.dev/docs/validation/dev/get-started/migration/#using-pydantic-v1-features-in-a-v1v2-environment.

### Decision points

Flipping the default of bare `array[...]` from eager to lazy is the single
highest-migration-cost item in the plan, so it is handled as an explicit
decision, not bundled into the additive work. The opt-in `array.lazy[...]`
accessor ships first, with no default change. Whether the default ever flips
hinges on whether Array API conformance at the bare-`__getitem__` surface turns
out to be a hard requirement; if it does, the flip happens as a long-window
deprecation with an explicit eager escape hatch and downstream coordination —
never as a reason to adopt a major version.

## How to get involved

- **Discuss the plans.** Comments and counter-proposals on any of the themes
  above are welcome on the
  [issue tracker](https://github.com/zarr-developers/zarr-python/issues) and in
  the [developer chat](https://ossci.zulipchat.com/).
- **Review in-flight work.** The `IndexTransform` algebra that lazy indexing is
  built on is in review at
  [#3906](https://github.com/zarr-developers/zarr-python/pull/3906).
- **Weigh in as a downstream maintainer.** If your project's use of
  Zarr-Python would be affected by the codec API rewrite, the stores rewrite,
  or the lazy-indexing work, the planning phase is the time to surface
  workloads or patterns that don't fit.