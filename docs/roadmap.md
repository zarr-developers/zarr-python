# Roadmap

This page describes where Zarr-Python is headed: the goals for the next major
cycle of work, the changes we intend to make, and how those changes will be
released. It is a living document; discussion and counter-proposals are welcome
on the
[zarr-python issue tracker](https://github.com/zarr-developers/zarr-python/issues).

*The history of this roadmap, including the detailed technical proposals it
was distilled from, can be traced in the
[zarr-python-planning](https://github.com/zarr-developers/zarr-python-planning)
repository.*

!!! note

    This roadmap reflects the current thinking of the core developers. It is a
    statement of direction, not a schedule. We don't know how long these changes
    will take, only that we are committed to moving the project in the direction outlined
    here.

## Where we are

The [3.0 release](https://github.com/zarr-developers/zarr-python/releases/tag/v3.0.0)
was a total redesign of the library's internals, with three goals: full support
for the Zarr V2 and V3 storage formats, storage APIs that are ergonomic for high-latency
storage (such as cloud storage), and backwards compatibility with Zarr-Python 2.x where
possible. Those goals were largely achieved, and more than a year on, the
2.x → 3.x transition is effectively resolved, in the sense that it the transition
is no longer the prevailing source of issues and pull requests.

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

!!! note
  Many of the features described below will not require breaking public 3.x APIs. We can and will
  ship those features in 3.x releases; at the same time, we consider it clarifying to frame the
  coherent development direction as vectored at a 4.0 milestone.

- Deliver excellent performance, out of the box, by whatever means necessary (e.g., Rust bindings).
- Make Zarr-Python APIs ergonomic and useful for developers.
- Expand our scope to cover vital quality-of-life routines like data copying,
  rechunking, and the like.
- Ease the growth of Python tools across all levels of the Zarr stack.
- Accelerate the implementation of new codecs, chunk grids, chunk key
  encodings, etc.

An important design input: [zarrs](https://github.com/zarrs/zarrs) (Rust) and
[TensorStore](https://github.com/google/tensorstore) (C++) are two independent
Zarr implementations that use architectural patterns we want to learn from.
We see them a complementary rather than competitive.

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

- **A focused package per level** — `zarr-metadata`, `zarr-storage`,
  `zarr-codec`, `zarr-dtype`, all composed in the `zarr` package. The Rust `zarrs` library
  successfully uses a structure like this, and we are keen to share the benefits of a more modular, maintainable codebase. The first two subpackages,
  [`zarr-metadata`](https://zarr.readthedocs.io/projects/zarr-metadata/en/latest/) and `[zarr-indexing`](https://zarr.readthedocs.io/projects/zarr-indexing/en/latest/), are already
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

### Foundation: swappable backends

Refactor the internals around a *swappable engine* — a single protocol that defines the core routines a
Zarr implementation must support. Zarr-Python becomes one user-facing API that can be driven by multiple
backends, including a Python-heavy backend, but also a Rust-based backend, via bindings to the `zarrs` crate.

Internally we will branch over two kinds of backends: synchronous and asynchronous. The synchronous backend is suitable for arrays and groups persisted to low-latency storage like in-memory stores or local file systems, where async scheduling is pure friction. The asynchronous backend will use Python's `async` support and will provide concurrent APIs where it helps: for arrays and groups persisted to high-latency storage.

### Lazy indexing

The Zarr-Python Array API was initially designed to mirror NumPy, with eager
array indexing syntax. `Array.__getitem__` performs IO eagerly and returns a NumPy arrays.
That was helpful to the dominant use-case at the time of its creation, but it
means deferred I/O and computation currently require an external library
such as Dask. It means there is no built-in support for representing multi
step reads as a single deferred plan. Further, it means that every chained
selection round-trips to storage independently.

We can fix this by introducing an API for lazy indexing. Under this model, an indexing operation
like `array[::2]`desugars to a declarative state like `(array, selection)`. Chained selections like
 `array[10:100][::2]` are fused immediately, and we defer actual IO for the time when the result of
 indexing is needed. TensorStore is an excellent role model for Zarr-Python here, and we can deliver
 this functionality without breaking ordinary indexing behavior. See this
 [classic discussion](https://github.com/zarr-developers/zarr-python/discussions/1603) for more
 background.

### Data types

First-class support for ML-specific dtypes — `bfloat16`, the `float8`
variants, packed `int4`/`uint4` — via
[`ml_dtypes`](https://github.com/jax-ml/ml_dtypes), using the exact identifiers
registered in `zarr-extensions` so the data stays readable by other
implementations. 

Ragged arrays, variable-length strings, and an investigation
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

### Configuration, registries, and plugins

Move configuration from "global mutable state read implicitly" to "typed data
passed explicitly": a typed config object replacing the untyped global `donfig`
dict, array-scoped runtime config passed at open time, a registry redesign that
addresses implementations by stable identity and resolves plugin name-conflicts
deliberately, and named profiles replacing global mutators.

### Coordinated and distributed writes

Give the two patterns that actually produce large Zarr archives — parallel
disjoint-region writes and append-along-axis growth — a design home: disjoint
chunk-aligned region writes with alignment *checked* rather than assumed, a
create-then-hand-out-regions primitive, and single-writer resize/append, all on
plain Zarr V3. Stronger guarantees (atomicity, reader isolation, concurrent
appenders) are enabled through the seam a transactional engine such as
[Icechunk](https://icechunk.io/) builds on, rather than implemented in
Zarr-Python itself.

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
