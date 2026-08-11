# Roadmap

This page describes where Zarr-Python is headed: the goals for the next major
cycle of work, the changes we intend to make, and how those changes will be
released. It is a living document; discussion and counter-proposals are welcome
on the
[Zarr-Python issue tracker](https://github.com/zarr-developers/zarr-python/issues).

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
possible. Those goals were largely achieved! Going by the content of issues and pull requests
submitted to the library, few users are grappling with 2.x → 3.x migration issues. Instead, we see
users asking for things like better APIs, where "better" usually means faster.

The 3.x redesign was carried out under hard backwards-compatibility
constraints, and it inherited many structural patterns from the 2.x
implementation it replaced. The library has never had a release cycle whose
primary goal was the *shape* of the internals. The next body of work — which we
call **"v4"** — is that overdue investment. We think iterating on the internals of
the library will make it *much* easier to bring faster, more expressive APIs to Zarr-Python
users.

## Goals

If the 3.0 goals could be sloganized as "migrate to Zarr V3, and improve cloud
storage support", the slogan for the v4 goals is:
**"a frictionless Zarr-based Python ecosystem for chunked arrays"**. Zarr-Python
should be *foundational* for the growing number of Python packages that work
with data in the Zarr format. Concretely, that means pushing in these
directions:

- Deliver excellent performance, out of the box, by whatever means necessary (e.g., Rust bindings).
- Make Zarr-Python APIs ergonomic and useful for developers.
- Expand our scope to cover vital quality-of-life routines like data copying,
  rechunking, and the like.
- Ease the growth of Python tools across all levels of the Zarr stack.
- Accelerate the implementation of new codecs, chunk grids, chunk key
  encodings, etc.

An important design input: [`zarrs`](https://github.com/zarrs/zarrs) (Rust) and
[TensorStore](https://github.com/google/tensorstore) (C++) are two independent
Zarr implementations that use architectural patterns we want to learn from.
We see them as complementary rather than competitive.

!!! note

    Many of the features in this roadmap will not require breaking public 3.x APIs. We can and will
    ship those features in 3.x releases; at the same time, we consider it clarifying to frame the
    coherent development direction as vectored at a 4.0 milestone.

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
on, conform to, or replace, without buying every other level.

We plan to "stackify" Zarr-Python by spinning core functionality out into separate Python packages, e.g. `zarr-metadata`, `zarr-indexing`, `zarr-storage`,
`zarr-codec`, `zarr-dtype`, each with narrow scope, all composed in the `zarr` package. The Rust `zarrs` library
successfully uses a structure like this, and we are keen to share the benefits of a more modular, maintainable codebase. Two of these subpackages,
[`zarr-metadata`](https://zarr.readthedocs.io/projects/zarr-metadata/en/latest/) and [`zarr-indexing`](https://zarr.readthedocs.io/projects/zarr-indexing/en/latest/), are already
published.

## What we intend to change

The following section details how we want to evolve the internal logic that drives Zarr-Python.

### Foundation: swappable backends

We propose to refactor Zarr-Python internals around a *swappable engine* — a protocol, or protocols,
that define the core routines a Zarr implementation must support. Zarr-Python becomes one user-facing
API that can be driven by multiple backends, including externally defined backends. We think this will allow users on many different platforms to get the best performance for their particular environment while retaining a familiar API.

#### Rust bindings

We want a Python backend (i.e., the status quo), but also a Rust-based backend, via bindings to the
[`zarrs`](https://docs.rs/zarrs/latest/zarrs/) crate. The [zarrs-python](https://zarrs-python.readthedocs.io/en/latest/) project demonstrates that
bridging `zarrs` and Zarr-Python buys a *lot* of performance in the specific case of chunk encoding. But zarrs-python is constrained today by limited
modularity in Zarr-Python internals. Refactoring our internals around swappable backends should address this limitation.

Any Python package that interfaces with `zarrs` will need Pythonic bindings to the Rust library. So we are *very* excited about the [zarrista](https://developmentseed.org/zarrista/latest/) package, which aims to provide complete Python bindings for `zarrs`.

#### Sync / Async partitioning

Internally we will branch over two kinds of backends: synchronous and asynchronous. The synchronous backend is suitable for arrays and groups persisted to low-latency storage like in-memory stores or local file systems, where async scheduling is pure friction. The asynchronous backend will use Python's `async` support and will provide concurrent APIs where it helps: for arrays and groups persisted to high-latency storage.

### Lazy indexing

The Zarr-Python Array API was initially designed to mirror NumPy, with eager
array indexing syntax. `Array.__getitem__` performs IO eagerly and returns a NumPy array.
That was helpful to the dominant use-case at the time of its creation, but it
means deferred IO and computation currently require an external library
such as Dask. It means there is no built-in support for representing multi-step
reads as a single deferred plan. Further, it means that every chained
selection round-trips to storage independently.

We can fix this by introducing an API for lazy indexing. Under this model, an array indexing operation
like `array[::2]` desugars to a declarative state like `(array, selection)`. Chained selections like
`array[10:100][::2]` are fused immediately, and we defer actual IO for the time when the result of
indexing is needed. [TensorStore](https://google.github.io/tensorstore/) is an excellent role model
for Zarr-Python here, and we can deliver this functionality without breaking ordinary indexing behavior.
See this [discussion](https://github.com/zarr-developers/zarr-python/discussions/1603) for more
background.

### Data types

First-class support for ML-specific dtypes — `bfloat16`, the `float8`
variants, packed `int4`/`uint4` — via
[`ml_dtypes`](https://github.com/jax-ml/ml_dtypes). These data types have specifications written up in `zarr-extensions`, but there's no simple to get them integrated in Zarr-Python today.

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

This area is actually an unfinished aspect of the 2.x → 3.0 migration: Zarr-Python 2.x supported
synchronization logic via file-based locks, and we have not implemented equivalent functionality in
3.x. We don't have *concrete* plans for closing this gap. Re-implementing simple object-based locking, for
backends that support it, is a direct solution we should consider. But a transactional storage model,
where a sequence of basic storage operations like reading and writing could be submitted in a batch and
executed serially, with rollbacks under failure, is also quite appealing.
As with array indexing, TensorStore is the trailblazer here, and we can learn from its example.

We can also avoid the need for synchronization mechanisms entirely with better planning.
Many users of the 2.x synchronization tooling needed to simply write values from one chunked source
to another, without worrying about chunk alignment. This can be addressed e.g. by creating a write
plan that partitions the input chunks into batches within which writes cannot race.

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
