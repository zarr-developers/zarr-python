# System-Memory Chunk Cache Documentation Design

**Date:** 2026-08-02

**Status:** Approved for implementation planning

**Package:** `packages/zarr-indexing`

## Purpose

The visual indexing guide currently uses half-open interval notation before it
defines that notation, and its napari-like integration stops at the point where
a real viewport consumer needs a chunk lifecycle. The next documentation change
will close both gaps:

1. teach half-open intervals before arbitrary coordinate origins; and
2. provide an executable, deterministic reference implementation of a
   system-memory chunk cache over a chunked source.

The cache example demonstrates how a consumer can combine `LazyArray` chunk
projections with loading, reuse, eviction, failure, and retry. It is reference
code in the documentation, not a package export or a promise of napari support.

## Goals

- Define `[0, 1)` as “start at 0, inclusive; stop at 1, exclusive” before the
  guide first uses interval notation.
- Show that adjacent half-open intervals concatenate without a gap or overlap:
  `[0, 1) ∪ [1, 3) = [0, 3)`.
- Connect literal integer coordinates to chunk layouts, including prepending
  storage into negative coordinates.
- Give napari-oriented consumers a small, executable model for keeping decoded
  chunks in system memory.
- Use public chunk projections to read chunk-local coordinates and place values
  at request coordinates.
- Make cache hits, de-duplication, least-recently-used eviction, failure, and
  explicit retry observable and testable.
- Keep all scheduling deterministic so the example is stable in documentation
  builds and straightforward to understand.

## Non-goals

- No public API will be added to `zarr_indexing`.
- The example will not import napari or claim to be a napari integration.
- It will not implement background threads, async execution, cancellation,
  priorities across viewports, codecs, storage keys, prefetching, or GPU memory.
- It will not be a production cache with locking, byte-based accounting, or a
  configurable replacement policy.
- It will not copy TensorStore or Neuroglancer source, prose, or artwork. Their
  systems are prior art; the example and diagrams will be independently authored.

## Teach interval notation first

Chapter 2, “Coordinates are addresses,” will begin with a short section titled
“How to read a half-open interval.” It will establish all four pieces of the
notation explicitly:

- `[` means the lower bound is included;
- `0` is the first coordinate in `[0, 1)`;
- `1` is the stopping boundary and is excluded; and
- `)` marks that exclusive upper bound.

The text will state that `[0, 1)` contains the single integer coordinate `0`.
A compact generated figure will then place `[0, 1)` next to `[1, 3)` and combine
them into `[0, 3)`. Coordinate `1` belongs only to the second interval, so the
pieces meet exactly once. This makes the practical benefit concrete: adjacent
slices and chunks can be concatenated by matching one interval's exclusive
upper bound to the next interval's inclusive lower bound.

Only after that explanation will the guide introduce `[-2, 3)`, negative
coordinates, and prepending. The existing prepend example will link the ideas:
chunks `[-3, 0)`, `[0, 3)`, and `[3, 6)` are ordinary adjacent half-open
intervals. Because coordinates are arbitrary integers, prepending a chunk grows
the array toward negative coordinates without renumbering existing cells.

## Executable reference architecture

The new example will live in the standalone documentation file
`docs/examples/napari_chunk_cache.py`. Keeping it separate from the short
integration example makes its teaching boundary clear and lets the complete
file execute as a documentation contract. It will not be imported or re-exported
by the library.

The example has four small units:

1. **Recording chunk source** — owns a NumPy array, shape, dtype, chunk shape,
   a `read_chunk(chunk_coords)` method, an exact read log, and a controlled way
   to fail a chosen chunk.
2. **Chunk record** — stores one chunk coordinate's state, decoded system-memory
   buffer or error, and a monotonic last-access generation.
3. **Stateful chunked-array wrapper** — exposes NumPy-like `shape`, `dtype`, and
   synchronous `__getitem__`; derives projections from `LazyArray`, queues each
   missing chunk once, loads it, assembles the requested result, and enforces a
   small LRU capacity.
4. **Event record** — captures the chunk coordinate, previous state, next state,
   and reason for every lifecycle transition. The worked example renders this
   history as a compact table and tests it exactly.

The wrapper is synchronous by design. “Queued” and “loading” remain separate
states because they explain the integration seam where a real viewport
scheduler could insert priorities, workers, or cancellation. A deterministic
`drain()` step performs the work immediately in projection order, avoiding
threads and timing-sensitive assertions.

## System-memory state machine

Each chunk record is in exactly one of these states:

| State | Meaning |
| --- | --- |
| `NEW` | The chunk has never been requested. |
| `QUEUED` | One or more selections need the chunk; it is scheduled once. |
| `LOADING` | The source read is in progress. |
| `READY` | The decoded chunk buffer is resident in system memory. |
| `FAILED` | The last source read failed and its error is retained. |
| `EVICTED` | A former `READY` buffer was dropped to respect capacity. |

The only legal transitions are:

```text
NEW     -> QUEUED -> LOADING -> READY
                         \---> FAILED -> QUEUED
READY   -> EVICTED -> QUEUED
```

All transitions pass through one validation-and-recording helper. Re-requesting
a `QUEUED`, `LOADING`, or `READY` chunk does not enqueue or read it again.
Re-requesting a `FAILED` chunk raises its retained failure without another
source read. `retry(chunk_coords)` is the explicit `FAILED -> QUEUED`
transition; the next drain attempts the read again.

The state diagram will use the guide's generated, theme-aware inline SVG system.
State names, arrow labels, line styles, a `<title>`, and a `<desc>` will carry
meaning independently of color in both light and dark themes.

## Selection and assembly data flow

For `wrapper[key]`, the example will:

1. build the lazy selection and obtain its public chunk projections without
   reading source data;
2. collect projection chunk coordinates in deterministic order;
3. mark already-`READY` required chunks as recently used and queue every
   `NEW` or `EVICTED` required chunk exactly once;
4. drain required queued chunks through `QUEUED -> LOADING -> READY` or
   `QUEUED -> LOADING -> FAILED`;
5. enumerate each projection's shared synthetic cell domain;
6. evaluate `chunk_transform` to read zero-origin chunk-local coordinates and
   `cell_transform` to place values at request coordinates;
7. evict least-recently-used `READY` chunks until the configured chunk-count
   capacity is restored; and
8. return the assembled NumPy result.

Chunks needed by the active request remain pinned through assembly. This lets a
request span more chunks than the steady-state capacity: eviction happens only
after all values have been placed. Cache capacity counts decoded `READY`
buffers, not queued records or bytes. Those simplifications will be stated
beside the code.

If loading any required chunk fails, the wrapper records `FAILED`, retains the
exception, and raises a small example-local `ChunkLoadError` with the source
exception as its cause. Already loaded chunks remain reusable. No partially
assembled result is returned.

## Worked viewport sequence

The prose will frame the wrapper as the system-memory portion of a chunked
rendering pipeline. It will use the guide's existing 6-by-8 image with 3-by-4
chunks and a two-chunk cache:

1. A viewport selecting `image[1:5, 2]` loads chunks `(0, 0)` and `(1, 0)` and
   reconstructs `[10, 18, 26, 34]` through the paired transforms.
2. An overlapping viewport that needs `(1, 0)` reuses its `READY` buffer and
   performs no source read, making that chunk most recently used.
3. A viewport in chunk `(0, 1)` loads it and evicts least-recently-used `(0, 0)`.
4. Repeating the first viewport reloads evicted `(0, 0)` while retaining the
   more recently used required chunk.
5. A request for chunk `(1, 1)`, which the recording source is configured to
   fail, records `FAILED`. Repeating it does not hammer the source. After the
   source is repaired and `retry((1, 1))` is called, the next request reaches
   `READY` and returns the expected values.

The page will show exact source reads and the transition log next to the state
diagram. It will explain that a real napari adapter could run queue draining in
workers and invalidate a canvas when chunks become ready, while the selection
and projection semantics would remain unchanged.

## Documentation placement

- Add the half-open-interval explanation at the very start of
  `docs/guide/02-transforms.md`.
- Expand the napari-like section of `docs/guide/integrations.md` with the
  system-memory lifecycle, state diagram, worked sequence, and executable
  snippet. Preserve the existing statement that this is not a napari
  integration.
- Put the complete cache implementation in its own executable example file.
- Add the state-machine SVG to the existing declarative diagram renderer and
  checked-in generated assets.
- Link to napari's array-like image input documentation and Neuroglancer's
  chunk-state source as context, while labeling Neuroglancer as conceptual
  prior art rather than an API being reproduced.

## Testing strategy

Implementation will begin with failing documentation tests.

One parameterized happy-path test will cover reasonable request sequences and
their expected values, source-read deltas, resident chunk coordinates, and
state histories. Focused tests will separately cover each error contract:

- a source failure becomes `FAILED` and preserves the cause;
- repeated access to `FAILED` does not perform an implicit retry;
- retry is rejected for a chunk not in `FAILED`; and
- an invalid state transition is rejected by the transition helper.

Additional assertions will prove that:

- constructing a selection performs no source reads;
- overlapping selections reuse `READY` buffers;
- duplicate demand queues and reads one chunk only once;
- assembly uses both public projection transforms and matches NumPy exactly;
- LRU eviction is deterministic and a later request reloads an evicted chunk;
- explicit retry performs exactly one new read and can reach `READY`;
- the example file executes from beginning to end and every displayed snippet
  region is balanced;
- the cache wrapper is absent from the package's public exports;
- the interval definition and concatenation example occur before the guide's
  first negative-origin interval; and
- the generated diagram is current, accessible, responsive, and legible under
  both Material light and dark color schemes.

The package's focused tests, diagram freshness check, strict documentation
build, lint, type checks, and full repository check remain the completion gate.

## Prior-art and licensing boundary

The example adopts only the general idea that chunks pass through explicit
lifecycle states before and after residency in system memory. State machines,
queues, caches, and LRU eviction are general software concepts. The code,
terminology, exposition, state diagram, and test scenarios will be independently
authored for this package. No TensorStore or Neuroglancer implementation will be
copied or translated, so the documentation does not incorporate their licensed
source. Attribution links are still valuable because they identify the systems
that motivated the integration model.
