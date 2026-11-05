# Release notes

<!-- towncrier release notes start -->

## 0.3.0 (2026-09-29)

### Features

- Chunk plans now have a factored, columnar form. `ChunkPlan.partition()`
  returns a `GridPartition`: one `StridedSet` or `IndexedSet` table per output
  dimension the transform reads independently, plus `joint_sets`, one `JointSet`
  per connected component of index arrays. Arrays that share request axes are
  sorted into chunks together; independent ones stay in separate tables, so
  `(u, v) -> (a[u], b[u], c[v])` stores 3,000 index values rather than
  3,000,000. A `ChunkProjection` is one row of each table and is derived on
  demand, so planning costs the sum of the touched chunks per axis rather than
  their product, and a vectorized consumer can read the tables (`chunk_coords()`,
  the CSR columns, the memoized read-only `local` coordinates) without
  materializing a projection per chunk.

  The projections a plan yields map the same cells to the same storage as
  before. Correlated projections now keep residual slices symbolic and index
  arrays compact along axes they do not vary over, and `vindex` compilation
  preserves singleton axes for the same reason.

  A hand-built transform in which two output maps read one input axis (a
  diagonal, which no selection produces) is rejected with `ValueError`; the
  whole-transform walk that previously served it produced wrong projections. ([#4310](https://github.com/zarr-developers/zarr-python/pull/4310))
- Make every `LazyArray` read supply a chunk projection, including unpartitioned reads and independently executed partition views. Partition views retain the source grid and full source base shape, so indexing and repartitioning use the same coordinate frame as other views. ([#4349](https://github.com/zarr-developers/zarr-python/pull/4349))
- Make `LazyArray` indexing lazy by default: use `view[...]`, `view.oindex[...]`,
  and `view.vindex[...]` directly, without a `.lazy` accessor. Iteration yields
  lazy views, so arithmetic over iterated elements no longer works on values;
  `result()` and NumPy conversion materialize. Add synchronous `write(values)`
  and assignment through composed views, and an explicit `EagerArrayAdapter` for
  consumers such as Dask that require eager indexing.

  Writes are planned against the source's write grid, discovered from
  `write_chunk_sizes` or `chunks`: an affine selection is one basic assignment,
  and any other selection reads, updates, and rewrites each touched cell once,
  so storage round trips scale with touched chunks rather than selected
  elements. A source with no advertised grid is written one element at a time.
  Writes bypass the reader, so a caching reader is not invalidated.

  Views keep their literal domain instead of re-zeroing after every selection:
  `a[10:20]` has domain `[10, 20)` and `a[10:20][2:5]` has domain `[12, 15)`,
  as in TensorStore, while NumPy keys stay positional. An `IndexDomain` key
  restricts a view to literal coordinates and an `IndexTransform` key composes
  onto it. Box partitions keep the request's coordinates, so a part view's
  domain is a sub-domain of its parent's. ([#4350](https://github.com/zarr-developers/zarr-python/pull/4350))

### Bugfixes

- Direct `IndexTransform.oindex` and `IndexTransform.vindex` selections now reject
  integer array coordinates outside the `np.intp` range before conversion.
  Previously, oversized `uint64` values could wrap to negative coordinates and
  silently select a different location in a domain containing negative coordinates. ([#4333](https://github.com/zarr-developers/zarr-python/pull/4333))
- Reject invalid wire index-array values and unrepresentable normalized bounds/ranks, and raise an explicit error for unsupported intersections sharing an affine and lookup input axis instead of returning incorrect coordinates. Group negative chunk-coordinate tuples without merging distinct chunks. ([#4345](https://github.com/zarr-developers/zarr-python/pull/4345))
- Validate inclusive `index_array_bounds` against all raw index values when loading transforms and output maps from JSON. Accept valid finite and one-sided constraints, and reject out-of-bounds values eagerly before offset, stride, or map simplification. Validated immutable maps need not retain the constraints; message normalization preserves the original bounds. ([#4347](https://github.com/zarr-developers/zarr-python/pull/4347))
- Delegate LazyArray source tokenization to Dask, honoring its registered normalizers, source hooks, and deterministic-token requirements. Remove local content-hashing and UUID fallbacks. Dask remains optional for indexing and reading. ([#4348](https://github.com/zarr-developers/zarr-python/pull/4348))

### Improved Documentation

- Rewrote the guide's prev/next navigation links as markdown so mkdocs
  validates them at build time; the rendered pages are unchanged. ([#4291](https://github.com/zarr-developers/zarr-python/pull/4291))
- The docs now render the release notes as a page instead of linking to
  `CHANGELOG.md` on GitHub, so each docs version shows its own changelog. ([#4311](https://github.com/zarr-developers/zarr-python/pull/4311))
- Correct indexing, reader, serialization, cache, and integration descriptions to match supported behavior; qualify NumPy/TensorStore compatibility and performance claims.

  Describe current contracts in source and test docstrings instead of narrating prior implementations. Clarify that immutable index coordinates do not snapshot source values. ([#4345](https://github.com/zarr-developers/zarr-python/pull/4345))

### Misc

- Expand planner property tests across signed origins and chunk IDs, custom grids, mixed affine and lookup dependencies, duplicate coordinates, and empty domains. Verify exact request coverage and storage mapping with an independent pointwise oracle. ([#4346](https://github.com/zarr-developers/zarr-python/pull/4346))
- Removed every `assert` statement from the package's runtime code and dropped
  the `zarr-indexing` exemption from the repo-wide ruff `S101` rule, so none can
  return. Asserts are stripped under `python -O`; the ones here narrowed types
  or guarded internal invariants rather than validating input, so most were
  restructured away (branching on the map type, carrying the narrowed value in a
  local) and the remainder became explicit `RuntimeError`s for states the
  public API cannot reach. No user-visible behavior changes. ([#4364](https://github.com/zarr-developers/zarr-python/pull/4364))
- The test suite now runs in the package's own environment instead of the zarr-python root environment, so `just test` works from an unpacked sdist, where the suites that need `zarr` skip. Inside the repository, the justfile recipes add the in-repo `zarr`. ([#4406](https://github.com/zarr-developers/zarr-python/pull/4406))


## 0.2.1 (2026-08-12)

### Improved Documentation

- The documentation builds from the package directory, so snippet includes
  resolve against this package's docs instead of zarr-python's; the 0.2.0 docs
  rendered some examples from the wrong files. ([#4252](https://github.com/zarr-developers/zarr-python/pull/4252))


## 0.2.0 (2026-08-12)

### Features

- `LazyArray` has an explicit reader boundary: an `IndexTransform` decides which
  values belong in a result, and a `Reader` decides how one backend obtains them,
  preserving the transform exactly. `LazyArray(source)` is conservative and assumes
  only basic indexing; `LazyArray.from_numpy(array)` selects the optimized NumPy
  reader; `with_reader` selects any other. Readers do not define indexing
  semantics, partitioning, scheduling, or result ownership. Both built-in readers
  lower through NumPy system memory, so a device array needs a custom reader that
  transfers into the supplied output buffer. ([#4222](https://github.com/zarr-developers/zarr-python/pull/4222))
- Added the `zarr_indexing.testing` subpackage, behind a `testing` extra
  (`pip install zarr-indexing[testing]`), carrying the Hypothesis machinery this
  package tests itself with. `ChainedIndexingStateMachine` composes basic,
  orthogonal and vectorized selections onto a `LazyArray` wrapping an array you
  supply, then checks every view's shape, `result()`, and assembled `parts()`
  against NumPy; `zarr_indexing.testing.strategies` exports the selection
  strategies alone, for a project with its own harness. Nothing outside the
  subpackage imports Hypothesis. ([#4222](https://github.com/zarr-developers/zarr-python/pull/4222))
- Added `UnitStepReader` / `unit_step_reader`: a backend adapter for sources
  whose basic indexing accepts only ascending step-1 slices (FFI bindings, HTTP
  range endpoints). Every key it presents is `slice(start, stop, 1)` per axis;
  strides, reversals, and gathers are applied to the in-memory block by the
  residual lowering. The integrations guide documents the companion dense-box
  re-partition idiom — resolving a unit-stride rectangular view as one backend
  slab read while keeping partitioned reads for strided and fancy selections. ([#4222](https://github.com/zarr-developers/zarr-python/pull/4222))
- Negative-step slices are supported, following merged ndsel 1.0-draft.2 and
  TensorStore 0.1.84: `arr[::-1]`, `arr[5:1:-2]`, and reversal composed over an
  already-strided or already-gathered view. One desugaring rule covers both signs —
  omitted bounds resolve on the side the traversal starts and stops, and the origin
  is `trunc(start / step)` — while a reversed interval is an error rather than a
  silently empty selection. A reversing slice normally yields a negative domain
  origin, since the result stays anchored to the source coordinate frame;
  `LazyArray` re-bases every view to origin 0, so its positional dialect is
  unaffected. ([#4222](https://github.com/zarr-developers/zarr-python/pull/4222))
- Fancy selections compose without restriction, on both `LazyArray` and
  `IndexTransform`: a second `oindex`/`vindex`/mask step may land on any axis of
  an already-fancy view, including axes an existing index array merely broadcasts
  along. Array-carrying transforms are chained through `compose`, and resolution
  handles the resulting mixed, correlated and diagonal index-array structures on
  one shared pointwise path, classified by `IndexTransform.index_array_structure`.
  Only hand-built affine diagonals — an index array and a slice map bound to the
  same axis — remain unsupported.

  `__dask_tokenize__` digests a view's canonical transform body rather than
  embedding it, so tokens stay small for large fancy selections. ([#4222](https://github.com/zarr-developers/zarr-python/pull/4222))
- Added `LazyArray`, which grafts the full NumPy indexing dialect onto any source
  exposing `shape`, `dtype`, and basic integer/slice `__getitem__` — a chunked
  store, an FFI binding, an HTTP endpoint. `view[...]`, `view.oindex[...]` and
  `view.vindex[...]` each compose an `IndexTransform` and return a new view
  without reading anything; `result()` and `numpy.asarray(view)` materialize.
  Selections use positional NumPy semantics (negatives wrap, scalars drop their
  axis, coordinate arrays keep order and duplicates), which the new
  `zarr_indexing.boundary` module translates into the algebra's literal
  coordinates. Consumers that need indexing to produce data, such as
  `dask.array.from_array`, wrap a view in `EagerArrayAdapter`.

  A read is divided along a **partitioning** — discovered from the wrapped array,
  or chosen with `with_parts` / `with_parts_per_axis` / `unpartitioned`.
  `parts()` yields one `Partition` per box, pairing a resolvable sub-view with
  where its cells belong in the result; `result()` is the assembly of that walk,
  and re-partitioning never changes what it returns. `base_shape` says which shape a partitioning is expressed in. `is_box`, `bounding_box()`
  and `strides()` report whether a selection is rectangular, so a consumer can
  dispatch a slab read against a gather.

  See the [guide](https://zarr-indexing.readthedocs.io/en/latest/guide/) for the
  model and the
  [design notes](https://zarr-indexing.readthedocs.io/en/latest/design-notes/)
  for the box/query distinction, the relationship to TensorStore, and current
  scope limits. ([#4222](https://github.com/zarr-developers/zarr-python/pull/4222))
- Added source-independent chunk planning. `plan_chunks(transform, grids)` returns
  a lazy, reusable `ChunkPlan` whose `ChunkProjection`s each pair a chunk-local
  transform with a transform back to the request, over one shared cell domain, so
  a consumer can read a chunk and place its values without re-deriving either. The
  same representation covers basic, orthogonal and vectorized indexing, and
  carries global chunk bounds plus conservative full/partial/unknown coverage.
  I/O, buffering and scheduling stay with the consumer. `zarr_indexing.grid` gained
  `EdgeDimensionGrid` and `dimension_grids_from_chunks` for building the per-axis
  grids it takes. ([#4222](https://github.com/zarr-developers/zarr-python/pull/4222))

### Bugfixes

- An adversarial review of the whole package found, and this fixes, several
  defects at its boundaries: `result()` and `__array__(copy=True)` could hand back
  a live view of a source that merely stored its data in NumPy; the wire format
  emitted a document nothing could load for a selection that selects nothing, and
  its domain loader validated nothing; chunk-selection lowering described a
  transposed block in two separate cases; a map derived from a vectorized
  selection carried a stale `input_dimension`, which made one view's answer depend
  on how it was partitioned; and `oindex` over a correlated view applied NumPy's
  vectorized rule instead of the outer product. ([#4222](https://github.com/zarr-developers/zarr-python/pull/4222))
- Correctness fixes to indexing and resolution, all reachable from 0.1.0:

  - An integer index applied to an axis a previous `oindex`/`vindex` step had
    already indexed left an all-singleton `ArrayMap` still naming the axis the
    integer removed, which after renumbering aliased a different one. Such a map
    now collapses to a `ConstantMap` at composition time.
  - A `vindex` selection whose coordinate arrays are not on the leading axes
    (`vindex[..., i, j]`, `vindex[..., mask]`) laid out its result incorrectly and
    raised a shape mismatch on a partitioned read. Gathered dimensions now follow
    NumPy's placement rule, and the per-part gather is realigned to the scatter.
  - An `oindex`/`vindex` step whose entries are all slices, applied to a view with
    a fancy-indexed axis, applied those slices positionally to every axis of the
    existing index array — including broadcast singletons — truncating it to size
    0, so `result()` returned an unwritten buffer. Reindexing is now
    dependency-aware.
  - `parts()` raised on a view emptied by a slice over an axis of extent 1; an
    empty domain now yields no parts, matching `result()`.
  - Negative-stride chunk projection swapped the endpoints while keeping the step
    negative, selecting nothing where the reversed axis was meant. Composition
    evaluated an inner index array over `range(size)` rather than the outer
    domain's own range, resolving every coordinate wrongly whenever that domain
    did not start at 0 — which both step-1 and negative-step slices produce.
  - A domain dimension no output map depends on, left behind when a later basic
    index consumes the axis a `vindex` array varied over, was miscounted in three
    places: the partition walk's out-selection rank, the lowering engine's axis
    restoration, and the correlated gather's broadcast.
  - The parts of a correlated view narrowed to a single point came back rank 1
    where the view was rank 0, so the documented
    `out[part.out_selection] = part.view.result()` assembly raised `ValueError`.
  - `result()` could return memory shared with the wrapped array: an unpartitioned
    read of a basic selection lowered to plain slicing and handed back a view of
    the source, and `numpy.array(view, copy=True)` inherited the alias. It now
    always allocates, and verifies the parts covered the output before returning.
  - `IndexTransform.from_json` rejects a non-integer `index_array` with an
    `NdselError` carrying `invalid_json`, rather than truncating a float array,
    coercing booleans, or leaking NumPy's conversion error for strings.
  - `result(parts=...)` raises `ValueError` rather than `AssertionError` when the
    supplied parts do not tile the view, and `with_parts` / `with_parts_per_axis`
    raise the documented `ValueError` for non-iterable input.

  ([#4222](https://github.com/zarr-developers/zarr-python/pull/4222))

### Deprecations and Removals

- The canonical JSON converters are now methods on the types that own the
  serialization: `IndexTransform.to_json()` / `IndexTransform.from_json()`,
  `IndexDomain.to_json()` / `IndexDomain.from_json()`, and `to_json()` on each
  output map kind. `output_index_map_from_json` remains a function, in
  `zarr_indexing.output_map`, because the wire form is a tagged union and
  loading it dispatches rather than belonging to any one kind.

  The free functions they replace — `transform_to_canonical`,
  `transform_from_canonical`, `index_domain_to_json`, `index_domain_from_json`,
  `output_index_map_to_json`, and the historical aliases
  `index_transform_to_json` / `index_transform_from_json` — are removed. There
  had been two spellings of each conversion; there is now one. ([#4222](https://github.com/zarr-developers/zarr-python/pull/4222))
- `ArrayMap` no longer has an `input_dimension` field: what a map depends on is
  read from its full-rank index array's shape (its non-singleton axes), the
  single source of truth. A selection narrowed to a single coordinate is now
  built as the `ConstantMap` it equals (`array_map_or_constant`), so a length-1
  fancy selection classifies as a box; hand-built all-singleton or shared-axis
  `ArrayMap`s resolve through the pointwise path. The wire format is unaffected —
  it never carried the field.

  The provisional tuple resolver and selector bridge are gone with it:
  `iter_chunk_transforms` and `sub_transform_to_selections` are removed, their
  role taken by `plan_chunks` and the paired projections it returns. ([#4222](https://github.com/zarr-developers/zarr-python/pull/4222))
- Operations moved onto the types that own them, following the arrangement
  TensorStore uses (public headers are the types; every transform operation
  lives in `internal/` and surfaces as a method):

  - `compose(outer, inner)` is now `outer.compose(inner)`, and the algorithm
    moved to the private `zarr_indexing._composition`.
  - `selection_to_transform(selection, transform, mode)` is now
    `transform.select(selection, mode)`.
  - `index_array_structure(transform)` is now the `transform.index_array_structure`
    property.
  - `array_map_dependent_axis(m)` is now the `ArrayMap.dependent_axis` property,
    alongside a new `ArrayMap.dependency_axes` giving every axis a map varies over.

  `zarr_indexing.affine` is now the private `zarr_indexing._affine`; it was
  never exported or documented.

  ([#4222](https://github.com/zarr-developers/zarr-python/pull/4222))
- `with_parts` is now three named methods — `with_parts`, `with_parts_per_axis`
  and `unpartitioned` — instead of one parameter whose meaning was decided by the
  type of what it was given. `Partition.array` is `Partition.view`, no longer the
  inverse of `LazyArray.array`. `ArrayMap`, `IndexTransform` and `Partition` can
  be compared and hashed, which `frozen=True` had implied and neither could do.
  `LazyArray.base_shape` says which shape a partitioning is expressed in. ([#4222](https://github.com/zarr-developers/zarr-python/pull/4222))

### Misc

- Restructured the documentation-contract tests: the snippet include graph is
  now discovered by scanning the rendered markdown instead of hand-maintained
  registries, prose and navigation assertions moved out of CI, and
  `pymdownx.snippets` now sets `check_paths: true` so an unresolvable include
  fails `mkdocs build --strict` instead of silently rendering nothing. ([#4222](https://github.com/zarr-developers/zarr-python/pull/4222))


## 0.1.0 (2026-07-31)

### Features

- First release of `zarr-indexing`: TensorStore-style index transforms and the
  ndsel wire format. ([#4196](https://github.com/zarr-developers/zarr-python/pull/4196))
- Reworked the JSON layer to conform to the [ndsel](https://github.com/zarr-developers/ndsel) draft wire format, which adapts TensorStore's `IndexTransform`. A new `zarr_indexing.messages` module (`parse_ndsel`, `normalize_ndsel`, `NdselError`) is a pure JSON-to-JSON layer that accepts all five message kinds (`point`/`box`/`slice`/`points`/`transform`) and normalizes them to the canonical transform body, enforcing the full ndsel error taxonomy. The package is checked against the vendored, language-agnostic ndsel conformance corpus. Serialization produces and consumes the canonical body (`IndexTransform.to_json`/`from_json`, and the `IndexDomain` pair). On serialization, orthogonal (`oindex`) `index_array` maps no longer emit `input_dimension` alongside `index_array` (a combination both ndsel and TensorStore reject), and degenerate all-singleton index arrays collapse to constant maps; the in-memory `input_dimension` is reconstructed from the array's dependency axes on load. ([#4196](https://github.com/zarr-developers/zarr-python/pull/4196))
