# TensorStore-Aligned Index Transform Corrections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make transform-based Zarr indexing preserve eager NumPy/Zarr semantics across lazy composition, orthogonal and vectorized indexing, zero-rank arrays, caller-provided output buffers, and writes.

**Architecture:** Normalize every `ArrayMap.index_array` over the complete transform input domain, following TensorStore. Derive independent versus correlated indexing from array dependency axes, preserve those axes through composition and chunk intersection, and keep public Zarr negative-index normalization at the array-selection boundary. Verify behavior with focused regressions plus generated indexing programs compared against NumPy.

**Tech Stack:** Python 3.12+, NumPy, pytest, Hypothesis, Zarr transform and codec pipeline internals.

## Global Constraints

- Keep the public indexing API unchanged.
- Use full-input-rank index arrays; do not add a transform-wide indexing-mode flag.
- Preserve TensorStore-style literal coordinates in the general `IndexTransform` API while applying NumPy-style negative wraparound at the public Zarr array boundary.
- Never silently bypass a lazy view transform.
- Keep negative slice steps unsupported.
- Follow test-first red/green/refactor cycles.

---

### Task 1: Normalize ArrayMap Dependency Axes

**Files:**
- Modify: `src/zarr/core/transforms/transform.py`
- Modify: `src/zarr/core/transforms/composition.py`
- Modify: `src/zarr/core/transforms/json.py`
- Test: `tests/test_transforms/test_transform.py`
- Test: `tests/test_transforms/test_composition.py`
- Test: `tests/test_transforms/test_json.py`

**Interfaces:**
- Consumes: `IndexTransform(domain, output)`, `_apply_oindex`, `_apply_vindex`, `compose`.
- Produces: full-input-rank `ArrayMap.index_array` values and an internal dependency-axis classifier used by later tasks.

- [ ] **Step 1: Write failing transform-construction tests**

Add assertions that orthogonal maps have shapes `(2, 1)` and `(1, 3)`, while two vectorized maps share shape `(2,)`. Assert JSON round-tripping preserves singleton axes.

```python
def test_oindex_multiple_arrays_preserves_independent_axes() -> None:
    t = IndexTransform.from_shape((10, 20))
    result = t.oindex[np.array([1, 3]), np.array([2, 4, 6])]
    assert result.domain.shape == (2, 3)
    assert result.output[0].index_array.shape == (2, 1)
    assert result.output[1].index_array.shape == (1, 3)


def test_vindex_multiple_arrays_preserves_shared_axes() -> None:
    t = IndexTransform.from_shape((10, 20))
    result = t.vindex[np.array([1, 3]), np.array([2, 4])]
    assert result.domain.shape == (2,)
    assert result.output[0].index_array.shape == (2,)
    assert result.output[1].index_array.shape == (2,)
```

- [ ] **Step 2: Run the focused tests and confirm the orthogonal shape assertion fails**

Run:

```bash
.venv/bin/python -m pytest tests/test_transforms/test_transform.py tests/test_transforms/test_composition.py tests/test_transforms/test_json.py -q
```

Expected: the new orthogonal singleton-axis assertions fail against squeezed arrays.

- [ ] **Step 3: Implement full-rank normalization**

Update `_apply_oindex` to reshape each public one-dimensional integer array with one varying axis and singleton axes elsewhere. Update `_apply_vindex` to align every broadcast array with the broadcast dimensions plus singleton slice dimensions. Add a private helper with this contract:

```python
def _array_map_dependency_axes(index_array: np.ndarray[Any, Any]) -> tuple[int, ...]:
    """Return input axes on which a normalized index array varies."""
    return tuple(axis for axis, size in enumerate(index_array.shape) if size != 1)
```

Preserve full rank in basic indexing, composition, and JSON conversion. Reject an `ArrayMap` whose rank differs from its containing transform input rank, except during a clearly defined compatibility normalization at construction.

- [ ] **Step 4: Run focused transform tests and refactor without changing behavior**

Run the command from Step 2. Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/zarr/core/transforms tests/test_transforms
git commit -m "fix(indexing): preserve array map dependency axes" -m "Assisted-by: Codex:GPT-5"
```

### Task 2: Resolve Independent and Correlated Array Maps Correctly

**Files:**
- Modify: `src/zarr/core/transforms/transform.py`
- Modify: `src/zarr/core/transforms/chunk_resolution.py`
- Test: `tests/test_transforms/test_transform.py`
- Test: `tests/test_transforms/test_chunk_resolution.py`
- Test: `tests/test_lazy_indexing.py`

**Interfaces:**
- Consumes: normalized full-rank `ArrayMap` values and `_array_map_dependency_axes` from Task 1.
- Produces: chunk intersections and output selectors that preserve outer-product or pairwise semantics.

- [ ] **Step 1: Write failing orthogonal resolution tests**

Create a chunked 2-D array and compare a lazy orthogonal selection with unequal index lengths to `np.ix_`. Cover both reads and writes across more than one chunk.

```python
def test_lazy_oindex_multiple_arrays_outer_product(arr, data) -> None:
    rows = np.array([1, 11])
    cols = np.array([2, 12, 22])
    actual = arr.lazy.oindex[rows, cols].result()
    np.testing.assert_array_equal(actual, data[np.ix_(rows, cols)])
```

- [ ] **Step 2: Run focused tests and confirm failure in intersection or placement**

```bash
.venv/bin/python -m pytest tests/test_transforms/test_chunk_resolution.py tests/test_lazy_indexing.py -q
```

Expected: the new multi-array orthogonal regression fails.

- [ ] **Step 3: Implement dependency-aware intersection and selectors**

Replace the `len(array_dims) >= 2` correlation test with dependency analysis. Jointly mask maps only when they share the same non-singleton dependency axes. Filter independent maps along their own axes and retain singleton axes. In `sub_transform_to_selections`, emit shared scatter indices only for correlated maps; build orthogonal selectors for independent maps.

- [ ] **Step 4: Verify orthogonal and vectorized behavior**

Run:

```bash
.venv/bin/python -m pytest tests/test_transforms/test_transform.py tests/test_transforms/test_chunk_resolution.py tests/test_lazy_indexing.py -q
```

Expected: PASS, including existing vectorized cases.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/zarr/core/transforms tests/test_transforms tests/test_lazy_indexing.py
git commit -m "fix(indexing): distinguish orthogonal array maps" -m "Assisted-by: Codex:GPT-5"
```

### Task 3: Preserve Lazy Views and Normalize Public Indices

**Files:**
- Modify: `src/zarr/core/array.py`
- Modify: `src/zarr/core/transforms/transform.py`
- Test: `tests/test_lazy_indexing.py`
- Test: `tests/test_indexing.py`

**Interfaces:**
- Consumes: `selection_to_transform(selection, transform, mode)`.
- Produces: array-boundary selection normalization relative to the current visible view and transform-aware fallback behavior.

- [ ] **Step 1: Write failing composed-view and validation tests**

Cover `arr.lazy[10:20].oindex[[0]]`, the corresponding write, `arr.lazy[-1]`, negative integer arrays, overly negative indices, positive out-of-range indices, and invalid Boolean masks. Assert writes by comparing the complete root array with NumPy.

- [ ] **Step 2: Confirm the regressions fail for the reviewed reasons**

```bash
.venv/bin/python -m pytest tests/test_lazy_indexing.py tests/test_indexing.py -q
```

Expected: composed advanced operations access the wrong offset or validation differs from eager Zarr.

- [ ] **Step 3: Add a public-selection normalization boundary**

Implement a helper in `array.py` that normalizes integer scalars and arrays against `self.shape`, validates bounds and mask shapes, and returns a selection suitable for `selection_to_transform`. Use it in `_LazyIndexAccessor`, `_LazyOIndex`, and `_LazyVIndex` before composing transforms.

For advanced access on a non-identity view, compose with `self._transform` instead of creating a legacy indexer against storage. Allow the legacy path only where it is guaranteed to operate on the identity transform. Raise `NotImplementedError` for transformed structured-field operations that cannot preserve field-aware semantics.

- [ ] **Step 4: Run focused lazy and eager indexing tests**

Run the command from Step 2. Expected: PASS.

- [ ] **Step 5: Commit Task 3**

```bash
git add src/zarr/core/array.py src/zarr/core/transforms/transform.py tests/test_lazy_indexing.py tests/test_indexing.py
git commit -m "fix(indexing): compose advanced lazy selections" -m "Assisted-by: Codex:GPT-5"
```

### Task 4: Correct Zero-Rank and Caller-Provided Buffer Handling

**Files:**
- Modify: `src/zarr/core/array.py`
- Test: `tests/test_indexing.py`
- Test: `tests/test_lazy_indexing.py`

**Interfaces:**
- Consumes: normalized transforms and chunk output selectors.
- Produces: scalar-shaped zero-rank I/O and safe multidimensional vectorized `out` behavior.

- [ ] **Step 1: Write failing result-kind and `out` tests**

Assert that `arr[...]` and `arr.lazy[...].result()` for a zero-dimensional array have shape `()` rather than `(1,)`. Add multidimensional coordinate arrays and an `NDBuffer` whose shape equals the broadcast selection shape; assert returned values and buffer contents match NumPy.

- [ ] **Step 2: Run tests and observe the exact shape/placement failures**

```bash
.venv/bin/python -m pytest tests/test_indexing.py::test_get_basic_selection_0d tests/test_indexing.py::test_get_selection_out tests/test_lazy_indexing.py -q
```

Expected: strict zero-rank shape assertions and multidimensional `out` regression fail.

- [ ] **Step 3: Implement correlation-aware flattening**

Define flattening as `bool(correlated_array_maps)` rather than `all(...)`. Keep `buffer_shape == ()` for a zero-rank identity transform. When `out` is supplied for correlated multidimensional indexing, read into a flat temporary, reshape to `out_shape`, and copy into `out`; return the caller-visible result without converting backend buffers unnecessarily.

- [ ] **Step 4: Verify focused buffer tests**

Run the command from Step 2. Expected: PASS.

- [ ] **Step 5: Commit Task 4**

```bash
git add src/zarr/core/array.py tests/test_indexing.py tests/test_lazy_indexing.py
git commit -m "fix(indexing): preserve transform result shapes" -m "Assisted-by: Codex:GPT-5"
```

### Task 5: Add Stateful Property-Based Indexing Programs

**Files:**
- Modify: `src/zarr/testing/strategies.py`
- Modify: `tests/test_properties.py`

**Interfaces:**
- Consumes: array shapes and existing basic/orthogonal/vectorized strategies.
- Produces: `indexing_programs(shape)` yielding valid operation sequences and execution modes for NumPy/Zarr differential testing.

- [ ] **Step 1: Add the program model and failing composed examples**

Define small immutable operation records:

```python
@dataclass(frozen=True)
class IndexOperation:
    mode: Literal["basic", "orthogonal", "vectorized"]
    selection: Any


@dataclass(frozen=True)
class IndexingProgram:
    operations: tuple[IndexOperation, ...]
    execution: Literal["materialize", "eager_on_lazy", "out", "set_scalar", "set_array"]
```

Generate one to three operations while carrying the visible shape forward. Keep sizes small enough for 300 Hypothesis examples. Add deterministic `@example` cases for slice-then-oindex, unequal orthogonal lengths, multidimensional vindex with `out`, and a zero-rank result.

- [ ] **Step 2: Run the new property with deterministic regression examples enabled**

```bash
.venv/bin/python -m pytest tests/test_properties.py -k indexing_program -q
```

Expected on the fixed tree: PASS. Temporarily reverse the Task 2 correlation predicate in the working tree, rerun the slice-then-oindex deterministic example, and confirm it fails before restoring the Task 2 code. This mutation check proves the generated-program oracle detects the original semantic error.

- [ ] **Step 3: Implement the NumPy/Zarr program runner**

For reads, compare exact shape, scalar-versus-array kind, dtype, values, and `out`. For writes, retain the NumPy root array, apply the equivalent indexed assignment, and compare the entire Zarr root array after the write. Use `np.ix_` to implement NumPy orthogonal semantics and normal NumPy indexing for vectorized semantics.

- [ ] **Step 4: Run property tests under the CI profile**

```bash
HYPOTHESIS_PROFILE=ci .venv/bin/python -m pytest --run-slow-hypothesis tests/test_properties.py -k "indexing_program or basic_indexing or oindex or vindex" -q
```

Expected: PASS with 300 deterministic examples per property.

- [ ] **Step 5: Commit Task 5**

```bash
git add src/zarr/testing/strategies.py tests/test_properties.py
git commit -m "test(indexing): generate composed indexing programs" -m "Assisted-by: Codex:GPT-5"
```

### Task 6: Full Verification and Review

**Files:**
- Verify all modified production and test files.

**Interfaces:**
- Consumes: Tasks 1-5.
- Produces: evidence that the branch is ready for review.

- [ ] **Step 1: Run the complete affected test modules**

```bash
.venv/bin/python -m pytest tests/test_transforms tests/test_lazy_indexing.py tests/test_indexing.py tests/test_array.py -q
```

Expected: PASS with only existing documented xfails.

- [ ] **Step 2: Run the focused Hypothesis suite**

```bash
HYPOTHESIS_PROFILE=ci .venv/bin/python -m pytest --run-slow-hypothesis tests/test_properties.py -q
```

Expected: PASS.

- [ ] **Step 3: Run formatting and static checks**

Use the repository's configured commands for Ruff, mypy, and pre-commit on modified files. Expected: no new errors.

- [ ] **Step 4: Inspect the final branch diff**

```bash
git diff --check origin/main...HEAD
git status --short
```

Expected: no whitespace errors and no uncommitted implementation changes.

- [ ] **Step 5: Request a branch code review**

Run the branch-review workflow and address any correctness findings before declaring completion.
