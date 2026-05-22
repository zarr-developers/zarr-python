# Indexing Test Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate and speed up `tests/test_indexing.py` by shrinking oversized arrays, replacing `np.random` selection loops with hand-picked parametrized cases, isolating one behavior per test, and unifying the duplicated `Expect` test-case dataclasses onto one canonical pair.

**Architecture:** First deduplicate the two divergent `Expect`/error dataclass pairs (`tests/conftest.py` vs `tests/test_codecs/conftest.py`) onto the richer `tests/conftest.py` pair, migrating the three codec/chunk-grid consumers. Then rewrite each indexing test family: smaller arrays (≥3 chunks/axis, partial edge), explicit `Expect[Selection, None]` / `ExpectFail[Selection]` case tables parametrized with `ids=lambda c: c.id`, comparing zarr against a numpy oracle. Error paths become their own named parametrized tests.

**Tech Stack:** Python 3.12, pytest, numpy, zarr; tests run via `uv run --frozen pytest`. Lint/type via `prek`.

**Spec:** `docs/superpowers/specs/2026-05-22-indexing-test-cleanup-design.md` (gist: https://gist.github.com/d-v-b/508a7294cba8bc702f36a4b85f4a8a90)

**Conventions (from project memory — apply throughout):**
- Every new/rewritten test gets a docstring stating the *behavior* it verifies.
- Compact happy-path via parametrization; each exception path gets its own named test.
- Single backticks in docstrings; no RST roles, no double-backticks.
- No falsy conditionals — test `is None` etc. explicitly.
- Run mypy via `prek --all-files`, never ad-hoc `uvx mypy`.
- Commit messages end with the `Co-Authored-By: Claude Opus 4.7 (1M context)` trailer.

---

## Part 0 — Deduplicate the `Expect` dataclasses

The canonical pair lives in `tests/conftest.py`:

```python
@dataclass
class Expect[TIn, TOut]:
    """A test case with explicit input, expected output, and a human-readable id."""
    input: TIn
    output: TOut
    id: str

@dataclass
class ExpectFail[TIn]:
    """A test case that should raise an exception."""
    input: TIn
    exception: type[Exception]
    id: str
    msg: str
```

The duplicate to delete lives in `tests/test_codecs/conftest.py`:

```python
@dataclass(frozen=True)
class Expect[TIn, TOut]:
    input: TIn
    expected: TOut

@dataclass(frozen=True)
class ExpectErr[TIn]:
    input: TIn
    msg: str
    exception_cls: type[Exception]
```

**Migration rule** (applied at every call site in the three consumer files):
- `Expect(input=X, expected=Y)` → `Expect(input=X, output=Y, id="<id>")`
- `ExpectErr(input=X, msg=M, exception_cls=E)` → `ExpectFail(input=X, exception=E, id="<id>", msg=M)`
- The `<id>` comes from the existing positional `ids=[...]` list on the `@pytest.mark.parametrize` for that block (1:1 by order). After moving ids onto cases, replace `ids=[...]` with `ids=lambda c: c.id`.
- Where a parametrize block has no `ids=` today, synthesize concise kebab-case ids.
- Import line in each consumer becomes `from tests.conftest import Expect, ExpectFail`.

Consumers (all import both classes from `tests.test_codecs.conftest`):
`tests/test_chunk_grids.py`, `tests/test_codecs/test_cast_value.py`, `tests/test_codecs/test_scale_offset.py`.

### Task 0.1: Make the canonical `Expect`/`ExpectFail` frozen

**Files:**
- Modify: `tests/conftest.py:67-83`

- [ ] **Step 1: Add `frozen=True` to both canonical dataclasses**

In `tests/conftest.py`, change the two decorators from `@dataclass` to `@dataclass(frozen=True)`:

```python
@dataclass(frozen=True)
class Expect[TIn, TOut]:
    """A test case with explicit input, expected output, and a human-readable id."""

    input: TIn
    output: TOut
    id: str


@dataclass(frozen=True)
class ExpectFail[TIn]:
    """A test case that should raise an exception."""

    input: TIn
    exception: type[Exception]
    id: str
    msg: str
```

- [ ] **Step 2: Verify the existing canonical consumer still passes**

Run: `uv run --frozen pytest tests/test_metadata/test_v3.py -q`
Expected: PASS (no field names changed for this consumer; only `frozen` added).

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "$(cat <<'EOF'
test: make canonical Expect/ExpectFail frozen

Prepares for deduplicating the second Expect pair in test_codecs/conftest.py.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 0.2: Migrate `tests/test_chunk_grids.py` to canonical pair

**Files:**
- Modify: `tests/test_chunk_grids.py` (import line 7; all `Expect(...)`, `ExpectErr(...)`, and `ids=[...]` blocks)

- [ ] **Step 1: Change the import**

Replace `from tests.test_codecs.conftest import Expect, ExpectErr` with `from tests.conftest import Expect, ExpectFail`.

- [ ] **Step 2: Migrate every `Expect(...)` and `ExpectErr(...)` call per the rule above**

Worked example — the `test_normalize_chunks_1d_errors` block (currently lines 131-175). The existing `ids=[...]` list maps 1:1 to the cases. After migration:

```python
@pytest.mark.parametrize(
    "case",
    [
        ExpectFail(input=(0, 100), exception=ValueError, id="zero-uniform", msg="Chunk size must be positive"),
        ExpectFail(input=(-2, 100), exception=ValueError, id="negative-uniform", msg="Chunk size must be positive"),
        ExpectFail(input=([], 100), exception=ValueError, id="empty-list", msg="must not be empty"),
        ExpectFail(input=([10, -1, 10], 100), exception=ValueError, id="negative-element", msg="must be positive"),
        ExpectFail(input=([10, 0, 10], 20), exception=ValueError, id="zero-element", msg="must be positive"),
        ExpectFail(input=([10, 20], 100), exception=ValueError, id="wrong-sum", msg="do not sum to span"),
        ExpectFail(input=([[3, 3], 1], 7), exception=TypeError, id="rle-single-dim", msg="non-integer element(s) ([3, 3],) at indices (0,)"),
        ExpectFail(input=([1, [2, 2], 1, [3]], 9), exception=TypeError, id="multiple-non-ints", msg="non-integer element(s) ([2, 2], [3]) at indices (1, 3)"),
        ExpectFail(input=([2, "3", 5], 10), exception=TypeError, id="string-element", msg="non-integer element(s) ('3',) at indices (1,)"),
    ],
    ids=lambda c: c.id,
)
def test_normalize_chunks_1d_errors(case: ExpectFail[tuple[Any, int]]) -> None:
    """Invalid 1D chunk specifications are rejected with informative error messages."""
    chunks, span = case.input
    with pytest.raises(case.exception, match=re.escape(case.msg)):
        normalize_chunks_1d(chunks, span=span)
```

Apply the same transformation to the remaining parametrize blocks in this file:
the `test_normalize_chunks_nd_errors` block (ids `["none", "true", "string", "too-many-dims", "too-few-dims", "rle-inner-dim"]`), and the success-case `Expect(...)` blocks (rename `expected=`→`output=`, add `id=` from the block's `ids=[...]`, switch `ids=` to `lambda c: c.id`). Update each function's `case:` type annotation: `ExpectErr[...]` → `ExpectFail[...]`, and references `case.exception_cls` → `case.exception`.

- [ ] **Step 3: Run the file to verify green**

Run: `uv run --frozen pytest tests/test_chunk_grids.py -q`
Expected: PASS, same test count as before (ids change but cases don't).

- [ ] **Step 4: Commit**

```bash
git add tests/test_chunk_grids.py
git commit -m "$(cat <<'EOF'
test: migrate test_chunk_grids to canonical Expect/ExpectFail

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 0.3: Migrate `tests/test_codecs/test_cast_value.py` to canonical pair

**Files:**
- Modify: `tests/test_codecs/test_cast_value.py` (import line 9; all `Expect(...)`/`ExpectErr(...)`/`ids=[...]` blocks)

- [ ] **Step 1: Change the import** to `from tests.conftest import Expect, ExpectFail`.

- [ ] **Step 2: Migrate every call site per the rule.** For success cases, `expected=`→`output=`, add `id=` taken 1:1 from the block's `ids=[...]` list (e.g. `["minimal", "full"]`, `["defaults", "explicit"]`, `["no-scalar-map", "with-scalar-map"]`, `["complex-source", "wrap-float-target"]`, `["f64→f32", "f32→f64", "i32→i64", "i64→i16", "f64→i32", "i32→f64"]`, `["towards-zero"]`, `["int32→int8"]`), then set `ids=lambda c: c.id`. For error blocks, `ExpectErr`→`ExpectFail`, `exception_cls=`→`exception=`, add `id=`, and update `case.exception_cls`→`case.exception` plus the `case:` type annotations.

- [ ] **Step 3: Run the file**

Run: `uv run --frozen pytest tests/test_codecs/test_cast_value.py -q`
Expected: PASS, same test count.

- [ ] **Step 4: Commit**

```bash
git add tests/test_codecs/test_cast_value.py
git commit -m "$(cat <<'EOF'
test: migrate test_cast_value to canonical Expect/ExpectFail

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 0.4: Migrate `tests/test_codecs/test_scale_offset.py` to canonical pair

**Files:**
- Modify: `tests/test_codecs/test_scale_offset.py` (import line 9; all call sites)

- [ ] **Step 1: Change the import** to `from tests.conftest import Expect, ExpectFail`.

- [ ] **Step 2: Migrate every call site per the rule** (same mechanical transformation as Task 0.2/0.3). This file's success cases at lines 27-38 have no `ids=` list today — synthesize ids: `id="default"`, `id="offset-only"`, `id="scale-only"`, `id="offset-and-scale"` matching the four `Expect(...)` cases in order. Error blocks (`ExpectErr` at ~82-87 and ~262-272): convert to `ExpectFail`, add ids (`"non-numeric-offset"`, `"non-numeric-scale"` and `"unrepresentable-1"`, `"unrepresentable-2"`, `"unrepresentable-3"` respectively — pick descriptive names from each case's input). Update annotations and `.exception_cls`→`.exception`.

- [ ] **Step 3: Run the file**

Run: `uv run --frozen pytest tests/test_codecs/test_scale_offset.py -q`
Expected: PASS, same test count.

- [ ] **Step 4: Commit**

```bash
git add tests/test_codecs/test_scale_offset.py
git commit -m "$(cat <<'EOF'
test: migrate test_scale_offset to canonical Expect/ExpectFail

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 0.5: Delete the duplicate dataclasses and verify nothing imports them

**Files:**
- Delete contents of: `tests/test_codecs/conftest.py` (file is only these two classes)

- [ ] **Step 1: Confirm no remaining importers**

Run: `grep -rn "from tests.test_codecs.conftest import\|test_codecs.conftest import" tests/`
Expected: no output (all three consumers migrated in 0.2–0.4).

- [ ] **Step 2: Delete the file**

`tests/test_codecs/conftest.py` contains only the two duplicate dataclasses, so remove it entirely:

```bash
git rm tests/test_codecs/conftest.py
```

- [ ] **Step 3: Run the full codecs + chunk-grids + v3 suites**

Run: `uv run --frozen pytest tests/test_chunk_grids.py tests/test_codecs/ tests/test_metadata/test_v3.py -q`
Expected: PASS, no collection errors.

- [ ] **Step 4: Run prek on touched files**

Run: `prek run --all-files`
Expected: all hooks pass (mypy in particular — the `case:` annotations now reference `ExpectFail`).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
test: delete duplicate Expect dataclasses in test_codecs/conftest.py

All consumers now use the canonical Expect/ExpectFail from tests/conftest.py.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Part 1 — Rewrite the indexing test families

All indexing tests use the file-local `store` fixture (MemoryStore, runs once
per test) and the `zarr_array_from_numpy_array(store, a, chunk_shape=...)`
helper. Both stay. The numpy-oracle assertion helpers
(`_test_get_basic_selection`, `_test_get_orthogonal_selection`,
`_test_set_orthogonal_selection`, `_test_get_coordinate_selection`,
`_test_set_coordinate_selection`, `_test_get_block_selection`,
`_test_set_block_selection`, `_test_get_mask_selection`,
`_test_set_mask_selection`) are **kept** — they encapsulate the
zarr-vs-numpy comparison and are now called once per parametrized case.

**Array-size rule (fixed):** every indexed axis spans ≥3 chunks; at least one
axis has a partial (non-full) edge chunk. Canonical shapes:
- 1d: `np.arange(30)`, chunks `(7,)` → 5 chunks, last size 2.
- 2d: `np.arange(60).reshape(12, 5)`, chunks `(5, 2)` → 3×3 chunks, partial edges.
- 3d: `np.arange(420).reshape(7, 6, 10)`, chunks `(3, 2, 4)` → 3×3×3 chunks, partial edges everywhere.

Add the import for the case dataclasses at the top of `tests/test_indexing.py`:
`from tests.conftest import Expect, ExpectFail`.

### Task 1.1 (EXEMPLAR): Rewrite `test_get_orthogonal_selection_1d_bool` and split its error paths

This task is the worked template. Tasks 1.2+ apply the same recipe and reference it.

**Recipe (apply to every family task):**
1. Shrink the array to the canonical shape for its dimensionality.
2. Replace the `np.random.seed` + `for p in ...` loop with an explicit
   module-level `Expect[Selection, None]` list named `_<FAMILY>_CASES`, each
   case carrying a hand-picked selection in `input`, `output=None`, and a
   descriptive `id`. Cover: empty (all-False mask / size-0 slice), full,
   alternating, single-element, sparse, and (for int arrays) sorted, unsorted,
   duplicate, negative/wraparound.
3. Make the test parametrized over that list with `ids=lambda c: c.id`, calling
   the kept oracle helper with `case.input`.
4. Extract the bundled `pytest.raises(IndexError)` block into a separate
   `test_<family>_raises` parametrized over an `ExpectFail[Selection]` list
   `_<FAMILY>_BAD_CASES` (each with `exception=IndexError`, an `id`, and a `msg`
   regex — use `msg=""` to match any `IndexError` message where the original
   only asserted the type).
5. Give both tests docstrings stating the behavior.

**Files:**
- Modify: `tests/test_indexing.py` (replace `test_get_orthogonal_selection_1d_bool`, lines 615-633)

- [ ] **Step 1: Add the case tables and rewritten tests**

Replace the existing `test_get_orthogonal_selection_1d_bool` with:

```python
_ORTHO_1D_BOOL_CASES: list[Expect[OrthogonalSelection, None]] = [
    Expect(input=np.zeros(30, dtype=bool), output=None, id="empty-mask"),
    Expect(input=np.ones(30, dtype=bool), output=None, id="full-mask"),
    Expect(input=np.arange(30) % 2 == 0, output=None, id="alternating-mask"),
    Expect(input=np.eye(1, 30, 7, dtype=bool)[0], output=None, id="single-true"),
    Expect(
        input=np.isin(np.arange(30), [0, 1, 8, 15, 29]),
        output=None,
        id="sparse-cross-chunk",
    ),
]

_ORTHO_1D_BOOL_BAD_CASES: list[ExpectFail[Any]] = [
    ExpectFail(input=np.zeros(5, dtype=bool), exception=IndexError, id="mask-too-short", msg=""),
    ExpectFail(input=np.zeros(50, dtype=bool), exception=IndexError, id="mask-too-long", msg=""),
    ExpectFail(
        input=[[True, False], [False, True]],
        exception=IndexError,
        id="mask-too-many-dims",
        msg="",
    ),
]


@pytest.mark.parametrize("case", _ORTHO_1D_BOOL_CASES, ids=lambda c: c.id)
def test_get_orthogonal_selection_1d_bool(store: StorePath, case: Expect[OrthogonalSelection, None]) -> None:
    """oindex with a 1D boolean mask matches numpy across chunk boundaries."""
    a = np.arange(30, dtype=int)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(7,))
    _test_get_orthogonal_selection(a, z, case.input)


@pytest.mark.parametrize("case", _ORTHO_1D_BOOL_BAD_CASES, ids=lambda c: c.id)
def test_get_orthogonal_selection_1d_bool_raises(
    store: StorePath, case: ExpectFail[Any]
) -> None:
    """oindex rejects masks of the wrong length or dimensionality with IndexError."""
    a = np.arange(30, dtype=int)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(7,))
    with pytest.raises(case.exception, match=case.msg):
        z.oindex[case.input]
```

- [ ] **Step 2: Run the rewritten tests to verify they pass**

Run: `uv run --frozen pytest "tests/test_indexing.py::test_get_orthogonal_selection_1d_bool" "tests/test_indexing.py::test_get_orthogonal_selection_1d_bool_raises" -v`
Expected: PASS — 5 parametrized happy-path cases (by id) + 3 raises cases.

- [ ] **Step 3: Run the whole file to confirm no breakage**

Run: `uv run --frozen pytest tests/test_indexing.py -q`
Expected: PASS (one fewer monolithic test, several new parametrized cases).

- [ ] **Step 4: Commit**

```bash
git add tests/test_indexing.py
git commit -m "$(cat <<'EOF'
test: rewrite orthogonal 1d bool indexing as parametrized cases

Smaller array (30 elems, chunks of 7), hand-picked deterministic masks
replacing np.random sparsity sweep, error paths split into their own test.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 1.2: Rewrite `test_get_orthogonal_selection_1d_int`

**Files:** Modify `tests/test_indexing.py` (lines 637-668).

- [ ] **Step 1: Apply the recipe.** Array `np.arange(30)`, chunks `(7,)`. Happy-path `_ORTHO_1D_INT_CASES` covering: `id="sorted"` (`[0, 8, 15, 29]`), `id="unsorted"` (`[3, 29, 1, 16]`), `id="duplicates"` (`[2, 2, 8, 8]`), `id="wraparound"` (`[0, 3, 10, -23, -12, -1]`), `id="single"` (`[15]`). Error `_ORTHO_1D_INT_BAD_CASES` (each `ExpectFail(exception=IndexError, msg="")`): `id="out-of-bounds-high"` (`[31]`), `id="out-of-bounds-low"` (`[-31]`), `id="too-many-dims"` (`[[2, 4], [6, 8]]`). The raises test asserts both `z.get_orthogonal_selection(case.input)` and `z.oindex[case.input]` raise (two `pytest.raises` blocks in the test body, as the original did). Docstrings on both.

- [ ] **Step 2: Run the two tests**

Run: `uv run --frozen pytest "tests/test_indexing.py::test_get_orthogonal_selection_1d_int" "tests/test_indexing.py::test_get_orthogonal_selection_1d_int_raises" -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_indexing.py
git commit -m "test: rewrite orthogonal 1d int indexing as parametrized cases

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 1.3: Rewrite `test_get_orthogonal_selection_2d` and `_test_get_orthogonal_selection_2d`

**Files:** Modify `tests/test_indexing.py` (lines 671-727).

- [ ] **Step 1: Apply the recipe.** Array `np.arange(60).reshape(12, 5)`, chunks `(5, 2)`. The private `_test_get_orthogonal_selection_2d` helper currently loops over 7 selection shapes built from two index arrays — instead build a `_ORTHO_2D_CASES: list[Expect[OrthogonalSelection, None]]` table directly with concrete hand-picked `ix0`/`ix1` (defined once at module scope), covering the same shapes: both-axes-array, array×slice, array×strided-slice, slice×array, array×int, int×array, plus the mixed int-array/bool-array pair. Use deterministic indices, e.g. `ix0_bool = np.isin(np.arange(12), [0, 5, 11])`, `ix1_bool = np.array([True, False, True, False, True])`, `ix0_int = np.array([0, 5, 11])`, `ix1_int = np.array([0, 2, 4])`. Fold the `basic_selections_2d` coverage of orthogonal into existing parametrize ids. Error cases from `basic_selections_2d_bad` → `_ORTHO_2D_BAD_CASES` (`ExpectFail`, `IndexError`, `msg=""`), raises test asserting both `get_orthogonal_selection` and `oindex`. Delete the now-unused `_test_get_orthogonal_selection_2d` helper. Docstrings on both tests.

- [ ] **Step 2: Run**

Run: `uv run --frozen pytest "tests/test_indexing.py::test_get_orthogonal_selection_2d" "tests/test_indexing.py::test_get_orthogonal_selection_2d_raises" -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_indexing.py
git commit -m "test: rewrite orthogonal 2d indexing as parametrized cases

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 1.4: Rewrite `test_get_orthogonal_selection_3d` and `_test_get_orthogonal_selection_3d`

**Files:** Modify `tests/test_indexing.py` (lines 729-792).

- [ ] **Step 1: Apply the recipe.** Array `np.arange(420).reshape(7, 6, 10)`, chunks `(3, 2, 4)`. The private helper enumerates 20 selection tuples — turn each into an `Expect` case in `_ORTHO_3D_CASES` with a descriptive id (e.g. `"single-value"`, `"all-negative"`, `"three-arrays"`, `"array-slice-slice"`, ... matching the comment groupings in the original). Define the hand-picked index arrays once at module scope: `ix0 = np.isin(np.arange(7), [0, 3, 6])` (bool) and integer variants per axis. Keep both bool-array and sorted-int-array variants by including both kinds as separate ids rather than a sparsity loop. Adjust the literal int indices in the tuples (e.g. `60, 15, 4`) to be in-bounds for the new `(7, 6, 10)` shape (e.g. `5, 3, 8`). Delete the unused helper. Docstring.

- [ ] **Step 2: Run**

Run: `uv run --frozen pytest "tests/test_indexing.py::test_get_orthogonal_selection_3d" -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_indexing.py
git commit -m "test: rewrite orthogonal 3d indexing as parametrized cases

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 1.5: Rewrite the set-orthogonal family (1d/2d/3d) and helpers

**Files:** Modify `tests/test_indexing.py` (`test_set_orthogonal_selection_1d/2d/3d` and their `_test_set_orthogonal_selection_2d/3d` helpers, lines 807-969).

- [ ] **Step 1: Apply the recipe** to all three set tests, reusing the same `_ORTHO_*_CASES` selection tables defined in Tasks 1.1-1.4 where the selections are valid for set (most are). The kept `_test_set_orthogonal_selection` helper already iterates over the three value forms (scalar / array / list) per selection and skips selections that produce an empty result (the existing `value == []` guard), so each parametrized case still exercises all applicable value forms. If a get-only case turns out not to round-trip through set, give it its own filtered list (e.g. `_ORTHO_3D_SET_CASES`) rather than forcing it. Use the same canonical shapes/chunks. For 1d, parametrize over `_ORTHO_1D_BOOL_CASES + _ORTHO_1D_INT_CASES` minus any that can't be set (e.g. empty selections that can't preserve dimensions — keep the existing skip-empty guard inside `_test_set_orthogonal_selection`). Delete the now-unused `_test_set_orthogonal_selection_2d` and `_test_set_orthogonal_selection_3d` helpers. Docstrings on all three.

- [ ] **Step 2: Run**

Run: `uv run --frozen pytest "tests/test_indexing.py::test_set_orthogonal_selection_1d" "tests/test_indexing.py::test_set_orthogonal_selection_2d" "tests/test_indexing.py::test_set_orthogonal_selection_3d" -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_indexing.py
git commit -m "test: rewrite set-orthogonal indexing family as parametrized cases

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 1.6: Rewrite the basic-selection 1d/2d families

**Files:** Modify `tests/test_indexing.py` (`basic_selections_1d`, `basic_selections_1d_bad`, `test_get_basic_selection_1d`, `basic_selections_2d`, `basic_selections_2d_bad`, `test_get_basic_selection_2d`, lines 204-385).

- [ ] **Step 1: Shrink the selection tables to the new array sizes and parametrize.**
  - 1d array `np.arange(30)`, chunks `(7,)`. Rewrite `basic_selections_1d` as a trimmed `Expect[BasicSelection, None]` list: keep one representative of each kind — single value (`5`, `-1`), bounded slice (`slice(3, 18)`), over-bounds slice (`slice(0, 100)`), negative slice (`slice(-18, -3)`), empty (`slice(0, 0)`, `slice(-1, 0)`), full (`slice(None)`, `Ellipsis`, `()`), and a few stepped slices (`slice(None, None, 3)`, `slice(3, 27, 5)`). Drop the giant step-sweep (steps 10..10000 on a 30-element array are redundant). Each gets an id.
  - `basic_selections_1d_bad` → `_BASIC_1D_BAD_CASES` (`ExpectFail`, `IndexError`, `msg=""`): keep one negative-step slice (`slice(None, None, -1)`), the type errors (`2.3`, `"foo"`, `b"xxx"`, `None`), the shape errors (`(0, 0)`, `(slice(None), slice(None))`), and the integer-list case `[1, 0]`. raises test asserts both `get_basic_selection` and `z[...]`.
  - Same treatment for 2d (`np.arange(60).reshape(12, 5)`, chunks `(5, 2)`): trim `basic_selections_2d` to representatives, adjust literal indices to be in-bounds (e.g. `42`→`5`, `slice(250, 350)`→`slice(2, 9)`). Keep the fancy-indexing fallback assertion (`z[([0, 1], [0, 1])]`) as a small standalone test `test_basic_2d_fancy_fallback` with a docstring.
  - Keep the kept `_test_get_basic_selection` oracle helper (it also checks the `out=` param). Docstrings on all.

- [ ] **Step 2: Run**

Run: `uv run --frozen pytest "tests/test_indexing.py::test_get_basic_selection_1d" "tests/test_indexing.py::test_get_basic_selection_2d" -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_indexing.py
git commit -m "test: rewrite basic 1d/2d selection families as parametrized cases

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 1.7: Rewrite the coordinate-selection family (get/set, 1d/2d)

**Files:** Modify `tests/test_indexing.py` (`test_get_coordinate_selection_1d/2d`, `test_set_coordinate_selection_1d/2d`, `_test_set_coordinate_selection`, and `coordinate_selections_1d_bad`, lines 1019-1186).

- [ ] **Step 1: Apply the recipe.** 1d array `np.arange(30)`, chunks `(7,)`. `_COORD_1D_CASES` (`Expect[CoordinateSelection, None]`): single (`5`, `-1`), wraparound (`[0, 3, 10, -23, -12, -1]` → adjust to in-bounds for 30: `[0, 3, 10, -23, -12, -1]` is valid since -23..−1 map into 0..29), out-of-order (`[3, 25, 8, 17]`), multi-dim (`np.array([[2, 4], [6, 8]])`), sorted (`[1, 8, 15, 29]`), reversed (`[29, 15, 8, 1]`). Replace the `for p in 2, 0.5, 0.1, 0.01` random sweep entirely. Error `_COORD_1D_BAD_CASES` from `coordinate_selections_1d_bad` + out-of-bounds (`[31]`, `[-31]`). 2d array `np.arange(60).reshape(12, 5)`, chunks `(5, 2)`: `_COORD_2D_CASES` covering single `(5, 4)`, `(-1, -1)`, both-axes-array `(ix0, ix1)` with deterministic `ix0=[0,5,11,2,8]`, `ix1=[1,3,4,0,2]`, mixed array/int, not-monotonic cases, multi-dim arrays. 2d error cases (slice mixed with array, Ellipsis) → `_COORD_2D_BAD_CASES`. Keep `_test_get_coordinate_selection`/`_test_set_coordinate_selection` helpers. Docstrings throughout.

- [ ] **Step 2: Run**

Run: `uv run --frozen pytest tests/test_indexing.py -k coordinate -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_indexing.py
git commit -m "test: rewrite coordinate selection family as parametrized cases

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 1.8: Rewrite the block-selection family (get/set, 1d/2d)

**Files:** Modify `tests/test_indexing.py` (`block_selections_*`, `test_get_block_selection_1d/2d`, `test_set_block_selection_1d/2d`, lines 1189-1378).

- [ ] **Step 1: Apply the recipe, preserving the selection↔projection pairing.** The block tests pair a block selection with the array slice it projects to. Keep that by making each case carry both, using `Expect[BasicSelection, slice | tuple[slice, ...]]` where `input` is the block selection and `output` is the expected array-index projection (this is the one family where `output` is meaningfully used). 1d array `np.arange(30)`, chunks `(7,)` → 5 blocks. Recompute the projections for the new chunking: block `0`→`slice(0,7)`, block `4`→`slice(28,30)`, `-1`→`slice(28,30)`, `slice(None,3)`→`slice(0,21)`, etc. Build `_BLOCK_1D_CASES` with ids. Error `_BLOCK_1D_BAD_CASES` from `block_selections_1d_bad` + out-of-bounds (`n_blocks+1`, `-(n_blocks+1)` computed from `z._chunk_grid.get_nchunks()`). 2d array `np.arange(60).reshape(12, 5)`, chunks `(5, 2)` → 3×3 blocks; recompute the 2d projections. Update `_test_get_block_selection`/`_test_set_block_selection` to take the projection from `case.output`. Docstrings.

- [ ] **Step 2: Run**

Run: `uv run --frozen pytest tests/test_indexing.py -k block -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_indexing.py
git commit -m "test: rewrite block selection family as parametrized cases

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 1.9: Rewrite the mask-selection family (get/set, 1d/2d)

**Files:** Modify `tests/test_indexing.py` (`mask_selections_1d_bad`, `test_get_mask_selection_1d/2d`, `test_set_mask_selection_1d/2d`, lines 1391-1497).

- [ ] **Step 1: Apply the recipe.** 1d array `np.arange(30)`, chunks `(7,)`. `_MASK_1D_CASES` (`Expect[Any, None]`): all-false, all-true, alternating (`np.arange(30) % 2 == 0`), sparse (`np.isin(np.arange(30), [0, 7, 14, 29])`). 2d array `np.arange(60).reshape(12, 5)`, chunks `(5, 2)`: `_MASK_2D_CASES`: all-false, all-true, checkerboard (`(np.add.outer(np.arange(12), np.arange(5)) % 2).astype(bool)`), sparse. Replace both `for p in 0.5, 0.1, 0.01` sweeps. Error `_MASK_1D_BAD_CASES` from `mask_selections_1d_bad` + too-short (`np.zeros(5, bool)`), too-long (`np.zeros(50, bool)`), too-many-dims (`[[True, False], [False, True]]`); and a `_MASK_2D_BAD_CASES` for the 2d wrong-shape/wrong-ndim cases. Keep `_test_get_mask_selection`/`_test_set_mask_selection`. Docstrings.

- [ ] **Step 2: Run**

Run: `uv run --frozen pytest tests/test_indexing.py -k mask -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_indexing.py
git commit -m "test: rewrite mask selection family as parametrized cases

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 1.10: Shrink `test_get_selection_out` and the remaining large-array tests

**Files:** Modify `tests/test_indexing.py` (`test_get_selection_out` lines 1500-1568; `test_indexing_equals_numpy` lines 1853-1880; `test_orthogonal_bool_indexing_like_numpy_ix` lines 1883-1900).

- [ ] **Step 1: Shrink arrays without changing test logic.**
  - `test_get_selection_out`: read the full body (lines 1500-1568) first; reduce `np.arange(1050)`/`(100,)` to `np.arange(30)`/`(7,)` and adjust the literal selection bounds proportionally, keeping the same `out=` buffer assertions.
  - `test_indexing_equals_numpy` and `test_orthogonal_bool_indexing_like_numpy_ix`: these already parametrize but on a `(1000, 10)` array. Shrink to `(12, 5)` chunks `(5, 2)` and rescale the selections (e.g. `np.arange(1000)`→`np.arange(12)`, `np.tile([True, False], (1000, 5))`→`np.tile([True, False], (12, ...))` sized to the new shape; `[100, 200, 300]`→in-bounds coords). Add/keep docstrings.

- [ ] **Step 2: Run**

Run: `uv run --frozen pytest "tests/test_indexing.py::test_get_selection_out" "tests/test_indexing.py::test_indexing_equals_numpy" "tests/test_indexing.py::test_orthogonal_bool_indexing_like_numpy_ix" -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_indexing.py
git commit -m "test: shrink arrays in selection_out and numpy-equivalence tests

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 1.11: Add docstrings to the leave-as-is tests and sweep already-parametrized ones

**Files:** Modify `tests/test_indexing.py` (the GH-regression, xfail/skip, helper, fallback, and iter tests that keep their logic).

- [ ] **Step 1: Add a one-line behavior docstring** to every test function that lacks one and was not rewritten above: `test_normalize_integer_selection`, `test_replace_ellipsis`, `test_fancy_indexing_fallback_on_get_setitem`, the `*_fallback_*` family, `test_setitem_zarr_array_as_value`, `test_set_basic_selection_0d`, `test_orthogonal_indexing_edge_cases`, `test_set_item_1d_last_two_chunks`, `test_orthogonal_indexing_fallback_on_get_setitem`, `test_slice_selection_uints`, `test_numpy_int_indexing`, `test_accessed_chunks`, `test_iter_grid_invalid`, `test_indexing_with_zarr_array`, `test_zero_sized_chunks`, `test_vectorized_indexing_incompatible_shape`, `test_iter_chunk_regions`, `test_iter_regions`, and the `TestAsync` methods. Do not change their logic or array sizes (regression tests keep their reproducing values per the spec).

- [ ] **Step 2: Run the whole file**

Run: `uv run --frozen pytest tests/test_indexing.py -q`
Expected: PASS, same skip/xfail counts as the baseline (1 skipped, 5 xfailed).

- [ ] **Step 3: Commit**

```bash
git add tests/test_indexing.py
git commit -m "test: add behavior docstrings to remaining indexing tests

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Part 2 — Final verification

**Baseline (measured 2026-05-22, before any changes):** full file
`150 passed, 1 skipped, 5 xfailed` in ~3.5 s (pytest) / ~4 s wall. Slowest:
`test_set_orthogonal_selection_3d` 0.92 s, `test_get_orthogonal_selection_1d_bool`
0.79 s, `test_set_orthogonal_selection_2d` 0.54 s,
`test_set_orthogonal_selection_1d` 0.41 s. The post-rewrite test *count* will
differ (monolithic tests split into many parametrized cases); skip/xfail counts
must stay `1`/`5`.

### Task 2.1: Full-suite verification and speed measurement

- [ ] **Step 1: Run the whole indexing file with durations**

Run: `uv run --frozen pytest tests/test_indexing.py --durations=25 -q`
Expected: PASS; the four tests that were >0.4 s at baseline
(`test_set_orthogonal_selection_3d`, `test_get_orthogonal_selection_1d_bool`,
`test_set_orthogonal_selection_2d`, `test_set_orthogonal_selection_1d`) each
now under ~0.15 s. Record the new total in the commit message.

- [ ] **Step 2: Run the dedup-affected suites once more**

Run: `uv run --frozen pytest tests/test_chunk_grids.py tests/test_codecs/ tests/test_metadata/test_v3.py -q`
Expected: PASS.

- [ ] **Step 3: Full lint + type pass**

Run: `prek run --all-files`
Expected: all hooks pass (ruff, mypy, codespell, etc.).

- [ ] **Step 4: Add a changelog fragment if the repo requires one**

Check `changes/` (towncrier). If indexing test-only changes need a fragment per `towncrier-check`, add one; otherwise skip. Run `prek run towncrier-check --all-files` to confirm.

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
test: finalize indexing test cleanup

Full suite green; orthogonal tests dropped from ~0.5-0.9s to <0.15s each.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Notes for the implementer

- **Read before you rewrite.** Several tasks reference line ranges that shift
  as earlier tasks edit the file. Re-locate each function by name before
  editing; don't trust stale line numbers.
- **`Selection` types** (`BasicSelection`, `OrthogonalSelection`,
  `CoordinateSelection`, `Selection`) are already imported at the top of
  `tests/test_indexing.py` — use them in the `Expect[...]` annotations.
- **`msg=""`** in `ExpectFail` matches any message (the original error tests
  only asserted `IndexError`, not text). Don't invent message regexes that the
  code doesn't actually emit.
- **Keep the oracle helpers.** Do not inline `_test_get_orthogonal_selection`
  etc. into the parametrized tests; calling them once per case is the design.
- If a rewritten happy-path case turns out to disagree with numpy (the oracle
  fails), that is a real finding — stop and investigate per
  systematic-debugging, don't tweak the case to make it pass.
```
